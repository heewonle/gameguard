"""정답 라벨 대비 평가. 정답을 읽을 수 있는 유일한 탐지 쪽 모듈.

  python -m gameguard.evaluate rules --db data/gameguard.duckdb --hits data/out/rule_hits.parquet
"""
import argparse
import datetime as dt
import json
import os
import re

import pandas as pd

from .db import attach_labels, connect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABUSE = ["farm_bot", "rmt_ring", "macro", "multi_account", "market_manip", "chargeback"]


def load_labels(db: str) -> pd.DataFrame:
    con = connect(db, read_only=True)
    attach_labels(con, db)
    return con.execute("SELECT account_id, label, role, stealth FROM labels.account_labels").df()


def stealth_table(flagged: set, lab: pd.DataFrame) -> pd.DataFrame:
    """유형별로 일반형 vs 은닉형 재현율."""
    ab = lab[lab["label"] != "normal"].assign(hit=lambda d: d["account_id"].isin(flagged))
    t = ab.pivot_table(index="label", columns="stealth", values="hit", aggfunc=["size", "mean"])
    t.columns = [f"{'n' if a == 'size' else 'recall'}_{'stealth' if b else 'plain'}" for a, b in t.columns]
    return t.reset_index()


def rule_table(hits: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """룰별 히트 수, 정밀도, 어떤 유형을 잡았는지."""
    h = hits.merge(lab, on="account_id")
    rows = []
    for rid, g in h.groupby("rule_id"):
        acc = g.drop_duplicates("account_id")
        tp = (acc["label"] != "normal").sum()
        row = {"rule": rid, "hits": len(acc), "precision": tp / len(acc)}
        for k in ABUSE:
            row[k] = (acc["label"] == k).sum()
        row["normal(FP)"] = (acc["label"] == "normal").sum()
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_table(flagged: set, lab: pd.DataFrame) -> pd.DataFrame:
    """유형·역할별 재현율 (아무 룰에나 걸린 비율)."""
    lab = lab.assign(hit=lab["account_id"].isin(flagged))
    t = lab.groupby(["label", "role"]).agg(n=("hit", "size"), caught=("hit", "sum")).reset_index()
    t["recall"] = t["caught"] / t["n"]
    return t


def summary(flagged: set, lab: pd.DataFrame) -> dict:
    y = lab["label"] != "normal"
    p = lab["account_id"].isin(flagged)
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"flagged": int(p.sum()), "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def report_rules(db: str, hits_path: str, out_md: str) -> str:
    con = connect(db, read_only=True)
    lab = load_labels(db)
    hits = pd.read_parquet(hits_path)
    flagged = set(hits["account_id"])
    s = summary(flagged, lab)
    md = [f"# 룰 단독 평가 ({dt.date.today()})", "",
          f"- 데이터: `{db}` — 계정 {len(lab):,} (어뷰징 {int((lab['label'] != 'normal').sum()):,})",
          f"- 아무 룰에나 걸린 계정 {s['flagged']:,} · 정밀도 **{s['precision']:.3f}** · 재현율 **{s['recall']:.3f}** · F1 **{s['f1']:.3f}**",
          f"- TP {s['tp']:,} / FP {s['fp']:,} / FN {s['fn']:,}", "",
          "## 룰별", "", fmt(rule_table(hits, lab)), "",
          "## 유형별 일반형 vs 은닉형 재현율", "", fmt(stealth_table(flagged, lab)), "",
          "## 유형·역할별 재현율 (룰 합집합)", "", fmt(coverage_table(flagged, lab)), ""]
    text = "\n".join(md)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write(text)
    return text


def _queue_table(scores: pd.Series, rule_flag: pd.Series, lab: pd.DataFrame, k: int) -> tuple:
    """룰에 안 걸린 계정을 점수순으로 세운 추가 검토 큐 상위 k 의 구성."""
    q = scores[~rule_flag].sort_values(ascending=False).head(k)
    comp = lab.set_index("account_id").loc[q.index]
    hit = (comp["label"] != "normal").mean()
    by = comp.assign(grp=comp["label"] + comp["stealth"].map({True: " (은닉)", False: ""}))["grp"].value_counts()
    return hit, by


def report_stage2(db: str, out_dir: str, out_md: str, budget: int = 300) -> str:
    con = connect(db, read_only=True)
    lab = load_labels(db)
    ids = lab["account_id"]
    rules = set(pd.read_parquet(os.path.join(out_dir, "rule_hits.parquet"))["account_id"])
    sc = {m: pd.read_parquet(os.path.join(out_dir, f"ml_scores_{m}.parquet")).set_index("account_id")
          for m in ("sanctioned", "oracle")}
    iso = sc["sanctioned"]
    methods = {
        "룰만": rules,
        "IsolationForest만 (비지도)": set(iso.index[iso["iso_flag"]]),
        "룰 ∪ IsolationForest": rules | set(iso.index[iso["iso_flag"]]),
        f"룰 ∪ ML-sanctioned (검토 예산 {budget})": rules | set(sc["sanctioned"].index[sc["sanctioned"]["ml_flag"]]),
        "룰 ∪ ML-oracle (상한선, 참고용)": rules | set(sc["oracle"].index[sc["oracle"]["ml_flag"]]),
    }
    rows, st_rows = [], []
    for name, flagged in methods.items():
        s = summary(flagged, lab)
        rows.append({"방법": name, "경보": s["flagged"], "정밀도": s["precision"], "재현율": s["recall"], "F1": s["f1"]})
        t = stealth_table(flagged, lab).set_index("label")["recall_stealth"].rename(name)
        st_rows.append(t)
    st = pd.concat(st_rows, axis=1).reset_index().rename(columns={"label": "유형 (은닉형 재현율)"})

    rule_flag = pd.Series(ids.isin(rules).to_numpy(), index=ids)
    q_lines = []
    for name, s in (("ML-sanctioned", sc["sanctioned"]["ml_score"]), ("IsolationForest", iso["iso_score"]),
                    ("ML-oracle (참고)", sc["oracle"]["ml_score"])):
        hit, by = _queue_table(s.reindex(ids), rule_flag, lab, budget)
        q_lines += [f"- **{name}** 상위 {budget}명 중 실제 어뷰저 **{hit:.1%}** — "
                    + ", ".join(f"{k} {v}" for k, v in by.items())]
    n_resid_abuse = int(((lab["label"] != "normal") & ~ids.isin(rules)).sum())

    md = [f"# 2단 평가 — 룰 + 계정 피처 ML ({dt.date.today()})", "",
          f"- 평가 월드: `{db}` — 계정 {len(lab):,}, 어뷰징 {int((lab['label'] != 'normal').sum()):,}",
          "- 모델·임계값은 개발 월드(seed 42)에서만 학습·결정. 이 월드의 라벨은 채점에만 쓴다.",
          "- ML-sanctioned: 개발 월드에서 **룰에 걸려 확정된 계정만** 양성으로 학습 (실서비스 조건). 룰 밖 상위 "
          f"{budget}명을 추가 검토 큐로 보낸다.",
          "- ML-oracle: 개발 월드의 모든 정답으로 학습. 합성 데이터라 지나치게 쉬우므로 **상한선으로만** 본다.", "",
          "## 방법별 성능", "", fmt(pd.DataFrame(rows)), "",
          "## 은닉형 재현율", "", fmt(st), "",
          f"## 추가 검토 큐 (룰이 놓친 어뷰저 {n_resid_abuse}명 중 얼마나 위로 올리나)", "", *q_lines, ""]
    text = "\n".join(md)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write(text)
    return text


def report_stage3(db: str, out_dir: str, out_md: str) -> str:
    con = connect(db, read_only=True)
    lab = load_labels(db)
    rules = set(pd.read_parquet(os.path.join(out_dir, "rule_hits.parquet"))["account_id"])
    ml = pd.read_parquet(os.path.join(out_dir, "ml_scores_sanctioned.parquet"))
    ml = set(ml.loc[ml["ml_flag"], "account_id"])
    fi = set(pd.read_parquet(os.path.join(out_dir, "graph_fanin.parquet"))["account_id"])
    pr = pd.read_parquet(os.path.join(out_dir, "graph_propagation.parquet"))
    pr = set(pr.loc[pr["flag"], "account_id"])
    steps = [("1단 룰", rules), ("+ 2단 ML-sanctioned", rules | ml), ("+ 3단 신원 그룹 송금 집중", rules | ml | fi),
             ("+ 3단 경보 전파 (최종)", rules | ml | fi | pr)]
    rows, st_rows = [], []
    for name, flagged in steps:
        s = summary(flagged, lab)
        rows.append({"단계": name, "경보": s["flagged"], "정밀도": s["precision"], "재현율": s["recall"], "F1": s["f1"],
                     "FP": s["fp"], "FN": s["fn"]})
        st_rows.append(stealth_table(flagged, lab).set_index("label")["recall_stealth"].rename(name))
    st = pd.concat(st_rows, axis=1).reset_index().rename(columns={"label": "유형 (은닉형 재현율)"})
    final = rules | ml | fi | pr
    md = [f"# 3단 평가 — 룰 + ML + 그래프 ({dt.date.today()})", "",
          f"- 평가 월드: `{db}` — 계정 {len(lab):,}, 어뷰징 {int((lab['label'] != 'normal').sum()):,}",
          "- 모든 임계값은 개발 월드(seed 42)에서 결정. ML은 룰로 확정된 계정만 양성으로 학습(sanctioned).",
          "- 신원 그룹 송금 집중: 주 IP·기기를 공유하는 4~30명 그룹에서 3명 이상이 한 계정에 송금의 70% 이상을 몰아줌 (라벨·씨앗 불필요)",
          "- 경보 전파: 룰 히트(확정에 가까운 경보)를 씨앗으로, 거래·신원 연결 가중치의 70% 이상이 씨앗 쪽인 계정. ML 경보는 검토 전이므로 씨앗에서 뺀다", "",
          "## 단계별 누적 성능", "", fmt(pd.DataFrame(rows)), "",
          "## 단계별 은닉형 재현율", "", fmt(st), "",
          "## 최종 유형·역할별 재현율", "", fmt(coverage_table(final, lab)), ""]
    text = "\n".join(md)
    open(out_md, "w", encoding="utf-8").write(text)
    return text


# ── 4단: LLM 조사 에이전트 ─────────────────────────────────────
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
# LLM 없이 경보 근거만으로 유형을 붙이는 기준선 (우선순위 순)
BASELINE_TYPE = [("R07", "chargeback"), ("R06", "market_manip"), ("R09", "market_manip"), ("R02", "macro"),
                 ("R01", "farm_bot"), ("R08", "rmt_ring"), ("R05", "rmt_ring"), ("fanin", "multi_account"),
                 ("R04", "multi_account"), ("R03", "farm_bot"), ("propagation", "rmt_ring")]


def _nums(text: str):
    out = []
    for m in NUM.findall(str(text)):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def grounded(value: str, sources: str) -> bool:
    """근거 값의 숫자가 모두 도구 결과(또는 경보 원문)에 있는가. 반올림 차이는 0.5% 까지 허용."""
    want = _nums(value)
    if not want:
        return True
    have = _nums(sources)
    for w in want:
        if not any(abs(w - h) <= max(abs(h) * 0.005, 0.0051) for h in have):
            return False
    return True


def baseline_type(details: str) -> str:
    for key, t in BASELINE_TYPE:
        if key in details:
            return t
    return "unknown"


def report_agent(db: str, out_dir: str, tag: str, out_md: str) -> str:
    lab = load_labels(db).set_index("account_id")
    alerts = pd.read_parquet(os.path.join(out_dir, "alerts.parquet")).set_index("account_id")
    rs = [json.loads(l) for l in open(os.path.join(out_dir, f"agent_{tag}.jsonl"), encoding="utf-8")]
    df = pd.DataFrame(rs)
    df["truth"] = df["account_id"].map(lab["label"])
    df["stealth"] = df["account_id"].map(lab["stealth"])
    df["is_abuse"] = df["truth"] != "normal"
    df["base_type"] = df["account_id"].map(lambda a: baseline_type(alerts.loc[a, "details"]))

    # 근거 사실성
    g_items, g_ok = 0, 0
    for r in rs:
        src = " ".join(t["result"] for t in r["transcript"]) + " " + r.get("brief", alerts.loc[r["account_id"], "details"])
        for e in r.get("evidence", []):
            g_items += 1
            g_ok += grounded(e.get("value", ""), src)

    conf = pd.crosstab(df["truth"].where(df["is_abuse"], "normal").map(lambda x: "어뷰징" if x != "normal" else "정상"),
                       df["verdict"]).reindex(columns=["abuse", "uncertain", "normal"], fill_value=0)
    ab, nm = df[df["is_abuse"]], df[~df["is_abuse"]]
    kept = (ab["verdict"] != "normal").mean() if len(ab) else float("nan")
    dismissed = (nm["verdict"] == "normal").mean() if len(nm) else float("nan")
    tp_typed = ab[ab["verdict"] == "abuse"]
    type_acc = (tp_typed["abuse_type"] == tp_typed["truth"]).mean() if len(tp_typed) else float("nan")
    base_acc = (ab["base_type"] == ab["truth"]).mean() if len(ab) else float("nan")
    by_type = ab.groupby("truth").apply(lambda g: pd.Series({
        "n": len(g), "어뷰징 유지": (g["verdict"] != "normal").mean(),
        "유형 정확(에이전트)": ((g["verdict"] == "abuse") & (g["abuse_type"] == g.name)).mean(),
        "유형 정확(기준선)": (g["base_type"] == g.name).mean()}), include_groups=False).reset_index()

    md = [f"# 4단 평가 — LLM 조사 에이전트 `{tag}` ({dt.date.today()})", "",
          f"- 평가 월드 `{db}` 경보 {len(alerts):,}건 중 표본 {len(df)}건 (룰 포함 경보 / 룰 밖 경보를 출처로만 층화 추출, 라벨 미사용)",
          f"- 모델 `{df['model'].iloc[0]}` (로컬 Ollama, 비용 0원) · 프롬프트 `{df['prompt'].iloc[0]}` — 개발 월드에서만 다듬고 고정",
          f"- 표본 구성: 실제 어뷰저 {int(df['is_abuse'].sum())} / 정상(1~3단 오탐) {int((~df['is_abuse']).sum())}", "",
          "## 판정", "",
          f"- 정상인데 경보가 뜬 계정 중 에이전트가 **정상으로 걸러낸 비율 {dismissed:.1%}** (검토자 부담 감소)",
          f"- 실제 어뷰저 중 **어뷰징/불확실로 유지한 비율 {kept:.1%}** (정상으로 잘못 넘기면 놓침)",
          f"- 어뷰징으로 판정한 실제 어뷰저의 **유형 정확도 {type_acc:.1%}** (LLM 없이 경보 근거로 붙인 기준선 {base_acc:.1%})",
          f"- 근거 수치 사실성: 근거 {g_items}개 중 숫자가 도구 결과와 일치 **{g_ok / max(g_items, 1):.1%}**", "",
          "| 실제 \\ 판정 | abuse | uncertain | normal |", "|---|---:|---:|---:|",
          *[f"| {i} | " + " | ".join(str(int(v)) for v in row) + " |" for i, row in conf.iterrows()], "",
          "## 유형별", "", fmt(by_type), "",
          "## 비용·속도 (RTX 4070 SUPER 로컬)", "",
          f"- 건당 중앙값 {df['seconds'].median():.1f}초, 도구 호출 {df['tool_calls'].median():.0f}회, "
          f"입력 토큰 {df['tokens_in'].median():,.0f} / 출력 {df['tokens_out'].median():,.0f}",
          f"- 전체 {df['seconds'].sum() / 60:.0f}분, API 비용 0원", ""]
    text = "\n".join(md)
    open(out_md, "w", encoding="utf-8").write(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rules", "stage2", "stage3", "agent"])
    ap.add_argument("--tag")
    ap.add_argument("--db", default="data/test.duckdb")
    ap.add_argument("--hits", default="data/out_test/rule_hits.parquet")
    ap.add_argument("--out", default="data/out_test")
    ap.add_argument("--md")
    a = ap.parse_args()
    if a.what == "rules":
        print(report_rules(a.db, a.hits, a.md or os.path.join(ROOT, "reports", "eval_rules_test.md")))
    elif a.what == "stage2":
        print(report_stage2(a.db, a.out, a.md or os.path.join(ROOT, "reports", "eval_stage2_test.md")))
    elif a.what == "stage3":
        print(report_stage3(a.db, a.out, a.md or os.path.join(ROOT, "reports", "eval_stage3_test.md")))
    else:
        print(report_agent(a.db, a.out, a.tag, a.md or os.path.join(ROOT, "reports", f"eval_agent_{a.tag}.md")))
