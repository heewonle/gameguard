"""정답 라벨 대비 평가. 정답을 읽을 수 있는 유일한 탐지 쪽 모듈.

  python -m gameguard.evaluate rules --db data/gameguard.duckdb --hits data/out/rule_hits.parquet
"""
import argparse
import datetime as dt
import os

import pandas as pd

from .db import connect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABUSE = ["farm_bot", "rmt_ring", "macro", "multi_account", "market_manip", "chargeback"]


def load_labels(con) -> pd.DataFrame:
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
    lab = load_labels(con)
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
    lab = load_labels(con)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rules", "stage2"])
    ap.add_argument("--db", default="data/test.duckdb")
    ap.add_argument("--hits", default="data/out_test/rule_hits.parquet")
    ap.add_argument("--out", default="data/out_test")
    ap.add_argument("--md")
    a = ap.parse_args()
    if a.what == "rules":
        print(report_rules(a.db, a.hits, a.md or os.path.join(ROOT, "reports", "eval_rules_test.md")))
    else:
        print(report_stage2(a.db, a.out, a.md or os.path.join(ROOT, "reports", "eval_stage2_test.md")))
