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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rules"])
    ap.add_argument("--db", default="data/gameguard.duckdb")
    ap.add_argument("--hits", default="data/out/rule_hits.parquet")
    ap.add_argument("--md", default=os.path.join(ROOT, "reports", "eval_rules.md"))
    a = ap.parse_args()
    print(report_rules(a.db, a.hits, a.md))
