"""최종 경보 테이블: 1·2·3단 결과를 계정당 한 행으로 모으고 단계별 근거를 붙인다.

  python -m gameguard.alerts --out data/out_test
"""
import argparse
import json
import os

import pandas as pd


def build_alerts(out_dir: str) -> pd.DataFrame:
    rules = pd.read_parquet(os.path.join(out_dir, "rule_hits.parquet"))
    ml = pd.read_parquet(os.path.join(out_dir, "ml_scores_sanctioned.parquet"))
    fi = pd.read_parquet(os.path.join(out_dir, "graph_fanin.parquet"))
    pr = pd.read_parquet(os.path.join(out_dir, "graph_propagation.parquet"))

    rows = {}

    def add(aid, source, detail):
        r = rows.setdefault(int(aid), {"account_id": int(aid), "sources": [], "details": []})
        if source not in r["sources"]:
            r["sources"].append(source)
        r["details"].append(detail)

    for t in rules.itertuples():
        add(t.account_id, "rule", {"stage": "rule", "rule_id": t.rule_id, "score": round(float(t.score), 3),
                                   "evidence": json.loads(t.evidence)})
    for t in ml[ml["ml_flag"]].itertuples():
        add(t.account_id, "ml", {"stage": "ml", "ml_score": round(float(t.ml_score), 4)})
    for t in fi.itertuples():
        add(t.account_id, "graph_fanin", {"stage": "graph", "signal": t.signal, "evidence": json.loads(t.evidence)})
    for t in pr[pr["flag"]].itertuples():
        add(t.account_id, "graph_propagation", {"stage": "graph", "signal": "propagation",
                                                "seed_share": round(float(t.seed_share), 3),
                                                "seed_neighbors": int(t.seed_neighbors)})
    df = pd.DataFrame(rows.values())
    df["sources"] = df["sources"].map(lambda s: ",".join(s))
    df["details"] = df["details"].map(lambda d: json.dumps(d, ensure_ascii=False))
    ms = ml.set_index("account_id")["ml_score"]
    df["ml_score"] = df["account_id"].map(ms).fillna(0.0)
    return df.sort_values("account_id").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/out_test")
    a = ap.parse_args()
    al = build_alerts(a.out)
    al.to_parquet(os.path.join(a.out, "alerts.parquet"), index=False)
    print(f"alerts {len(al):,}")
    print(al["sources"].value_counts().head(10).to_string())
