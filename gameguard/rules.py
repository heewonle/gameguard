"""SQL 룰 실행 → 룰 히트 테이블.

  python -m gameguard.rules --db data/gameguard.duckdb --out data/out
"""
import argparse
import os
import time

import pandas as pd
import yaml

from .db import connect, run_sql_file, sql_files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_rules(con, rules_cfg: dict, only=None, verbose=True) -> pd.DataFrame:
    out = []
    for f in sql_files(os.path.join(ROOT, "sql", "rules", "R*.sql")):
        name = os.path.splitext(os.path.basename(f))[0]
        if only and not any(name.startswith(o) for o in only):
            continue
        t0 = time.time()
        df = run_sql_file(con, f, **rules_cfg.get(name, {})).df()
        if verbose:
            print(f"  {name:24s} {len(df):6,} hits  {time.time() - t0:5.1f}s", flush=True)
        out.append(df)
    hits = pd.concat(out, ignore_index=True)
    hits["evidence"] = hits["evidence"].astype(str)
    return hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gameguard.duckdb")
    ap.add_argument("--rules", default=os.path.join(ROOT, "config", "rules.yaml"))
    ap.add_argument("--out", default="data/out")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.rules, encoding="utf-8"))
    hits = run_rules(connect(a.db, read_only=True), cfg, a.only)
    os.makedirs(a.out, exist_ok=True)
    hits.to_parquet(os.path.join(a.out, "rule_hits.parquet"), index=False)
    print(f"total hits {len(hits):,} / accounts {hits['account_id'].nunique():,}")
