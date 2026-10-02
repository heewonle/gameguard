"""데이터 품질 체크 실행: sql/quality/*.sql 은 각각 (check_name, n_bad) 한 행을 반환한다.

  python -m gameguard.quality --db data/gameguard.duckdb
"""
import argparse
import os

import pandas as pd

from .db import connect, run_sql_file, sql_files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_quality(con) -> pd.DataFrame:
    rows = [run_sql_file(con, f).df() for f in sql_files(os.path.join(ROOT, "sql", "quality", "*.sql"))]
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gameguard.duckdb")
    a = ap.parse_args()
    res = run_quality(connect(a.db, read_only=True))
    print(res.to_string(index=False))
    raise SystemExit(1 if res["n_bad"].sum() else 0)
