"""DuckDB 적재와 연결.

공개 테이블은 main 스키마, 정답은 labels 스키마에 둔다.
탐지 코드(sql/rules, sql/features)는 labels 스키마를 절대 참조하지 않는다 — tests/test_no_label_leak.py 가 검사.

  python -m gameguard.db load --raw data/raw --db data/gameguard.duckdb
"""
import argparse
import glob
import os

import duckdb

PUBLIC = ["accounts", "sessions", "actions", "currency_log", "trades",
          "market_listings", "payments", "items", "maps"]
LABELS = ["account_labels", "item_truth", "macro_sessions"]


def connect(db_path: str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=read_only)


def load(raw: str, db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    con = connect(db_path)
    for t in PUBLIC:
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet(?)", [os.path.join(raw, f"{t}.parquet")])
    con.execute("CREATE SCHEMA labels")
    for t in LABELS:
        con.execute(f"CREATE TABLE labels.{t} AS SELECT * FROM read_parquet(?)",
                    [os.path.join(raw, "labels", f"{t}.parquet")])
    for t in PUBLIC:
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"{t:16s} {n:>12,}")
    con.close()


def run_sql_file(con, path: str, **params):
    """SQL 파일 실행. {name} 자리표시자는 params 로 치환 (숫자·식별자 설정값 전용)."""
    sql = open(path, encoding="utf-8").read()
    if params:
        sql = sql.format(**params)
    return con.execute(sql)


def sql_files(pattern: str):
    return sorted(glob.glob(pattern))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["load"])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--db", default="data/gameguard.duckdb")
    a = ap.parse_args()
    load(a.raw, a.db)
