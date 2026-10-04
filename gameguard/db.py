"""DuckDB 적재와 연결.

공개 테이블은 <db>.duckdb, 정답은 별도 파일 <db>_labels.duckdb 에 둔다.
탐지 DB 파일에는 정답이 물리적으로 없다 — 탐지 코드·LLM 에이전트가 어떤 SQL 을 써도 정답을 읽을 수 없다.
채점(evaluate.py)만 attach_labels() 로 정답 파일을 붙여 labels 스키마로 읽는다.

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
    lpath = labels_path(db_path)
    if os.path.exists(lpath):
        os.remove(lpath)
    lcon = connect(lpath)
    for t in LABELS:
        lcon.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet(?)",
                     [os.path.join(raw, "labels", f"{t}.parquet")])
    lcon.close()
    for t in PUBLIC:
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"{t:16s} {n:>12,}")
    con.close()


def labels_path(db_path: str) -> str:
    return os.path.splitext(db_path)[0] + "_labels.duckdb"


def attach_labels(con, db_path: str) -> None:
    """채점 전용: 정답 파일을 labels 스키마로 붙인다."""
    con.execute(f"ATTACH '{labels_path(db_path)}' AS labels (READ_ONLY)")


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
