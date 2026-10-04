"""에이전트 도구 안전장치: 어떤 SQL 로도 정답에 닿지 못하고, 쓰기가 불가능해야 한다."""
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gameguard.agent.tools import Toolbox  # noqa: E402
from gameguard.db import load  # noqa: E402


@pytest.fixture(scope="module")
def tb(tmp_path_factory):
    out = tmp_path_factory.mktemp("agent_raw")
    subprocess.run([sys.executable, "-m", "sim.generate", "--config", "config/sim_small.yaml", "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    db = str(out / "a.duckdb")
    load(str(out), db)
    acc = pd.read_parquet(out / "accounts.parquet")[["account_id"]]
    return Toolbox(db, acc.assign(dummy=0), acc.head(0)), out


@pytest.mark.parametrize("q", [
    "SELECT * FROM labels.account_labels",
    "SELECT * FROM query_table('labels.account_labels')",
    "SELECT * FROM read_parquet('{out}/labels/account_labels.parquet')",
    "SELECT * FROM '{out}/labels/account_labels.parquet'",
    "WITH x AS (SELECT 1) SELECT * FROM x; ATTACH '{out}/a_labels.duckdb' AS l",
    "DELETE FROM trades",
])
def test_sql_cannot_reach_labels_or_write(tb, q):
    t, out = tb
    res = t.run_readonly_sql(q.format(out=str(out).replace("\\", "/")))
    assert "rows" not in res, res


def test_detection_db_has_no_label_tables(tb):
    t, _ = tb
    names = {r[0] for r in t.con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert not names & {"account_labels", "item_truth", "macro_sessions"}


def test_external_access_disabled_even_without_filter(tb):
    t, out = tb
    with pytest.raises(Exception):
        t.con.execute(f"SELECT * FROM read_parquet('{str(out).replace(chr(92), '/')}/labels/account_labels.parquet')").df()


def test_plain_select_works(tb):
    t, _ = tb
    assert t.run_readonly_sql("SELECT count(*) AS n FROM trades")["rows"][0]["n"] > 0
