"""생성기·적재·품질 테스트. 소규모 월드를 한 번 만들어 공유한다."""
import glob
import os
import re
import subprocess
import sys

import duckdb
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gameguard.db import load  # noqa: E402
from gameguard.quality import run_quality  # noqa: E402


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    out = tmp_path_factory.mktemp("raw")
    subprocess.run([sys.executable, "-m", "sim.generate", "--config", "config/sim_small.yaml", "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    db = str(out / "t.duckdb")
    load(str(out), db)
    return out, db


def test_quality_checks_pass(world):
    _, db = world
    res = run_quality(duckdb.connect(db, read_only=True))
    assert res["n_bad"].sum() == 0, res.to_string()


def test_every_label_present(world):
    out, _ = world
    lab = pd.read_parquet(out / "labels" / "account_labels.parquet")
    assert set(lab["label"]) == {"normal", "farm_bot", "rmt_ring", "macro", "multi_account",
                                 "market_manip", "chargeback"}


def test_ids_do_not_reveal_label(world):
    """account_id 순서와 라벨이 무관해야 한다 (ID 정렬만으로 봇을 찾을 수 없게)."""
    out, _ = world
    lab = pd.read_parquet(out / "labels" / "account_labels.parquet").sort_values("account_id")
    bots = (lab["label"] == "farm_bot").to_numpy()
    first_half = bots[: len(bots) // 2].sum()
    assert 0.25 < first_half / bots.sum() < 0.75


def test_public_tables_have_no_label_columns(world):
    out, _ = world
    for f in glob.glob(str(out / "*.parquet")):
        cols = set(pd.read_parquet(f).columns)
        assert not cols & {"label", "kind", "role", "ring_id", "is_macro", "base_price"}, f


def test_reproducible(tmp_path):
    outs = []
    for k in range(2):
        o = tmp_path / f"r{k}"
        subprocess.run([sys.executable, "-m", "sim.generate", "--config", "config/sim_small.yaml", "--out", str(o)],
                       cwd=ROOT, check=True, capture_output=True)
        outs.append(pd.read_parquet(o / "trades.parquet"))
    pd.testing.assert_frame_equal(outs[0], outs[1])


DETECTION_DIRS = ["sql/rules", "sql/features", "gameguard"]
ALLOWED = {"gameguard/db.py", "gameguard/evaluate.py"}   # 적재·평가만 정답을 읽을 수 있다


def test_no_label_leak():
    """탐지 코드가 labels 스키마나 정답 파일을 참조하지 않는다."""
    pat = re.compile(r"labels\.|account_labels|item_truth|macro_sessions")
    bad = []
    for d in DETECTION_DIRS:
        for f in glob.glob(os.path.join(ROOT, d, "**", "*.*"), recursive=True):
            rel = os.path.relpath(f, ROOT).replace("\\", "/")
            if rel in ALLOWED or not rel.endswith((".py", ".sql")):
                continue
            if pat.search(open(f, encoding="utf-8").read()):
                bad.append(rel)
    assert not bad, bad
