"""일일 배치: 실패해도 멈추지 않고, 앞 단계가 실패하면 뒤 단계를 건너뛴다. 조사 우선순위."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gameguard.run import Run, priority  # noqa: E402


def test_failure_skips_dependents(tmp_path):
    R = Run(str(tmp_path))
    R.step("a", lambda: {"x": 1})
    R.step("b", lambda: 1 / 0)
    R.step("c", lambda: {}, needs=["b"])
    R.step("d", lambda: {}, needs=["a"])
    path = R.finish()
    st = {s["step"]: s["status"] for s in R.steps}
    assert st == {"a": "ok", "b": "failed", "c": "skipped", "d": "ok"}
    assert os.path.exists(path)


def test_priority_multi_stage_and_no_rule_first():
    a = pd.DataFrame({"account_id": [1, 2, 3, 4],
                      "sources": ["rule", "rule,ml,graph_fanin", "ml", "rule,ml"],
                      "ml_score": [0.9, 0.1, 0.5, 0.2]})
    assert priority(a)["account_id"].tolist() == [2, 4, 3, 1]
