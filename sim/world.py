"""월드 정적 데이터: 사냥터와 아이템."""
import numpy as np
import pandas as pd


def make_maps(n_maps: int) -> pd.DataFrame:
    ids = np.arange(1, n_maps + 1)
    return pd.DataFrame({
        "map_id": ids.astype("int16"),
        "min_level": ((ids - 1) * 6 + 1).astype("int16"),
        # 킬당 평균 골드: 상위 사냥터일수록 많다
        "gold_per_kill": (40 * 1.35 ** (ids - 1)).round().astype("int32"),
    })


GRADE_BASE = {1: 2_000, 2: 20_000, 3: 200_000, 4: 2_000_000}


def make_items(n_items: int, rng: np.random.Generator) -> pd.DataFrame:
    grade = rng.choice([1, 2, 3, 4], size=n_items, p=[0.45, 0.30, 0.18, 0.07])
    base = np.array([GRADE_BASE[g] for g in grade]) * rng.lognormal(0, 0.35, n_items)
    return pd.DataFrame({
        "item_id": np.arange(1, n_items + 1, dtype="int32"),
        "grade": grade.astype("int8"),
        "base_price": base.round(-1).astype("int64"),
    })


# 시간대별 접속 시작 가중치 (0~23시). 저녁 피크.
HOUR_WEIGHTS = np.array([3, 2, 1.2, 0.8, 0.6, 0.6, 0.8, 1.2, 1.8, 2.2, 2.5, 2.8,
                         3.2, 3.0, 2.8, 2.8, 3.0, 3.5, 4.5, 5.5, 6.5, 7.0, 6.0, 4.5])
HOUR_P = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()
