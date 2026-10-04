"""2단: 계정 피처 + ML.

- LightGBM (지도), 라벨 두 가지
    oracle     개발 월드의 모든 정답으로 학습 — 상한선 (합성 데이터라 지나치게 쉽다)
    sanctioned 개발 월드에서 "룰에 걸려 확정된" 계정만 양성으로 학습 — 실서비스와 같은 조건.
               놓친 어뷰저는 정상으로 섞여 들어간다. 은닉형을 본 적 없는 모델이 그것을 찾는가?
- IsolationForest (비지도): 라벨 없이 동작하는 기준선.
임계값은 개발 월드의 out-of-fold 예측으로 정한다 — 평가 월드는 점수만 매긴다.

  python -m gameguard.models train --mode sanctioned
  python -m gameguard.models score --mode sanctioned --db data/test.duckdb --out data/out_test
"""
import argparse
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold

from .db import connect, run_sql_file
from .evaluate import load_labels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_SQL = os.path.join(ROOT, "sql", "features", "account_features.sql")
MODEL_DIR = os.path.join(ROOT, "data", "models")
REVIEW_BUDGET = 300   # 30일치 데이터에서 검수팀이 룰 외에 추가로 볼 수 있는 경보 수 (가정)
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)


def build_features(db: str) -> pd.DataFrame:
    return run_sql_file(connect(db, read_only=True), FEATURE_SQL).df()


def _xy(feat: pd.DataFrame):
    X = feat.drop(columns=["account_id"]).astype(float)
    return X


def _best_threshold(oof: np.ndarray, y: np.ndarray, rule_flag: np.ndarray):
    """룰 ∪ ML 합집합의 F1 이 가장 높은 ML 임계값 (개발 월드 OOF 기준)."""
    best = (0.0, 0.5)
    for th in np.linspace(0.05, 0.95, 91):
        p = rule_flag | (oof >= th)
        tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, th)
    return best


def train(db: str, hits_path: str, mode: str) -> dict:
    feat = build_features(db)
    lab = load_labels(db).set_index("account_id").loc[feat["account_id"]]
    X = _xy(feat)
    rule_flag = feat["account_id"].isin(set(pd.read_parquet(hits_path)["account_id"])).to_numpy()
    y = (lab["label"] != "normal").to_numpy()
    if mode == "sanctioned":
        y = y & rule_flag          # 룰이 잡고 검수로 확정된 것만 안다

    oof = np.zeros(len(X))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    if mode == "sanctioned":
        # 학습 라벨이 룰 히트뿐이라 F1 로 고르면 "룰 밖은 다 정상"이 최적이 된다.
        # 대신 검수 인력 예산: 룰에 안 걸린 계정 중 상위 REVIEW_BUDGET 명을 추가 검토 큐로 보낸다.
        resid = np.sort(oof[~rule_flag])[::-1]
        th = float(resid[min(REVIEW_BUDGET, len(resid)) - 1])
        f1 = float("nan")
    else:
        f1, th = _best_threshold(oof, y, rule_flag)

    model = lgb.LGBMClassifier(**LGB_PARAMS).fit(X, y)
    iso = IsolationForest(n_estimators=300, contamination=float(y.mean()), random_state=0)
    iso.fit(X.fillna(X.median()))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"lgb": model, "iso": iso, "columns": list(X.columns), "fill": X.median().to_dict()},
                os.path.join(MODEL_DIR, f"stage2_{mode}.joblib"))
    imp = pd.Series(model.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    meta = {"mode": mode, "threshold": float(th), "dev_oof_union_f1_vs_train_labels": float(f1), "n_train": int(len(X)),
            "n_pos": int(y.sum()), "top_features": imp.head(15).round(0).to_dict()}
    json.dump(meta, open(os.path.join(MODEL_DIR, f"stage2_{mode}_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return meta


def score(db: str, out_dir: str, mode: str) -> pd.DataFrame:
    bundle = joblib.load(os.path.join(MODEL_DIR, f"stage2_{mode}.joblib"))
    meta = json.load(open(os.path.join(MODEL_DIR, f"stage2_{mode}_meta.json"), encoding="utf-8"))
    feat = build_features(db)
    X = _xy(feat)[bundle["columns"]]
    s = pd.DataFrame({"account_id": feat["account_id"]})
    s["ml_score"] = bundle["lgb"].predict_proba(X)[:, 1]
    s["ml_flag"] = s["ml_score"] >= meta["threshold"]
    Xf = X.fillna(pd.Series(bundle["fill"]))
    s["iso_score"] = -bundle["iso"].score_samples(Xf)
    s["iso_flag"] = bundle["iso"].predict(Xf) == -1
    os.makedirs(out_dir, exist_ok=True)
    feat.to_parquet(os.path.join(out_dir, "features.parquet"), index=False)
    s.to_parquet(os.path.join(out_dir, f"ml_scores_{mode}.parquet"), index=False)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--db", default="data/gameguard.duckdb")
    ap.add_argument("--hits", default="data/out_gameguard/rule_hits.parquet")
    ap.add_argument("--out", default="data/out_test")
    ap.add_argument("--mode", choices=["oracle", "sanctioned"], default="sanctioned")
    a = ap.parse_args()
    if a.cmd == "train":
        print(json.dumps(train(a.db, a.hits, a.mode), ensure_ascii=False, indent=2))
    else:
        s = score(a.db, a.out, a.mode)
        print(f"scored {len(s):,} · ml_flag {int(s['ml_flag'].sum()):,} · iso_flag {int(s['iso_flag'].sum()):,}")
