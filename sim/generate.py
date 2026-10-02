"""합성 MMORPG 로그 생성.

  python -m sim.generate                         # config/sim.yaml → data/raw/
  python -m sim.generate --config config/sim_small.yaml --out data/raw_small

출력 (Parquet)
  공개 테이블: accounts, sessions, actions, currency_log, trades, market_listings, payments, items, maps
  정답 (labels/): account_labels, item_truth  ← 탐지 코드는 읽으면 안 된다
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import yaml

from .actors import make_accounts
from .activity import make_actions, make_sessions
from .economy import build_economy
from .world import make_items, make_maps


def finalize_currency(L, acc, start, end):
    cur = pd.concat(L.cur, ignore_index=True)
    cur["ts"] = pd.to_datetime(cur["ts"]).astype("datetime64[ns]")
    cur = cur[(cur["ts"] < end) & (cur["delta"] != 0)]
    # 기초 잔액 행: 그 계정의 첫 이벤트 직전 (기간 이전 생성 계정은 기간 시작 직전)
    first_ev = cur.groupby("account_id")["ts"].min()
    start_ts = acc.set_index("account_id")["created_at"].clip(lower=start - pd.Timedelta(seconds=1))
    start_ts = pd.concat([start_ts, first_ev - pd.Timedelta(seconds=1)], axis=1).min(axis=1)
    init = pd.DataFrame({"ts": start_ts.reindex(acc["account_id"]).to_numpy(), "account_id": acc["account_id"].to_numpy(),
                         "delta": 0, "reason": "carryover"})
    cur = pd.concat([init, cur], ignore_index=True)
    cur["_o"] = np.arange(len(cur))
    cur = cur.sort_values(["account_id", "ts", "_o"], kind="stable").reset_index(drop=True)

    # 잔액이 음수가 되지 않도록 기초 잔액을 정한다. 단, 차지백 결제자는 환불로 음수가 될 수 있다.
    run_all = cur.groupby("account_id")["delta"].cumsum()
    nonref = cur["delta"].where(cur["reason"] != "refund", 0)
    run_nr = nonref.groupby(cur["account_id"]).cumsum()
    cb = set(acc.loc[acc["role"].eq("payer"), "account_id"])
    is_cb = cur["account_id"].isin(cb)
    need = np.where(is_cb, -run_nr, -run_all)
    shift = pd.Series(need).groupby(cur["account_id"].to_numpy()).max().clip(lower=0)
    rng = np.random.default_rng(0)
    base = pd.Series(rng.lognormal(np.log(200_000), 1.0, len(shift)).round(), index=shift.index)
    base[base.index.isin(cb)] = rng.uniform(0, 5_000, base.index.isin(cb).sum()).round()
    init_amt = (shift + base).round().astype("int64")
    first = cur["reason"].eq("carryover")
    cur.loc[first, "delta"] = init_amt.loc[cur.loc[first, "account_id"]].to_numpy()
    cur["balance_after"] = cur.groupby("account_id")["delta"].cumsum().astype("int64")
    cur["delta"] = cur["delta"].astype("int64")
    cur["log_id"] = np.arange(1, len(cur) + 1, dtype="int64")
    return cur.drop(columns="_o")[["log_id", "ts", "account_id", "delta", "balance_after", "reason"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sim.yaml")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, help="설정 파일의 seed 대신 사용 (평가용 월드)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    rng = np.random.default_rng(cfg["seed"])
    start = pd.Timestamp(cfg["start"])
    end = start + pd.Timedelta(days=cfg["days"])
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    maps = make_maps(cfg["world"]["n_maps"])
    items = make_items(cfg["world"]["n_items"], rng)
    acc = make_accounts(cfg, rng, start)
    log(f"accounts {len(acc):,}")
    sess = make_sessions(acc, cfg, rng, start)
    log(f"sessions {len(sess):,}")
    actions = make_actions(sess, acc, cfg, rng)
    log(f"actions {len(actions):,}")
    L = build_economy(acc, sess, actions, maps, items, cfg, rng, start)
    cur = finalize_currency(L, acc, start, end)
    trades = pd.concat(L.trades, ignore_index=True)
    trades["ts"] = pd.to_datetime(trades["ts"]).astype("datetime64[ns]")
    trades = trades[trades["ts"] < end].sort_values("ts", kind="stable")
    for c in ("item_id", "qty"):
        trades[c] = trades[c].astype("int32")
    listings = pd.concat(L.listings, ignore_index=True)
    listings["ts"] = pd.to_datetime(listings["ts"]).astype("datetime64[ns]")
    listings = listings[listings["ts"] < end].sort_values("ts", kind="stable")
    listings["trade_id"] = listings["trade_id"].astype("Int64")
    pays = pd.concat(L.pays, ignore_index=True)
    pays["ts"] = pd.to_datetime(pays["ts"]).astype("datetime64[ns]")
    pays = pays[pays["ts"] < end].sort_values("ts", kind="stable")
    log(f"currency {len(cur):,} trades {len(trades):,} listings {len(listings):,} payments {len(pays):,}")

    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
    w = lambda df, name: df.to_parquet(os.path.join(args.out, f"{name}.parquet"), index=False)

    w(acc[["account_id", "created_at", "country", "device_id", "ip_hash", "guild_id", "level"]]
      .sort_values("account_id"), "accounts")
    sess_out = sess.assign(account_id=acc["account_id"].to_numpy()[sess["acc_i"]])
    w(sess_out[["session_id", "account_id", "login_at", "logout_at", "ip_hash", "device_id"]], "sessions")
    w(actions.drop(columns="_sess_i").sort_values(["ts"], kind="stable"), "actions")
    w(cur, "currency_log")
    w(trades, "trades")
    w(listings, "market_listings")
    w(pays, "payments")
    w(items[["item_id", "grade"]], "items")
    w(maps[["map_id", "min_level"]], "maps")
    # 정답
    lab = acc[["account_id", "kind", "role", "ring_id", "stealth"]].rename(columns={"kind": "label"})
    lab.to_parquet(os.path.join(args.out, "labels", "account_labels.parquet"), index=False)
    items.to_parquet(os.path.join(args.out, "labels", "item_truth.parquet"), index=False)
    macro_sess = sess_out.loc[sess["is_macro"].to_numpy(), ["session_id", "account_id"]]
    macro_sess.to_parquet(os.path.join(args.out, "labels", "macro_sessions.parquet"), index=False)

    summary = {"config": args.config, "accounts": len(acc), "sessions": len(sess), "actions": len(actions),
               "currency_log": len(cur), "trades": len(trades), "market_listings": len(listings),
               "payments": len(pays), "labels": lab["label"].value_counts().to_dict()}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log("done " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
