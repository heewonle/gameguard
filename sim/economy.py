"""재화 흐름: 드랍·퀘스트·상점, 거래(1:1·거래소), 결제/환불, 그리고 어뷰징 시나리오.

거래 테이블 의미: buyer 가 seller 에게 gold 를 내고, seller 는 item_id 를 qty 만큼 준다.
item_id = 0 이면 아이템 없이 골드만 이동(선물·송금). gold 이동 방향은 항상 buyer → seller.
"""
import numpy as np
import pandas as pd

MARKET_FEE = 0.05


class Ledger:
    """거래·통화 이벤트를 모아 두었다가 DataFrame 으로 만든다."""

    def __init__(self):
        self.cur, self.trades, self.listings, self.pays = [], [], [], []
        self._tid = 9_000_000
        self._lid = 7_000_000

    def currency(self, ts, acc, delta, reason):
        self.cur.append(pd.DataFrame({"ts": ts, "account_id": acc, "delta": delta, "reason": reason}))

    def trade(self, ts, seller, buyer, item, qty, gold, channel):
        n = len(ts)
        if n == 0:
            return None
        tid = np.arange(self._tid, self._tid + n)
        self._tid += n
        gold = np.asarray(gold).round().astype("int64")
        self.trades.append(pd.DataFrame({"ts": ts, "trade_id": tid, "seller": seller, "buyer": buyer,
                                         "item_id": item, "qty": qty, "gold": gold, "channel": channel}))
        fee = MARKET_FEE if channel == "market" else 0.0
        self.currency(ts, buyer, -gold, channel)
        self.currency(ts, seller, (gold * (1 - fee)).round().astype("int64"), channel)
        return tid

    def listing(self, ts, seller, item, price, status, trade_id=None):
        n = len(ts)
        lid = np.arange(self._lid, self._lid + n)
        self._lid += n
        self.listings.append(pd.DataFrame({"ts": ts, "listing_id": lid, "seller": seller, "item_id": item,
                                           "price": np.asarray(price).round().astype("int64"),
                                           "status": status,
                                           "trade_id": trade_id if trade_id is not None else pd.NA}))

    def payment(self, ts, pid, acc, krw, event):
        self.pays.append(pd.DataFrame({"ts": ts, "payment_id": pid, "account_id": acc,
                                       "amount_krw": krw, "event": event}))


def _rand_ts_in(sess_rows: pd.DataFrame, rng, lo=0.0, hi=1.0):
    """세션들 안의 무작위 시각 (세션 길이의 lo~hi 구간)."""
    dur = (sess_rows["logout_at"] - sess_rows["login_at"]).dt.total_seconds().to_numpy()
    u = rng.uniform(lo, hi, len(sess_rows))
    return (sess_rows["login_at"] + pd.to_timedelta(dur * u, unit="s")).dt.floor("s").to_numpy()


def build_economy(acc, sess, actions, maps, items, cfg, rng, start):
    L = Ledger()
    aid = acc["account_id"].to_numpy()
    kind = acc["kind"].to_numpy()
    role = acc["role"].to_numpy()
    grp = acc["ring_id"].to_numpy()
    days = cfg["days"]
    base_price = items.set_index("item_id")["base_price"]
    grade = items.set_index("item_id")["grade"]
    junk = items.loc[items["grade"] == 1, "item_id"].to_numpy()

    sess = sess.copy()
    sess["account_id"] = aid[sess["acc_i"]]
    sess["kind"] = kind[sess["acc_i"]]
    sess["role"] = role[sess["acc_i"]]
    sess["day"] = ((sess["login_at"] - start).dt.days).astype(int)

    # ── 드랍 골드: 세션별 킬 × 사냥터 킬당 골드 ──
    gpk = maps.set_index("map_id")["gold_per_kill"].reindex(range(0, maps["map_id"].max() + 1)).fillna(0).to_numpy()
    kill = actions["action_type"].to_numpy() == "kill"
    w = gpk[actions["map_id"].to_numpy()] * kill * rng.lognormal(0, 0.3, len(actions))
    drop = np.bincount(actions["_sess_i"].to_numpy(), weights=w, minlength=len(sess)).round()
    sess["drop"] = drop
    end_ts = sess["logout_at"].to_numpy()
    L.currency(end_ts, sess["account_id"].to_numpy(), drop.astype("int64"), "drop")

    # 퀘스트 보상 (사람 세션 30%) / 상점 소비
    human = ~np.isin(sess["kind"].to_numpy(), ["farm_bot"]) & (sess["role"].to_numpy() != "alt") & ~sess["is_macro"].to_numpy()
    q = human & (rng.random(len(sess)) < 0.3)
    lvl = acc["level"].to_numpy()[sess["acc_i"]]
    L.currency(end_ts[q] - np.timedelta64(60, "s"), sess["account_id"].to_numpy()[q],
               (lvl[q] * 500 * rng.lognormal(0, 0.4, q.sum())).round().astype("int64"), "quest")
    spend = np.where(human, rng.uniform(0.3, 0.6, len(sess)), rng.uniform(0.02, 0.08, len(sess))) * drop
    sp = spend > 0
    L.currency(end_ts[sp] - np.timedelta64(120, "s"), sess["account_id"].to_numpy()[sp],
               -spend[sp].round().astype("int64"), "shop")

    # 하루 대표 세션(계정·날짜별 첫 세션) — 거래 시각 뽑기용
    first_sess = sess[~sess["is_macro"]].drop_duplicates(["account_id", "day"])
    normal_like = first_sess[first_sess["kind"].isin(["normal", "macro", "market_manip"]) |
                             first_sess["role"].isin(["buyer", "main"])]

    # ── 정상 1:1 거래 (길드원 간) ──
    gid_of = pd.Series(acc["guild_id"].to_numpy(), index=aid)
    members = acc[acc["guild_id"] > 0].groupby("guild_id")["account_id"].apply(np.array)
    t = normal_like[rng.random(len(normal_like)) < 0.04]
    if len(t):
        g = gid_of.loc[t["account_id"]].to_numpy()
        partner = np.array([rng.choice(members.get(x, np.array([a]))) if x > 0 else a
                            for x, a in zip(g, t["account_id"].to_numpy())])
        ok = partner != t["account_id"].to_numpy()
        t, partner = t[ok], partner[ok]
        it = rng.choice(items.loc[items["grade"] <= 3, "item_id"].to_numpy(), len(t))
        L.trade(_rand_ts_in(t, rng), partner, t["account_id"].to_numpy(), it, 1,
                base_price.loc[it].to_numpy() * rng.lognormal(0, 0.25, len(t)), "p2p")

    # ── 길드장 선물 (헷갈리는 정상: 여러 명이 한 명에게 골드를 보냄) ──
    leader = acc.loc[role == "guild_leader"].set_index("guild_id")["account_id"]
    t = normal_like[(rng.random(len(normal_like)) < 0.012)]
    g = gid_of.loc[t["account_id"]].to_numpy()
    has = np.isin(g, leader.index.to_numpy())
    t, g = t[has], g[has]
    ld = leader.loc[g].to_numpy()
    ok = ld != t["account_id"].to_numpy()
    t, ld = t[ok], ld[ok]
    L.trade(_rand_ts_in(t, rng), ld, t["account_id"].to_numpy(), 0, 0,
            rng.uniform(10_000, 100_000, len(t)), "p2p")

    # ── 정상 거래소 구매 ──
    t = normal_like[rng.random(len(normal_like)) < 0.15]
    sellers_pool = acc.loc[kind == "normal", "account_id"].to_numpy()
    sel = rng.choice(sellers_pool, len(t))
    ok = sel != t["account_id"].to_numpy()
    t, sel = t[ok], sel[ok]
    blvl = acc.set_index("account_id").loc[t["account_id"], "level"].to_numpy()
    gmax = np.clip(blvl // 15 + 1, 1, 4)
    it = np.array([rng.choice(items.loc[items["grade"] <= m, "item_id"].to_numpy()) for m in gmax])
    ts = _rand_ts_in(t, rng)
    price = base_price.loc[it].to_numpy() * rng.lognormal(0, 0.15, len(t))
    tid = L.trade(ts, sel, t["account_id"].to_numpy(), it, 1, price, "market")
    L.listing(ts - pd.to_timedelta(rng.uniform(0.5, 48, len(t)), unit="h").to_numpy(), sel, it, price, "sold", tid)
    # 안 팔린 매물
    k = int(len(t) * 0.4)
    it2 = rng.choice(items["item_id"].to_numpy(), k)
    L.listing((start + pd.to_timedelta(rng.uniform(0, days, k), unit="D")).floor("s").to_numpy(),
              rng.choice(sellers_pool, k), it2, base_price.loc[it2].to_numpy() * rng.lognormal(0.2, 0.2, k), "expired")

    # ── 작업장 링 ──
    for rid in np.unique(grp[kind == "farm_bot"]):
        coll = aid[(grp == rid) & (role == "collector")]
        selr = aid[(grp == rid) & (role == "seller")]
        buyers = aid[(grp == rid) & (role == "buyer")]
        bs = sess[(sess["kind"] == "farm_bot") & np.isin(sess["account_id"], aid[grp == rid])]
        # 봇 → 수거: 세션 끝 무렵 드랍의 85~95%
        to = rng.choice(coll, len(bs))
        amt = bs["drop"].to_numpy() * rng.uniform(0.85, 0.95, len(bs))
        tsb = bs["logout_at"].to_numpy() - np.timedelta64(30, "s")
        it = np.where(rng.random(len(bs)) < 0.5, rng.choice(junk, len(bs)), 0)
        L.trade(tsb, to, bs["account_id"].to_numpy(), it, (it > 0).astype(int), amt, "p2p")
        # 수거 → 판매: 매일 23시대, 받은 금액의 90%
        daily = pd.DataFrame({"c": to, "d": bs["day"].to_numpy(), "amt": amt}).groupby(["c", "d"], as_index=False)["amt"].sum()
        tsc = (start + pd.to_timedelta(daily["d"], unit="D") + pd.Timedelta(hours=23)
               + pd.to_timedelta(rng.uniform(0, 3000, len(daily)), unit="s")).dt.floor("s").to_numpy()
        L.trade(tsc, rng.choice(selr, len(daily)), daily["c"].to_numpy(), 0, 0, daily["amt"].to_numpy() * 0.9, "p2p")
        # 판매 → RMT 구매자: 구매자가 잡템을 넘기고 판매자가 큰 골드를 지불 (단가 이상치)
        bsess = first_sess[np.isin(first_sess["account_id"], buyers)]
        npur = rng.integers(1, 5, len(buyers))
        rows = []
        for b, n in zip(buyers, npur):
            cand = bsess[bsess["account_id"] == b]
            if len(cand):
                rows.append(cand.sample(min(n, len(cand)), random_state=int(rng.integers(1 << 30))))
        if rows:
            bt = pd.concat(rows)
            it = rng.choice(junk, len(bt))
            L.trade(_rand_ts_in(bt, rng), bt["account_id"].to_numpy(), rng.choice(selr, len(bt)), it, 1,
                    rng.uniform(200_000, 3_000_000, len(bt)), "p2p")

    # ── 다계정: 부계정 → 본계정 송금 ──
    for gid_ in np.unique(grp[kind == "multi_account"]):
        main = aid[(grp == gid_) & (role == "main")][0]
        alts = sess[(sess["role"] == "alt") & np.isin(sess["account_id"], aid[grp == gid_])]
        alts = alts[alts["drop"] > 0]
        L.trade(alts["logout_at"].to_numpy() - np.timedelta64(20, "s"), np.full(len(alts), main),
                alts["account_id"].to_numpy(), 0, 0, alts["drop"].to_numpy() * rng.uniform(0.9, 0.98, len(alts)), "p2p")

    # ── 시세조작: 매집 → 고가 재등록 → 자전거래 ──
    g3 = items.loc[items["grade"] == 3, "item_id"].to_numpy()
    for gid_ in np.unique(grp[kind == "market_manip"]):
        mem = aid[grp == gid_]
        target = int(rng.choice(g3))
        bp = base_price.loc[target]
        d0 = int(rng.integers(3, max(4, days - 6)))
        n = int(rng.integers(30, 61))
        ts = (start + pd.Timedelta(days=d0) + pd.to_timedelta(rng.uniform(0, 48, n), unit="h")).floor("s").to_numpy()
        sel = rng.choice(sellers_pool, n)
        price = bp * rng.lognormal(0, 0.1, n)
        tid = L.trade(ts, sel, rng.choice(mem, n), np.full(n, target), 1, price, "market")
        L.listing(ts - np.timedelta64(3600, "s"), sel, np.full(n, target), price, "sold", tid)
        mult = rng.uniform(2.5, 4.0)
        k = int(rng.integers(10, 21))
        ts2 = (start + pd.Timedelta(days=d0 + 2) + pd.to_timedelta(rng.uniform(0, 72, k), unit="h")).floor("s").to_numpy()
        s2 = rng.choice(mem, k)
        p2 = bp * mult * rng.lognormal(0, 0.05, k)
        buyers2 = rng.choice(acc.loc[kind == "normal", "account_id"].to_numpy(), k)
        tid2 = L.trade(ts2, s2, buyers2, np.full(k, target), 1, p2, "market")
        L.listing(ts2 - np.timedelta64(7200, "s"), s2, np.full(k, target), p2, "sold", tid2)
        L.listing(ts2, rng.choice(mem, k), np.full(k, target), bp * mult * rng.lognormal(0, 0.05, k), "active")
        # 자전거래: 같은 쌍이 같은 아이템을 반복 왕복
        a_, b_ = mem[0], mem[1]
        w = int(rng.integers(5, 16))
        tsw = np.sort((start + pd.Timedelta(days=d0 + 1) + pd.to_timedelta(rng.uniform(0, 96, w), unit="h")).floor("s").to_numpy())
        sw = np.where(np.arange(w) % 2 == 0, a_, b_)
        bw = np.where(np.arange(w) % 2 == 0, b_, a_)
        L.trade(tsw, sw, bw, np.full(w, target), 1, bp * mult * rng.lognormal(0, 0.03, w), "p2p")

    # ── 결제: 정상 과금 유저 ──
    gpk_krw = cfg["world"]["gold_per_krw"]
    pid = [3_000_000]

    def pay(rows, krw, refund_after_h=None):
        n = len(rows)
        if n == 0:
            return None
        ids = np.arange(pid[0], pid[0] + n)
        pid[0] += n
        ts = _rand_ts_in(rows, rng)
        L.payment(ts, ids, rows["account_id"].to_numpy(), krw, "paid")
        L.currency(ts, rows["account_id"].to_numpy(), (krw * gpk_krw).astype("int64"), "purchase")
        if refund_after_h is not None:
            m = ~np.isnan(refund_after_h)
            rts = (ts[m] + pd.to_timedelta(refund_after_h[m], unit="h").to_numpy()).astype("datetime64[s]")
            L.payment(rts, ids[m], rows["account_id"].to_numpy()[m], krw[m], "refunded")
            L.currency(rts, rows["account_id"].to_numpy()[m], -(krw[m] * gpk_krw).astype("int64"), "refund")
        return ts

    payers = acc.loc[(kind == "normal") & (rng.random(len(acc)) < cfg["normal"]["payer_frac"]), "account_id"].to_numpy()
    ps = normal_like[np.isin(normal_like["account_id"], payers) & (rng.random(len(normal_like)) < 0.25)]
    krw = rng.choice([5_000, 11_000, 33_000, 55_000], len(ps)).astype("int64")
    rf = np.where(rng.random(len(ps)) < cfg["normal"]["legit_refund_rate"], rng.uniform(2, 7 * 24, len(ps)), np.nan)
    pay(ps, krw, rf)

    # ── 차지백: 결제 → 1~12시간 내 수령자에게 골드 이전 → 1~14일 뒤 환불 ──
    cb_sess = first_sess[first_sess["role"] == "payer"]
    for gid_ in np.unique(grp[kind == "chargeback"]):
        recv = aid[(grp == gid_) & (role == "receiver")][0]
        rows = cb_sess[np.isin(cb_sess["account_id"], aid[(grp == gid_) & (role == "payer")])]
        if len(rows) == 0:
            continue
        rows = rows.sample(min(len(rows), int(rng.integers(1, 4)) * 2), random_state=int(rng.integers(1 << 30)))
        krw = rng.choice([55_000, 110_000, 330_000], len(rows)).astype("int64")
        ts = pay(rows, krw, rng.uniform(24, 14 * 24, len(rows)))
        tts = (ts + pd.to_timedelta(rng.uniform(1, 12, len(rows)), unit="h").to_numpy()).astype("datetime64[s]")
        L.trade(tts, np.full(len(rows), recv), rows["account_id"].to_numpy(), 0, 0,
                krw * gpk_krw * rng.uniform(0.85, 0.95, len(rows)), "p2p")

    return L
