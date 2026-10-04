"""계정·행위자 생성. 계정마다 정답 라벨(kind), 역할(role), 그룹(ring_id)과 행동 파라미터를 붙인다.

account_id 는 마지막에 무작위로 섞어서 부여한다 — ID 순서로 유형을 알 수 없게.
"""
import numpy as np
import pandas as pd

# 행동 파라미터 열
#   p_active   하루 접속 확률
#   n_sess     접속한 날 평균 세션 수 (1 + 포아송)
#   dur_med_m  세션 길이 중앙값(분), dur_sd 로그정규 표준편차
#   int_mean   행동 간격 평균(초), int_cv 행동 간격 변동계수
#   single_map True 면 한 사냥터만 반복
#   n_skills   쓰는 스킬 가짓수


def _block(n, kind, role, **kw):
    df = pd.DataFrame({"kind": [kind] * n, "role": role if np.ndim(role) else [role] * n})
    for k, v in kw.items():
        df[k] = v if np.ndim(v) else [v] * n
    return df


def make_accounts(cfg: dict, rng: np.random.Generator, start: pd.Timestamp) -> pd.DataFrame:
    nc, ab = cfg["normal"], cfg["abuse"]
    days = cfg["days"]
    blocks = []

    # ── 정상 유저 ──
    n = nc["n"]
    hard = rng.random(n) < nc["hardcore_frac"]
    normal = _block(n, "normal", np.where(hard, "hardcore", "player"), ring_id="",
                    p_active=np.where(hard, 0.92, rng.beta(2, 3, n)),
                    n_sess=np.where(hard, 2.0, 0.35),
                    dur_med_m=np.where(hard, 170.0, rng.uniform(30, 70, n)),
                    dur_sd=np.where(hard, 0.35, 0.6), int_mean=nc["action_interval_s"] * rng.uniform(0.8, 1.25, n),
                    int_cv=rng.uniform(0.9, 1.6, n), single_map=False, n_skills=8)
    # 길드: guild_size 명씩, 첫 번째가 길드장
    gid = rng.permutation(n) // nc["guild_size"]
    normal["guild_id"] = gid + 1
    first = pd.Series(np.arange(n)).groupby(gid).transform("min").to_numpy()
    normal.loc[np.arange(n) == first, "role"] = "guild_leader"
    blocks.append(normal)

    # ── 작업장 링: 봇 → 수거 → 판매 → RMT 구매자 ──
    for r in range(ab["farm_rings"]):
        rid = f"ring{r + 1:02d}"
        nb = int(rng.integers(15, 26))
        blocks.append(_block(nb, "farm_bot", "bot", ring_id=rid, p_active=0.97, n_sess=0.0,
                             dur_med_m=20 * 60.0, dur_sd=0.05, int_mean=rng.uniform(55, 70, nb),
                             int_cv=rng.uniform(0.05, 0.15, nb), single_map=True, n_skills=2))
        for role in ("collector", "seller"):
            blocks.append(_block(2, "rmt_ring", role, ring_id=rid, p_active=0.85, n_sess=0.3,
                                 dur_med_m=25.0, dur_sd=0.5, int_mean=nnc(nc), int_cv=1.2,
                                 single_map=False, n_skills=8))
        nbuy = int(rng.integers(20, 41))
        blocks.append(_block(nbuy, "rmt_ring", "buyer", ring_id=rid, p_active=rng.beta(2, 3, nbuy),
                             n_sess=0.35, dur_med_m=rng.uniform(30, 70, nbuy), dur_sd=0.6,
                             int_mean=nnc(nc), int_cv=rng.uniform(0.9, 1.6, nbuy),
                             single_map=False, n_skills=8))

    # ── 매크로: 평소엔 정상, 새벽에 고정 주기 반복 세션 ──
    m = ab["macro"]
    blocks.append(_block(m, "macro", "macro_user", ring_id="", p_active=rng.beta(3, 2, m),
                         n_sess=0.35, dur_med_m=rng.uniform(30, 70, m), dur_sd=0.6,
                         int_mean=nnc(nc), int_cv=rng.uniform(0.9, 1.6, m), single_map=False,
                         n_skills=8, macro_hour=rng.integers(1, 6, m), macro_period=rng.uniform(15, 25, m)))

    # ── 다계정: 본계정 1 + 부계정 4~8 (기간 중 생성, 한 기기) ──
    for g in range(ab["multi_account_groups"]):
        gid_ = f"multi{g + 1:03d}"
        blocks.append(_block(1, "multi_account", "main", ring_id=gid_, p_active=0.85, n_sess=1.0,
                             dur_med_m=120.0, dur_sd=0.5, int_mean=nnc(nc), int_cv=1.2,
                             single_map=False, n_skills=8))
        na = int(rng.integers(4, 9))
        blocks.append(_block(na, "multi_account", "alt", ring_id=gid_, p_active=0.8, n_sess=0.5,
                             dur_med_m=rng.uniform(60, 180, na), dur_sd=0.4, int_mean=rng.uniform(70, 100, na),
                             int_cv=rng.uniform(0.4, 0.7, na), single_map=True, n_skills=3))

    # ── 시세조작 그룹 ──
    for g in range(ab["market_manip_groups"]):
        k = int(rng.integers(3, 6))
        blocks.append(_block(k, "market_manip", "manipulator", ring_id=f"manip{g + 1:02d}",
                             p_active=rng.uniform(0.5, 0.8, k), n_sess=0.4, dur_med_m=60.0, dur_sd=0.6,
                             int_mean=nnc(nc), int_cv=1.2, single_map=False, n_skills=8))

    # ── 차지백: 결제자 2 + 수령자 1 씩 ──
    c = ab["chargeback"]
    ng = max(1, c // 3)
    for g in range(ng):
        gid_ = f"cb{g + 1:03d}"
        blocks.append(_block(2, "chargeback", "payer", ring_id=gid_, p_active=0.6, n_sess=0.2,
                             dur_med_m=40.0, dur_sd=0.6, int_mean=nnc(nc), int_cv=1.2,
                             single_map=False, n_skills=8))
        blocks.append(_block(1, "chargeback", "receiver", ring_id=gid_, p_active=0.6, n_sess=0.2,
                             dur_med_m=40.0, dur_sd=0.6, int_mean=nnc(nc), int_cv=1.2,
                             single_map=False, n_skills=8))

    acc = pd.concat(blocks, ignore_index=True)
    N = len(acc)
    for col, default in (("macro_hour", -1), ("macro_period", 0.0), ("guild_id", 0)):
        acc[col] = acc[col].fillna(default) if col in acc else default
    acc["guild_id"] = acc["guild_id"].astype(int)
    acc["macro_hour"] = acc["macro_hour"].astype(int)

    # ── 은닉형(stealth): 룰을 피하도록 사람처럼 행동하는 어뷰저 ──
    # 봇·매크로·RMT 구매자는 계정 단위, 다계정·시세조작·차지백은 그룹 단위로 정한다.
    sf = ab.get("stealth_frac", 0.0)
    kinds_ = acc["kind"].to_numpy()
    stealth = (kinds_ != "normal") & (rng.random(N) < sf)
    for k in ("multi_account", "market_manip", "chargeback"):
        for gid_ in acc.loc[kinds_ == k, "ring_id"].unique():
            m = (acc["ring_id"] == gid_).to_numpy()
            stealth[m] = rng.random() < sf
    acc["stealth"] = stealth
    sb = stealth & (kinds_ == "farm_bot")                      # 사람 같은 리듬, 짧은 가동
    acc.loc[sb, "int_cv"] = rng.uniform(0.5, 0.9, sb.sum())
    acc.loc[sb, "dur_med_m"] = rng.uniform(600, 900, sb.sum())
    acc.loc[sb, "dur_sd"] = 0.15
    acc.loc[sb, "n_skills"] = 5
    acc["macro_cv"] = 0.02
    sm = stealth & (kinds_ == "macro")                         # 지터를 넣은 매크로
    acc.loc[sm, "macro_cv"] = rng.uniform(0.25, 0.45, sm.sum())

    # 비정상 유저 일부도 길드에 넣어 길드 정보로 바로 구분되지 않게
    nguild = int(acc["guild_id"].max())
    no_g = acc["guild_id"] == 0
    acc.loc[no_g, "guild_id"] = np.where(rng.random(no_g.sum()) < 0.6,
                                         rng.integers(1, nguild + 1, no_g.sum()), 0)

    # ── 레벨·주 사냥터 ──
    lvl = np.clip(rng.gamma(2.2, 12, N), 1, 60).astype(int).copy()
    lvl[acc["kind"].eq("farm_bot").to_numpy()] = rng.integers(36, 50, acc["kind"].eq("farm_bot").sum())
    alt = acc["role"].eq("alt").to_numpy()
    lvl[alt] = rng.integers(8, 20, alt.sum())
    acc["level"] = lvl
    acc["home_map"] = np.clip(lvl // 6 + 1, 1, cfg["world"]["n_maps"])

    # ── 생성일 ──
    created = start - pd.to_timedelta(rng.integers(1, 900, N), unit="D")
    new_mask = acc["role"].isin(["alt", "payer", "receiver"]).to_numpy()
    new_mask = new_mask | acc["role"].eq("buyer").to_numpy() & (rng.random(N) < 0.3)
    # 기간 중 새로 들어온 정상 유저 (신규 계정 = 어뷰저라는 지름길을 막는다)
    new_mask = new_mask | acc["kind"].eq("normal").to_numpy() & (rng.random(N) < cfg["normal"].get("new_player_frac", 0.0))
    offs = rng.uniform(0, days * 0.8, N)
    created = np.where(new_mask, start + pd.to_timedelta(offs, unit="D"), created)
    acc["created_at"] = pd.to_datetime(created).floor("s")

    # ── 기기·IP ──
    dev = rng.integers(10**8, 10**9, N)
    ip = rng.integers(10**8, 10**9, N)
    kinds = acc["kind"].to_numpy()
    roles = acc["role"].to_numpy()
    groups = acc["ring_id"].to_numpy()
    # 정상 가족 기기 공유 (2~3명)
    fam = np.flatnonzero((kinds == "normal") & (rng.random(N) < nc["family_device_frac"]))
    for i in range(0, len(fam) - 2, 3):
        k = int(rng.integers(2, 4))
        dev[fam[i:i + k]] = dev[fam[i]]
        ip[fam[i:i + k]] = ip[fam[i]]
    # 링 봇: 기기 3~5대, IP 2~3개 공유
    for rid in np.unique(groups[kinds == "farm_bot"]):
        idx = np.flatnonzero((groups == rid) & (kinds == "farm_bot"))
        dpool = rng.integers(10**8, 10**9, int(rng.integers(3, 6)))
        ipool = rng.integers(10**8, 10**9, int(rng.integers(2, 4)))
        dev[idx] = rng.choice(dpool, len(idx))
        ip[idx] = rng.choice(ipool, len(idx))
    # 다계정: 한 기기·IP. 은닉형은 기기를 계정마다 바꾸고(기기 정보 변조) IP만 공유
    st = acc["stealth"].to_numpy()
    for gid_ in np.unique(groups[kinds == "multi_account"]):
        idx = np.flatnonzero(groups == gid_)
        if not st[idx[0]]:
            dev[idx] = dev[idx[0]]
        ip[idx] = ip[idx[0]]
    acc["device_id"] = [f"D{d:09d}" for d in dev]
    acc["ip_hash"] = [f"{(i * 2654435761) % 16**10:010x}" for i in ip]
    acc["country"] = rng.choice(["KR", "KR", "KR", "KR", "KR", "JP", "TW", "US"], N)
    acc.loc[acc["kind"].eq("farm_bot"), "country"] = rng.choice(["KR", "CN", "VN"], acc["kind"].eq("farm_bot").sum())

    # ── ID 무작위 부여 ──
    acc["account_id"] = rng.permutation(np.arange(100_001, 100_001 + N))
    return acc


def nnc(nc):
    return nc["action_interval_s"]
