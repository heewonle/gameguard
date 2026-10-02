"""세션과 행동 로그 생성 (벡터화)."""
import numpy as np
import pandas as pd

from .world import HOUR_P

ACTIONS = np.array(["kill", "loot", "skill", "move", "chat", "craft"])
P_ACT = {
    "human": [0.45, 0.20, 0.15, 0.10, 0.05, 0.05],
    "bot":   [0.60, 0.35, 0.05, 0.00, 0.00, 0.00],
    "macro": [0.55, 0.40, 0.05, 0.00, 0.00, 0.00],
}
PCBANG_IPS = 200


def make_sessions(acc: pd.DataFrame, cfg: dict, rng, start: pd.Timestamp) -> pd.DataFrame:
    days = cfg["days"]
    N = len(acc)
    # 계정 × 날짜 격자
    a_idx = np.repeat(np.arange(N), days)
    d_idx = np.tile(np.arange(days), N)
    day_start = start + pd.to_timedelta(d_idx, unit="D")
    alive = day_start + pd.Timedelta(days=1) > acc["created_at"].to_numpy()[a_idx]
    active = alive & (rng.random(len(a_idx)) < acc["p_active"].to_numpy()[a_idx])
    a_idx, d_idx = a_idx[active], d_idx[active]

    kind = acc["kind"].to_numpy()
    is_bot = kind[a_idx] == "farm_bot"
    n_s = np.where(is_bot, 1, 1 + rng.poisson(acc["n_sess"].to_numpy()[a_idx]))
    s_acc = np.repeat(a_idx, n_s)
    s_day = np.repeat(d_idx, n_s)
    S = len(s_acc)
    bot = kind[s_acc] == "farm_bot"

    hour = rng.choice(24, S, p=HOUR_P) + rng.random(S)
    hour[bot] = rng.uniform(0, 1, bot.sum())                      # 봇은 자정 직후 시작해 하루 종일
    dur_m = np.exp(np.log(acc["dur_med_m"].to_numpy()[s_acc]) + acc["dur_sd"].to_numpy()[s_acc] * rng.standard_normal(S))
    dur_m[bot] = rng.uniform(18 * 60, 22 * 60, bot.sum())
    s = pd.DataFrame({
        "acc_i": s_acc,
        "login_at": start + pd.to_timedelta(s_day, unit="D") + pd.to_timedelta(hour * 3600, unit="s"),
        "dur_s": np.clip(dur_m * 60, 120, 23 * 3600),
        "is_macro": False,
    })

    # 매크로 세션: 지정 시각(새벽)에 2~4시간
    mac = np.flatnonzero(acc["macro_hour"].to_numpy() >= 0)
    if len(mac):
        ma = np.repeat(mac, days)
        md = np.tile(np.arange(days), len(mac))
        keep = rng.random(len(ma)) < 0.7
        ma, md = ma[keep], md[keep]
        hr = acc["macro_hour"].to_numpy()[ma] + rng.uniform(0, 0.25, len(ma))
        s = pd.concat([s, pd.DataFrame({
            "acc_i": ma,
            "login_at": start + pd.to_timedelta(md, unit="D") + pd.to_timedelta(hr * 3600, unit="s"),
            "dur_s": rng.uniform(2 * 3600, 4 * 3600, len(ma)),
            "is_macro": True,
        })], ignore_index=True)

    # 계정 생성 전 세션 제거
    s = s[s["login_at"].to_numpy() > acc["created_at"].to_numpy()[s["acc_i"].to_numpy()]]
    s = s.sort_values(["acc_i", "login_at"], kind="stable").reset_index(drop=True)
    s["login_at"] = s["login_at"].dt.floor("s")
    s = _remove_overlap(s)
    s["logout_at"] = s["login_at"] + pd.to_timedelta(s["dur_s"].round(), unit="s")
    end = start + pd.Timedelta(days=days)
    s = s[s["login_at"] < end].reset_index(drop=True)
    s["logout_at"] = s["logout_at"].clip(upper=end)

    # 접속 기기·IP: 기본은 계정 고유, 정상 계열 세션 8%는 PC방
    acc_dev = acc["device_id"].to_numpy()[s["acc_i"]]
    acc_ip = acc["ip_hash"].to_numpy()[s["acc_i"]]
    human = ~np.isin(kind[s["acc_i"]], ["farm_bot", "multi_account"])
    pc = human & (rng.random(len(s)) < 0.08)
    pcn = rng.integers(0, PCBANG_IPS, len(s))
    s["device_id"] = np.where(pc, [f"PC{p:04d}{q:03d}" for p, q in zip(pcn, rng.integers(0, 60, len(s)))], acc_dev)
    s["ip_hash"] = np.where(pc, [f"pcb{p:07x}" for p in pcn], acc_ip)
    s["session_id"] = np.arange(1, len(s) + 1, dtype="int64") + 5_000_000
    return s


def _remove_overlap(s: pd.DataFrame) -> pd.DataFrame:
    """같은 계정의 세션이 겹치지 않게 뒤 세션을 민다 (최소 1분 간격)."""
    acc_i = s["acc_i"].to_numpy()
    t = s["login_at"].to_numpy().astype("datetime64[s]").astype("int64")
    d = s["dur_s"].to_numpy()
    prev_end = -1
    prev_acc = -1
    for i in range(len(t)):
        if acc_i[i] == prev_acc and t[i] < prev_end + 60:
            t[i] = prev_end + 60
        prev_acc = acc_i[i]
        prev_end = t[i] + int(d[i])
    s["login_at"] = pd.to_datetime(t, unit="s")
    return s


def make_actions(s: pd.DataFrame, acc: pd.DataFrame, cfg: dict, rng) -> pd.DataFrame:
    a = s["acc_i"].to_numpy()
    kind = acc["kind"].to_numpy()[a]
    macro = s["is_macro"].to_numpy()
    imean = np.where(macro, acc["macro_period"].to_numpy()[a], acc["int_mean"].to_numpy()[a])
    icv = np.where(macro, acc["macro_cv"].to_numpy()[a], acc["int_cv"].to_numpy()[a])
    dur = (s["logout_at"] - s["login_at"]).dt.total_seconds().to_numpy()
    n = np.maximum(1, np.floor(dur / imean)).astype(np.int64)

    sess_i = np.repeat(np.arange(len(s)), n)
    shape = 1.0 / icv[sess_i] ** 2
    gap = rng.gamma(shape, imean[sess_i] / shape)
    # 세션 안 누적합
    cs = np.cumsum(gap)
    first = np.concatenate([[0], np.cumsum(n)[:-1]])
    base = np.repeat(cs[first] - gap[first], n)
    offset = cs - base
    keep = offset <= dur[sess_i]
    sess_i, offset = sess_i[keep], offset[keep]
    M = len(sess_i)

    # 행동 유형
    style = np.where(kind[sess_i] == "farm_bot", 1, 0)
    style = np.where(macro[sess_i], 2, style)
    single = acc["single_map"].to_numpy()[a][sess_i] | macro[sess_i]
    style = np.where((style == 0) & single, 1, style)   # 부계정 자동사냥도 봇형 분포
    act = np.empty(M, dtype=object)
    for k, name in enumerate(["human", "bot", "macro"]):
        m = style == k
        act[m] = rng.choice(ACTIONS, m.sum(), p=P_ACT[name])

    # 사냥터: 세션 주 사냥터 ± 1, 사람은 행동의 25%가 인접 맵
    n_maps = cfg["world"]["n_maps"]
    home = acc["home_map"].to_numpy()[a]
    sess_map = np.where(acc["single_map"].to_numpy()[a] | macro, home,
                        np.clip(home + rng.integers(-1, 2, len(s)), 1, n_maps))
    amap = sess_map[sess_i].copy()
    wander = (~single) & (rng.random(M) < 0.25)
    amap[wander] = np.clip(amap[wander] + rng.choice([-1, 1], wander.sum()), 1, n_maps)

    # 스킬: 계정마다 쓰는 스킬 묶음
    nsk = acc["n_skills"].to_numpy()[a][sess_i]
    skill_base = (acc["account_id"].to_numpy()[a][sess_i] * 7) % 30
    skill = ((skill_base + rng.integers(0, 1 << 30, M) % nsk) % 30 + 1).astype("int16")
    skill[~np.isin(act, ["kill", "skill"])] = 0

    # 좌표: 사람은 맵 전체, 봇/매크로는 반경 30 안을 맴돔
    cx = (acc["account_id"].to_numpy()[a] * 37 % 900 + 50)[sess_i]
    cy = (acc["account_id"].to_numpy()[a] * 91 % 900 + 50)[sess_i]
    x = np.where(single, cx + rng.normal(0, 15, M), rng.uniform(0, 1000, M)).astype("float32")
    y = np.where(single, cy + rng.normal(0, 15, M), rng.uniform(0, 1000, M)).astype("float32")

    ts = s["login_at"].to_numpy()[sess_i] + (offset * 1000).astype("timedelta64[ms]")
    return pd.DataFrame({
        "ts": ts,
        "account_id": acc["account_id"].to_numpy()[a][sess_i].astype("int64"),
        "session_id": s["session_id"].to_numpy()[sess_i],
        "action_type": pd.Categorical(act, categories=ACTIONS),
        "map_id": amap.astype("int16"),
        "skill_id": skill,
        "x": x, "y": y,
        "_sess_i": sess_i,
    })
