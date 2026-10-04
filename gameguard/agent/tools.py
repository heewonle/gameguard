"""조사 에이전트 도구. 전부 읽기 전용이고, 결과는 LLM 컨텍스트에 맞게 작게 요약한다.

안전장치
  - DuckDB 를 read_only 로 열고 외부 파일 접근(enable_external_access)을 끈다 → 정답 Parquet 을 읽을 수 없다
  - run_readonly_sql: SELECT/WITH 만, 정답 스키마 이름이 들어간 쿼리 거부, 결과 최대 50행
  - 모든 호출은 감사 로그(calls)에 남는다
"""
import json
import re
import time

import duckdb
import pandas as pd

FORBIDDEN = re.compile(r"\b(labels|attach|copy|install|load|pragma|set|export|import|create|insert|update|delete|"
                       r"drop|alter|read_parquet|read_csv|read_json|glob)\b", re.I)


def _r(x, nd=2):
    if isinstance(x, float):
        return round(x, nd)
    return x


def _records(df: pd.DataFrame, n=20):
    df = df.head(n).copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(3)
    return json.loads(df.to_json(orient="records", force_ascii=False))


class Toolbox:
    def __init__(self, db: str, features: pd.DataFrame, alerts: pd.DataFrame):
        self.con = duckdb.connect(db, read_only=True, config={"enable_external_access": False})
        self.feat = features.set_index("account_id")
        self.alerted = set(alerts["account_id"])
        self.calls = []

    def q(self, sql, params=None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    # ── 도구 ─────────────────────────────────────────────
    def get_account_profile(self, account_id: int) -> dict:
        a = self.q("SELECT account_id, created_at, level, country, guild_id FROM accounts WHERE account_id = ?",
                   [account_id])
        if a.empty:
            return {"error": "no such account"}
        f = self.feat.loc[account_id]
        p0 = self.q("SELECT min(login_at) FROM sessions").iloc[0, 0]
        pct = lambda c: round(float((self.feat[c] <= f[c]).mean() * 100), 1) if pd.notna(f[c]) else None
        return {
            **_records(a)[0],
            "created_during_period": bool(a["created_at"].iloc[0] >= p0),
            "active_days": int(f["active_days"]),
            "hours_per_active_day": _r(float(f["hours_per_day"] or 0)),
            "hours_per_day_percentile": pct("hours_per_day"),
            "max_day_hours": _r(float(f["max_day_hours"] or 0)),
            "n_sessions": int(f["n_sessions"] or 0),
            "income_gold": int(f["income"]),
            "income_percentile": pct("income"),
            "payments": int(f["n_paid"]), "refunds": int(f["n_refund"]), "paid_krw": int(f["paid_krw"]),
            "min_balance": int(f["min_balance"]) if pd.notna(f["min_balance"]) else None,
        }

    def get_session_pattern(self, account_id: int) -> dict:
        f = self.feat.loc[account_id]
        hours = self.q("""SELECT hour(login_at) AS h, count(*) AS n FROM sessions WHERE account_id = ?
                          GROUP BY 1 ORDER BY n DESC LIMIT 3""", [account_id])
        return {
            "action_interval_cv_median": _r(f["median_cv"], 3), "action_interval_cv_min": _r(f["min_cv"], 3),
            "share_sessions_cv_below_0.5": _r(f["frac_sess_cv_lt_05"], 3),
            "note_cv": "사람은 보통 CV 0.9~1.6, 기계적 반복일수록 0에 가깝다",
            "login_hour_top3": _records(hours),
            "share_login_0_to_6h": _r(f["frac_dawn_login"], 3), "top_login_hour_share": _r(f["top_hour_share"], 3),
            "action_share": {"chat": _r(f["share_chat"], 3), "move": _r(f["share_move"], 3),
                             "craft": _r(f["share_craft"], 3), "loot": _r(f["share_loot"], 3)},
            "distinct_skills": int(f["n_skills"] or 0), "distinct_maps": int(f["n_maps"] or 0),
            "position_spread": _r(f["mean_spread"], 1),
            "note_spread": "사냥 거점 주변 좌표 표준편차. 사람 60~200, 한자리 반복은 15 안팎",
        }

    def get_currency_flow(self, account_id: int) -> dict:
        by = self.q("""SELECT reason, count(*) AS n, sum(delta) AS gold FROM currency_log WHERE account_id = ?
                       GROUP BY 1 ORDER BY abs(sum(delta)) DESC""", [account_id])
        f = self.feat.loc[account_id]
        return {"by_reason": _records(by),
                "p2p_out_without_return_gold": int(f["p2p_out_unrecip"]),
                "p2p_out_ratio_of_income": _r(f["out_ratio"], 3),
                "p2p_in_without_return_gold": int(f["p2p_in_unrecip"]),
                "p2p_in_ratio_of_income": _r(f["in_ratio"], 3),
                "gift_like_out_count": int(f["p2p_out_gift_n"]), "gift_like_in_count": int(f["p2p_in_gift_n"]),
                "market_buys": int(f["mkt_buy_n"]), "market_sells": int(f["mkt_sell_n"]),
                "max_market_sell_price_vs_median": _r(f["mkt_sell_max_ratio"], 2),
                "max_market_buy_price_vs_median": _r(f["mkt_buy_max_ratio"], 2),
                "max_same_item_market_buys": int(f["max_same_item_buys"])}

    def get_trade_partners(self, account_id: int, top_n: int = 8) -> dict:
        df = self.q("""
            WITH t AS (
              SELECT seller AS partner, 'received_from' AS dir, gold, item_id, channel FROM trades WHERE buyer = ?
              UNION ALL
              SELECT buyer, 'paid_by', gold, item_id, channel FROM trades WHERE seller = ?)
            SELECT partner,
                   sum(gold) FILTER (WHERE dir = 'paid_by') AS gold_received,
                   sum(gold) FILTER (WHERE dir = 'received_from') AS gold_sent,
                   count(*) AS trades, count(*) FILTER (WHERE item_id = 0) AS gold_only_trades,
                   string_agg(DISTINCT channel, '/') AS channels
            FROM t GROUP BY 1 ORDER BY coalesce(gold_received, 0) + coalesce(gold_sent, 0) DESC
            LIMIT ?""", [account_id, account_id, int(top_n)])
        df["partner_alerted"] = df["partner"].isin(self.alerted)
        total = self.q("SELECT count(DISTINCT seller) + count(DISTINCT buyer) FROM trades WHERE buyer = ? OR seller = ?",
                       [account_id, account_id]).iloc[0, 0]
        return {"note": "gold_sent = 이 계정이 상대에게 낸 골드, gold_received = 상대에게서 받은 골드",
                "partners": _records(df.fillna(0)), "distinct_partners_approx": int(total)}

    def get_identity_links(self, account_id: int) -> dict:
        df = self.q("""
            WITH prim AS (
              SELECT account_id, arg_max(device_id, n_dev) AS dev, arg_max(ip_hash, n_ip) AS ip FROM (
                SELECT account_id, device_id, ip_hash,
                       count(*) OVER (PARTITION BY account_id, device_id) AS n_dev,
                       count(*) OVER (PARTITION BY account_id, ip_hash) AS n_ip FROM sessions) GROUP BY 1),
            me AS (SELECT * FROM prim WHERE account_id = ?)
            SELECT p.account_id AS other, p.ip = me.ip AS same_main_ip, p.dev = me.dev AS same_main_device
            FROM prim p, me WHERE p.account_id <> me.account_id AND (p.ip = me.ip OR p.dev = me.dev)
            LIMIT 40""", [account_id])
        df["other_alerted"] = df["other"].isin(self.alerted)
        n_ip = int(df["same_main_ip"].sum())
        return {"accounts_sharing_main_ip": n_ip, "accounts_sharing_main_device": int(df["same_main_device"].sum()),
                "note": "가족은 2~3명이 공유하기도 한다. 30명 넘게 공유하면 PC방 같은 공용 환경",
                "linked_accounts": _records(df, 15)}

    def run_readonly_sql(self, query: str) -> dict:
        q = query.strip().rstrip(";")
        if not re.match(r"^(select|with)\b", q, re.I) or FORBIDDEN.search(q) or ";" in q:
            return {"error": "SELECT/WITH 단일 조회만 허용. 공개 테이블: accounts, sessions, actions, currency_log, "
                             "trades, market_listings, payments, items, maps"}
        try:
            df = self.con.execute(f"SELECT * FROM ({q}) LIMIT 50").df()
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)[:300]}
        return {"rows": _records(df, 50)}

    # ── 실행기 ─────────────────────────────────────────────
    def call(self, name: str, args: dict) -> dict:
        t0 = time.time()
        fn = getattr(self, name, None)
        if name.startswith("_") or name not in TOOL_NAMES or fn is None:
            out = {"error": f"unknown tool {name}"}
        else:
            try:
                out = fn(**args)
            except Exception as e:  # noqa: BLE001
                out = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        self.calls.append({"tool": name, "args": args, "ms": int((time.time() - t0) * 1000)})
        return out


TOOL_SPECS = [
    {"name": "get_account_profile", "description": "계정 기본 정보와 활동량·수입·결제 요약 (모집단 대비 백분위 포함)",
     "params": {"account_id": "integer"}},
    {"name": "get_session_pattern", "description": "접속 시각, 행동 간격 규칙성(CV), 행동 종류 비율, 좌표 퍼짐",
     "params": {"account_id": "integer"}},
    {"name": "get_currency_flow", "description": "골드 수입·지출 사유별 합계, 대가 없는 송금/수령, 거래소 고가 매매",
     "params": {"account_id": "integer"}},
    {"name": "get_trade_partners", "description": "골드를 주고받은 상위 거래 상대와 그 상대의 경보 여부",
     "params": {"account_id": "integer", "top_n": "integer"}},
    {"name": "get_identity_links", "description": "주 IP·주 기기를 공유하는 계정과 그 계정들의 경보 여부",
     "params": {"account_id": "integer"}},
    {"name": "run_readonly_sql", "description": "공개 테이블에 대한 SELECT 조회 (최대 50행). 위 도구로 부족할 때만",
     "params": {"query": "string"}},
]
TOOL_NAMES = {t["name"] for t in TOOL_SPECS}


def ollama_tools():
    out = []
    for t in TOOL_SPECS:
        req = [k for k in t["params"] if k != "top_n"]
        out.append({"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": {"type": "object", "properties": {k: {"type": v} for k, v in t["params"].items()},
                           "required": req}}})
    return out
