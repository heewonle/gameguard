"""GameGuard 검토 대시보드 — 경보 큐, 계정 조사, 사람 검토 기록, 배치 실행 기록.

  streamlit run app/streamlit_app.py -- --db data/test.duckdb --out data/out_test

정답 라벨을 읽지 않는다 (운영 화면). 성능 수치는 '평가 보고서' 탭에서 reports/*.md 로만 본다.
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gameguard.agent.tools import Toolbox  # noqa: E402

# 색: 크기는 단일 파랑, 판정은 예약된 상태색 (항상 글자와 함께 표시)
BLUE = "#2a78d6"
STATUS = {"abuse": ("#d03b3b", "🔴 어뷰징"), "uncertain": ("#fab219", "🟡 불확실"),
          "normal": ("#0ca30c", "🟢 정상"), None: ("#9a9893", "⚪ 미조사")}
TYPE_KO = {"farm_bot": "작업장 봇", "rmt_ring": "RMT 링", "macro": "매크로", "multi_account": "다계정",
           "market_manip": "시세조작", "chargeback": "차지백", "none": "-"}
SRC_KO = {"rule": "룰", "ml": "ML", "graph_fanin": "그래프·송금집중", "graph_propagation": "그래프·전파"}


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "test.duckdb"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "out_test"))
    a, _ = ap.parse_known_args(sys.argv[1:])
    return a


A = args()
st.set_page_config(page_title="GameGuard 검토", page_icon="🛡️", layout="wide")


@st.cache_data
def load_tables(out: str, _mtime: float):
    al = pd.read_parquet(os.path.join(out, "alerts.parquet"))
    ml_all = pd.read_parquet(os.path.join(out, "ml_scores_sanctioned.parquet"))["ml_score"]
    al["ml_percentile"] = al["ml_score"].map(lambda v: round(float((ml_all <= v).mean() * 100), 2))
    edges = pd.read_parquet(os.path.join(out, "graph_edges.parquet"))
    feats = pd.read_parquet(os.path.join(out, "features.parquet"))
    return al, edges, feats


def load_agent(out: str) -> pd.DataFrame:
    p = os.path.join(out, "agent_reports.jsonl")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["account_id"])
    rs = [json.loads(l) for l in open(p, encoding="utf-8")]
    return pd.DataFrame(rs).drop_duplicates("account_id", keep="last").set_index("account_id")


def load_reviews(out: str) -> pd.DataFrame:
    p = os.path.join(out, "reviews.jsonl")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["account_id", "decision", "abuse_type", "note", "reviewer", "at"]).set_index("account_id")
    rs = [json.loads(l) for l in open(p, encoding="utf-8")]
    return pd.DataFrame(rs).drop_duplicates("account_id", keep="last").set_index("account_id")


def toolbox() -> Toolbox:
    if "tb" not in st.session_state:
        al, _, feats = load_tables(A.out, os.path.getmtime(os.path.join(A.out, "alerts.parquet")))
        st.session_state.tb = Toolbox(A.db, feats, al)
    return st.session_state.tb


alerts, edges, feats = load_tables(A.out, os.path.getmtime(os.path.join(A.out, "alerts.parquet")))
agent = load_agent(A.out)
reviews = load_reviews(A.out)

q = alerts.copy()
q["단계"] = q["sources"].map(lambda s: " + ".join(SRC_KO.get(x, x) for x in s.split(",")))
q["n_sources"] = q["sources"].str.count(",") + 1
q["verdict"] = q["account_id"].map(agent["verdict"]) if len(agent) else None
q["에이전트 판정"] = q["verdict"].map(lambda v: STATUS.get(v if isinstance(v, str) else None)[1])
q["유형"] = q["account_id"].map(agent["abuse_type"]).map(TYPE_KO) if len(agent) else "-"
q["확신도"] = q["account_id"].map(agent["confidence"]) if len(agent) else None
q["검토"] = q["account_id"].map(reviews["decision"]).fillna("대기") if len(reviews) else "대기"

st.title("🛡️ GameGuard 검토 대시보드")
st.caption(f"DB `{os.path.relpath(A.db, ROOT)}` · 결과 `{os.path.relpath(A.out, ROOT)}` · "
           "제재는 사람이 결정한다. 에이전트 판정은 검토 순서를 정하는 참고 정보다.")

tab_over, tab_queue, tab_acc, tab_runs, tab_eval = st.tabs(["개요", "경보 큐", "계정 조사", "배치 기록", "평가 보고서"])

# ── 개요 ───────────────────────────────────────────────
with tab_over:
    c = st.columns(5)
    c[0].metric("경보", f"{len(q):,}")
    c[1].metric("에이전트 조사", f"{q['verdict'].notna().sum():,}")
    c[1].caption(f"경보의 {q['verdict'].notna().mean():.0%}")
    c[2].metric("🔴 어뷰징 판정", f"{(q['verdict'] == 'abuse').sum():,}")
    c[3].metric("🟢 정상 판정 (우선순위 낮춤)", f"{(q['verdict'] == 'normal').sum():,}")
    c[4].metric("사람 검토 완료", f"{(q['검토'] != '대기').sum():,}")

    left, right = st.columns(2)
    by_src = q["단계"].value_counts().rename_axis("단계").reset_index(name="경보 수")
    left.subheader("탐지 단계 조합별 경보")
    left.altair_chart(
        alt.Chart(by_src).mark_bar(color=BLUE, cornerRadiusEnd=4, height={"band": 0.7}).encode(
            x=alt.X("경보 수:Q", title="경보 수"), y=alt.Y("단계:N", sort="-x", title=None),
            tooltip=["단계", "경보 수"]).properties(height=260), use_container_width=True)
    right.subheader("에이전트 판정")
    vd = q["에이전트 판정"].value_counts().rename_axis("판정").reset_index(name="경보 수")
    order = [v[1] for v in STATUS.values()]
    right.altair_chart(
        alt.Chart(vd).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
            x=alt.X("경보 수:Q"), y=alt.Y("판정:N", sort=order, title=None),
            color=alt.Color("판정:N", scale=alt.Scale(domain=order, range=[v[0] for v in STATUS.values()]),
                            legend=None),
            tooltip=["판정", "경보 수"]).properties(height=260), use_container_width=True)
    right.caption("상태색은 글자 표시와 함께 쓴다. 미조사 경보는 배치가 매일 우선순위 순으로 예산만큼 조사한다.")

# ── 경보 큐 ─────────────────────────────────────────────
with tab_queue:
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    v_sel = f1.multiselect("에이전트 판정", order, default=order)
    s_sel = f2.multiselect("탐지 단계", list(SRC_KO.values()), default=list(SRC_KO.values()))
    r_sel = f3.selectbox("검토 상태", ["전체", "대기만", "완료만"])
    only_no_rule = f4.toggle("룰 밖 경보만")
    view = q[q["에이전트 판정"].isin(v_sel) & q["단계"].map(lambda t: any(s in t for s in s_sel))]
    if r_sel == "대기만":
        view = view[view["검토"] == "대기"]
    elif r_sel == "완료만":
        view = view[view["검토"] != "대기"]
    if only_no_rule:
        view = view[~view["sources"].str.contains("rule")]
    rank = {"abuse": 0, "uncertain": 1, None: 2, "normal": 3}
    view = view.assign(_r=view["verdict"].map(lambda v: rank.get(v if isinstance(v, str) else None)))
    view = view.sort_values(["_r", "n_sources", "ml_percentile"], ascending=[True, False, False])
    st.caption(f"{len(view):,}건 · 정렬: 어뷰징 → 불확실 → 미조사 → 정상, 그 안에서 걸린 단계 수·ML 백분위 순")
    ev = st.dataframe(
        view[["account_id", "에이전트 판정", "유형", "확신도", "단계", "ml_percentile", "검토"]].rename(
            columns={"account_id": "계정", "ml_percentile": "ML 백분위"}),
        hide_index=True, use_container_width=True, height=520, on_select="rerun", selection_mode="single-row")
    if ev.selection.rows:
        st.session_state.acc = int(view.iloc[ev.selection.rows[0]]["account_id"])
        st.info(f"계정 {st.session_state.acc} 선택됨 → '계정 조사' 탭에서 확인")

# ── 계정 조사 ───────────────────────────────────────────
with tab_acc:
    investigated = q[q["verdict"] == "abuse"].sort_values("n_sources", ascending=False)
    first = investigated if len(investigated) else q.sort_values("n_sources", ascending=False)
    default = st.session_state.get("acc", int(first["account_id"].iloc[0]))
    acc = st.number_input("계정 ID", value=default, step=1, format="%d")
    tb = toolbox()
    row = q[q["account_id"] == acc]
    if row.empty:
        st.warning("경보가 없는 계정이다. 도구로 조회한 정보만 표시한다.")
    else:
        r = row.iloc[0]
        st.markdown(f"**경보 단계:** {r['단계']} · **ML 백분위:** {r['ml_percentile']} · **에이전트:** {r['에이전트 판정']} "
                    f"{'· ' + r['유형'] if isinstance(r['유형'], str) else ''}")

    a_col, b_col = st.columns([3, 2])
    with a_col:
        if acc in agent.index:
            rep = agent.loc[acc]
            st.subheader("에이전트 조사 보고서")
            st.write(rep["summary"])
            st.dataframe(pd.DataFrame(rep["evidence"]).rename(columns={"claim": "주장", "value": "근거 값", "tool": "도구"}),
                         hide_index=True, use_container_width=True)
            st.caption(f"권고 {rep['recommended_action']} · 확신도 {rep['confidence']} · 도구 {rep['tool_calls']}회 · "
                       f"{rep['seconds']}초 · 모델 {rep['model']} / 프롬프트 {rep['prompt']}")
            with st.expander("조사 과정 (도구 호출 기록)"):
                for t in rep["transcript"]:
                    st.markdown(f"**{t['tool']}** `{json.dumps(t['args'], ensure_ascii=False)}`")
                    st.code(t["result"][:1500], language="json")
        else:
            st.info("아직 에이전트가 조사하지 않은 경보다.")
        if not row.empty:
            with st.expander("탐지 단계별 근거 (원문)"):
                st.json(json.loads(row.iloc[0]["details"]))

    with b_col:
        st.subheader("사람 검토")
        prev = reviews.loc[acc] if acc in reviews.index else None
        if prev is not None:
            st.success(f"기록됨: {prev['decision']} ({TYPE_KO.get(prev['abuse_type'], prev['abuse_type'])}) — {prev['at']}")
        with st.form("review"):
            d = st.radio("결정", ["제재 확정", "모니터링", "오탐(정상)"], horizontal=True)
            t = st.selectbox("유형", list(TYPE_KO), format_func=lambda k: TYPE_KO[k],
                             index=list(TYPE_KO).index(agent.loc[acc, "abuse_type"]) if acc in agent.index
                             and agent.loc[acc, "abuse_type"] in TYPE_KO else len(TYPE_KO) - 1)
            note = st.text_area("메모")
            who = st.text_input("검토자", value="reviewer")
            if st.form_submit_button("검토 결과 저장"):
                rec = {"account_id": int(acc), "decision": d, "abuse_type": t, "note": note, "reviewer": who,
                       "agent_verdict": agent.loc[acc, "verdict"] if acc in agent.index else None,
                       "at": dt.datetime.now().isoformat(timespec="seconds")}
                with open(os.path.join(A.out, "reviews.jsonl"), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                st.success("저장했다. 확정된 제재는 다음 ML 재학습의 양성 라벨이 된다.")

    st.divider()
    p1, p2, p3 = st.columns(3)
    prof = tb.get_account_profile(int(acc))
    if "error" not in prof:
        p1.metric("하루 접속 (활동일 평균)", f"{prof['hours_per_active_day']}시간")
        p1.caption(f"전체 계정 중 상위 {100 - prof['hours_per_day_percentile']:.1f}%")
        pat = tb.get_session_pattern(int(acc))
        p2.metric("행동 간격 CV (중앙값)", pat["action_interval_cv_median"])
        p2.caption("사람은 보통 0.9~1.6, 기계적 반복일수록 0에 가깝다")
        flow = tb.get_currency_flow(int(acc))
        p3.metric("대가 없이 받은 골드 / 보낸 골드", f"{flow['p2p_in_without_return_gold']:,} / {flow['p2p_out_without_return_gold']:,}")

        daily = tb.q("""SELECT login_at::DATE AS 날짜, sum(epoch(logout_at) - epoch(login_at)) / 3600.0 AS 시간
                        FROM sessions WHERE account_id = ? GROUP BY 1 ORDER BY 1""", [int(acc)])
        g1, g2 = st.columns(2)
        g1.subheader("일별 접속 시간")
        g1.altair_chart(alt.Chart(daily).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
            x=alt.X("날짜:T", title=None, axis=alt.Axis(format="%m-%d")), y=alt.Y("시간:Q", title="시간", scale=alt.Scale(domain=[0, 24])),
            tooltip=[alt.Tooltip("날짜:T"), alt.Tooltip("시간:Q", format=".1f")]).properties(height=220),
            use_container_width=True)

        # 1단계 이웃 그래프 (골드 이동 = 실선 화살표, IP·기기 공유 = 점선)
        g2.subheader("거래·신원 연결")
        e = edges[(edges["src"] == acc) | (edges["dst"] == acc)].copy()
        if e.empty:
            g2.caption("연결된 계정 없음")
        else:
            # 같은 쌍의 IP·기기 공유는 한 선으로, 골드 이동은 방향 있는 선으로. 이웃은 연결 강도 상위 12개만
            ident = (e[e["kind"] != "transfer"].assign(a=lambda d: d[["src", "dst"]].min(axis=1), b=lambda d: d[["src", "dst"]].max(axis=1))
                     .groupby(["a", "b"])["kind"].apply(lambda k: "·".join(sorted({"IP" if x == "ip" else "기기" for x in k}))).reset_index())
            tr = e[e["kind"] == "transfer"]
            other = pd.concat([ident.assign(n=lambda d: d["a"].where(d["a"] != acc, d["b"]), w=1.0),
                               tr.assign(n=lambda d: d["src"].where(d["src"] != acc, d["dst"]), w=lambda d: d["gold"] / 1e5)])
            keep = set(other.groupby("n")["w"].sum().sort_values(ascending=False).head(12).index) | {acc}
            nodes = keep
            col = lambda n: STATUS.get(agent.loc[n, "verdict"] if n in agent.index else None)[0]
            lines = ['digraph G { rankdir=LR; size="7,5"; bgcolor="transparent"; nodesep=0.25; ranksep=0.6;',
                     'node [shape=box style="rounded,filled" fillcolor="#f4f4f2" fontname="Malgun Gothic" fontsize=10 fontcolor="#0b0b0b"];',
                     'edge [fontname="Malgun Gothic" fontsize=9 color="#8a8984" fontcolor="#8a8984"];']
            for n in nodes:
                tag = "경보" if n in set(q["account_id"]) else "경보 없음"
                lines.append(f'"{n}" [label="{n}\\n{tag}" color="{col(n)}" penwidth={3 if n == acc else 2}];')
            for t in tr.itertuples():
                if t.src in keep and t.dst in keep:
                    lines.append(f'"{t.src}" -> "{t.dst}" [label="{int(t.gold):,}G" penwidth=1.5];')
            for t in ident.itertuples():
                if t.a in keep and t.b in keep:
                    lines.append(f'"{t.a}" -> "{t.b}" [dir=none style=dashed label="{t.kind}"];')
            lines.append("}")
            g2.graphviz_chart("\n".join(lines), use_container_width=True)
            g2.caption("테두리 색 = 에이전트 판정 (🔴 어뷰징 🟡 불확실 🟢 정상 ⚪ 미조사). 실선 화살표 = 대가 없는 골드 이동, "
                       "점선 = 주 IP·기기 공유. 연결 강도 상위 12개 계정만 표시")

        partners = tb.get_trade_partners(int(acc))["partners"]
        if partners:
            st.subheader("거래 상대")
            st.dataframe(pd.DataFrame(partners), hide_index=True, use_container_width=True)

# ── 배치 기록 ───────────────────────────────────────────
with tab_runs:
    runs = sorted(glob.glob(os.path.join(A.out, "runs", "*.json")), reverse=True)
    if not runs:
        st.info("배치 실행 기록이 없다. `python -m gameguard.run` 으로 실행한다.")
    for p in runs[:10]:
        r = json.load(open(p, encoding="utf-8"))
        bad = [s for s in r["steps"] if s["status"] != "ok"]
        icon = "✅" if not bad else "⚠️"
        with st.expander(f"{icon} {r['stamp']} — {len(r['steps']) - len(bad)}/{len(r['steps'])} 단계 성공", expanded=p == runs[0]):
            st.dataframe(pd.DataFrame(r["steps"]), hide_index=True, use_container_width=True)

# ── 평가 보고서 ─────────────────────────────────────────
with tab_eval:
    st.caption("정답을 아는 합성 월드에서만 계산한 성능. 운영 화면은 정답을 읽지 않는다.")
    files = sorted(glob.glob(os.path.join(ROOT, "reports", "*.md")))
    pick = st.selectbox("보고서", files, format_func=os.path.basename,
                        index=next((i for i, f in enumerate(files) if "stage3" in f), 0))
    st.markdown(open(pick, encoding="utf-8").read())
