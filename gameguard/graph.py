"""3단: 골드 이동 + 신원 공유 그래프.

두 가지 신호
  fanin       신원 공유 그룹(주 IP·기기가 같은 4~30명) 안에서 여러 계정이 한 계정에게 골드를 몰아줌
              — 라벨·씨앗 없이 구조만으로 판단 (은닉형 다계정: 기기는 바꿔도 IP와 송금 구조는 남는다)
  propagation 1·2단 경보(씨앗)와 강하게 연결된 계정. 거래·신원 연결 가중치 중 씨앗 쪽 비율로 판단하고,
              순위용으로 개인화 PageRank 점수도 함께 남긴다.

  python -m gameguard.graph --db data/test.duckdb --out data/out_test          # 씨앗 = 룰 히트
"""
import argparse
import os

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from .db import connect, run_sql_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_edges(con, cfg) -> pd.DataFrame:
    return run_sql_file(con, os.path.join(ROOT, "sql", "graph", "edges.sql"), **cfg["edges"]).df()


def identity_groups(edges: pd.DataFrame) -> dict:
    """주 IP·기기 공유 간선의 연결 요소 → {account_id: group_id}."""
    g = nx.Graph()
    ident = edges[edges["kind"].isin(["ip", "device"])]
    g.add_edges_from(zip(ident["src"], ident["dst"]))
    out = {}
    for i, comp in enumerate(nx.connected_components(g)):
        for a in comp:
            out[a] = (i, len(comp))
    return out


def fanin(edges: pd.DataFrame, cfg) -> pd.DataFrame:
    c = cfg["fanin"]
    grp = identity_groups(edges)
    tr = edges[edges["kind"] == "transfer"].copy()
    tr["gs"] = tr["src"].map(lambda a: grp.get(a, (None, 0))[0])
    tr["gd"] = tr["dst"].map(lambda a: grp.get(a, (None, 0))[0])
    tr["size"] = tr["src"].map(lambda a: grp.get(a, (None, 0))[1])
    tot = tr.groupby("src")["gold"].sum()
    inside = tr[(tr["gs"].notna()) & (tr["gs"] == tr["gd"]) & (tr["size"] >= c["min_group"])].copy()
    inside["share"] = inside["gold"] / inside["src"].map(tot)
    funnel = inside[inside["share"] >= c["min_share"]]
    hubs = funnel.groupby("dst").agg(senders=("src", "nunique"), gold=("gold", "sum"), group=("gs", "first"))
    hubs = hubs[hubs["senders"] >= c["min_senders"]]
    rows = []
    for hub, h in hubs.iterrows():
        members = funnel.loc[funnel["dst"] == hub, "src"].tolist()
        ev = f'{{"hub": {hub}, "senders": {int(h["senders"])}, "gold_to_hub": {int(h["gold"])}}}'
        rows += [{"account_id": hub, "signal": "fanin_hub", "evidence": ev}]
        rows += [{"account_id": m, "signal": "fanin_sender", "evidence": ev} for m in members]
    return pd.DataFrame(rows, columns=["account_id", "signal", "evidence"])


def propagate(edges: pd.DataFrame, seeds: set, cfg) -> pd.DataFrame:
    p = cfg["propagation"]
    w = np.where(edges["kind"] == "transfer", np.log10(edges["gold"].clip(lower=10)) * p["w_transfer_per_log10"],
                 np.where(edges["kind"] == "ip", p["w_ip"], p["w_device"]))
    e = edges.assign(w=w).groupby(["src", "dst"], as_index=False)["w"].sum()
    g = nx.Graph()
    for s, d, ww in e.itertuples(index=False):
        if g.has_edge(s, d):
            g[s][d]["weight"] += ww
        else:
            g.add_edge(s, d, weight=ww)
    present = [s for s in seeds if s in g]
    if not present:
        return pd.DataFrame(columns=["account_id", "ppr", "seed_share", "seed_neighbors", "flag"])
    ppr = nx.pagerank(g, alpha=p["alpha"], personalization={s: 1.0 for s in present}, weight="weight")
    rows = []
    seed_set = set(present)
    for n in g.nodes:
        if n in seed_set:
            continue
        nb = g[n]
        tot = sum(v["weight"] for v in nb.values())
        sw = sum(v["weight"] for k, v in nb.items() if k in seed_set)
        k = sum(1 for k in nb if k in seed_set)
        rows.append({"account_id": n, "ppr": ppr.get(n, 0.0), "seed_share": sw / tot if tot else 0.0,
                     "seed_neighbors": k})
    df = pd.DataFrame(rows)
    df["flag"] = (df["seed_neighbors"] >= p["min_seed_neighbors"]) & (df["seed_share"] >= p["min_score_ratio"])
    return df


def load_seeds(out_dir: str, which) -> set:
    seeds = set()
    if "rules" in which:
        seeds |= set(pd.read_parquet(os.path.join(out_dir, "rule_hits.parquet"))["account_id"])
    if "ml_sanctioned" in which:
        s = pd.read_parquet(os.path.join(out_dir, "ml_scores_sanctioned.parquet"))
        seeds |= set(s.loc[s["ml_flag"], "account_id"])
    return seeds


def run(db: str, out_dir: str, seeds_from, cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    edges = load_edges(connect(db, read_only=True), cfg)
    fi = fanin(edges, cfg)
    seeds = load_seeds(out_dir, seeds_from)
    pr = propagate(edges, seeds, cfg)
    edges.to_parquet(os.path.join(out_dir, "graph_edges.parquet"), index=False)
    fi.to_parquet(os.path.join(out_dir, "graph_fanin.parquet"), index=False)
    pr.to_parquet(os.path.join(out_dir, "graph_propagation.parquet"), index=False)
    return edges, fi, pr, seeds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/test.duckdb")
    ap.add_argument("--out", default="data/out_test")
    # 씨앗은 확정에 가까운 경보만: 룰 히트. ML 경보는 검토 전이라 오탐에서 전파가 번진다.
    ap.add_argument("--seeds", nargs="*", default=["rules"])
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "graph.yaml"))
    a = ap.parse_args()
    edges, fi, pr, seeds = run(a.db, a.out, a.seeds, a.config)
    print(f"edges {len(edges):,} ({edges['kind'].value_counts().to_dict()}) · seeds {len(seeds):,}")
    print(f"fanin flagged {fi['account_id'].nunique():,} · propagation flagged {int(pr['flag'].sum()):,}")
