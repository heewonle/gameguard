"""일일 배치: 품질 체크 → 룰 → ML → 그래프 → 경보 → 조사 에이전트 → 요약.

ProjectMC 무인 실행 원칙을 따른다
  - 단계마다 로그를 남기고, 실패해도 예외를 밖으로 던지지 않는다 (스케줄러가 오류 창으로 멈추지 않게)
  - 앞 단계가 실패하면 그 결과에 기대는 뒤 단계는 건너뛰고 이유를 남긴다
  - 에이전트는 아직 조사하지 않은 경보만, 우선순위 순으로 예산(--agent-budget)만큼 조사한다 → 매일 돌려도 중복 조사 없음
  - 할 일이 없으면 조용히 끝난다

실서비스라면 매일 새 로그가 쌓인 DB 에 대해 돌린다. 이 저장소의 합성 월드는 30일치 고정 스냅숏이다.

  python -m gameguard.run --db data/test.duckdb --out data/out_test --agent-budget 20
"""
import argparse
import datetime as dt
import json
import os
import time
import traceback

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_LOG = "agent_reports.jsonl"


class Run:
    def __init__(self, out_dir: str):
        self.out = out_dir
        self.steps = []
        os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
        self.stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logf = open(os.path.join(out_dir, "runs", f"{self.stamp}.log"), "w", encoding="utf-8")

    def log(self, msg):
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def step(self, name, fn, needs=()):
        failed = [n for n in needs if not self.ok(n)]
        if failed:
            self.steps.append({"step": name, "status": "skipped", "reason": f"앞 단계 실패: {', '.join(failed)}"})
            self.log(f"SKIP {name} — 앞 단계 실패 {failed}")
            return None
        t0 = time.time()
        try:
            info = fn() or {}
            self.steps.append({"step": name, "status": "ok", "seconds": round(time.time() - t0, 1), **info})
            self.log(f"OK   {name} {round(time.time() - t0, 1)}s {json.dumps(info, ensure_ascii=False)}")
            return info
        except Exception as e:  # noqa: BLE001 — 배치는 멈추지 않는다
            self.steps.append({"step": name, "status": "failed", "error": f"{type(e).__name__}: {e}"})
            self.log(f"FAIL {name}: {e}\n{traceback.format_exc()}")
            return None

    def ok(self, name):
        return any(s["step"] == name and s["status"] == "ok" for s in self.steps)

    def finish(self):
        path = os.path.join(self.out, "runs", f"{self.stamp}.json")
        json.dump({"stamp": self.stamp, "steps": self.steps}, open(path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        self.log(f"done → {path}")
        self.logf.close()
        return path


def priority(alerts: pd.DataFrame) -> pd.DataFrame:
    """조사 순서: 여러 단계에 걸친 경보 → 룰 밖 경보(사람이 근거를 모르는 것) → ML 점수."""
    a = alerts.copy()
    a["n_sources"] = a["sources"].str.count(",") + 1
    a["no_rule"] = ~a["sources"].str.contains("rule")
    return a.sort_values(["n_sources", "no_rule", "ml_score"], ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/test.duckdb")
    ap.add_argument("--out", default="data/out_test")
    ap.add_argument("--agent-budget", type=int, default=20, help="이번 실행에서 에이전트가 조사할 최대 경보 수 (0 이면 생략)")
    ap.add_argument("--model", default=os.environ.get("GAMEGUARD_LLM", "qwen3-vl:8b-instruct"))
    ap.add_argument("--prompt", default="v3")
    a = ap.parse_args()

    from .alerts import build_alerts
    from .db import connect
    from .graph import run as run_graph
    from .models import score
    from .quality import run_quality
    from .rules import run_rules

    R = Run(a.out)
    R.log(f"start db={a.db} out={a.out}")

    def s_quality():
        q = run_quality(connect(a.db, read_only=True))
        bad = int(q["n_bad"].sum())
        if bad:
            raise RuntimeError(f"품질 체크 실패 {bad}건: {q[q['n_bad'] > 0].to_dict('records')}")
        return {"checks": len(q)}

    def s_rules():
        cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "rules.yaml"), encoding="utf-8"))
        hits = run_rules(connect(a.db, read_only=True), cfg, verbose=False)
        hits.to_parquet(os.path.join(a.out, "rule_hits.parquet"), index=False)
        return {"hits": len(hits), "accounts": int(hits["account_id"].nunique())}

    def s_ml():
        s = score(a.db, a.out, "sanctioned")
        return {"ml_flag": int(s["ml_flag"].sum())}

    def s_graph():
        _, fi, pr, seeds = run_graph(a.db, a.out, ["rules"], os.path.join(ROOT, "config", "graph.yaml"))
        return {"seeds": len(seeds), "fanin": int(fi["account_id"].nunique()), "propagation": int(pr["flag"].sum())}

    def s_alerts():
        al = build_alerts(a.out)
        al.to_parquet(os.path.join(a.out, "alerts.parquet"), index=False)
        return {"alerts": len(al)}

    def s_agent():
        if a.agent_budget <= 0:
            return {"investigated": 0, "note": "예산 0"}
        from .agent.agent import OllamaBackend, investigate
        from .agent.tools import Toolbox
        alerts = pd.read_parquet(os.path.join(a.out, "alerts.parquet"))
        ml_all = pd.read_parquet(os.path.join(a.out, "ml_scores_sanctioned.parquet"))["ml_score"]
        alerts["ml_percentile"] = alerts["ml_score"].map(lambda v: round(float((ml_all <= v).mean() * 100), 2))
        path = os.path.join(a.out, AGENT_LOG)
        done = {json.loads(l)["account_id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
        todo = priority(alerts[~alerts["account_id"].isin(done)]).head(a.agent_budget)
        if todo.empty:
            return {"investigated": 0, "note": "새 경보 없음"}
        tb = Toolbox(a.db, pd.read_parquet(os.path.join(a.out, "features.parquet")), alerts)
        be = OllamaBackend(a.model)
        n = 0
        with open(path, "a", encoding="utf-8") as fh:
            for _, row in todo.iterrows():
                res = investigate(be, tb, row, a.prompt)
                res.update(model=a.model, prompt=a.prompt, investigated_at=dt.datetime.now().isoformat(timespec="seconds"))
                fh.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
                fh.flush()
                n += 1
        return {"investigated": n, "remaining": int((~alerts["account_id"].isin(done)).sum()) - n}

    R.step("quality", s_quality)
    R.step("rules", s_rules, needs=["quality"])
    R.step("ml", s_ml, needs=["quality"])
    R.step("graph", s_graph, needs=["rules"])
    R.step("alerts", s_alerts, needs=["rules", "ml", "graph"])
    R.step("agent", s_agent, needs=["alerts"])
    R.finish()


if __name__ == "__main__":
    main()
