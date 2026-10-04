"""LLM 조사 에이전트: 경보 1건 → 도구로 증거 수집 → 구조화된 판정.

백엔드는 LLM 호출 한 가지 인터페이스(chat)만 맞추면 바꿀 수 있다. 기본은 로컬 Ollama (무료, 데이터가 PC 밖으로 나가지 않음).

  python -m gameguard.agent.agent --db data/test.duckdb --out data/out_test --n-rule 50 --n-other 100
  python -m gameguard.agent.agent --account 123456          # 한 건만
"""
import argparse
import json
import os
import random
import time

import pandas as pd

from .tools import Toolbox, ollama_tools

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TOOL_ROUNDS = 6

# 속성 순서가 생성 순서다: 설명·근거를 먼저 쓰게 해서 판정이 설명을 따르도록 한다 (v1 에서 설명과 판정이 어긋난 사례)
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "evidence": {"type": "array", "maxItems": 5, "items": {
            "type": "object",
            "properties": {"claim": {"type": "string"}, "value": {"type": "string"}, "tool": {"type": "string"}},
            "required": ["claim", "value", "tool"]}},
        "abuse_type": {"type": "string", "enum": ["farm_bot", "rmt_ring", "macro", "multi_account",
                                                  "market_manip", "chargeback", "none"]},
        "verdict": {"type": "string", "enum": ["abuse", "normal", "uncertain"]},
        "confidence": {"type": "number"},
        "recommended_action": {"type": "string", "enum": ["sanction_review", "monitor", "dismiss"]},
    },
    "required": ["summary", "evidence", "abuse_type", "verdict", "confidence", "recommended_action"],
}
MIN_TOOL_CALLS = 3


class OllamaBackend:
    # 한 번의 응답 길이 상한. 정상 응답은 수백~1,700 토큰인데, 가끔 같은 문장을 반복하며 1만 토큰 넘게
    # 생성해 한 건에 4분씩 걸렸다 (개발·평가 월드 모두 관찰). 상한에 걸리면 판정 JSON 이 잘려 uncertain 처리된다.
    MAX_TOKENS = 2048

    def __init__(self, model: str, num_ctx: int = 16384):
        import ollama
        self.client = ollama.Client()
        self.model = model
        self.opts = {"num_ctx": num_ctx, "temperature": 0.1, "num_predict": self.MAX_TOKENS}

    def chat(self, messages, tools=None, fmt=None):
        r = self.client.chat(model=self.model, messages=messages, tools=tools, format=fmt, options=self.opts)
        m = r.message
        calls = [{"name": c.function.name, "args": dict(c.function.arguments or {})} for c in (m.tool_calls or [])]
        return {"content": m.content or "", "tool_calls": calls,
                "tokens_in": r.prompt_eval_count or 0, "tokens_out": r.eval_count or 0}


def alert_brief(row) -> str:
    """ML 은 원 확률 대신 전체 계정 대비 백분위로 준다 (v1: 검토 큐 계정의 확률이 0.0으로 반올림돼 '정상'으로 읽힘)."""
    details = json.loads(row["details"])
    for d in details:
        if d.get("stage") == "ml":
            d.pop("ml_score", None)
            d["ml_percentile"] = row.get("ml_percentile")
    return json.dumps({"account_id": int(row["account_id"]), "alert_sources": row["sources"].split(","),
                       "details": details}, ensure_ascii=False)


def investigate(backend, tb: Toolbox, row, prompt_version="v1") -> dict:
    system = open(os.path.join(HERE, "prompts", f"system_{prompt_version}.md"), encoding="utf-8").read()
    brief = alert_brief(row)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "다음 경보를 조사해줘.\n" + brief}]
    t0, tin, tout, transcript = time.time(), 0, 0, []
    tb.calls = []
    for _ in range(MAX_TOOL_ROUNDS):
        r = backend.chat(msgs, tools=ollama_tools())
        tin, tout = tin + r["tokens_in"], tout + r["tokens_out"]
        if not r["tool_calls"]:
            msgs.append({"role": "assistant", "content": r["content"]})
            if len(transcript) < MIN_TOOL_CALLS and prompt_version not in ("v1",):
                msgs.append({"role": "user", "content": f"아직 도구를 {len(transcript)}번만 호출했다. "
                             f"판정 전에 최소 {MIN_TOOL_CALLS}번 도구로 증거를 확인해줘."})
                continue
            break
        msgs.append({"role": "assistant", "content": r["content"],
                     "tool_calls": [{"function": {"name": c["name"], "arguments": c["args"]}} for c in r["tool_calls"]]})
        for c in r["tool_calls"]:
            out = tb.call(c["name"], c["args"])
            txt = json.dumps(out, ensure_ascii=False, default=str)
            transcript.append({"tool": c["name"], "args": c["args"], "result": txt})
            msgs.append({"role": "tool", "content": txt, "tool_name": c["name"]})
    msgs.append({"role": "user", "content": "조사를 마치고 최종 판정을 JSON 으로만 답해줘. evidence 의 value 는 도구 결과의 숫자를 그대로 쓴다. summary 는 한국어 2~3문장."})
    r = backend.chat(msgs, fmt=VERDICT_SCHEMA)
    tin, tout = tin + r["tokens_in"], tout + r["tokens_out"]
    try:
        verdict = json.loads(r["content"])
    except json.JSONDecodeError:
        verdict = {"verdict": "uncertain", "abuse_type": "none", "confidence": 0.0, "evidence": [],
                   "recommended_action": "monitor", "summary": "판정 JSON 파싱 실패", "_raw": r["content"][:500]}
    # 정책: 어뷰징 판정은 항상 사람 검토를 거친다 (LLM이 정하지 않음)
    verdict["needs_human_review"] = verdict.get("verdict") != "normal"
    return {"account_id": int(row["account_id"]), "sources": row["sources"], "brief": brief, **verdict,
            "tool_calls": len(transcript), "transcript": transcript,
            "tokens_in": tin, "tokens_out": tout, "seconds": round(time.time() - t0, 1)}


def sample_alerts(alerts: pd.DataFrame, n_rule: int, n_other: int, seed: int = 0, exclude=()) -> pd.DataFrame:
    """평가 표본: 룰에 걸린 경보 n_rule + 룰 밖(ML·그래프만) 경보 n_other. 라벨이 아니라 경보 출처로만 고른다."""
    alerts = alerts[~alerts["account_id"].isin(set(exclude))]
    has_rule = alerts["sources"].str.contains("rule")
    a = alerts[has_rule].sample(min(n_rule, has_rule.sum()), random_state=seed)
    b = alerts[~has_rule].sample(min(n_other, (~has_rule).sum()), random_state=seed)
    return pd.concat([a, b]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/test.duckdb")
    ap.add_argument("--out", default="data/out_test")
    ap.add_argument("--model", default=os.environ.get("GAMEGUARD_LLM", "qwen3-vl:8b-instruct"))
    ap.add_argument("--prompt", default="v3")
    ap.add_argument("--n-rule", type=int, default=50)
    ap.add_argument("--n-other", type=int, default=100)
    ap.add_argument("--account", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--exclude-from", help="이 jsonl 에 있는 계정은 표본에서 뺀다 (프롬프트 확인용 별도 표본)")
    a = ap.parse_args()

    alerts = pd.read_parquet(os.path.join(a.out, "alerts.parquet"))
    feats = pd.read_parquet(os.path.join(a.out, "features.parquet"))
    ml_all = pd.read_parquet(os.path.join(a.out, "ml_scores_sanctioned.parquet"))["ml_score"]
    alerts["ml_percentile"] = alerts["ml_score"].map(lambda v: round(float((ml_all <= v).mean() * 100), 2))
    tb = Toolbox(a.db, feats, alerts)
    be = OllamaBackend(a.model)
    excl = [json.loads(l)["account_id"] for l in open(a.exclude_from, encoding="utf-8")] if a.exclude_from else []
    rows = (alerts[alerts["account_id"] == a.account] if a.account
            else sample_alerts(alerts, a.n_rule, a.n_other, a.sample_seed, excl))
    tag = a.tag or f"{a.model.replace(':', '_').replace('/', '_')}_{a.prompt}"
    path = os.path.join(a.out, f"agent_{tag}.jsonl")
    done = set()
    if os.path.exists(path) and not a.account:
        done = {json.loads(l)["account_id"] for l in open(path, encoding="utf-8")}
    random.seed(0)
    with open(path, "a", encoding="utf-8") as fh:
        for i, row in rows.iterrows():
            if int(row["account_id"]) in done:
                continue
            res = investigate(be, tb, row, a.prompt)
            res["model"], res["prompt"] = a.model, a.prompt
            fh.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            print(f"[{i + 1}/{len(rows)}] {res['account_id']} {res.get('verdict')}/{res.get('abuse_type')} "
                  f"tools={res['tool_calls']} {res['seconds']}s", flush=True)
    print("saved", path)


if __name__ == "__main__":
    main()
