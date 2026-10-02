# GameGuard — MMORPG 경제 로그 이상탐지 + LLM 조사 에이전트

> 작업 중 (2026-10). 설계: `JobApplication2026/design/P1_GameGuard_이상탐지.md`

정답 라벨이 있는 **합성 MMORPG 로그**에서 작업장 봇·RMT 거래망·매크로·다계정·시세조작·차지백을 탐지한다.
탐지는 **SQL 룰 → 계정 피처 ML → 거래 그래프** 3단으로 쌓고, 걸린 계정은 LLM 조사 에이전트가 증거를 모아 보고서를 쓴다(제재는 사람 승인).

## 현재 결과 — 1단: SQL 룰 9개 (평가 월드, seed 2026)

룰 임계값은 개발 월드(seed 42)에서만 조정했고, 아래 숫자는 **한 번도 보지 않은 평가 월드** 기준이다.

| | 값 |
|---|---|
| 계정 / 어뷰징 | 21,159 / 1,159 |
| 정밀도 | **0.996** (오탐 3) |
| 재현율 | **0.683** |
| F1 | **0.811** |

| 유형 | 일반형 재현율 | 은닉형 재현율 |
|---|---:|---:|
| 작업장 봇 | 1.000 | 1.000 |
| 매크로 | 1.000 | 0.290 |
| RMT 링 | 0.986 | 0.113 |
| 다계정 | 1.000 | 0.005 |
| 시세조작 | 0.769 | 1.000 |
| 차지백 | 0.910 | 0.333 |

룰은 정밀하지만 **사람처럼 행동하는 은닉형 어뷰저를 거의 못 잡는다.** → 2단(ML)·3단(그래프)의 목표.
전체 표: [reports/eval_rules_test.md](reports/eval_rules_test.md)

## 데이터

`sim/` 이 30일치 월드를 만든다 (기본 설정 기준).

| 테이블 | 행 수 | 내용 |
|---|---:|---|
| accounts | 2.1만 | 생성일, 기기, IP, 길드, 레벨 |
| sessions | 39만 | 접속·종료, 기기·IP (PC방 접속 포함) |
| actions | 2,300만+ | 사냥·루팅·스킬·이동·채팅·제작, 사냥터, 좌표 |
| currency_log | 106만 | 골드 증감과 잔액 (드랍·퀘스트·상점·거래·결제·환불) |
| trades / market_listings | 6.7만 / 5.5만 | 1:1 거래, 거래소 |
| payments | 6.6천 | 결제·환불 |

**행위자:** 정상 2만 명 (헷갈리는 정상: 하루 8~14시간 하드코어, 기기를 공유하는 가족, 선물을 받는 길드장, 정상 환불)
\+ 어뷰저 6유형. 각 유형의 40%는 **은닉형**이다: 사람 같은 리듬의 봇, 지터를 넣은 매크로, 기기를 바꾸는 다계정, 골드를 쪼개 넘기는 RMT, 천천히 매집하는 시세조작, 거래소를 거치는 차지백.

정답 라벨은 `labels` 스키마에 따로 두고, 탐지 코드가 라벨을 참조하면 테스트가 실패한다.

## 탐지 1단 — SQL 룰 (`sql/rules/`, 임계값 `config/rules.yaml`)

| 룰 | 잡는 것 |
|---|---|
| R01 장시간 접속 | 하루 18시간+ 접속일 5일 이상 |
| R02 기계적 리듬 | 행동 간격 변동계수 < 0.3 세션 3개 이상 |
| R03 골드 유출 | 번 골드의 70% 이상을 대가 없이 3명 이하에게 송금 |
| R04 주 기기 공유 | 같은 기기를 주 기기로 쓰는 계정 4~30개 (PC방 제외) |
| R05 비정상 단가 | 아이템별 robust z ≥ 6 (거래소는 판매자만) |
| R06 자전거래 | 같은 쌍·같은 아이템 7일 내 양방향 3회+ |
| R07 차지백 순서 | 결제 → 24시간 내 송금 → 환불 |
| R08 골드 수거 | 5명+에게서 100만+ 골드, 자기 수입의 3배+ |
| R09 매집 | 같은 아이템 48시간 내 거래소 15개+ 구매 |

## 실행

```bash
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
python -m sim.generate                                   # 개발 월드 → data/raw
python -m sim.generate --seed 2026 --out data/raw_test   # 평가 월드
python -m gameguard.db load --raw data/raw_test --db data/test.duckdb
python -m gameguard.quality --db data/test.duckdb        # 품질 체크 6종
python -m gameguard.rules --db data/test.duckdb --out data/out_test
python -m gameguard.evaluate rules --db data/test.duckdb --hits data/out_test/rule_hits.parquet --md reports/eval_rules_test.md
python -m pytest
```

## 로드맵
- [x] 합성 로그 생성기 (어뷰징 6유형 + 은닉형 + 헷갈리는 정상)
- [x] DuckDB 적재, 품질 체크, 라벨 누수 테스트
- [x] SQL 룰 9개 + 개발/평가 월드 분리 평가
- [ ] 계정 피처 SQL + IsolationForest / LightGBM
- [ ] 거래 그래프 (RMT 링 단위 탐지)
- [ ] LLM 조사 에이전트 + 평가
- [ ] Streamlit 대시보드, 일일 배치
- [ ] 방법론 문서
