# GameGuard — 작업 안내

- 설계·진행 로그: 취업 준비 폴더의 design/P1 문서 (먼저 읽고, 작업 후 진행 로그 갱신)
- Python: `.venv\Scripts\python` (Windows). 출력에 한글이 있으므로 `PYTHONIOENCODING=utf-8`
- 개발 월드 `data/gameguard.duckdb` (seed 42)에서만 임계값·모델을 조정한다. 성능은 평가 월드 `data/test.duckdb` (seed 2026) 기준으로만 보고한다.
- 정답은 별도 파일 `data/<world>_labels.duckdb` 에 있고, 채점(`gameguard/evaluate.py`)만 ATTACH 한다. 탐지 코드·에이전트는 정답에 닿을 수 없다 (tests/test_data.py, tests/test_agent_tools.py 가 검사)
- 모델 학습(`gameguard/models.py`)은 개발 월드 정답만 `evaluate.load_labels` 로 읽는다
- README·이력서에는 실제로 측정한 숫자만 쓴다.
- 커밋 메시지는 한국어, 작성자 heewonle.
- LLM 조사 에이전트는 **로컬 Ollama**(무료, API 키 불필요)로 돌린다. 기본 모델 `qwen3-vl:8b-instruct`, 바꾸려면 `--model` 또는 환경변수 `GAMEGUARD_LLM`. 유료 API 백엔드는 `gameguard/agent/agent.py` 의 `chat()` 인터페이스만 맞춰 추가
- 에이전트 프롬프트도 개발 월드 경보로만 다듬고, 평가 월드는 고정된 프롬프트로 한 번만 돌린다
