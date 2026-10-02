# GameGuard — 작업 안내

- 설계·진행 로그: `C:\Users\lhw45\Downloads\JobApplication2026\design\P1_GameGuard_이상탐지.md` (먼저 읽고, 작업 후 진행 로그 갱신)
- Python: `.venv\Scripts\python` (Windows). 출력에 한글이 있으므로 `PYTHONIOENCODING=utf-8`
- 개발 월드 `data/gameguard.duckdb` (seed 42)에서만 임계값·모델을 조정한다. 성능은 평가 월드 `data/test.duckdb` (seed 2026) 기준으로만 보고한다.
- 탐지 코드(`sql/rules`, `sql/features`, `gameguard/`)는 `labels` 스키마를 읽으면 안 된다. 예외: `gameguard/db.py`, `gameguard/evaluate.py` (tests/test_data.py 가 검사)
- README·이력서에는 실제로 측정한 숫자만 쓴다.
- 커밋 메시지는 한국어, 작성자 heewonle.
