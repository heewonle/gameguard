-- 같은 계정의 세션이 겹치지 않는가
SELECT 'Q02_session_overlap' AS check_name, count(*) AS n_bad
FROM (SELECT login_at, lag(logout_at) OVER (PARTITION BY account_id ORDER BY login_at) AS prev_out
      FROM sessions)
WHERE login_at < prev_out
