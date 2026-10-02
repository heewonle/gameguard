-- 행동이 자기 세션의 접속 구간 안에 있고, 세션 주인과 계정이 같은가
SELECT 'Q01_action_in_session' AS check_name, count(*) AS n_bad
FROM actions a JOIN sessions s USING (session_id)
WHERE a.ts < s.login_at OR a.ts > s.logout_at OR a.account_id <> s.account_id
