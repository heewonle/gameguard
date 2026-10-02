-- R01 장시간 접속: 하루 접속 시간이 {long_hours}시간 이상인 날이 {min_days}일 이상
-- 근거: 작업장 봇은 하루 20시간 안팎을 돌린다. 하드코어 정상 유저도 14시간을 넘기기 어렵다.
WITH daily AS (
    SELECT account_id, login_at::DATE AS d,
           sum(epoch(logout_at) - epoch(login_at)) / 3600.0 AS hours
    FROM sessions GROUP BY 1, 2
)
SELECT account_id, 'R01' AS rule_id,
       least(1.0, count(*) FILTER (WHERE hours >= {long_hours}) / 20.0) AS score,
       json_object('long_days', count(*) FILTER (WHERE hours >= {long_hours}),
                   'active_days', count(*), 'max_hours', round(max(hours), 1)) AS evidence
FROM daily GROUP BY 1
HAVING count(*) FILTER (WHERE hours >= {long_hours}) >= {min_days}
