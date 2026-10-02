-- R02 기계적 리듬: 행동 간격 변동계수(표준편차/평균)가 {max_cv} 미만인 세션이 {min_sessions}개 이상
-- 근거: 사람의 행동 간격은 들쭉날쭉하다(CV≈1). 봇·매크로는 거의 일정하다.
WITH gaps AS (
    SELECT session_id, account_id,
           epoch(ts) - epoch(lag(ts) OVER (PARTITION BY session_id ORDER BY ts)) AS gap
    FROM actions
), sess AS (
    SELECT account_id, session_id, count(*) AS n, stddev_samp(gap) / avg(gap) AS cv
    FROM gaps WHERE gap IS NOT NULL GROUP BY 1, 2
    HAVING count(*) >= {min_actions}
)
SELECT account_id, 'R02' AS rule_id,
       least(1.0, count(*) FILTER (WHERE cv < {max_cv}) / 10.0) AS score,
       json_object('regular_sessions', count(*) FILTER (WHERE cv < {max_cv}),
                   'sessions', count(*), 'min_cv', round(min(cv), 3)) AS evidence
FROM sess GROUP BY 1
HAVING count(*) FILTER (WHERE cv < {max_cv}) >= {min_sessions}
