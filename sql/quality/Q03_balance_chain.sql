-- 잔액 연속성: balance_after = 직전 balance_after + delta
SELECT 'Q03_balance_chain' AS check_name, count(*) AS n_bad
FROM (SELECT delta, balance_after,
             lag(balance_after) OVER (PARTITION BY account_id ORDER BY log_id) AS prev
      FROM currency_log)
WHERE prev IS NOT NULL AND balance_after <> prev + delta
