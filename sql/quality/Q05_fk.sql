-- 참조 무결성: 세션·거래·결제의 계정이 accounts 에 있는가
SELECT 'Q05_fk' AS check_name,
  (SELECT count(*) FROM sessions WHERE account_id NOT IN (SELECT account_id FROM accounts))
+ (SELECT count(*) FROM trades WHERE seller NOT IN (SELECT account_id FROM accounts)
                               OR buyer NOT IN (SELECT account_id FROM accounts))
+ (SELECT count(*) FROM payments WHERE account_id NOT IN (SELECT account_id FROM accounts)) AS n_bad
