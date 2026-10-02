-- 자기 자신과의 거래 금지
SELECT 'Q06_self_trade' AS check_name, count(*) AS n_bad FROM trades WHERE seller = buyer
