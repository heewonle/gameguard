-- 거래 1건마다 구매자 출금·판매자 입금 2행이 통화 로그에 있는가
WITH t AS (SELECT ts, seller, buyer, channel FROM trades),
c AS (SELECT ts, account_id, reason, delta FROM currency_log WHERE reason IN ('p2p', 'market'))
SELECT 'Q04_trade_mirror' AS check_name,
       (SELECT count(*) FROM t) * 2 - (SELECT count(*) FROM c) AS n_bad
