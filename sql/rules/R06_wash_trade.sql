-- R06 자전거래: 같은 두 계정이 같은 아이템을 {window_days}일 안에 양방향으로 {min_round_trips}회 이상 주고받음
WITH t AS (
    SELECT least(seller, buyer) AS a, greatest(seller, buyer) AS b, item_id, ts,
           seller < buyer AS dir
    FROM trades WHERE item_id > 0
), pairs AS (
    SELECT a, b, item_id, count(*) AS n
    FROM t GROUP BY 1, 2, 3
    HAVING count(*) >= {min_round_trips} AND count(DISTINCT dir) = 2
       AND (epoch(max(ts)) - epoch(min(ts))) / 86400.0 <= {window_days}
), acc AS (
    SELECT a AS account_id, n, item_id, b AS partner FROM pairs
    UNION ALL SELECT b, n, item_id, a FROM pairs
)
SELECT account_id, 'R06' AS rule_id, least(1.0, max(n) / 10.0) AS score,
       json_object('round_trips', max(n), 'item_id', arg_max(item_id, n), 'partner', arg_max(partner, n)) AS evidence
FROM acc GROUP BY 1
