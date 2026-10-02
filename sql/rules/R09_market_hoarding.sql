-- R09 매집: 같은 아이템을 거래소에서 {window_hours}시간 안에 {min_units}개 이상 구매
WITH b AS (
    SELECT buyer AS account_id, item_id, ts, qty FROM trades WHERE channel = 'market' AND item_id > 0
), w AS (
    SELECT account_id, item_id,
           sum(qty) OVER (PARTITION BY account_id, item_id ORDER BY ts
                          RANGE BETWEEN INTERVAL '{window_hours} hours' PRECEDING AND CURRENT ROW) AS units
    FROM b
)
SELECT account_id, 'R09' AS rule_id, least(1.0, max(units) / 30.0) AS score,
       json_object('max_units_in_window', max(units), 'item_id', arg_max(item_id, units)) AS evidence
FROM w GROUP BY 1 HAVING max(units) >= {min_units}
