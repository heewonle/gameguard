-- R03 골드 유출 깔때기: 번 골드의 {min_out_ratio} 이상을 대가 없이 {max_partners}명 이하에게 1:1로 내보냄
-- 근거: 작업장 봇·부계정은 번 골드를 수거 계정/본계정으로 넘긴다. 차지백 결제자는 결제 골드를 넘긴다.
-- "대가 없이" = 지불 골드 − 받은 아이템의 거래소 중앙가. 정상가에 아이템을 산 거래는 유출이 아니다.
WITH item_px AS (
    SELECT item_id, median(gold::DOUBLE / qty) AS px
    FROM trades WHERE channel = 'market' AND item_id > 0 GROUP BY 1
), income AS (
    SELECT account_id, sum(delta) AS income
    FROM currency_log WHERE reason IN ('drop', 'quest', 'purchase') GROUP BY 1
), outflow AS (
    SELECT t.buyer AS account_id,
           sum(greatest(t.gold - t.qty * coalesce(p.px, 0), 0)) AS out_gold,
           count(DISTINCT t.seller) AS partners
    FROM trades t LEFT JOIN item_px p USING (item_id)
    WHERE t.channel = 'p2p' GROUP BY 1
)
SELECT o.account_id, 'R03' AS rule_id,
       least(1.0, o.out_gold / i.income) AS score,
       json_object('unreciprocated_out', round(o.out_gold), 'income', i.income,
                   'out_ratio', round(o.out_gold / i.income, 3), 'partners', o.partners) AS evidence
FROM outflow o JOIN income i USING (account_id)
WHERE i.income >= {min_income} AND o.out_gold >= {min_out_ratio} * i.income AND o.partners <= {max_partners}
