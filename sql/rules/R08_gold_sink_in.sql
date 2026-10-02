-- R08 골드 수거: {min_senders}명 이상에게서 1:1로 받은 골드가 {min_in_gold} 이상이고 자기 수입의 {min_in_ratio}배 이상
-- 근거: 작업장 수거·판매 계정, 다계정 본계정은 남의 골드가 대부분이다.
-- 길드장도 선물을 받지만 금액이 작다 (헷갈리는 정상).
WITH inflow AS (
    SELECT seller AS account_id, sum(gold) AS in_gold, count(DISTINCT buyer) AS senders
    FROM trades WHERE channel = 'p2p' GROUP BY 1
), own AS (
    SELECT account_id, sum(delta) AS own_income
    FROM currency_log WHERE reason IN ('drop', 'quest', 'purchase') GROUP BY 1
)
SELECT i.account_id, 'R08' AS rule_id,
       least(1.0, ln(i.in_gold / greatest(coalesce(o.own_income, 0), 1)) / 5.0) AS score,
       json_object('in_gold', i.in_gold, 'senders', i.senders, 'own_income', coalesce(o.own_income, 0)) AS evidence
FROM inflow i LEFT JOIN own o USING (account_id)
WHERE i.senders >= {min_senders} AND i.in_gold >= {min_in_gold}
  AND i.in_gold >= {min_in_ratio} * greatest(coalesce(o.own_income, 0), 1)
