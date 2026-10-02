-- R07 차지백 순서: 결제 → {transfer_hours}시간 안에 결제 골드의 {min_transfer_ratio} 이상을 1:1 송금 → 그 결제 환불
-- 결제자와 송금 받은 상대 모두 표시한다.
WITH paid AS (SELECT payment_id, account_id, ts, amount_krw FROM payments WHERE event = 'paid'),
refund AS (SELECT payment_id, ts AS refund_ts FROM payments WHERE event = 'refunded'),
rate AS (  -- 원 → 골드 환율 (결제 시 지급 로그로 추정)
    SELECT median(c.delta / p.amount_krw) AS gpk
    FROM currency_log c JOIN paid p ON c.account_id = p.account_id AND c.ts = p.ts AND c.reason = 'purchase'
), xfer AS (
    SELECT p.payment_id, p.account_id, t.seller AS receiver, sum(t.gold) AS gold,
           any_value(p.amount_krw) AS krw
    FROM paid p JOIN trades t
      ON t.buyer = p.account_id AND t.channel = 'p2p'
     AND t.ts BETWEEN p.ts AND p.ts + INTERVAL '{transfer_hours} hours'
    GROUP BY 1, 2, 3
), hits AS (
    SELECT x.* FROM xfer x JOIN refund r USING (payment_id), rate
    WHERE x.gold >= {min_transfer_ratio} * x.krw * rate.gpk
), acc AS (
    SELECT account_id, payment_id, receiver AS partner FROM hits
    UNION ALL SELECT receiver, payment_id, account_id FROM hits
)
SELECT account_id, 'R07' AS rule_id, least(1.0, 0.6 + 0.2 * count(DISTINCT payment_id)) AS score,
       json_object('refunded_payments_with_transfer', count(DISTINCT payment_id),
                   'partner', any_value(partner)) AS evidence
FROM acc GROUP BY 1
