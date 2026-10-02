-- R05 비정상 단가: 아이템별 거래 단가 로그값의 robust z(중앙값·MAD 기준) ≥ {min_robust_z}
-- 근거: RMT는 잡템에 큰 골드를 얹어 넘기고, 시세조작은 시세의 몇 배로 판다.
-- 아이템 없이 골드만 오가는 거래(item_id=0)는 제외.
-- 1:1 거래는 양쪽이 가격에 합의하므로 둘 다 표시. 거래소 구매자는 정해진 매물가를 받아들였을 뿐(피해자일 수 있음)이라 판매자만 표시.
WITH t AS (
    SELECT trade_id, seller, buyer, channel, item_id, ln(gold::DOUBLE / qty) AS lp
    FROM trades WHERE item_id > 0 AND qty > 0 AND gold > 0
), med AS (
    SELECT item_id, median(lp) AS med, count(*) AS n FROM t GROUP BY 1
), stat AS (
    SELECT item_id, any_value(m.med) AS med, median(abs(t.lp - m.med)) AS mad
    FROM t JOIN med m USING (item_id)
    WHERE m.n >= {min_item_trades}
    GROUP BY 1
), flagged AS (
    SELECT t.*, (t.lp - s.med) / (1.4826 * greatest(s.mad, 0.05)) AS z
    FROM t JOIN stat s USING (item_id)
    WHERE abs((t.lp - s.med) / (1.4826 * greatest(s.mad, 0.05))) >= {min_robust_z}
), both_sides AS (
    SELECT seller AS account_id, trade_id, z FROM flagged
    UNION ALL SELECT buyer, trade_id, z FROM flagged WHERE channel = 'p2p'
)
SELECT account_id, 'R05' AS rule_id,
       least(1.0, count(*) / 3.0) AS score,
       json_object('outlier_trades', count(*), 'max_abs_z', round(max(abs(z)), 1)) AS evidence
FROM both_sides GROUP BY 1
