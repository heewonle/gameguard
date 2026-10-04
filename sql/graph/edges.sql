-- 3단 그래프의 간선. 계정 쌍마다 한 행 (src → dst 방향이 있는 골드 이동 + 방향 없는 신원 공유).
--   transfer : src 가 dst 에게 대가 없이 넘긴 골드 합 (1:1 거래 + 거래소 고가 구매의 초과 지불분)
--   ip / device : 주 IP·주 기기가 같은 계정 쌍 (공용 IP·기기를 빼기 위해 {max_group}명 이하 그룹만)
WITH item_px AS (
    SELECT item_id, median(gold::DOUBLE / qty) AS px
    FROM trades WHERE channel = 'market' AND item_id > 0 GROUP BY 1
), tr AS (
    -- 거래소 구매는 시세의 {min_market_ratio}배 이상일 때만 의도적 이전으로 본다
    -- (2~4배 고가 매물을 산 사람은 시세조작의 피해자일 수 있다)
    SELECT t.buyer AS src, t.seller AS dst,
           greatest(t.gold - t.qty * coalesce(p.px, 0), 0) AS unrecip
    FROM trades t LEFT JOIN item_px p USING (item_id)
    WHERE t.channel = 'p2p' OR t.gold >= {min_market_ratio} * t.qty * coalesce(p.px, 1e18)
), transfer AS (
    SELECT src, dst, 'transfer' AS kind, sum(unrecip) AS gold, count(*) AS n
    FROM tr GROUP BY 1, 2 HAVING sum(unrecip) >= {min_edge_gold}
), prim AS (
    SELECT account_id,
           arg_max(device_id, n_dev) AS main_device,
           arg_max(ip_hash, n_ip) AS main_ip
    FROM (SELECT account_id, device_id, ip_hash,
                 count(*) OVER (PARTITION BY account_id, device_id) AS n_dev,
                 count(*) OVER (PARTITION BY account_id, ip_hash) AS n_ip
          FROM sessions)
    GROUP BY 1
), ip_grp AS (
    SELECT main_ip FROM prim GROUP BY 1 HAVING count(*) BETWEEN 2 AND {max_group}
), dev_grp AS (
    SELECT main_device FROM prim GROUP BY 1 HAVING count(*) BETWEEN 2 AND {max_group}
), ident AS (
    SELECT a.account_id AS src, b.account_id AS dst, 'ip' AS kind, 0 AS gold, 1 AS n
    FROM prim a JOIN prim b ON a.main_ip = b.main_ip AND a.account_id < b.account_id
    WHERE a.main_ip IN (SELECT main_ip FROM ip_grp)
    UNION ALL
    SELECT a.account_id, b.account_id, 'device', 0, 1
    FROM prim a JOIN prim b ON a.main_device = b.main_device AND a.account_id < b.account_id
    WHERE a.main_device IN (SELECT main_device FROM dev_grp)
)
SELECT * FROM transfer
UNION ALL
SELECT * FROM ident
