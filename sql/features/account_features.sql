-- 계정 피처 (계정당 1행). 2단 ML 입력.
-- 룰이 놓친 은닉형을 겨냥한 피처를 함께 둔다:
--   은닉형 봇   → 하루 가동 시간 분포, 행동 다양성(채팅·이동 비율), 좌표 퍼짐, 스킬 가짓수
--   은닉형 매크로 → 접속 시각 고정성, 새벽 접속 비율, 세션별 리듬 최솟값
--   은닉형 다계정 → 주 IP 공유 계정 수(기기는 바꿔도 IP는 같음), 신규 계정 여부, 송금 상대 집중도
--   은닉형 RMT  → 아이템 없는 소액 입금 횟수, 입금 대비 자기 수입
--   은닉형 차지백 → 결제·환불, 거래소 고가 구매
WITH
sess AS (
    SELECT account_id, session_id, login_at, logout_at, device_id, ip_hash,
           (epoch(logout_at) - epoch(login_at)) / 3600.0 AS hours,
           hour(login_at) + minute(login_at) / 60.0 AS login_hour
    FROM sessions
),
daily AS (
    SELECT account_id, login_at::DATE AS d, sum(hours) AS h FROM sess GROUP BY 1, 2
),
f_time AS (
    SELECT d.account_id,
           count(*) AS active_days,
           sum(h) AS total_hours,
           avg(h) AS hours_per_day,
           max(h) AS max_day_hours,
           quantile_cont(h, 0.9) AS p90_day_hours,
           avg((h >= 10)::INT) AS frac_days_10h
    FROM daily d GROUP BY 1
),
f_sess AS (
    SELECT account_id,
           count(*) AS n_sessions,
           avg(hours) AS mean_session_h,
           stddev_pop(login_hour) AS login_hour_std,
           avg((login_hour < 6)::INT) AS frac_dawn_login
    FROM sess GROUP BY 1
),
f_hour AS (  -- 가장 자주 시작하는 1시간대가 차지하는 비율 (예약 실행 매크로면 높다)
    SELECT account_id, max(cnt) / sum(cnt) AS top_hour_share
    FROM (SELECT account_id, floor(login_hour) AS hb, count(*) AS cnt FROM sess GROUP BY 1, 2)
    GROUP BY 1
),
period AS (SELECT min(login_at)::DATE::TIMESTAMP AS p0 FROM sessions),
gaps AS (
    SELECT account_id, session_id, x, y, action_type, map_id, skill_id,
           epoch(ts) - epoch(lag(ts) OVER (PARTITION BY session_id ORDER BY ts)) AS gap
    FROM actions
),
per_sess AS (
    SELECT account_id, session_id, count(*) AS n,
           stddev_samp(gap) / nullif(avg(gap), 0) AS cv,
           (stddev_pop(x) + stddev_pop(y)) / 2 AS spread,
           count(DISTINCT map_id) AS maps
    FROM gaps GROUP BY 1, 2
),
f_rhythm AS (
    SELECT account_id,
           median(cv) FILTER (WHERE n >= 30) AS median_cv,
           min(cv) FILTER (WHERE n >= 30) AS min_cv,
           avg((cv < 0.5)::INT) FILTER (WHERE n >= 30) AS frac_sess_cv_lt_05,
           avg(spread) FILTER (WHERE n >= 30) AS mean_spread,
           avg(maps) AS maps_per_session
    FROM per_sess GROUP BY 1
),
f_act AS (
    SELECT account_id,
           count(*) AS n_actions,
           avg((action_type = 'chat')::INT) AS share_chat,
           avg((action_type = 'move')::INT) AS share_move,
           avg((action_type = 'craft')::INT) AS share_craft,
           avg((action_type = 'loot')::INT) AS share_loot,
           count(DISTINCT skill_id) AS n_skills,
           count(DISTINCT map_id) AS n_maps
    FROM actions GROUP BY 1
),
f_cur AS (
    SELECT account_id,
           sum(delta) FILTER (WHERE reason IN ('drop', 'quest', 'purchase')) AS income,
           sum(delta) FILTER (WHERE reason = 'drop') AS drop_gold,
           -sum(delta) FILTER (WHERE reason = 'shop') AS shop_spend,
           min(balance_after) AS min_balance
    FROM currency_log GROUP BY 1
),
item_px AS (
    SELECT item_id, median(gold::DOUBLE / qty) AS px
    FROM trades WHERE channel = 'market' AND item_id > 0 GROUP BY 1
),
tr AS (
    SELECT t.*, greatest(t.gold - t.qty * coalesce(p.px, 0), 0) AS unrecip,
           CASE WHEN t.item_id > 0 AND p.px > 0 THEN t.gold / (t.qty * p.px) END AS price_ratio
    FROM trades t LEFT JOIN item_px p USING (item_id)
),
f_out AS (   -- 내가 골드를 낸 쪽 (buyer)
    SELECT buyer AS account_id,
           sum(unrecip) FILTER (WHERE channel = 'p2p') AS p2p_out_unrecip,
           count(*) FILTER (WHERE channel = 'p2p') AS p2p_out_n,
           count(DISTINCT seller) FILTER (WHERE channel = 'p2p') AS p2p_out_partners,
           count(*) FILTER (WHERE channel = 'p2p' AND item_id = 0) AS p2p_out_gift_n,
           count(*) FILTER (WHERE channel = 'market') AS mkt_buy_n,
           max(price_ratio) FILTER (WHERE channel = 'market') AS mkt_buy_max_ratio,
           sum(unrecip) FILTER (WHERE channel = 'market') AS mkt_buy_overpay
    FROM tr GROUP BY 1
),
f_out_top AS (
    SELECT account_id, max(g) / sum(g) AS out_top_partner_share FROM (
        SELECT buyer AS account_id, seller, sum(gold) AS g FROM trades WHERE channel = 'p2p' GROUP BY 1, 2
    ) GROUP BY 1
),
f_in AS (    -- 내가 골드를 받은 쪽 (seller)
    SELECT seller AS account_id,
           sum(unrecip) FILTER (WHERE channel = 'p2p') AS p2p_in_unrecip,
           count(DISTINCT buyer) FILTER (WHERE channel = 'p2p') AS p2p_in_senders,
           count(*) FILTER (WHERE channel = 'p2p' AND item_id = 0) AS p2p_in_gift_n,
           count(*) FILTER (WHERE channel = 'market') AS mkt_sell_n,
           max(price_ratio) FILTER (WHERE channel = 'market') AS mkt_sell_max_ratio
    FROM tr GROUP BY 1
),
f_hoard AS (
    SELECT account_id, max(units) AS max_same_item_buys FROM (
        SELECT buyer AS account_id, item_id, count(*) AS units
        FROM trades WHERE channel = 'market' AND item_id > 0 GROUP BY 1, 2
    ) GROUP BY 1
),
f_pay AS (
    SELECT account_id,
           count(*) FILTER (WHERE event = 'paid') AS n_paid,
           count(*) FILTER (WHERE event = 'refunded') AS n_refund,
           sum(amount_krw) FILTER (WHERE event = 'paid') AS paid_krw
    FROM payments GROUP BY 1
),
-- 주 기기·주 IP (세션 과반) 를 같이 쓰는 계정 수
prim AS (
    SELECT account_id,
           arg_max(device_id, n_dev) AS main_device,
           arg_max(ip_hash, n_ip) AS main_ip
    FROM (SELECT account_id, device_id, ip_hash,
                 count(*) OVER (PARTITION BY account_id, device_id) AS n_dev,
                 count(*) OVER (PARTITION BY account_id, ip_hash) AS n_ip
          FROM sessions)
    GROUP BY 1
),
f_ident AS (
    SELECT p.account_id,
           count(*) OVER (PARTITION BY p.main_device) AS accounts_on_main_device,
           count(*) OVER (PARTITION BY p.main_ip) AS accounts_on_main_ip
    FROM prim p
)
SELECT a.account_id,
       a.level,
       date_diff('day', a.created_at, period.p0) AS account_age_days,
       (a.created_at >= period.p0)::INT AS created_in_period,
       coalesce(t.active_days, 0) AS active_days, t.total_hours, t.hours_per_day, t.max_day_hours,
       t.p90_day_hours, t.frac_days_10h,
       s.n_sessions, s.mean_session_h, s.login_hour_std, s.frac_dawn_login, fh.top_hour_share,
       r.median_cv, r.min_cv, r.frac_sess_cv_lt_05, r.mean_spread, r.maps_per_session,
       ac.n_actions, ac.share_chat, ac.share_move, ac.share_craft, ac.share_loot, ac.n_skills, ac.n_maps,
       coalesce(c.income, 0) AS income, coalesce(c.drop_gold, 0) AS drop_gold,
       coalesce(c.shop_spend, 0) / nullif(c.income, 0) AS shop_ratio, c.min_balance,
       coalesce(o.p2p_out_unrecip, 0) AS p2p_out_unrecip,
       coalesce(o.p2p_out_unrecip, 0) / nullif(c.income, 0) AS out_ratio,
       coalesce(o.p2p_out_n, 0) AS p2p_out_n, coalesce(o.p2p_out_partners, 0) AS p2p_out_partners,
       coalesce(o.p2p_out_gift_n, 0) AS p2p_out_gift_n, ot.out_top_partner_share,
       coalesce(o.mkt_buy_n, 0) AS mkt_buy_n, o.mkt_buy_max_ratio,
       coalesce(o.mkt_buy_overpay, 0) / nullif(c.income, 0) AS mkt_overpay_ratio,
       coalesce(i.p2p_in_unrecip, 0) AS p2p_in_unrecip,
       coalesce(i.p2p_in_unrecip, 0) / greatest(coalesce(c.income, 0), 1000) AS in_ratio,
       coalesce(i.p2p_in_senders, 0) AS p2p_in_senders, coalesce(i.p2p_in_gift_n, 0) AS p2p_in_gift_n,
       coalesce(i.mkt_sell_n, 0) AS mkt_sell_n, i.mkt_sell_max_ratio,
       coalesce(h.max_same_item_buys, 0) AS max_same_item_buys,
       coalesce(p.n_paid, 0) AS n_paid, coalesce(p.n_refund, 0) AS n_refund, coalesce(p.paid_krw, 0) AS paid_krw,
       coalesce(id.accounts_on_main_device, 0) AS accounts_on_main_device,
       coalesce(id.accounts_on_main_ip, 0) AS accounts_on_main_ip
FROM accounts a CROSS JOIN period
LEFT JOIN f_time t USING (account_id)
LEFT JOIN f_sess s USING (account_id)
LEFT JOIN f_hour fh USING (account_id)
LEFT JOIN f_rhythm r USING (account_id)
LEFT JOIN f_act ac USING (account_id)
LEFT JOIN f_cur c USING (account_id)
LEFT JOIN f_out o USING (account_id)
LEFT JOIN f_out_top ot USING (account_id)
LEFT JOIN f_in i USING (account_id)
LEFT JOIN f_hoard h USING (account_id)
LEFT JOIN f_pay p USING (account_id)
LEFT JOIN f_ident id USING (account_id)
ORDER BY a.account_id
