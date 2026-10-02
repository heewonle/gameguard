-- R04 주 사용 기기 공유: 같은 기기를 "주 기기"로 쓰는 계정이 {min_accounts}~{max_accounts}개
-- 근거: 가족 공유는 2~3명. 그보다 많으면 다계정/작업장.
-- 주 기기 = 그 계정 세션의 과반을 차지한 기기. PC방처럼 가끔 들르는 공용 기기는 주 기기가 되지 않는다.
WITH per AS (
    SELECT account_id, device_id, count(*) AS n,
           count(*) / sum(count(*)) OVER (PARTITION BY account_id) AS share
    FROM sessions GROUP BY 1, 2
), main_dev AS (
    SELECT account_id, device_id FROM per WHERE share > 0.5
), dev AS (
    SELECT device_id, count(*) AS n_acc FROM main_dev GROUP BY 1
    HAVING count(*) BETWEEN {min_accounts} AND {max_accounts}
)
SELECT m.account_id, 'R04' AS rule_id,
       least(1.0, d.n_acc / 10.0) AS score,
       json_object('device_id', m.device_id, 'accounts_on_device', d.n_acc) AS evidence
FROM main_dev m JOIN dev d USING (device_id)
