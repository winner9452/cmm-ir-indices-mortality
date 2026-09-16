-- ============================================================
-- 01-population.sql
-- CMM(心血管代谢多病共存)队列提取
-- 暴露(8): TyG, SPISE, TG-HDL, METS-IR, AIP, TyG-BMI, TyG-RC, CHG
--   * 口径对齐 Wu et al. (eICU-CRD, Cardiovasc Diabetol 2026):
--     - AIP = Log10(TG/HDL)        [常用对数 log10, 标准 AIP 定义]
--     - CHG = Ln(TC×FBG/(2×HDL))    [Mansoori et al. 2025 标准定义, HDL 在分母; 审稿意见后纠正]
--     - TyG-RC = TyG × RC, RC = TC - HDL - LDL  [残余胆固醇, 须 > 0]
--     - SPISE = 600×HDL^0.185 / (TG^0.2 × BMI^1.338)
-- 结局: 院内死亡(主要) + 30/90/365天全因死亡率
-- 数据库: MIMIC-IV v2.2
-- ============================================================

-- Step 1: 心血管代谢疾病(CMD)标记
-- 注: CMM 采用三成分定义 = 糖尿病 / 心脏病(CHD或CHF或AF) / 中风, 中 ≥2 项共存
--     高血压/血脂异常/CKD/PAD 仍抽取, 但仅作协变量, 不计入 CMM
--     * 房颤(AF) 归入"心脏病"成分(与 CHD/CHF 并列), 参与 CMM 计数; 不单列为协变量
WITH cm_diseases AS (
    SELECT
        hadm_id,
        -- 1. 糖尿病(ICD, 全部类型: 250 / E08-E13)
        MAX(CASE WHEN
            (icd_version = 9 AND icd_code LIKE '250%')
            OR (icd_version = 10 AND (icd_code LIKE 'E08%' OR icd_code LIKE 'E09%'
                OR icd_code LIKE 'E10%' OR icd_code LIKE 'E11%' OR icd_code LIKE 'E13%'))
            THEN 1 ELSE 0 END) AS dm_icd,
        -- 2. 高血压
        MAX(CASE WHEN
            (icd_version = 9 AND (icd_code LIKE '401%' OR icd_code LIKE '402%' OR icd_code LIKE '403%' OR icd_code LIKE '404%' OR icd_code LIKE '405%'))
            OR (icd_version = 10 AND (icd_code LIKE 'I10%' OR icd_code LIKE 'I11%' OR icd_code LIKE 'I12%' OR icd_code LIKE 'I13%' OR icd_code LIKE 'I15%'))
            THEN 1 ELSE 0 END) AS hypertension,
        -- 3. 血脂异常
        MAX(CASE WHEN
            (icd_version = 9 AND icd_code LIKE '272%')
            OR (icd_version = 10 AND icd_code LIKE 'E78%')
            THEN 1 ELSE 0 END) AS dyslipidemia,
        -- 4. 冠心病/缺血性心脏病 (含心绞痛 413/I20)
        MAX(CASE WHEN
            (icd_version = 9 AND (icd_code LIKE '410%' OR icd_code LIKE '411%' OR icd_code LIKE '412%' OR icd_code LIKE '413%' OR icd_code LIKE '414%'))
            OR (icd_version = 10 AND (icd_code LIKE 'I20%' OR icd_code LIKE 'I21%' OR icd_code LIKE 'I22%' OR icd_code LIKE 'I24%' OR icd_code LIKE 'I25%'))
            THEN 1 ELSE 0 END) AS chd,
        -- 5. 心力衰竭
        MAX(CASE WHEN
            (icd_version = 9 AND (icd_code LIKE '428%' OR icd_code LIKE '4254%'))
            OR (icd_version = 10 AND (icd_code LIKE 'I50%' OR icd_code LIKE 'I42%'))
            THEN 1 ELSE 0 END) AS hf,
        -- 6. 卒中 (缺血性 + 出血性)
        MAX(CASE WHEN
            (icd_version = 9 AND (icd_code LIKE '430%' OR icd_code LIKE '431%' OR icd_code LIKE '432%'
                OR icd_code LIKE '433%' OR icd_code LIKE '434%' OR icd_code = '436'))
            OR (icd_version = 10 AND (icd_code LIKE 'I60%' OR icd_code LIKE 'I61%' OR icd_code LIKE 'I62%'
                OR icd_code LIKE 'I63%' OR icd_code LIKE 'I65%' OR icd_code LIKE 'I66%'))
            THEN 1 ELSE 0 END) AS stroke,
        -- 7. 慢性肾脏病
        MAX(CASE WHEN
            (icd_version = 9 AND icd_code LIKE '585%')
            OR (icd_version = 10 AND icd_code LIKE 'N18%')
            THEN 1 ELSE 0 END) AS ckd,
        -- 8. 外周动脉疾病
        MAX(CASE WHEN
            (icd_version = 9 AND icd_code LIKE '440%')
            OR (icd_version = 10 AND icd_code LIKE 'I70%')
            THEN 1 ELSE 0 END) AS pad,
        -- 9. 房颤(AF, 房颤/房扑)  [对齐 Wu et al. Model 3 协变量, 不计入 CMM 三成分]
        --    ICD-9 427.31(房颤)/427.32(房扑); ICD-10 I48(含 I48.0/.1/.2/.91 等)
        MAX(CASE WHEN
            (icd_version = 9 AND (icd_code = '42731' OR icd_code = '42732'))
            OR (icd_version = 10 AND icd_code LIKE 'I48%')
            THEN 1 ELSE 0 END) AS af
    FROM mimiciv_hosp.diagnoses_icd
    GROUP BY hadm_id
)

, cmm_flag AS (
    SELECT
        hadm_id,
        -- chd / hf / af 保留为独立列 (Table1/亚组分层用); 其余作协变量
        dm_icd, hypertension, dyslipidemia, chd, hf, stroke, ckd, pad, af,
        -- ★ 心脏病 = CHD 或 CHF 或 房颤(AF)  [房颤纳入心脏病成分, 参与 CMM 计数]
        GREATEST(chd, hf, af) AS heart_disease
        -- 注: 糖尿病综合判定(ICD+降糖药+HbA1c) 与 cm_count 在 Step 13 计算
    FROM cm_diseases
)

-- Step 2: 排除严重合并症 (终末期肾衰、肝硬化/肝衰竭、恶性肿瘤)
, severe_excl AS (
    SELECT DISTINCT hadm_id
    FROM mimiciv_hosp.diagnoses_icd
    WHERE
        (icd_version = 9 AND icd_code IN ('5855', '5856'))
        OR (icd_version = 10 AND (icd_code LIKE 'N18.5%' OR icd_code LIKE 'N18.6%'))
        OR (icd_version = 9 AND icd_code LIKE '571%')
        OR (icd_version = 10 AND (icd_code LIKE 'K70%' OR icd_code LIKE 'K71%' OR icd_code LIKE 'K74%'))
        OR (icd_version = 9 AND icd_code BETWEEN '140' AND '2099')
        OR (icd_version = 10 AND icd_code BETWEEN 'C00' AND 'C97')
)

-- Step 3: 首次ICU入住
, first_icu AS (
    SELECT
        s.subject_id, s.hadm_id, s.stay_id,
        s.intime, s.outtime,
        s.los AS icu_los_days,
        ROW_NUMBER() OVER (PARTITION BY s.subject_id ORDER BY s.intime) AS icu_order
    FROM mimiciv_icu.icustays s
)

-- Step 4: 人口统计学
, demographics AS (
    SELECT
        p.subject_id, p.gender,
        p.anchor_age + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year) AS age,
        a.hadm_id, a.admittime, a.dischtime, a.deathtime,
        a.hospital_expire_flag,
        a.race AS ethnicity,   -- MIMIC-IV v2.0+ 将 ethnicity 列改名为 race
        a.race AS race         -- ★ 同源另出一列 race, 供 R 端作协变量(R 内折叠 White/Black/Other)
    FROM mimiciv_hosp.patients p
    INNER JOIN mimiciv_hosp.admissions a ON p.subject_id = a.subject_id
)

-- Step 5: 身高 (chartevents + omr)
, height_all AS (
    SELECT ce.subject_id, ce.valuenum AS height_cm, 1 AS priority
    FROM mimiciv_icu.chartevents ce
    WHERE ce.itemid = 226730 AND ce.valuenum > 100 AND ce.valuenum < 250

    UNION ALL

    SELECT ce.subject_id, ROUND((ce.valuenum * 2.54)::numeric, 1), 2
    FROM mimiciv_icu.chartevents ce
    WHERE ce.itemid = 226707 AND ce.valuenum > 40 AND ce.valuenum < 100

    UNION ALL

    SELECT o.subject_id, ROUND((o.result_value::numeric * 2.54)::numeric, 1), 3
    FROM mimiciv_hosp.omr o
    WHERE o.result_name = 'Height (Inches)'
      AND o.result_value ~ '^\d+\.?\d*$'
      AND o.result_value::numeric > 40 AND o.result_value::numeric < 100

    UNION ALL

    SELECT o.subject_id,
           CASE WHEN o.result_value::numeric < 100
                THEN ROUND((o.result_value::numeric * 2.54)::numeric, 1)
                ELSE ROUND(o.result_value::numeric, 1) END,
           4
    FROM mimiciv_hosp.omr o
    WHERE o.result_name = 'Height'
      AND o.result_value ~ '^\d+\.?\d*$'
      AND o.result_value::numeric > 40 AND o.result_value::numeric < 250
)

, height_final AS (
    SELECT DISTINCT ON (subject_id) subject_id, height_cm
    FROM height_all WHERE height_cm > 100 AND height_cm < 250
    ORDER BY subject_id, priority
)

-- Step 6: 体重 (chartevents kg + lbs + omr)
, weight_sources AS (
    SELECT ie.stay_id, ie.subject_id, ce.valuenum AS weight_kg,
           ce.charttime::date AS wt_date, 1 AS priority
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_icu.chartevents ce ON ie.stay_id = ce.stay_id
    WHERE ce.itemid IN (224639, 226512) AND ce.valuenum > 20 AND ce.valuenum < 300
      AND ce.charttime BETWEEN ie.intime AND ie.intime + INTERVAL '24 hours'

    UNION ALL

    SELECT ie.stay_id, ie.subject_id,
           ROUND((ce.valuenum / 2.2046)::numeric, 1), ce.charttime::date, 2
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_icu.chartevents ce ON ie.stay_id = ce.stay_id
    WHERE ce.itemid = 226531 AND ce.valuenum > 50 AND ce.valuenum < 660
      AND ce.charttime BETWEEN ie.intime AND ie.intime + INTERVAL '24 hours'

    UNION ALL

    SELECT ie.stay_id, ie.subject_id,
           ROUND((o.result_value::numeric / 2.2046)::numeric, 1), o.chartdate, 3
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_hosp.omr o ON ie.subject_id = o.subject_id
    WHERE o.result_name = 'Weight (Lbs)'
      AND o.result_value ~ '^\d+\.?\d*$'
      AND o.result_value::numeric > 50 AND o.result_value::numeric < 660
)

, weight_final AS (
    SELECT DISTINCT ON (stay_id) stay_id, weight_kg
    FROM weight_sources WHERE weight_kg > 20 AND weight_kg < 300
    ORDER BY stay_id, priority, wt_date DESC
)

-- Step 7: 生命体征 (ICU 24h内)
, vitals AS (
    SELECT
        ce.stay_id,
        ROUND(AVG(CASE WHEN ce.itemid IN (220052, 220181, 225312) THEN ce.valuenum END)::numeric, 1) AS map,
        ROUND(AVG(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END)::numeric, 1) AS heart_rate,
        ROUND(AVG(CASE WHEN ce.itemid IN (220210, 224690) THEN ce.valuenum END)::numeric, 1) AS resp_rate,
        ROUND(AVG(CASE WHEN ce.itemid IN (220179, 220050) THEN ce.valuenum END)::numeric, 1) AS sbp,
        ROUND(AVG(CASE WHEN ce.itemid IN (220180, 220051) THEN ce.valuenum END)::numeric, 1) AS dbp,
        -- ★ 修正: 223761=华氏(°F), 223762=摄氏(°C); 原 SQL 直接混平均两种刻度 -> 数值无意义
        --   统一为摄氏度: F 转 C ((F-32)/1.8), C 原值; 各自加合理范围过滤掉异常值
        ROUND(AVG(CASE
            WHEN ce.itemid = 223762 AND ce.valuenum BETWEEN 30 AND 45 THEN ce.valuenum
            WHEN ce.itemid = 223761 AND ce.valuenum BETWEEN 86 AND 113 THEN (ce.valuenum - 32) / 1.8
        END)::numeric, 2) AS temperature,
        ROUND(AVG(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END)::numeric, 1) AS spo2
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_icu.chartevents ce ON ie.stay_id = ce.stay_id
        AND ce.charttime BETWEEN ie.intime AND ie.intime + INTERVAL '24 hours'
        AND ce.valuenum IS NOT NULL
    GROUP BY ce.stay_id
)

-- Step 8: 实验室 (★ 已对齐 Wu et al., Cardiovasc Diabetol 2026 的取数口径)
--   原版(已弃用): 窗口=整段住院(admittime→dischtime), 取离 ICU intime 最近一条(当初为对齐口径/降低缺失)。
--   现版: 改回该文口径 —— "first measurement upon ICU admission",
--         即入 ICU 首日 [intime, intime+24h] 内的【首次】测值
--         (该文 Fig.1 流程图要求 TG/FBG/TC/HDL 在 "first day of admission" 非缺失)。
--   ★ 与 eICU 版(01-population-eicu.sql 的 [0,1440] 分钟)现采用完全一致的"入 ICU 首日 24h"窗口 -> 两库口径对齐。
--   收益: 暴露血脂/葡萄糖更接近基线·空腹态, 规避病程后段治疗与濒死过程对指标的反向污染(reverse causation);
--         生命体征/体重/干预/严重度评分(first_day_*)本就为首日 -> 化验回到首日后, 基线快照时间口径统一。
--   代价: 缺失上升、可算 8 指标的 N 下降(该文亦因缺失 TG/FBG/TC/HDL 排除患者)。
--   ⚠ MIMIC 特有: 入院/急诊常在 intime 之前抽血(charttime < intime), 这类化验会被本窗排除 ->
--     血脂缺失可能比整段住院版更明显。若 MIMIC 血脂覆盖过稀, 可给窗口加一个"入ICU前缓冲"
--     (把下方 labs_raw 的 ie.intime 改成 ie.intime - INTERVAL '6 hours' 等), 仍属"首日就近取首次", 但略偏离本文字面。
--   ⚠ 例外 HbA1c: 反映近 3 月慢性血糖, 与采血时点无关且首日内极少开单; 故【单列 hba1c_raw】仍放宽到
--     整段住院取首次值, 仅供糖尿病判定(cmm_composite), 不作首日暴露 -> 避免无谓缺失, 不改变入组口径。
, labs_raw AS (   -- 急性暴露 + 报告化验: 入 ICU 首日 [intime, intime+24h], 取最早(首次)一条
    SELECT
        ie.stay_id, le.itemid, le.valuenum, le.charttime,
        ROW_NUMBER() OVER (
            PARTITION BY ie.stay_id, le.itemid
            ORDER BY le.charttime              -- ★ 首次 = charttime 最小(最早)
        ) AS rn
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_hosp.labevents le
        ON ie.hadm_id = le.hadm_id
        AND le.charttime BETWEEN ie.intime AND ie.intime + INTERVAL '24 hours'   -- ★ 入 ICU 首日 24h
        AND le.valuenum IS NOT NULL
)

-- HbA1c 单列(慢性指标): 整段住院窗(admittime→dischtime)取最早值, 仅供 DM 判定; 详见上方 ⚠ 说明
, hba1c_raw AS (
    SELECT
        ie.stay_id, le.valuenum,
        ROW_NUMBER() OVER (
            PARTITION BY ie.stay_id
            ORDER BY le.charttime
        ) AS rn
    FROM mimiciv_icu.icustays ie
    INNER JOIN mimiciv_hosp.admissions adm ON ie.hadm_id = adm.hadm_id
    INNER JOIN mimiciv_hosp.labevents le
        ON ie.hadm_id = le.hadm_id
        AND le.itemid = 50852
        AND le.charttime BETWEEN adm.admittime AND adm.dischtime
        AND le.valuenum IS NOT NULL
)

, labs AS (
    SELECT
        lr.stay_id,
        MAX(CASE WHEN lr.itemid = 51006    AND lr.rn = 1 THEN lr.valuenum END) AS bun,
        MAX(CASE WHEN lr.itemid = 50912    AND lr.rn = 1 THEN lr.valuenum END) AS creatinine,
        MAX(CASE WHEN lr.itemid = 51222    AND lr.rn = 1 THEN lr.valuenum END) AS hemoglobin,
        MAX(CASE WHEN lr.itemid = 51265    AND lr.rn = 1 THEN lr.valuenum END) AS platelet,
        MAX(CASE WHEN lr.itemid = 51279    AND lr.rn = 1 THEN lr.valuenum END) AS rbc,
        MAX(CASE WHEN lr.itemid = 51301    AND lr.rn = 1 THEN lr.valuenum END) AS wbc,
        MAX(CASE WHEN lr.itemid = 50983    AND lr.rn = 1 THEN lr.valuenum END) AS sodium,
        MAX(CASE WHEN lr.itemid = 50902    AND lr.rn = 1 THEN lr.valuenum END) AS chloride,
        MAX(CASE WHEN lr.itemid = 50893    AND lr.rn = 1 THEN lr.valuenum END) AS calcium,
        MAX(CASE WHEN lr.itemid = 50971    AND lr.rn = 1 THEN lr.valuenum END) AS potassium,
        MAX(CASE WHEN lr.itemid IN (50809, 50931) AND lr.rn = 1 THEN lr.valuenum END) AS glucose,
        -- ★ HbA1c 来自整段住院窗(hba1c_raw.rn=1), 慢性指标不受首日限制(详见上方 Step 8 ⚠ 说明)
        MAX(h.hba1c)                                                          AS hba1c,
        -- ★ 修正: MIMIC-IV 中 50907=Cholesterol Total, 50904=Cholesterol HDL (原 SQL 二者颠倒)
        MAX(CASE WHEN lr.itemid = 50907    AND lr.rn = 1 THEN lr.valuenum END) AS cholesterol,
        MAX(CASE WHEN lr.itemid = 50904    AND lr.rn = 1 THEN lr.valuenum END) AS hdl,
        -- LDL: 优先计算值(50905), 缺失则回退实测值(50906)
        --   注: ICU 中 50905(LDL-Calc, Friedewald, 需 TG<400)较稀疏; 仅取 50905 会致 ldl 几乎全空 -> tyg_rc 全空
        COALESCE(
            MAX(CASE WHEN lr.itemid = 50905 AND lr.rn = 1 THEN lr.valuenum END),
            MAX(CASE WHEN lr.itemid = 50906 AND lr.rn = 1 THEN lr.valuenum END)
        ) AS ldl,
        MAX(CASE WHEN lr.itemid = 51000    AND lr.rn = 1 THEN lr.valuenum END) AS triglycerides,
        MAX(CASE WHEN lr.itemid = 51237    AND lr.rn = 1 THEN lr.valuenum END) AS inr,
        MAX(CASE WHEN lr.itemid = 51274    AND lr.rn = 1 THEN lr.valuenum END) AS pt,
        MAX(CASE WHEN lr.itemid = 51275    AND lr.rn = 1 THEN lr.valuenum END) AS ptt,
        MAX(CASE WHEN lr.itemid = 50813    AND lr.rn = 1 THEN lr.valuenum END) AS lactate,
        MAX(CASE WHEN lr.itemid = 50882    AND lr.rn = 1 THEN lr.valuenum END) AS bicarbonate,
        MAX(CASE WHEN lr.itemid = 50885    AND lr.rn = 1 THEN lr.valuenum END) AS bilirubin,
        MAX(CASE WHEN lr.itemid = 50862    AND lr.rn = 1 THEN lr.valuenum END) AS albumin
    FROM labs_raw lr
    LEFT JOIN (SELECT stay_id, valuenum AS hba1c FROM hba1c_raw WHERE rn = 1) h
           ON lr.stay_id = h.stay_id
    GROUP BY lr.stay_id
)

-- Step 9: 疾病严重度评分
, severity AS (
    SELECT s.stay_id, s.sofa,
           sa.sapsii AS saps_ii,
           oa.oasis,
           ap.apsiii AS aps_iii
    FROM mimiciv_derived.first_day_sofa s
    LEFT JOIN mimiciv_derived.sapsii sa ON s.stay_id = sa.stay_id
    LEFT JOIN mimiciv_derived.oasis oa ON s.stay_id = oa.stay_id
    LEFT JOIN mimiciv_derived.apsiii ap ON s.stay_id = ap.stay_id
)

-- Step 10: 干预措施
, interventions AS (
    SELECT
        ie.stay_id,
        MAX(CASE WHEN vt.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS ventilation,
        MAX(CASE WHEN va.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS vasopressor
    FROM mimiciv_icu.icustays ie
    LEFT JOIN mimiciv_derived.ventilation vt ON ie.stay_id = vt.stay_id
        AND vt.starttime < ie.intime + INTERVAL '24 hours' AND vt.endtime > ie.intime
    LEFT JOIN mimiciv_derived.vasoactive_agent va ON ie.stay_id = va.stay_id
        AND va.starttime < ie.intime + INTERVAL '24 hours' AND va.endtime > ie.intime
    GROUP BY ie.stay_id
)

-- Step 11: 药物
, medications AS (
    SELECT
        ie.stay_id,
        MAX(CASE WHEN LOWER(pr.drug) LIKE '%aspirin%' THEN 1 ELSE 0 END) AS aspirin,
        MAX(CASE WHEN LOWER(pr.drug) SIMILAR TO '%(statin|atorva|simva|rosuva|prava|lova|fluva|pitava)%'
            THEN 1 ELSE 0 END) AS statin,
        MAX(CASE WHEN LOWER(pr.drug) SIMILAR TO '%(metformin|glyburide|glipizide|insulin|pioglitazone|sitagliptin)%'
            THEN 1 ELSE 0 END) AS antidiabetic,
        MAX(CASE WHEN LOWER(pr.drug) SIMILAR TO '%(lisinopril|enalapril|losartan|valsartan|amlodipine|metoprolol|atenolol|carvedilol)%'
            THEN 1 ELSE 0 END) AS antihypertensive,
        MAX(CASE WHEN LOWER(pr.drug) LIKE '%heparin%' OR LOWER(pr.drug) LIKE '%warfarin%'
            OR LOWER(pr.drug) LIKE '%enoxaparin%' THEN 1 ELSE 0 END) AS anticoagulant
    FROM mimiciv_icu.icustays ie
    LEFT JOIN mimiciv_hosp.prescriptions pr ON ie.hadm_id = pr.hadm_id
        AND pr.starttime BETWEEN ie.intime AND ie.intime + INTERVAL '24 hours'
    GROUP BY ie.stay_id
)

-- Step 12: 死亡结局 (院内死亡 + 入ICU后 30/90/365 天全因死亡率)
, outcomes AS (
    SELECT
        fi.stay_id, fi.subject_id, fi.intime,
        -- 院内死亡 (本次住院期间死亡, MIMIC hospital_expire_flag)
        COALESCE(demo.hospital_expire_flag, 0) AS inhosp_outcome,
        -- 院内死亡生存时间: ICU入住 → 院内死亡(事件)或出院(删失), 单位天, 下限 0.01 防 coxph 0/负时间
        GREATEST(
            EXTRACT(EPOCH FROM (COALESCE(demo.deathtime, demo.dischtime, fi.outtime) - fi.intime)) / 86400.0,
            0.01) AS surv_time_inhosp,
        -- 30天
        CASE WHEN demo.deathtime IS NOT NULL AND demo.deathtime <= fi.intime + INTERVAL '30 days' THEN 1
             WHEN pt.dod IS NOT NULL AND pt.dod <= CAST(fi.intime AS DATE) + 30 THEN 1
             ELSE 0 END AS day30_outcome,
        -- 90天
        CASE WHEN demo.deathtime IS NOT NULL AND demo.deathtime <= fi.intime + INTERVAL '90 days' THEN 1
             WHEN pt.dod IS NOT NULL AND pt.dod <= CAST(fi.intime AS DATE) + 90 THEN 1
             ELSE 0 END AS day90_outcome,
        -- 365天
        CASE WHEN demo.deathtime IS NOT NULL AND demo.deathtime <= fi.intime + INTERVAL '365 days' THEN 1
             WHEN pt.dod IS NOT NULL AND pt.dod <= CAST(fi.intime AS DATE) + 365 THEN 1
             ELSE 0 END AS day365_outcome,
        -- 生存时间 (右截尾于对应时间点)
        LEAST(COALESCE(
            EXTRACT(EPOCH FROM (demo.deathtime - fi.intime)) / 86400.0,
            (pt.dod - CAST(fi.intime AS DATE))::numeric, 30), 30) AS surv_time_30,
        LEAST(COALESCE(
            EXTRACT(EPOCH FROM (demo.deathtime - fi.intime)) / 86400.0,
            (pt.dod - CAST(fi.intime AS DATE))::numeric, 90), 90) AS surv_time_90,
        LEAST(COALESCE(
            EXTRACT(EPOCH FROM (demo.deathtime - fi.intime)) / 86400.0,
            (pt.dod - CAST(fi.intime AS DATE))::numeric, 365), 365) AS surv_time_365
    FROM first_icu fi
    LEFT JOIN demographics demo ON fi.hadm_id = demo.hadm_id
    LEFT JOIN mimiciv_hosp.patients pt ON fi.subject_id = pt.subject_id
)

-- Step 13: 三成分 CMM 计数 (糖尿病 = ICD诊断 OR 降糖药 OR HbA1c≥6.5%)
-- 注: 此处才能引用 stay_id 级的 medications / labs, 故 cm_count 在此计算
, cmm_composite AS (
    SELECT
        fi.stay_id,
        cf.hypertension, cf.dyslipidemia, cf.chd, cf.hf, cf.ckd, cf.pad, cf.af,
        cf.heart_disease, cf.stroke,
        -- 糖尿病(综合): ICD诊断 OR 降糖药 OR HbA1c≥6.5%
        GREATEST(cf.dm_icd,
                 COALESCE(med.antidiabetic, 0),
                 CASE WHEN l.hba1c >= 6.5 THEN 1 ELSE 0 END) AS dm,
        -- ★ CMM 三成分计数: 糖尿病 + 心脏病(CHD或CHF或AF) + 中风
        (GREATEST(cf.dm_icd,
                  COALESCE(med.antidiabetic, 0),
                  CASE WHEN l.hba1c >= 6.5 THEN 1 ELSE 0 END)
         + cf.heart_disease + cf.stroke) AS cm_count
    FROM first_icu fi
    INNER JOIN cmm_flag cf ON fi.hadm_id = cf.hadm_id
    LEFT JOIN medications med ON fi.stay_id = med.stay_id
    LEFT JOIN labs l ON fi.stay_id = l.stay_id
)

-- ============ 最终汇总 + 指标计算 ============
SELECT
    fi.subject_id, fi.hadm_id, fi.stay_id,
    fi.intime, fi.outtime, fi.icu_los_days, fi.icu_order,
    -- 人口统计
    demo.age, demo.gender, demo.ethnicity, demo.race,
    hf_t.height_cm, wf.weight_kg,
    CASE WHEN hf_t.height_cm > 0 AND wf.weight_kg > 0
         THEN ROUND((wf.weight_kg / ((hf_t.height_cm / 100.0) ^ 2))::numeric, 2)
         ELSE NULL END AS bmi,
    -- 生命体征
    v.map, v.heart_rate, v.resp_rate, v.sbp, v.dbp, v.temperature, v.spo2,
    -- 实验室
    l.bun, l.creatinine, l.hemoglobin, l.platelet, l.rbc, l.wbc,
    l.sodium, l.chloride, l.calcium, l.potassium, l.glucose,
    l.hba1c, l.cholesterol, l.hdl, l.ldl, l.triglycerides,
    l.inr, l.pt, l.ptt, l.lactate, l.bicarbonate,
    l.bilirubin, l.albumin,

    -- ★★★ 四大代谢指标 ★★★
    -- TyG = ln(TG × Glucose / 2)
    CASE WHEN l.triglycerides > 0 AND l.glucose > 0
         THEN ROUND(LN(l.triglycerides * l.glucose / 2.0)::numeric, 4)
         ELSE NULL END AS tyg,
    -- TyG-BMI = TyG × BMI
    CASE WHEN l.triglycerides > 0 AND l.glucose > 0 AND hf_t.height_cm > 0 AND wf.weight_kg > 0
         THEN ROUND((LN(l.triglycerides * l.glucose / 2.0) *
               (wf.weight_kg / ((hf_t.height_cm / 100.0) ^ 2)))::numeric, 2)
         ELSE NULL END AS tyg_bmi,
    -- METS-IR = ln(2×Glucose + TG) × BMI / ln(HDL)
    CASE WHEN l.glucose > 0 AND l.triglycerides > 0 AND l.hdl > 1
              AND hf_t.height_cm > 0 AND wf.weight_kg > 0
         THEN ROUND((LN(2.0 * l.glucose + l.triglycerides) *
               (wf.weight_kg / ((hf_t.height_cm / 100.0) ^ 2)) /
               LN(l.hdl))::numeric, 4)
         ELSE NULL END AS mets_ir,
    -- AIP = Log10(TG / HDL)   [标准 AIP 定义(Dobiášová): 常用对数 log10; PostgreSQL LOG()=log10]
    CASE WHEN l.triglycerides > 0 AND l.hdl > 0
         THEN ROUND(LOG(l.triglycerides / l.hdl)::numeric, 4)
         ELSE NULL END AS aip,
    -- CHG = Ln(TC × FBG / (2 × HDL))   [Mansoori et al., J Diabetes Investig 2025;16(2):309-14 标准定义, HDL 在分母; 审稿意见后纠正]
    --   * 标准定义下 CHG 约为 5–6 (原 "HDL 在分子" 误写版本约为 12.9, 已废弃)
    --   * 单位均 mg/dL; ICU 无真正空腹血糖, 以入科首次 glucose 代 FBG (与 TyG/METS-IR 一致)
    CASE WHEN l.cholesterol > 0 AND l.glucose > 0 AND l.hdl > 0
         THEN ROUND(LN(l.cholesterol * l.glucose / (2.0 * l.hdl))::numeric, 4)
         ELSE NULL END AS chg,

    -- ★★★ 新增 3 指标 (补齐 8 指标) ★★★
    -- TG-HDL = TG / HDL  (比值)
    CASE WHEN l.triglycerides > 0 AND l.hdl > 0
         THEN ROUND((l.triglycerides / l.hdl)::numeric, 4)
         ELSE NULL END AS tg_hdl,
    -- TyG-RC = TyG × RC,  RC(残余胆固醇) = TC - HDL - LDL (mg/dL); 须 RC > 0
    CASE WHEN l.triglycerides > 0 AND l.glucose > 0
              AND l.cholesterol > 0 AND l.hdl > 0 AND l.ldl > 0
              AND (l.cholesterol - l.hdl - l.ldl) > 0
         THEN ROUND((LN(l.triglycerides * l.glucose / 2.0)
                     * (l.cholesterol - l.hdl - l.ldl))::numeric, 4)
         ELSE NULL END AS tyg_rc,
    -- SPISE = 600 × HDL^0.185 / (TG^0.2 × BMI^1.338)   [胰岛素敏感性, 越高越好]
    CASE WHEN l.hdl > 0 AND l.triglycerides > 0
              AND hf_t.height_cm > 0 AND wf.weight_kg > 0
         THEN ROUND((600.0 * POWER(l.hdl, 0.185)
                     / (POWER(l.triglycerides, 0.2)
                        * POWER(wf.weight_kg / ((hf_t.height_cm / 100.0) ^ 2), 1.338)))::numeric, 4)
         ELSE NULL END AS spise,

    -- 疾病严重度
    sev.sofa, sev.saps_ii, sev.oasis, sev.aps_iii,
    -- CMM组分 (心脏病 = CHD 或 CHF 或 AF); af 已并入 heart_disease 参与 cm_count, 此处保留 af 独立列供 Table1
    cmm.dm, cmm.hypertension, cmm.dyslipidemia, cmm.chd, cmm.hf,
    cmm.heart_disease, cmm.stroke, cmm.ckd, cmm.pad, cmm.af, cmm.cm_count,
    -- 干预
    COALESCE(itv.ventilation, 0)   AS ventilation,
    COALESCE(itv.vasopressor, 0)   AS vasopressor,
    COALESCE(med.aspirin, 0)       AS aspirin,
    COALESCE(med.statin, 0)        AS statin,
    COALESCE(med.antidiabetic, 0)  AS antidiabetic,
    COALESCE(med.antihypertensive, 0) AS antihypertensive,
    COALESCE(med.anticoagulant, 0) AS anticoagulant,
    -- 结局 (院内死亡 + 30/90/365天全因死亡率)
    o.inhosp_outcome,
    o.day30_outcome, o.day90_outcome, o.day365_outcome,
    o.surv_time_inhosp,
    o.surv_time_30, o.surv_time_90, o.surv_time_365,
    -- 排除标记
    CASE WHEN se.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS severe_comorbid

FROM first_icu fi
INNER JOIN demographics demo ON fi.hadm_id = demo.hadm_id AND fi.subject_id = demo.subject_id
INNER JOIN cmm_composite cmm ON fi.stay_id = cmm.stay_id
LEFT JOIN severe_excl se ON fi.hadm_id = se.hadm_id
LEFT JOIN height_final hf_t ON fi.subject_id = hf_t.subject_id
LEFT JOIN weight_final wf ON fi.stay_id = wf.stay_id
LEFT JOIN vitals v ON fi.stay_id = v.stay_id
LEFT JOIN labs l ON fi.stay_id = l.stay_id
LEFT JOIN severity sev ON fi.stay_id = sev.stay_id
LEFT JOIN interventions itv ON fi.stay_id = itv.stay_id
LEFT JOIN medications med ON fi.stay_id = med.stay_id
LEFT JOIN outcomes o ON fi.stay_id = o.stay_id

WHERE cmm.cm_count >= 2;  -- CMM: 三成分(糖尿病/心脏病[CHD或CHF或AF]/中风)中 ≥2 项
