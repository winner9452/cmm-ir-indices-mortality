-- ============================================================
-- 01-population-eicu.sql
-- CMM(心血管代谢多病共存)队列提取 —— eICU-CRD 版
--   (迁移自 MIMIC-IV 版 01-population.sql)
--
-- 暴露(8): TyG, SPISE, TG-HDL, METS-IR, AIP, TyG-BMI, TyG-RC, CHG
--   * 口径对齐 Wu et al. (eICU-CRD, Cardiovasc Diabetol 2026):
--     - TyG     = Ln(TG × Glucose / 2)
--     - TyG-BMI = TyG × BMI
--     - METS-IR = Ln(2×Glucose + TG) × BMI / Ln(HDL)
--     - AIP     = Log10(TG / HDL)            [标准 Dobiášová 定义; PostgreSQL LOG()=log10]
--     - CHG     = Ln(TC × FBG / (2 × HDL))   [Mansoori et al. 2025 标准定义, HDL 在分母; 审稿意见后纠正]
--     - TG-HDL  = TG / HDL
--     - TyG-RC  = TyG × RC, RC = TC - HDL - LDL  [残余胆固醇, 须 > 0]
--     - SPISE   = 600 × HDL^0.185 / (TG^0.2 × BMI^1.338)   [越高=越敏感]
--   * 单位与 MIMIC 版一致, 均为 mg/dL(eICU 实验室即美制单位), 故公式直接沿用。
--
-- 结局: 院内死亡 (唯一终点)
--   ★ eICU 无出院后随访(无死亡日期), 故 MIMIC 版的 30/90/365 天终点在此删除,
--     ICU 死亡次要终点亦已移除; 院内死亡来自 patient.hospitaldischargestatus = 'Expired'。
--
-- 关键的 eICU vs MIMIC 差异:
--   1. 时间为"偏移量(分钟, 相对入 ICU 时刻)", 非时间戳; 24h = offset∈[0,1440]。
--   2. age 为字符, '> 89' 统一记为 90。
--   3. 身高/体重直接在 patient 表(admissionheight cm / admissionweight kg)。
--   4. 严重度改用 SOFA/SAPS-II/OASIS 三评分(Step 11, 口径=mimic-code 官方实现移植);
--      eICU 无原生表, 故按首日 min/max 自算; 不再使用 APACHE IVa。
--   5. 实验室/合并症名称为字符串, 用 labname / diagnosisstring + pastHistory 关键词匹配。
--   6. "首次 ICU"用 unitvisitnumber=1; 跨住院去重用 uniquepid(eICU 无跨次绝对时间, 近似)。
--
-- 数据库: eICU Collaborative Research Database v2.0, schema = eicu_icu
-- ============================================================

-- Step 1: 诊断/既往史合并文本源 (诊断 diagnosisstring + 既往史 pastHistory)
--   eICU 的 icd9code 可能同时含 ICD-9/10 且逗号分隔, 前缀匹配不可靠;
--   故以 diagnosisstring / pastHistory 关键词为主, icd9code 包含匹配为辅。
WITH dx_text AS (
    SELECT patientunitstayid,
           LOWER(diagnosisstring) AS txt,
           LOWER(COALESCE(icd9code, '')) AS icd
    FROM eicu_icu.diagnosis
    UNION ALL
    SELECT patientunitstayid,
           LOWER(COALESCE(pasthistorypath, '') || ' ' || COALESCE(pasthistoryvalue, '')) AS txt,
           '' AS icd
    FROM eicu_icu.pasthistory
)

-- Step 1b: 心血管代谢疾病(CMD)与协变量合并症标记 (每 ICU 停留)
--   CMM 三成分 = 糖尿病 / 心脏病(CHD或CHF或AF) / 中风, ≥2 项共存
--   高血压/血脂异常/CKD/PAD 仍抽取, 仅作协变量
, comorbid AS (
    SELECT
        patientunitstayid,
        -- 1. 糖尿病(排除尿崩症 diabetes insipidus)
        MAX(CASE WHEN (txt LIKE '%diabet%' AND txt NOT LIKE '%insipidus%')
                  OR icd LIKE '250%' OR icd LIKE '%e08%' OR icd LIKE '%e09%'
                  OR icd LIKE '%e10%' OR icd LIKE '%e11%' OR icd LIKE '%e13%'
                 THEN 1 ELSE 0 END) AS dm_dx,
        -- 2. 高血压(排除肺动脉高压)
        MAX(CASE WHEN (txt LIKE '%hypertension%' AND txt NOT LIKE '%pulmonary hypertension%')
                  OR icd LIKE '401%' OR icd LIKE '402%' OR icd LIKE '403%'
                  OR icd LIKE '404%' OR icd LIKE '405%'
                  OR icd LIKE '%i10%' OR icd LIKE '%i11%' OR icd LIKE '%i12%'
                  OR icd LIKE '%i13%' OR icd LIKE '%i15%'
                 THEN 1 ELSE 0 END) AS hypertension,
        -- 3. 血脂异常
        MAX(CASE WHEN txt LIKE '%hyperlipidemia%' OR txt LIKE '%dyslipidemia%'
                  OR txt LIKE '%hypercholesterolemia%' OR txt LIKE '%hypertriglyceridemia%'
                  OR icd LIKE '272%' OR icd LIKE '%e78%'
                 THEN 1 ELSE 0 END) AS dyslipidemia,
        -- 4. 冠心病/缺血性心脏病(含心绞痛/心梗/ACS)
        MAX(CASE WHEN txt LIKE '%coronary%' OR txt LIKE '%myocardial infarction%'
                  OR txt LIKE '%angina%' OR txt LIKE '%acute coronary%'
                  OR icd LIKE '410%' OR icd LIKE '411%' OR icd LIKE '412%'
                  OR icd LIKE '413%' OR icd LIKE '414%'
                  OR icd LIKE '%i20%' OR icd LIKE '%i21%' OR icd LIKE '%i22%'
                  OR icd LIKE '%i24%' OR icd LIKE '%i25%'
                 THEN 1 ELSE 0 END) AS chd,
        -- 5. 心力衰竭
        MAX(CASE WHEN txt LIKE '%heart failure%' OR txt LIKE '%cardiomyopathy%'
                  OR txt LIKE '%chf%'
                  OR icd LIKE '428%' OR icd LIKE '4254%'
                  OR icd LIKE '%i50%' OR icd LIKE '%i42%'
                 THEN 1 ELSE 0 END) AS hf,
        -- 5b. 房颤/房扑(★ 已并入 CMM "心脏病"成分; ICD-9 427.31/427.32, ICD-10 I48)
        MAX(CASE WHEN txt LIKE '%atrial fibrillation%' OR txt LIKE '%atrial flutter%'
                  OR txt LIKE '%afib%' OR txt LIKE '%a-fib%' OR txt LIKE '%a fib%'
                  OR icd LIKE '42731%' OR icd LIKE '42732%'
                  OR icd LIKE '%i48%'
                 THEN 1 ELSE 0 END) AS af,
        -- 6. 卒中(缺血性+出血性; 不含 TIA, 与 MIMIC 版 430-434/436 对齐)
        MAX(CASE WHEN txt LIKE '%stroke%' OR txt LIKE '%cerebrovascular accident%'
                  OR txt LIKE '%cerebral infarction%' OR txt LIKE '%intracerebral hemorrhage%'
                  OR txt LIKE '%subarachnoid hemorrhage%'
                  OR icd LIKE '430%' OR icd LIKE '431%' OR icd LIKE '432%'
                  OR icd LIKE '433%' OR icd LIKE '434%' OR icd LIKE '436%'
                  OR icd LIKE '%i60%' OR icd LIKE '%i61%' OR icd LIKE '%i62%'
                  OR icd LIKE '%i63%' OR icd LIKE '%i65%' OR icd LIKE '%i66%'
                 THEN 1 ELSE 0 END) AS stroke,
        -- 7. 慢性肾脏病(协变量; 终末期在 severe 单独排除)
        MAX(CASE WHEN txt LIKE '%chronic kidney%' OR txt LIKE '%chronic renal%'
                  OR txt LIKE '%ckd%'
                  OR icd LIKE '585%' OR icd LIKE '%n18%'
                 THEN 1 ELSE 0 END) AS ckd,
        -- 8. 外周动脉疾病
        MAX(CASE WHEN txt LIKE '%peripheral vascular%' OR txt LIKE '%peripheral arterial%'
                  OR txt LIKE '%pvd%'
                  OR icd LIKE '440%' OR icd LIKE '%i70%'
                 THEN 1 ELSE 0 END) AS pad
    FROM dx_text
    GROUP BY patientunitstayid
)

-- Step 2: 排除严重合并症 (终末期肾衰、肝硬化/肝衰竭、恶性肿瘤)
, severe AS (
    SELECT DISTINCT patientunitstayid
    FROM dx_text
    WHERE txt LIKE '%esrd%' OR txt LIKE '%end-stage renal%' OR txt LIKE '%end stage renal%'
       OR txt LIKE '%dialysis dependent%' OR txt LIKE '%hemodialysis dependent%'
       OR txt LIKE '%cirrhosis%' OR txt LIKE '%hepatic failure%'
       OR txt LIKE '%liver failure%' OR txt LIKE '%end-stage liver%'
       OR txt LIKE '%malignan%' OR txt LIKE '%carcinoma%' OR txt LIKE '%metasta%'
       OR txt LIKE '%lymphoma%' OR txt LIKE '%leukemia%' OR txt LIKE '%sarcoma%'
       OR txt LIKE '%melanoma%' OR txt LIKE '% cancer%' OR txt LIKE 'cancer%'
       OR icd LIKE '5855%' OR icd LIKE '5856%' OR icd LIKE '%n185%' OR icd LIKE '%n186%'
       OR icd LIKE '571%' OR icd LIKE '%k70%' OR icd LIKE '%k71%' OR icd LIKE '%k74%'
)

-- Step 3: 患者基础(首次 ICU)+ 人口学 + 身高体重 + 偏移/结局原料
--   age '> 89' → 90; gender 映射为 'F'/'M' 以对齐 R 端 factor(levels=c("F","M"))。
, base_pt AS (
    SELECT
        p.patientunitstayid,
        p.uniquepid,
        p.patienthealthsystemstayid,
        CASE WHEN p.gender = 'Female' THEN 'F'
             WHEN p.gender = 'Male'   THEN 'M' ELSE NULL END AS gender,
        CASE WHEN p.age = '> 89' THEN 90
             WHEN p.age ~ '^[0-9]+$' THEN p.age::int ELSE NULL END AS age,
        p.ethnicity,
        CASE WHEN p.admissionheight BETWEEN 100 AND 250 THEN p.admissionheight ELSE NULL END AS height_cm,
        CASE WHEN p.admissionweight BETWEEN 20  AND 300 THEN p.admissionweight ELSE NULL END AS weight_kg,
        p.unitvisitnumber,
        p.unittype,
        p.hospitalid,
        -- 偏移(分钟): 入院相对入 ICU 为负; 出 ICU / 出院相对入 ICU 为正
        p.hospitaladmitoffset,
        p.unitdischargeoffset,
        p.hospitaldischargeoffset,
        p.hospitaldischargestatus,
        ROUND((p.unitdischargeoffset / 1440.0)::numeric, 3) AS icu_los_days
    FROM eicu_icu.patient p
    WHERE p.unitvisitnumber = 1     -- 本次住院的首次 ICU 停留
)

, first_icu AS (
    SELECT *,
        -- 同一 uniquepid 跨多次住院时取其一(eICU 无跨次绝对时间, 按住院/停留号近似排序)
        ROW_NUMBER() OVER (PARTITION BY uniquepid
                           ORDER BY patienthealthsystemstayid, patientunitstayid) AS icu_order
    FROM base_pt
)

-- Step 4: 生命体征(入 ICU 24h 内均值)
--   vitalPeriodic: 心率/呼吸/SpO2/体温/有创动脉压; vitalAperiodic: 无创血压
, vp AS (
    SELECT patientunitstayid,
        ROUND(AVG(CASE WHEN heartrate          BETWEEN 0  AND 300 THEN heartrate          END)::numeric, 1) AS heart_rate,
        ROUND(AVG(CASE WHEN respiration        BETWEEN 0  AND 100 THEN respiration        END)::numeric, 1) AS resp_rate,
        ROUND(AVG(CASE WHEN sao2               BETWEEN 0  AND 100 THEN sao2               END)::numeric, 1) AS spo2,
        ROUND(AVG(CASE WHEN temperature BETWEEN 25 AND 45   THEN temperature
                       WHEN temperature BETWEEN 90 AND 110  THEN (temperature - 32) / 1.8
                  END)::numeric, 2) AS temperature,
        ROUND(AVG(CASE WHEN systemicsystolic   BETWEEN 0  AND 300 THEN systemicsystolic   END)::numeric, 1) AS sbp_ibp,
        ROUND(AVG(CASE WHEN systemicdiastolic  BETWEEN 0  AND 200 THEN systemicdiastolic  END)::numeric, 1) AS dbp_ibp,
        ROUND(AVG(CASE WHEN systemicmean       BETWEEN 0  AND 250 THEN systemicmean       END)::numeric, 1) AS map_ibp
    FROM eicu_icu.vitalperiodic
    WHERE observationoffset BETWEEN 0 AND 1440
    GROUP BY patientunitstayid
)

, va AS (
    SELECT patientunitstayid,
        ROUND(AVG(CASE WHEN noninvasivesystolic  BETWEEN 0 AND 300 THEN noninvasivesystolic  END)::numeric, 1) AS sbp_nibp,
        ROUND(AVG(CASE WHEN noninvasivediastolic BETWEEN 0 AND 200 THEN noninvasivediastolic END)::numeric, 1) AS dbp_nibp,
        ROUND(AVG(CASE WHEN noninvasivemean      BETWEEN 0 AND 250 THEN noninvasivemean      END)::numeric, 1) AS map_nibp
    FROM eicu_icu.vitalaperiodic
    WHERE observationoffset BETWEEN 0 AND 1440
    GROUP BY patientunitstayid
)

-- Step 4b: 体温补充来源 —— nurseCharting(eICU 体温主要落点)
--   vitalPeriodic.temperature 极稀疏(患者级缺失~92%); 体温实际多记于 nurseCharting,
--   且按医院分别以 'Temperature (C)' / 'Temperature (F)' 两种单位的通用文本列存储。
--   nursingchartvalue 为 varchar, 需安全转数字(正则过滤非数值); F 统一换算为 C。
, nc_temp AS (
    SELECT patientunitstayid,
        ROUND(AVG(
            CASE
              WHEN nursingchartcelltypevalname = 'Temperature (C)'
                   AND nursingchartvalue ~ '^-?[0-9]+(\.[0-9]+)?$'
                   AND nursingchartvalue::numeric BETWEEN 25 AND 45
                THEN nursingchartvalue::numeric
              WHEN nursingchartcelltypevalname = 'Temperature (F)'
                   AND nursingchartvalue ~ '^-?[0-9]+(\.[0-9]+)?$'
                   AND nursingchartvalue::numeric BETWEEN 90 AND 110
                THEN (nursingchartvalue::numeric - 32) / 1.8
            END)::numeric, 2) AS temperature_nc
    FROM eicu_icu.nursecharting
    WHERE nursingchartoffset BETWEEN 0 AND 1440
      AND nursingchartcelltypevalname IN ('Temperature (C)', 'Temperature (F)')
    GROUP BY patientunitstayid
)

-- Step 5: 实验室(★ 已对齐 Wu et al., Cardiovasc Diabetol 2026 的取数口径)
--   原版(已弃用): 窗口=整段住院, 取离入 ICU 最近一条(当初为对齐 MIMIC 版、降低缺失)。
--   现版: 改回该文口径 —— "first measurement upon ICU admission",
--         即入 ICU 首日 [0, 1440] 分钟 内的【首次】测值
--         (该文 Fig.1 流程图要求 TG/FBG/TC/HDL 在 "first day of admission" 非缺失)。
--   收益: (1) 暴露血脂/葡萄糖更接近基线·空腹态, 规避病程后段治疗与濒死过程对指标的反向污染(reverse causation);
--         (2) 生命体征(Step 4, 同为 [0,1440])与化验现共享同一首日窗口 -> 基线快照时间口径一致;
--         (3) 取"首次"值 -> 以 glucose 代 FBG 更合理。
--   代价: 缺失上升、可算 8 指标的 N 下降(该文 Fig.1 亦因缺失 TG/FBG/TC/HDL 排除 1812 例)。
--   ⚠ 例外 HbA1c: 反映近 3 月慢性血糖, 与采血时点无关且首日内极少开单; 故【单列】仍放宽到
--     整段住院取首次值, 仅供糖尿病判定(cmm), 不作首日暴露 -> 避免无谓缺失, 不改变入组口径。
, lab_ranked AS (   -- 急性暴露 + 报告化验: 首日 [0,1440], 取最早(首次)一条
    SELECT
        l.patientunitstayid,
        -- HCO3 归一化为 bicarbonate(同为血清碳酸氢盐, 与 sev_lab 口径一致), 归一化后再分区
        CASE WHEN l.labname = 'HCO3' THEN 'bicarbonate' ELSE l.labname END AS labname,
        l.labresult,
        ROW_NUMBER() OVER (PARTITION BY l.patientunitstayid,
                                        CASE WHEN l.labname = 'HCO3' THEN 'bicarbonate' ELSE l.labname END
                           ORDER BY l.labresultoffset) AS rn          -- ★ 首次 = offset 最小(最早)
    FROM eicu_icu.lab l
    WHERE l.labresult IS NOT NULL
      AND l.labresultoffset BETWEEN 0 AND 1440                        -- ★ 入 ICU 首日 24h (对齐本文)
      AND l.labname IN (
          'triglycerides','HDL','LDL','total cholesterol','glucose',
          'BUN','creatinine','Hgb','platelets x 1000','RBC','WBC x 1000',
          'sodium','potassium','chloride','calcium','bicarbonate','HCO3','lactate',
          'albumin','total bilirubin','PT - INR','PT','PTT')
)

-- HbA1c 单列(慢性指标): 整段住院窗取最早值, 仅供 DM 判定; 详见上方 ⚠ 说明
, hba1c_ranked AS (
    SELECT l.patientunitstayid, l.labresult,
           ROW_NUMBER() OVER (PARTITION BY l.patientunitstayid
                              ORDER BY l.labresultoffset) AS rn
    FROM eicu_icu.lab l
    JOIN eicu_icu.patient p ON l.patientunitstayid = p.patientunitstayid
    WHERE l.labresult IS NOT NULL
      AND l.labname = 'HbA1c'
      AND l.labresultoffset BETWEEN COALESCE(p.hospitaladmitoffset, -1440)
                                AND COALESCE(p.hospitaldischargeoffset, 43200)
)

, labs AS (
    SELECT
        lr.patientunitstayid,
        MAX(CASE WHEN lr.labname='BUN'               AND lr.rn=1 THEN lr.labresult END) AS bun,
        MAX(CASE WHEN lr.labname='creatinine'        AND lr.rn=1 THEN lr.labresult END) AS creatinine,
        MAX(CASE WHEN lr.labname='Hgb'               AND lr.rn=1 THEN lr.labresult END) AS hemoglobin,
        MAX(CASE WHEN lr.labname='platelets x 1000'  AND lr.rn=1 THEN lr.labresult END) AS platelet,
        MAX(CASE WHEN lr.labname='RBC'               AND lr.rn=1 THEN lr.labresult END) AS rbc,
        MAX(CASE WHEN lr.labname='WBC x 1000'        AND lr.rn=1 THEN lr.labresult END) AS wbc,
        MAX(CASE WHEN lr.labname='sodium'            AND lr.rn=1 THEN lr.labresult END) AS sodium,
        MAX(CASE WHEN lr.labname='chloride'          AND lr.rn=1 THEN lr.labresult END) AS chloride,
        MAX(CASE WHEN lr.labname='calcium'           AND lr.rn=1 THEN lr.labresult END) AS calcium,
        MAX(CASE WHEN lr.labname='potassium'         AND lr.rn=1 THEN lr.labresult END) AS potassium,
        MAX(CASE WHEN lr.labname='glucose'           AND lr.rn=1 THEN lr.labresult END) AS glucose,
        -- ★ HbA1c 来自整段住院窗(hba1c_ranked.rn=1), 慢性指标不受首日限制
        MAX(h.hba1c)                                                              AS hba1c,
        MAX(CASE WHEN lr.labname='total cholesterol' AND lr.rn=1 THEN lr.labresult END) AS cholesterol,
        MAX(CASE WHEN lr.labname='HDL'               AND lr.rn=1 THEN lr.labresult END) AS hdl,
        MAX(CASE WHEN lr.labname='LDL'               AND lr.rn=1 THEN lr.labresult END) AS ldl,
        MAX(CASE WHEN lr.labname='triglycerides'     AND lr.rn=1 THEN lr.labresult END) AS triglycerides,
        MAX(CASE WHEN lr.labname='PT - INR'          AND lr.rn=1 THEN lr.labresult END) AS inr,
        MAX(CASE WHEN lr.labname='PT'                AND lr.rn=1 THEN lr.labresult END) AS pt,
        MAX(CASE WHEN lr.labname='PTT'               AND lr.rn=1 THEN lr.labresult END) AS ptt,
        MAX(CASE WHEN lr.labname='lactate'           AND lr.rn=1 THEN lr.labresult END) AS lactate,
        MAX(CASE WHEN lr.labname='bicarbonate'       AND lr.rn=1 THEN lr.labresult END) AS bicarbonate,
        MAX(CASE WHEN lr.labname='total bilirubin'   AND lr.rn=1 THEN lr.labresult END) AS bilirubin,
        MAX(CASE WHEN lr.labname='albumin'           AND lr.rn=1 THEN lr.labresult END) AS albumin
    FROM lab_ranked lr
    LEFT JOIN (SELECT patientunitstayid, labresult AS hba1c
               FROM hba1c_ranked WHERE rn = 1) h
           ON lr.patientunitstayid = h.patientunitstayid
    GROUP BY lr.patientunitstayid
)

-- Step 6 (原 APACHE 评分已移除): 严重度改用 SOFA / SAPS-II / OASIS 三评分,
--   在 Step 11 按 mimic-code 官方口径计算; apachepatientresult 不再使用。

-- APACHE APS 变量表的呼吸机/插管标记(用于补充 ventilation)
, aps_extra AS (
    SELECT patientunitstayid,
        MAX(CASE WHEN vent = 1 THEN 1 ELSE 0 END)      AS apsvar_vent,
        MAX(CASE WHEN intubated = 1 THEN 1 ELSE 0 END) AS apsvar_intub
    FROM eicu_icu.apacheapsvar
    GROUP BY patientunitstayid
)

-- Step 7: 药物(入 ICU 首日; 含入科前带入医嘱)
, meds AS (
    SELECT
        patientunitstayid,
        MAX(CASE WHEN drugname ILIKE '%aspirin%' OR drugname ILIKE '%acetylsalicylic%'
                 THEN 1 ELSE 0 END) AS aspirin,
        MAX(CASE WHEN drugname ILIKE ANY (ARRAY['%statin%','%atorva%','%simva%','%rosuva%',
                                                '%prava%','%lova%','%fluva%','%pitava%'])
                 THEN 1 ELSE 0 END) AS statin,
        MAX(CASE WHEN drugname ILIKE ANY (ARRAY['%metformin%','%glyburide%','%glipizide%',
                                                '%insulin%','%pioglitazone%','%sitagliptin%',
                                                '%glimepiride%'])
                 THEN 1 ELSE 0 END) AS antidiabetic,
        MAX(CASE WHEN drugname ILIKE ANY (ARRAY['%lisinopril%','%enalapril%','%losartan%',
                                                '%valsartan%','%amlodipine%','%metoprolol%',
                                                '%atenolol%','%carvedilol%'])
                 THEN 1 ELSE 0 END) AS antihypertensive,
        MAX(CASE WHEN drugname ILIKE '%heparin%' OR drugname ILIKE '%warfarin%'
                  OR drugname ILIKE '%enoxaparin%' THEN 1 ELSE 0 END) AS anticoagulant
    FROM eicu_icu.medication
    WHERE drugname IS NOT NULL
      AND drugstartoffset <= 1440        -- 首日(含负偏移=入科前带入)
    GROUP BY patientunitstayid
)

-- Step 8: 升压药(infusionDrug, 入 ICU 24h 内)
, vaso AS (
    SELECT DISTINCT patientunitstayid, 1 AS vasopressor
    FROM eicu_icu.infusiondrug
    WHERE drugname ILIKE ANY (ARRAY['%norepinephrine%','%levophed%','%epinephrine%',
                                    '%dopamine%','%dobutamine%','%vasopressin%',
                                    '%phenylephrine%','%neo-synephrine%'])
      AND infusionoffset BETWEEN -60 AND 1440
)

-- Step 9: 机械通气(treatment 字符串, 入 ICU 24h 内)
, vent_tx AS (
    SELECT DISTINCT patientunitstayid, 1 AS vent_treat
    FROM eicu_icu.treatment
    WHERE (treatmentstring ILIKE '%mechanical ventilation%'
           OR treatmentstring ILIKE '%ventilator%'
           OR treatmentstring ILIKE '%intubation%')
      AND treatmentoffset BETWEEN -60 AND 1440
)

-- Step 10: 三成分 CMM 计数 (糖尿病 = 诊断 OR 降糖药 OR HbA1c≥6.5%)
, cmm AS (
    SELECT
        fi.patientunitstayid,
        COALESCE(c.hypertension, 0) AS hypertension,
        COALESCE(c.dyslipidemia, 0) AS dyslipidemia,
        COALESCE(c.chd, 0)          AS chd,
        COALESCE(c.hf, 0)           AS hf,
        COALESCE(c.ckd, 0)          AS ckd,
        COALESCE(c.pad, 0)          AS pad,
        COALESCE(c.af, 0)           AS af,
        COALESCE(c.stroke, 0)       AS stroke,
        -- 糖尿病(综合)
        GREATEST(COALESCE(c.dm_dx, 0),
                 COALESCE(m.antidiabetic, 0),
                 CASE WHEN lb.hba1c >= 6.5 THEN 1 ELSE 0 END) AS dm,
        -- 心脏病 = CHD 或 CHF 或 房颤(AF) ★ AF 已并入心脏病成分
        GREATEST(COALESCE(c.chd, 0), COALESCE(c.hf, 0), COALESCE(c.af, 0)) AS heart_disease,
        -- ★ CMM 三成分计数: 糖尿病 + 心脏病(CHD/CHF/AF) + 中风
        (GREATEST(COALESCE(c.dm_dx, 0),
                  COALESCE(m.antidiabetic, 0),
                  CASE WHEN lb.hba1c >= 6.5 THEN 1 ELSE 0 END)
         + GREATEST(COALESCE(c.chd, 0), COALESCE(c.hf, 0), COALESCE(c.af, 0))
         + COALESCE(c.stroke, 0)) AS cm_count
    FROM first_icu fi
    LEFT JOIN comorbid c ON fi.patientunitstayid = c.patientunitstayid
    LEFT JOIN meds     m ON fi.patientunitstayid = m.patientunitstayid
    LEFT JOIN labs    lb ON fi.patientunitstayid = lb.patientunitstayid
)

-- ============================================================
-- Step 11: 严重度三评分 SOFA / SAPS-II / OASIS
--   口径 = mimic-code 官方实现 (mimic-iv oasis.sql + sapsii.sql 精确分档),
--   数据源换成 eICU; SAPS-II/OASIS 严格按官方"首日 min/max"重算, 不用 APACHE 单值替代。
--   时间窗 = 入 ICU 后 0-1440 分钟(首日)。★末尾列出需校验的近似点。
-- ============================================================

-- 首日生命体征 min/max (有创 vitalperiodic + 无创 vitalaperiodic 合并)
, sev_vital_raw AS (
    SELECT patientunitstayid,
           NULLIF(heartrate,-1) AS hr, NULLIF(respiration,-1) AS rr,
           CASE WHEN temperature BETWEEN 25 AND 46 THEN temperature END AS temp,
           NULLIF(systemicmean,-1) AS map, NULLIF(systemicsystolic,-1) AS sbp
    FROM eicu_icu.vitalperiodic
    WHERE observationoffset BETWEEN 0 AND 1440
    UNION ALL
    SELECT patientunitstayid, NULL, NULL, NULL,
           NULLIF(noninvasivemean,-1) AS map, NULLIF(noninvasivesystolic,-1) AS sbp
    FROM eicu_icu.vitalaperiodic
    WHERE observationoffset BETWEEN 0 AND 1440
)
, sev_vital AS (
    SELECT patientunitstayid,
        MIN(hr) AS hr_min, MAX(hr) AS hr_max, MIN(rr) AS rr_min, MAX(rr) AS rr_max,
        MIN(temp) AS temp_min, MAX(temp) AS temp_max,
        MIN(map) AS map_min, MAX(map) AS map_max, MIN(sbp) AS sbp_min, MAX(sbp) AS sbp_max
    FROM sev_vital_raw GROUP BY patientunitstayid
)
-- 首日实验室 min/max (labname 取自官方 pivoted-lab 标准名)
, sev_lab AS (
    SELECT patientunitstayid,
        MIN(CASE WHEN labname='BUN' THEN labresult END) AS bun_min,
        MAX(CASE WHEN labname='BUN' THEN labresult END) AS bun_max,
        MIN(CASE WHEN labname='WBC x 1000' THEN labresult END) AS wbc_min,
        MAX(CASE WHEN labname='WBC x 1000' THEN labresult END) AS wbc_max,
        MIN(CASE WHEN labname='potassium' THEN labresult END) AS k_min,
        MAX(CASE WHEN labname='potassium' THEN labresult END) AS k_max,
        MIN(CASE WHEN labname='sodium' THEN labresult END) AS na_min,
        MAX(CASE WHEN labname='sodium' THEN labresult END) AS na_max,
        MIN(CASE WHEN labname IN ('bicarbonate','HCO3') THEN labresult END) AS hco3_min,
        MAX(CASE WHEN labname IN ('bicarbonate','HCO3') THEN labresult END) AS hco3_max,
        MIN(CASE WHEN labname='total bilirubin' THEN labresult END) AS bili_min,
        MAX(CASE WHEN labname='total bilirubin' THEN labresult END) AS bili_max,
        MAX(CASE WHEN labname='creatinine' THEN labresult END) AS creat_max,
        MIN(CASE WHEN labname='platelets x 1000' THEN labresult END) AS plt_min
    FROM eicu_icu.lab
    WHERE labresultoffset BETWEEN 0 AND 1440
    GROUP BY patientunitstayid
)
-- 首日尿量 (24h 总和)
, sev_uo AS (
    SELECT patientunitstayid, SUM(cellvaluenumeric) AS urineoutput
    FROM eicu_icu.intakeoutput
    WHERE intakeoutputoffset BETWEEN 0 AND 1440
      AND (LOWER(celllabel) LIKE '%urine%' OR LOWER(cellpath) LIKE '%urine%')
    GROUP BY patientunitstayid
)
-- APACHE 物理量: GCS(合成) / PaO2:FiO2(取最差=最低) / 通气  (eICU 无干净逐时 GCS 与配对血气)
, sev_aps AS (
    SELECT patientunitstayid,
        MAX(CASE WHEN eyes>=1 AND motor>=1 AND verbal>=1 THEN eyes+motor+verbal END) AS gcs,
        MIN(CASE WHEN NULLIF(pao2,-1) IS NOT NULL AND NULLIF(fio2,-1) IS NOT NULL AND fio2>0
                 THEN NULLIF(pao2,-1)/(CASE WHEN fio2>1 THEN fio2/100.0 ELSE fio2 END) END) AS pafi,
        MAX(GREATEST(COALESCE(vent,0),COALESCE(intubated,0))) AS vent
    FROM eicu_icu.apacheapsvar
    GROUP BY patientunitstayid
)
-- 人口学 / 入科前住院时长(分钟)/ 单元类型
, sev_pt AS (
    SELECT patientunitstayid,
        CASE WHEN age LIKE '>%' THEN 90 WHEN age ~ '^[0-9]+$' THEN age::int END AS age,
        GREATEST(-hospitaladmitoffset, 0) AS preiculos,
        LOWER(unittype) AS unittype
    FROM eicu_icu.patient
)
-- 择期手术标志 (OASIS / SAPS-II 入院类型)
, sev_pred AS (
    SELECT patientunitstayid,
        CASE WHEN electivesurgery=1 THEN 1 WHEN electivesurgery=0 THEN 0 ELSE NULL END AS electivesurgery
    FROM eicu_icu.apachepredvar
)
-- SAPS-II 慢病(AIDS/血液恶性肿瘤/转移癌)+ 手术标志  (eICU 编码不可靠, 文本+ICD9 辅助; ★需校验)
, sev_comorb AS (
    SELECT patientunitstayid,
        MAX(CASE WHEN icd9code SIMILAR TO '04[2-4]%'
                  OR LOWER(diagnosisstring) LIKE '%hiv%' OR LOWER(diagnosisstring) LIKE '%aids%'
                 THEN 1 ELSE 0 END) AS aids,
        MAX(CASE WHEN icd9code SIMILAR TO '20[0-8]%' OR icd9code IN ('238.6','273.3')
                  OR LOWER(diagnosisstring) LIKE '%lymphoma%'
                  OR LOWER(diagnosisstring) LIKE '%leukemia%'
                  OR LOWER(diagnosisstring) LIKE '%myeloma%'
                 THEN 1 ELSE 0 END) AS hem,
        MAX(CASE WHEN icd9code SIMILAR TO '19[6-9]%' OR icd9code = '789.51'
                  OR LOWER(diagnosisstring) LIKE '%metasta%'
                 THEN 1 ELSE 0 END) AS mets,
        MAX(CASE WHEN LOWER(diagnosisstring) LIKE '%surg%'
                  OR LOWER(diagnosisstring) LIKE '%post-op%'
                  OR LOWER(diagnosisstring) LIKE '%s/p %'
                 THEN 1 ELSE 0 END) AS surgical
    FROM eicu_icu.diagnosis
    GROUP BY patientunitstayid
)
-- SOFA (首日, 0-24)
, sofa_calc AS (
    SELECT fi.patientunitstayid,
        (CASE WHEN a.pafi IS NULL THEN 0
              WHEN a.pafi < 100 AND a.vent=1 THEN 4
              WHEN a.pafi < 200 AND a.vent=1 THEN 3
              WHEN a.pafi < 300 THEN 2
              WHEN a.pafi < 400 THEN 1 ELSE 0 END)
      + (CASE WHEN l.plt_min IS NULL THEN 0
              WHEN l.plt_min<20 THEN 4 WHEN l.plt_min<50 THEN 3
              WHEN l.plt_min<100 THEN 2 WHEN l.plt_min<150 THEN 1 ELSE 0 END)
      + (CASE WHEN l.bili_max IS NULL THEN 0
              WHEN l.bili_max>=12 THEN 4 WHEN l.bili_max>=6 THEN 3
              WHEN l.bili_max>=2 THEN 2 WHEN l.bili_max>=1.2 THEN 1 ELSE 0 END)
      + (CASE WHEN v.vasopressor=1 THEN 3                       -- ★无剂量分级, 用药即记 3
              WHEN vt.map_min IS NOT NULL AND vt.map_min<70 THEN 1 ELSE 0 END)
      + (CASE WHEN a.gcs IS NULL THEN 0
              WHEN a.gcs<6 THEN 4 WHEN a.gcs<10 THEN 3
              WHEN a.gcs<13 THEN 2 WHEN a.gcs<15 THEN 1 ELSE 0 END)
      + (CASE WHEN l.creat_max IS NULL THEN 0
              WHEN l.creat_max>=5 THEN 4 WHEN l.creat_max>=3.5 THEN 3
              WHEN l.creat_max>=2 THEN 2 WHEN l.creat_max>=1.2 THEN 1 ELSE 0 END)
        AS sofa
    FROM first_icu fi
    LEFT JOIN sev_aps   a  ON fi.patientunitstayid=a.patientunitstayid
    LEFT JOIN sev_lab   l  ON fi.patientunitstayid=l.patientunitstayid
    LEFT JOIN sev_vital vt ON fi.patientunitstayid=vt.patientunitstayid
    LEFT JOIN vaso      v  ON fi.patientunitstayid=v.patientunitstayid
)
-- OASIS (mimic-iv oasis.sql 精确分档)
, oasis_calc AS (
    SELECT fi.patientunitstayid,
        (CASE WHEN pt.preiculos IS NULL THEN 0
              WHEN pt.preiculos < 10.2 THEN 5 WHEN pt.preiculos < 297 THEN 3
              WHEN pt.preiculos < 1440 THEN 0 WHEN pt.preiculos < 18708 THEN 2 ELSE 1 END)
      + (CASE WHEN pt.age IS NULL THEN 0
              WHEN pt.age < 24 THEN 0 WHEN pt.age <= 53 THEN 3
              WHEN pt.age <= 77 THEN 6 WHEN pt.age <= 89 THEN 9
              WHEN pt.age >= 90 THEN 7 ELSE 0 END)
      + (CASE WHEN a.gcs IS NULL THEN 0
              WHEN a.gcs <= 7 THEN 10 WHEN a.gcs < 14 THEN 4 WHEN a.gcs = 14 THEN 3 ELSE 0 END)
      + (CASE WHEN v.hr_max IS NULL THEN 0
              WHEN v.hr_max > 125 THEN 6 WHEN v.hr_min < 33 THEN 4
              WHEN v.hr_max BETWEEN 107 AND 125 THEN 3
              WHEN v.hr_max BETWEEN 89 AND 106 THEN 1 ELSE 0 END)
      + (CASE WHEN v.map_min IS NULL THEN 0
              WHEN v.map_min < 20.65 THEN 4 WHEN v.map_min < 51 THEN 3
              WHEN v.map_max > 143.44 THEN 3
              WHEN v.map_min >= 51 AND v.map_min < 61.33 THEN 2 ELSE 0 END)
      + (CASE WHEN v.rr_min IS NULL THEN 0
              WHEN v.rr_min < 6 THEN 10 WHEN v.rr_max > 44 THEN 9
              WHEN v.rr_max > 30 THEN 6 WHEN v.rr_max > 22 THEN 1
              WHEN v.rr_min < 13 THEN 1 ELSE 0 END)
      + (CASE WHEN v.temp_max IS NULL THEN 0
              WHEN v.temp_max > 39.88 THEN 6
              WHEN v.temp_min BETWEEN 33.22 AND 35.93 THEN 4
              WHEN v.temp_max BETWEEN 33.22 AND 35.93 THEN 4
              WHEN v.temp_min < 33.22 THEN 3
              WHEN v.temp_min > 35.93 AND v.temp_min <= 36.39 THEN 2
              WHEN v.temp_max BETWEEN 36.89 AND 39.88 THEN 2 ELSE 0 END)
      + (CASE WHEN uo.urineoutput IS NULL THEN 0
              WHEN uo.urineoutput < 671.09 THEN 10 WHEN uo.urineoutput > 6896.80 THEN 8
              WHEN uo.urineoutput BETWEEN 671.09 AND 1426.99 THEN 5
              WHEN uo.urineoutput BETWEEN 1427.00 AND 2544.14 THEN 1 ELSE 0 END)
      + (CASE WHEN a.vent = 1 THEN 9 ELSE 0 END)
      + (CASE WHEN pr.electivesurgery IS NULL THEN 0
              WHEN pr.electivesurgery = 1 THEN 0 ELSE 6 END)
        AS oasis
    FROM first_icu fi
    LEFT JOIN sev_pt    pt ON fi.patientunitstayid=pt.patientunitstayid
    LEFT JOIN sev_aps   a  ON fi.patientunitstayid=a.patientunitstayid
    LEFT JOIN sev_vital v  ON fi.patientunitstayid=v.patientunitstayid
    LEFT JOIN sev_uo    uo ON fi.patientunitstayid=uo.patientunitstayid
    LEFT JOIN sev_pred  pr ON fi.patientunitstayid=pr.patientunitstayid
)
-- SAPS-II (mimic-code sapsii.sql 精确分档)
, saps_calc AS (
    SELECT fi.patientunitstayid,
        (CASE WHEN pt.age IS NULL THEN 0
              WHEN pt.age < 40 THEN 0 WHEN pt.age < 60 THEN 7
              WHEN pt.age < 70 THEN 12 WHEN pt.age < 75 THEN 15
              WHEN pt.age < 80 THEN 16 ELSE 18 END)
      + (CASE WHEN v.hr_max IS NULL THEN 0
              WHEN v.hr_min < 40 THEN 11 WHEN v.hr_max >= 160 THEN 7
              WHEN v.hr_max >= 120 THEN 4 WHEN v.hr_min < 70 THEN 2 ELSE 0 END)
      + (CASE WHEN v.sbp_min IS NULL THEN 0
              WHEN v.sbp_min < 70 THEN 13 WHEN v.sbp_min < 100 THEN 5
              WHEN v.sbp_max >= 200 THEN 2 ELSE 0 END)
      + (CASE WHEN v.temp_max IS NULL THEN 0
              WHEN v.temp_min < 39.0 THEN 0 WHEN v.temp_max >= 39.0 THEN 3 ELSE 0 END)
      + (CASE WHEN (a.vent=1 AND a.pafi IS NOT NULL)
                THEN (CASE WHEN a.pafi < 100 THEN 11 WHEN a.pafi < 200 THEN 9 ELSE 6 END)
                ELSE 0 END)
      + (CASE WHEN uo.urineoutput IS NULL THEN 0
              WHEN uo.urineoutput < 500 THEN 11 WHEN uo.urineoutput < 1000 THEN 4 ELSE 0 END)
      + (CASE WHEN l.bun_max IS NULL THEN 0
              WHEN l.bun_max < 28 THEN 0 WHEN l.bun_max < 84 THEN 6 ELSE 10 END)
      + (CASE WHEN l.wbc_max IS NULL THEN 0
              WHEN l.wbc_min < 1 THEN 12 WHEN l.wbc_max >= 20 THEN 3 ELSE 0 END)
      + (CASE WHEN l.k_max IS NULL THEN 0
              WHEN l.k_min < 3 THEN 3 WHEN l.k_max >= 5 THEN 3 ELSE 0 END)
      + (CASE WHEN l.na_max IS NULL THEN 0
              WHEN l.na_min < 125 THEN 5 WHEN l.na_max >= 145 THEN 1 ELSE 0 END)
      + (CASE WHEN l.hco3_max IS NULL THEN 0
              WHEN l.hco3_min < 15 THEN 6 WHEN l.hco3_min < 20 THEN 3 ELSE 0 END)
      + (CASE WHEN l.bili_max IS NULL THEN 0
              WHEN l.bili_max < 4 THEN 0 WHEN l.bili_max < 6 THEN 4 ELSE 9 END)
      + (CASE WHEN a.gcs IS NULL THEN 0
              WHEN a.gcs < 3 THEN 0 WHEN a.gcs < 6 THEN 26 WHEN a.gcs < 9 THEN 13
              WHEN a.gcs < 11 THEN 7 WHEN a.gcs < 14 THEN 5 ELSE 0 END)
      + (CASE WHEN cm.aids=1 THEN 17 WHEN cm.hem=1 THEN 10 WHEN cm.mets=1 THEN 9 ELSE 0 END)
      + (CASE WHEN pr.electivesurgery=1 THEN 0      -- ScheduledSurgical
              WHEN cm.surgical=1 THEN 8             -- UnscheduledSurgical ★近似
              ELSE 6 END)                           -- Medical
        AS saps_ii
    FROM first_icu fi
    LEFT JOIN sev_pt     pt ON fi.patientunitstayid=pt.patientunitstayid
    LEFT JOIN sev_vital  v  ON fi.patientunitstayid=v.patientunitstayid
    LEFT JOIN sev_lab    l  ON fi.patientunitstayid=l.patientunitstayid
    LEFT JOIN sev_aps    a  ON fi.patientunitstayid=a.patientunitstayid
    LEFT JOIN sev_uo     uo ON fi.patientunitstayid=uo.patientunitstayid
    LEFT JOIN sev_comorb cm ON fi.patientunitstayid=cm.patientunitstayid
    LEFT JOIN sev_pred   pr ON fi.patientunitstayid=pr.patientunitstayid
)

-- ============ 最终汇总 + 指标计算 ============
SELECT
    -- ID(对齐 MIMIC 版列名, 便于 R 端通用)
    fi.uniquepid                 AS subject_id,
    fi.patienthealthsystemstayid AS hadm_id,
    fi.patientunitstayid         AS stay_id,
    fi.icu_los_days,
    fi.icu_order,
    -- 人口统计 (ethnicity 为原始字段; R 端重编码为 race=White/Black/Other 以对齐 MIMIC)
    fi.age, fi.gender, fi.ethnicity,
    fi.height_cm, fi.weight_kg,
    CASE WHEN fi.height_cm > 0 AND fi.weight_kg > 0
         THEN ROUND((fi.weight_kg / ((fi.height_cm / 100.0) ^ 2))::numeric, 2)
         ELSE NULL END AS bmi,
    -- 生命体征(无创血压优先, 缺则有创)
    COALESCE(va.map_nibp, vp.map_ibp)  AS map,
    vp.heart_rate,
    vp.resp_rate,
    COALESCE(va.sbp_nibp, vp.sbp_ibp)  AS sbp,
    COALESCE(va.dbp_nibp, vp.dbp_ibp)  AS dbp,
    COALESCE(nct.temperature_nc, vp.temperature) AS temperature,
    vp.spo2,
    -- 实验室
    l.bun, l.creatinine, l.hemoglobin, l.platelet, l.rbc, l.wbc,
    l.sodium, l.chloride, l.calcium, l.potassium, l.glucose,
    l.hba1c, l.cholesterol, l.hdl, l.ldl, l.triglycerides,
    l.inr, l.pt, l.ptt, l.lactate, l.bicarbonate,
    l.bilirubin, l.albumin,

    -- ★★★ 八大代谢指标(口径同 MIMIC 版) ★★★
    -- TyG = ln(TG × Glucose / 2)
    CASE WHEN l.triglycerides > 0 AND l.glucose > 0
         THEN ROUND(LN(l.triglycerides * l.glucose / 2.0)::numeric, 4)
         ELSE NULL END AS tyg,
    -- TyG-BMI = TyG × BMI
    CASE WHEN l.triglycerides > 0 AND l.glucose > 0 AND fi.height_cm > 0 AND fi.weight_kg > 0
         THEN ROUND((LN(l.triglycerides * l.glucose / 2.0) *
               (fi.weight_kg / ((fi.height_cm / 100.0) ^ 2)))::numeric, 2)
         ELSE NULL END AS tyg_bmi,
    -- METS-IR = ln(2×Glucose + TG) × BMI / ln(HDL)
    CASE WHEN l.glucose > 0 AND l.triglycerides > 0 AND l.hdl > 1
              AND fi.height_cm > 0 AND fi.weight_kg > 0
         THEN ROUND((LN(2.0 * l.glucose + l.triglycerides) *
               (fi.weight_kg / ((fi.height_cm / 100.0) ^ 2)) /
               LN(l.hdl))::numeric, 4)
         ELSE NULL END AS mets_ir,
    -- AIP = Log10(TG / HDL)   [标准 Dobiášová 定义; LOG()=log10]
    --   注: 上游 R 头注释提到"改为 LN", 但此处沿用 SQL 原始 LOG(=log10, 标准 AIP)。
    --       若想与该注释完全一致, 把下面 LOG 换成 LN 即可。
    CASE WHEN l.triglycerides > 0 AND l.hdl > 0
         THEN ROUND(LOG(l.triglycerides / l.hdl)::numeric, 4)
         ELSE NULL END AS aip,
    -- CHG = Ln(TC × FBG / (2 × HDL))   [Mansoori et al., J Diabetes Investig 2025;16(2):309-14 标准定义, HDL 在分母; ICU 以入科首次 glucose 代 FBG]
    CASE WHEN l.cholesterol > 0 AND l.glucose > 0 AND l.hdl > 0
         THEN ROUND(LN(l.cholesterol * l.glucose / (2.0 * l.hdl))::numeric, 4)
         ELSE NULL END AS chg,
    -- TG-HDL = TG / HDL
    CASE WHEN l.triglycerides > 0 AND l.hdl > 0
         THEN ROUND((l.triglycerides / l.hdl)::numeric, 4)
         ELSE NULL END AS tg_hdl,
    -- TyG-RC = TyG × RC, RC = TC - HDL - LDL (mg/dL); 须 RC > 0
    CASE WHEN l.triglycerides > 0 AND l.glucose > 0
              AND l.cholesterol > 0 AND l.hdl > 0 AND l.ldl > 0
              AND (l.cholesterol - l.hdl - l.ldl) > 0
         THEN ROUND((LN(l.triglycerides * l.glucose / 2.0)
                     * (l.cholesterol - l.hdl - l.ldl))::numeric, 4)
         ELSE NULL END AS tyg_rc,
    -- SPISE = 600 × HDL^0.185 / (TG^0.2 × BMI^1.338)
    CASE WHEN l.hdl > 0 AND l.triglycerides > 0
              AND fi.height_cm > 0 AND fi.weight_kg > 0
         THEN ROUND((600.0 * POWER(l.hdl, 0.185)
                     / (POWER(l.triglycerides, 0.2)
                        * POWER(fi.weight_kg / ((fi.height_cm / 100.0) ^ 2), 1.338)))::numeric, 4)
         ELSE NULL END AS spise,

    -- 疾病严重度(三评分; 口径=mimic-code 移植, 见 Step 11)
    sf.sofa,
    sp.saps_ii,
    oa.oasis,
    -- CMM 组分(心脏病 = CHD 或 CHF 或 AF; af 已并入心脏病, 计入 cm_count)
    --   ★ cm_count / heart_disease 仅供入组(CMM≥2)与定义逻辑使用, R 端不纳入 Table 1 报告(对齐 MIMIC)
    cmm.dm, cmm.hypertension, cmm.dyslipidemia, cmm.chd, cmm.hf,
    cmm.heart_disease, cmm.stroke, cmm.ckd, cmm.pad, cmm.af, cmm.cm_count,
    -- 干预
    GREATEST(COALESCE(vt.vent_treat, 0),
             COALESCE(ax.apsvar_vent, 0),
             COALESCE(ax.apsvar_intub, 0)) AS ventilation,
    COALESCE(vs.vasopressor, 0)      AS vasopressor,
    COALESCE(m.aspirin, 0)           AS aspirin,
    COALESCE(m.statin, 0)            AS statin,
    COALESCE(m.antidiabetic, 0)      AS antidiabetic,
    COALESCE(m.antihypertensive, 0)  AS antihypertensive,
    COALESCE(m.anticoagulant, 0)     AS anticoagulant,

    -- ★ 结局: 院内死亡 (唯一终点) ★
    CASE WHEN fi.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS inhosp_outcome,
    GREATEST(ROUND((fi.hospitaldischargeoffset / 1440.0)::numeric, 4), 0.01) AS surv_time_inhosp,

    -- 排除标记
    CASE WHEN se.patientunitstayid IS NOT NULL THEN 1 ELSE 0 END AS severe_comorbid

FROM first_icu fi
INNER JOIN cmm        ON fi.patientunitstayid = cmm.patientunitstayid
LEFT  JOIN severe se  ON fi.patientunitstayid = se.patientunitstayid
LEFT  JOIN vp         ON fi.patientunitstayid = vp.patientunitstayid
LEFT  JOIN va         ON fi.patientunitstayid = va.patientunitstayid
LEFT  JOIN nc_temp nct ON fi.patientunitstayid = nct.patientunitstayid
LEFT  JOIN labs l     ON fi.patientunitstayid = l.patientunitstayid
LEFT  JOIN sofa_calc  sf ON fi.patientunitstayid = sf.patientunitstayid
LEFT  JOIN saps_calc  sp ON fi.patientunitstayid = sp.patientunitstayid
LEFT  JOIN oasis_calc oa ON fi.patientunitstayid = oa.patientunitstayid
LEFT  JOIN aps_extra ax ON fi.patientunitstayid = ax.patientunitstayid
LEFT  JOIN meds m     ON fi.patientunitstayid = m.patientunitstayid
LEFT  JOIN vaso vs    ON fi.patientunitstayid = vs.patientunitstayid
LEFT  JOIN vent_tx vt ON fi.patientunitstayid = vt.patientunitstayid

WHERE cmm.cm_count >= 2;   -- CMM: 三成分(糖尿病/心脏病[CHD或CHF或AF]/中风)中 ≥2 项
