######################################################################
# CMM 研究: 8 个胰岛素抵抗(IR)代用指标与重症CMM患者死亡风险
# 复刻 Wu B. et al. "Association of various insulin resistance surrogate
#   markers with mortality risk in critically ill patients with ischemic
#   stroke" (eICU-CRD, Cardiovasc Diabetol 2026) 的处理流程, 落到 MIMIC-IV CMM 队列。
#
# 8 指标 (顺序对齐本文 Fig A-H): TyG, SPISE, TG-HDL, METS-IR, AIP, TyG-BMI, TyG-RC, CHG
# 关键口径 (与本文一致):
#   - AIP = Log10(TG/HDL)         常用对数 log10 (SQL 用 PostgreSQL LOG()=log10; Dobiášová 标准定义)
#   - CHG = Ln(TC×FBG/(2×HDL))    HDL 在分母 (Mansoori 2025 标准定义; 审稿意见1后纠正, SQL 已同步)
#   - SPISE 为敏感性指标, 方向相反 (HR<1 为危险)
# 流程要点 (本次改动):
#   - per-SD 为主 + per-1-unit 为辅 (本文用 per-unit; 跨指标不可比, 故 per-SD 主报)
#   - 三个模型(输出标签): Crude 未校正; Model 1 = age+sex+BMI(本文 Adjusted I);
#     Model 2 = ★Step 3b【LASSO + VIF<5】筛选结果(Adjusted II): 候选池 = 各严重度评分(SOFA/SAPS II/OASIS)
#       + 组成各评分的变量 + 其余临床协变量; 强制人口学(age/sex/BMI)不参与惩罚、始终保留。
#       (Step 3c 固定手动清单作为可选覆盖, 默认 USE_MANUAL_COVARS_M3=FALSE 不启用)
#   - ★ 协变量筛选 (Step 3b): ★本次改为 LASSO(glmnet family=cox, alpha=1) -> VIF<5 去共线; 已【不再】做 单因素 Cox / 逐步 Cox。
#     ★ LASSO 前对连续变量标准化(均值0/标准差1, 哑变量不标准化); 强制人口学(age/sex/BMI; race 不纳入任何模型)不罚不剔。
#     候选池剔除暴露成分(TG/Glu/HDL/TC/LDL)及身高/体重, 防对 8 暴露过度校正(设 DROP_EXPOSURE_COMPONENTS=FALSE 可保留);
#     ★严重度评分整体纳入候选池(INCLUDE_SEVERITY_SCORES=TRUE)+其组成变量(DROP_SEVERITY_COMPONENTS=FALSE), 经 LASSO->VIF<5 取舍
#   - 主要终点 = 院内死亡 (Table 1 按院内死亡 Survivor/Non-survivor 分层); 30/90/365 天作次要
#     (注: 以院内死亡为主要终点, 与本文原始口径一致; 其余终点保留作敏感性分析)
#   - 缺失: 暴露相关实验室作纳入标准要求非缺; 其余协变量用 mice PMM 多重插补 (取首个完成集)
#   - PH 违反者补 时依 Cox (tt: 与 log(t) 交互) 作敏感性分析
#   - RCS 用 3 节点, 以中位数为参考
#   - ROC: 单指标(univariate) AUC + Youden 最佳切点; 指标间 DeLong 两两比较
# MIMIC-IV v2.2
######################################################################

rm(list = ls())
options(stringsAsFactors = FALSE, scipen = 999)

if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse, DBI, odbc,
  tableone, flextable,
  survival, survminer, rms,
  ggplot2, patchwork, forestploter,
  mice, nricens, gtsummary, pROC,
  xgboost, shapviz, glmnet
)

set.seed(20260531)   # 复现: 多重插补/随机过程

DIR_LOAD <- "./load"; DIR_CSV <- "./csv"; DIR_DOCX <- "./docx"; DIR_SQL <- "./sql"
for (d in c(DIR_LOAD, DIR_CSV, DIR_DOCX)) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

# ====== 八大暴露指标 (顺序对齐本文 Fig A-H) ======
EXPOSURES       <- c("tyg", "spise", "tg_hdl", "mets_ir", "aip", "tyg_bmi", "tyg_rc", "chg")
EXPOSURE_LABELS <- c("TyG", "SPISE", "TG-HDL", "METS-IR", "AIP", "TyG-BMI", "TyG-RC", "CHG")
# 注: SPISE 越高=胰岛素越敏感, 其 HR 预期 < 1, 方向与其余 7 个相反。

# ★ 主要 + 次要时间终点
TIMEPOINTS  <- c(30, 90, 365)            # 天数终点, 均作次要/敏感性分析
PRIMARY_KEY <- "inhosp"                  # ← 主要终点 = 院内死亡; 在 ENDPOINTS 中的 key, 供 Table1 分层 / ROC 主终点引用
PRIMARY_LBL <- "In-hospital"
# ★ 主要终点 = 院内死亡; Table 1 按院内死亡(Survivor/Non-survivor)分层。
#   注: 与本文原始口径一致(以院内死亡为主); 30/90/365 天保留为次要(敏感性)分析。

# 统一终点表: 院内死亡 + 3 个天数终点, 驱动 KM/Cox/RCS/亚组/NRI-IDI/ROC
#   - 院内死亡(inhosp): outcome = hospital_expire_flag, 生存时间删失于出院 (非固定窗口)
#   - 天数终点: outcome = day{N}_outcome, 生存时间右截尾于第 N 天
ENDPOINTS <- c(
  list(list(key = "inhosp", suffix = "inhosp",
            surv = "surv_time_inhosp", out = "inhosp_outcome",
            group = "inhosp_group", label = "In-hospital", brk = 5)),
  lapply(TIMEPOINTS, function(tp) list(
    key    = as.character(tp), suffix = paste0(tp, "d"),
    surv   = paste0("surv_time_", tp), out = paste0("day", tp, "_outcome"),
    group  = paste0("day", tp, "_group"), label = paste0(tp, "-day"),
    brk    = max(round(tp / 6), 5)))
)

# 将主要终点(院内死亡)移到列表首位 -> 所有"按终点循环"的图/表/CSV 均以院内死亡为首(主), 其余为次
.prim_idx <- which(vapply(ENDPOINTS, function(e) e$key, character(1)) == PRIMARY_KEY)
stopifnot(length(.prim_idx) == 1)
ENDPOINTS <- c(ENDPOINTS[.prim_idx], ENDPOINTS[-.prim_idx])

# ====================================================================
# 01 数据提取
# ====================================================================
cat(">>> Step 1: 数据提取\n")

con <- DBI::dbConnect(odbc::odbc(), dsn = "mimic4", database = "mimic4_v31",
                      uid = "postgres", pwd = "1",
                      server = "localhost", port = 5432)

sql_pop <- readr::read_file(file.path(DIR_SQL, "01-population.sql"))
population <- DBI::dbGetQuery(con, sql_pop)
DBI::dbDisconnect(con)

# R 端队列筛选 —— 仅按"暴露相关实验室"做完全病例 (定义暴露/分位, 不可插补);
#   其余协变量留待 Step 1b 多重插补。
cohort_raw <- population %>%
  filter(
    cm_count >= 2,                  # CMM ≥2项
    age >= 18,                      # 成人
    icu_order == 1,                 # 首次ICU
    icu_los_days >= 1,              # ICU ≥24h
    severe_comorbid == 0,           # 排除严重合并症
    # --- 暴露定义所需实验室/体测, 全部要求非缺 (与本文纳入标准一致) ---
    !is.na(triglycerides), !is.na(glucose), !is.na(hdl),
    !is.na(cholesterol),            # CHG / TyG-RC 需 TC
    !is.na(ldl),                    # TyG-RC 需 LDL (RC = TC-HDL-LDL)
    !is.na(height_cm), !is.na(weight_kg),
    # --- 8 指标均可计算 (含 RC>0 约束体现在 tyg_rc 非缺) ---
    !is.na(tyg), !is.na(spise), !is.na(tg_hdl), !is.na(mets_ir),
    !is.na(aip), !is.na(tyg_bmi), !is.na(tyg_rc), !is.na(chg),
    !is.na(surv_time_365)
  )

# ★ 按需求彻底删除 hba1c 与 aps_iii: 二者自此不进入 cohort_raw, 故不会传导到
#   cohort_imp / df, 也就不再出现在 Table 1、单/多因素 Cox、亚组、RCS、ROC、
#   列线图、SHAP, 以及导出的 cohort_for_ml.csv 与 all_results.Rdata。
#   说明: hba1c 仅在 01-population.sql 内部用于判定 dm / cm_count(已在 SQL 算好),
#         此处删除不影响该判定; aps_iii 在 R 端本就未被任何模型/筛选引用, 仅因
#         Table 1 的"全量纳入"逻辑被动现身, 现一并移除。
cohort_raw <- cohort_raw %>% select(-any_of(c("hba1c", "aps_iii")))

# 按需求彻底删除 insurance: 整个分析不再使用/携带该变量
# (会传导至 extra_cols/cohort_imp/df/Table 1 及 save; 若连数据库都不想取出, 请在 01-population.sql 的 SELECT 里移除 insurance)

# --- 种族 race: 若 SQL 提供则折叠为 White/Black/Other 三类 (对齐本文; 参考级 = White) ---
#   注: 当前 01-population.sql 似未 SELECT race; 若缺失, 本块与下游 race 处理均被 guard 自动跳过。
#       MIMIC-IV 中 race 取自 admissions.race; 原始类别繁多, 这里用宽松正则归并:
#       含 WHITE -> White; 含 BLACK -> Black; 其余(ASIAN/HISPANIC/OTHER/UNKNOWN/NA 等) -> Other。
if ("race" %in% names(cohort_raw)) {
  .r <- toupper(trimws(as.character(cohort_raw$race)))
  race3 <- ifelse(grepl("WHITE", .r), "White",
           ifelse(grepl("BLACK", .r), "Black", "Other"))
  race3[is.na(.r) | .r == ""] <- "Other"
  cohort_raw$race <- factor(race3, levels = c("White", "Black", "Other"))
  cat(sprintf("  race 已折叠为 White/Black/Other: %s\n",
              paste(sprintf("%s=%d", names(table(cohort_raw$race)),
                            as.integer(table(cohort_raw$race))), collapse = ", ")))
} else {
  cat("  [race] cohort_raw 无 race 列 -> 跳过 (如需纳入, 请在 01-population.sql 选出 admissions.race)\n")
}

cat(sprintf("  全人群: %d, 暴露完整队列: %d\n", nrow(population), nrow(cohort_raw)))
save(population, cohort_raw, file = file.path(DIR_LOAD, "raw_data.Rdata"))

# ====================================================================
# 01b 多重插补 (mice PMM) —— 替代原"完全病例删除"
#   思路: 暴露/结局/生存时间/人口学已完整 -> mice 自动不插补 (method="");
#         仅对缺失率可接受(<=20%)的协变量做 PMM; 缺失>20% 的协变量按本文做法剔除。
#   产出: cohort_imp = 取首个完成数据集 (与本文"插补后直接分析"一致, 兼容 strong:: 单 df 接口)。
#   * 如需严格 Rubin 合并: 对 coxph 用 with(mice_obj, ...) + pool();
#     但本文未做合并, 且 KM/RCS/ROC/NRI-IDI 难合并, 故主流程用单完成集。
# ====================================================================
cat(">>> Step 1b: 多重插补 (mice PMM)\n")

# 候选协变量/Table1 变量 (用于缺失评估; 不含 ID/时间)
impute_candidates <- c(
  "age","gender","race","bmi","height_cm","weight_kg",
  "heart_rate","sbp","dbp","map","resp_rate","temperature","spo2",
  "hemoglobin","platelet","rbc","wbc","creatinine","bun","glucose","triglycerides",
  "cholesterol","hdl","ldl","lactate","albumin","bilirubin",
  "sodium","potassium","chloride","calcium","bicarbonate","inr","pt","ptt",
  "sofa","saps_ii","oasis",
  "dm","hypertension","dyslipidemia","chd","hf","stroke","ckd","pad","af",
  "ventilation","vasopressor","aspirin","statin","antidiabetic","antihypertensive",
  EXPOSURES
)
impute_candidates <- intersect(impute_candidates, names(cohort_raw))

# 缺失率评估 + 剔除高缺失(>20%)协变量 (本文剔除 APTT比/CRP/hsCRP/ESR/尿酸 等 >~20%)
miss_rate <- sapply(cohort_raw[impute_candidates], function(x) mean(is.na(x)))
drop_hi   <- names(miss_rate)[miss_rate > 0.20]
drop_hi   <- setdiff(drop_hi, EXPOSURES)   # 暴露不剔除(已完整, 不会触发)
miss_tab  <- data.frame(var = names(miss_rate),
                        miss_pct = round(100 * miss_rate, 1),
                        dropped = names(miss_rate) %in% drop_hi)
write.csv(miss_tab, file.path(DIR_CSV, "missingness_report.csv"), row.names = FALSE)
cat("  --- 缺失率(剔除>20%) ---\n"); print(miss_tab[order(-miss_tab$miss_pct), ], row.names = FALSE)

vars_keep  <- setdiff(impute_candidates, drop_hi)
id_time    <- cohort_raw %>% select(any_of(c("subject_id","hadm_id","stay_id","intime","outtime")))

dat_mice <- cohort_raw[, vars_keep, drop = FALSE]
# 二分类协变量转因子, 利于 mice 选 logreg
bin_vars <- intersect(c("dm","hypertension","dyslipidemia","chd","hf","stroke","ckd","pad","af",
                        "ventilation","vasopressor","aspirin","statin","antidiabetic",
                        "antihypertensive"), names(dat_mice))
dat_mice[bin_vars] <- lapply(dat_mice[bin_vars], factor)
if ("gender" %in% names(dat_mice)) dat_mice$gender <- factor(dat_mice$gender)

# 方法向量: 数值缺失列用 PMM (本文口径); 二分类用默认 logreg; 完整列自动不插补 ("")
ini  <- mice(dat_mice, maxit = 0, printFlag = FALSE)
meth <- ini$method
num_cols <- names(dat_mice)[vapply(dat_mice, is.numeric, logical(1))]
sel  <- num_cols[meth[num_cols] != ""]
meth[sel] <- "pmm"
imp <- mice(dat_mice, m = 5, maxit = 50, method = meth,
            predictorMatrix = ini$predictorMatrix, printFlag = FALSE, seed = 20260531)
dat_done <- mice::complete(imp, 1)
# 还原二分类为 0/1 数值, 与下游 factor(.,0:1) 一致
dat_done[bin_vars] <- lapply(dat_done[bin_vars], function(x) as.integer(as.character(x)))
if ("gender" %in% names(dat_done)) dat_done$gender <- as.character(dat_done$gender)

# 合并: 完成的协变量 + ID/时间 + (未进入 mice 的)结局/生存等列
extra_cols <- setdiff(names(cohort_raw), c(names(dat_done), names(id_time)))
cohort_imp <- bind_cols(id_time, dat_done, cohort_raw[, extra_cols, drop = FALSE])
saveRDS(imp, file.path(DIR_LOAD, "mice_imp.rds"))
# ★审稿意见3: 同时保存 5 个插补完成集 (供 Step 6c/6d Rubin 法则合并)
#   与 eICU 流程一致; 二分类/性别做与 dat_done 相同的还原转换。
imp_data_list <- lapply(seq_len(imp$m), function(i) {
  d <- mice::complete(imp, i)
  d[bin_vars] <- lapply(d[bin_vars], function(x) as.integer(as.character(x)))
  if ("gender" %in% names(d)) d$gender <- as.character(d$gender)
  d
})
saveRDS(imp_data_list, file.path(DIR_LOAD, "mice_completed_list.rds"))
cat(sprintf("  已构建 %d 个插补完成集 (供 Rubin 法则合并)\n", length(imp_data_list)))
cat(sprintf("  插补后队列: %d 行, %d 列\n", nrow(cohort_imp), ncol(cohort_imp)))

# ====================================================================
# 02 流程图
# ====================================================================
cat(">>> Step 2: 流程图\n")

fc <- tribble(
  ~step, ~n,
  "CMM队列(SQL已筛≥2项CMD)",  nrow(population),
  "首次ICU",                  population %>% filter(icu_order == 1) %>% nrow(),
  "成人(≥18岁)",              population %>% filter(icu_order == 1, age >= 18) %>% nrow(),
  "ICU≥24h",                  population %>% filter(icu_order == 1, age >= 18, icu_los_days >= 1) %>% nrow(),
  "排除严重合并症",           population %>% filter(icu_order == 1, age >= 18, icu_los_days >= 1, severe_comorbid == 0) %>% nrow(),
  "暴露实验室完整(TG/Glu/HDL/TC/LDL/身高/体重)+8指标可算", nrow(cohort_raw)
)
fc$excluded <- c(NA, -diff(fc$n))
write.csv(fc, file.path(DIR_CSV, "flowchart.csv"), row.names = FALSE)
print(fc)

# ====================================================================
# 03 变量标签化
# ====================================================================
cat(">>> Step 3: 变量标签化\n")

df <- cohort_imp %>%
  mutate(
    gender          = factor(gender, levels = c("F","M"), labels = c("Female","Male")),
    dm              = factor(dm, 0:1, c("No","Yes")),
    hypertension    = factor(hypertension, 0:1, c("No","Yes")),
    dyslipidemia    = factor(dyslipidemia, 0:1, c("No","Yes")),
    chd             = factor(chd, 0:1, c("No","Yes")),
    hf              = factor(hf, 0:1, c("No","Yes")),
    stroke          = factor(stroke, 0:1, c("No","Yes")),
    ckd             = factor(ckd, 0:1, c("No","Yes")),
    pad             = factor(pad, 0:1, c("No","Yes")),
    af              = factor(af, 0:1, c("No","Yes")),
    ventilation     = factor(ventilation, 0:1, c("No","Yes")),
    vasopressor     = factor(vasopressor, 0:1, c("No","Yes")),
    aspirin         = factor(aspirin, 0:1, c("No","Yes")),
    statin          = factor(statin, 0:1, c("No","Yes")),
    antidiabetic    = factor(antidiabetic, 0:1, c("No","Yes")),
    antihypertensive= factor(antihypertensive, 0:1, c("No","Yes")),
    anticoagulant   = factor(anticoagulant, 0:1, c("No","Yes")),
    # 亚组分层用 (对齐本文 Fig4: 年龄<60, BMI<30)
    age_grp = factor(ifelse(age < 60, "Age<60", "Age>=60"), levels = c("Age<60","Age>=60")),
    bmi_grp = factor(ifelse(bmi < 30, "BMI<30", "BMI>=30"), levels = c("BMI<30","BMI>=30"))
  )

# 暴露指标四分位 + 标准化 (按 EXPOSURES 循环, 8 个)
for (exp_var in EXPOSURES) {
  df[[paste0(exp_var, "_q")]] <- cut(
    df[[exp_var]], breaks = quantile(df[[exp_var]], 0:4/4, na.rm = TRUE),
    include.lowest = TRUE, labels = paste0("Q", 1:4))
  df[[paste0(exp_var, "_std")]] <- as.numeric(scale(df[[exp_var]]))
}

# 各终点 生存/非生存 分组因子
for (ep in ENDPOINTS) df[[ep$group]] <- factor(df[[ep$out]], 0:1, c("Survivor","Non-survivor"))

# Table 1 变量 —— 自动纳入 df 全部变量, 仅排除 ID/时间/派生列/终点列
#   排除集: 主键ID、ICU时间戳、暴露派生列(_q 四分位 / _std 标准化)、
#           各终点的生存时间/结局/分组列(否则与分层变量自身重复、且互为决定)。
.exposure_derived <- c(paste0(EXPOSURES, "_q"), paste0(EXPOSURES, "_std"))
.endpoint_cols    <- unlist(lapply(ENDPOINTS, function(e) c(e$surv, e$out, e$group)))
tbl1_exclude <- unique(c(
  "subject_id","hadm_id","stay_id","intime","outtime",   # ID / 时间戳
  "icu_order","icu_los_days","severe_comorbid",          # 入排辅助字段(severe_comorbid 纳入后恒为0/零方差)
  "height_cm","weight_kg",                               # 体测原始值(已并入 BMI, 不单列)
  "age_grp","bmi_grp",                                   # 年龄/BMI 分组(亚组分析用, 不进基线表)
  "cm_count","ethnicity","heart_disease",                # CMM计数; ethnicity(与折叠后 race 重复, 类别过多); heart_disease(=CHD|CHF|AF 复合, 近似恒定, 成分已单列)
  .exposure_derived, .endpoint_cols,                     # 派生列 / 终点列
  "hospital_expire_flag","surv_time_inhosp"              # 院内死亡原始列(冗余保险)
))
# 全量纳入: df 中除排除集外的所有列
tbl1_vars <- setdiff(names(df), tbl1_exclude)

# 顺序: 本文偏好列在前, 其余(新增/未列出)按 df 原序追加在后
.tbl1_pref <- c("age","gender","race","bmi",
                "heart_rate","sbp","dbp","map","resp_rate","temperature","spo2",
                "dm","hypertension","dyslipidemia","chd","hf","stroke","ckd","pad","af",
                "hemoglobin","platelet","rbc","wbc","creatinine","bun","glucose","triglycerides",
                "cholesterol","hdl","ldl","lactate","albumin","bilirubin",
                "sodium","potassium","chloride","calcium","bicarbonate","inr","pt","ptt",
                "tyg","spise","tg_hdl","mets_ir","aip","tyg_bmi","tyg_rc","chg",
                "sofa","saps_ii","oasis",
                "ventilation","vasopressor","aspirin","statin","antidiabetic","antihypertensive","anticoagulant")
tbl1_vars <- c(intersect(.tbl1_pref, tbl1_vars),
               setdiff(tbl1_vars, .tbl1_pref))
cat(sprintf("  Table 1 纳入 %d 个变量(全量, 已排除 ID/时间/派生/终点列)\n", length(tbl1_vars)))

# 兜底标签化: 上面 mutate 是逐个手列的, 易漏掉上游新增的 0/1 列(如曾漏标的 anticoagulant)。
# 这里把基线表中"仍是数值且取值⊆{0,1}"的列统一转成 No/Yes 因子, 防止再有遗漏(已是因子的不受影响)。
.is_bin01 <- function(x) {
  if (!is.numeric(x)) return(FALSE)
  u <- unique(x[!is.na(x)]); length(u) <= 2 && all(u %in% c(0, 1))
}
.bin_to_label <- intersect(tbl1_vars, names(df)[vapply(df, .is_bin01, logical(1))])
for (.v in .bin_to_label) df[[.v]] <- factor(df[[.v]], levels = c(0, 1), labels = c("No", "Yes"))
if (length(.bin_to_label))
  cat(sprintf("  [标签化] 兜底将 0/1 列转 No/Yes: %s\n", paste(.bin_to_label, collapse = ", ")))

# 分类变量: 自动识别(因子列, 或仅含 {0,1}/{两类} 的列), 覆盖全量纳入后新增的二分类
.is_cat <- function(x) {
  if (is.factor(x) || is.character(x) || is.logical(x)) return(TRUE)
  u <- unique(x[!is.na(x)])
  length(u) <= 2 && all(u %in% c(0, 1))   # 0/1 哑变量
}
cat_vars <- tbl1_vars[vapply(df[tbl1_vars], .is_cat, logical(1))]

# 去除零方差连续变量: method="auto" 会对连续列跑 shapiro.test, 单一取值会报
# "所有'x'的值都是一样的"。分类列不受影响(卡方/Fisher), 故仅在连续列中剔除常量列。
.const_continuous <- setdiff(tbl1_vars, cat_vars)
.const_continuous <- .const_continuous[vapply(
  df[.const_continuous],
  function(x) length(unique(x[!is.na(x)])) < 2, logical(1))]
if (length(.const_continuous)) {
  cat(sprintf("  [Table 1] 剔除零方差连续变量: %s\n",
              paste(.const_continuous, collapse = ", ")))
  tbl1_vars <- setdiff(tbl1_vars, .const_continuous)
}

# ====== 调整模型: Crude(未校正) / Model 1(age+sex+BMI) / Model 2(变量筛选结果); 标签对齐本文 Adjusted I/II ======
covars_m2 <- c("age","gender","bmi")                         # → 输出 "Model 1"(Adjusted I): age/sex/BMI
# Model 2(Adjusted II) 不再使用任何预设协变量, 一律由下方 Step 3b 的 ★LASSO + VIF<5 筛选产生;
# covars_m3 在 Step 3b 末尾统一赋值(此处不预先定义)。
# 注1: 严重度评分(sofa/saps_ii/oasis)与"组成各评分的变量"★一同纳入 LASSO 候选池
#      (开关 INCLUDE_SEVERITY_SCORES=TRUE + SEVERITY_SCORE_MODE="candidate"; DROP_SEVERITY_COMPONENTS=FALSE):
#      其"参与评分计算"的生理/化验/干预组成变量(heart_rate/sbp/map/resp_rate/temperature/spo2/po2/
#      wbc/bun/creatinine/bilirubin/sodium/potassium/bicarbonate/platelet/ventilation/vasopressor 等)
#      与评分一同经 ★LASSO -> VIF<5 取舍(评分与组成变量天然共线, 由 LASSO 择优 + VIF 消解)。评分与各成分均仍保留于数据与 Table 1。
# 注2: 房颤(AF) 已归入"心脏病"成分(SQL heart_disease = CHD|CHF|AF), 参与 CMM 三成分计数,
#      故 AF 不作独立协变量, 也不作亚组分层(避免与心脏病成分重复)。
# 注3: 若想保留你原先更全的协变量(map/lactate/wbc/creatinine/aspirin/statin/ventilation/vasopressor),
#      可另设 Model 4 作敏感性分析。

# ====================================================================
# 03b 协变量筛选 (★本次改为 LASSO + VIF<5; 已【不再】做 单因素 Cox / 逐步 Cox)
#   流程 (强制人口学基线全程锁定, 不参与 LASSO 惩罚):
#     (0) 强制保留 人口学基线 (age/gender/bmi; race 按需求不纳入任何模型); 对应本文 Model I, 维持 M3 ⊇ M2。
#     (1) LASSO Cox (glmnet, family="cox", alpha=1): 候选 = 强制人口学 + cand_pool
#         (★候选池含 严重度评分 sofa/saps_ii/oasis 及"组成各评分的变量" + 其余临床协变量);
#         强制人口学 penalty.factor=0 永不剔除, 其余 penalty.factor=1;
#         ★ LASSO 前对"连续候选变量"标准化(均值0/标准差1), 因子哑变量不标准化(glmnet standardize=FALSE);
#         cv.glmnet 交叉验证选 lambda(LASSO_LAMBDA: "1se" 更稀疏 / "min" 偏差最小), 取非零系数变量为入选。
#     (2) VIF < 5 迭代去共线 (强制项不剔) 作收尾; 评分与其组成变量天然高度共线, 由此一步消解。
#   与上一版差异: 删除"单因素 Cox P<0.05"与"逐步 Cox(AIC)"两步, 改用 LASSO 选变量(VIF<5 不变)。
#   ★ 暴露成分(TG/Glu/HDL/TC/LDL): 本研究 8 暴露共享这些成分, 默认剔出候选池以防对所有暴露过度校正;
#     如需逐字复刻本文候选池(含血糖), 设 DROP_EXPOSURE_COMPONENTS <- FALSE。
# ====================================================================
cat(">>> Step 3b: 协变量筛选 (LASSO Cox -> VIF<5; 强制人口学全程锁定, 不参与惩罚)\n")

# (已按需求移除 USE_VARIABLE_SELECTION 与预设协变量: Model 2 一律采用下方筛选结果, 无预设回退)
VIF_THRESHOLD            <- 5       # 本文: VIF < 5 (LASSO 后去共线)
DROP_EXPOSURE_COMPONENTS <- TRUE    # 候选池是否剔除暴露计算成分(见上注; FALSE 可逐字复刻本文)

# ★★ 本次需求: Model 2 协变量改用 LASSO 选取(候选池 = 各评分 + 组成各评分的变量 + 其余临床协变量),
#    LASSO 前标准化连续变量, LASSO 后 VIF<5 去共线; 【不再】做 单因素 Cox / 逐步 Cox。
#    (评分整体纳入候选池: INCLUDE_SEVERITY_SCORES=TRUE + candidate; 组成变量留池: DROP_SEVERITY_COMPONENTS=FALSE。)
INCLUDE_SEVERITY_SCORES  <- TRUE    # ★本次需求: TRUE=三评分(sofa/saps_ii/oasis)整体纳入 LASSO 候选池, 由 LASSO->VIF 取舍
SEVERITY_SCORES_TO_USE   <- c("sofa", "saps_ii", "oasis")  # 纳入哪"几种"评分(可按需删减, 例如仅留 "sofa")
SEVERITY_SCORE_MODE      <- "candidate" # "force"=强制锁入 Model 2(LASSO 不罚 + VIF 不剔, 保证"几种评分"全部在模型内);
                                    # "candidate"=仅放入候选池, 由 LASSO -> VIF 自行取舍(可自动消解评分间/评分-成分共线)
DROP_SEVERITY_COMPONENTS <- FALSE   # ★本次需求: FALSE = 将"组成各评分的变量"(参与评分计算的生理/化验/干预变量)纳入候选池,
                                    #   与评分一同进入 LASSO 竞争, 由 LASSO 选取 + VIF<5 消解共线;
                                    #   TRUE = 把组成变量剔出候选池(防与评分双重计分)。
                                    # 注: FALSE 时评分(SOFA/SAPS II/OASIS)与其组成变量同处候选池, 二者天然高度共线 ->
                                    #     LASSO 倾向在共线组内择优保留, 残余共线再由 VIF<5 迭代剔除, 这正是"LASSO+VIF"的预期行为。
                                    #     若想要"只用组成变量、不要复合评分", 改设 INCLUDE_SEVERITY_SCORES <- FALSE(评分不入池, 仅留组成变量竞争)。

# --- LASSO 参数 (替代旧 单因素/逐步 参数) ---
LASSO_ENDPOINT       <- ENDPOINTS[[1]]  # LASSO 所用单一生存终点(默认主要终点=院内死亡; LASSO Cox 仅能用单一结局)
LASSO_LAMBDA         <- "min"           # 选 lambda: "min"(CV 偏差最小, 默认) / "1se"(更稀疏)
                                        #   注: 本数据评分+组成变量高度共线、单终点事件有限, "1se" 会收缩到仅强制人口学(空模型), 故默认用 "min"
LASSO_NFOLDS         <- 10              # cv.glmnet 交叉验证折数
STANDARDIZE_CONTINUOUS <- TRUE          # ★ LASSO 前标准化连续候选变量(均值0/标准差1); 二分类/哑变量不标准化

# --- (1) 强制保留: 人口学基线 = 本文 Model I(age/sex/bmi); race 不纳入任何模型(按需求) ---
force_keep_demo  <- intersect(covars_m2, names(df))             # age/gender/bmi (race 已排除); 维持 M3 ⊇ M2
# 注: 本文强制 gender/race; 此处按需求去掉 race(仅强制 age/gender/bmi); 如需额外强制临床先验在此填入。
# ★本次改动: 若以 "force" 方式纳入严重度评分, 在此把选用评分锁入强制保留集 ->
#   后续 LASSO(penalty.factor=0 不罚)与 VIF 去共线均不会剔除它们, 保证"几种评分"全部进入 Model 2。
#   (mode="candidate" 时此处为空, 评分改由候选池经 LASSO/VIF 自行取舍)
force_keep_prior <- if (INCLUDE_SEVERITY_SCORES && SEVERITY_SCORE_MODE == "force")
  intersect(SEVERITY_SCORES_TO_USE, names(df)) else character(0)

# --- 暴露计算成分: 据开关决定是否从候选池剔除 ---
exposure_components <- if (DROP_EXPOSURE_COMPONENTS)
  c("triglycerides", "glucose", "hdl", "cholesterol", "ldl") else character(0)

# --- 严重度评分(纳入 Model 2 候选) + 其"计算成分/组成变量"处理(★本次改动: 默认纳入候选池) ---
ALL_SEVERITY_SCORES <- c("sofa", "saps_ii", "oasis")
if (INCLUDE_SEVERITY_SCORES) {
  severity_scores_use <- intersect(SEVERITY_SCORES_TO_USE, names(df))   # 实际纳入 Model 2 候选的评分
  # 「评分计算成分/组成变量」: 参与 SOFA / SAPS II / OASIS 评分计算的 生理/化验/干预 变量。
  #   ★本次改动: 默认(DROP_SEVERITY_COMPONENTS=FALSE)将这些组成变量保留在候选池, 与评分一同进入
  #     LASSO -> VIF<5 筛选; 二者天然共线, 由 LASSO 择优保留 + VIF<5 消解(可能剔评分或剔部分组成变量)。
  #   (若设 DROP_SEVERITY_COMPONENTS=TRUE 则改回旧法: 把组成变量剔出候选池, 防"评分 + 原始变量"双重计分。)
  #   逐分量来源(MIMIC-IV 派生口径; GCS/尿量/术前LOS 不在本队列协变量池, 无需列出):
  #     SOFA    : po2 + ventilation(呼吸) / platelet(凝血) / bilirubin(肝) / map + vasopressor(循环) / creatinine(肾)
  #     SAPS II : heart_rate sbp temperature / po2 ventilation / bun wbc potassium sodium bicarbonate bilirubin
  #     OASIS   : heart_rate map resp_rate temperature ventilation
  #   ※ age 虽为 SAPS II/OASIS 分量, 但属核心人口学混杂(本文 Model I 强制项), 全程保留、不作成分剔除(维持 M3 ⊇ M2)。
  #   ※ dbp / rbc / chloride / calcium / lactate / inr / pt / ptt 不参与上述 3 评分计算, 仍留在候选池。
  #   ★ 若你的 sofa/oasis 等派生口径不同(如以 spo2 替代 po2), 在此向量增删即可。
  ALL_SCORE_COMPONENTS <- c(
    "heart_rate", "sbp", "map", "resp_rate", "temperature", "spo2",   # 生命体征 / 氧合
    "po2",                                              # 血气 / 酸碱
    "platelet", "wbc",                                  # 血液
    "creatinine", "bun", "bilirubin",                      # 生化(肾/肝; glucose 由暴露成分剔除)
    "sodium", "potassium", "bicarbonate",                  # 电解质 / 生化
    "ventilation", "vasopressor"                                      # 干预
  )
  # 仅 DROP_SEVERITY_COMPONENTS=TRUE 时把组成变量剔出候选池; 本次默认 FALSE -> 为空 -> 组成变量保留在池内参与筛选
  score_component_vars <- if (DROP_SEVERITY_COMPONENTS) ALL_SCORE_COMPONENTS else character(0)
  # 未被点名选用的其余评分, 仍排除出候选池(防其被自动筛入)
  severity_scores_exclude <- setdiff(ALL_SEVERITY_SCORES, severity_scores_use)
} else {
  # 旧默认: 完全不纳入评分, 也不剔成分
  severity_scores_use     <- character(0)
  ALL_SCORE_COMPONENTS    <- character(0)
  score_component_vars    <- character(0)
  severity_scores_exclude <- ALL_SEVERITY_SCORES
}
# 兼容旧名: 下游/日志中的 severity_scores 仍代表"被排除出候选池的评分"
severity_scores <- severity_scores_exclude

# --- 候选池 = 低缺失协变量集(vars_keep, 已剔>20%缺失, 与本文一致)
#     再去掉 暴露/(成分)/(严重度评分)/强制保留/ID/时间/结局/派生 列 ---
exclude_from_cand <- unique(c(
  EXPOSURES, paste0(EXPOSURES, "_std"), paste0(EXPOSURES, "_q"),
  exposure_components, severity_scores, score_component_vars,   # ← score_component_vars 仅在 DROP_SEVERITY_COMPONENTS=TRUE 时非空(剔出组成变量防双重计分); 本次默认 FALSE -> 为空 -> 组成变量保留在候选池
  force_keep_demo, force_keep_prior,
  "subject_id", "hadm_id", "stay_id", "intime", "outtime",
  "age_grp", "bmi_grp", "cm_count", "race",   # race 不作模型协变量(仅保留于数据/Table 1), 排除出候选池防被筛入
  "height_cm", "weight_kg",   # ← 身高/体重不纳入协变量(仍保留用于 SQL 计算 BMI/指标)
  vapply(ENDPOINTS, function(e) e$out,   character(1)),
  vapply(ENDPOINTS, function(e) e$surv,  character(1)),
  vapply(ENDPOINTS, function(e) e$group, character(1))
))
cand_pool <- setdiff(intersect(vars_keep, names(df)), exclude_from_cand)

# --- (1) LASSO Cox 变量选择 (★替代旧 单因素 Cox + 逐步 Cox) ---
#   候选 = 强制人口学(penalty.factor=0, 永不剔) + cand_pool(penalty.factor=1, 含评分及其组成变量);
#   ★ LASSO 前标准化连续设计列(均值0/标准差1), 哑变量不标准化, glmnet standardize=FALSE 避免重复;
#   family="cox" 以 LASSO_ENDPOINT(默认主要终点=院内死亡)为生存结局, cv.glmnet 交叉验证选 lambda。
lasso_forced <- intersect(unique(c(force_keep_demo, force_keep_prior)), names(df))
lasso_vars   <- intersect(unique(c(lasso_forced, cand_pool)), names(df))

# 去除零方差/单水平候选(model.matrix 对单水平因子会报 contrasts 错; 常量数值无信息); 强制人口学不剔
.has_var2   <- function(x) length(unique(x[!is.na(x)])) >= 2
.drop_const <- setdiff(lasso_vars[!vapply(df[lasso_vars], .has_var2, logical(1))], lasso_forced)
if (length(.drop_const)) {
  cat(sprintf("  [LASSO] 剔除零方差/单水平候选: %s\n", paste(.drop_const, collapse = ", ")))
  lasso_vars <- setdiff(lasso_vars, .drop_const)
}

# 设计矩阵 (因子 -> 哑变量); 去截距
X_full <- model.matrix(as.formula(paste("~", paste(lasso_vars, collapse = " + "))), data = df)
X_full <- X_full[, colnames(X_full) != "(Intercept)", drop = FALSE]

# 每个设计列回溯来源变量名(因子哑变量取最长前缀匹配) -> 供 penalty.factor 赋值 + 还原变量级入选
col2srcvar <- vapply(colnames(X_full), function(cn) {
  hit <- lasso_vars[vapply(lasso_vars, function(x) startsWith(cn, x), logical(1))]
  if (length(hit)) hit[which.max(nchar(hit))] else cn
}, character(1))

# ★ 连续设计列标准化(均值0/标准差1); 二分类/哑变量(取值⊆{0,1})不标准化; 常量列不缩放(避免除0)
n_cont <- 0L; n_dummy <- 0L
if (STANDARDIZE_CONTINUOUS) {
  is_dummy01 <- apply(X_full, 2, function(z) all(z %in% c(0, 1)))
  cont_cols  <- which(!is_dummy01); n_cont <- length(cont_cols); n_dummy <- sum(is_dummy01)
  if (n_cont) {
    ctr <- colMeans(X_full[, cont_cols, drop = FALSE])
    sdv <- apply(X_full[, cont_cols, drop = FALSE], 2, sd); sdv[sdv == 0] <- 1
    X_full[, cont_cols] <- sweep(sweep(X_full[, cont_cols, drop = FALSE], 2, ctr, "-"), 2, sdv, "/")
  }
  cat(sprintf("  [LASSO] 已标准化连续设计列 %d 个(二分类哑变量 %d 个不标准化)\n", n_cont, n_dummy))
}

# 惩罚因子: 强制人口学来源列=0(不惩罚, 永不剔除); 其余=1
pen_fac <- ifelse(col2srcvar %in% lasso_forced, 0, 1)

# 生存结局(LASSO_ENDPOINT, 默认主要终点=院内死亡; LASSO Cox 仅能用单一结局)
y_surv <- survival::Surv(df[[LASSO_ENDPOINT$surv]], df[[LASSO_ENDPOINT$out]])

set.seed(20260531)   # 复现交叉验证折分
cvfit <- tryCatch(
  glmnet::cv.glmnet(x = X_full, y = y_surv, family = "cox", alpha = 1,
                    penalty.factor = pen_fac, standardize = FALSE, nfolds = LASSO_NFOLDS),
  error = function(e) { cat(sprintf("  [LASSO] cv.glmnet 失败(%s) -> 仅保留强制人口学\n",
                                    conditionMessage(e))); NULL })

if (is.null(cvfit)) {
  selected_lasso <- lasso_forced
  src_abs_coef   <- setNames(rep(0, length(lasso_vars)), lasso_vars)
} else {
  # ★ 诊断: 事件数 + 两个 lambda 各自"非强制入选"个数(判断 1se 是否过度稀疏到仅剩强制项)
  .n_event <- sum(df[[LASSO_ENDPOINT$out]] == 1, na.rm = TRUE)
  .nz_at   <- function(s) {
    bb <- as.matrix(coef(cvfit, s = s))[, 1]
    length(setdiff(unique(col2srcvar[colnames(X_full) %in% names(bb)[bb != 0]]), lasso_forced))
  }
  .n_min <- .nz_at(cvfit$lambda.min); .n_1se <- .nz_at(cvfit$lambda.1se)
  cat(sprintf("  [LASSO 诊断] 终点=%s 事件数=%d; 非强制入选个数: lambda.min=%d, lambda.1se=%d\n",
              LASSO_ENDPOINT$label, .n_event, .n_min, .n_1se))
  if (.n_1se == 0 && LASSO_LAMBDA == "1se")
    cat("  [LASSO 提示] lambda.1se 收缩到仅强制人口学(空模型); 建议设 LASSO_LAMBDA <- \"min\"\n")

  lambda_sel <- if (LASSO_LAMBDA == "min") cvfit$lambda.min else cvfit$lambda.1se
  beta_sel   <- as.matrix(coef(cvfit, s = lambda_sel))         # family=cox 无截距, rownames=设计列名
  beta_vec   <- setNames(beta_sel[, 1], rownames(beta_sel))
  nz_cols    <- setdiff(names(beta_vec)[beta_vec != 0], "(Intercept)")   # 非零系数设计列
  lasso_keep_src <- unique(col2srcvar[colnames(X_full) %in% nz_cols])    # 还原为变量级
  selected_lasso <- intersect(unique(c(lasso_forced, lasso_keep_src)), names(df))  # 强制项始终并入
  # 变量级 |系数|(因子=各哑变量绝对值之和), 供审计排序
  src_abs_coef <- tapply(abs(beta_vec[colnames(X_full)]), col2srcvar, sum, na.rm = TRUE)
  cat(sprintf("  LASSO(family=cox, 终点=%s, lambda.%s=%.4g): 候选 %d -> 入选 %d : %s\n",
              LASSO_ENDPOINT$label, LASSO_LAMBDA, lambda_sel,
              length(setdiff(lasso_vars, lasso_forced)),
              length(setdiff(selected_lasso, lasso_forced)),
              paste(setdiff(selected_lasso, lasso_forced), collapse = ", ")))
}

# --- 审计报表: 每个候选变量的 LASSO |系数| 与是否入选(强制项标记 forced=TRUE) ---
sel_report <- bind_rows(lapply(lasso_vars, function(v) {
  ac <- if (v %in% names(src_abs_coef)) as.numeric(src_abs_coef[[v]]) else 0
  data.frame(variable = v, forced = v %in% lasso_forced,
             lasso_abscoef = round(ifelse(is.na(ac), 0, ac), 4),
             selected = v %in% selected_lasso)
}))

# --- (LASSO 入选 + 强制人口学) 交 VIF<5 去共线 ---
covars_sel <- intersect(selected_lasso, names(df))

# --- VIF 去共线 (本文 VIF<5; 强制保留项不剔; 因子哑变量按变量名取最大 VIF; base lm, 无需 car) ---
compute_vif <- function(vars) {
  vars <- intersect(vars, names(df))
  if (length(vars) < 2) return(setNames(rep(NA_real_, length(vars)), vars))
  mm <- model.matrix(as.formula(paste("~", paste(vars, collapse = " + "))), data = df)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  col2var <- vapply(colnames(mm), function(cn) {
    hit <- vars[vapply(vars, function(x) startsWith(cn, x), logical(1))]
    if (length(hit)) hit[which.max(nchar(hit))] else cn
  }, character(1))
  vif_col <- vapply(seq_len(ncol(mm)), function(j) {
    r2 <- summary(lm(mm[, j] ~ mm[, -j, drop = FALSE]))$r.squared
    1 / (1 - min(r2, 0.999999))
  }, numeric(1))
  tapply(vif_col, col2var, max)
}
protected <- unique(c(force_keep_demo, force_keep_prior))
repeat {
  vv  <- compute_vif(covars_sel)
  bad <- setdiff(names(vv)[which(vv >= VIF_THRESHOLD)], protected)
  if (!length(bad)) break
  drop_v <- bad[which.max(vv[bad])]
  cat(sprintf("  [VIF>=%g] 剔除 %s (VIF=%.1f)\n", VIF_THRESHOLD, drop_v, vv[drop_v]))
  covars_sel <- setdiff(covars_sel, drop_v)
  if (length(covars_sel) <= length(protected)) break
}
vif_final <- compute_vif(covars_sel)

# --- 落地: Model 2 协变量 = Step 3b 筛选结果(★LASSO -> VIF<5);
#     ★本次改动: 严重度评分 + 其组成变量 一同作为候选, 共同经 LASSO + VIF<5 取舍(组成变量不再预先剔出) ---
covars_m3 <- intersect(covars_sel, names(df))
# ★ 显式兜底(force 模式): 确保选用的严重度评分一定保留在 Model 2。
#   force 模式下评分已并入 force_keep_prior -> LASSO penalty.factor=0(不罚)与 protected(VIF 跳过表)双重保护,
#   即便评分间 VIF 远超阈值也不会被剔。此处再用 union 显式兜底, 防任何上游改动意外破坏锁定。
if (INCLUDE_SEVERITY_SCORES && SEVERITY_SCORE_MODE == "force")
  covars_m3 <- union(covars_m3, intersect(severity_scores_use, names(df)))
.sev_in_m3 <- intersect(severity_scores_use, covars_m3)
if (INCLUDE_SEVERITY_SCORES) {
  .comp_note <- if (DROP_SEVERITY_COMPONENTS) {
    "评分计算成分已剔出候选池防双重计分"
  } else {
    "评分组成变量已纳入候选池, 与评分一同经 LASSO->VIF<5 筛选"
  }
  cat(sprintf("  >>> Model 2 协变量 = 筛选结果(含评分{%s}, 方式=%s); %s\n",
              paste(.sev_in_m3, collapse = "/"), SEVERITY_SCORE_MODE, .comp_note))
} else {
  cat("  >>> Model 2 协变量 = 筛选结果(LASSO -> VIF<5; 严重度评分已排除候选池)\n")
}

# ====================================================================
# 03c   手动指定 Model 2 协变量 (可选覆盖; ★本次需求已置 FALSE -> Model 2 用 Step 3b LASSO 结果)
#   - 上方 Step 3b 的 LASSO + VIF<5 筛选照常运行并输出审计报表
#     (covariate_selection_report.csv + 控制台)。当 USE_MANUAL_COVARS_M3=TRUE 时,
#     "最终用于建模"的 covars_m3(本脚本输出标签 = "Model 2" / Adjusted II)改用下面这份显式清单,
#     使发表模型固定、可复现 -> 此时 Step 3b LASSO 仅作审计, 不决定最终 Model 2。
#   - ★本次需求: Model 2 协变量由 LASSO(评分 + 组成变量 + 其余临床协变量)经 VIF<5 产生, 故此处置 FALSE。
#     若想改回"固定 12 项手动清单", 把 USE_MANUAL_COVARS_M3 改回 TRUE 即可(一行, 无需删代码)。
#   - intersect 兜底: 清单中若有变量因高缺失(>20%)被剔出 df, 自动跳过并提示。
# ====================================================================
USE_MANUAL_COVARS_M3 <- FALSE    # ★本次需求: FALSE = Model 2 用 Step 3b LASSO+VIF<5 结果; TRUE = 改用下方固定 12 项清单
MANUAL_COVARS_M3 <- c("age", "gender", "bmi", "heart_rate", "resp_rate", "temperature",
                      "wbc", "stroke", "pad", "ventilation", "vasopressor", "antihypertensive")
if (USE_MANUAL_COVARS_M3) {
  .m2_missing <- setdiff(MANUAL_COVARS_M3, names(df))
  if (length(.m2_missing))
    cat(sprintf("  [手动 Model 2] 以下指定变量不在 df 中(可能高缺失被剔), 已跳过: %s\n",
                paste(.m2_missing, collapse = ", ")))
  covars_m3  <- intersect(MANUAL_COVARS_M3, names(df))      # 保持给定顺序; covars_m3 即输出标签 "Model 2"
  .sev_in_m3 <- intersect(severity_scores_use, covars_m3)   # 同步刷新(此处恒为空, 清单无评分), 供下方报表/日志
  cat(sprintf("  >>> ★ Model 2 改用手动清单(%d 项, 覆盖自动筛选): %s\n",
              length(covars_m3), paste(covars_m3, collapse = ", ")))
} else {
  cat("  >>> ★ Model 2 采用 Step 3b LASSO+VIF<5 筛选结果(USE_MANUAL_COVARS_M3=FALSE)\n")
}

# --- 报表输出 + 控制台摘要 ---
sel_report$in_final <- sel_report$variable %in% covars_m3
sel_report <- sel_report[order(-sel_report$selected, -sel_report$lasso_abscoef), ]
write.csv(sel_report, file.path(DIR_CSV, "covariate_selection_report.csv"), row.names = FALSE)
write.csv(data.frame(variable = names(vif_final), VIF = round(as.numeric(vif_final), 2)),
          file.path(DIR_CSV, "covariate_vif_final.csv"), row.names = FALSE)
cat("  强制保留(人口学/嵌套, LASSO 不罚): ", paste(force_keep_demo, collapse = ", "), "\n")
if (length(force_keep_prior)) {
  cat("  强制保留(临床先验, LASSO 不罚)   : ", paste(force_keep_prior, collapse = ", "), "\n")
}
cat(sprintf("  候选池%s: %s\n",
            if (DROP_EXPOSURE_COMPONENTS) "(已剔暴露成分)" else "(含暴露成分/逐字复刻本文)",
            paste(cand_pool, collapse = ", ")))
cat("  LASSO 入选(非零系数, 含强制项): ", paste(selected_lasso, collapse = ", "), "\n")
cat("  最终 covars_m3 (Model 2, 过 VIF<5): ", paste(covars_m3, collapse = ", "), "\n")
if (INCLUDE_SEVERITY_SCORES) {
  cat(sprintf("  严重度评分(纳入 LASSO 候选池, 方式=%s): %s\n", SEVERITY_SCORE_MODE, paste(.sev_in_m3, collapse = ", ")))
  if (DROP_SEVERITY_COMPONENTS) {
    cat("  评分计算成分(已剔出候选池, 防双重计分): ",
        paste(intersect(score_component_vars, names(df)), collapse = ", "), "\n")
  } else {
    # ★本次改动: 报告评分组成变量在 LASSO 各阶段的去向, 便于核对"协变量纳入了组成变量"
    .comp_in_df   <- intersect(ALL_SCORE_COMPONENTS, names(df))
    .comp_in_pool <- intersect(ALL_SCORE_COMPONENTS, cand_pool)        # 进入 LASSO 候选池的组成变量
    .comp_lasso   <- intersect(ALL_SCORE_COMPONENTS, selected_lasso)   # LASSO 入选(非零系数)的组成变量
    .comp_final   <- intersect(ALL_SCORE_COMPONENTS, covars_m3)        # 最终(过 VIF<5)进入 Model 2 的组成变量
    cat("  ★评分组成变量(本次纳入 LASSO 候选池): ", paste(.comp_in_pool, collapse = ", "), "\n")
    cat("     - LASSO 入选(非零系数): ", paste(.comp_lasso, collapse = ", "), "\n")
    cat("     - 最终入 Model 2(过 VIF<5): ", paste(.comp_final, collapse = ", "), "\n")
    .comp_missing <- setdiff(.comp_in_df, .comp_in_pool)
    if (length(.comp_missing))
      cat("     - (组成变量在 df 中但未入池, 多因高缺失被剔/或在排除集): ",
          paste(.comp_missing, collapse = ", "), "\n")
  }
  if (length(severity_scores_exclude))
    cat("  (未点名选用的评分, 仍排除出候选池): ", paste(severity_scores_exclude, collapse = ", "), "\n")
} else {
  cat("  (严重度评分已排除候选池, 未进入 Model 2: ", paste(severity_scores, collapse = ", "), ")\n")
}
print(sel_report, row.names = FALSE)

save(df, tbl1_vars, cat_vars, covars_m2, covars_m3,
     sel_report, vif_final, EXPOSURES, EXPOSURE_LABELS,
     file = file.path(DIR_LOAD, "labeled_df.Rdata"))

# ====================================================================
# 04 Table 1 (按院内死亡分层) —— strong::ok_tbl1
# ====================================================================
cat(">>> Step 4: Table 1 (按院内死亡分层, ok_tbl1)\n")

tbl1_ft <- strong::ok_tbl1(
  vars        = tbl1_vars,
  col_name_gp = ENDPOINTS[[1]]$group,                  # ← 主要终点(院内死亡)分层 = inhosp_group
  df          = df,
  method      = "auto",
  overall     = TRUE,
  smd         = TRUE,
  bold_sig    = TRUE,
  eps         = c(0.001, 0.01, 0.05)
)

tbl1_ft <- tbl1_ft %>%
  flextable::set_table_properties(layout = "autofit", width = 1) %>%
  flextable::fontsize(size = 8, part = "all") %>%
  flextable::padding(padding = 2, part = "all")

sect_landscape <- officer::prop_section(
  page_size    = officer::page_size(orient = "landscape",
                                    width = 11.69, height = 8.27),
  page_margins = officer::page_mar(top = 0.5, bottom = 0.5,
                                   left = 0.5, right = 0.5,
                                   header = 0.3, footer = 0.3, gutter = 0)
)

flextable::save_as_docx(
  tbl1_ft,
  path       = file.path(DIR_DOCX, "table1_baseline.docx"),
  pr_section = sect_landscape
)
cat("  Table 1 已输出: ", file.path(DIR_DOCX, "table1_baseline.docx"), "\n")

# ====================================================================
# 05 Kaplan-Meier (8 指标 × 院内+3个时间点) —— strong::ok_unadj_km
# ====================================================================
cat(">>> Step 5: KM 分析 (院内 + 30/90/365 天, ok_unadj_km)\n")

km_palette <- c("#2E9FDF", "#00BA38", "#F8766D", "#E76BF3")

for (ep in ENDPOINTS) {
  surv_var <- ep$surv
  out_var  <- ep$out
  brk      <- ep$brk

  km_plots <- map2(EXPOSURES, EXPOSURE_LABELS, function(exp_var, lab) {
    q_var <- paste0(exp_var, "_q")
    strong::ok_unadj_km(
      time          = surv_var,
      outcome       = out_var,
      col_name_gp   = q_var,
      df            = df,
      break.x.by    = brk,
      conf.int      = TRUE,
      risk.table    = TRUE,
      tables.height = 0.25,
      legend.title  = lab,
      legend.labs   = paste0("Q", 1:4),
      palette       = km_palette,
      ggtheme       = theme_bw(),
      title         = paste0(lab, " Quartiles - ", ep$label, " Mortality"),
      ylim          = c(0.5, 1)
    )
  })

  pdf(file.path(DIR_DOCX, paste0("fig_km_curves_", ep$suffix, ".pdf")), width = 14, height = 18)
  for (p in km_plots) print(p)
  dev.off()
  cat(sprintf("  %s KM 已输出\n", ep$label))
}

# ====================================================================
# 06 Cox 回归 (8 指标 × 三模型 × 院内+3个时间点)
#    per-SD 为主 + per-1-unit 为辅; 四分位(Q1参考) + P-trend
# ====================================================================
cat(">>> Step 6: Cox 回归 (院内 + 30/90/365 天)\n")

run_cox <- function(exp_var, exp_label, surv_var, out_var, ep_label) {
  std_var  <- paste0(exp_var, "_std")
  q_var    <- paste0(exp_var, "_q")

  # 公共: 拟合一个模型并抽取某个项的 HR
  fit_extract <- function(term, rhs, model_lab, type_lab) {
    fml <- as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ", rhs))
    mod <- coxph(fml, data = df)
    s <- summary(mod); ci <- s$conf.int; co <- s$coefficients
    r <- which(rownames(co) == term); if (length(r) == 0) r <- 1
    data.frame(endpoint = ep_label, index = exp_label, model = model_lab, type = type_lab,
               HR = round(ci[r,1],3), CI_lo = round(ci[r,3],3), CI_hi = round(ci[r,4],3),
               p = round(co[r,5],4))
  }

  rhs_m2 <- paste(covars_m2, collapse = " + ")
  rhs_m3 <- paste(covars_m3, collapse = " + ")

  rows <- bind_rows(
    # --- per-SD (主) --- 标签对齐本文: Crude(未校正) / Model 1(age+sex+BMI=Adjusted I) / Model 2(全校正=Adjusted II)
    fit_extract(std_var, std_var,                                   "Crude",   "Per SD"),
    fit_extract(std_var, paste0(std_var, " + ", rhs_m2),            "Model 1", "Per SD"),
    fit_extract(std_var, paste0(std_var, " + ", rhs_m3),            "Model 2", "Per SD"),
    # --- per-1-unit (辅, 与本文 Table 2 同尺度) ---
    fit_extract(exp_var, exp_var,                                   "Crude",   "Per 1 unit"),
    fit_extract(exp_var, paste0(exp_var, " + ", rhs_m2),            "Model 1", "Per 1 unit"),
    fit_extract(exp_var, paste0(exp_var, " + ", rhs_m3),            "Model 2", "Per 1 unit")
  )

  # --- 四分位 (Model 2 全校正, Q1 参考) ---
  m3_q <- coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ", q_var, " + ", rhs_m3)), data = df)
  s_q <- summary(m3_q); ci_q <- s_q$conf.int; co_q <- s_q$coefficients
  q_rows <- data.frame(
    endpoint = ep_label, index = exp_label, model = "Model 2",
    type = c("Q2 vs Q1","Q3 vs Q1","Q4 vs Q1"),
    HR = round(ci_q[1:3,1],3), CI_lo = round(ci_q[1:3,3],3),
    CI_hi = round(ci_q[1:3,4],3), p = round(co_q[1:3,5],4)
  )

  # --- P for trend (四分位序号作连续, Model 2 全校正) ---
  df_tmp <- df; df_tmp$q_num <- as.numeric(df_tmp[[q_var]])
  trend_mod <- coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ q_num + ", rhs_m3)), data = df_tmp)
  p_trend <- round(summary(trend_mod)$coefficients[1,5], 4)

  # --- PH 检验 (Model 2 全校正, per-SD) 记录全局 P (供 Step 6b 决定是否做时依 Cox) ---
  m3_std <- coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ", std_var, " + ", rhs_m3)), data = df)
  ph_global <- tryCatch(cox.zph(m3_std)$table["GLOBAL","p"], error = function(e) NA_real_)
  cat(sprintf("  %s %s PH Global P = %.3f\n", ep_label, exp_label, ph_global))

  bind_rows(rows, q_rows) %>%
    mutate(p_trend  = ifelse(type == "Q4 vs Q1", p_trend, NA),
           ph_global = ifelse(type == "Per SD" & model == "Model 2", ph_global, NA))
}

cox_all <- map_dfr(ENDPOINTS, function(ep) {
  map_dfr(seq_along(EXPOSURES),
          ~run_cox(EXPOSURES[.x], EXPOSURE_LABELS[.x], ep$surv, ep$out, ep$label))
})
write.csv(cox_all, file.path(DIR_CSV, "table2_cox_all_endpoints.csv"), row.names = FALSE)
print(cox_all)

# ====================================================================
# 06b 时依 Cox (对 PH 违反指标的敏感性分析) —— 与 log(t) 交互
#     本文做法: Schoenfeld P<0.05 的指标改用时依 Cox。
# ====================================================================
cat(">>> Step 6b: 时依 Cox (PH 违反者)\n")

td_cox <- map_dfr(ENDPOINTS, function(ep) {
  surv_var <- ep$surv; out_var <- ep$out; ep_label <- ep$label
  rhs_m3 <- paste(covars_m3, collapse = " + ")

  map_dfr(seq_along(EXPOSURES), function(i) {
    exp_var <- EXPOSURES[i]; exp_label <- EXPOSURE_LABELS[i]; std_var <- paste0(exp_var, "_std")

    m3_std <- coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ", std_var, " + ", rhs_m3)), data = df)
    ph_p   <- tryCatch(cox.zph(m3_std)$table["GLOBAL","p"], error = function(e) NA_real_)
    if (is.na(ph_p) || ph_p >= 0.05) return(NULL)   # 仅对违反者做

    fit_td <- tryCatch(
      coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ", std_var,
                              " + tt(", std_var, ") + ", rhs_m3)),
            data = df, tt = function(x, t, ...) x * log(pmax(t, 1e-3))),
      error = function(e) NULL)
    if (is.null(fit_td)) return(NULL)

    s <- summary(fit_td); ci <- s$conf.int; co <- s$coefficients
    main_r <- which(rownames(co) == std_var)
    tt_r   <- grep("^tt\\(", rownames(co))
    data.frame(
      endpoint = ep_label, index = exp_label, ph_global = round(ph_p, 4),
      HR_main = round(ci[main_r,1],3), HR_main_lo = round(ci[main_r,3],3),
      HR_main_hi = round(ci[main_r,4],3), p_main = round(co[main_r,5],4),
      HR_time = round(ci[tt_r,1],3), p_time = round(co[tt_r,5],4)   # 与 log(t) 交互: <1 表示效应随时间衰减
    )
  })
})
write.csv(td_cox, file.path(DIR_CSV, "cox_timevarying_phviolators.csv"), row.names = FALSE)
print(td_cox)
# ====================================================================
# 06c Cox 回归 —— 5 个插补数据集 Rubin 法则合并 (★审稿意见 3/8)
#   - 与 Step 6 相同设定 (Crude/Model 1/Model 2; per-SD / per-unit; 四分位+趋势 仅全校正),
#     分别在 5 个完成集上拟合, 对 log(HR) 及 SE 按 Rubin 法则合并 (Barnard-Rubin 自由度)。
#   - 暴露/结局/生存时间完整未插补, 四分位切点与标准化暴露在各完成集间完全一致,
#     仅被插补协变量不同 -> 合并仅反映协变量插补不确定性。
#   - ★修订稿 MIMIC 外部复制表应以本步合并结果为准 (table2_cox_pooled_rubin.csv)。
# ====================================================================
cat(">>> Step 6c: Cox 回归 (Rubin 法则合并 5 个插补集)\n")

imputed_cols <- names(miss_rate[vars_keep])[miss_rate[vars_keep] > 0]
imputed_cols <- intersect(imputed_cols, names(df))

df_imp_list <- lapply(imp_data_list, function(d) {
  out <- df
  for (cl in imputed_cols) out[[cl]] <- d[[cl]]
  out
})

pool_rubin <- function(beta, se) {
  m    <- length(beta)
  qbar <- mean(beta)
  ubar <- mean(se^2)
  b    <- if (m > 1) stats::var(beta) else 0
  tvar <- ubar + (1 + 1/m) * b
  se_p <- sqrt(tvar)
  lam  <- if (tvar > 0) (1 + 1/m) * b / tvar else 0
  nu   <- if (lam > 0) (m - 1) / lam^2 else Inf
  stat <- qbar / se_p
  p    <- 2 * stats::pt(abs(stat), df = nu, lower.tail = FALSE)
  crit <- stats::qt(0.975, df = nu)
  c(HR = exp(qbar), CI_lo = exp(qbar - crit * se_p),
    CI_hi = exp(qbar + crit * se_p), p = p)
}

fit_term <- function(data, term, rhs, surv_var, out_var) {
  mod <- tryCatch(
    survival::coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ", rhs)),
                    data = data),
    error = function(e) NULL)
  if (is.null(mod)) return(c(beta = NA_real_, se = NA_real_))
  co <- summary(mod)$coefficients
  r  <- which(rownames(co) == term)
  if (length(r) == 0) return(c(beta = NA_real_, se = NA_real_))
  c(beta = co[r, 1], se = co[r, 3])
}

pool_one <- function(term, rhs, surv_var, out_var) {
  fits <- lapply(df_imp_list, fit_term, term = term, rhs = rhs,
                 surv_var = surv_var, out_var = out_var)
  beta <- sapply(fits, `[[`, "beta"); se <- sapply(fits, `[[`, "se")
  ok <- !is.na(beta) & !is.na(se)
  if (sum(ok) < 2) return(c(HR = NA_real_, CI_lo = NA_real_, CI_hi = NA_real_, p = NA_real_))
  pool_rubin(beta[ok], se[ok])
}

rhs_m2 <- paste(covars_m2, collapse = " + ")
rhs_m3 <- paste(covars_m3, collapse = " + ")

cox_pooled <- map_dfr(ENDPOINTS, function(ep) {
  surv_var <- ep$surv; out_var <- ep$out
  map_dfr(seq_along(EXPOSURES), function(i) {
    exp_var <- EXPOSURES[i]; exp_label <- EXPOSURE_LABELS[i]
    std_var <- paste0(exp_var, "_std"); q_var <- paste0(exp_var, "_q")
    specs <- list(
      list(std_var, std_var,                        "Crude",   "Per SD"),
      list(std_var, paste0(std_var, " + ", rhs_m2), "Model 1", "Per SD"),
      list(std_var, paste0(std_var, " + ", rhs_m3), "Model 2", "Per SD"),
      list(exp_var, exp_var,                        "Crude",   "Per 1 unit"),
      list(exp_var, paste0(exp_var, " + ", rhs_m2), "Model 1", "Per 1 unit"),
      list(exp_var, paste0(exp_var, " + ", rhs_m3), "Model 2", "Per 1 unit")
    )
    rows <- map_dfr(specs, function(sp) {
      pr <- pool_one(sp[[1]], sp[[2]], surv_var, out_var)
      data.frame(endpoint = ep$label, index = exp_label, model = sp[[3]], type = sp[[4]],
                 HR = round(unname(pr["HR"]), 3), CI_lo = round(unname(pr["CI_lo"]), 3),
                 CI_hi = round(unname(pr["CI_hi"]), 3), p = round(unname(pr["p"]), 4), row.names = NULL)
    })
    # 四分位 (Q2/Q3/Q4 vs Q1) + P-trend, 仅全校正 Model 2 (同 Step 6 口径)
    df_imp_list <<- lapply(df_imp_list, function(d) { d$q_num <- as.numeric(d[[q_var]]); d })
    qr <- map_dfr(paste0("Q", 2:4), function(qq) {
      pr <- pool_one(paste0(q_var, qq), paste0(q_var, " + ", rhs_m3), surv_var, out_var)
      data.frame(endpoint = ep$label, index = exp_label, model = "Model 2",
                 type = paste0(qq, " vs Q1"),
                 HR = round(unname(pr["HR"]), 3), CI_lo = round(unname(pr["CI_lo"]), 3),
                 CI_hi = round(unname(pr["CI_hi"]), 3), p = round(unname(pr["p"]), 4), row.names = NULL)
    })
    pr_tr <- pool_one("q_num", paste0("q_num + ", rhs_m3), surv_var, out_var)
    qr$p_trend <- c(NA, NA, round(unname(pr_tr["p"]), 4))   # P-trend 记在 Q4 行 (同 Step 6)
    bind_rows(rows, qr)
  })
})
cox_pooled$n_imp <- length(df_imp_list)
write.csv(cox_pooled, file.path(DIR_CSV, "table2_cox_pooled_rubin.csv"), row.names = FALSE)
cat("  Rubin 合并 Cox 结果已输出: table2_cox_pooled_rubin.csv\n")
print(cox_pooled)

# ====================================================================
# 06d 完整病例 (complete-case) 敏感性分析 (★审稿意见 3)
#   - 仅在 Model 1/2 协变量完整的病例上拟合全校正 Model 2 (暴露完整 by design);
#     与 Step 6c 合并结果并列输出 (per-SD / per-unit / 四分位 全口径)。
# ====================================================================
cat(">>> Step 6d: 完整病例敏感性分析 (Model 2 全校正)\n")

cc_vars <- intersect(c(covars_m2, covars_m3), names(cohort_raw))
cc_idx  <- stats::complete.cases(cohort_raw[, cc_vars, drop = FALSE])
df_cc   <- df[cc_idx, , drop = FALSE]
cat(sprintf("  完整病例: %d / %d (%.1f%%)\n", nrow(df_cc), nrow(df),
            100 * nrow(df_cc) / nrow(df)))

cc_res <- map_dfr(ENDPOINTS, function(ep) {
  surv_var <- ep$surv; out_var <- ep$out
  map_dfr(seq_along(EXPOSURES), function(i) {
    exp_var <- EXPOSURES[i]; exp_label <- EXPOSURE_LABELS[i]
    std_var <- paste0(exp_var, "_std"); q_var <- paste0(exp_var, "_q")
    rows <- map_dfr(c("Per SD", "Per 1 unit"), function(tp) {
      term <- if (tp == "Per SD") std_var else exp_var
      ft   <- fit_term(df_cc, term, paste0(term, " + ", rhs_m3), surv_var, out_var)
      z    <- ft["beta"] / ft["se"]
      data.frame(endpoint = ep$label, index = exp_label, model = "Model 2", type = tp,
                 n_cc = nrow(df_cc),
                 HR_cc    = round(exp(ft["beta"]), 3),
                 CI_lo_cc = round(exp(ft["beta"] - 1.96 * ft["se"]), 3),
                 CI_hi_cc = round(exp(ft["beta"] + 1.96 * ft["se"]), 3),
                 p_cc     = round(2 * pnorm(-abs(z)), 4), row.names = NULL)
    })
    df_cc_q <- df_cc; df_cc_q$q_num <- as.numeric(df_cc_q[[q_var]])
    qr <- map_dfr(paste0("Q", 2:4), function(qq) {
      ft <- fit_term(df_cc, paste0(q_var, qq), paste0(q_var, " + ", rhs_m3), surv_var, out_var)
      z  <- ft["beta"] / ft["se"]
      data.frame(endpoint = ep$label, index = exp_label, model = "Model 2",
                 type = paste0(qq, " vs Q1"), n_cc = nrow(df_cc),
                 HR_cc    = round(exp(ft["beta"]), 3),
                 CI_lo_cc = round(exp(ft["beta"] - 1.96 * ft["se"]), 3),
                 CI_hi_cc = round(exp(ft["beta"] + 1.96 * ft["se"]), 3),
                 p_cc     = round(2 * pnorm(-abs(z)), 4), row.names = NULL)
    })
    ft_tr <- fit_term(df_cc_q, "q_num", paste0("q_num + ", rhs_m3), surv_var, out_var)
    qr$p_trend_cc <- c(NA, NA, round(2 * pnorm(-abs(ft_tr["beta"] / ft_tr["se"])), 4))
    bind_rows(rows, qr)
  })
})

# 与 Rubin 合并结果并排 (Model 2 全部类型)
cc_compare <- cc_res %>%
  left_join(cox_pooled %>%
              filter(model == "Model 2") %>%
              select(endpoint, index, type, p_trend_pooled = p_trend,
                     HR_pooled = HR, CI_lo_pooled = CI_lo,
                     CI_hi_pooled = CI_hi, p_pooled = p),
            by = c("endpoint", "index", "type"))
write.csv(cc_compare, file.path(DIR_CSV, "sensitivity_complete_case_vs_pooled.csv"),
          row.names = FALSE)
cat("  完整病例 vs Rubin 合并 对照已输出: sensitivity_complete_case_vs_pooled.csv\n")
print(cc_compare)

# ====================================================================
# 07 RCS (8 指标 × 院内+3个时间点) —— 3 节点, 中位数参考, strong::rcs_cph
# ====================================================================
cat(">>> Step 7: RCS (3 节点, 院内 + 30/90/365 天, rcs_cph)\n")
dd <- datadist(df); options(datadist = 'dd')

# 数值型非线性 P (rms::cph + anova); 3 节点与本文一致
get_p_nl <- function(exp_var, surv_var, out_var) {
  fml <- as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ rcs(", exp_var, ", 3) + ",
                           paste(covars_m3, collapse = " + ")))
  fit <- cph(fml, data = df, x = TRUE, y = TRUE)
  an  <- stats::anova(fit)
  rn  <- trimws(rownames(an))
  pcol <- grep("^P$", colnames(an))
  ridx <- which(rn == "Nonlinear")
  if (length(ridx) == 0) ridx <- grep("NONLINEAR", toupper(rn))
  if (length(ridx) == 0 || length(pcol) == 0) return(NA_real_)
  unname(an[ridx[1], pcol[1]])
}

rcs_summary <- list()

for (ep in ENDPOINTS) {
  surv_var <- ep$surv; out_var <- ep$out; ep_label <- ep$label

  rcs_plots <- map2(EXPOSURES, EXPOSURE_LABELS, function(exp_var, lab) {
    strong::rcs_cph(
      df            = df,
      time          = surv_var,
      outcome       = out_var,
      x             = exp_var,
      covars        = covars_m3,
      x_label       = lab,
      outcome_label = ep_label,
      knot          = 3,                      # ← 3 节点 (本文)
      y_axis_lab    = "Adjusted HR (95% CI)"  # Cox -> HR
    ) + ggtitle(paste0(lab, " (", ep_label, ")"))
    # 注: rms/rcs_cph 默认以预测变量中位数为参考 (HR=1), 与本文一致;
    #     若 strong::rcs_cph 暴露了 ref 参数, 可显式设为 median(df[[exp_var]])。
  })

  combined <- patchwork::wrap_plots(rcs_plots, ncol = 2) +
    patchwork::plot_annotation(
      title = paste0(ep_label, " mortality — RCS (Model 2 adjusted, 3 knots)"))
  ggsave(file.path(DIR_DOCX, paste0("fig_rcs_curves_", ep$suffix, ".pdf")),
         combined, width = 14, height = 16)
  cat(sprintf("  %s RCS 已输出\n", ep_label))

  p_nl_vec <- map2_dbl(EXPOSURES, EXPOSURE_LABELS, ~get_p_nl(.x, surv_var, out_var))
  rcs_summary[[ep$key]] <- data.frame(
    endpoint = ep_label, index = EXPOSURE_LABELS, p_nonlinear = round(p_nl_vec, 4))
  for (j in seq_along(EXPOSURE_LABELS))
    cat(sprintf("  %s %s P-nonlinear = %.3f\n", ep_label, EXPOSURE_LABELS[j], p_nl_vec[j]))
}

rcs_summary_df <- bind_rows(rcs_summary)
write.csv(rcs_summary_df, file.path(DIR_CSV, "rcs_pnonlinear_all_endpoints.csv"), row.names = FALSE)

# ====================================================================
# 08 亚组分析 (8 指标 × 院内+3个时间点) —— 对齐本文 Fig4 分层
# ====================================================================
cat(">>> Step 8: 亚组分析 (院内 + 30/90/365 天, ok_subgroup_mult_cox)\n")

# 分层因子 (对齐本文 Fig4): 性别 / 年龄<60 / BMI<30 / 冠心病 / 高血压 / 糖尿病 / 卒中史
#   注: 房颤已并入"心脏病"成分参与 CMM 计数, 不再单独作分层因子
subgroup_vars <- c("gender","age_grp","bmi_grp","chd","hypertension","dm","stroke")
subgroup_vars <- intersect(subgroup_vars, names(df))

num_covar <- covars_m3[vapply(df[covars_m3], is.numeric, logical(1))]

flatten_subgroup <- function(res, exp_var, exp_label, ep_label) {
  q_var <- paste0(exp_var, "_q")
  rows_out <- list()
  for (sv in names(res)) {
    for (lev in names(res[[sv]])) {
      fm <- tryCatch(res[[sv]][[lev]]$fit_mod, error = function(e) NULL)
      if (is.null(fm)) next
      est <- tryCatch({
        b <- coef(fm); v <- sqrt(diag(vcov(fm)))
        k <- grep(paste0("^", q_var), names(b))
        if (length(k) == 0) NULL else data.frame(
          endpoint = ep_label, index = exp_label, strata = sv, level = lev,
          n        = tryCatch(stats::nobs(fm), error = function(e) NA_integer_),
          contrast = gsub(paste0("^", q_var, "=?"), "", names(b)[k]),
          HR    = round(exp(b[k]), 3),
          CI_lo = round(exp(b[k] - 1.96 * v[k]), 3),
          CI_hi = round(exp(b[k] + 1.96 * v[k]), 3),
          p     = round(2 * pnorm(-abs(b[k] / v[k])), 4),
          row.names = NULL)
      }, error = function(e) NULL)
      if (!is.null(est)) rows_out[[length(rows_out) + 1]] <- est
    }
  }
  bind_rows(rows_out)
}

subgroup_obj  <- list()
subgroup_flat <- list()
int_all       <- list()

for (ep in ENDPOINTS) {
  surv_var <- ep$surv; out_var <- ep$out; ep_label <- ep$label

  for (i in seq_along(EXPOSURES)) {
    exp_var <- EXPOSURES[i]; exp_label <- EXPOSURE_LABELS[i]
    std_var <- paste0(exp_var, "_std"); q_var <- paste0(exp_var, "_q")
    key     <- paste0(ep$key, "_", exp_var)

    res <- strong::ok_subgroup_mult_cox(
      time          = surv_var,
      outcome       = out_var,
      col_name_gp   = q_var,
      covars        = covars_m3,
      subgroup_vars = subgroup_vars,
      df            = df,
      num_covar     = num_covar
    )
    subgroup_obj[[key]]  <- res
    subgroup_flat[[key]] <- flatten_subgroup(res, exp_var, exp_label, ep_label)

    # P-for-interaction (连续 per-SD × 分层) + BH-FDR; 通用提取交互项(不假设 "Yes" 水平)
    ip <- map_dfr(subgroup_vars, function(sv) {
      fml <- as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ", std_var, " * ", sv, " + ",
                               paste(setdiff(covars_m3, sv), collapse = " + ")))
      fit <- tryCatch(survival::coxph(fml, data = df), error = function(e) NULL)
      if (is.null(fit)) return(data.frame(strata = sv, p_int = NA_real_))
      co <- summary(fit)$coefficients
      it_idx <- grep(paste0("^", std_var, ":", sv), rownames(co))
      data.frame(strata = sv,
                 p_int = if (length(it_idx)) round(co[it_idx[1], 5], 4) else NA_real_)
    })
    ip$fdr_p    <- round(p.adjust(ip$p_int, method = "BH"), 4)
    ip$endpoint <- ep_label; ip$index <- exp_label
    int_all[[key]] <- ip

    cat(sprintf("  %s %s 亚组完成\n", ep_label, exp_label))
  }
}

subgroup_all <- bind_rows(subgroup_flat) %>%
  left_join(bind_rows(int_all), by = c("endpoint", "index", "strata"))
write.csv(subgroup_all, file.path(DIR_CSV, "subgroup_analysis_all_endpoints.csv"), row.names = FALSE)
save(subgroup_obj, file = file.path(DIR_LOAD, "subgroup_strong_objects.Rdata"))

# ====================================================================
# 09 NRI / IDI (8 指标 × 院内+3个时间点) —— 增量价值 (本文 Suppl Table 2)
#    插补后 df 已无缺失 -> 完整病例集即全队列。
# ====================================================================
cat(">>> Step 9: NRI / IDI (院内 + 30/90/365 天)\n")

df_model <- df   # 已无缺失 (Step 1b 插补)
cat(sprintf("  分析集: %d (插补后无缺失)\n", nrow(df_model)))

base_preds <- list()

nri_idi <- map_dfr(ENDPOINTS, function(ep) {
  out_var  <- ep$out; ep_label <- ep$label

  glm_base <- glm(as.formula(paste(out_var, "~", paste(covars_m3, collapse = " + "))),
                  data = df_model, family = binomial)
  p_base <- predict(glm_base, type = "response")
  base_preds[[ep$key]] <<- p_base
  y <- df_model[[out_var]]

  map_dfr(seq_along(EXPOSURES), function(i) {
    fml <- as.formula(paste(out_var, "~", EXPOSURES[i], "+", paste(covars_m3, collapse = " + ")))
    glm_new <- glm(fml, data = df_model, family = binomial)
    p_new <- predict(glm_new, type = "response")

    idi <- (mean(p_new[y==1]) - mean(p_new[y==0])) -
           (mean(p_base[y==1]) - mean(p_base[y==0]))

    roc_base <- pROC::roc(y, p_base, quiet = TRUE)
    roc_new  <- pROC::roc(y, p_new,  quiet = TRUE)
    delong   <- pROC::roc.test(roc_base, roc_new, method = "delong")

    data.frame(endpoint = ep_label, index = EXPOSURE_LABELS[i],
               AUC_base = round(auc(roc_base),3), AUC_basePlus = round(auc(roc_new),3),
               delong_p = round(delong$p.value, 4), IDI = round(idi, 4))
  })
})
write.csv(nri_idi, file.path(DIR_CSV, "table_nri_idi_all_endpoints.csv"), row.names = FALSE)
print(nri_idi)

# ====================================================================
# 10 ROC: 单指标(univariate) AUC + Youden 切点 + 指标间 DeLong 两两比较
#    对齐本文 Fig5 / Suppl Table 1 / Suppl Table 3 (均为"指标本身"的判别力)
# ====================================================================
cat(">>> Step 10: 单指标 ROC + Youden 切点 + DeLong 两两\n")

exp_colors <- grDevices::hcl.colors(length(EXPOSURES), palette = "Dark 3")

auc_uni  <- list()   # 各终点 × 指标 单指标 AUC
cutoffs  <- list()   # Youden 最佳切点 + 敏感度/特异度/PPV/NPV (Suppl Table 1)
roc_store_primary <- list()  # 主要终点(院内死亡)各指标 roc 对象, 供 DeLong

for (ep in ENDPOINTS) {
  out_var <- ep$out; ep_label <- ep$label
  y <- df_model[[out_var]]

  # SPISE 为敏感性指标(越高越好), 单指标 ROC 用 direction 自动处理;
  # pROC 默认按 AUC>=0.5 方向, 这里显式让其按"越大越危险"统一, SPISE 取负向。
  pdf(file.path(DIR_DOCX, paste0("fig_roc_univariate_", ep$suffix, ".pdf")), width = 8, height = 7)
  first <- TRUE
  for (i in seq_along(EXPOSURES)) {
    pred <- df[[EXPOSURES[i]]]
    roc_i <- pROC::roc(y, pred, quiet = TRUE, direction = "auto")
    if (ep$key == PRIMARY_KEY) roc_store_primary[[EXPOSURE_LABELS[i]]] <- roc_i
    if (first) { plot(roc_i, col = exp_colors[i], legacy.axes = TRUE,
                      main = paste0("ROC (univariate IR index) — ", ep_label)); first <- FALSE }
    else       plot(roc_i, col = exp_colors[i], add = TRUE)

    ci_auc <- as.numeric(pROC::ci.auc(roc_i))
    auc_uni[[paste0(ep$key,"_",EXPOSURES[i])]] <- data.frame(
      endpoint = ep_label, index = EXPOSURE_LABELS[i],
      AUC = round(ci_auc[2],3), AUC_lo = round(ci_auc[1],3), AUC_hi = round(ci_auc[3],3))

    co <- pROC::coords(roc_i, "best", best.method = "youden",
                       ret = c("threshold","sensitivity","specificity","ppv","npv"),
                       transpose = FALSE)
    cutoffs[[paste0(ep$key,"_",EXPOSURES[i])]] <- data.frame(
      endpoint = ep_label, index = EXPOSURE_LABELS[i],
      cutoff = round(co$threshold[1],3),
      youden = round(co$sensitivity[1] + co$specificity[1] - 1, 3),
      sensitivity = round(co$sensitivity[1],3), specificity = round(co$specificity[1],3),
      PPV = round(co$ppv[1],3), NPV = round(co$npv[1],3))
  }
  legend("bottomright", legend = paste0(EXPOSURE_LABELS,
         " (AUC=", sapply(EXPOSURES, function(e) round(as.numeric(auc(pROC::roc(y, df[[e]], quiet=TRUE))),3)), ")"),
         col = exp_colors, lwd = 2, cex = 0.75)
  dev.off()
  cat(sprintf("  %s 单指标 ROC 已输出\n", ep_label))
}

write.csv(bind_rows(auc_uni), file.path(DIR_CSV, "roc_auc_univariate.csv"), row.names = FALSE)
write.csv(bind_rows(cutoffs), file.path(DIR_CSV, "roc_cutoffs_youden.csv"), row.names = FALSE)

# 指标间 DeLong 两两比较 (院内死亡, 主要终点; 对齐本文 Suppl Table 3)
delong_pairs <- list()
labs <- names(roc_store_primary)
if (length(labs) >= 2) {
  cmb <- combn(labs, 2)
  for (k in seq_len(ncol(cmb))) {
    a <- cmb[1,k]; b <- cmb[2,k]
    tt <- tryCatch(pROC::roc.test(roc_store_primary[[a]], roc_store_primary[[b]], method = "delong"),
                   error = function(e) NULL)
    if (!is.null(tt))
      delong_pairs[[k]] <- data.frame(
        endpoint = PRIMARY_LBL, index_A = a, index_B = b,
        AUC_A = round(as.numeric(auc(roc_store_primary[[a]])),3),
        AUC_B = round(as.numeric(auc(roc_store_primary[[b]])),3),
        delong_p = round(tt$p.value, 4))
  }
}
write.csv(bind_rows(delong_pairs), file.path(DIR_CSV, paste0("roc_delong_pairwise_", PRIMARY_KEY, ".csv")), row.names = FALSE)
cat("  DeLong 两两比较已输出\n")
