######################################################################
# CMM 研究: 8 个胰岛素抵抗(IR)代用指标与重症 CMM 患者死亡风险 —— eICU-CRD 版
#   迁移自 MIMIC-IV 版 main_analysis.R; 复刻 Wu B. et al.
#   "Association of various insulin resistance surrogate markers with
#    mortality risk in critically ill patients with ischemic stroke"
#   (eICU-CRD, Cardiovasc Diabetol 2026) 的处理流程。
#
# 8 指标 (顺序对齐本文 Fig A-H): TyG, SPISE, TG-HDL, METS-IR, AIP, TyG-BMI, TyG-RC, CHG
# 关键口径 (与本文/SQL 一致):
#   - AIP = Log10(TG/HDL)         常用对数 (SQL 用 LOG=log10, 标准 Dobiášová 定义)
#   - CHG = Ln(TC×FBG×HDL/2)      HDL 在分子 (本文口径)
#   - SPISE 为敏感性指标, 方向相反 (HR<1 为危险)
#
# ★ 与 MIMIC 版的差异 (本脚本改动):
#   1. 数据源 = eICU-CRD (schema eicu_crd), 经 01-population-eicu.sql 提取。
#   2. ★ 唯一终点 = 院内死亡 (in-hospital mortality)。
#      eICU 无出院后随访(无死亡日期), 故 MIMIC 版的 30/90/365 天终点全部删除;
#      ICU 死亡次要终点亦已移除, 仅分析院内死亡。
#   3. Table 1 按"院内死亡 Survivor/Non-survivor"分层。
#   4. 严重度协变量 = APACHE IVa 总分 + APS (eICU 无 SOFA/SAPS-II 原生表; 最贴近原文 APS+APACHE)。
#   5. 其余流程(多重插补/KM/Cox/时依Cox/RCS/亚组/NRI-IDI/ROC)与 MIMIC 版一致。
#   6. ★审稿意见 3 修改 (2026-09): Cox 主分析与亚组分析改为 5 个插补集 Rubin 法则合并
#      (Step 6c/8b); 新增完整病例敏感性分析 (Step 6d); ML 流程明确为单一插补并附论证
#      (Step 11 注释)。KM/RCS/时依 Cox 等图形与敏感性流程仍用第 1 完成集。
######################################################################

rm(list = ls())
options(stringsAsFactors = FALSE, scipen = 999)

if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse, DBI, odbc, RPostgres,
  tableone, flextable,
  survival, survminer, rms,
  ggplot2, patchwork, forestploter,
  mice, nricens, gtsummary, pROC, glmnet
)

set.seed(20260531)   # 复现: 多重插补/随机过程

DIR_LOAD <- "./load"; DIR_CSV <- "./csv"; DIR_DOCX <- "./docx"; DIR_SQL <- "./sql"
for (d in c(DIR_LOAD, DIR_CSV, DIR_DOCX)) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

# ====== 八大暴露指标 (顺序对齐本文 Fig A-H) ======
EXPOSURES       <- c("tyg", "spise", "tg_hdl", "mets_ir", "aip", "tyg_bmi", "tyg_rc", "chg")
EXPOSURE_LABELS <- c("TyG", "SPISE", "TG-HDL", "METS-IR", "AIP", "TyG-BMI", "TyG-RC", "CHG")
# 注: SPISE 越高=胰岛素越敏感, 其 HR 预期 < 1, 方向与其余 7 个相反。

# ★ 终点定义 (eICU): 院内死亡 (唯一终点)
#   - 院内死亡(inhosp): outcome = inhosp_outcome, 生存时间 surv_time_inhosp 删失于出院。
#   * eICU 无出院后随访 -> 不设 30/90/365 天终点; 已移除 ICU 死亡次要终点。
PRIMARY_KEY <- "inhosp"                 # ← 唯一终点 = 院内死亡
PRIMARY_LBL <- "In-hospital"
PRIMARY_TP  <- "inhosp"

ENDPOINTS <- list(
  list(key = "inhosp", suffix = "inhosp",
       surv = "surv_time_inhosp", out = "inhosp_outcome",
       group = "inhosp_group", label = "In-hospital", brk = 20)
)
# ★ 仅保留院内死亡为唯一终点(已移除 ICU 死亡次要终点);
#   所有"按终点循环"的图/表/CSV 均仅针对院内死亡。

# ★ 防呆断言: 若 ENDPOINTS 因旧会话残留而混入 ICU(或任何非院内)终点, 立即报错中止,
#   避免"文件已改、内存未刷新"导致继续输出 ICU 结果。
stopifnot(
  length(ENDPOINTS) == 1L,
  identical(ENDPOINTS[[1]]$key, "inhosp"),
  !any(vapply(ENDPOINTS, function(e) e$key, character(1)) == "icu")
)

# ====================================================================
# 01 数据提取
# ====================================================================
cat(">>> Step 1: 数据提取 (eICU-CRD)\n")

# --- 方式 A: ODBC DSN(若已配置名为 "eicu" 的 DSN) ---
con <- DBI::dbConnect(odbc::odbc(), dsn = "eicu", database = "eicu",
                      uid = "postgres", pwd = "1",
                      server = "localhost", port = 5432)



sql_pop <- readr::read_file(file.path(DIR_SQL, "01-population-eicu.sql"))
population <- DBI::dbGetQuery(con, sql_pop)
DBI::dbDisconnect(con)

# ★ 防呆: 若 SQL/缓存仍残留 ICU 死亡相关列, 一律剔除, 确保下游绝不引用 ICU 终点。
#   (icu_order / icu_los_days 为"首次ICU/住满24h"纳入变量, 须保留; 故仅精确删下列三列。)
population <- population %>% select(-any_of(c("icu_outcome", "surv_time_icu", "icu_group", "unitdischargestatus")))


# R 端队列筛选 —— 仅按"暴露相关实验室"做完全病例 (定义暴露/分位, 不可插补);
#   其余协变量留待 Step 1b 多重插补。
cohort_raw <- population %>%
  filter(
    cm_count >= 2,                  # CMM ≥2项 (★ cm_count 仅用于入组筛选; 已从 Table 1 报告变量移除以对齐 MIMIC)
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
    # ★ eICU: 用院内死亡生存时间非缺(原 MIMIC 版此处为 surv_time_365)
    !is.na(surv_time_inhosp)
  )

cat(sprintf("  全人群: %d, 暴露完整队列: %d\n", nrow(population), nrow(cohort_raw)))
save(population, cohort_raw, file = file.path(DIR_LOAD, "raw_data.Rdata"))

# ====================================================================
# 01a 纳入 vs 排除患者比较 (审稿意见 #4: 选择偏倚评估)
#   "otherwise eligible" = 满足全部入组标准、但因暴露定义实验室/体测不完整
#   而无法计算 8 项指标者。仅比较两组本身可得的基线变量与院内死亡率;
#   血脂/血糖等缺失变量不参与比较(缺失本身即选择机制)。
# ====================================================================
cat(">>> Step 1a: 纳入 vs 排除患者特征比较\n")

pop_eligible <- population %>%
  filter(cm_count >= 2, age >= 18, icu_order == 1, icu_los_days >= 1,
         severe_comorbid == 0, !is.na(surv_time_inhosp))

inc_cmp <- pop_eligible %>%
  filter(!is.na(triglycerides), !is.na(glucose), !is.na(hdl), !is.na(cholesterol),
         !is.na(ldl), !is.na(height_cm), !is.na(weight_kg),
         !is.na(tyg), !is.na(spise), !is.na(tg_hdl), !is.na(mets_ir),
         !is.na(aip), !is.na(tyg_bmi), !is.na(tyg_rc), !is.na(chg))
exc_cmp <- pop_eligible %>%
  filter(is.na(triglycerides) | is.na(glucose) | is.na(hdl) | is.na(cholesterol) |
         is.na(ldl) | is.na(height_cm) | is.na(weight_kg) |
         is.na(tyg) | is.na(spise) | is.na(tg_hdl) | is.na(mets_ir) |
         is.na(aip) | is.na(tyg_bmi) | is.na(tyg_rc) | is.na(chg))

stopifnot(nrow(inc_cmp) + nrow(exc_cmp) == nrow(pop_eligible),
          nrow(inc_cmp) == nrow(cohort_raw))   # 防呆: 口径须与主队列一致

cat(sprintf("  其他入组条件合格: %d; 纳入(暴露完整): %d (%.1f%%); 因暴露不全排除: %d (%.1f%%)\n",
            nrow(pop_eligible), nrow(inc_cmp), 100 * nrow(inc_cmp) / nrow(pop_eligible),
            nrow(exc_cmp), 100 * nrow(exc_cmp) / nrow(pop_eligible)))

cmp_cont <- intersect(c("age", "sofa", "saps_ii", "oasis"), names(pop_eligible))
cmp_cat  <- intersect(c("gender", "dm", "hypertension", "dyslipidemia", "chd", "hf",
                        "af", "stroke", "ckd", "pad", "ventilation", "vasopressor",
                        "inhosp_outcome"), names(pop_eligible))

rows_cont <- lapply(cmp_cont, function(v) {
  x1 <- inc_cmp[[v]]; x0 <- exc_cmp[[v]]
  p  <- tryCatch(wilcox.test(x1, x0)$p.value, error = function(e) NA_real_)
  m1 <- mean(x1, na.rm = TRUE); m0 <- mean(x0, na.rm = TRUE)
  s1 <- sd(x1, na.rm = TRUE);   s0 <- sd(x0, na.rm = TRUE)
  smd <- (m1 - m0) / sqrt((s1^2 + s0^2) / 2)
  data.frame(variable = v,
             included = sprintf("%.1f (%.1f)", m1, s1),
             excluded = sprintf("%.1f (%.1f)", m0, s0),
             p_value = signif(p, 3), smd = round(smd, 3),
             stringsAsFactors = FALSE)
})

rows_cat <- lapply(cmp_cat, function(v) {
  x1 <- inc_cmp[[v]]; x0 <- exc_cmp[[v]]
  lev <- if (v == "gender") "M" else 1          # gender 原始编码 F/M; 其余为 0/1
  t1 <- sum(x1 == lev, na.rm = TRUE); n1 <- sum(!is.na(x1))
  t0 <- sum(x0 == lev, na.rm = TRUE); n0 <- sum(!is.na(x0))
  p1 <- t1 / n1; p0 <- t0 / n0
  p  <- tryCatch(chisq.test(matrix(c(t1, n1 - t1, t0, n0 - t0), nrow = 2))$p.value,
                 error = function(e) NA_real_)
  pp <- (t1 + t0) / (n1 + n0)
  smd <- if (pp > 0 & pp < 1) (p1 - p0) / sqrt(pp * (1 - pp)) else NA_real_
  data.frame(variable = if (v == "gender") "gender (Male)" else v,
             included = sprintf("%d (%.1f%%)", t1, 100 * p1),
             excluded = sprintf("%d (%.1f%%)", t0, 100 * p0),
             p_value = signif(p, 3), smd = round(smd, 3),
             stringsAsFactors = FALSE)
})

cmp_tab <- rbind(do.call(rbind, rows_cont), do.call(rbind, rows_cat))
write.csv(cmp_tab, file.path(DIR_CSV, "included_vs_excluded_comparison.csv"),
          row.names = FALSE)
print(cmp_tab)

# ====================================================================
# 01b 多重插补 (mice PMM) —— 替代原"完全病例删除"
#   暴露/结局/生存时间/人口学已完整 -> mice 自动不插补 (method="");
#   仅对缺失率可接受(<=20%)的协变量做 PMM; 缺失>20% 的协变量按本文做法剔除。
# ====================================================================
cat(">>> Step 1b: 多重插补 (mice PMM)\n")

# 候选协变量/Table1 变量 (★ 严重度改为 SOFA/SAPS-II/OASIS, 口径=mimic-code 官方实现移植到 eICU)
impute_candidates <- c(
  "age","gender","bmi","height_cm","weight_kg",
  "heart_rate","sbp","dbp","map","resp_rate","temperature","spo2",
  "hemoglobin","platelet","rbc","wbc","creatinine","bun","glucose","triglycerides",
  "cholesterol","hdl","ldl","lactate","albumin","bilirubin",   # hba1c 已剔除: eICU-CRD lab 表无 HbA1c (100% 缺失, 经 labname 全名核查确认无任何拼写变体)
  "sodium","potassium","chloride","calcium","bicarbonate","inr","pt","ptt",
  "sofa","saps_ii","oasis",
  "dm","hypertension","dyslipidemia","chd","hf","af","stroke","ckd","pad",
  "ventilation","vasopressor","aspirin","statin","antidiabetic","antihypertensive","anticoagulant",  # ★ 新增 anticoagulant(对齐 MIMIC)
  EXPOSURES
)
impute_candidates <- intersect(impute_candidates, names(cohort_raw))

# 缺失率评估 + 剔除高缺失(>20%)协变量
miss_rate <- sapply(cohort_raw[impute_candidates], function(x) mean(is.na(x)))
drop_hi   <- names(miss_rate)[miss_rate > 0.20]
drop_hi   <- setdiff(drop_hi, EXPOSURES)   # 暴露不剔除(已完整)
miss_tab  <- data.frame(var = names(miss_rate),
                        miss_pct = round(100 * miss_rate, 1),
                        dropped = names(miss_rate) %in% drop_hi)
write.csv(miss_tab, file.path(DIR_CSV, "missingness_report.csv"), row.names = FALSE)
cat("  --- 缺失率(剔除>20%) ---\n"); print(miss_tab[order(-miss_tab$miss_pct), ], row.names = FALSE)

vars_keep  <- setdiff(impute_candidates, drop_hi)
id_time    <- cohort_raw %>% select(any_of(c("subject_id","hadm_id","stay_id","intime","outtime")))

dat_mice <- cohort_raw[, vars_keep, drop = FALSE]
# 二分类协变量转因子, 利于 mice 选 logreg
bin_vars <- intersect(c("dm","hypertension","dyslipidemia","chd","hf","af","stroke","ckd","pad",
                        "ventilation","vasopressor","aspirin","statin","antidiabetic",
                        "antihypertensive","anticoagulant"), names(dat_mice))
dat_mice[bin_vars] <- lapply(dat_mice[bin_vars], factor)
if ("gender" %in% names(dat_mice)) dat_mice$gender <- factor(dat_mice$gender)

# 方法向量: 数值缺失列用 PMM; 二分类用默认 logreg; 完整列自动不插补 ("")
ini  <- mice(dat_mice, maxit = 0, printFlag = FALSE)
meth <- ini$method
num_cols <- names(dat_mice)[vapply(dat_mice, is.numeric, logical(1))]
sel  <- num_cols[meth[num_cols] != ""]
meth[sel] <- "pmm"
imp <- mice(dat_mice, m = 5, maxit = 50, method = meth,
            predictorMatrix = ini$predictorMatrix, printFlag = FALSE, seed = 20260531)
dat_done <- mice::complete(imp, 1)
dat_done[bin_vars] <- lapply(dat_done[bin_vars], function(x) as.integer(as.character(x)))
if ("gender" %in% names(dat_done)) dat_done$gender <- as.character(dat_done$gender)

# 合并: 完成的协变量 + ID/时间 + (未进入 mice 的)结局/生存等列
extra_cols <- setdiff(names(cohort_raw), c(names(dat_done), names(id_time)))
cohort_imp <- bind_cols(id_time, dat_done, cohort_raw[, extra_cols, drop = FALSE])
saveRDS(imp, file.path(DIR_LOAD, "mice_imp.rds"))
cat(sprintf("  插补后队列: %d 行, %d 列\n", nrow(cohort_imp), ncol(cohort_imp)))

# ====================================================================
# 01c 全部 5 个插补完成集 (★审稿意见 3: Rubin 法则合并所需)
#   - 原流程仅用第 1 个完成集 (complete(imp, 1)) 做全部下游分析;
#     现保留该口径用于 KM/RCS/时依 Cox/ML 等流程,
#     另构建 5 个完成集列表, 供 Step 6c (主 Cox) 与 Step 8b (亚组) 按 Rubin 法则合并。
#   - 暴露/结局/生存时间完整未插补 -> 各完成集间仅"被插补协变量"不同,
#     四分位切点与标准化暴露在各完成集间完全一致。
# ====================================================================
imp_data_list <- lapply(seq_len(imp$m), function(i) {
  d <- mice::complete(imp, i)
  d[bin_vars] <- lapply(d[bin_vars], function(x) as.integer(as.character(x)))
  if ("gender" %in% names(d)) d$gender <- as.character(d$gender)
  bind_cols(id_time, d, cohort_raw[, extra_cols, drop = FALSE])
})
saveRDS(imp_data_list, file.path(DIR_LOAD, "mice_completed_list.rds"))
cat(sprintf("  已构建 %d 个插补完成集 (供 Rubin 法则合并)\n", length(imp_data_list)))

# ====================================================================
# 02 流程图
# ====================================================================
cat(">>> Step 2: 流程图\n")

fc <- tribble(
  ~step, ~n,
  "CMM队列(SQL已筛≥2项CMD, eICU)",  nrow(population),
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
    af              = factor(af, 0:1, c("No","Yes")),
    stroke          = factor(stroke, 0:1, c("No","Yes")),
    ckd             = factor(ckd, 0:1, c("No","Yes")),
    pad             = factor(pad, 0:1, c("No","Yes")),
    ventilation     = factor(ventilation, 0:1, c("No","Yes")),
    vasopressor     = factor(vasopressor, 0:1, c("No","Yes")),
    aspirin         = factor(aspirin, 0:1, c("No","Yes")),
    statin          = factor(statin, 0:1, c("No","Yes")),
    antidiabetic    = factor(antidiabetic, 0:1, c("No","Yes")),
    antihypertensive= factor(antihypertensive, 0:1, c("No","Yes")),
    anticoagulant   = factor(anticoagulant, 0:1, c("No","Yes")),   # ★ 新增(对齐 MIMIC)
    # ★ race: 由 eICU ethnicity 重编码为 White/Black/Other 三分类, 对齐 MIMIC race
    #   Caucasian->White; African American->Black; 其余(Hispanic/Asian/Native American/Other-Unknown/缺失)->Other
    race = factor(dplyr::case_when(
             grepl("caucasian|white",  ethnicity, ignore.case = TRUE) ~ "White",
             grepl("african|black",    ethnicity, ignore.case = TRUE) ~ "Black",
             TRUE                                                     ~ "Other"),
           levels = c("White","Black","Other")),
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

# 各终点 生存/非生存 分组因子 (院内)
for (ep in ENDPOINTS) df[[ep$group]] <- factor(df[[ep$out]], 0:1, c("Survivor","Non-survivor"))

# Table 1 变量 (★ 对齐 MIMIC 版, 共 60 项)
#   - 移除 cm_count(MIMIC 无此变量; cm_count 仅用于 CMM≥2 入组筛选, 不作基线报告)
#   - 移除 po2/pco2/ph 血气(eICU 动脉血气覆盖不足, 已从两库分析中剔除)
#     ★ 注意: MIMIC 侧若仍保留 po2/pco2/ph, 需同样删除这 3 项才能与本表严格一致
#   - 新增 10 项(MIMIC 有而 eICU 原 Table1 缺): race / temperature / spo2 / rbc /
#     bilirubin / inr / pt / ptt / antihypertensive / anticoagulant
#   - 顺序逐行对齐 MIMIC Table 1, 便于双队列行对行比对
tbl1_vars <- c("age","gender","race","bmi","heart_rate","sbp","dbp","map","resp_rate","temperature","spo2",
               "dm","hypertension","dyslipidemia","chd","hf","stroke","ckd","pad","af",
               "hemoglobin","platelet","rbc","wbc","creatinine","bun","glucose","triglycerides",
               "cholesterol","hdl","ldl","lactate","albumin","bilirubin",
               "sodium","potassium","chloride","calcium","bicarbonate","inr","pt","ptt",
               "tyg","spise","tg_hdl","mets_ir","aip","tyg_bmi","tyg_rc","chg",
               "sofa","saps_ii","oasis",
               "ventilation","vasopressor","aspirin","statin","antidiabetic","antihypertensive","anticoagulant")
tbl1_vars <- intersect(tbl1_vars, names(df))   # 防高缺失被剔除的变量报错

cat_vars <- c("gender","race","dm","hypertension","dyslipidemia","chd","hf","af","stroke","ckd","pad",
              "ventilation","vasopressor","aspirin","statin","antidiabetic","antihypertensive","anticoagulant")
cat_vars <- intersect(cat_vars, names(df))

# ====== 调整模型 (eICU, 镜像本文 Model 1/2/3; 输出标签见 Step 6 run_cox) ======
#   Model 1 = Crude(仅暴露);  Model 2 = age+sex+BMI;  Model 3 = ★变量筛选结果(本次改动)。
#   ★ 协变量选择方法: 与 MIMIC 版 main_analysis.R 的 Step 3b 一致, ★本次改为 LASSO + VIF<5
#     (已【不再】做 单因素 Cox / 逐步 Cox)。eICU 的 Model 3 等价于 MIMIC 的 covars_m3 / 本文 Adjusted II。
#     两脚本的 covars_m2 / covars_m3 变量名与角色完全一致, 仅 Step 6 打印的模型序号标签不同
#     (MIMIC: Crude/Model 1/Model 2;  eICU: Model 1/Model 2/Model 3), 本次不改动该标签。
covars_m2 <- c("age","gender","bmi")     # → 输出 "Model 2"(人口学): age/sex/BMI
# Model 3 不再手列预设协变量, 一律由下方 Step 3b 变量筛选产生; covars_m3 在 Step 3b 末尾赋值。

# ====================================================================
# 03b 协变量筛选 (复刻 MIMIC 版; ★本次改为 LASSO + VIF<5; 已【不再】做 单因素 Cox / 逐步 Cox)
#   流程 (强制人口学基线全程锁定, 不参与 LASSO 惩罚):
#     (0) 强制保留 人口学基线 (age/gender/bmi; race 按需求不纳入任何模型); 维持 M3 ⊇ M2。
#     (1) LASSO Cox (glmnet, family="cox", alpha=1): 候选 = 强制人口学 + cand_pool
#         (★候选池含 严重度评分 sofa/saps_ii/oasis 及"组成各评分的变量" + 其余临床协变量);
#         强制人口学 penalty.factor=0 永不剔除, 其余 penalty.factor=1;
#         ★ LASSO 前对"连续候选变量"标准化(均值0/标准差1), 因子哑变量不标准化(glmnet standardize=FALSE);
#         cv.glmnet 交叉验证选 lambda(LASSO_LAMBDA: "min" CV 偏差最小 / "1se" 更稀疏), 取非零系数变量为入选。
#     (2) VIF < 5 迭代去共线 (强制项不剔) 收尾; 评分与其组成变量天然高度共线, 由此一步消解。
#   与上一版差异: 删除"单因素 Cox P<0.05"与"逐步 Cox(AIC)"两步, 改用 LASSO 选变量(VIF<5 不变)。
#   ★ eICU 仅"院内死亡"单终点 -> LASSO 即以该终点为生存结局(LASSO_ENDPOINT 指向唯一的 inhosp)。
#   ★ 暴露成分(TG/Glu/HDL/TC/LDL): 默认从候选池剔除以防对 8 个暴露过度校正(DROP_EXPOSURE_COMPONENTS=TRUE);
#     如需逐字复刻本文候选池(含血糖)设 FALSE。
#   ★ 严重度评分(sofa/saps_ii/oasis, eICU 已按 mimic-code 口径在 SQL 自算): 整体纳入 LASSO 候选池(INCLUDE_SEVERITY_SCORES=TRUE);
#     ★本次改动: 其"参与评分计算"的组成变量也纳入候选池(DROP_SEVERITY_COMPONENTS=FALSE), 与评分一同经
#     LASSO->VIF<5 取舍; 手动清单关闭(USE_MANUAL_COVARS_M3=FALSE), 使该筛选结果即为 Model 3。
# ====================================================================
cat(">>> Step 3b: 协变量筛选 (LASSO Cox -> VIF<5; 强制人口学全程锁定, 不参与惩罚)\n")

VIF_THRESHOLD            <- 5       # 本文: VIF < 5 (LASSO 后去共线)
DROP_EXPOSURE_COMPONENTS <- TRUE    # 候选池是否剔除暴露计算成分(FALSE 可逐字复刻本文)

INCLUDE_SEVERITY_SCORES  <- T    # ★本次需求: TRUE=三评分(sofa/saps_ii/oasis)整体纳入 LASSO 候选池, 由 LASSO->VIF 取舍
SEVERITY_SCORES_TO_USE   <- c("sofa", "saps_ii", "oasis")  # 纳入哪"几种"评分(可按需删减)
SEVERITY_SCORE_MODE      <- "candidate" # "force"=强制锁入(LASSO 不罚 + VIF 不剔);
                                    # "candidate"=仅放入候选池, 由 LASSO -> VIF 自行取舍(可自动消解评分间/评分-成分共线)
DROP_SEVERITY_COMPONENTS <- T   # ★本次需求: FALSE = 将"组成各评分的变量"(参与评分计算的生理/化验/干预变量)纳入候选池,
                                    #   与评分一同进入 LASSO 竞争, 由 LASSO 选取 + VIF<5 消解共线;
                                    #   TRUE = 把组成变量剔出候选池(防与评分双重计分)。
                                    # 注: FALSE 时评分(SOFA/SAPS II/OASIS)与其组成变量同处候选池, 二者天然高度共线 ->
                                    #     LASSO 倾向在共线组内择优保留, 残余共线再由 VIF<5 迭代剔除, 这正是"LASSO+VIF"的预期行为。
                                    # ★★ 注: 本配置已配合 Step 03c 的 USE_MANUAL_COVARS_M3=FALSE, 故该筛选结果即为最终 Model 3。

# --- LASSO 参数 (替代旧 单因素/逐步 参数) ---
LASSO_ENDPOINT       <- ENDPOINTS[[1]]  # LASSO 所用单一生存终点(eICU 唯一终点=院内死亡)
LASSO_LAMBDA         <- "min"           # 选 lambda: "min"(CV 偏差最小, 默认) / "1se"(更稀疏)
                                        #   注: 评分+组成变量高度共线、单终点事件有限时 "1se" 易收缩到仅强制人口学(空模型), 故默认 "min"
LASSO_NFOLDS         <- 10              # cv.glmnet 交叉验证折数
STANDARDIZE_CONTINUOUS <- TRUE          # ★ LASSO 前标准化连续候选变量(均值0/标准差1); 二分类/哑变量不标准化

# --- (1) 强制保留: 人口学基线 = 本文 Model I(age/sex/bmi); race 不纳入任何模型(按需求) ---
force_keep_demo  <- intersect(covars_m2, names(df))             # age/gender/bmi (race 已排除); 维持 M3 ⊇ M2
# 若以 "force" 方式纳入严重度评分, 在此把选用评分锁入强制保留集 -> LASSO 不罚 + VIF 不剔;
# mode="candidate" 时此处为空, 评分改由候选池经 LASSO/VIF 自行取舍。
force_keep_prior <- if (INCLUDE_SEVERITY_SCORES && SEVERITY_SCORE_MODE == "force")
  intersect(SEVERITY_SCORES_TO_USE, names(df)) else character(0)

# --- 暴露计算成分: 据开关决定是否从候选池剔除 ---
exposure_components <- if (DROP_EXPOSURE_COMPONENTS)
  c("triglycerides", "glucose", "hdl", "cholesterol", "ldl") else character(0)

# --- 严重度评分(纳入 Model 3 候选) + 其"计算成分/组成变量"处理(★本次改动: 默认纳入候选池) ---
ALL_SEVERITY_SCORES <- c("sofa", "saps_ii", "oasis")
if (INCLUDE_SEVERITY_SCORES) {
  severity_scores_use <- intersect(SEVERITY_SCORES_TO_USE, names(df))   # 实际纳入 Model 3 候选的评分
  # 「评分计算成分/组成变量」: 参与 SOFA / SAPS II / OASIS 评分计算的 生理/化验/干预 变量。
  #   ★本次改动: 默认(DROP_SEVERITY_COMPONENTS=FALSE)将这些组成变量保留在候选池, 与评分一同进入
  #     LASSO -> VIF<5 筛选; 二者天然共线, 由 LASSO 择优保留 + VIF<5 消解(可能剔评分或剔部分组成变量)。
  #   (若设 DROP_SEVERITY_COMPONENTS=TRUE 则改回旧法: 把组成变量剔出候选池, 防"评分 + 原始变量"双重计分。)
  #   逐分量来源(mimic-code 派生口径; GCS/尿量/术前LOS 不在本队列协变量池, 无需列出):
  #     SOFA    : po2 + ventilation(呼吸) / platelet(凝血) / bilirubin(肝) / map + vasopressor(循环) / creatinine(肾)
  #     SAPS II : heart_rate sbp temperature / po2 ventilation / bun wbc potassium sodium bicarbonate bilirubin
  #     OASIS   : heart_rate map resp_rate temperature ventilation
  #   ※ age 虽为 SAPS II/OASIS 分量, 但属核心人口学混杂(本文 Model I 强制项), 全程保留(维持 M3 ⊇ M2)。
  #   ※ dbp / rbc / chloride / calcium / lactate / inr / pt / ptt 不参与上述 3 评分计算, 仍留在候选池。
  #   ★ eICU 已剔除 po2 血气(覆盖不足), 此处列出 po2 仅作占位, setdiff/intersect 自动无副作用。
  ALL_SCORE_COMPONENTS <- c(
    "heart_rate", "sbp", "map", "resp_rate", "temperature", "spo2",   # 生命体征 / 氧合
    "po2",                                              # 血气 / 酸碱(eICU 已无, 占位)
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
  # 完全不纳入评分, 也不剔成分
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
  exposure_components, severity_scores, score_component_vars,   # ← score_component_vars 仅在 DROP_SEVERITY_COMPONENTS=TRUE 时非空; 本次默认 FALSE -> 为空 -> 组成变量保留在候选池
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
#   family="cox" 以 LASSO_ENDPOINT(eICU 唯一终点=院内死亡)为生存结局, cv.glmnet 交叉验证选 lambda。
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

# 生存结局(LASSO_ENDPOINT, eICU 唯一终点=院内死亡)
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

# --- 落地: Model 3 协变量 = Step 3b 筛选结果(★LASSO -> VIF<5);
#     ★本次改动: 严重度评分 + 其组成变量 一同作为候选, 共同经 LASSO + VIF<5 取舍(组成变量不再预先剔出) ---
covars_m3 <- intersect(covars_sel, names(df))
# 显式兜底(force 模式): 确保选用的严重度评分一定保留在 Model 3 (force 模式下评分已并入 force_keep_prior, LASSO 不罚)。
if (INCLUDE_SEVERITY_SCORES && SEVERITY_SCORE_MODE == "force")
  covars_m3 <- union(covars_m3, intersect(severity_scores_use, names(df)))
.sev_in_m3 <- intersect(severity_scores_use, covars_m3)
if (INCLUDE_SEVERITY_SCORES) {
  .comp_note <- if (DROP_SEVERITY_COMPONENTS) {
    "评分计算成分已剔出候选池防双重计分"
  } else {
    "评分组成变量已纳入候选池, 与评分一同经 LASSO->VIF<5 筛选"
  }
  cat(sprintf("  >>> Model 3 协变量(自动筛选) = 筛选结果(含评分{%s}, 方式=%s); %s\n",
              paste(.sev_in_m3, collapse = "/"), SEVERITY_SCORE_MODE, .comp_note))
} else {
  cat("  >>> Model 3 协变量(自动筛选) = 筛选结果(LASSO -> VIF<5; 严重度评分已排除候选池)\n")
}

# ====================================================================
# 03c   手动指定 Model 3 协变量 (可选; 覆盖上方 Step 3b 自动筛选结果)
#   - 上方 Step 3b 的 LASSO + VIF<5 筛选始终照常运行并输出审计报表
#     (covariate_selection_report.csv + 控制台)。USE_MANUAL_COVARS_M3=TRUE 时,
#     "最终用于建模"的 covars_m3 改用下面这份显式清单(使发表模型固定、可复现)。
#   - ★当前配置: USE_MANUAL_COVARS_M3 = FALSE -> Model 3 由 Step 3b LASSO+VIF<5 筛选决定(不使用下方手动清单)。
#     下方 MANUAL_COVARS_M3(22 项固定清单)保留备查但当前【不生效】; 仅当此开关改回 TRUE 才覆盖筛选结果。
#     上方 LASSO+VIF 仍照常运行并输出审计报表(covariate_selection_report.csv), 只是不决定最终建模集。
#     如需改回"固定手动清单"发表模型, 把 USE_MANUAL_COVARS_M3 改回 TRUE 即可(一行, 无需删代码)。
#   - intersect 兜底: 清单中若有变量因高缺失(>20%)被剔出 df, 自动跳过并提示。
#   ※ 注: sofa/saps_ii/oasis 三评分高度共线(LASSO/VIF 通常只留 1 个)。手动清单按需求强制三者
#     并入, 仅作"调整用"协变量 —— 暴露(8 指标)的 HR 不受影响, 但三评分各自的 HR 因共线会不稳。
# ====================================================================
USE_MANUAL_COVARS_M3 <- FALSE    # ★ FALSE = Model 3 由 Step 3b LASSO+VIF<5 筛选决定; 不用固定手动清单(本次需求)
MANUAL_COVARS_M3 <- c("age", "gender", "bmi", "resp_rate", "temperature", "wbc", "bun", "bicarbonate",
                      "sofa", "saps_ii", "oasis", "chd", "hf", "stroke", "ventilation", "antihypertensive",
                      "dbp", "calcium", "spo2", "ptt", "hemoglobin", "sbp")
if (USE_MANUAL_COVARS_M3) {
  .m3_missing <- setdiff(MANUAL_COVARS_M3, names(df))
  if (length(.m3_missing))
    cat(sprintf("  [手动 Model 3] 以下指定变量不在 df 中(可能高缺失被剔), 已跳过: %s\n",
                paste(.m3_missing, collapse = ", ")))
  covars_m3  <- intersect(MANUAL_COVARS_M3, names(df))      # 保持给定顺序
  .sev_in_m3 <- intersect(severity_scores_use, covars_m3)   # 同步刷新, 供下方报表 / 日志
  cat(sprintf("  >>> ★ Model 3 改用手动清单(%d 项, 覆盖自动筛选): %s\n",
              length(covars_m3), paste(covars_m3, collapse = ", ")))
} else {
  cat("  >>> ★ Model 3 采用 Step 3b 自动筛选结果(USE_MANUAL_COVARS_M3=FALSE; 评分组成变量经 LASSO->VIF<5 取舍)\n")
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
cat("  最终 covars_m3 (Model 3, 过 VIF<5): ", paste(covars_m3, collapse = ", "), "\n")
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
    .comp_final   <- intersect(ALL_SCORE_COMPONENTS, covars_m3)        # 最终进入 Model 3 的组成变量(过 VIF<5)
    cat("  ★评分组成变量(本次纳入 LASSO 候选池): ", paste(.comp_in_pool, collapse = ", "), "\n")
    cat("     - LASSO 入选(非零系数): ", paste(.comp_lasso, collapse = ", "), "\n")
    cat("     - 最终入 Model 3  : ", paste(.comp_final, collapse = ", "),
        if (USE_MANUAL_COVARS_M3) "  [注: 当前用手动清单, 此项受手动清单限制]" else "  [过 VIF<5]", "\n")
    .comp_missing <- setdiff(.comp_in_df, .comp_in_pool)
    if (length(.comp_missing))
      cat("     - (组成变量在 df 中但未入池, 多因高缺失被剔/或在排除集): ",
          paste(.comp_missing, collapse = ", "), "\n")
  }
  if (length(severity_scores_exclude))
    cat("  (未点名选用的评分, 仍排除出候选池): ", paste(severity_scores_exclude, collapse = ", "), "\n")
} else {
  cat("  (严重度评分已排除候选池, 未进入 Model 3: ", paste(severity_scores, collapse = ", "), ")\n")
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
  col_name_gp = "inhosp_group",         # ← 主要终点(院内死亡)分层
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
# 05b Model 3 协变量调整后的生存曲线 (直接调整 / G-computation)
#   目的: 与 Step 5 未调整 KM 对应, 给出"调整 Model 3 协变量后"的四分位生存曲线。
#   方法 (direct adjusted survival = corrected group prognosis / G-formula):
#     1) 对每个指标拟合 Cox: Surv(time, event) ~ <指标>_q + covars_m3
#        (与 Step 6 的四分位 Model 3 m3_q 完全同口径)。
#     2) 对每个四分位 g(Q1-Q4): 把"标准化总体"的协变量整体设为"假设全部位于 g",
#        survfit() 预测每个个体的生存曲线, 再在该总体上求平均 -> 标准化生存曲线。
#     ★ 4 条曲线共用同一(全队列)协变量分布, 组间差异仅反映四分位本身,
#       即"协变量已被调整/标准化"; 区别于 Step 5 仅按四分位分层的未调整 KM。
#   注: 直接调整曲线的逐点 CI 需对"个体生存均值"做方差传播(个体间相关), 较繁琐;
#       此处仅绘点估计(发表常规做法)。如需 CI 可对 Cox 做 bootstrap 重抽后取分位。
# ====================================================================
cat(">>> Step 5b: Model 3 协变量调整后的生存曲线 (direct adjusted)\n")

# Q1→Q4 配色 (原 Step 5 未调整 KM 已整段删除, 其配色定义移至此处供调整后曲线复用)
km_palette <- c("#2E9FDF", "#00BA38", "#F8766D", "#E76BF3")

rhs_m3_adj <- paste(covars_m3, collapse = " + ")

# ★ 调整后曲线 X 轴上限 / 刻度间隔 (单位 = surv_time_inhosp 的单位, 名义为"天")
#   背景: 院内死亡随访删失于出院, x 轴若沿用"观测到的最长随访时间"(原 x_max=max(time)),
#   会被极端长住院的离群值撑到 ~380(天) —— eICU-CRD 已知存在异常巨大的出院偏移量。
#   后果: 绝大多数院内死亡集中在最初 1-2 周, 被压扁在最左侧; 右尾仅基于个位数样本极不稳定。
#   做法: 截断到临床有意义的窗口(院内死亡常用 28/30 天; 亦可 60/90)。
#     ADJ_KM_XMAX = 数值 -> 在该处截断 X 轴(用 coord_cartesian 裁剪, 不丢数据, 阶梯线仍正确画到边缘);
#     ADJ_KM_XMAX = NA   -> 沿用观测最大随访时间(旧行为, 会出现 ~380)。
#   ⚠ 截断值与 surv_time_inhosp 同单位; 若回查 SQL 发现单位其实是小时/分钟(未做 /1440), 请先修 SQL。
ADJ_KM_XMAX <- 30      # 院内死亡 X 轴上限(天); 设 NA 用观测最大值(旧的 ~380 行为)
ADJ_KM_XBRK <- 5       # X 轴刻度间隔(天); 设 NA 则沿用各终点 ep$brk(=20, 适配 380 大尺度)

# ★ 调整后曲线的 P 值标注 (★注意: 这是 G-computation 标准化曲线, 不是真 KM, 故【不用 log-rank】;
#   改取与曲线同源、同 Step 6 口径的 Cox 检验):
#     "overall" = 四分位整体差异(似然比检验, 3df, 已校正 covars_m3) —— log-rank 的"调整版"类比
#     "trend"   = 四分位序号(1-4)作连续项的 Wald P (= Step 6 的 P-trend, 第782-784行)
#     "both"    = 两者都标(默认);  "none" = 不标
ADJ_KM_PVAL <- "both"
fmt_p <- function(p) if (is.na(p)) "= NA" else if (p < 0.001) "< 0.001" else
  paste0("= ", formatC(p, format = "f", digits = 3))

for (ep in ENDPOINTS) {
  surv_var <- ep$surv; out_var <- ep$out; brk <- ep$brk

  adj_plots <- map2(EXPOSURES, EXPOSURE_LABELS, function(exp_var, lab) {
    q_var    <- paste0(exp_var, "_q")
    q_levels <- levels(df[[q_var]])                       # "Q1".."Q4"

    # (1) Model 3 Cox: 四分位 + covars_m3 (与 Step 6 m3_q 同口径)
    fml <- as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ",
                             q_var, " + ", rhs_m3_adj))
    fit <- survival::coxph(fml, data = df, x = TRUE)

    # 标准化总体: 在模型变量上完整的行 (对 4 条曲线统一, 保证仅四分位不同)
    std_pop <- df[stats::complete.cases(
      df[, c(surv_var, out_var, q_var, covars_m3)]), , drop = FALSE]

    # (2) G-computation: 把全体反事实置于四分位 g -> 预测 -> 全队列平均
    adj_df <- map_dfr(q_levels, function(g) {
      nd <- std_pop
      nd[[q_var]] <- factor(g, levels = q_levels)         # 反事实: 全员置于 g
      sf <- survival::survfit(fit, newdata = nd)
      data.frame(time  = sf$time,
                 surv  = rowMeans(sf$surv),                # 全队列平均 = 标准化生存
                 group = factor(g, levels = q_levels))
    })

    # 加 (t=0, S=1) 起点, 令阶梯曲线自 1 出发
    adj_df <- bind_rows(
      data.frame(time = 0, surv = 1,
                 group = factor(q_levels, levels = q_levels)),
      adj_df
    ) %>% arrange(group, time)

    # X 轴上限/刻度: 按 Step 5b 顶部开关; 截断不丢数据(coord_cartesian 裁剪)
    x_max <- if (is.na(ADJ_KM_XMAX)) max(adj_df$time, na.rm = TRUE) else ADJ_KM_XMAX
    x_brk <- if (is.na(ADJ_KM_XBRK)) brk else ADJ_KM_XBRK

    # (3) 调整后 P 值 (★调整/标准化曲线不能用 log-rank; 取 Cox 检验, 与 Step 6 同口径)
    #     P-overall = 四分位因子整体差异(似然比检验, 已校正 covars_m3) —— log-rank 的"调整版"类比
    #     P-trend   = 四分位序号(1-4)作连续项的 Wald P (= Step 6 第782-784行)
    #   ★ P-overall 直接用两模型对数偏似然手算 LR, 不依赖 anova 列名:
    #     anova(coxph,coxph) 的 P 值列名随版本为 "P(>|Chi|)" 或 "Pr(>|Chi|)", 正则匹配易取空 -> 误回退 NA。
    fit0 <- survival::coxph(
      as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ", rhs_m3_adj)), data = df)
    p_overall <- if (isTRUE(fit$n == fit0$n)) {           # 两模型须同 n 才可比(此处 q 因子无缺、covars 已插补, 必相等)
      lr_stat <- 2 * (fit$loglik[2] - fit0$loglik[2])      # full(q+covars) vs reduced(covars) 的对数偏似然差
      lr_df   <- length(stats::coef(fit)) - length(stats::coef(fit0))   # = 四分位哑变量个数(3)
      if (lr_df > 0) stats::pchisq(lr_stat, df = lr_df, lower.tail = FALSE) else NA_real_
    } else NA_real_
    df_tr <- df; df_tr$q_num <- as.numeric(df_tr[[q_var]])   # Q1..Q4 -> 1..4 (同 Step 6 第782-784行)
    fit_tr <- survival::coxph(
      as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ q_num + ", rhs_m3_adj)), data = df_tr)
    p_trend <- tryCatch(summary(fit_tr)$coefficients["q_num", 5], error = function(e) NA_real_)
    p_lab <- switch(ADJ_KM_PVAL,
      trend   = paste0("Adjusted P-trend ",   fmt_p(p_trend)),
      overall = paste0("Adjusted P-overall ", fmt_p(p_overall)),
      both    = paste0("Adjusted P-overall ", fmt_p(p_overall), "\n",
                       "Adjusted P-trend ",   fmt_p(p_trend)),
      none    = NA_character_, NA_character_)
    # NULL 图层可被 ggplot 安全忽略; 标注置于左下空白区(曲线在上方高生存区, 不遮挡)
    p_layer <- if (!is.na(p_lab))
      annotate("text", x = 0.02 * x_max, y = 0.58, hjust = 0, vjust = 1,
               label = p_lab, size = 5, color = "grey20", lineheight = 0.95) else NULL

    ggplot(adj_df, aes(time, surv, color = group)) +
      geom_step(linewidth = 1.1) +
      scale_color_manual(values = km_palette, name = lab,
                         labels = paste0("Q", 1:4)) +
      scale_x_continuous(breaks = seq(0, x_max, by = x_brk)) +
      # coord_cartesian 仅裁剪显示范围, 阶梯线用全量数据计算后再裁到 x_max(不会提前截断)
      coord_cartesian(xlim = c(0, x_max), ylim = c(0.5, 1)) +
      p_layer +
      labs(title = paste0(lab, " Quartiles - ", ep$label,
                          " Mortality (Model 3 adjusted)"),
           x = "Time (days)", y = "Adjusted survival probability") +
      theme_bw(base_size = 16) +
      theme(
        plot.title      = element_text(size = 20, face = "bold", hjust = 0.5),
        axis.title      = element_text(size = 18),
        axis.text       = element_text(size = 15),
        legend.title    = element_text(size = 17, face = "bold"),
        legend.text     = element_text(size = 15),
        legend.key.size = grid::unit(1.1, "cm")
      )
  })

  # ★ 4×2 拼图: 8 个 IR 指标(TyG…CHG)排成 4 行 × 2 列, 输出到单页 PDF(正好填满, 无空格)。
  #   adj_plots 是纯 ggplot 对象 -> 直接用 patchwork::wrap_plots() 拼网格 (无需 arrange_ggsurvplots)。
  adj_grid <- patchwork::wrap_plots(adj_plots, ncol = 2, nrow = 4)
  pdf(file.path(DIR_DOCX, paste0("fig_km_adjusted_curves_", ep$suffix, ".pdf")),
      width = 14, height = 28)
  print(adj_grid)
  dev.off()
  cat(sprintf("  %s Model 3 调整后生存曲线已输出 (4×2 拼图): %s\n", ep$label,
              file.path(DIR_DOCX, paste0("fig_km_adjusted_curves_", ep$suffix, ".pdf"))))
}

# ====================================================================
# 06 Cox 回归 (8 指标 × 三模型 × 院内)
#    per-SD 为主 + per-1-unit 为辅; 四分位(Q1参考) + P-trend
# ====================================================================
cat(">>> Step 6: Cox 回归 (院内)\n")

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
    # --- per-SD (主) ---
    fit_extract(std_var, std_var,                                   "Model 1", "Per SD"),
    fit_extract(std_var, paste0(std_var, " + ", rhs_m2),            "Model 2", "Per SD"),
    fit_extract(std_var, paste0(std_var, " + ", rhs_m3),            "Model 3", "Per SD"),
    # --- per-1-unit (辅, 与本文 Table 2 同尺度) ---
    fit_extract(exp_var, exp_var,                                   "Model 1", "Per 1 unit"),
    fit_extract(exp_var, paste0(exp_var, " + ", rhs_m2),            "Model 2", "Per 1 unit"),
    fit_extract(exp_var, paste0(exp_var, " + ", rhs_m3),            "Model 3", "Per 1 unit")
  )

  # --- 四分位 HR + P-trend (Model 1/2/3, Q1 参考) —— 镜像本文 Table 2 三列 ---
  #   Model 1 = Crude(仅四分位); Model 2 = +covars_m2(age/sex/BMI); Model 3 = +covars_m3。
  #   每个模型各自给出 Q2/Q3/Q4 vs Q1 的 HR, 及该模型自己的 P-trend(四分位序号作连续项)。
  df_tmp <- df; df_tmp$q_num <- as.numeric(df_tmp[[q_var]])   # Q1..Q4 -> 1..4
  quart_rows <- function(model_lab, cov_rhs) {
    rhs_q <- if (nzchar(cov_rhs)) paste0(q_var, " + ", cov_rhs) else q_var
    mq    <- coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ", rhs_q)), data = df)
    sq    <- summary(mq); ciq <- sq$conf.int; coq <- sq$coefficients   # 四分位哑变量(Q2/Q3/Q4)在前 3 行
    rhs_t <- if (nzchar(cov_rhs)) paste0("q_num + ", cov_rhs) else "q_num"
    mt    <- coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ", rhs_t)), data = df_tmp)
    ptr   <- round(summary(mt)$coefficients["q_num", 5], 4)
    data.frame(
      endpoint = ep_label, index = exp_label, model = model_lab,
      type = c("Q2 vs Q1","Q3 vs Q1","Q4 vs Q1"),
      HR = round(ciq[1:3,1],3), CI_lo = round(ciq[1:3,3],3),
      CI_hi = round(ciq[1:3,4],3), p = round(coq[1:3,5],4),
      p_trend = c(NA, NA, ptr)            # P-trend 记在该模型的 Q4 行
    )
  }
  q_rows <- bind_rows(
    quart_rows("Model 1", ""),
    quart_rows("Model 2", rhs_m2),
    quart_rows("Model 3", rhs_m3)
  )

  # --- PH 检验 (Model 3, per-SD) 记录全局 P (供 Step 6b 决定是否做时依 Cox) ---
  m3_std <- coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ", std_var, " + ", rhs_m3)), data = df)
  ph_global <- tryCatch(cox.zph(m3_std)$table["GLOBAL","p"], error = function(e) NA_real_)
  cat(sprintf("  %s %s PH Global P = %.3f\n", ep_label, exp_label, ph_global))

  bind_rows(rows, q_rows) %>%
    mutate(ph_global = ifelse(type == "Per SD" & model == "Model 3", ph_global, NA))
}

cox_all <- map_dfr(ENDPOINTS, function(ep) {
  map_dfr(seq_along(EXPOSURES),
          ~run_cox(EXPOSURES[.x], EXPOSURE_LABELS[.x], ep$surv, ep$out, ep$label))
})
write.csv(cox_all, file.path(DIR_CSV, "table2_cox_all_endpoints.csv"), row.names = FALSE)
print(cox_all)

# ====================================================================
# 06b 时依 Cox (对 PH 违反指标的敏感性分析) —— 与 log(t) 交互
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
# 06c Cox 回归 —— 5 个插补数据集 Rubin 法则合并 (★审稿意见 3)
#   - 与 Step 6 相同的模型设定 (Model 1/2/3; per-SD / per-unit / 四分位 / P-trend),
#     分别在 5 个完成集上拟合, 对 log(HR) 及其标准误按 Rubin 法则合并
#     (Barnard-Rubin 自由度, 大样本近似)。
#   - 暴露/结局/生存时间完整未插补, 四分位切点与标准化暴露在各完成集间完全一致,
#     仅被插补协变量不同 -> 合并仅反映协变量插补不确定性。
#   - ★修订稿 Table 2 应以本步合并结果为准 (table2_cox_pooled_rubin.csv);
#     Step 6 (第 1 完成集) 结果保留作对照。
# ====================================================================
cat(">>> Step 6c: Cox 回归 (Rubin 法则合并 5 个插补集)\n")

# 被插补的协变量列 (缺失率>0 且未被剔除); 各完成集间仅这些列不同
imputed_cols <- names(miss_rate[vars_keep])[miss_rate[vars_keep] > 0]
imputed_cols <- intersect(imputed_cols, names(df))

# 以 df (第 1 完成集, 含标签/因子/派生变量) 为模板, 逐列替换为第 i 完成集的取值
df_imp_list <- lapply(imp_data_list, function(d) {
  out <- df
  for (cl in imputed_cols) out[[cl]] <- d[[cl]]
  out
})

# Rubin 法则: 输入各完成集的 log-HR 与 SE, 输出合并 HR/CI/P
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

# 在单个完成集上拟合并提取某项的 log-HR 与 SE
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

# 对单个 (暴露 × 模型 × 类型) 在 5 个完成集上拟合并合并
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
      list(std_var, std_var,                        "Model 1", "Per SD"),
      list(std_var, paste0(std_var, " + ", rhs_m2), "Model 2", "Per SD"),
      list(std_var, paste0(std_var, " + ", rhs_m3), "Model 3", "Per SD"),
      list(exp_var, exp_var,                        "Model 1", "Per 1 unit"),
      list(exp_var, paste0(exp_var, " + ", rhs_m2), "Model 2", "Per 1 unit"),
      list(exp_var, paste0(exp_var, " + ", rhs_m3), "Model 3", "Per 1 unit")
    )
    rows <- map_dfr(specs, function(sp) {
      pr <- pool_one(sp[[1]], sp[[2]], surv_var, out_var)
      data.frame(endpoint = ep$label, index = exp_label, model = sp[[3]], type = sp[[4]],
                 HR = round(pr["HR"], 3), CI_lo = round(pr["CI_lo"], 3),
                 CI_hi = round(pr["CI_hi"], 3), p = round(pr["p"], 4), row.names = NULL)
    })
    # 四分位 (Q2/Q3/Q4 vs Q1) + P-trend, 三模型; q_num 逐暴露临时加入各完成集
    df_imp_list <<- lapply(df_imp_list, function(d) { d$q_num <- as.numeric(d[[q_var]]); d })
    q_rows <- map_dfr(seq_len(3), function(mi) {
      cov_rhs   <- c("", rhs_m2, rhs_m3)[mi]
      model_lab <- paste0("Model ", mi)
      rhs_q <- if (nzchar(cov_rhs)) paste0(q_var, " + ", cov_rhs) else q_var
      qr <- map_dfr(paste0("Q", 2:4), function(qq) {
        pr <- pool_one(paste0(q_var, qq), rhs_q, surv_var, out_var)
        data.frame(endpoint = ep$label, index = exp_label, model = model_lab,
                   type = paste0(qq, " vs Q1"),
                   HR = round(pr["HR"], 3), CI_lo = round(pr["CI_lo"], 3),
                   CI_hi = round(pr["CI_hi"], 3), p = round(pr["p"], 4), row.names = NULL)
      })
      rhs_t <- if (nzchar(cov_rhs)) paste0("q_num + ", cov_rhs) else "q_num"
      pr_tr <- pool_one("q_num", rhs_t, surv_var, out_var)
      qr$p_trend <- c(NA, NA, round(pr_tr["p"], 4))   # P-trend 记在该模型 Q4 行 (同 Step 6)
      qr
    })
    bind_rows(rows, q_rows)
  })
})
cox_pooled$n_imp <- length(df_imp_list)
write.csv(cox_pooled, file.path(DIR_CSV, "table2_cox_pooled_rubin.csv"), row.names = FALSE)
cat("  Rubin 合并 Cox 结果已输出: table2_cox_pooled_rubin.csv\n")
print(cox_pooled)

# ====================================================================
# 06d 完整病例 (complete-case) 敏感性分析 (★审稿意见 3)
#   - 仅在 Model 2/3 协变量完整的病例上拟合 Model 3 (暴露完整 by design);
#     与 Step 6c 合并结果并列输出, 供修订稿明确报告 (不再仅 Methods 提及)。
# ====================================================================
cat(">>> Step 6d: 完整病例敏感性分析 (Model 3)\n")

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
    # per-SD / per-unit
    rows <- map_dfr(c("Per SD", "Per 1 unit"), function(tp) {
      term <- if (tp == "Per SD") std_var else exp_var
      ft   <- fit_term(df_cc, term, paste0(term, " + ", rhs_m3), surv_var, out_var)
      z    <- ft["beta"] / ft["se"]
      data.frame(endpoint = ep$label, index = exp_label, model = "Model 3", type = tp,
                 n_cc = nrow(df_cc),
                 HR_cc    = round(exp(ft["beta"]), 3),
                 CI_lo_cc = round(exp(ft["beta"] - 1.96 * ft["se"]), 3),
                 CI_hi_cc = round(exp(ft["beta"] + 1.96 * ft["se"]), 3),
                 p_cc     = round(2 * pnorm(-abs(z)), 4), row.names = NULL)
    })
    # 四分位 Q2-Q4 vs Q1 + P-trend (口径同 Table 2; q_num 临时加入完整病例集)
    df_cc_q <- df_cc; df_cc_q$q_num <- as.numeric(df_cc_q[[q_var]])
    qr <- map_dfr(paste0("Q", 2:4), function(qq) {
      ft <- fit_term(df_cc, paste0(q_var, qq), paste0(q_var, " + ", rhs_m3), surv_var, out_var)
      z  <- ft["beta"] / ft["se"]
      data.frame(endpoint = ep$label, index = exp_label, model = "Model 3",
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

# 与 Rubin 合并结果并排 (Model 3 全部类型: per-SD / per-unit / 四分位)
cc_compare <- cc_res %>%
  left_join(cox_pooled %>%
              filter(model == "Model 3") %>%
              select(endpoint, index, type, p_trend_pooled = p_trend,
                     HR_pooled = HR, CI_lo_pooled = CI_lo,
                     CI_hi_pooled = CI_hi, p_pooled = p),
            by = c("endpoint", "index", "type"))
write.csv(cc_compare, file.path(DIR_CSV, "sensitivity_complete_case_vs_pooled.csv"),
          row.names = FALSE)
cat("  完整病例 vs Rubin 合并 对照已输出: sensitivity_complete_case_vs_pooled.csv\n")
print(cc_compare)

# ====================================================================
# 06e Logistic 回归敏感性分析 (★审稿意见 7)
#   - 院内死亡按二分类结局处理 (桥接 Cox 生存框架与 ML 二分类框架);
#     Model 3 协变量, 每完成集 glm(binomial), log(OR) 按 Rubin 法则合并。
#   - 输出 OR (非 HR): sensitivity_logistic_rubin.csv (补充表)。
# ====================================================================
cat(">>> Step 6e: Logistic 回归敏感性分析 (Rubin 合并)\n")

fit_glm_term <- function(data, term, rhs, out_var) {
  mod <- tryCatch(
    stats::glm(as.formula(paste0(out_var, " ~ ", rhs)), family = binomial, data = data),
    error = function(e) NULL)
  if (is.null(mod)) return(c(beta = NA_real_, se = NA_real_))
  co <- summary(mod)$coefficients
  r  <- which(rownames(co) == term)
  if (length(r) == 0) return(c(beta = NA_real_, se = NA_real_))
  c(beta = co[r, 1], se = co[r, 2])
}

pool_one_glm <- function(term, rhs, out_var) {
  fits <- lapply(df_imp_list, fit_glm_term, term = term, rhs = rhs, out_var = out_var)
  beta <- sapply(fits, `[[`, "beta"); se <- sapply(fits, `[[`, "se")
  ok <- !is.na(beta) & !is.na(se)
  if (sum(ok) < 2) return(c(HR = NA_real_, CI_lo = NA_real_, CI_hi = NA_real_, p = NA_real_))
  pool_rubin(beta[ok], se[ok])
}

logi_res <- map_dfr(ENDPOINTS, function(ep) {
  out_var <- ep$out
  map_dfr(seq_along(EXPOSURES), function(i) {
    exp_var <- EXPOSURES[i]; exp_label <- EXPOSURE_LABELS[i]
    std_var <- paste0(exp_var, "_std"); q_var <- paste0(exp_var, "_q")
    specs <- list(
      list(std_var, paste0(std_var, " + ", rhs_m3), "Per SD"),
      list(exp_var, paste0(exp_var, " + ", rhs_m3), "Per 1 unit")
    )
    rows <- map_dfr(specs, function(sp) {
      pr <- pool_one_glm(sp[[1]], sp[[2]], out_var)
      data.frame(endpoint = ep$label, index = exp_label, model = "Model 3", type = sp[[3]],
                 OR = round(unname(pr["HR"]), 3), CI_lo = round(unname(pr["CI_lo"]), 3),
                 CI_hi = round(unname(pr["CI_hi"]), 3), p = round(unname(pr["p"]), 4),
                 row.names = NULL)
    })
    # 四分位 OR + P-trend (q_num 逐暴露刷新到各完成集, 同 Step 6c 口径)
    df_imp_list <<- lapply(df_imp_list, function(d) { d$q_num <- as.numeric(d[[q_var]]); d })
    qr <- map_dfr(paste0("Q", 2:4), function(qq) {
      pr <- pool_one_glm(paste0(q_var, qq), paste0(q_var, " + ", rhs_m3), out_var)
      data.frame(endpoint = ep$label, index = exp_label, model = "Model 3",
                 type = paste0(qq, " vs Q1"),
                 OR = round(unname(pr["HR"]), 3), CI_lo = round(unname(pr["CI_lo"]), 3),
                 CI_hi = round(unname(pr["CI_hi"]), 3), p = round(unname(pr["p"]), 4),
                 row.names = NULL)
    })
    pr_tr <- pool_one_glm("q_num", paste0("q_num + ", rhs_m3), out_var)
    qr$p_trend <- c(NA, NA, round(unname(pr_tr["p"]), 4))   # P-trend 记在 Q4 行 (同 Step 6)
    bind_rows(rows, qr)
  })
})
logi_res$n_imp <- length(df_imp_list)
write.csv(logi_res, file.path(DIR_CSV, "sensitivity_logistic_rubin.csv"), row.names = FALSE)
cat("  Logistic 敏感性 (Rubin 合并) 已输出: sensitivity_logistic_rubin.csv\n")
print(logi_res)

# ====================================================================
# 06f PH 检验完整报告 + 时依 Cox (5 插补集 Rubin 合并) (★审稿意见 7)
#   - (1) PH: 每完成集拟合 Model 3 (per-SD) Cox, cox.zph 取 GLOBAL 与暴露行 P,
#         报告 5 次的中位数与范围 -> ph_test_results.csv (补充表);
#   - (2) 时依 Cox: 全部 8 指标 (不再只跑违反者), tt = x*log(t),
#         主效应与交互项分别 Rubin 合并 -> cox_timevarying_pooled_rubin.csv (补充表)。
# ====================================================================
cat(">>> Step 6f: PH 检验完整报告 + 时依 Cox (Rubin 合并)\n")

# --- (1) PH 检验 (Schoenfeld) ---
ph_res <- map_dfr(ENDPOINTS, function(ep) {
  surv_var <- ep$surv; out_var <- ep$out
  map_dfr(seq_along(EXPOSURES), function(i) {
    exp_var <- EXPOSURES[i]; exp_label <- EXPOSURE_LABELS[i]
    std_var <- paste0(exp_var, "_std")
    res <- lapply(df_imp_list, function(d) {
      m <- tryCatch(survival::coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ",
                                    std_var, " + ", rhs_m3)), data = d),
                    error = function(e) NULL)
      if (is.null(m)) return(c(global = NA_real_, expo = NA_real_))
      tb <- tryCatch(survival::cox.zph(m)$table, error = function(e) NULL)
      if (is.null(tb) || !("GLOBAL" %in% rownames(tb))) return(c(global = NA_real_, expo = NA_real_))
      c(global = unname(tb["GLOBAL", "p"]),
        expo   = if (std_var %in% rownames(tb)) unname(tb[std_var, "p"]) else NA_real_)
    })
    g <- sapply(res, `[`, "global"); e <- sapply(res, `[`, "expo")
    data.frame(endpoint = ep$label, index = exp_label,
               ph_global_median   = round(median(g, na.rm = TRUE), 4),
               ph_global_range    = sprintf("%.4g-%.4g", min(g, na.rm = TRUE), max(g, na.rm = TRUE)),
               ph_exposure_median = round(median(e, na.rm = TRUE), 4),
               ph_exposure_range  = sprintf("%.4g-%.4g", min(e, na.rm = TRUE), max(e, na.rm = TRUE)),
               n_imp = sum(!is.na(g)), row.names = NULL)
  })
})
write.csv(ph_res, file.path(DIR_CSV, "ph_test_results.csv"), row.names = FALSE)
cat("  PH 检验 (Schoenfeld, 5 插补集中位数/范围) 已输出: ph_test_results.csv\n")
print(ph_res)

# --- (2) 时依 Cox (全部指标, Rubin 合并) ---
fit_td_term <- function(data, std_var, rhs_m3, surv_var, out_var) {
  mod <- tryCatch(
    survival::coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ",
                    std_var, " + tt(", std_var, ") + ", rhs_m3)),
                    data = data, tt = function(x, t, ...) x * log(pmax(t, 1e-3))),
    error = function(e) NULL)
  rb <- function(r, co) if (length(r) == 0) c(NA_real_, NA_real_) else c(co[r, 1], co[r, 3])
  if (is.null(mod)) return(rbind(c(NA_real_, NA_real_), c(NA_real_, NA_real_)))
  co <- summary(mod)$coefficients
  r_main <- which(rownames(co) == std_var)
  r_tt   <- grep("^tt\\(", rownames(co))
  rbind(rb(r_main, co), rb(r_tt, co))
}

td_pooled <- map_dfr(ENDPOINTS, function(ep) {
  surv_var <- ep$surv; out_var <- ep$out
  map_dfr(seq_along(EXPOSURES), function(i) {
    exp_var <- EXPOSURES[i]; exp_label <- EXPOSURE_LABELS[i]
    std_var <- paste0(exp_var, "_std")
    fits <- lapply(df_imp_list, fit_td_term, std_var = std_var, rhs_m3 = rhs_m3,
                   surv_var = surv_var, out_var = out_var)
    b_main <- sapply(fits, function(m) m[1, 1]); s_main <- sapply(fits, function(m) m[1, 2])
    b_tt   <- sapply(fits, function(m) m[2, 1]); s_tt   <- sapply(fits, function(m) m[2, 2])
    okm <- !is.na(b_main) & !is.na(s_main); okt <- !is.na(b_tt) & !is.na(s_tt)
    pr_main <- if (sum(okm) >= 2) pool_rubin(b_main[okm], s_main[okm]) else c(HR = NA_real_, CI_lo = NA_real_, CI_hi = NA_real_, p = NA_real_)
    pr_tt   <- if (sum(okt) >= 2) pool_rubin(b_tt[okt], s_tt[okt])     else c(HR = NA_real_, CI_lo = NA_real_, CI_hi = NA_real_, p = NA_real_)
    data.frame(endpoint = ep$label, index = exp_label,
               HR_main = round(unname(pr_main["HR"]), 3), HR_main_lo = round(unname(pr_main["CI_lo"]), 3),
               HR_main_hi = round(unname(pr_main["CI_hi"]), 3), p_main = round(unname(pr_main["p"]), 4),
               HR_time = round(unname(pr_tt["HR"]), 3), p_time = round(unname(pr_tt["p"]), 4),
               n_imp = sum(okm), row.names = NULL)
  })
})
# 附 PH 全局 P 中位数, 便于补充表合并呈现
td_pooled <- td_pooled %>%
  left_join(ph_res %>% select(endpoint, index, ph_global_median), by = c("endpoint", "index")) %>%
  relocate(ph_global_median, .after = index)
write.csv(td_pooled, file.path(DIR_CSV, "cox_timevarying_pooled_rubin.csv"), row.names = FALSE)
cat("  时依 Cox (Rubin 合并, 全指标) 已输出: cox_timevarying_pooled_rubin.csv\n")
print(td_pooled)

# ====================================================================
# 06g 临床预设协变量集敏感性分析 (★审稿人2 意见4/意见5)
#   - 意见4: Model 3 协变量由结局驱动的 LASSO 选出; 病因学/关联分析宜对比临床预设集。
#   - 意见5: SOFA/SAPS II/OASIS 三评分概念重叠; 增设"同一临床预设集但仅留 SOFA"
#     (去掉 saps_ii/oasis) 的对照校正集, 直接检验三评分冗余对暴露效应的影响。
#   - 临床预设集 (不依据本数据结局选择): 人口学(age/gender/bmi) + 严重度评分
#     (sofa/saps_ii/oasis) + 合并症(dm/hypertension/stroke/ckd/hf/chd)
#     + 干预与用药(ventilation/vasopressor/antidiabetic/antihypertensive)。
#   - 模型设定同 Step 6c (per-SD / per-unit / 四分位 / P-trend), 5 完成集 Rubin 合并;
#     复用 Step 6c 的 pool_rubin / fit_term / pool_one / df_imp_list。
# ====================================================================
cat(">>> Step 6g: 临床预设协变量集敏感性分析 (双校正集, Rubin 合并)\n")

covars_clin <- intersect(c("age", "gender", "bmi",
                           "sofa", "saps_ii", "oasis",
                           "dm", "hypertension", "stroke", "ckd", "hf", "chd",
                           "ventilation", "vasopressor",
                           "antidiabetic", "antihypertensive"), names(df))
adj_sets <- list(
  "Clinical preset (SOFA+SAPS II+OASIS)" = covars_clin,
  "Clinical preset (SOFA only)"          = setdiff(covars_clin, c("saps_ii", "oasis"))
)

clin_res <- map_dfr(names(adj_sets), function(adj_lab) {
  covs <- adj_sets[[adj_lab]]
  rhs_adj <- paste(covs, collapse = " + ")
  cat(sprintf("  校正集 [%s] (%d 项): %s\n", adj_lab, length(covs), rhs_adj))
  res <- map_dfr(ENDPOINTS, function(ep) {
    surv_var <- ep$surv; out_var <- ep$out
    map_dfr(seq_along(EXPOSURES), function(i) {
      exp_var <- EXPOSURES[i]; exp_label <- EXPOSURE_LABELS[i]
      std_var <- paste0(exp_var, "_std"); q_var <- paste0(exp_var, "_q")
      rows <- map_dfr(list(list(std_var, "Per SD"), list(exp_var, "Per 1 unit")),
        function(sp) {
          pr <- pool_one(sp[[1]], paste0(sp[[1]], " + ", rhs_adj), surv_var, out_var)
          data.frame(endpoint = ep$label, index = exp_label, type = sp[[2]],
                     HR = round(pr["HR"], 3), CI_lo = round(pr["CI_lo"], 3),
                     CI_hi = round(pr["CI_hi"], 3), p = round(pr["p"], 4),
                     row.names = NULL)
        })
      # 四分位 + P-trend (q_num 逐暴露重建, 同 Step 6c 口径)
      df_imp_list <<- lapply(df_imp_list, function(d) { d$q_num <- as.numeric(d[[q_var]]); d })
      qr <- map_dfr(paste0("Q", 2:4), function(qq) {
        pr <- pool_one(paste0(q_var, qq), paste0(q_var, " + ", rhs_adj), surv_var, out_var)
        data.frame(endpoint = ep$label, index = exp_label, type = paste0(qq, " vs Q1"),
                   HR = round(pr["HR"], 3), CI_lo = round(pr["CI_lo"], 3),
                   CI_hi = round(pr["CI_hi"], 3), p = round(pr["p"], 4),
                   row.names = NULL)
      })
      pr_tr <- pool_one("q_num", paste0("q_num + ", rhs_adj), surv_var, out_var)
      qr$p_trend <- c(NA, NA, round(pr_tr["p"], 4))
      bind_rows(rows, qr)
    })
  })
  res$adjustment <- adj_lab
  res
})
clin_res <- clin_res %>% relocate(adjustment, .after = index)
clin_res$n_imp <- length(df_imp_list)
write.csv(clin_res, file.path(DIR_CSV, "sensitivity_clinical_covars_rubin.csv"),
          row.names = FALSE)
cat("  临床预设协变量集敏感性分析 (双校正集) 已输出: sensitivity_clinical_covars_rubin.csv\n")
print(clin_res)


# ====================================================================
# 07 RCS (8 指标 × 院内) —— 3 节点, 中位数参考, strong::rcs_cph
# ====================================================================
cat(">>> Step 7: RCS (3 节点, 院内, rcs_cph)\n")
# ★ datadist 会遍历传入 df 的每一列: 全缺失列(如 hba1c, <2 非缺失)会【报错】, 常数列(如 icu_order)会【警告】。
#   这些列并不在 RCS 模型中, 故构造"datadist 安全"的子集 df_rcs(剔除全缺失/常数列, 但保留全部建模所需列),
#   供本步 datadist 与 strong::rcs_cph 使用。strong::rcs_cph 内部也会对传入的 df 再跑 datadist, 故必须传 df_rcs。
.need_rcs <- unique(c(vapply(ENDPOINTS, function(e) e$surv, character(1)),
                      vapply(ENDPOINTS, function(e) e$out,  character(1)),
                      EXPOSURES, covars_m3))
.bad_rcs  <- names(df)[vapply(df, function(v)
                 sum(!is.na(v)) < 2L || length(unique(v[!is.na(v)])) < 2L, logical(1))]
.drop_rcs <- setdiff(.bad_rcs, .need_rcs)
df_rcs <- df[, setdiff(names(df), .drop_rcs), drop = FALSE]
if (length(.drop_rcs))
  cat(sprintf("  [RCS] datadist 前已剔除全缺失/常数列: %s\n", paste(.drop_rcs, collapse = ", ")))
dd <- datadist(df_rcs); options(datadist = 'dd')

get_p_nl <- function(exp_var, surv_var, out_var) {
  fml <- as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ rcs(", exp_var, ", 3) + ",
                           paste(covars_m3, collapse = " + ")))
  fit <- cph(fml, data = df_rcs, x = TRUE, y = TRUE)
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
      df            = df_rcs,
      time          = surv_var,
      outcome       = out_var,
      x             = exp_var,
      covars        = covars_m3,
      x_label       = lab,
      outcome_label = ep_label,
      knot          = 3,                      # ← 3 节点 (本文)
      y_axis_lab    = "Adjusted HR (95% CI)"  # Cox -> HR
    ) + ggtitle(paste0(lab, " (", ep_label, ")"))
  })

  combined <- patchwork::wrap_plots(rcs_plots, ncol = 2) +
    patchwork::plot_annotation(
      title = paste0(ep_label, " mortality — RCS (Model 3 adjusted, 3 knots)"))
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
# 08 亚组分析 (8 指标 × 院内) —— 对齐本文 Fig4 分层
#   ★ 改动: 暴露不再按四分位(Q1 参考), 而是作为【连续变量(per-SD)】进入各亚组 Cox。
#     - 每个亚组层内拟合 Cox: Surv ~ <exp>_std + covars_m3(剔除该分层因子自身),
#       报告 per-1-SD 的 HR(95%CI)。方向与全队列 per-SD 主分析一致(SPISE 预期 HR<1)。
#     - P-for-interaction 仍为 连续 per-SD × 分层 的 LRT(口径不变), 与点估计同标尺。
#     - 不再依赖 strong::ok_subgroup_mult_cox(其按 col_name_gp 分位估计)。
# ====================================================================
cat(">>> Step 8: 亚组分析 (院内, 连续暴露 per-SD)\n")

# 分层因子 (★按需求 4 项): 性别 / 年龄(<60 vs ≥60) / BMI(<30 vs ≥30) / 族裔(race: White/Black/Other)
#   - age_grp / bmi_grp 见 Step 3 (年龄<60、BMI<30 切点); race 由 eICU ethnicity 重编码(Step 3, White/Black/Other 三分类)。
#   - race 不在 covars_m3 中(Step 3b 已显式排除), 故按其分层并用 covars_m3 校正无自身共线。
#   - 原 chd / hypertension / dm / stroke 分层因子已按需求移除。
subgroup_vars <- c("gender","age_grp","bmi_grp","race")
subgroup_vars <- intersect(subgroup_vars, names(df))

# 层内逐一拟合连续(per-SD)暴露 Cox, 返回该指标 × 该分层的所有水平 HR
fit_subgroup_cont <- function(surv_var, out_var, std_var, exp_label, ep_label,
                              sv, covars, df) {
  rhs_adj <- setdiff(covars, sv)            # 层内校正剔除分层因子自身, 避免层内常数共线
  levs    <- levels(df[[sv]])
  if (is.null(levs)) levs <- sort(unique(stats::na.omit(df[[sv]])))
  rows_out <- list()
  for (lev in levs) {
    sub <- df[!is.na(df[[sv]]) & df[[sv]] == lev, , drop = FALSE]
    fml <- as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ",
                             std_var, " + ", paste(rhs_adj, collapse = " + ")))
    fm  <- tryCatch(survival::coxph(fml, data = sub), error = function(e) NULL)
    if (is.null(fm)) next
    est <- tryCatch({
      b <- coef(fm); v <- sqrt(diag(vcov(fm)))
      k <- match(std_var, names(b))
      if (is.na(k)) NULL else data.frame(
        endpoint = ep_label, index = exp_label, strata = sv, level = as.character(lev),
        n        = tryCatch(stats::nobs(fm), error = function(e) nrow(sub)),
        n_event  = tryCatch(sum(fm$y[, "status"] == 1), error = function(e) NA_integer_),
        contrast = "per_1SD",
        HR    = round(exp(b[k]), 3),
        CI_lo = round(exp(b[k] - 1.96 * v[k]), 3),
        CI_hi = round(exp(b[k] + 1.96 * v[k]), 3),
        p     = round(2 * pnorm(-abs(b[k] / v[k])), 4),
        row.names = NULL)
    }, error = function(e) NULL)
    if (!is.null(est)) rows_out[[length(rows_out) + 1]] <- est
  }
  bind_rows(rows_out)
}

subgroup_flat <- list()
int_all       <- list()

for (ep in ENDPOINTS) {
  surv_var <- ep$surv; out_var <- ep$out; ep_label <- ep$label

  for (i in seq_along(EXPOSURES)) {
    exp_var <- EXPOSURES[i]; exp_label <- EXPOSURE_LABELS[i]
    std_var <- paste0(exp_var, "_std")
    key     <- paste0(ep$key, "_", exp_var)

    # 各分层各水平: per-SD 连续暴露 HR
    subgroup_flat[[key]] <- map_dfr(subgroup_vars, function(sv)
      fit_subgroup_cont(surv_var, out_var, std_var, exp_label, ep_label,
                        sv, covars_m3, df))

    # P-for-interaction (连续 per-SD × 分层) + BH-FDR —— 口径与点估计一致
    ip <- map_dfr(subgroup_vars, function(sv) {
      rhs_adj  <- paste(setdiff(covars_m3, sv), collapse = " + ")
      # 全模型: 含 暴露(per-SD) × 分层 交互项
      fml_full <- as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ",
                                    std_var, " * ", sv, " + ", rhs_adj))
      # 简约模型: 仅主效应(去掉交互项)
      fml_red  <- as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ",
                                    std_var, " + ", sv, " + ", rhs_adj))
      fit_full <- tryCatch(survival::coxph(fml_full, data = df), error = function(e) NULL)
      fit_red  <- tryCatch(survival::coxph(fml_red,  data = df), error = function(e) NULL)
      if (is.null(fit_full) || is.null(fit_red))
        return(data.frame(strata = sv, p_int = NA_real_))
      # ★ 似然比检验(LRT)给出全局 P-for-interaction:
      #   race(White/Black/Other, 3 水平 -> 2 个交互系数)也得到单一全局 p,
      #   优于旧版仅取首个交互系数 Wald-p(多分类时会漏检其余水平对比)。
      df_int <- length(coef(fit_full)) - length(coef(fit_red))
      p_int  <- if (df_int > 0)
        round(pchisq(2 * (as.numeric(logLik(fit_full)) - as.numeric(logLik(fit_red))),
                     df = df_int, lower.tail = FALSE), 4) else NA_real_
      data.frame(strata = sv, p_int = p_int)
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

# ====================================================================
# 08b 亚组分析 —— 5 个插补集 Rubin 法则合并 (★审稿意见 3)
#   - 层内 per-SD HR: 与 Step 8 同口径, 5 个完成集分别拟合后按 Rubin 法则合并。
#   - 交互作用 P: 二分类分层(gender/age_grp/bmi_grp)的单一交互系数按 Rubin 合并;
#     race(3 水平, 2 df LRT)报告各完成集 LRT P 的中位数与范围 (p_int_detail)。
#   - FDR: 对每个暴露跨 4 个分层因子做 BH 校正 (与 Step 8 同口径)。
#   - ★修订稿亚组表 (Suppl Table) 建议以本步合并结果为准。
# ====================================================================
cat(">>> Step 8b: 亚组分析 (Rubin 法则合并 5 个插补集)\n")

subgroup_pooled <- map_dfr(ENDPOINTS, function(ep) {
  surv_var <- ep$surv; out_var <- ep$out
  map_dfr(seq_along(EXPOSURES), function(i) {
    exp_var <- EXPOSURES[i]; exp_label <- EXPOSURE_LABELS[i]
    std_var <- paste0(exp_var, "_std")

    # (1) 层内 per-SD HR 合并
    flat <- map_dfr(subgroup_vars, function(sv) {
      rhs_adj <- paste(setdiff(covars_m3, sv), collapse = " + ")
      levs <- levels(df[[sv]])
      if (is.null(levs)) levs <- sort(unique(stats::na.omit(df[[sv]])))
      map_dfr(levs, function(lev) {
        fits <- lapply(df_imp_list, function(d) {
          sub <- d[!is.na(d[[sv]]) & d[[sv]] == lev, , drop = FALSE]
          fit_term(sub, std_var, paste0(std_var, " + ", rhs_adj), surv_var, out_var)
        })
        beta <- sapply(fits, `[[`, "beta"); se <- sapply(fits, `[[`, "se")
        ok <- !is.na(beta) & !is.na(se)
        if (sum(ok) < 2) return(NULL)
        pr   <- pool_rubin(beta[ok], se[ok])
        sub1 <- df[!is.na(df[[sv]]) & df[[sv]] == lev, , drop = FALSE]
        data.frame(endpoint = ep$label, index = exp_label, strata = sv,
                   level = as.character(lev), n = nrow(sub1),
                   n_event = sum(sub1[[out_var]] == 1), contrast = "per_1SD",
                   HR = round(pr["HR"], 3), CI_lo = round(pr["CI_lo"], 3),
                   CI_hi = round(pr["CI_hi"], 3), p = round(pr["p"], 4),
                   row.names = NULL)
      })
    })

    # (2) 交互作用 P
    ip <- map_dfr(subgroup_vars, function(sv) {
      rhs_adj  <- paste(setdiff(covars_m3, sv), collapse = " + ")
      fml_full <- paste0("Surv(", surv_var, ", ", out_var, ") ~ ",
                         std_var, " * ", sv, " + ", rhs_adj)
      n_lev <- length(unique(stats::na.omit(df[[sv]])))
      if (n_lev == 2) {
        # 二分类: 单一交互系数 -> Rubin 合并
        fits <- lapply(df_imp_list, function(d) {
          mod <- tryCatch(coxph(as.formula(fml_full), data = d), error = function(e) NULL)
          if (is.null(mod)) return(c(beta = NA_real_, se = NA_real_))
          co <- summary(mod)$coefficients
          r  <- grep(":", rownames(co), fixed = TRUE)
          if (length(r) != 1) return(c(beta = NA_real_, se = NA_real_))
          c(beta = co[r, 1], se = co[r, 3])
        })
        beta <- sapply(fits, `[[`, "beta"); se <- sapply(fits, `[[`, "se")
        ok <- !is.na(beta) & !is.na(se)
        p_int <- if (sum(ok) >= 2) round(unname(pool_rubin(beta[ok], se[ok])["p"]), 4) else NA_real_
        data.frame(strata = sv, p_int = p_int, p_int_detail = NA_character_)
      } else {
        # 多分类 (race): 各完成集 LRT P 的中位数与范围
        ps <- sapply(df_imp_list, function(d) {
          f_full <- tryCatch(coxph(as.formula(fml_full), data = d), error = function(e) NULL)
          f_red  <- tryCatch(
            coxph(as.formula(paste0("Surv(", surv_var, ", ", out_var, ") ~ ",
                                    std_var, " + ", sv, " + ", rhs_adj)), data = d),
            error = function(e) NULL)
          if (is.null(f_full) || is.null(f_red)) return(NA_real_)
          df_i <- length(coef(f_full)) - length(coef(f_red))
          if (df_i <= 0) return(NA_real_)
          pchisq(2 * (as.numeric(logLik(f_full)) - as.numeric(logLik(f_red))),
                 df = df_i, lower.tail = FALSE)
        })
        ps <- ps[!is.na(ps)]
        data.frame(strata = sv,
                   p_int = if (length(ps)) round(median(ps), 4) else NA_real_,
                   p_int_detail = if (length(ps))
                     sprintf("median of %d imputations; range %.4f-%.4f",
                             length(ps), min(ps), max(ps)) else NA_character_)
      }
    })
    ip$fdr_p    <- round(p.adjust(ip$p_int, method = "BH"), 4)
    ip$endpoint <- ep$label; ip$index <- exp_label

    left_join(flat, ip, by = c("endpoint", "index", "strata"))
  })
})
write.csv(subgroup_pooled, file.path(DIR_CSV, "subgroup_analysis_pooled_rubin.csv"),
          row.names = FALSE)
cat("  Rubin 合并亚组结果已输出: subgroup_analysis_pooled_rubin.csv\n")

# ====================================================================
# 09 NRI / IDI (8 指标 × 院内) —— 增量价值 (本文 Suppl Table 2)
# ====================================================================
cat(">>> Step 9: NRI / IDI (院内)\n")

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
#    对齐本文 Fig5 / Suppl Table 1 / Suppl Table 3
# ====================================================================
cat(">>> Step 10: 单指标 ROC + Youden 切点 + DeLong 两两\n")

exp_colors <- grDevices::hcl.colors(length(EXPOSURES), palette = "Dark 3")

auc_uni  <- list()
cutoffs  <- list()
roc_store_primary <- list()  # 主要终点(院内)各指标 roc 对象, 供 DeLong

for (ep in ENDPOINTS) {
  out_var <- ep$out; ep_label <- ep$label
  y <- df_model[[out_var]]

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
write.csv(bind_rows(delong_pairs), file.path(DIR_CSV, "roc_delong_pairwise_inhosp.csv"), row.names = FALSE)
cat("  DeLong 两两比较已输出\n")

# ====================================================================
# 11 导出 Python 机器学习数据 (traindata.csv / valdata.csv)
#   ★审稿意见 3 —— ML 流程的插补策略 (单一插补) 说明与论证:
#   (1) 机器学习流程 (7 算法 × 网格调参 × 1000 次 bootstrap × SMOTE × SHAP) 计算量大,
#       对 5 个插补集分别训练并合并预测在工程上不可行, 文献中亦无统一做法;
#   (2) 暴露指标 (含 CHG) 完整未插补, 插补仅涉及少数协变量 (缺失率均 ≤20%,
#       见 missingness_report.csv), 插补不确定性对 ML 特征影响有限;
#   (3) 与本文 ML 方法学模板 (Luo et al., Int J Med Inform 2025;198:105874) 一致,
#       采用第 1 个完成集 (complete(imp, 1));
#   (4) 插补不确定性已由 Step 6c (主要 Cox 结果 Rubin 法则合并) 与
#       Step 6d (完整病例敏感性分析) 系统评估。
#   修订稿 Methods 将按上述口径明确描述: "a single imputed dataset (the first
#   completed dataset) was used for the machine-learning pipeline"。
#   供 python 端「Python机器学习建模与评估.py」直接读取。
#   与 python 脚本严格对齐的口径:
#     - 第 1 列 = 结局 Outcome (0/1); 其后为 9 个最终建模特征 (列名须与 python 完全一致)
#     - 9 个特征(n=9): age, sbp, temperature, stroke, glucose, sofa, saps_ii, oasis, tyg_bmi
#     - 分类变量 stroke 转 0/1; 其余为连续值; 无缺失(Step 1b 已插补)
#     - 文件编码 = GBK (python: pd.read_csv(..., encoding="GBK"))
#     - 训练集 : 验证集 = 按 Outcome 分层 7:3 随机拆分 (set.seed 可复现)
# ====================================================================
cat(">>> Step 11: (已停用) Python 建模数据导出改由 11_ml_section_eicu.R Step 13d 完成 (防泄漏, 审稿意见#5)\n")
#   ★★ 旧流程(全队列插补 + 全队列筛选 + 硬编码 9 变量 -> 再切分导出)存在
#       插补/筛选 -> 验证集 的信息泄漏, 已按审稿意见#5 停用。
#   ★★ 新流程: traindata.csv / valdata.csv 由 11_ml_section_eicu.R 的 Step 13d 导出
#       (先切分 -> 两分区独立插补 -> 训练集内筛选 -> 训练集内调参)。请运行:
#           source("11_ml_section_eicu.R")
#       python 端硬编码的特征列表须与 Step 13d 打印的最终建模特征一致。

# --- (保留) 全队列 ML 宽表, 供留档/其他用途 ---
df %>%
  mutate(across(where(is.factor), ~as.integer(.) - 1L)) %>%
  select(-any_of(c("intime", "outtime", vapply(ENDPOINTS, function(e) e$group, character(1))))) %>%
  write.csv(file.path(DIR_CSV, "cohort_for_ml.csv"), row.names = FALSE)

save(df, cox_all, td_cox, subgroup_all, nri_idi, rcs_summary_df,
     file = file.path(DIR_LOAD, "all_results.Rdata"))

cat("\n=== R 分析完成 (eICU, 院内死亡为唯一终点); 已生成 python 建模所需 traindata.csv / valdata.csv ===\n")
