######################################################################
# Step 11–13: 机器学习部分 (eICU-CRD 版)
#   复刻 Luo X. et al. "Development and validation of an interpretable
#   machine learning model for predicting in-hospital mortality for
#   ischemic stroke patients in ICU", Int J Med Inform 2025;198:105874。
#
#   与 MIMIC 版 11_ml_section.R 完全同源 —— 因为 main_analysis_eicu.R 与
#   main_analysis.R 向 labeled_df.Rdata 写入的对象同名(df / tbl1_vars /
#   covars_m3 / EXPOSURES / EXPOSURE_LABELS ...), 主要结局列同为 inhosp_outcome,
#   8 个 IR 指标同为 EXPOSURES。本模块与数据库无关, 仅依赖上述已保存对象。
#
#   ★ 与 MIMIC 版的唯一实质差异: eICU 仅"院内死亡"单终点。
#     本 ML 部分本就只用院内死亡 (ML_OUTCOME = inhosp_outcome), 故无需任何改动。
#   ★ 自包含改动: 本文件自行 p_load(xgboost)(eICU 主脚本未加载), 以支持【直接训练 XGBoost】
#     与 TreeSHAP; 单独运行也可用。
#
# 总体思路 (★审稿意见#5 防泄漏修订版 —— 先切分, 后一切开发步骤):
#   Step 0d  【先切分】数据按结局 7:3 分层切分为 训练/内部验证; 此后预处理/插补/
#            特征筛选/SMOTE/调参【全部只在训练集】进行, 内部验证集仅在 Step 13 评估时用一次。
#   Step 0d  【分区独立插补】用 cohort_raw(raw_data.Rdata) 还原 NA 模式, 训练/验证两分区
#            【各自独立】单插补(mice, m=1, PMM; 与"ML 用单插补"口径一致), 训练集信息
#            不进入验证集插补; >20% 高缺失(按训练集缺失率判定)不插补并剔出 ML 池。
#   Step 11  特征筛选: LASSO ∩ Boruta ∩ RFE 三法取交集 —— ★仅在训练集上运行
#            ★(IR_IN_SELECTION, 默认 TRUE): 8 个 IR 指标(EXPOSURES)与临床变量平等进入
#              三法筛选; 结果拆为 sel_clin(临床幸存)/sel_ir(IR 幸存)。
#            ★原 Step 12 (IR 增量价值比较) 已按需求移除; 最终模型直接用筛选幸存变量。
#   Step 13  多算法临床模型: 最终变量 = sel_ml (三法筛选幸存的 临床+IR 变量) 训练 9 个 ML 模型
#            (LR/KNN/NB/DT/SVM/RF/XGBoost/LightGBM/ANN; 仅训练集 + SMOTE + 5 折网格调参),
#            在【从未参与开发】的内部验证集上评估一次
#            (AUROC/Acc/Sens/Spec/PPV/NPV/F1/Kappa/Youden/PR-AUC/Brier + 校准曲线 + DCA),
#            最优模型 = 内部验证 AUROC 最优者(开发队列内部选定并【锁定】);
#            ★MIMIC-IV 外部队列仅对锁定模型评估一次, 不参与任何算法/变量/超参选择;
#            最后对最优模型做 SHAP (beeswarm / dependence / 个体 force/waterfall)
#
# 与本文一致的关键口径:
#   - 主要结局 = 院内死亡 (inhosp_outcome, 与生存分析终点一致)
#   - ★先切分后一切(审稿意见#5): 7:3 分层切分 (train:test) 在插补/特征筛选/调参【之前】
#     完成 (Step 0d), idx_tr 全程复用; 内部验证集仅在 Step 13 评估时使用一次
#   - 训练集 SMOTE 至 1:1 (仅训练集; 测试集保持原始分布)
#   - ★本次改动: 因 SMOTE 把训练正类抬到 1:1, 会使测试集预测概率系统性高估(校准曲线整体偏下),
#     Step 13 增加【概率重标定】(RECALIBRATE, 默认 "prior"): 用原始未 SMOTE 训练集的真实患病率
#     把预测概率校正回真实尺度, 再据校正后概率算 性能/校准/DCA。重标定为单调变换 -> 不改 AUROC/ROC,
#     仅修正校准(截距/斜率)、Brier 及 0.5 切点的 Sens/Spec/F1。(亦可直接设 USE_SMOTE<-FALSE 从源头避免高估。)
#   - 连续变量 min-max 归一到 [0,1] (scaler 仅用训练集拟合, 防信息泄漏)
#   - 5 折交叉验证调参; LASSO 用 10 折 CV + lambda.min
#   - RFE 用随机森林基模型 + 5 折 CV + AUC; AUROC 95%CI 用 Bootstrap 1000 次
#   - ★本次改动: Step 11 三法筛选【前】对连续候选变量标准化(均值0/标准差1; 因子/二分类 No/Yes 不动),
#     与生存分析 LASSO 口径一致。LASSO 量纲敏感(已配合 glmnet standardize=FALSE); Boruta/RFE 为随机森林基、
#     对单调变换不敏感, 标准化不改变其结果, 仅为三法口径统一。Step 12/13 建模仍用各自的 min-max 归一(不受影响)。
#   - ★本次优化(调参 + 指标): 由 TUNE_MODE 控制("grid" 默认 / "length" 回退旧行为)。
#     (1) 调参由 tuneLength(自动粗网格) 升级为【显式网格搜索 tuneGrid + 5 折 CV, 以 AUC 选优】,
#         覆盖 8/9 个模型(LR 无超参除外): KNN(k) / NB(laplace,usekernel,adjust) / DT(cp) /
#         SVM(sigma,C) / RF(mtry) / ANN(size,decay); XGBoost、LightGBM 另对核心超参做网格搜索
#         (每个组合做一次 5 折 xgb.cv / lgb.cv, 选 CV-AUC 最优组合 + 其最佳迭代轮数)。
#     (2) 评估指标在原 AUROC/Acc/Sens/Spec/F1/Brier 上新增 Precision(PPV)/NPV/Kappa/Youden's J/PR-AUC,
#         并报告约登最优切点及该切点的 Sens/Spec/F1(固定 0.5 切点在 SMOTE/重标定后未必最优)。
#     注: caret 的 rf 仅调 mtry(ntree 默认 500)、rpart 仅调 cp; 若要扩展见 make_tune_grid() 处注释。
#         网格搜索比 tuneLength 慢: 可设 ALLOW_PARALLEL<-TRUE(需先注册 doParallel) 或 TUNE_MODE<-"length" 提速。
#
# 运行方式 (两者皆可):
#   (A) 在 main_analysis_eicu.R 末尾追加  source("11_ml_section_eicu.R")  —— 复用内存对象
#   (B) 单独运行本文件 —— 自动从 ./load/labeled_df.Rdata 载入 df/EXPOSURES 等
#   注: 两个数据库请在各自的工作目录内运行 (主脚本本身的 ./csv、./docx 输出文件名相同,
#       同目录运行会互相覆盖; ML 输出同理)。
######################################################################

# ====================================================================
# 0. 环境引导 + 依赖
# ====================================================================
if (!exists("df") || !exists("covars_m3") || !exists("EXPOSURES")) {
  .ld <- file.path(if (exists("DIR_LOAD")) DIR_LOAD else "./load", "labeled_df.Rdata")
  if (!file.exists(.ld))
    stop("未找到 labeled_df.Rdata, 请先运行 main_analysis_eicu.R (至 Step 3b/3c 即可)。")
  load(.ld)   # 提供: df, tbl1_vars, cat_vars, covars_m2, covars_m3, EXPOSURES, EXPOSURE_LABELS ...
}
# ★审稿意见#5: 还需插补前的原始队列 cohort_raw (raw_data.Rdata) 以还原 NA 模式 -> 分区独立插补
if (!exists("cohort_raw")) {
  .rd <- file.path(if (exists("DIR_LOAD")) DIR_LOAD else "./load", "raw_data.Rdata")
  if (!file.exists(.rd))
    stop("未找到 raw_data.Rdata, 请先运行 main_analysis_eicu.R 的 Step 1 (会保存 population/cohort_raw)。")
  .e <- new.env(); load(.rd, envir = .e); cohort_raw <- .e$cohort_raw
  rm(.e)
}
stopifnot(nrow(cohort_raw) == nrow(df))   # df 由 cohort_raw 经插补+标签化而来, 行序一致
if (!exists("DIR_CSV"))  DIR_CSV  <- "./csv"
if (!exists("DIR_DOCX")) DIR_DOCX <- "./docx"
if (!exists("DIR_LOAD")) DIR_LOAD <- "./load"
for (d in c(DIR_CSV, DIR_DOCX, DIR_LOAD)) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

if (!require("pacman")) install.packages("pacman")
# 核心依赖 (必需); ★ 本模块自带 xgboost(eICU 主脚本未加载), 以支持【直接训练 XGBoost】 + TreeSHAP
pacman::p_load(tidyverse, glmnet, Boruta, caret, randomForest,
               e1071, naivebayes, rpart, kernlab, nnet, xgboost,
               pROC, shapviz)
# caret/部分模型可能需要 plyr —— 仅安装、不 attach, 以免遮蔽 dplyr 同名函数
if (!requireNamespace("plyr", quietly = TRUE)) install.packages("plyr")
# 可选依赖 (缺失则相关步骤自动降级/跳过)
for (.p in c("lightgbm", "kernelshap", "ggvenn"))
  if (!requireNamespace(.p, quietly = TRUE))
    cat(sprintf("  [提示] 未检测到 %-11s 相关步骤将自动降级/跳过 (如需: install.packages('%s'))\n", .p, .p))

# ====================================================================
# 0b. 全局参数 (可按需调整)
# ====================================================================
SEED            <- 20260531           # 复现随机过程 (CV / SMOTE / RF / 切分)
ML_OUTCOME      <- "inhosp_outcome"   # 主要结局: 院内死亡 (eICU 唯一终点)
TRAIN_PROP      <- 0.70               # 训练:测试 = 7:3 (本文)
USE_SMOTE       <- TRUE               # 训练集 SMOTE 至 1:1 (本文)
SMOTE_K         <- 5                  # SMOTE 近邻数
RECALIBRATE     <- "prior"            # ★概率重标定(修复 SMOTE 致测试集高估): "prior"(按真实患病率校正截距, 默认/最稳) / "platt"(逻辑重标定, 截距+斜率; 拟合不稳自动回退 prior) / "none"(不校正)
BOOT_N          <- 1000               # AUROC 95%CI Bootstrap 次数 (本文)
BORUTA_MAXRUNS  <- 200                # Boruta 最大迭代
TUNE_MODE       <- "grid"             # ★调参模式: "grid"=显式网格搜索(本次升级, 默认) / "length"=旧的 tuneLength 自动粗网格
TUNE_LEN        <- 3                  # tuneLength 规模 (仅 TUNE_MODE=="length" 或某算法无网格如 glm 时作后备)
ALLOW_PARALLEL  <- FALSE              # ★caret 网格搜索是否并行 (TRUE 需先注册 doParallel 后端; 默认 FALSE 保证可复现/稳定)
MIN_INTERSECT   <- 2                  # 三法交集最少变量数; 不足则退回">=2法共选"
IR_IN_SELECTION <- TRUE               # ★ 8 个 IR 指标纳入 Step11 特征筛选(LASSO∩Boruta∩RFE; 本次定稿口径)
                                      #   TRUE  = 8 个 IR 与临床变量平等参与三法筛选 (训练集内)
                                      #   FALSE = 候选池剔除 IR (旧口径)
SHAP_MODEL_NAME <- NA                 # SHAP 用哪个模型: NA=测试集 AUROC 最优者 (本文选 RF, 可设 "RF")
SHAP_BG_N       <- 100                # SHAP 背景样本数
SHAP_EXP_N      <- 200                # SHAP 解释样本数 (从测试集抽样)

set.seed(SEED)

# 结局向量 (0/1)
.y_raw <- df[[ML_OUTCOME]]
if (is.factor(.y_raw)) .y_raw <- as.integer(.y_raw) - 1L
y_all <- as.integer(.y_raw)
stopifnot(all(y_all %in% c(0, 1)), length(unique(y_all)) == 2)
cat(sprintf(">>> ML 主要结局 = %s; 阳性(死亡)率 = %.1f%% (%d/%d)\n",
            ML_OUTCOME, 100 * mean(y_all), sum(y_all), length(y_all)))

# ====================================================================
# 0d. ★审稿意见#5 防泄漏重排: 先切分 -> 两分区独立插补 -> 后续开发仅用训练集
# ====================================================================
cat("\n>>> Step 0d: 先切分 + 分区独立插补 (防信息泄漏)\n")

# --- (i) 还原插补前 NA 模式 ---
#   df 是"第 1 完成集"经标签化后的数据; 其中曾被插补的单元格须先还原为 NA,
#   才能在两个分区内独立重插补。cohort_raw 与 df 行序一致(主脚本 Step1b->3 无行过滤, 已核实)。
.na_restore_cols <- intersect(names(df), names(cohort_raw))
.n_restored <- 0L
for (.v in .na_restore_cols) {
  .na_pos <- is.na(cohort_raw[[.v]])
  .n_restored <- .n_restored + sum(.na_pos & !is.na(df[[.v]]))
  df[[.v]][.na_pos] <- NA
}
cat(sprintf("  NA 模式已还原 (%d 个单元格回到缺失态, 供分区独立插补)\n", .n_restored))

# --- (ii) 7:3 分层切分 —— ★先于插补/筛选/调参 (原流程在特征筛选之后切分, 存在泄漏) ---
set.seed(SEED)
idx_tr <- caret::createDataPartition(factor(y_all), p = TRAIN_PROP, list = FALSE)[, 1]
cat(sprintf("  数据切分(先于插补/筛选): 训练集 %d (死亡 %d) / 内部验证集 %d (死亡 %d)\n",
            length(idx_tr), sum(y_all[idx_tr]),
            length(y_all) - length(idx_tr), sum(y_all[-idx_tr])))

# --- (iii) 两分区独立单插补 (mice, m=1, PMM; 与"ML 采用单插补"口径一致) ---
#   >20% 高缺失(按【训练集】缺失率判定)不插补并保持 NA -> 下游自动剔出 ML 候选池(与主分析口径一致)
if (!requireNamespace("mice", quietly = TRUE)) install.packages("mice")
.imp_cols <- intersect(union(union(tbl1_vars, EXPOSURES), ML_OUTCOME), names(df))
.miss_tr  <- vapply(df[idx_tr, .imp_cols, drop = FALSE], function(x) mean(is.na(x)), numeric(1))
.drop_hi  <- setdiff(names(.miss_tr)[.miss_tr > 0.20], EXPOSURES)
if (length(.drop_hi))
  cat("  [缺失剔除] 训练集缺失率>20%, 不插补并剔出 ML 池: ", paste(.drop_hi, collapse = ", "), "\n")

impute_partition <- function(d_part, cols, drop_hi, seed) {
  dat <- d_part[, cols, drop = FALSE]
  chr <- names(dat)[vapply(dat, is.character, logical(1))]
  dat[chr] <- lapply(dat[chr], factor)                 # mice 需要因子而非字符
  ini  <- mice::mice(dat, maxit = 0, printFlag = FALSE)
  meth <- ini$method
  meth[intersect(drop_hi, names(meth))] <- ""          # 高缺失列不插补
  nums <- names(dat)[vapply(dat, is.numeric, logical(1))]
  sel  <- nums[meth[nums] != ""]
  meth[sel] <- "pmm"                                   # 数值列 PMM (与主分析一致)
  imp  <- mice::mice(dat, m = 1, maxit = 50, method = meth,
                     predictorMatrix = ini$predictorMatrix, printFlag = FALSE, seed = seed)
  done <- mice::complete(imp, 1)
  d_part[, cols] <- done[, cols]
  d_part
}
cat("  训练集内独立插补 ...\n")
df_tr_imp <- impute_partition(df[idx_tr, , drop = FALSE],  .imp_cols, .drop_hi, SEED)
cat("  内部验证集内独立插补 (不使用训练集任何信息) ...\n")
df_te_imp <- impute_partition(df[-idx_tr, , drop = FALSE], .imp_cols, .drop_hi, SEED + 1L)

# --- (iv) 重组 df: 前 n_tr 行为训练分区, 其余为内部验证分区; idx_tr / y_all 同步 ---
df <- dplyr::bind_rows(df_tr_imp, df_te_imp)
idx_tr <- seq_len(nrow(df_tr_imp))
.y_raw <- df[[ML_OUTCOME]]
if (is.factor(.y_raw)) .y_raw <- as.integer(.y_raw) - 1L
y_all <- as.integer(.y_raw)
rm(df_tr_imp, df_te_imp)
cat(sprintf("  分区独立插补完成; 训练 %d / 验证 %d (两分区插补模型互不可见)\n",
            length(idx_tr), nrow(df) - length(idx_tr)))

# ====================================================================
# 0c. 通用工具函数 (设计矩阵 / 归一 / SMOTE / 训练评估 / 指标)
# ====================================================================

# 设计矩阵: 因子 one-hot 展开并删截距; 同时返回"列 -> 原变量"映射 (用于把哑变量名映射回原变量)
make_design <- function(vars, data) {
  vars <- intersect(vars, names(data))
  mm <- model.matrix(as.formula(paste("~", paste(vars, collapse = " + "))), data = data)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  col2var <- vapply(colnames(mm), function(cn) {
    hit <- vars[vapply(vars, function(v) startsWith(cn, v), logical(1))]
    if (length(hit)) hit[which.max(nchar(hit))] else cn   # 取最长匹配, 正确处理 pt/ptt 等前缀重叠
  }, character(1))
  list(X = mm, map = col2var)
}

# min-max 归一: 仅用训练集拟合 (防泄漏), 应用到训练/测试
fit_scaler   <- function(X) { mn <- apply(X, 2, min); mx <- apply(X, 2, max)
                              rng <- mx - mn; rng[rng == 0] <- 1; list(min = mn, range = rng) }
apply_scaler <- function(X, sc) sweep(sweep(X, 2, sc$min, "-"), 2, sc$range, "/")

# 按既定 7:3 切分 + 训练集归一 (一次切分, 全程复用 idx_tr); 同时丢弃训练集内零方差列
prep_split <- function(vars) {
  d <- make_design(vars, df)
  Xtr0 <- d$X[idx_tr, , drop = FALSE]; Xte0 <- d$X[-idx_tr, , drop = FALSE]
  keep <- which(apply(Xtr0, 2, function(z) length(unique(z)) > 1))   # 去训练集常量列(防某因子水平在训练集缺席)
  Xtr0 <- Xtr0[, keep, drop = FALSE];  Xte0 <- Xte0[, keep, drop = FALSE]
  sc <- fit_scaler(Xtr0)
  list(Xtr = apply_scaler(Xtr0, sc), Xte = apply_scaler(Xte0, sc),
       ytr = y_all[idx_tr], yte = y_all[-idx_tr], map = d$map[colnames(Xtr0)])
}

# 训练集 SMOTE 至 ~1:1 (本文); 仅训练集; 少数类过少或包缺失则自动跳过
do_smote <- function(X, y01, K = SMOTE_K, seed = SEED) {
  if (!USE_SMOTE || !requireNamespace("smotefamily", quietly = TRUE)) {
    if (USE_SMOTE && !requireNamespace("smotefamily", quietly = TRUE))
      cat("  [SMOTE] 未安装 smotefamily, 跳过过采样 (如需: install.packages('smotefamily'))\n")
    return(list(X = X, y = y01))
  }
  if (min(table(y01)) < K + 1) { cat("  [SMOTE] 少数类样本过少, 跳过\n"); return(list(X = X, y = y01)) }
  set.seed(seed)
  sm  <- smotefamily::SMOTE(X = as.data.frame(X), target = as.character(y01), K = K)
  d   <- sm$data; cls <- d[["class"]]; d[["class"]] <- NULL
  list(X = as.matrix(sapply(d, as.numeric)), y = as.integer(cls))
}

# ★各 caret 算法的显式调参网格; p = 设计矩阵列数(one-hot 后), 仅 rf 的 mtry 依赖 p。
#   说明: caret 的 method="rf" 内置可调参数只有 mtry(ntree 固定默认 500), method="rpart" 只有 cp。
#   若要进一步搜 RF 的 ntree 或 DT 的 maxdepth, 需:
#     - RF: 在外层对 ntree 取值循环, 每个 ntree 用 fit_eval_caret 训一遍按 CV-AUC 取最优(改建模流程); 或
#     - DT: 把 method 改为 "rpart2"(调 maxdepth) 或自写 caret model 同时调 cp+maxdepth。
#   本版保持原文方法不变(rf→mtry, rpart→cp), 仅由自动网格升级为显式网格。
make_tune_grid <- function(method, p) {
  switch(method,
    "glm"         = NULL,                                            # 逻辑回归无超参 -> 走 tuneLength 后备(实际被忽略)
    "knn"         = expand.grid(k = c(3, 5, 7, 9, 11, 15, 19, 25)),
    "naive_bayes" = expand.grid(laplace   = c(0, 0.5, 1),
                                usekernel = c(FALSE, TRUE),
                                adjust    = c(1)),
    "rpart"       = expand.grid(cp = c(0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1)),
    "svmRadial"   = expand.grid(sigma = c(0.005, 0.01, 0.02, 0.05, 0.1),
                                C     = c(0.25, 0.5, 1, 2, 4)),
    "rf"          = expand.grid(mtry = sort(unique(pmin(p,
                                  pmax(1, floor(sqrt(p) * c(0.5, 1, 1.5, 2))))))),
    "nnet"        = expand.grid(size  = c(2, 4, 6, 8, 10),
                                decay = c(0, 1e-3, 1e-2, 1e-1)),
    NULL)
}

# 用 caret 训练单个模型: 5 折 CV + 网格搜索 (metric=ROC), 返回 模型 + 测试集 P(Yes)
#   tuneGrid 不传时: TUNE_MODE=="grid" -> 自动用 make_tune_grid() 构造显式网格; 否则用 tuneLength。
fit_eval_caret <- function(method, Xtr, ytr01, Xte, tuneLength = TUNE_LEN,
                           tuneGrid = NULL, seed = SEED, extra = list()) {
  ytr  <- factor(ifelse(ytr01 == 1, "Yes", "No"), levels = c("No", "Yes"))
  ctrl <- caret::trainControl(method = "cv", number = 5, classProbs = TRUE,
                              summaryFunction = caret::twoClassSummary,
                              allowParallel = ALLOW_PARALLEL)
  if (is.null(tuneGrid) && identical(TUNE_MODE, "grid"))
    tuneGrid <- make_tune_grid(method, ncol(Xtr))                   # 自动构造该算法显式网格
  set.seed(seed)
  base_args <- list(x = as.data.frame(Xtr), y = ytr, method = method,
                    metric = "ROC", trControl = ctrl)
  tune_args <- if (!is.null(tuneGrid) && nrow(tuneGrid) > 0)
    list(tuneGrid = tuneGrid) else list(tuneLength = tuneLength)    # 有网格走网格, 否则走 tuneLength
  mod <- tryCatch(suppressWarnings(do.call(caret::train, c(base_args, tune_args, extra))),
                  error = function(e) { cat(sprintf("    [%s] 训练失败: %s\n", method, conditionMessage(e))); NULL })
  if (is.null(mod)) return(NULL)
  p <- as.numeric(predict(mod, newdata = as.data.frame(Xte), type = "prob")[, "Yes"])
  list(model = mod, prob = p)
}

# PR-AUC (精确率-召回率曲线下面积; 不平衡数据比 ROC-AUC 更敏感; 手算梯形积分, 无需新包)
pr_auc <- function(p, y) {
  o  <- order(p, decreasing = TRUE); y <- y[o]
  P  <- sum(y == 1); if (P == 0) return(NA_real_)
  tp <- cumsum(y == 1); fp <- cumsum(y == 0)
  recall    <- c(0, tp / P)
  precision <- c(tp[1] / (tp[1] + fp[1]), tp / (tp + fp))
  sum(diff(recall) * (head(precision, -1) + tail(precision, -1)) / 2)
}

# 由 测试集 P(Yes) 计算一整套判别/校准指标 (对齐本文 Table 2, 并扩充)
#   扩充: + Precision(PPV)/NPV/Kappa/Youden/PR-AUC + 约登最优切点(及该切点 Sens/Spec/F1)
metrics_from_prob <- function(p, yte01, boot_n = BOOT_N, seed = SEED) {
  yte   <- factor(ifelse(yte01 == 1, "Yes", "No"), levels = c("No", "Yes"))
  roc_o <- pROC::roc(yte01, p, levels = c(0, 1), direction = "<", quiet = TRUE)
  set.seed(seed)
  ci    <- as.numeric(pROC::ci.auc(roc_o, method = "bootstrap", boot.n = boot_n))

  # (1) 固定 0.5 切点的混淆矩阵指标
  pred05 <- factor(ifelse(p >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
  cm05   <- caret::confusionMatrix(pred05, yte, positive = "Yes", mode = "everything")

  # (2) 约登最优切点(由 ROC 求)及其指标 —— SMOTE/重标定后 0.5 常非最优, 故并列报告
  thr_opt <- tryCatch({
    co <- pROC::coords(roc_o, x = "best", best.method = "youden",
                       ret = "threshold", transpose = FALSE)
    as.numeric(co[["threshold"]])[1]
  }, error = function(e) 0.5)
  if (length(thr_opt) == 0 || !is.finite(thr_opt)) thr_opt <- 0.5
  predY <- factor(ifelse(p >= thr_opt, "Yes", "No"), levels = c("No", "Yes"))
  cmY   <- caret::confusionMatrix(predY, yte, positive = "Yes", mode = "everything")

  data.frame(
    # —— 判别能力 ——
    AUROC  = round(ci[2], 3), AUC_lo = round(ci[1], 3), AUC_hi = round(ci[3], 3),
    PR_AUC = round(pr_auc(p, yte01), 3),
    # —— 固定 0.5 切点的分类指标 ——
    Accuracy    = round(unname(cm05$overall["Accuracy"]), 3),
    Sensitivity = round(unname(cm05$byClass["Sensitivity"]), 3),
    Specificity = round(unname(cm05$byClass["Specificity"]), 3),
    Precision   = round(unname(cm05$byClass["Precision"]), 3),        # = PPV 阳性预测值
    NPV         = round(unname(cm05$byClass["Neg Pred Value"]), 3),   # 阴性预测值
    F1          = round(unname(cm05$byClass["F1"]), 3),
    Kappa       = round(unname(cm05$overall["Kappa"]), 3),
    Youden      = round(unname(cm05$byClass["Sensitivity"]) +
                          unname(cm05$byClass["Specificity"]) - 1, 3),
    # —— 校准 ——
    Brier       = round(mean((p - yte01)^2), 3),
    # —— 约登最优切点 ——
    Thr_opt     = round(thr_opt, 3),
    Sens_optThr = round(unname(cmY$byClass["Sensitivity"]), 3),
    Spec_optThr = round(unname(cmY$byClass["Specificity"]), 3),
    F1_optThr   = round(unname(cmY$byClass["F1"]), 3),
    row.names = NULL)
}

# 概率重标定: 修复"训练集 SMOTE 改变患病率 -> 测试集系统性高估"。
#   在【原始(未 SMOTE)训练集】上按真实患病率拟合校正映射, 再套到任意预测概率(测试/外部), 不泄漏测试。
#   - "prior": 先验/截距校正  logit(p*) = logit(p) + logit(π) - logit(τ)  (仅平移截距, 最稳健, 无拟合风险);
#              τ = 训练时正类占比(SMOTE 后≈0.5), π = 原始训练集真实正类占比; 未 SMOTE 时 τ=π -> 无校正。
#   - "platt": 逻辑重标定  logit(p*) = a + b·logit(p)  (截距 a + 斜率 b, 由原始训练集拟合); 拟合不稳
#              (不收敛/斜率≤0 或 >10, 多因树模型在训练集近乎可分)时自动回退 prior。
#   - "none" : 原样返回。
#   返回"输入概率向量 -> 校正后概率向量"的函数(每个模型各拟合一个)。
#   注: 重标定是单调变换 -> 不改变 AUROC/ROC 与 DeLong; 仅修正校准(截距/斜率)、Brier 及 0.5 切点 Sens/Spec/F1。
make_recalibrator <- function(p_dev, y_dev, method = RECALIBRATE, tau = NULL) {
  .clamp  <- function(p) pmin(pmax(p, 1e-6), 1 - 1e-6)
  pi_true <- mean(y_dev)
  prior_fun <- function() {
    tt <- if (is.null(tau)) pi_true else tau
    shift <- stats::qlogis(.clamp(pi_true)) - stats::qlogis(.clamp(tt))   # logit(π) - logit(τ)
    function(p) stats::plogis(stats::qlogis(.clamp(p)) + shift)
  }
  if (method == "none")  return(function(p) p)
  if (method == "prior") return(prior_fun())
  # platt: 逻辑重标定(在原始训练集预测上拟合)
  g <- suppressWarnings(stats::glm(y_dev ~ stats::qlogis(.clamp(p_dev)), family = binomial))
  a <- unname(stats::coef(g)[1]); b <- unname(stats::coef(g)[2])
  if (!is.finite(a) || !is.finite(b) || b <= 0 || b > 10) {
    cat("    [重标定] platt 拟合不稳(训练集可能过拟合/可分) -> 退回 prior 截距校正\n")
    return(prior_fun())
  }
  function(p) stats::plogis(a + b * stats::qlogis(.clamp(p)))
}

# ====================================================================
# 11 ML 特征筛选: LASSO ∩ Boruta ∩ RFE (★ IR_IN_SELECTION 控制是否纳入 8 个 IR 指标)
# ====================================================================
cat(sprintf("\n>>> Step 11: ML 特征筛选 (LASSO ∩ Boruta ∩ RFE; 候选池%s 8 个 IR 指标)\n",
            if (IR_IN_SELECTION) "纳入" else "剔除"))

# 候选池 = Table 1 临床变量 (已排除 ID/时间/派生/终点列); race 始终不作模型协变量而剔除。
# ★ IR_IN_SELECTION 控制 8 个 IR 指标是否进入候选池一同筛选:
#   TRUE  -> 8 个 IR 与其余临床变量平等地进入 LASSO ∩ Boruta ∩ RFE (本次需求)
#   FALSE -> 沿用原口径: 候选池剔除 8 个 IR, 只在临床变量中筛选
if (IR_IN_SELECTION) {
  ml_cand <- union(setdiff(tbl1_vars, "race"), EXPOSURES)   # 临床变量 + 8 个 IR (race 仍排除)
} else {
  ml_cand <- setdiff(tbl1_vars, c(EXPOSURES, "race"))       # ★本次定稿口径: 不含 IR
  .leak <- intersect(ml_cand, EXPOSURES)                    # 防呆: IR 不得残留
  if (length(.leak))
    stop("IR 指标仍残留在候选池: ", paste(.leak, collapse = ", "))
  cat(sprintf("  [IR 排除] 已确认 %d 个 IR 指标不进入筛选: %s\n",
              length(EXPOSURES), paste(EXPOSURES, collapse = ", ")))
}
# ★ 关键: 剔除仍含缺失(NA)的候选变量。
#   df 虽经 Step 1b 多重插补, 但缺失率 >20% 的协变量按本文做法未被插补, 仍以 NA 形态
#   保留在 df(经 extra_cols 并入)。这些 NA 列会让下游 model.matrix() 触发 na.omit 整行删除,
#   造成设计矩阵行数 << y, 进而 glmnet 报 "number of observations in y not equal to rows of x"。
#   按本文口径(>20% 缺失即剔除), 这里直接把含 NA 的候选剔出 ML 特征筛选池(保留全部样本行)。
.na_cols <- ml_cand[vapply(df[idx_tr, ml_cand, drop = FALSE], anyNA, logical(1))]  # ★仅按训练集判定(审稿意见#5)
if (length(.na_cols)) {
  cat("  [缺失剔除] 以下候选仍含 NA(未插补, 多为 >20% 高缺失), 已剔出 ML 特征池: ",
      paste(.na_cols, collapse = ", "), "\n")
  ml_cand <- setdiff(ml_cand, .na_cols)
}
# 去近零方差 (防 one-hot 后出现常量列 / 单水平因子)
.nzv <- caret::nearZeroVar(df[idx_tr, ml_cand, drop = FALSE])   # ★仅按训练集判定(审稿意见#5)
if (length(.nzv)) {
  cat("  [nearZeroVar] 剔除近零方差候选: ", paste(ml_cand[.nzv], collapse = ", "), "\n")
  ml_cand <- ml_cand[-.nzv]
}
# ★ 若开启 IR 入池, 提示哪些 IR 因 NA / 近零方差被自动剔出(无法参与筛选)
if (IR_IN_SELECTION) {
  .ir_dropped <- setdiff(EXPOSURES, ml_cand)
  if (length(.ir_dropped))
    cat("  [提示] 以下 IR 指标因含 NA 或近零方差被剔出特征池, 未参与本次筛选: ",
        paste(.ir_dropped, collapse = ", "), "\n")
}
cat(sprintf("  候选变量池 (n=%d): %s\n", length(ml_cand), paste(ml_cand, collapse = ", ")))

# ★ 本次改动: ML 特征筛选【前】对【连续候选变量】标准化(均值0/标准差1); 因子/二分类(No/Yes)不动。
#   - LASSO 对量纲敏感: 标准化使 L1 惩罚在各连续变量间可比(下方已设 glmnet standardize=FALSE, 避免重复标准化);
#   - Boruta/RFE 为随机森林基, 对单调变换不敏感, 结果不受标准化影响, 此处统一喂入标准化数据仅为口径一致;
#   - 仅作用于 Step 11 特征筛选所用副本 df_sel; 全局 df 不变 -> Step 12/13 建模仍用原 df + 各自 min-max 归一(不受影响)。
df_sel <- df[idx_tr, , drop = FALSE]   # ★审稿意见#5: 特征筛选仅用训练集
y_tr   <- y_all[idx_tr]                # ★训练集结局 (LASSO/Boruta/RFE 均只用 y_tr)
.is_cont_ml <- function(x) {
  if (!is.numeric(x)) return(FALSE)                                   # 因子/字符(含 No/Yes 二分类)不标准化
  u <- unique(x[!is.na(x)]); !(length(u) <= 2 && all(u %in% c(0, 1))) # 排除 0/1 二分类
}
.ml_cont <- ml_cand[vapply(df_sel[ml_cand], .is_cont_ml, logical(1))]
if (length(.ml_cont)) {
  df_sel[.ml_cont] <- lapply(df_sel[.ml_cont], function(x) {
    s <- sd(x, na.rm = TRUE); if (!is.finite(s) || s == 0) s <- 1     # 常量列不缩放(避免除0)
    as.numeric((x - mean(x, na.rm = TRUE)) / s)
  })
  cat(sprintf("  [标准化] ML 筛选前已标准化 %d 个连续候选(均值0/标准差1): %s\n",
              length(.ml_cont), paste(.ml_cont, collapse = ", ")))
}

# --- (1) LASSO: 10 折 CV, lambda.min; ★连续变量已标准化 -> standardize=FALSE ---
.des <- make_design(ml_cand, df_sel)
set.seed(SEED)
.fit_path <- glmnet::glmnet(.des$X, y_tr, family = "binomial", alpha = 1, standardize = FALSE)  # 供 11b 路径图
set.seed(SEED)
.cvf <- glmnet::cv.glmnet(.des$X, y_tr, family = "binomial", alpha = 1, nfolds = 10, standardize = FALSE)
.co  <- as.matrix(coef(.cvf, s = "lambda.min"))
.nz  <- setdiff(rownames(.co)[which(.co[, 1] != 0)], "(Intercept)")
lasso_vars <- unique(.des$map[.nz])                       # 哑变量名映射回原变量
cat(sprintf("  LASSO (lambda.min) 选中 %d : %s\n", length(lasso_vars), paste(lasso_vars, collapse = ", ")))

# --- (2) Boruta: 随机森林重要性置换检验; TentativeRoughFix 裁决待定项 ---
set.seed(SEED)
.bor_raw <- Boruta::Boruta(x = df_sel[ml_cand], y = factor(y_tr), doTrace = 0, maxRuns = BORUTA_MAXRUNS)
.bor <- Boruta::TentativeRoughFix(.bor_raw)
boruta_vars <- Boruta::getSelectedAttributes(.bor, withTentative = FALSE)
cat(sprintf("  Boruta 选中 %d : %s\n", length(boruta_vars), paste(boruta_vars, collapse = ", ")))

# --- (3) RFE: 随机森林基模型 + 5 折 CV + AUC (本文); 失败则退回 Accuracy ---
.y_fac <- factor(ifelse(y_tr == 1, "Yes", "No"), levels = c("No", "Yes"))
.rfe_fit <- NULL
rfe_vars <- tryCatch({
  ff <- caret::rfFuncs
  ff$summary <- caret::twoClassSummary
  ff$pred <- function(object, x) {
    cbind(data.frame(pred = predict(object, x)), as.data.frame(predict(object, x, type = "prob")))
  }
  rc <- caret::rfeControl(functions = ff, method = "cv", number = 5, verbose = FALSE)
  set.seed(SEED)
  .rfe_fit <<- caret::rfe(x = df_sel[ml_cand], y = .y_fac, sizes = 2:length(ml_cand),
                          rfeControl = rc, metric = "ROC")
  caret::predictors(.rfe_fit)
}, error = function(e) {
  cat("  [RFE] ROC 配置失败, 退回 Accuracy: ", conditionMessage(e), "\n")
  rc <- caret::rfeControl(functions = caret::rfFuncs, method = "cv", number = 5, verbose = FALSE)
  set.seed(SEED)
  .rfe_fit <<- caret::rfe(x = df_sel[ml_cand], y = .y_fac, sizes = 2:length(ml_cand),
                          rfeControl = rc, metric = "Accuracy")
  caret::predictors(.rfe_fit)
})
cat(sprintf("  RFE 选中 %d : %s\n", length(rfe_vars), paste(rfe_vars, collapse = ", ")))

# --- 三法交集 (本文 Venn 取交集) ---
sel_ml <- Reduce(intersect, list(lasso_vars, boruta_vars, rfe_vars))

# 选择矩阵 + Venn 图 + 落盘
.sets <- list(LASSO = lasso_vars, Boruta = boruta_vars, RFE = rfe_vars)
.allv <- sort(unique(unlist(.sets)))
sel_matrix <- data.frame(
  variable = .allv,
  LASSO  = .allv %in% lasso_vars,
  Boruta = .allv %in% boruta_vars,
  RFE    = .allv %in% rfe_vars)
sel_matrix$n_methods <- rowSums(sel_matrix[, c("LASSO", "Boruta", "RFE")])
sel_matrix$intersection <- sel_matrix$variable %in% sel_ml
sel_matrix <- sel_matrix[order(-sel_matrix$n_methods, sel_matrix$variable), ]
write.csv(sel_matrix, file.path(DIR_CSV, "ml_feature_selection.csv"), row.names = FALSE)

# ★保存三法筛选的拟合对象, 供 11b_ml_selection_figures.R 直接绘图 (保证图与本文筛选完全一致,
#   且无需再次跑筛选 -> 也就不会再次触碰全量数据)
saveRDS(list(cvf = .cvf, fit_path = .fit_path, bor_raw = .bor_raw, bor_fixed = .bor,
             rfe_fit = .rfe_fit, ml_cand = ml_cand, col2var = .des$map,
             lasso_vars = lasso_vars, boruta_vars = boruta_vars, rfe_vars = rfe_vars,
             IR_IN_SELECTION = IR_IN_SELECTION, seed = SEED),
        file.path(DIR_LOAD, "ml_selection_fits.rds"))

if (requireNamespace("ggvenn", quietly = TRUE)) {
  .vp <- ggvenn::ggvenn(.sets, show_percentage = FALSE,
                        fill_color = c("#00BA38", "#2E9FDF", "#F8766D"),
                        stroke_size = 0.5, set_name_size = 5, text_size = 4) +
    ggplot2::ggtitle("Feature selection: intersection of LASSO, Boruta and RFE")
  ggsave(file.path(DIR_DOCX, "fig_ml_venn.pdf"), .vp, width = 7, height = 6)
  cat("  Venn 图已输出: fig_ml_venn.pdf\n")
}

# 交集过小则退回 ">=2 法共选" (并告警), 保证后续可建模
if (length(sel_ml) < MIN_INTERSECT) {
  .alt <- sel_matrix$variable[sel_matrix$n_methods >= 2]
  cat(sprintf("  [警告] 三法交集仅 %d 个 (<%d) -> 退回 '>=2 法共选' %d 个: %s\n",
              length(sel_ml), MIN_INTERSECT, length(.alt), paste(.alt, collapse = ", ")))
  sel_ml <- .alt
}
stopifnot(length(sel_ml) >= 1)

# ★ 把最终筛选集拆成 "临床变量" 与 "IR 指标" 两部分:
#   sel_clin = 选中的非 IR(临床)变量; sel_ir = 选中的 IR 指标(IR 未入池时为空)
#   两者互斥; 下游 Step12 以 sel_clin 为基线, Step13 据此组装最终建模变量。
sel_clin <- setdiff(sel_ml, EXPOSURES)
sel_ir   <- intersect(sel_ml, EXPOSURES)
cat(sprintf("  >>> 最终筛选变量集 (n=%d): %s\n", length(sel_ml), paste(sel_ml, collapse = ", ")))
if (IR_IN_SELECTION) {
  cat(sprintf("      其中 临床变量 %d 个: %s\n", length(sel_clin),
              if (length(sel_clin)) paste(sel_clin, collapse = ", ") else "(无)"))
  cat(sprintf("      其中 IR 指标 %d 个 (经三法筛选幸存): %s\n", length(sel_ir),
              if (length(sel_ir)) paste(sel_ir, collapse = ", ") else "(无 -- 无 IR 通过筛选)"))
}

# ====================================================================
# 11b 7:3 分层切分 —— ★已上移至 Step 0d (审稿意见#5: 切分先于插补/筛选/调参);
#     idx_tr 全程复用; 内部验证集仅在 Step 13 评估时使用一次。
# ====================================================================
stopifnot(length(idx_tr) > 0, max(idx_tr) <= nrow(df))

# ====================================================================
# 13 多算法临床模型: [三法筛选幸存变量 sel_ml = 临床 + IR] 训练 9 个 ML 模型 + 评估 + SHAP
# ====================================================================
# ★ 最终建模变量 = 三法交集筛选结果 sel_ml (8 个 IR 与临床变量平等筛选, 审稿后定稿口径);
#   原 Step 12 (IR 增量价值比较) 已按需求移除, 不再有 best_ir 回退。
final_vars <- sel_ml
if (length(sel_ir) > 0) {
  .ir_note <- sprintf("三法筛选幸存 IR(%d): %s", length(sel_ir), paste(sel_ir, collapse = ", "))
} else {
  .ir_note <- "无 IR 通过三法筛选, 最终模型仅含临床变量"
  cat("  [提示] 无任何 IR 指标通过 LASSO∩Boruta∩RFE; 最终模型仅含临床变量。\n")
}
cat(sprintf("\n>>> Step 13: 多算法临床模型 (最终 %d 变量; %s)\n", length(final_vars), .ir_note))
cat(sprintf("  最终建模变量 (n=%d): %s\n", length(final_vars), paste(final_vars, collapse = ", ")))
sp  <- prep_split(final_vars)
bal <- do_smote(sp$Xtr, sp$ytr)
cat(sprintf("  训练集 %d (SMOTE 后 %d, 死亡占比 %.0f%%); 测试集 %d\n",
            length(sp$ytr), length(bal$y), 100 * mean(bal$y), length(sp$yte)))

# --- 9 个模型: 7 个走 caret(显式网格搜索), XGBoost / LightGBM 单列(自带网格搜索) ---
#   caret 模型的调参网格由 fit_eval_caret() 内部按 make_tune_grid() 自动构造(TUNE_MODE=="grid")。
caret_specs <- list(
  LR      = list(method = "glm",         extra = list()),                                # 无超参
  KNN     = list(method = "knn",         extra = list()),                                # 调 k
  NB      = list(method = "naive_bayes", extra = list()),                                # 调 laplace/usekernel/adjust
  DT      = list(method = "rpart",       extra = list()),                                # 调 cp
  SVM     = list(method = "svmRadial",   extra = list()),                                # 调 sigma/C
  RF      = list(method = "rf",          extra = list()),                                # 调 mtry
  ANN     = list(method = "nnet",        extra = list(trace = FALSE, MaxNWts = 10000))   # 调 size/decay
)

# caret 模型训练集 5 折 CV 的最优 ROC(AUC) —— 开发内指标, 供与内部验证对照
best_cv_auc <- function(mod) {
  if (is.null(mod) || !inherits(mod, "train") || is.null(mod$results)) return(NA_real_)
  res <- mod$results; bt <- mod$bestTune
  for (nm2 in names(bt)) res <- res[res[[nm2]] == bt[[nm2]], ]
  if ("ROC" %in% names(res)) round(res$ROC[1], 3) else NA_real_
}
ml_models <- list(); ml_probs <- list(); ml_cv_auc <- list()
for (nm in names(caret_specs)) {
  cat(sprintf("  训练 %s (%s) ...\n", nm,
              if (TUNE_MODE == "grid" && caret_specs[[nm]]$method != "glm") "5 折网格搜索"
              else if (caret_specs[[nm]]$method == "glm") "无超参/5 折 CV"
              else "5 折 CV/tuneLength"))
  fe <- fit_eval_caret(caret_specs[[nm]]$method, bal$X, bal$y, sp$Xte,
                       tuneLength = TUNE_LEN, extra = caret_specs[[nm]]$extra)
  if (!is.null(fe)) { ml_models[[nm]] <- fe$model; ml_probs[[nm]] <- fe$prob
                      ml_cv_auc[[nm]] <- best_cv_auc(fe$model) }
}

# XGBoost (★直接用 xgboost 训练, 绕开 caret 的 xgbTree 封装 —— 后者常与新版 xgboost 不兼容,
#          折内预测概率变 NaN -> caret 报 "all the ROC metric values are missing" 并 Stopping;
#          直接训练更稳。TUNE_MODE=="grid" 时对核心超参做网格搜索: 每个组合 5 折 xgb.cv 选 CV-AUC + 最佳轮数。)
if (requireNamespace("xgboost", quietly = TRUE)) {
  cat(sprintf("  训练 XGBoost (%s) ...\n", if (TUNE_MODE == "grid") "5 折网格搜索" else "5 折 CV 选轮数"))
  .xgb <- tryCatch({
    dtr_x <- xgboost::xgb.DMatrix(as.matrix(bal$X), label = bal$y)
    .cv_auc_x <- NA_real_

    if (TUNE_MODE == "grid") {
      xgb_grid <- expand.grid(max_depth = c(3, 5, 7), eta = c(0.05, 0.1),
                              subsample = c(0.8, 1.0), colsample_bytree = c(0.8, 1.0))  # 24 组合
      best <- list(auc = -Inf, params = NULL, nrounds = NA)
      for (i in seq_len(nrow(xgb_grid))) {
        pr <- list(objective = "binary:logistic", eval_metric = "auc",
                   eta = xgb_grid$eta[i], max_depth = xgb_grid$max_depth[i],
                   subsample = xgb_grid$subsample[i], colsample_bytree = xgb_grid$colsample_bytree[i],
                   min_child_weight = 1, nthread = 1)
        set.seed(SEED)   # 同一随机种子 -> 各组合用同一折, 公平可比
        cvx <- xgboost::xgb.cv(pr, dtr_x, nrounds = 500, nfold = 5,
                               early_stopping_rounds = 30, verbose = 0, maximize = TRUE)
        bi  <- if (!is.null(cvx$best_iteration) && cvx$best_iteration >= 1)
                 cvx$best_iteration else which.max(cvx$evaluation_log$test_auc_mean)
        a   <- cvx$evaluation_log$test_auc_mean[bi]
        if (is.finite(a) && a > best$auc) best <- list(auc = a, params = pr, nrounds = bi)
      }
      cat(sprintf("    XGB 最优: depth=%d eta=%.2f subsample=%.1f colsample=%.1f nrounds=%d (CV-AUC=%.3f)\n",
                  best$params$max_depth, best$params$eta, best$params$subsample,
                  best$params$colsample_bytree, best$nrounds, best$auc))
      .cv_auc_x <- best$auc
      m_x <- xgboost::xgb.train(best$params, dtr_x, nrounds = best$nrounds, verbose = 0)

    } else {  # 旧的快速路径: 固定常用超参, 仅 5 折 CV 选轮数
      pr <- list(objective = "binary:logistic", eval_metric = "auc", eta = 0.05,
                 max_depth = 5, subsample = 0.8, colsample_bytree = 0.8,
                 min_child_weight = 1, nthread = 1)
      set.seed(SEED)
      cvx <- xgboost::xgb.cv(pr, dtr_x, nrounds = 500, nfold = 5,
                             early_stopping_rounds = 30, verbose = 0, maximize = TRUE)
      bx  <- if (!is.null(cvx$best_iteration) && cvx$best_iteration >= 1) cvx$best_iteration else 200
      .cv_auc_x <- max(cvx$evaluation_log$test_auc_mean, na.rm = TRUE)
      m_x <- xgboost::xgb.train(pr, dtr_x, nrounds = bx, verbose = 0)
    }
    list(model = m_x, prob = as.numeric(predict(m_x, as.matrix(sp$Xte))),
         cv_auc = round(.cv_auc_x, 3))
  }, error = function(e) { cat("    [XGBoost] 失败:", conditionMessage(e), "\n"); NULL })
  if (!is.null(.xgb)) { ml_models[["XGBoost"]] <- .xgb$model; ml_probs[["XGBoost"]] <- .xgb$prob
                        ml_cv_auc[["XGBoost"]] <- .xgb$cv_auc }
}

# LightGBM (单列; TUNE_MODE=="grid" 时对核心超参做网格搜索: 每个组合 5 折 lgb.cv 选 CV-AUC + 最佳轮数)
if (requireNamespace("lightgbm", quietly = TRUE)) {
  cat(sprintf("  训练 LightGBM (%s) ...\n", if (TUNE_MODE == "grid") "5 折网格搜索" else "5 折 CV 选轮数"))
  .lgb <- tryCatch({
    dtr    <- lightgbm::lgb.Dataset(as.matrix(bal$X), label = bal$y)
    .cv_auc_l <- NA_real_
    base_p <- list(objective = "binary", metric = "auc", bagging_freq = 1,
                   min_data_in_leaf = 20, verbosity = -1, num_threads = 1)

    if (TUNE_MODE == "grid") {
      lgb_grid <- expand.grid(num_leaves = c(15, 31, 63), learning_rate = c(0.05, 0.1),
                              feature_fraction = c(0.8, 1.0), bagging_fraction = c(0.8, 1.0))  # 24 组合
      best <- list(auc = -Inf, params = NULL, nrounds = NA)
      for (i in seq_len(nrow(lgb_grid))) {
        pr <- c(base_p, list(num_leaves = lgb_grid$num_leaves[i],
                             learning_rate    = lgb_grid$learning_rate[i],
                             feature_fraction = lgb_grid$feature_fraction[i],
                             bagging_fraction = lgb_grid$bagging_fraction[i]))
        set.seed(SEED)
        cvr <- lightgbm::lgb.cv(pr, dtr, nrounds = 500, nfold = 5,
                                early_stopping_rounds = 30, verbose = -1)
        a  <- tryCatch(as.numeric(cvr$best_score), error = function(e) NA_real_)
        bi <- tryCatch(as.integer(cvr$best_iter),  error = function(e) NA_integer_)
        if (is.finite(a) && a > best$auc) best <- list(auc = a, params = pr, nrounds = bi)
      }
      cat(sprintf("    LGB 最优: leaves=%d lr=%.2f feat_frac=%.1f bag_frac=%.1f nrounds=%d (CV-AUC=%.3f)\n",
                  best$params$num_leaves, best$params$learning_rate, best$params$feature_fraction,
                  best$params$bagging_fraction, best$nrounds, best$auc))
      .cv_auc_l <- best$auc
      m <- lightgbm::lgb.train(best$params, dtr,
                               nrounds = if (is.finite(best$nrounds) && best$nrounds >= 1) best$nrounds else 200,
                               verbose = -1)

    } else {  # 旧的快速路径: 固定常用超参, 仅 5 折 CV 选轮数
      pr <- c(base_p, list(num_leaves = 31, learning_rate = 0.05,
                           feature_fraction = 0.9, bagging_fraction = 0.8))
      set.seed(SEED)
      cvr <- lightgbm::lgb.cv(pr, dtr, nrounds = 500, nfold = 5,
                              early_stopping_rounds = 30, verbose = -1)
      bi  <- if (!is.null(cvr$best_iter) && cvr$best_iter >= 1) cvr$best_iter else 200
      .cv_auc_l <- tryCatch(as.numeric(cvr$best_score), error = function(e) NA_real_)
      m   <- lightgbm::lgb.train(pr, dtr, nrounds = bi, verbose = -1)
    }
    list(model = m, prob = as.numeric(predict(m, as.matrix(sp$Xte))),
         cv_auc = round(.cv_auc_l, 3))
  }, error = function(e) { cat("    [LightGBM] 失败:", conditionMessage(e), "\n"); NULL })
  if (!is.null(.lgb)) { ml_models[["LightGBM"]] <- .lgb$model; ml_probs[["LightGBM"]] <- .lgb$prob
                        ml_cv_auc[["LightGBM"]] <- .lgb$cv_auc }
}

# --- ★ 概率重标定: 修复 SMOTE 致的系统性高估 (在原始未 SMOTE 训练集上拟合, 套到测试集; 不泄漏测试) ---
#   重标定为单调变换 -> 不改 AUROC/ROC; 仅修正校准(截距/斜率)、Brier 及 0.5 切点的 Sens/Spec/F1。
ml_probs_raw <- ml_probs                          # 备份原始(未校正)测试概率, 供对照
.predict_prob <- function(model, newX) {          # 统一取 P(Yes): caret 模型 vs LightGBM/XGBoost(booster)
  if (inherits(model, "lgb.Booster") || inherits(model, "xgb.Booster"))
    as.numeric(predict(model, as.matrix(newX)))
  else as.numeric(predict(model, newdata = as.data.frame(newX), type = "prob")[, "Yes"])
}
if (RECALIBRATE != "none") {
  .tau_smote <- mean(bal$y)                        # 训练时正类占比(SMOTE 后≈0.5; 未 SMOTE 时=原始患病率)
  for (nm in names(ml_models)) {
    .p_tr  <- .predict_prob(ml_models[[nm]], sp$Xtr)              # 原始训练集预测(真实患病率)
    .recal <- make_recalibrator(.p_tr, sp$ytr, RECALIBRATE, tau = .tau_smote)
    ml_probs[[nm]] <- .recal(ml_probs[[nm]])                     # 校正测试集概率
  }
  cat(sprintf("  [重标定] 测试集概率已做 %s 校正 (训练正类占比 τ=%.2f, 原始患病率 π=%.3f); 下列性能/校准/DCA 基于校正后概率\n",
              RECALIBRATE, .tau_smote, mean(sp$ytr)))
}

# --- 性能表 (对齐本文 Table 2, 已扩充指标列) ---
cv_auc_vec <- vapply(names(ml_probs), function(nm) {
  v <- ml_cv_auc[[nm]]
  if (is.null(v) || !is.finite(v)) NA_real_ else v
}, numeric(1))
perf_tab <- purrr::map_dfr(names(ml_probs), function(nm)
  cbind(Model = nm, CV_AUC_train = unname(cv_auc_vec[nm]),
        metrics_from_prob(ml_probs[[nm]], sp$yte)))
perf_tab <- perf_tab[order(-perf_tab$AUROC), ]
write.csv(perf_tab, file.path(DIR_CSV, "ml_model_performance.csv"), row.names = FALSE)
cat("  --- 各模型内部验证集性能 (按 AUROC 降序; CV_AUC_train = 训练集 5 折 CV) ---\n"); print(perf_tab, row.names = FALSE)

# ★审稿意见#5: 最优模型仅在开发队列内部(内部验证集 AUROC)选定并锁定;
#   MIMIC-IV 外部队列只对锁定模型评估一次, 不参与任何算法/变量/超参选择。
final_model_lock <- data.frame(
  final_model     = perf_tab$Model[1],
  selection_basis = "Internal validation (held-out 30% of eICU-CRD) AUROC; model development confined to eICU-CRD only",
  external_use    = "MIMIC-IV used once for external evaluation of the locked model only (no selection)")
write.csv(final_model_lock, file.path(DIR_CSV, "ml_final_model_lock.csv"), row.names = FALSE)
cat(sprintf("  >>> 最优模型(锁定) = %s (内部验证 AUROC=%.3f, 训练集 CV AUC=%.3f); 外部队列仅评估一次, 不参与选择\n",
            perf_tab$Model[1], perf_tab$AUROC[1], perf_tab$CV_AUC_train[1]))

# --- 模型 ROC 叠加图 (本文 Fig3) ---
.mcols <- grDevices::hcl.colors(length(ml_probs), "Dark 3")
pdf(file.path(DIR_DOCX, "fig_ml_model_roc.pdf"), width = 8, height = 7)
.first <- TRUE; .ord <- perf_tab$Model
for (k in seq_along(.ord)) {
  nm <- .ord[k]
  r  <- pROC::roc(sp$yte, ml_probs[[nm]], levels = c(0, 1), direction = "<", quiet = TRUE)
  if (.first) { plot(r, col = .mcols[k], lwd = 2, legacy.axes = TRUE,
                     main = "ROC of ML models (testing set)"); .first <- FALSE }
  else        plot(r, col = .mcols[k], lwd = 2, add = TRUE)
}
legend("bottomright",
       legend = sprintf("%s (AUC=%.3f)", perf_tab$Model, perf_tab$AUROC),
       col = .mcols, lwd = 2, cex = 0.75)
dev.off()
cat("  模型 ROC 叠加图已输出: fig_ml_model_roc.pdf\n")

# --- 校准曲线 (本文 Fig4) + 校准斜率/截距 ---
calib_df <- purrr::map_dfr(names(ml_probs), function(nm) {
  p <- ml_probs[[nm]]; g <- dplyr::ntile(p, 10)
  data.frame(model = nm, pred = as.numeric(tapply(p, g, mean)),
             obs = as.numeric(tapply(sp$yte, g, mean)))
})
calib_stats <- purrr::map_dfr(names(ml_probs), function(nm) {
  p <- pmin(pmax(ml_probs[[nm]], 1e-6), 1 - 1e-6)
  g <- suppressWarnings(glm(sp$yte ~ qlogis(p), family = binomial))
  data.frame(model = nm, cal_intercept = round(unname(coef(g)[1]), 3),
             cal_slope = round(unname(coef(g)[2]), 3))
})
write.csv(calib_stats, file.path(DIR_CSV, "ml_calibration_stats.csv"), row.names = FALSE)
.cal_p <- ggplot2::ggplot(calib_df, ggplot2::aes(pred, obs, color = model)) +
  ggplot2::geom_abline(linetype = 2, color = "grey50") +
  ggplot2::geom_line() + ggplot2::geom_point(size = 1) +
  ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  ggplot2::labs(x = "Predicted probability", y = "Observed proportion",
                title = sprintf("Calibration curves (testing set%s)",
                                if (RECALIBRATE != "none") paste0(", recalibrated: ", RECALIBRATE) else "")) +
  ggplot2::theme_bw()
ggsave(file.path(DIR_DOCX, "fig_ml_calibration.pdf"), .cal_p, width = 7.5, height = 7)
cat("  校准曲线已输出: fig_ml_calibration.pdf\n")

# --- 决策曲线分析 DCA (本文 Fig5; 手写净获益, 无需额外包) ---
dca_manual <- function(probs_list, y, thr = seq(0.01, 0.90, by = 0.01)) {
  n <- length(y); prev <- mean(y)
  out <- list(
    All  = data.frame(model = "Treat all",  threshold = thr, NB = prev - (1 - prev) * thr / (1 - thr)),
    None = data.frame(model = "Treat none", threshold = thr, NB = 0))
  for (nm in names(probs_list)) {
    p <- probs_list[[nm]]
    nb <- vapply(thr, function(t) {
      pos <- p >= t
      sum(pos & y == 1) / n - (sum(pos & y == 0) / n) * (t / (1 - t))
    }, numeric(1))
    out[[nm]] <- data.frame(model = nm, threshold = thr, NB = nb)
  }
  dplyr::bind_rows(out)
}
dca_df <- dca_manual(ml_probs, sp$yte)
write.csv(dca_df, file.path(DIR_CSV, "ml_dca_netbenefit.csv"), row.names = FALSE)
.dca_p <- ggplot2::ggplot(dca_df, ggplot2::aes(threshold, NB, color = model)) +
  ggplot2::geom_line() +
  ggplot2::coord_cartesian(ylim = c(-0.05, max(0.05, mean(sp$yte) * 1.1))) +
  ggplot2::labs(x = "High risk threshold", y = "Standardized net benefit",
                title = sprintf("Decision curve analysis (testing set%s)",
                                if (RECALIBRATE != "none") paste0(", recalibrated: ", RECALIBRATE) else "")) +
  ggplot2::theme_bw()
ggsave(file.path(DIR_DOCX, "fig_ml_dca.pdf"), .dca_p, width = 8, height = 6)
cat("  决策曲线已输出: fig_ml_dca.pdf\n")

# ====================================================================
# 13b SHAP 解释 (本文: beeswarm / dependence / 个体 force-waterfall)
#   默认解释 测试集 AUROC 最优模型 (本文选 RF; 可用 SHAP_MODEL_NAME 指定)
#   - 树模型(XGBoost/LightGBM): 用 TreeSHAP (快, 精确)
#   - 其余模型(RF/SVM/...): 用 kernelshap 模型无关法 (需 kernelshap 包)
# ====================================================================
cat("\n>>> Step 13b: SHAP 解释\n")
shap_name <- if (!is.na(SHAP_MODEL_NAME) && SHAP_MODEL_NAME %in% names(ml_probs))
  SHAP_MODEL_NAME else perf_tab$Model[1]
cat(sprintf("  解释模型 = %s\n", shap_name))

# 背景集/解释集 (来自归一后的训练/测试矩阵)
set.seed(SEED)
.bg_idx  <- sample(seq_len(nrow(sp$Xtr)), min(SHAP_BG_N, nrow(sp$Xtr)))
.exp_idx <- sample(seq_len(nrow(sp$Xte)), min(SHAP_EXP_N, nrow(sp$Xte)))
X_bg  <- as.data.frame(sp$Xtr[.bg_idx, , drop = FALSE])
X_exp <- as.data.frame(sp$Xte[.exp_idx, , drop = FALSE])

shp <- NULL
if (shap_name == "XGBoost" && !is.null(ml_models[["XGBoost"]])) {
  # 直接训练的 xgb.Booster -> TreeSHAP (shapviz 直接支持 xgb.Booster)
  shp <- tryCatch(shapviz::shapviz(ml_models[["XGBoost"]], X_pred = as.matrix(X_exp), X = X_exp),
                  error = function(e) { cat("    [TreeSHAP/XGB] 失败:", conditionMessage(e), "\n"); NULL })
} else if (shap_name == "LightGBM" && !is.null(ml_models[["LightGBM"]])) {
  shp <- tryCatch(shapviz::shapviz(ml_models[["LightGBM"]], X_pred = as.matrix(X_exp), X = X_exp),
                  error = function(e) { cat("    [TreeSHAP/LGB] 失败:", conditionMessage(e), "\n"); NULL })
}
# 其余模型 (或 TreeSHAP 失败) -> kernelshap 模型无关法
if (is.null(shp)) {
  if (!requireNamespace("kernelshap", quietly = TRUE)) {
    cat("  [SHAP] 非树模型需 kernelshap (install.packages('kernelshap')); 已跳过 SHAP。\n")
  } else {
    if (shap_name %in% c("LightGBM", "XGBoost")) {
      pf <- function(object, X) as.numeric(predict(object, as.matrix(X)))
      obj <- ml_models[[shap_name]]
    } else {
      pf <- function(object, X) as.numeric(predict(object, as.data.frame(X), type = "prob")[, "Yes"])
      obj <- ml_models[[shap_name]]
    }
    set.seed(SEED)
    ks  <- tryCatch(kernelshap::kernelshap(obj, X = X_exp, bg_X = X_bg, pred_fun = pf, verbose = FALSE),
                    error = function(e) { cat("    [kernelshap] 失败:", conditionMessage(e), "\n"); NULL })
    if (!is.null(ks)) shp <- shapviz::shapviz(ks)
  }
}

if (!is.null(shp)) {
  # (1) 全局重要性: bar + beeswarm (本文 Fig6)
  ggsave(file.path(DIR_DOCX, "fig_ml_shap_importance_bar.pdf"),
         shapviz::sv_importance(shp, kind = "bar")  + ggplot2::ggtitle(paste0("SHAP importance — ", shap_name)),
         width = 7, height = 6)
  ggsave(file.path(DIR_DOCX, "fig_ml_shap_beeswarm.pdf"),
         shapviz::sv_importance(shp, kind = "beeswarm") + ggplot2::ggtitle(paste0("SHAP beeswarm — ", shap_name)),
         width = 8, height = 6)

  # (2) 依赖图: 每个特征 (本文 Fig7), patchwork 拼图
  .feats   <- colnames(shapviz::get_shap_values(shp))
  .dep_lst <- lapply(.feats, function(v) shapviz::sv_dependence(shp, v = v))
  ggsave(file.path(DIR_DOCX, "fig_ml_shap_dependence.pdf"),
         patchwork::wrap_plots(.dep_lst, ncol = 3) +
           patchwork::plot_annotation(title = paste0("SHAP dependence — ", shap_name)),
         width = 14, height = ceiling(length(.feats) / 3) * 3.2, limitsize = FALSE)

  # (3) 个体解释: 测试集中 预测概率最高(死亡) 与 最低(生存) 各 1 例 (本文 Fig8 force/waterfall)
  .p_exp <- if (shap_name %in% c("LightGBM", "XGBoost"))
    as.numeric(predict(ml_models[[shap_name]], as.matrix(X_exp)))
  else as.numeric(predict(ml_models[[shap_name]], X_exp, type = "prob")[, "Yes"])
  .hi <- which.max(.p_exp); .lo <- which.min(.p_exp)
  ggsave(file.path(DIR_DOCX, "fig_ml_shap_force_highrisk.pdf"),
         shapviz::sv_waterfall(shp, row_id = .hi) +
           ggplot2::ggtitle(sprintf("High-risk case (predicted p=%.2f)", .p_exp[.hi])),
         width = 9, height = 5)
  ggsave(file.path(DIR_DOCX, "fig_ml_shap_force_lowrisk.pdf"),
         shapviz::sv_waterfall(shp, row_id = .lo) +
           ggplot2::ggtitle(sprintf("Low-risk case (predicted p=%.2f)", .p_exp[.lo])),
         width = 9, height = 5)
  cat("  SHAP 图已输出: fig_ml_shap_importance_bar / _beeswarm / _dependence / _force_highrisk / _force_lowrisk .pdf\n")
}

# ====================================================================
# 13c 落盘 (供报告/复核)
# ====================================================================
save(ml_cand, lasso_vars, boruta_vars, rfe_vars, sel_ml, sel_clin, sel_ir, sel_matrix,
     IR_IN_SELECTION, idx_tr,
     final_vars, ml_models, ml_probs, ml_probs_raw, ml_cv_auc, perf_tab, calib_stats, dca_df,
     final_model_lock, shap_name,
     file = file.path(DIR_LOAD, "ml_results.Rdata"))

# ====================================================================
# 13d ★导出 Python 建模数据 (traindata.csv / valdata.csv) —— 防泄漏版
#   供 python 端「Python机器学习建模与评估.py」直接读取 (手稿 ML 结果的实际来源)。
#   口径: df 已在 Step 0d 完成"先切分 + 两分区独立插补"; 特征 = Step 11 训练集内
#   三法筛选 + Step 12/13 确定的最终变量 final_vars; 训练/验证 = idx_tr 切分。
#     - 第 1 列 = 结局 Outcome (0/1); 其后为最终建模特征 (列名与 python 端一致)
#     - 因子 (如 stroke 的 No/Yes) 转 0/1 整数; 连续变量保持原值; 无缺失
#     - 文件编码 = GBK (python: pd.read_csv(..., encoding="GBK"))
#   ★ python 脚本中 5 处硬编码特征列表 (X_train / X_val / X_test 等)
#     必须改为下方打印的清单后再运行!
# ====================================================================
cat("\n>>> Step 13d: 导出 Python 建模数据 (traindata.csv / valdata.csv; 防泄漏版)\n")

ml_export <- df %>%
  mutate(across(all_of(final_vars) & where(is.factor), ~ as.integer(.) - 1L)) %>%
  mutate(Outcome = as.integer(round(as.numeric(.data[[ML_OUTCOME]])))) %>%
  select(Outcome, all_of(final_vars))

.na_cnt <- colSums(is.na(ml_export))
if (any(.na_cnt > 0))
  stop("  [导出中止] 建模数据存在缺失, 请检查: ",
       paste(names(.na_cnt)[.na_cnt > 0], collapse = ", "))

train_data <- ml_export[idx_tr, , drop = FALSE]
val_data   <- ml_export[-idx_tr, , drop = FALSE]
write.csv(train_data, file.path(DIR_CSV, "traindata.csv"), row.names = FALSE, fileEncoding = "GBK")
write.csv(val_data,   file.path(DIR_CSV, "valdata.csv"),   row.names = FALSE, fileEncoding = "GBK")
cat(sprintf("  已导出(GBK): %s (%d 行, 阳性 %d)\n              %s (%d 行, 阳性 %d)\n",
            file.path(DIR_CSV, "traindata.csv"), nrow(train_data), sum(train_data$Outcome),
            file.path(DIR_CSV, "valdata.csv"),   nrow(val_data),   sum(val_data$Outcome)))
cat(sprintf("  ★最终建模特征 (n=%d, 训练集内筛选): %s\n",
            length(final_vars), paste(final_vars, collapse = ", ")))
cat("  ★请确认 python 端硬编码变量列表与上述清单一致后再运行!\n")

cat("\n==================== ML 部分完成 (eICU) ====================\n")
cat(sprintf("  Step 11 候选池含 IR : %s\n", if (IR_IN_SELECTION) "是 (8 个 IR 与临床变量同筛)" else "否 (仅临床变量)"))
cat(sprintf("  Step 11 筛选变量 (n=%d): %s\n", length(sel_ml), paste(sel_ml, collapse = ", ")))
if (IR_IN_SELECTION)
  cat(sprintf("          其中 IR 幸存 (n=%d): %s\n", length(sel_ir),
              if (length(sel_ir)) paste(sel_ir, collapse = ", ") else "(无)"))
cat(sprintf("  Step 13 最终建模变量 (n=%d): %s\n", length(final_vars), paste(final_vars, collapse = ", ")))
cat(sprintf("  Step 13 最优 ML 模型            : %s (内部验证集 AUROC=%.3f; 模型开发全程仅用 eICU 训练分区)\n", perf_tab$Model[1], perf_tab$AUROC[1]))
cat(sprintf("  调参模式                        : %s\n", if (TUNE_MODE == "grid") "显式网格搜索 + 5 折 CV" else "tuneLength + 5 折 CV"))
cat("  输出 CSV : ml_feature_selection / ml_model_performance / ml_calibration_stats / ml_dca_netbenefit / ml_final_model_lock\n")
cat("  输出 图  : fig_ml_venn / fig_ml_model_roc / fig_ml_calibration / fig_ml_dca / fig_ml_shap_*\n")
cat("===========================================================\n")
