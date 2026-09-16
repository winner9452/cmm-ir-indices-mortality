######################################################################
# 11b_ml_selection_figures.R  (★审稿意见#5 修订版)
# ------------------------------------------------------------------
# 为三种特征筛选方法各自生成一张【独立】出版级图:
#
#   (A) LASSO   -> 系数路径图 fig_sel_lasso_path.pdf
#                  (随 log(lambda) 变化的系数轨迹; 标出 lambda.min / lambda.1se)
#   (B) Boruta  -> 重要性箱线图 fig_sel_boruta_importance.pdf
#                  (各变量 Z-score 分布, 按 Confirmed/Tentative/Rejected 着色;
#                   含 shadowMax/Min 阴影特征参照线)
#   (C) RFE     -> 性能-特征数曲线 fig_sel_rfe_auc.pdf
#                  (AUC ~ 纳入特征数; 标出 CV 最优 size)
#
# ★本次改动(审稿意见#5): 不再自行重跑特征筛选(旧版在全量数据上筛选, 存在泄漏,
#   且与主模块口径可能不一致); 改为直接读取 11_ml_section_eicu.R 在
#   【训练集】上完成筛选时保存的拟合对象 ./load/ml_selection_fits.rds 绘图,
#   保证图与正文筛选结果完全一致。
#
# 运行前提: 已运行 11_ml_section_eicu.R (其 Step 11 会保存 ml_selection_fits.rds)
######################################################################

# ====================================================================
# 0. 环境引导 + 依赖
# ====================================================================
if (!exists("DIR_DOCX")) DIR_DOCX <- "./docx"
if (!exists("DIR_LOAD")) DIR_LOAD <- "./load"
for (d in c(DIR_DOCX, DIR_LOAD)) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

.fit_file <- file.path(DIR_LOAD, "ml_selection_fits.rds")
if (!file.exists(.fit_file))
  stop("未找到 ", .fit_file, "\n  请先运行 11_ml_section_eicu.R (Step 11 会保存该文件)。")
.fits <- readRDS(.fit_file)
cvf        <- .fits$cvf          # cv.glmnet (10 折 CV, 训练集)
fit_path   <- .fits$fit_path     # glmnet 全路径 (训练集)
bor        <- .fits$bor_raw      # Boruta 原始对象 (含 ImpHistory, 训练集)
bor_fixed  <- .fits$bor_fixed    # TentativeRoughFix 后
rfe_fit    <- .fits$rfe_fit      # caret::rfe 结果 (随机森林 + 5 折 CV, 训练集)
col2var    <- .fits$col2var      # 哑变量列 -> 原变量 映射
SEED       <- .fits$seed
cat(sprintf(">>> 已载入训练集筛选拟合对象 (IR_IN_SELECTION = %s)\n", .fits$IR_IN_SELECTION))

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse)

# 出版级主题 (无网格杂线, 适合期刊)
.theme_pub <- ggplot2::theme_bw(base_size = 12) +
  ggplot2::theme(panel.grid.minor = ggplot2::element_blank(),
                 plot.title = ggplot2::element_text(face = "bold", size = 13),
                 legend.position = "right")

# ====================================================================
# (A) LASSO 系数路径图
# ====================================================================
cat("\n>>> (A) LASSO 系数路径图\n")
lam_1se <- cvf$lambda.1se
lam_min <- cvf$lambda.min

# 整理系数轨迹为长表 (每列变量随 log-lambda 的系数)
beta <- as.matrix(fit_path$beta)               # 行=哑变量列, 列=lambda 序列
loglam <- log(fit_path$lambda)
path_df <- as.data.frame(t(beta))
path_df$loglam <- loglam
path_long <- tidyr::pivot_longer(path_df, -loglam,
                                 names_to = "term", values_to = "coef")
# 在 lambda.min 处被选中的变量(用于高亮 + 标签; 与主模块 lambda.min 口径一致)
co_1se <- as.matrix(coef(cvf, s = "lambda.min"))
sel_terms <- setdiff(rownames(co_1se)[which(co_1se[, 1] != 0)], "(Intercept)")
lasso_vars <- unique(col2var[sel_terms])
path_long$selected <- path_long$term %in% sel_terms

# 末端标签 (仅给最终选中的变量, 放在路径左端系数最展开处, 便于读)
lab_df <- path_long %>%
  dplyr::filter(selected) %>%
  dplyr::group_by(term) %>%
  dplyr::filter(loglam == min(loglam)) %>%   # 最小 lambda 端系数最展开, 便于读
  dplyr::ungroup()

p_lasso <- ggplot2::ggplot(path_long,
              ggplot2::aes(loglam, coef, group = term)) +
  ggplot2::geom_line(data = dplyr::filter(path_long, !selected),
                     color = "grey80", linewidth = 0.4) +
  ggplot2::geom_line(data = dplyr::filter(path_long, selected),
                     ggplot2::aes(color = term), linewidth = 0.8) +
  ggplot2::geom_vline(xintercept = log(lam_min), linetype = 1, color = "red") +
  ggplot2::geom_vline(xintercept = log(lam_1se), linetype = 2, color = "grey40") +
  ggplot2::annotate("text", x = log(lam_min), y = Inf, label = "lambda.min",
                    vjust = 1.4, hjust = -0.05, size = 3.2, color = "red") +
  ggplot2::annotate("text", x = log(lam_1se), y = Inf, label = "lambda.1se",
                    vjust = 1.4, hjust = 1.05, size = 3.2, color = "grey30") +
  ggrepel::geom_text_repel(data = lab_df,
                    ggplot2::aes(label = term, color = term),
                    size = 3, max.overlaps = 50, direction = "y",
                    hjust = 1, nudge_x = -0.15, segment.size = 0.2, show.legend = FALSE) +
  ggplot2::labs(x = expression(log(lambda)), y = "Coefficient",
                title = "LASSO coefficient paths",
                color = "Selected (at lambda.min)") +
  .theme_pub +
  ggplot2::guides(color = ggplot2::guide_legend(ncol = 1))

# 若没装 ggrepel 则去掉标签层(避免报错)
if (!requireNamespace("ggrepel", quietly = TRUE)) {
  cat("  [提示] 未安装 ggrepel, 路径图省略变量标签 (install.packages('ggrepel') 可加上)\n")
  p_lasso <- ggplot2::ggplot(path_long, ggplot2::aes(loglam, coef, group = term)) +
    ggplot2::geom_line(data = dplyr::filter(path_long, !selected),
                       color = "grey80", linewidth = 0.4) +
    ggplot2::geom_line(data = dplyr::filter(path_long, selected),
                       ggplot2::aes(color = term), linewidth = 0.8) +
    ggplot2::geom_vline(xintercept = log(lam_min), linetype = 1, color = "red") +
    ggplot2::geom_vline(xintercept = log(lam_1se), linetype = 2, color = "grey40") +
    ggplot2::labs(x = expression(log(lambda)), y = "Coefficient",
                  title = "LASSO coefficient paths",
                  color = "Selected (at lambda.min)") +
    .theme_pub
}

ggplot2::ggsave(file.path(DIR_DOCX, "fig_sel_lasso_path.pdf"),
                p_lasso, width = 8, height = 6)
cat(sprintf("  LASSO 选中 %d 个变量: %s\n", length(lasso_vars), paste(lasso_vars, collapse = ", ")))
cat("  已输出: fig_sel_lasso_path.pdf\n")

# ====================================================================
# (B) Boruta 重要性箱线图
# ====================================================================
cat("\n>>> (B) Boruta 重要性箱线图\n")
boruta_vars <- Boruta::getSelectedAttributes(bor_fixed, withTentative = FALSE)

# 取每个属性(含 shadowMin/Mean/Max)的历次 Z-score, 转长表
imp_hist <- as.data.frame(bor$ImpHistory)
imp_hist[imp_hist == -Inf] <- NA                      # Boruta 用 -Inf 占位未评估轮
imp_long <- tidyr::pivot_longer(imp_hist, dplyr::everything(),
                                names_to = "feature", values_to = "Z") %>%
  dplyr::filter(is.finite(Z))

# 决策(来自 TentativeRoughFix 后); shadow* 单列为参照
dec <- bor_fixed$finalDecision                        # Confirmed/Rejected (Tentative 已裁决)
dec_map <- tibble::tibble(feature = names(dec), decision = as.character(dec))
imp_long <- imp_long %>%
  dplyr::left_join(dec_map, by = "feature") %>%
  dplyr::mutate(decision = dplyr::case_when(
    grepl("^shadow", feature) ~ "Shadow",
    is.na(decision)           ~ "Tentative",
    TRUE                      ~ decision))

# 按中位 Z 排序 (shadow 排最前作基准带)
ord <- imp_long %>% dplyr::group_by(feature) %>%
  dplyr::summarise(med = median(Z, na.rm = TRUE), .groups = "drop") %>%
  dplyr::arrange(med)
imp_long$feature <- factor(imp_long$feature, levels = ord$feature)

pal <- c(Confirmed = "#00BA38", Tentative = "#F0C808",
         Rejected = "#F8766D", Shadow = "#9AA0A6")

p_boruta <- ggplot2::ggplot(imp_long,
              ggplot2::aes(feature, Z, fill = decision)) +
  ggplot2::geom_boxplot(outlier.size = 0.5, linewidth = 0.3) +
  ggplot2::scale_fill_manual(values = pal, name = "Decision") +
  ggplot2::coord_flip() +
  ggplot2::labs(x = NULL, y = "Importance (Z-score)",
                title = "Boruta feature importance") +
  .theme_pub

ggplot2::ggsave(file.path(DIR_DOCX, "fig_sel_boruta_importance.pdf"),
                p_boruta, width = 8,
                height = max(6, 0.22 * length(unique(imp_long$feature))),
                limitsize = FALSE)
cat(sprintf("  Boruta 确认 %d 个变量: %s\n", length(boruta_vars), paste(boruta_vars, collapse = ", ")))
cat("  已输出: fig_sel_boruta_importance.pdf\n")

# ====================================================================
# (C) RFE 性能-特征数曲线
# ====================================================================
cat("\n>>> (C) RFE 性能-特征数曲线 (随机森林 + 5 折 CV + ROC; 训练集)\n")
rfe_vars <- caret::predictors(rfe_fit)

# 取每个 size 的 CV 性能 (ROC 优先, 否则 Accuracy)
res <- rfe_fit$results
metric_col <- if ("ROC" %in% names(res)) "ROC" else "Accuracy"
sd_col     <- if (paste0(metric_col, "SD") %in% names(res)) paste0(metric_col, "SD") else NA
best_size  <- rfe_fit$optsize
res$ymin <- if (!is.na(sd_col)) res[[metric_col]] - res[[sd_col]] else res[[metric_col]]
res$ymax <- if (!is.na(sd_col)) res[[metric_col]] + res[[sd_col]] else res[[metric_col]]

p_rfe <- ggplot2::ggplot(res, ggplot2::aes(Variables, .data[[metric_col]])) +
  { if (!is.na(sd_col))
      ggplot2::geom_ribbon(ggplot2::aes(ymin = ymin, ymax = ymax),
                           fill = "#2E9FDF", alpha = 0.15) } +
  ggplot2::geom_line(color = "#2E9FDF", linewidth = 0.8) +
  ggplot2::geom_point(color = "#2E9FDF", size = 1.6) +
  ggplot2::geom_vline(xintercept = best_size, linetype = 2, color = "red") +
  ggplot2::annotate("point", x = best_size,
                    y = res[[metric_col]][res$Variables == best_size],
                    color = "red", size = 3) +
  ggplot2::annotate("text", x = best_size,
                    y = res[[metric_col]][res$Variables == best_size],
                    label = sprintf("optimal = %d", best_size),
                    vjust = -1, color = "red", size = 3.3) +
  ggplot2::labs(x = "Number of features",
                y = if (metric_col == "ROC") "Cross-validated AUC" else "Cross-validated accuracy",
                title = "RFE performance vs. number of features") +
  .theme_pub + ggplot2::theme(legend.position = "none")

ggplot2::ggsave(file.path(DIR_DOCX, "fig_sel_rfe_auc.pdf"),
                p_rfe, width = 8, height = 6)
cat(sprintf("  RFE 最优 size = %d; 选中变量: %s\n", best_size, paste(rfe_vars, collapse = ", ")))
cat("  已输出: fig_sel_rfe_auc.pdf\n")

cat("\n==================== 三法筛选图完成 ====================\n")
cat(sprintf("  LASSO  (lambda.min) 选中 %d: %s\n", length(lasso_vars), paste(lasso_vars, collapse = ", ")))
cat(sprintf("  Boruta (confirmed)  选中 %d: %s\n", length(boruta_vars), paste(boruta_vars, collapse = ", ")))
cat(sprintf("  RFE    (opt size=%d) 选中 %d: %s\n", best_size, length(rfe_vars), paste(rfe_vars, collapse = ", ")))
cat("  输出图: fig_sel_lasso_path.pdf / fig_sel_boruta_importance.pdf / fig_sel_rfe_auc.pdf\n")
cat("=======================================================\n")

