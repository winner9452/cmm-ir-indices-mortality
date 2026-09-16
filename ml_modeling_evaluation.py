#################### 导入需要的python库 ####################
# pip install pandas
# pip install scikit-learn
# pip install statsmodels
# pip install xgboost
# pip install lightgbm
# pip install matplotlib
# pip install seaborn
# pip install shap
# pip install scipy
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import pickle
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample
import seaborn as sns
from scipy.stats import norm
from sklearn.calibration import calibration_curve
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score

####################### 一、数据导入及预处理 #######################
os.getcwd() # 当前工作路径
os.chdir(r"E:\cmm_study_eicu\python")
plt.rcParams['font.family'] = 'Times New Roman' # 设置字体为Times New Roman，与r统一
# ★★★ 关键修改：让导出的 PDF 文字保持「可编辑文本」，而非被转成轮廓路径 ★★★
#   matplotlib 默认 pdf.fonttype=3 (Type3)，文字会变成一堆矢量路径，
#   在 Adobe Illustrator 里无法选中、无法改字号。改成 42 (TrueType) 即可正常编辑。
#   下面这几行放在所有画图代码之前，对全脚本所有 plt.savefig 出的 PDF 统一生效。
plt.rcParams['pdf.fonttype'] = 42   # PDF 输出真文本(TrueType)，AI 可编辑/可改字号/可搜索 ✅
plt.rcParams['ps.fonttype']  = 42   # 若另存 EPS 同理
plt.rcParams['svg.fonttype'] = 'none'  # 若另存 SVG，文字保留为 <text> 而非路径
# 全局字号设置(投稿前可按目标期刊统一调整，比事后在 AI 里逐个改更省事)
plt.rcParams.update({
    'font.size':       12,   # 全局基准字号
    'axes.titlesize':  14,   # 标题
    'axes.labelsize':  12,   # 坐标轴标签
    'xtick.labelsize': 10,   # x 轴刻度
    'ytick.labelsize': 10,   # y 轴刻度
    'legend.fontsize': 10,   # 图例
})
# 导入数据
train_data = pd.read_csv("traindata.csv", encoding="GBK")   
val_data = pd.read_csv("valdata.csv", encoding="GBK")
##假如有外部测试集，标准化代码如下：
# test_data = pd.read_csv("testdata.csv", encoding="GBK")

####2025-07-26更新：smote过采样法，针对训练集使用，用于平衡结局分布#########################
# pip install imbalanced-learn  装python库⭐⭐⭐
from imblearn.over_sampling import SMOTE
# 第一列是结局，其余为特征
X_origin = train_data.iloc[:, 1:]  # 特征：从第2列开始
y_origin = train_data.iloc[:, 0]   # 结局：第1列
# 自定义smote过采样比例：让少数类变成多数类的 0.8 倍
smote = SMOTE(sampling_strategy=0.8, random_state=123)
# 应用SMOTE过采样
X_resampled, y_resampled = smote.fit_resample(X_origin, y_origin)
# 合并为新的训练集
train_data_resampled = pd.concat([pd.Series(y_resampled, name=y_origin.name),
                                  pd.DataFrame(X_resampled, columns=X_origin.columns)], axis=1)
# 导出过采样后的训练集
train_data_resampled.to_csv("traindata_smote.csv", index=False)
#检查一下原始类别分布和SMOTE过采样后的类别分布
print("原始类别分布:\n", y_origin.value_counts())
print("SMOTE过采样后类别分布:\n", y_resampled.value_counts())
##########################################2025-07-26更新内容结束，不做smote的可以不用管这里面的内容######

##############################################################################
####################### 二、在训练集构建机器学习预测模型 #######################
##############################################################################
#做了这么多准备，终于到了激动人心的模型构建部分🤦冲冲冲！！！

############################### 1.Logistic回归模型 ###############################
# 因为这个是线性模型，和其他机器学法不一样，很简单不用调参，所以分开做
train_data = pd.read_csv("traindata.csv", encoding="GBK")  # logistic回归一般不用过采样数据⭐⭐⭐
print(train_data.info()) 
categorical_vars = [c for c in ['Outcome', 'stroke'] if c in train_data.columns]  # ★本数据集仅 stroke 为分类变量(结局 Outcome 同按分类处理); 连续变量保持数值, 不再误转 category
for var in categorical_vars:
    train_data[var] = train_data[var].astype('category') 
# 查看训练集数据变量类型是否定义成功
print(train_data.info()) 
# 定义模型关键变量 
# ★★★将我们在R那边筛选出来的重要变量放进来★★★
X_train = train_data[[c for c in train_data.columns if c != 'Outcome']] 
X_train_const = sm.add_constant(X_train) # 给训练集添加常数项，就是初中学的二元一次方程y=kx+b中的b 👍
Y_train = train_data['Outcome'] # 定义结局变量
# 开始训练模型 
logist_model = sm.Logit(Y_train, X_train_const).fit(disp=0)
logist_model.summary() 
# 查看这个多因素逻辑回归的具体参数，投稿中文杂志需要给出这个内容，直接用r语言变量筛选部分结果里的“2.多因素逻辑回归中华级表格”，是一样的
# 计算训练集AUC，越高越好，0.5就意味是随机猜测的，1就意味着百分百对，所以70%以上最好，不过训练集一般都会很高接近1
Y_train_logist = logist_model.predict(X_train_const) 
auc_logist = roc_auc_score(Y_train, Y_train_logist) 
auc_logist # 输出训练集AUC
# 保存训练好的Logistic模型等最后一起出图，★记得先创建一个文件夹
with open("2.训练集构建模型/logistic_model.pkl", 'wb') as f:
    pickle.dump(logist_model, f)

#☆☆☆二分类诊断模型机器学习预测模型全流程代码
#☆☆☆小红书：派大珍
#☆☆☆bilibili：派大珍
#☆☆☆微信：sci02024
#☆☆☆预测模型的细节有很多，只有在以上方式购买代码享有答更详细教学视频与答疑
#☆☆☆感谢支持👍👍👍我会继续增加更多的见到过的预测模型图表

############################### 2. 决策树模型 ###############################
# 定义训练集和验证集里的自变量和结局
train_data_ml = pd.read_csv("traindata.csv", encoding="GBK")    # 想用过采样的训练集的话需要改名字⭐⭐⭐
val_data_ml = pd.read_csv("valdata.csv", encoding="GBK")
X_train = train_data_ml[[c for c in train_data_ml.columns if c != 'Outcome']] # 批量定义自变量
y_train = train_data_ml['Outcome'] 
X_val = val_data_ml[[c for c in val_data_ml.columns if c != 'Outcome']] # 批量定义自变量
y_val = val_data_ml['Outcome'] 

# 设置环境变量，避免路径问题
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

# ★先用默认参数训练决策树，这个大论文往往需要放，小论文直接往下走，关注超参数调优结果即可
tree_default = DecisionTreeClassifier(random_state=123)  # 创建决策树分类模型，使用默认参数
tree_default.fit(X_train, y_train)  # 在训练集上拟合决策树模型
# 查看模型的默认参数，这个参数可以写进大论文里，水字数增加工作量😀
print("决策树模型默认参数:", pd.DataFrame.from_dict(tree_default.get_params(), orient='index'))
# 计算使用默认参数时的模型在验证集的AUC，★注意模型好不好我们关注的是验证集的AUC高不高，以及外部数据集的AUC
y_val_pred_prob_treed = tree_default.predict_proba(X_val)[:, 1]
auc_treed = roc_auc_score(y_val, y_val_pred_prob_treed)
print("默认参数决策树模型的验证集 AUC:", auc_treed)

## ★★★★★模型超参数的网格搜索和5折交叉验证这个调参方法要写进论文的方法部分！
# 1. 定义超参数搜索范围
param_grid = {
    'max_depth': [3, 5, 10, None],  # 树的最大深度
    'min_samples_split': [5, 10, 20],  # 节点至少需要多少个样本，才会继续分裂
    'max_features': ['sqrt', None],  # 在每次分裂时，决策树可以考虑的最大特征数
    'ccp_alpha': [0.0, 0.01, 0.1]  # 剪枝时的复杂度惩罚系数
}

# 2. 使用 GridSearchCV 进行网格搜索和 k 折交叉验证
grid_search_tree = GridSearchCV(
    estimator=tree_default,
    param_grid=param_grid,
    scoring='roc_auc',  # 使用 AUC 作为评价指标
    cv=5,  # 5 折交叉验证
    n_jobs=1,  # 使用单线程
    verbose=1
)

grid_search_tree.fit(X_train, y_train)  # 在训练集上进行网格搜索

# 3. 输出最优超参数   👍这个结果要写进文章里
best_auc_tree = grid_search_tree.best_score_  # 获取最佳模型的交叉验证 AUC
tree_model_best = grid_search_tree.best_estimator_  # 获取最佳模型
best_params = grid_search_tree.best_params_  # 获取最佳超参数组合
print("最佳决策树模型参数组合:", best_params)
print("调优后决策树模型详细参数:", pd.DataFrame.from_dict(tree_model_best.get_params(), orient='index'))
print("参数调优决策树模型最佳模型的交叉验证 AUC:", best_auc_tree)
print("默认参数决策树模型的验证集 AUC:", auc_treed)

# 5. 保存训练好的决策树模型等最后一起出图
with open("2.训练集构建模型/tree_model.pkl", 'wb') as f:
    pickle.dump(tree_model_best, f)

############################### 3. 随机森林模型 ###############################
# 首先用默认参数训练随机森林模型
rf_model_default = RandomForestClassifier(random_state=123, oob_score=True)
rf_model_default.fit(X_train, y_train)
# 查看模型默认参数
print("模型默认参数:", pd.DataFrame.from_dict(rf_model_default.get_params(), orient='index'))
# 计算默认参数模型的验证集AUC
y_val_pred_prob_rfd = rf_model_default.predict_proba(X_val)[:, 1]
auc_rfd = roc_auc_score(y_val, y_val_pred_prob_rfd)
print("默认参数模型的验证集 AUC:", auc_rfd)

# 进行超参数网格搜索调优
# 1. 定义超参数搜索范围
param_grid = {
    'n_estimators': np.arange(50, 500, 50),  # 树的数量：50 到 450，每次增加 50
    'max_features': list(range(2, round(np.sqrt(X_train.shape[1])) + 1))  # 每棵树的最大特征使用数：2 到 自变量数
}

# 2. 使用 GridSearchCV 进行网格搜索和 k 折交叉验证
grid_search_rf = GridSearchCV(
    estimator=rf_model_default,  # 使用之前定义的默认参数模型
    param_grid=param_grid,  # 使用之前定义的超参数网格
    scoring='roc_auc',  # 使用 AUC 作为评价指标
    cv=5,  # 5 折交叉验证
    n_jobs=-1,  # 并行计算
    verbose=1
)

grid_search_rf.fit(X_train, y_train)  # 在训练集上进行网格搜索

# 3. 输出最优超参数
best_auc_rf = grid_search_rf.best_score_  # 获取最佳模型的交叉验证 AUC
rf_model_best = grid_search_rf.best_estimator_  # 获取最佳模型
best_params_rf = grid_search_rf.best_params_  # 获取最佳超参数组合
best_ntree = best_params_rf['n_estimators']
best_mtry = best_params_rf['max_features']
print("最佳RF模型参数组合: n_estimators =", best_ntree, ", max_features =", best_mtry)
print("调优后RF模型详细参数:", pd.DataFrame.from_dict(rf_model_best.get_params(), orient='index'))
print("默认参数RF模型的验证集 AUC:", auc_rfd)
print("调优后RF模型最佳模型的交叉验证 AUC", best_auc_rf)
# 保存训练好的模型
with open("2.训练集构建模型/rf_model.pkl", 'wb') as f:
    pickle.dump(rf_model_best, f)

###################### 4. Xgboost模型 ##########################
# 使用默认参数训练XGBoost
xgb_default = XGBClassifier(random_state=123, eval_metric='logloss')
xgb_default.fit(X_train, y_train)
# 计算默认参数模型的验证集AUC
y_val_pred_prob_xgbd = xgb_default.predict_proba(X_val)[:, 1]
auc_xgbd = roc_auc_score(y_val, y_val_pred_prob_xgbd)
print("默认参数XGBoost模型的验证集 AUC:", auc_xgbd)

# 定义超参数搜索范围
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'max_depth': [3, 5, 10],  # 最大深度
    'n_estimators': [50, 100, 200],  # 弱分类器数量
    'subsample': [0.6, 0.8, 1.0]  # 采样比例
}

# 使用 GridSearchCV 进行网格搜索和 k 折交叉验证
grid_search_xgb = GridSearchCV(
    estimator=xgb_default,  # 使用之前定义的默认参数模型
    param_grid=param_grid,  # 使用之前定义的超参数网格
    scoring='roc_auc',  # 使用 AUC 作为评价指标
    cv=5,  # 5 折交叉验证
    n_jobs=-1,  # 并行计算
    verbose=1
)

grid_search_xgb.fit(X_train, y_train)  # 在训练集上进行网格搜索

# 输出最优超参数组合
best_auc_xgb = grid_search_xgb.best_score_  # 获取最佳模型的交叉验证 AUC
xgb_model_best = grid_search_xgb.best_estimator_  # 获取最佳模型
best_params_xgb = grid_search_xgb.best_params_  # 获取最佳超参数组合
print("最佳XGBoost参数组合:", best_params_xgb)
print("调优后XGBoost模型详细参数:", pd.DataFrame.from_dict(xgb_model_best.get_params(), orient='index'))
print("默认参数XGBoost模型的验证集 AUC:", auc_xgbd)
print("参数调优XGBoost模型最佳模型的交叉验证 AUC:", best_auc_xgb)
# 保存训练好的模型
with open("2.训练集构建模型/xgb_model.pkl", 'wb') as f:
    pickle.dump(xgb_model_best, f)

###################### 5. LightGBM模型 ##########################
# 使用默认参数训练LightGBM
lgb_default = lgb.LGBMClassifier(random_state=123)
lgb_default.fit(X_train, y_train)
# 计算默认参数模型的验证集AUC
y_val_pred_prob_lgbd = lgb_default.predict_proba(X_val)[:, 1]
auc_lgbd = roc_auc_score(y_val, y_val_pred_prob_lgbd)
print("默认参数LightGBM模型的验证集 AUC:", auc_lgbd)

# 定义超参数搜索范围
param_grid = {
    'learning_rate': [0.1, 0.2],  # 学习率
    'num_leaves': [31, 50, 100],  # 叶子节点数
    'n_estimators': [100],  # 弱分类器数量
    'subsample': [0.6, 0.8, 1.0],  # 采样比例
    'colsample_bytree': [0.6, 0.8, 1.0]  # 特征子集采样比例
}

# 使用 GridSearchCV 进行网格搜索和 k 折交叉验证
grid_search_lgb = GridSearchCV(
    estimator=lgb_default,  # 使用之前定义的默认参数模型
    param_grid=param_grid,  # 使用之前定义的超参数网格
    scoring='roc_auc',  # 使用 AUC 作为评价指标
    cv=5,  # 5 折交叉验证
    n_jobs=-1,  # 并行计算
    verbose=1
)

grid_search_lgb.fit(X_train, y_train)  # 在训练集上进行网格搜索

# 输出最优超参数组合
best_auc_lgb = grid_search_lgb.best_score_  # 获取最佳模型的交叉验证 AUC
lgb_model_best = grid_search_lgb.best_estimator_  # 获取最佳模型
best_params_lgb = grid_search_lgb.best_params_  # 获取最佳超参数组合
print("最佳LightGBM参数组合:", best_params_lgb)
print("调优LightGBM模型详细参数:", pd.DataFrame.from_dict(lgb_model_best.get_params(), orient='index'))
print("默认参数LightGBM模型的验证集 AUC:", auc_lgbd)
print("参数调优LightGBM模型最佳模型的交叉验证 AUC:", best_auc_lgb)
# 保存训练好的模型
with open("2.训练集构建模型/lgb_model.pkl", 'wb') as f:
    pickle.dump(lgb_model_best, f)

###################### 6. 支持向量机（SVM）模型 ##########################
# 使用默认参数训练SVM
svm_default = SVC(probability=True, random_state=123)
svm_default.fit(X_train, y_train)
# 计算默认参数模型的验证集AUC
y_val_pred_prob_svmd = svm_default.predict_proba(X_val)[:, 1]
auc_svmd = roc_auc_score(y_val, y_val_pred_prob_svmd)
print("默认参数SVM模型的验证集 AUC:", auc_svmd)

# 定义超参数搜索范围
param_grid = {
    'C': [0.1],  # 正则化参数
    'kernel': ['linear', 'rbf'],  # 核函数类型
    'gamma': ['scale', 'auto', 0.01],  # 核函数系数
    'degree': [2, 3, 4]  # 多项式核的阶数
}

# 使用 GridSearchCV 进行网格搜索和 k 折交叉验证
grid_search_svm = GridSearchCV(
    estimator=svm_default,  # 使用之前定义的默认参数模型
    param_grid=param_grid,  # 使用之前定义的超参数网格
    scoring='roc_auc',  # 使用 AUC 作为评价指标
    cv=5,  # 5 折交叉验证
    n_jobs=-1,  # 并行计算
    verbose=1
)

grid_search_svm.fit(X_train, y_train)  # 在训练集上进行网格搜索

# 输出最优超参数组合
best_auc_svm = grid_search_svm.best_score_  # 获取最佳模型的交叉验证 AUC
svm_model_best = grid_search_svm.best_estimator_  # 获取最佳模型
best_params_svm = grid_search_svm.best_params_  # 获取最佳超参数组合
print("最佳SVM参数组合:", best_params_svm)
print("调优SVM模型详细参数:", pd.DataFrame.from_dict(svm_model_best.get_params(), orient='index'))
print("默认参数SVM模型的验证集 AUC:", auc_svmd)
print("参数调优SVM模型最佳模型的交叉验证 AUC:", best_auc_svm)
# 保存训练好的模型
with open("2.训练集构建模型/svm_model.pkl", 'wb') as f:
    pickle.dump(svm_model_best, f)

###################### 7. 人工神经网络（ANN）模型 ##########################
# 使用默认参数训练ANN
ann_default = MLPClassifier(random_state=123, max_iter=500)
ann_default.fit(X_train, y_train)
# 计算默认参数模型的验证集AUC
y_val_pred_prob_annd = ann_default.predict_proba(X_val)[:, 1]
auc_annd = roc_auc_score(y_val, y_val_pred_prob_annd)
print("默认参数ANN模型的验证集 AUC:", auc_annd)

# 定义超参数搜索范围
param_grid = {
    'hidden_layer_sizes': [(25,), (50,), (100,), (10, 10), (50, 50), (100, 100), (50, 50, 50)],  # 隐藏层神经元数
    'activation': ['relu', 'tanh', 'logistic']  # 激活函数
}

# 使用 GridSearchCV 进行网格搜索和 k 折交叉验证
grid_search_ann = GridSearchCV(
    estimator=ann_default,  # 使用之前定义的默认参数模型
    param_grid=param_grid,  # 使用之前定义的超参数网格
    scoring='roc_auc',  # 使用 AUC 作为评价指标
    cv=5,  # 5 折交叉验证
    n_jobs=-1,  # 并行计算
    verbose=1
)

grid_search_ann.fit(X_train, y_train)  # 在训练集上进行网格搜索

# 输出最优超参数组合
best_auc_ann = grid_search_ann.best_score_  # 获取最佳模型的交叉验证 AUC
ann_model_best = grid_search_ann.best_estimator_  # 获取最佳模型
best_params_ann = grid_search_ann.best_params_  # 获取最佳超参数组合
print("最佳ANN参数组合:", best_params_ann)
print("调优ANN模型详细参数:", pd.DataFrame.from_dict(ann_model_best.get_params(), orient='index'))
print("默认参数ANN模型的验证集 AUC:", auc_annd)
print("参数调优ANN模型最佳模型的交叉验证 AUC:", best_auc_ann)
# 保存训练好的模型
with open("2.训练集构建模型/ann_model.pkl", 'wb') as f:
    pickle.dump(ann_model_best, f)

################################################################################################
################################## 验证集预测模型评估 #################################################
################################################################################################
# 先读取验证集
# 1.得到验证集数据（对于Logistic模型）
val_data = pd.read_csv("valdata.csv", encoding="GBK")
X_val_logist = val_data[[c for c in val_data.columns if c != 'Outcome']]
X_val_logist_const = sm.add_constant(X_val_logist)
y_val_logist = val_data['Outcome']
# 定义变量类型
categorical_vars = [c for c in ['Outcome', 'stroke'] if c in val_data.columns]  # ★本数据集仅 stroke 为分类变量(结局 Outcome 同按分类处理); 连续变量保持数值, 不再误转 category
for var in categorical_vars:
    val_data[var] = val_data[var].astype('category') 
print(val_data.info()) 
# 2.得到验证数据集（对于机器学习模型）
val_data_ml = pd.read_csv("valdata.csv", encoding="GBK")
X_val = val_data_ml[[c for c in val_data_ml.columns if c != 'Outcome']] # 批量定义自变量
y_val = val_data_ml['Outcome'] 

###################### 1. 加载训练好的模型 ##########################
# 1.1. Logistic模型
with open("2.训练集构建模型/logistic_model.pkl", 'rb') as f:
    logist_model = pickle.load(f)
# 1.2. 决策树模型
with open("2.训练集构建模型/tree_model.pkl", 'rb') as f:
    tree_model = pickle.load(f)
# 1.3. 随机森林模型
with open("2.训练集构建模型/rf_model.pkl", 'rb') as f:
    rf_model = pickle.load(f)
# 1.4. XGBoost模型
with open("2.训练集构建模型/xgb_model.pkl", 'rb') as f:
    xgb_model = pickle.load(f)
# 1.5. LightGBM模型
with open("2.训练集构建模型/lgb_model.pkl", 'rb') as f:
    lgb_model = pickle.load(f)
# 1.6. SVM模型
with open("2.训练集构建模型/svm_model.pkl", 'rb') as f:
    svm_model = pickle.load(f)
# 1.7. ANN模型
with open("2.训练集构建模型/ann_model.pkl", 'rb') as f:
    ann_model = pickle.load(f)

###################### 2. 得到验证数据集预测结果，包括预测概率和预测分类 ##########################
# 2.1. Logistic模型
y_val_pred_prob_logist = logist_model.predict(X_val_logist_const) # 预测概率
y_val_pred_logist = (y_val_pred_prob_logist >= 0.5).astype(int) # 预测分类值（阈值0.5）
# 1.2. 决策树模型
y_val_pred_prob_tree = tree_model.predict_proba(X_val)[:, 1]
y_val_pred_tree = (y_val_pred_prob_tree >= 0.5).astype(int)
# 2.3. 随机森林模型
y_val_pred_prob_rf = rf_model.predict_proba(X_val)[:, 1]
y_val_pred_rf = (y_val_pred_prob_rf >= 0.5).astype(int)
# 2.4. XGBoost模型
y_val_pred_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]
y_val_pred_xgb = (y_val_pred_prob_xgb >= 0.5).astype(int)
# 2.5. LightGBM模型
y_val_pred_prob_lgb = lgb_model.predict_proba(X_val)[:, 1]
y_val_pred_lgb = (y_val_pred_prob_lgb >= 0.5).astype(int)
# 2.6. SVM模型
y_val_pred_prob_svm = svm_model.predict_proba(X_val)[:, 1]
y_val_pred_svm = (y_val_pred_prob_svm >= 0.5).astype(int)
# 2.7. ANN模型
y_val_pred_prob_ann = ann_model.predict_proba(X_val)[:, 1]
y_val_pred_ann = (y_val_pred_prob_ann >= 0.5).astype(int)

###################### 3. 计算混淆矩阵并可视化 ##########################
## 编写混淆矩阵可视化函数，方便调用 ##注意修改标题Outcome★★★★★★★
output_folder = "3.验证集混淆矩阵"    #建好文件夹
def CM_plot(cm, model_name):
    plt.figure(figsize=(5, 4)) 
    # 计算百分比
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 创建注释，包含计数和百分比
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n{cm_percentage[i, j]:.2%}"
    sns.heatmap(cm_percentage, annot=annotations, fmt="", cmap="Blues", xticklabels=['No Outcome', 'Outcome'], yticklabels=['No Outcome', 'Outcome'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    # 保存为PDF格式，并指定保存到指定文件夹
    plt.savefig(os.path.join(output_folder, f"{model_name}_confusion_matrix.pdf"), bbox_inches='tight')
    plt.show()

# 3.1. Logistic模型
cm_logist = confusion_matrix(y_val_logist, y_val_pred_logist) # 混淆矩阵
CM_plot(cm_logist, "Logistic")

# 3.2. 决策树模型
cm_tree = confusion_matrix(y_val, y_val_pred_tree)
CM_plot(cm_tree, "DecisionTree")

# 3.3. 随机森林模型
cm_rf = confusion_matrix(y_val, y_val_pred_rf)
CM_plot(cm_rf, "RandomForest")

# 3.4. XGBoost模型
cm_xgb = confusion_matrix(y_val, y_val_pred_xgb)
CM_plot(cm_xgb, "XGBoost")

# 3.5. LightGBM模型
cm_lgb = confusion_matrix(y_val, y_val_pred_lgb)
CM_plot(cm_lgb, "LightGBM")

# 3.6. SVM模型
cm_svm = confusion_matrix(y_val, y_val_pred_svm)
CM_plot(cm_svm, "SVM")

# 3.7. ANN模型
cm_ann = confusion_matrix(y_val, y_val_pred_ann)
CM_plot(cm_ann, "ANN")

###################### 4. 计算准确率、精确率、灵敏度、f1分数、特异度、kappa、Youden's J、PPV、NPV ##########################
def calculate_acc_pre_sen_f1_spc(cm, y_true, y_pred):
    tn, fp, fn, tp = cm.ravel()
    accuracy  = (tp + tn) / (tp + fn + tn + fp)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    f1score   = 2 * (precision * sensitivity) / (precision + sensitivity)
    specificity = tn / (tn + fp)
    kappa = cohen_kappa_score(y_true, y_pred)
    youden = sensitivity + specificity - 1
    ppv = precision                    # 阳性预测值（同 Precision）
    npv = tn / (tn + fn)               # 阴性预测值

    return accuracy, precision, sensitivity, f1score, specificity, kappa, youden, ppv, npv

# 4.1. Logistic 模型
accuracy_logist, precision_logist, sensitivity_logist, f1_logist, specificity_logist, \
kappa_logist, youden_logist, ppv_logist, npv_logist = calculate_acc_pre_sen_f1_spc(cm_logist, y_val_logist, y_val_pred_logist)
print(f"Logistic Model → Accuracy: {accuracy_logist:.3f}, Precision: {precision_logist:.3f}, "
      f"Sensitivity: {sensitivity_logist:.3f}, F1 Score: {f1_logist:.3f}, Specificity: {specificity_logist:.3f}, "
      f"Kappa: {kappa_logist:.3f}, Youden's J: {youden_logist:.3f}, PPV: {ppv_logist:.3f}, NPV: {npv_logist:.3f}")

# 4.2. 决策树模型
accuracy_tree, precision_tree, sensitivity_tree, f1_tree, specificity_tree, \
kappa_tree, youden_tree, ppv_tree, npv_tree = calculate_acc_pre_sen_f1_spc(cm_tree, y_val, y_val_pred_tree)
print(f"Decision Tree Model → Accuracy: {accuracy_tree:.3f}, Precision: {precision_tree:.3f}, "
      f"Sensitivity: {sensitivity_tree:.3f}, F1 Score: {f1_tree:.3f}, Specificity: {specificity_tree:.3f}, "
      f"Kappa: {kappa_tree:.3f}, Youden's J: {youden_tree:.3f}, PPV: {ppv_tree:.3f}, NPV: {npv_tree:.3f}")

# 4.3. 随机森林模型
accuracy_rf, precision_rf, sensitivity_rf, f1_rf, specificity_rf, \
kappa_rf, youden_rf, ppv_rf, npv_rf = calculate_acc_pre_sen_f1_spc(cm_rf, y_val, y_val_pred_rf)
print(f"Random Forest Model → Accuracy: {accuracy_rf:.3f}, Precision: {precision_rf:.3f}, "
      f"Sensitivity: {sensitivity_rf:.3f}, F1 Score: {f1_rf:.3f}, Specificity: {specificity_rf:.3f}, "
      f"Kappa: {kappa_rf:.3f}, Youden's J: {youden_rf:.3f}, PPV: {ppv_rf:.3f}, NPV: {npv_rf:.3f}")

# 4.4. XGBoost 模型
accuracy_xgb, precision_xgb, sensitivity_xgb, f1_xgb, specificity_xgb, \
kappa_xgb, youden_xgb, ppv_xgb, npv_xgb = calculate_acc_pre_sen_f1_spc(cm_xgb, y_val, y_val_pred_xgb)
print(f"XGBoost Model → Accuracy: {accuracy_xgb:.3f}, Precision: {precision_xgb:.3f}, "
      f"Sensitivity: {sensitivity_xgb:.3f}, F1 Score: {f1_xgb:.3f}, Specificity: {specificity_xgb:.3f}, "
      f"Kappa: {kappa_xgb:.3f}, Youden's J: {youden_xgb:.3f}, PPV: {ppv_xgb:.3f}, NPV: {npv_xgb:.3f}")

# 4.5. LightGBM 模型
accuracy_lgb, precision_lgb, sensitivity_lgb, f1_lgb, specificity_lgb, \
kappa_lgb, youden_lgb, ppv_lgb, npv_lgb = calculate_acc_pre_sen_f1_spc(cm_lgb, y_val, y_val_pred_lgb)
print(f"LightGBM Model → Accuracy: {accuracy_lgb:.3f}, Precision: {precision_lgb:.3f}, "
      f"Sensitivity: {sensitivity_lgb:.3f}, F1 Score: {f1_lgb:.3f}, Specificity: {specificity_lgb:.3f}, "
      f"Kappa: {kappa_lgb:.3f}, Youden's J: {youden_lgb:.3f}, PPV: {ppv_lgb:.3f}, NPV: {npv_lgb:.3f}")

# 4.6. SVM 模型
accuracy_svm, precision_svm, sensitivity_svm, f1_svm, specificity_svm, \
kappa_svm, youden_svm, ppv_svm, npv_svm = calculate_acc_pre_sen_f1_spc(cm_svm, y_val, y_val_pred_svm)
print(f"SVM Model → Accuracy: {accuracy_svm:.3f}, Precision: {precision_svm:.3f}, "
      f"Sensitivity: {sensitivity_svm:.3f}, F1 Score: {f1_svm:.3f}, Specificity: {specificity_svm:.3f}, "
      f"Kappa: {kappa_svm:.3f}, Youden's J: {youden_svm:.3f}, PPV: {ppv_svm:.3f}, NPV: {npv_svm:.3f}")

# 4.7. ANN 模型
accuracy_ann, precision_ann, sensitivity_ann, f1_ann, specificity_ann, \
kappa_ann, youden_ann, ppv_ann, npv_ann = calculate_acc_pre_sen_f1_spc(cm_ann, y_val, y_val_pred_ann)
print(f"ANN Model → Accuracy: {accuracy_ann:.3f}, Precision: {precision_ann:.3f}, "
      f"Sensitivity: {sensitivity_ann:.3f}, F1 Score: {f1_ann:.3f}, Specificity: {specificity_ann:.3f}, "
      f"Kappa: {kappa_ann:.3f}, Youden's J: {youden_ann:.3f}, PPV: {ppv_ann:.3f}, NPV: {npv_ann:.3f}")

###################### 5. 计算AUC及其95%置信区间 ##########################
## 编写AUC及其95%置信区间计算的函数 boostrap=1000次##
def calculate_auc(y_label, y_pred_prob, n_boot=1000, random_state=123):
    auc_value = roc_auc_score(y_label, y_pred_prob)
    data = np.column_stack([y_label, y_pred_prob])
    rng = np.random.default_rng(random_state)
    auc_boots = []
    for _ in range(n_boot):
        # 有放回重抽样
        idx = rng.choice(len(data), size=len(data), replace=True)
        y_label_boot = data[idx, 0]
        y_pred_boot = data[idx, 1]
        # 如果重抽样后只包含一类样本，则跳过
        if len(np.unique(y_label_boot)) < 2:
            continue
        auc_boot = roc_auc_score(y_label_boot, y_pred_boot)
        auc_boots.append(auc_boot)
    auc_ci_lower, auc_ci_upper = np.percentile(auc_boots, [2.5, 97.5])
    return auc_value, auc_ci_lower, auc_ci_upper

# 5.1. Logistic 模型
auc_value_logist, auc_ci_lower_logist, auc_ci_upper_logist = calculate_auc(y_val_logist, y_val_pred_prob_logist)
print(f"Logistic Model AUC: {auc_value_logist:.3f} (95% CI: {auc_ci_lower_logist:.3f} - {auc_ci_upper_logist:.3f})")

# 5.2. 决策树模型
auc_value_tree, auc_ci_lower_tree, auc_ci_upper_tree = calculate_auc(y_val, y_val_pred_prob_tree)
print(f"Decision Tree Model AUC: {auc_value_tree:.3f} (95% CI: {auc_ci_lower_tree:.3f} - {auc_ci_upper_tree:.3f})")

# 5.3. 随机森林模型
auc_value_rf, auc_ci_lower_rf, auc_ci_upper_rf = calculate_auc(y_val, y_val_pred_prob_rf)
print(f"Random Forest Model AUC: {auc_value_rf:.3f} (95% CI: {auc_ci_lower_rf:.3f} - {auc_ci_upper_rf:.3f})")

# 5.4. XGBoost 模型
auc_value_xgb, auc_ci_lower_xgb, auc_ci_upper_xgb = calculate_auc(y_val, y_val_pred_prob_xgb)
print(f"XGBoost Model AUC: {auc_value_xgb:.3f} (95% CI: {auc_ci_lower_xgb:.3f} - {auc_ci_upper_xgb:.3f})")

# 5.5. LightGBM 模型
auc_value_lgb, auc_ci_lower_lgb, auc_ci_upper_lgb = calculate_auc(y_val, y_val_pred_prob_lgb)
print(f"LightGBM Model AUC: {auc_value_lgb:.3f} (95% CI: {auc_ci_lower_lgb:.3f} - {auc_ci_upper_lgb:.3f})")

# 5.6. SVM 模型
auc_value_svm, auc_ci_lower_svm, auc_ci_upper_svm = calculate_auc(y_val, y_val_pred_prob_svm)
print(f"SVM Model AUC: {auc_value_svm:.3f} (95% CI: {auc_ci_lower_svm:.3f} - {auc_ci_upper_svm:.3f})")

# 5.7. ANN 模型
auc_value_ann, auc_ci_lower_ann, auc_ci_upper_ann = calculate_auc(y_val, y_val_pred_prob_ann)
print(f"ANN Model AUC: {auc_value_ann:.3f} (95% CI: {auc_ci_lower_ann:.3f} - {auc_ci_upper_ann:.3f})")

# 汇总模型评估指标并保存
model_results_validation = pd.DataFrame({
    "Model": ["Logistic", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "SVM", "ANN"],
    "AUC": [auc_value_logist, auc_value_tree, auc_value_rf, auc_value_xgb, auc_value_lgb, auc_value_svm, auc_value_ann],
    "95% CI Lower": [auc_ci_lower_logist, auc_ci_lower_tree, auc_ci_lower_rf, auc_ci_lower_xgb, auc_ci_lower_lgb, auc_ci_lower_svm, auc_ci_lower_ann],
    "95% CI Upper": [auc_ci_upper_logist, auc_ci_upper_tree, auc_ci_upper_rf, auc_ci_upper_xgb, auc_ci_upper_lgb, auc_ci_upper_svm, auc_ci_upper_ann],
    "Accuracy": [accuracy_logist, accuracy_tree, accuracy_rf, accuracy_xgb, accuracy_lgb, accuracy_svm, accuracy_ann],
    "Precision": [precision_logist, precision_tree, precision_rf, precision_xgb, precision_lgb, precision_svm, precision_ann],
    "Sensitivity": [sensitivity_logist, sensitivity_tree, sensitivity_rf, sensitivity_xgb, sensitivity_lgb, sensitivity_svm, sensitivity_ann],
    "Specificity": [specificity_logist, specificity_tree, specificity_rf, specificity_xgb, specificity_lgb, specificity_svm, specificity_ann],
    "F1 Score": [f1_logist, f1_tree, f1_rf, f1_xgb, f1_lgb, f1_svm, f1_ann],
    "Kappa": [kappa_logist, kappa_tree, kappa_rf, kappa_xgb, kappa_lgb, kappa_svm, kappa_ann],
    "Youden's J": [youden_logist, youden_tree, youden_rf, youden_xgb, youden_lgb, youden_svm, youden_ann],
    "PPV": [ppv_logist, ppv_tree, ppv_rf, ppv_xgb, ppv_lgb, ppv_svm, ppv_ann],
    "NPV": [npv_logist, npv_tree, npv_rf, npv_xgb, npv_lgb, npv_svm, npv_ann]
})

model_results_validation
model_results_validation.to_csv("4.验证集模型评估/model_performance_validation.csv", index=False)

###################### 6. 绘制ROC曲线 ##########################
## 编写绘制ROC曲线的函数 ##
output_folder = "4.验证集模型评估/单模型评估"  # 建好文件夹
def ROC_plot(y_label, y_pred_prob, auc_value, auc_ci_lower, auc_ci_upper, model_name):
    fpr, tpr, _ = roc_curve(y_label, y_pred_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="#c0392b", linewidth=2,
             label=f'AUC = {auc_value:.3f} (95% CI: {auc_ci_lower:.3f} - {auc_ci_upper:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Chance')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.gca().set_aspect('equal', adjustable='box')   # ROC 用正方形坐标更规范
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curve - {model_name}")            # ★ 标题带上模型名(原来缺失)
    plt.legend(loc='lower right', frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{model_name}_ROC_curve.pdf"), bbox_inches='tight')
    plt.show()
    plt.close()

# 6.1. Logistic模型
ROC_plot(y_val_logist, y_val_pred_prob_logist, auc_value_logist, auc_ci_lower_logist, auc_ci_upper_logist, "Logistic")
# 6.2. 决策树模型
ROC_plot(y_val, y_val_pred_prob_tree, auc_value_tree, auc_ci_lower_tree, auc_ci_upper_tree, "DecisionTree")
# 6.3. 随机森林模型
ROC_plot(y_val, y_val_pred_prob_rf, auc_value_rf, auc_ci_lower_rf, auc_ci_upper_rf, "RandomForest")
# 6.4. XGBoost模型
ROC_plot(y_val, y_val_pred_prob_xgb, auc_value_xgb, auc_ci_lower_xgb, auc_ci_upper_xgb, "XGBoost")
# 6.5. LightGBM模型
ROC_plot(y_val, y_val_pred_prob_lgb, auc_value_lgb, auc_ci_lower_lgb, auc_ci_upper_lgb, "LightGBM")
# 6.6. SVM模型
ROC_plot(y_val, y_val_pred_prob_svm, auc_value_svm, auc_ci_lower_svm, auc_ci_upper_svm, "SVM")
# 6.7. ANN模型
ROC_plot(y_val, y_val_pred_prob_ann, auc_value_ann, auc_ci_lower_ann, auc_ci_upper_ann, "ANN")

# 在一张图上绘制所有模型的ROC曲线，并显示AUC的95%置信区间
plt.figure(figsize=(8, 6))
models = {
    "Logistic": (y_val_logist, y_val_pred_prob_logist, auc_value_logist, auc_ci_lower_logist, auc_ci_upper_logist),
    "Decision Tree": (y_val, y_val_pred_prob_tree, auc_value_tree, auc_ci_lower_tree, auc_ci_upper_tree),
    "Random Forest": (y_val, y_val_pred_prob_rf, auc_value_rf, auc_ci_lower_rf, auc_ci_upper_rf),
    "XGBoost": (y_val, y_val_pred_prob_xgb, auc_value_xgb, auc_ci_lower_xgb, auc_ci_upper_xgb),
    "LightGBM": (y_val, y_val_pred_prob_lgb, auc_value_lgb, auc_ci_lower_lgb, auc_ci_upper_lgb),
    "SVM": (y_val, y_val_pred_prob_svm, auc_value_svm, auc_ci_lower_svm, auc_ci_upper_svm),
    "ANN": (y_val, y_val_pred_prob_ann, auc_value_ann, auc_ci_lower_ann, auc_ci_upper_ann)
}
for model_name, (y_true, y_pred_prob, auc_value, ci_lower, ci_upper) in models.items():
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f}, 95% CI: {ci_lower:.3f}-{ci_upper:.3f})")  # 添加置信区间信息
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("1-Specificity")
plt.ylabel("Sensitivity")
plt.title("ROC Curve - Validation Set Model Comparison")
plt.legend(loc='lower right')  # 调整图例字体位置
plt.grid()
plt.savefig("4.验证集模型评估/ROC_curves_allmodel_validation.pdf", dpi=300)
plt.show()

###################### 7. 绘制校准曲线 ##########################
# 确保文件夹存在
output_folder = "4.验证集模型评估/单模型评估" 
# 计算Brier分数的置信区间函数
def brier_score_confidence_interval(y_true, y_pred_prob, n_bootstraps=1000, alpha=0.05):
    brier_scores = []  # 存储每次重采样计算的Brier分数
    
    # 使用自助法（bootstrap）进行重采样
    for _ in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), n_samples=len(y_true), replace=True)  # 随机重采样数据索引
        y_true_sample = np.array(y_true)[indices]  # 获取重采样后的真实标签
        y_pred_prob_sample = np.array(y_pred_prob)[indices]  # 获取重采样后的预测概率
        
        # 计算当前重采样的Brier分数
        brier_score = np.mean((y_pred_prob_sample - y_true_sample) ** 2)  # Brier分数是预测概率与真实标签之间的均方误差
        brier_scores.append(brier_score)  # 将计算得到的Brier分数添加到列表中
    
    # 计算Brier分数的均值和置信区间
    brier_mean = np.mean(brier_scores)  # Brier分数的均值
    brier_lower = np.percentile(brier_scores, (alpha / 2) * 100)  # 计算置信区间的下限
    brier_upper = np.percentile(brier_scores, (1 - alpha / 2) * 100)  # 计算置信区间的上限
    
    return brier_mean, brier_lower, brier_upper  # 返回Brier分数均值和置信区间上下限

# 编写绘制校准曲线的函数
def CaliC_plot(y_label, y_pred_prob, model_name, n_bins=10):
    # ★ 改用等频(quantile)分箱: 每个分箱样本量相等, 避免高概率区样本稀少导致的锯齿/尖刺
    #   (uniform 等宽分箱在结局不平衡时, 右侧分箱常只有个位数阳性, 曲线极不稳定)
    prob_true, prob_pred = calibration_curve(
        y_label, y_pred_prob, n_bins=n_bins, strategy='quantile')

    # 计算Brier分数及其置信区间
    brier_mean, brier_lower, brier_upper = brier_score_confidence_interval(y_label, y_pred_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', color="#1f77b4", linewidth=2,
             label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1,
             label='Perfect calibration')
    # Brier 分数放进文本框, 不再塞进图例标签(原来图例太长被挤在角落)
    plt.text(0.04, 0.96,
             f'Brier score = {brier_mean:.3f} (95% CI: {brier_lower:.3f}-{brier_upper:.3f})',
             transform=plt.gca().transAxes, va='top', ha='left', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgray', alpha=0.85))
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend(loc='lower right', frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{model_name}_Calibration_Curve.pdf"), bbox_inches='tight')
    plt.show()
    plt.close()

# Logistic模型
CaliC_plot(y_val_logist, y_val_pred_prob_logist, "Logistic")
# 决策树模型
CaliC_plot(y_val, y_val_pred_prob_tree, "DecisionTree")
# 随机森林模型
CaliC_plot(y_val, y_val_pred_prob_rf, "RandomForest")
# XGBoost模型
CaliC_plot(y_val, y_val_pred_prob_xgb, "XGBoost")
# LightGBM模型
CaliC_plot(y_val, y_val_pred_prob_lgb, "LightGBM")
# SVM模型
CaliC_plot(y_val, y_val_pred_prob_svm, "SVM")
# ANN模型
CaliC_plot(y_val, y_val_pred_prob_ann, "ANN")

# 在一张图上绘制所有模型的校准曲线
plt.figure(figsize=(12, 10))
models = {
    "Logistic": (y_val_logist, y_val_pred_prob_logist),
    "Decision Tree": (y_val, y_val_pred_prob_tree),
    "Random Forest": (y_val, y_val_pred_prob_rf),
    "XGBoost": (y_val, y_val_pred_prob_xgb),
    "LightGBM": (y_val, y_val_pred_prob_lgb),
    "SVM": (y_val, y_val_pred_prob_svm),
    "ANN": (y_val, y_val_pred_prob_ann)
}
for model_name, (y_true, y_pred_prob) in models.items():
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    
    # 计算Brier分数及其置信区间
    brier_mean, brier_lower, brier_upper = brier_score_confidence_interval(y_true, y_pred_prob)
    
    plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name} (Brier Score: {brier_mean:.3f} [95% CI: {brier_lower:.3f} - {brier_upper:.3f}])')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curves - Validation Set Model Comparison")
plt.legend(loc='lower right')
plt.grid()
plt.savefig("4.验证集模型评估/Calibration_curves_allmodel_validation.pdf", dpi=300)
plt.show()
###################### 8. 绘制决策分析曲线 (DCA) ##########################
## 编写计算净收益的函数，方便调用 ##
output_folder = "4.验证集模型评估/单模型评估"
# 计算净收益的函数
def calculate_net_benefit(y_label, y_pred_prob, thresholds=np.linspace(0.01, 0.99, 100)):
    net_benefit_model = []  # 用于保存在不同阳性阈值下基于模型的净收益
    net_benefit_alltrt = []  # 用于保存在不同阳性阈值下假定所有人都接受治疗时的净收益
    net_benefits_notrt = [0] * len(thresholds)  # 假定所有人都不接受治疗时的净收益，始终为0
    total_obs = len(y_label)
    for thresh in thresholds:
        # 对于基于模型的净收益
        y_pred_label = y_pred_prob > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        net_benefit = (tp / total_obs) - (fp / total_obs) * (thresh / (1 - thresh))
        net_benefit_model.append(net_benefit)
        # 对于假定所有人都接受治疗时的净收益
        tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
        total_right = tp + tn
        net_benefit = (tp / total_right) - (tn / total_right) * (thresh / (1 - thresh))
        net_benefit_alltrt.append(net_benefit)
    return net_benefit_model, net_benefit_alltrt, net_benefits_notrt

# 编写绘制DCA曲线的函数
def DCA_plot(net_benefit_model, net_benefit_alltrt, net_benefits_notrt, model_name, thresholds=np.linspace(0.01, 0.99, 100)):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, net_benefit_model, label="Model Net Benefit", color='blue', linewidth=2)
    plt.plot(thresholds, net_benefit_alltrt, label="Treat All", color="red", linewidth=2)
    plt.plot(thresholds, net_benefits_notrt, linestyle='--', color='green', label="Treat None", linewidth=2)
    plt.xlim(0, 1)                                     # ★ 锁定 x 轴范围
    plt.ylim(-0.15, np.nanmax(np.array(net_benefit_model)) + 0.1)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title(f"Decision Curve Analysis - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{model_name}_DCA_curve.pdf"), bbox_inches='tight')
    plt.show()
    plt.close()

# 单个模型的DCA曲线
# Logistic模型
net_benefit_logist, net_benefit_alltrt_logist, net_benefits_notrt_logist = calculate_net_benefit(y_val_logist, y_val_pred_prob_logist)
DCA_plot(net_benefit_logist, net_benefit_alltrt_logist, net_benefits_notrt_logist, "Logistic")
# 决策树模型
net_benefit_tree, net_benefit_alltrt_tree, net_benefits_notrt_tree = calculate_net_benefit(y_val, y_val_pred_prob_tree)
DCA_plot(net_benefit_tree, net_benefit_alltrt_tree, net_benefits_notrt_tree, "DecisionTree")
# 随机森林模型
net_benefit_rf, net_benefit_alltrt_rf, net_benefits_notrt_rf = calculate_net_benefit(y_val, y_val_pred_prob_rf)
DCA_plot(net_benefit_rf, net_benefit_alltrt_rf, net_benefits_notrt_rf, "RandomForest")
# XGBoost模型
net_benefit_xgb, net_benefit_alltrt_xgb, net_benefits_notrt_xgb = calculate_net_benefit(y_val, y_val_pred_prob_xgb)
DCA_plot(net_benefit_xgb, net_benefit_alltrt_xgb, net_benefits_notrt_xgb, "XGBoost")
# LightGBM模型
net_benefit_lgb, net_benefit_alltrt_lgb, net_benefits_notrt_lgb = calculate_net_benefit(y_val, y_val_pred_prob_lgb)
DCA_plot(net_benefit_lgb, net_benefit_alltrt_lgb, net_benefits_notrt_lgb, "LightGBM")
# SVM模型
net_benefit_svm, net_benefit_alltrt_svm, net_benefits_notrt_svm = calculate_net_benefit(y_val, y_val_pred_prob_svm)
DCA_plot(net_benefit_svm, net_benefit_alltrt_svm, net_benefits_notrt_svm, "SVM")
# ANN模型
net_benefit_ann, net_benefit_alltrt_ann, net_benefits_notrt_ann = calculate_net_benefit(y_val, y_val_pred_prob_ann)
DCA_plot(net_benefit_ann, net_benefit_alltrt_ann, net_benefits_notrt_ann, "ANN")

# 在一张图上绘制所有模型的DCA曲线
plt.figure(figsize=(10, 8))
thresholds = np.linspace(0.01, 0.99, 100)

# 计算所有模型的DCA曲线
models = {
    "Logistic": (y_val_logist, y_val_pred_prob_logist),
    "Decision Tree": (y_val, y_val_pred_prob_tree),
    "Random Forest": (y_val, y_val_pred_prob_rf),
    "XGBoost": (y_val, y_val_pred_prob_xgb),
    "LightGBM": (y_val, y_val_pred_prob_lgb),
    "SVM": (y_val, y_val_pred_prob_svm),
    "ANN": (y_val, y_val_pred_prob_ann)
}

# 计算所有模型的净收益
all_net_benefits = {}
for model_name, (y_true, y_pred_prob) in models.items():
    net_benefit, _, _ = calculate_net_benefit(y_true, y_pred_prob)
    all_net_benefits[model_name] = net_benefit

# 绘制所有模型的DCA曲线
for model_name, net_benefit in all_net_benefits.items():
    plt.plot(thresholds, net_benefit, label=model_name)

# 绘制“Treat All”和“Treat None”曲线
plt.plot(thresholds, net_benefit_alltrt_logist, linestyle="--", color="red", label="Treat All")
plt.plot(thresholds, net_benefits_notrt_logist, linestyle="--", color="green", label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylim(-0.3, np.nanmax(np.array([max(net_benefit) for net_benefit in all_net_benefits.values()])) + 0.1)
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis - Validation Set Model Comparison")
plt.legend(loc="lower right", fontsize=12)
plt.grid()
plt.savefig("4.验证集模型评估/DCA_curves_allmodel_validation.pdf", dpi=300)
plt.show()


#☆☆☆二分类诊断模型机器学习预测模型全流程代码
#☆☆☆小红书：派大珍
#☆☆☆bilibili：派大珍
#☆☆☆微信：sci02024    添加时记得备注
#☆☆☆预测模型的细节有很多，只有在以上方式购买代码享有答更详细教学视频与答疑
#☆☆☆感谢支持👍👍👍我会继续增加更多的见到过的预测模型图表

################################################################################################
##################################   训练集模型评估  #############################################   
################################################################################################
# 得到训练集数据（Logistic模型）
train_data = pd.read_csv("traindata.csv", encoding="GBK")    # logistic回归不需要用过采样数据⭐⭐⭐
X_train_logist = train_data[[c for c in train_data.columns if c != 'Outcome']]
X_train_logist_const = sm.add_constant(X_train_logist)
y_train_logist = train_data['Outcome']
# 定义变量类型
categorical_vars = [c for c in ['Outcome', 'stroke'] if c in train_data.columns]  # ★本数据集仅 stroke 为分类变量(结局 Outcome 同按分类处理); 连续变量保持数值, 不再误转 category
for var in categorical_vars:
    train_data[var] = train_data[var].astype('category') 
print(train_data.info()) 
# 得到训练数据集（机器学习模型）
train_data_ml = pd.read_csv("traindata.csv", encoding="GBK")   # 选择过采样数据集需要改名字⭐⭐⭐
X_train = train_data_ml[[c for c in train_data_ml.columns if c != 'Outcome']]
y_train = train_data_ml['Outcome'] 

###################### 1. 计算测试数据集预测结果 ##########################
# 1.1. Logistic模型
y_train_pred_prob_logist = logist_model.predict(X_train_logist_const) # 预测概率
y_train_pred_logist = (y_train_pred_prob_logist >= 0.5).astype(int) # 预测分类值（阈值0.5）
# 1.2. 决策树模型
y_train_pred_prob_tree = tree_model.predict_proba(X_train)[:, 1]
y_train_pred_tree = (y_train_pred_prob_tree >= 0.5).astype(int)
# 1.3. 随机森林模型
y_train_pred_prob_rf = rf_model.predict_proba(X_train)[:, 1]
y_train_pred_rf = (y_train_pred_prob_rf >= 0.5).astype(int)
# 1.4. XGBoost模型
y_train_pred_prob_xgb = xgb_model.predict_proba(X_train)[:, 1]
y_train_pred_xgb = (y_train_pred_prob_xgb >= 0.5).astype(int)
# 1.5. LightGBM模型
y_train_pred_prob_lgb = lgb_model.predict_proba(X_train)[:, 1]
y_train_pred_lgb = (y_train_pred_prob_lgb >= 0.5).astype(int)
# 1.6. SVM模型
y_train_pred_prob_svm = svm_model.predict_proba(X_train)[:, 1]
y_train_pred_svm = (y_train_pred_prob_svm >= 0.5).astype(int)
# 1.7. ANN模型
y_train_pred_prob_ann = ann_model.predict_proba(X_train)[:, 1]
y_train_pred_ann = (y_train_pred_prob_ann >= 0.5).astype(int)

###################### 2. 计算混淆矩阵并可视化 ##########################
output_folder = "3.训练集混淆矩阵"    #建好文件夹
# 2.1. Logistic模型
cm_logist_train = confusion_matrix(y_train_logist, y_train_pred_logist)
CM_plot(cm_logist_train, "Logistic")
# 2.2. 决策树模型
cm_tree_train = confusion_matrix(y_train, y_train_pred_tree)
CM_plot(cm_tree_train, "DecisionTree")
# 2.3. 随机森林模型
cm_rf_train = confusion_matrix(y_train, y_train_pred_rf)
CM_plot(cm_rf_train, "RandomForest")
# 2.4. XGBoost模型
cm_xgb_train = confusion_matrix(y_train, y_train_pred_xgb)
CM_plot(cm_xgb_train, "XGBoost")
# 2.5. LightGBM模型
cm_lgb_train = confusion_matrix(y_train, y_train_pred_lgb)
CM_plot(cm_lgb_train, "LightGBM")
# 2.6. SVM模型
cm_svm_train = confusion_matrix(y_train, y_train_pred_svm)
CM_plot(cm_svm_train, "SVM")
# 2.7. ANN模型
cm_ann_train = confusion_matrix(y_train, y_train_pred_ann)
CM_plot(cm_ann_train, "ANN")

###################### 3. 计算准确率、精确率、灵敏度、f1分数、特异度、kappa、约登、等等 ##########################
# 4.1. Logistic 模型
accuracy_logist_train, precision_logist_train, sensitivity_logist_train, f1_logist_train, specificity_logist_train, \
kappa_logist_train, youden_logist_train, ppv_logist_train, npv_logist_train = \
    calculate_acc_pre_sen_f1_spc(cm_logist_train, y_train_logist, y_train_pred_logist)

# 4.2. 决策树模型
accuracy_tree_train, precision_tree_train, sensitivity_tree_train, f1_tree_train, specificity_tree_train, \
kappa_tree_train, youden_tree_train, ppv_tree_train, npv_tree_train = \
    calculate_acc_pre_sen_f1_spc(cm_tree_train, y_train, y_train_pred_tree)

# 4.3. 随机森林模型
accuracy_rf_train, precision_rf_train, sensitivity_rf_train, f1_rf_train, specificity_rf_train, \
kappa_rf_train, youden_rf_train, ppv_rf_train, npv_rf_train = \
    calculate_acc_pre_sen_f1_spc(cm_rf_train, y_train, y_train_pred_rf)

# 4.4. XGBoost 模型
accuracy_xgb_train, precision_xgb_train, sensitivity_xgb_train, f1_xgb_train, specificity_xgb_train, \
kappa_xgb_train, youden_xgb_train, ppv_xgb_train, npv_xgb_train = \
    calculate_acc_pre_sen_f1_spc(cm_xgb_train, y_train, y_train_pred_xgb)

# 4.5. LightGBM 模型
accuracy_lgb_train, precision_lgb_train, sensitivity_lgb_train, f1_lgb_train, specificity_lgb_train, \
kappa_lgb_train, youden_lgb_train, ppv_lgb_train, npv_lgb_train = \
    calculate_acc_pre_sen_f1_spc(cm_lgb_train, y_train, y_train_pred_lgb)

# 4.6. SVM 模型
accuracy_svm_train, precision_svm_train, sensitivity_svm_train, f1_svm_train, specificity_svm_train, \
kappa_svm_train, youden_svm_train, ppv_svm_train, npv_svm_train = \
    calculate_acc_pre_sen_f1_spc(cm_svm_train, y_train, y_train_pred_svm)

# 4.7. ANN 模型
accuracy_ann_train, precision_ann_train, sensitivity_ann_train, f1_ann_train, specificity_ann_train, \
kappa_ann_train, youden_ann_train, ppv_ann_train, npv_ann_train = \
    calculate_acc_pre_sen_f1_spc(cm_ann_train, y_train, y_train_pred_ann)

###################### 4. 计算AUC及其95%置信区间 ##########################
# 4.1. Logistic模型
auc_value_logist_train, auc_ci_lower_logist_train, auc_ci_upper_logist_train = calculate_auc(y_train_logist, y_train_pred_prob_logist)
# 4.2. 决策树模型
auc_value_tree_train, auc_ci_lower_tree_train, auc_ci_upper_tree_train = calculate_auc(y_train, y_train_pred_prob_tree)
# 4.3. 随机森林模型
auc_value_rf_train, auc_ci_lower_rf_train, auc_ci_upper_rf_train = calculate_auc(y_train, y_train_pred_prob_rf)
# 4.4. XGBoost模型
auc_value_xgb_train, auc_ci_lower_xgb_train, auc_ci_upper_xgb_train = calculate_auc(y_train, y_train_pred_prob_xgb)
# 4.5. LightGBM模型
auc_value_lgb_train, auc_ci_lower_lgb_train, auc_ci_upper_lgb_train = calculate_auc(y_train, y_train_pred_prob_lgb)
# 4.6. SVM模型
auc_value_svm_train, auc_ci_lower_svm_train, auc_ci_upper_svm_train = calculate_auc(y_train, y_train_pred_prob_svm)
# 4.7. ANN模型
auc_value_ann_train, auc_ci_lower_ann_train, auc_ci_upper_ann_train = calculate_auc(y_train, y_train_pred_prob_ann)

###################### 5. 所有模型的训练集预测效果汇总，方便对比 ##########################
model_results_train = pd.DataFrame({
    "Model": ["Logistic", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "SVM", "ANN"],
    "AUC": [auc_value_logist_train, auc_value_tree_train, auc_value_rf_train, auc_value_xgb_train, auc_value_lgb_train, auc_value_svm_train, auc_value_ann_train],
    "AUC 95% CI Lower": [auc_ci_lower_logist_train, auc_ci_lower_tree_train, auc_ci_lower_rf_train, auc_ci_lower_xgb_train, auc_ci_lower_lgb_train, auc_ci_lower_svm_train, auc_ci_lower_ann_train],
    "AUC 95% CI Upper": [auc_ci_upper_logist_train, auc_ci_upper_tree_train, auc_ci_upper_rf_train, auc_ci_upper_xgb_train, auc_ci_upper_lgb_train, auc_ci_upper_svm_train, auc_ci_upper_ann_train],
    "Accuracy":  [accuracy_logist_train,  accuracy_tree_train,  accuracy_rf_train, accuracy_xgb_train,  accuracy_lgb_train,  accuracy_svm_train,  accuracy_ann_train],
    "Precision": [precision_logist_train, precision_tree_train, precision_rf_train, precision_xgb_train, precision_lgb_train, precision_svm_train, precision_ann_train],
    "Sensitivity": [sensitivity_logist_train, sensitivity_tree_train, sensitivity_rf_train, sensitivity_xgb_train, sensitivity_lgb_train, sensitivity_svm_train, sensitivity_ann_train],
    "Specificity": [specificity_logist_train, specificity_tree_train, specificity_rf_train, specificity_xgb_train, specificity_lgb_train, specificity_svm_train, specificity_ann_train],
    "F1 Score": [f1_logist_train, f1_tree_train, f1_rf_train, f1_xgb_train, f1_lgb_train, f1_svm_train, f1_ann_train],
    "Kappa": [kappa_logist_train, kappa_tree_train, kappa_rf_train, kappa_xgb_train, kappa_lgb_train, kappa_svm_train, kappa_ann_train],
    "Youden's J": [youden_logist_train, youden_tree_train, youden_rf_train, youden_xgb_train, youden_lgb_train, youden_svm_train, youden_ann_train],
    "PPV": [ppv_logist_train, ppv_tree_train, ppv_rf_train, ppv_xgb_train, ppv_lgb_train, ppv_svm_train, ppv_ann_train],
    "NPV": [npv_logist_train, npv_tree_train, npv_rf_train, npv_xgb_train, npv_lgb_train, npv_svm_train, npv_ann_train]
})

model_results_train
model_results_train.to_csv("4.训练集模型评估/model_performance_train.csv", index=False)  # 保存为CSV文件

###################### 6. 绘制ROC曲线 ##########################
plt.figure(figsize=(8, 6))
models_train = {
    "Logistic": (y_train_logist, y_train_pred_prob_logist),
    "Decision Tree": (y_train, y_train_pred_prob_tree),
    "Random Forest": (y_train, y_train_pred_prob_rf),
    "XGBoost": (y_train, y_train_pred_prob_xgb),
    "LightGBM": (y_train, y_train_pred_prob_lgb),
    "SVM": (y_train, y_train_pred_prob_svm),
    "ANN": (y_train, y_train_pred_prob_ann)
}

# 计算每个模型的AUC及其95%置信区间
for model_name, (y_true, y_pred_prob) in models_train.items():
    auc_value, auc_ci_lower, auc_ci_upper = calculate_auc(y_true, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f} [95% CI: {auc_ci_lower:.3f} - {auc_ci_upper:.3f}])")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("1-Specificity")
plt.ylabel("Sensitivity")
plt.title("ROC Curve - Training Set Model Comparison")
plt.legend(loc='lower right')  # 设置图例字体大小为8
plt.grid()
plt.savefig("4.训练集模型评估/ROC_curves_allmodel_train.pdf", dpi=300)
plt.show()

###################### 7. 绘制校准曲线 ##########################
plt.figure(figsize=(10, 8))
models_train = {
    "Logistic": (y_train_logist, y_train_pred_prob_logist),
    "Decision Tree": (y_train, y_train_pred_prob_tree),
    "Random Forest": (y_train, y_train_pred_prob_rf),
    "XGBoost": (y_train, y_train_pred_prob_xgb),
    "LightGBM": (y_train, y_train_pred_prob_lgb),
    "SVM": (y_train, y_train_pred_prob_svm),
    "ANN": (y_train, y_train_pred_prob_ann)
}
for model_name, (y_true, y_pred_prob) in models_train.items():
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    
    # 计算Brier分数及其置信区间
    brier_mean, brier_lower, brier_upper = brier_score_confidence_interval(y_true, y_pred_prob)
    
    plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name} (Brier Score: {brier_mean:.3f} [95% CI: {brier_lower:.3f} - {brier_upper:.3f}])')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curves - Training Set Model Comparison")
plt.legend(loc='lower right')  # 统一图例位置
plt.grid()
plt.savefig("4.训练集模型评估/Calibration_curves_allmodel_train.pdf", dpi=300)  # 修改保存路径
plt.show()

###################### 8. 绘制DCA曲线 ##########################
plt.figure(figsize=(10, 8))
thresholds = np.linspace(0.01, 0.99, 100)  # 统一阈值范围

models_train = {
    "Logistic": (y_train_logist, y_train_pred_prob_logist),
    "Decision Tree": (y_train, y_train_pred_prob_tree),
    "Random Forest": (y_train, y_train_pred_prob_rf),
    "XGBoost": (y_train, y_train_pred_prob_xgb),
    "LightGBM": (y_train, y_train_pred_prob_lgb),
    "SVM": (y_train, y_train_pred_prob_svm),
    "ANN": (y_train, y_train_pred_prob_ann)
}

# 计算所有模型的净收益
for model_name, (y_true, y_pred_prob) in models_train.items():
    net_benefit, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefit(y_true, y_pred_prob, thresholds=thresholds)
    plt.plot(thresholds, net_benefit, label=model_name)

# 绘制“Treat All”和“Treat None”曲线
plt.plot(thresholds, net_benefit_alltrt, linestyle="--", color="red", label="Treat All")
plt.plot(thresholds, net_benefits_notrt, linestyle="--", color="green", label="Treat None")

plt.xlabel("Threshold Probability")
plt.ylim(-0.2, np.nanmax(np.array(net_benefit)) + 0.1)
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis - Training Set Model Comparison")
plt.legend(loc="lower right", fontsize=8)
plt.grid()
plt.savefig("4.训练集模型评估/DCA_curves_allmodel_train.pdf", dpi=300)  
plt.show()

################################################################################################
####################★★★★★★★★★外部测试集验证★★★★★★★★★ ###################################
################################################################################################
test_data = pd.read_csv("testdata.csv", encoding="GBK")     #我用的训练集数据做演示，记得换成你的测试集的数据名字⭐⭐
# 得到测试集数据（Logistic模型）
X_test_logist = test_data[[c for c in test_data.columns if c != 'Outcome']]
X_test_logist_const = sm.add_constant(X_test_logist)
y_test_logist = test_data['Outcome']
# 定义变量类型
categorical_vars = [c for c in ['Outcome', 'stroke'] if c in test_data.columns]  # ★本数据集仅 stroke 为分类变量(结局 Outcome 同按分类处理); 连续变量保持数值, 不再误转 category
for var in categorical_vars:
    test_data[var] = test_data[var].astype('category') 
print(test_data.info()) 
# 得到测试数据集（机器学习模型）
test_data_ml = pd.read_csv("testdata.csv", encoding="GBK")  #我用的训练集数据做演示，记得换成你的测试集的数据名字⭐⭐
X_test = test_data_ml[[c for c in test_data_ml.columns if c != 'Outcome']]
y_test = test_data_ml['Outcome'] 

###################### 1. 计算测试数据集预测结果 ##########################
# 1.1. Logistic模型
y_test_pred_prob_logist = logist_model.predict(X_test_logist_const) # 预测概率
y_test_pred_logist = (y_test_pred_prob_logist >= 0.5).astype(int) # 预测分类值（阈值0.5）
# 1.2. 决策树模型
y_test_pred_prob_tree = tree_model.predict_proba(X_test)[:, 1]
y_test_pred_tree = (y_test_pred_prob_tree >= 0.5).astype(int)
# 1.3. 随机森林模型
y_test_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_test_pred_rf = (y_test_pred_prob_rf >= 0.5).astype(int)
# 1.4. XGBoost模型
y_test_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_test_pred_xgb = (y_test_pred_prob_xgb >= 0.5).astype(int)
# 1.5. LightGBM模型
y_test_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_test_pred_lgb = (y_test_pred_prob_lgb >= 0.5).astype(int)
# 1.6. SVM模型
y_test_pred_prob_svm = svm_model.predict_proba(X_test)[:, 1]
y_test_pred_svm = (y_test_pred_prob_svm >= 0.5).astype(int)
# 1.7. ANN模型
y_test_pred_prob_ann = ann_model.predict_proba(X_test)[:, 1]
y_test_pred_ann = (y_test_pred_prob_ann >= 0.5).astype(int)

###################### 2. 计算混淆矩阵并可视化 ##########################
output_folder = "3.外部测试集混淆矩阵"    #建好文件夹
# 2.1. Logistic模型
cm_logist_test = confusion_matrix(y_test_logist, y_test_pred_logist)
CM_plot(cm_logist_test, "Logistic")
# 2.2. 决策树模型
cm_tree_test = confusion_matrix(y_test, y_test_pred_tree)
CM_plot(cm_tree_test, "DecisionTree")
# 2.3. 随机森林模型
cm_rf_test = confusion_matrix(y_test, y_test_pred_rf)
CM_plot(cm_rf_test, "RandomForest")
# 2.4. XGBoost模型
cm_xgb_test = confusion_matrix(y_test, y_test_pred_xgb)
CM_plot(cm_xgb_test, "XGBoost")
# 2.5. LightGBM模型
cm_lgb_test = confusion_matrix(y_test, y_test_pred_lgb)
CM_plot(cm_lgb_test, "LightGBM")
# 2.6. SVM模型
cm_svm_test = confusion_matrix(y_test, y_test_pred_svm)
CM_plot(cm_svm_test, "SVM")
# 2.7. ANN模型
cm_ann_test = confusion_matrix(y_test, y_test_pred_ann)
CM_plot(cm_ann_test, "ANN")

###################### 3. 计算准确率、精确率、灵敏度、f1分数、特异度、kappa、约登、等等 ##########################
# 4.1. Logistic 模型
accuracy_logist_test, precision_logist_test, sensitivity_logist_test, f1_logist_test, specificity_logist_test, \
kappa_logist_test, youden_logist_test, ppv_logist_test, npv_logist_test = \
    calculate_acc_pre_sen_f1_spc(cm_logist_test, y_test_logist, y_test_pred_logist)

# 4.2. 决策树模型
accuracy_tree_test, precision_tree_test, sensitivity_tree_test, f1_tree_test, specificity_tree_test, \
kappa_tree_test, youden_tree_test, ppv_tree_test, npv_tree_test = \
    calculate_acc_pre_sen_f1_spc(cm_tree_test, y_test, y_test_pred_tree)

# 4.3. 随机森林模型
accuracy_rf_test, precision_rf_test, sensitivity_rf_test, f1_rf_test, specificity_rf_test, \
kappa_rf_test, youden_rf_test, ppv_rf_test, npv_rf_test = \
    calculate_acc_pre_sen_f1_spc(cm_rf_test, y_test, y_test_pred_rf)

# 4.4. XGBoost 模型
accuracy_xgb_test, precision_xgb_test, sensitivity_xgb_test, f1_xgb_test, specificity_xgb_test, \
kappa_xgb_test, youden_xgb_test, ppv_xgb_test, npv_xgb_test = \
    calculate_acc_pre_sen_f1_spc(cm_xgb_test, y_test, y_test_pred_xgb)

# 4.5. LightGBM 模型
accuracy_lgb_test, precision_lgb_test, sensitivity_lgb_test, f1_lgb_test, specificity_lgb_test, \
kappa_lgb_test, youden_lgb_test, ppv_lgb_test, npv_lgb_test = \
    calculate_acc_pre_sen_f1_spc(cm_lgb_test, y_test, y_test_pred_lgb)

# 4.6. SVM 模型
accuracy_svm_test, precision_svm_test, sensitivity_svm_test, f1_svm_test, specificity_svm_test, \
kappa_svm_test, youden_svm_test, ppv_svm_test, npv_svm_test = \
    calculate_acc_pre_sen_f1_spc(cm_svm_test, y_test, y_test_pred_svm)

# 4.7. ANN 模型
accuracy_ann_test, precision_ann_test, sensitivity_ann_test, f1_ann_test, specificity_ann_test, \
kappa_ann_test, youden_ann_test, ppv_ann_test, npv_ann_test = \
    calculate_acc_pre_sen_f1_spc(cm_ann_test, y_test, y_test_pred_ann)

###################### 4. 计算AUC及其95%置信区间 ##########################
# 4.1. Logistic模型
auc_value_logist_test, auc_ci_lower_logist_test, auc_ci_upper_logist_test = calculate_auc(y_test_logist, y_test_pred_prob_logist)
# 4.2. 决策树模型
auc_value_tree_test, auc_ci_lower_tree_test, auc_ci_upper_tree_test = calculate_auc(y_test, y_test_pred_prob_tree)
# 4.3. 随机森林模型
auc_value_rf_test, auc_ci_lower_rf_test, auc_ci_upper_rf_test = calculate_auc(y_test, y_test_pred_prob_rf)
# 4.4. XGBoost模型
auc_value_xgb_test, auc_ci_lower_xgb_test, auc_ci_upper_xgb_test = calculate_auc(y_test, y_test_pred_prob_xgb)
# 4.5. LightGBM模型
auc_value_lgb_test, auc_ci_lower_lgb_test, auc_ci_upper_lgb_test = calculate_auc(y_test, y_test_pred_prob_lgb)
# 4.6. SVM模型
auc_value_svm_test, auc_ci_lower_svm_test, auc_ci_upper_svm_test = calculate_auc(y_test, y_test_pred_prob_svm)
# 4.7. ANN模型
auc_value_ann_test, auc_ci_lower_ann_test, auc_ci_upper_ann_test = calculate_auc(y_test, y_test_pred_prob_ann)

###################### 5. 所有模型的测试集预测效果汇总，方便对比 ##########################
model_results_test = pd.DataFrame({
    "Model": ["Logistic", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "SVM", "ANN"],
    "AUC": [auc_value_logist_test, auc_value_tree_test, auc_value_rf_test, auc_value_xgb_test, auc_value_lgb_test, auc_value_svm_test, auc_value_ann_test],
    "AUC 95% CI Lower": [auc_ci_lower_logist_test, auc_ci_lower_tree_test, auc_ci_lower_rf_test, auc_ci_lower_xgb_test, auc_ci_lower_lgb_test, auc_ci_lower_svm_test, auc_ci_lower_ann_test],
    "AUC 95% CI Upper": [auc_ci_upper_logist_test, auc_ci_upper_tree_test, auc_ci_upper_rf_test, auc_ci_upper_xgb_test, auc_ci_upper_lgb_test, auc_ci_upper_svm_test, auc_ci_upper_ann_test],
    "Accuracy":  [accuracy_logist_test,  accuracy_tree_test,  accuracy_rf_test, accuracy_xgb_test,  accuracy_lgb_test,  accuracy_svm_test,  accuracy_ann_test],
    "Precision": [precision_logist_test, precision_tree_test, precision_rf_test, precision_xgb_test, precision_lgb_test, precision_svm_test, precision_ann_test],
    "Sensitivity": [sensitivity_logist_test, sensitivity_tree_test, sensitivity_rf_test, sensitivity_xgb_test, sensitivity_lgb_test, sensitivity_svm_test, sensitivity_ann_test],
    "Specificity": [specificity_logist_test, specificity_tree_test, specificity_rf_test, specificity_xgb_test, specificity_lgb_test, specificity_svm_test, specificity_ann_test],
    "F1 Score": [f1_logist_test, f1_tree_test, f1_rf_test, f1_xgb_test, f1_lgb_test, f1_svm_test, f1_ann_test],
    "Kappa": [kappa_logist_test, kappa_tree_test, kappa_rf_test, kappa_xgb_test, kappa_lgb_test, kappa_svm_test, kappa_ann_test],
    "Youden's J": [youden_logist_test, youden_tree_test, youden_rf_test, youden_xgb_test, youden_lgb_test, youden_svm_test, youden_ann_test],
    "PPV": [ppv_logist_test, ppv_tree_test, ppv_rf_test, ppv_xgb_test, ppv_lgb_test, ppv_svm_test, ppv_ann_test],
    "NPV": [npv_logist_test, npv_tree_test, npv_rf_test, npv_xgb_test, npv_lgb_test, npv_svm_test, npv_ann_test]
})

model_results_test
model_results_test.to_csv("4.外部测试集模型评估/model_performance_test.csv", index=False)  # 保存为CSV文件

###################### 6. 绘制ROC曲线 ##########################
# 汇总模型评估
plt.figure(figsize=(10, 8))
models_test = {
    "Logistic": (y_test_logist, y_test_pred_prob_logist),
    "Decision Tree": (y_test, y_test_pred_prob_tree),
    "Random Forest": (y_test, y_test_pred_prob_rf),
    "XGBoost": (y_test, y_test_pred_prob_xgb),
    "LightGBM": (y_test, y_test_pred_prob_lgb),
    "SVM": (y_test, y_test_pred_prob_svm),
    "ANN": (y_test, y_test_pred_prob_ann)
}

# 计算每个模型的AUC及其95%置信区间
for model_name, (y_true, y_pred_prob) in models_test.items():
    auc_value, auc_ci_lower, auc_ci_upper = calculate_auc(y_true, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f} [95% CI: {auc_ci_lower:.3f} - {auc_ci_upper:.3f}])")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("1-Specificity")
plt.ylabel("Sensitivity")
plt.title("ROC Curve - Test Set Model Comparison")
plt.legend(loc='lower right')  # 设置图例字体大小为8
plt.grid()
plt.savefig("4.外部测试集模型评估/ROC_curves_allmodel_test.pdf", dpi=300)
plt.show()

# 单模型评估
output_folder = "4.外部测试集模型评估/单模型评估"    #建好文件夹⭐
# 1. Logistic
ROC_plot(y_test_logist, y_test_pred_prob_logist, auc_value_logist_test, auc_ci_lower_logist_test, auc_ci_upper_logist_test, "Logistic")
# 2. 决策树
ROC_plot(y_test, y_test_pred_prob_tree, auc_value_tree_test, auc_ci_lower_tree_test, auc_ci_upper_tree_test, "DecisionTree")
# 3. 随机森林
ROC_plot(y_test, y_test_pred_prob_rf, auc_value_rf_test, auc_ci_lower_rf_test, auc_ci_upper_rf_test, "RandomForest")
# 4. XGBoost
ROC_plot(y_test, y_test_pred_prob_xgb, auc_value_xgb_test, auc_ci_lower_xgb_test, auc_ci_upper_xgb_test, "XGBoost")
# 5. LightGBM
ROC_plot(y_test, y_test_pred_prob_lgb, auc_value_lgb_test, auc_ci_lower_lgb_test, auc_ci_upper_lgb_test, "LightGBM")
# 6. SVM
ROC_plot(y_test, y_test_pred_prob_svm, auc_value_svm_test, auc_ci_lower_svm_test, auc_ci_upper_svm_test, "SVM")
# 7. ANN
ROC_plot(y_test, y_test_pred_prob_ann, auc_value_ann_test, auc_ci_lower_ann_test, auc_ci_upper_ann_test, "ANN")

###################### 7. 绘制校准曲线 ##########################
plt.figure(figsize=(12, 10))
models_test = {
    "Logistic": (y_test_logist, y_test_pred_prob_logist),
    "Decision Tree": (y_test, y_test_pred_prob_tree),
    "Random Forest": (y_test, y_test_pred_prob_rf),
    "XGBoost": (y_test, y_test_pred_prob_xgb),
    "LightGBM": (y_test, y_test_pred_prob_lgb),
    "SVM": (y_test, y_test_pred_prob_svm),
    "ANN": (y_test, y_test_pred_prob_ann)
}
for model_name, (y_true, y_pred_prob) in models_test.items():
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    
    # 计算Brier分数及其置信区间
    brier_mean, brier_lower, brier_upper = brier_score_confidence_interval(y_true, y_pred_prob)
    
    plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name} (Brier Score: {brier_mean:.3f} [95% CI: {brier_lower:.3f} - {brier_upper:.3f}])')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curves - Test Set Model Comparison")
plt.legend(loc='lower right')  # 统一图例位置
plt.grid()
plt.savefig("4.外部测试集模型评估//Calibration_curves_allmodel_test.pdf", dpi=300)  # 修改保存路径
plt.show()

# 单模型评估
output_folder = "4.外部测试集模型评估/单模型评估"    #建好文件夹⭐
# 1. Logistic
CaliC_plot(y_test_logist, y_test_pred_prob_logist, "Logistic")
# 2. 决策树
CaliC_plot(y_test, y_test_pred_prob_tree, "DecisionTree")
# 3. 随机森林
CaliC_plot(y_test, y_test_pred_prob_rf, "RandomForest")
# 4. XGBoost
CaliC_plot(y_test, y_test_pred_prob_xgb, "XGBoost")
# 5. LightGBM
CaliC_plot(y_test, y_test_pred_prob_lgb, "LightGBM")
# 6. SVM
CaliC_plot(y_test, y_test_pred_prob_svm, "SVM")
# 7. ANN
CaliC_plot(y_test, y_test_pred_prob_ann, "ANN")

###################### 8. 绘制DCA曲线 ##########################
plt.figure(figsize=(10, 8))
thresholds = np.linspace(0.01, 0.99, 100)  # 统一阈值范围

models_test = {
    "Logistic": (y_test_logist, y_test_pred_prob_logist),
    "Decision Tree": (y_test, y_test_pred_prob_tree),
    "Random Forest": (y_test, y_test_pred_prob_rf),
    "XGBoost": (y_test, y_test_pred_prob_xgb),
    "LightGBM": (y_test, y_test_pred_prob_lgb),
    "SVM": (y_test, y_test_pred_prob_svm),
    "ANN": (y_test, y_test_pred_prob_ann)
}

# 计算所有模型的净收益
for model_name, (y_true, y_pred_prob) in models_test.items():
    net_benefit, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefit(y_true, y_pred_prob, thresholds=thresholds)
    plt.plot(thresholds, net_benefit, label=model_name)

# 绘制“Treat All”和“Treat None”曲线
plt.plot(thresholds, net_benefit_alltrt, linestyle="--", color="red", label="Treat All")
plt.plot(thresholds, net_benefits_notrt, linestyle="--", color="green", label="Treat None")

plt.xlabel("Threshold Probability")
plt.ylim(-0.2, np.nanmax(np.array(net_benefit)) + 0.1)
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis - Test Set Model Comparison")
plt.legend(loc="lower right", fontsize=8)
plt.grid()
plt.savefig("4.外部测试集模型评估/DCA_curves_allmodel_test.pdf", dpi=300)  
plt.show()

# 单模型评估
output_folder = "4.外部测试集模型评估/单模型评估"    #建好文件夹⭐
# 1. Logistic
net_benefit_logist, net_benefit_alltrt_logist, net_benefits_notrt_logist = calculate_net_benefit(y_test_logist, y_test_pred_prob_logist)
DCA_plot(net_benefit_logist, net_benefit_alltrt_logist, net_benefits_notrt_logist, "Logistic")

# 2. 决策树
net_benefit_tree, net_benefit_alltrt_tree, net_benefits_notrt_tree = calculate_net_benefit(y_test, y_test_pred_prob_tree)
DCA_plot(net_benefit_tree, net_benefit_alltrt_tree, net_benefits_notrt_tree, "DecisionTree")

# 3. 随机森林
net_benefit_rf, net_benefit_alltrt_rf, net_benefits_notrt_rf = calculate_net_benefit(y_test, y_test_pred_prob_rf)
DCA_plot(net_benefit_rf, net_benefit_alltrt_rf, net_benefits_notrt_rf, "RandomForest")

# 4. XGBoost
net_benefit_xgb, net_benefit_alltrt_xgb, net_benefits_notrt_xgb = calculate_net_benefit(y_test, y_test_pred_prob_xgb)
DCA_plot(net_benefit_xgb, net_benefit_alltrt_xgb, net_benefits_notrt_xgb, "XGBoost")

# 5. LightGBM
net_benefit_lgb, net_benefit_alltrt_lgb, net_benefits_notrt_lgb = calculate_net_benefit(y_test, y_test_pred_prob_lgb)
DCA_plot(net_benefit_lgb, net_benefit_alltrt_lgb, net_benefits_notrt_lgb, "LightGBM")

# 6. SVM
net_benefit_svm, net_benefit_alltrt_svm, net_benefits_notrt_svm = calculate_net_benefit(y_test, y_test_pred_prob_svm)
DCA_plot(net_benefit_svm, net_benefit_alltrt_svm, net_benefits_notrt_svm, "SVM")

# 7. ANN
net_benefit_ann, net_benefit_alltrt_ann, net_benefits_notrt_ann = calculate_net_benefit(y_test, y_test_pred_prob_ann)
DCA_plot(net_benefit_ann, net_benefit_alltrt_ann, net_benefits_notrt_ann, "ANN")

################################################################################################
############################### 最优模型的SHAP解释 ###############################################
################################################################################################

####################################特征图、蜂群图、依赖图##############################################
save_path = "5.SHAP解释/"
def shap_explain_model(model, X_train, X_val, model_name, n_interpret=150, obs_index=5):
    # 1. 建立 SHAP 核解释器（KernelExplainer 适用于任何模型）
    explainer = shap.KernelExplainer(lambda X: model.predict_proba(X), X_train)
    shap_values = explainer(X_val.iloc[:n_interpret, :])
    # 2.1 特征重要性条形图（全局 SHAP 重要性）
    plt.figure(figsize=(15, 12))
    shap.plots.bar(shap_values[:, :, 1], max_display=15, show=False)
    plt.savefig(f"{save_path}bar_{model_name}.pdf", dpi=200, bbox_inches='tight')
    plt.close()
    # 2.2 特征重要性蜂群图
    plt.figure(figsize=(15, 12))
    shap.plots.beeswarm(shap_values[:, :, 1], max_display=15, show=False)
    plt.savefig(f"{save_path}beeswarm_{model_name}.pdf", dpi=200, bbox_inches='tight')
    plt.close()
    # 2.3 特征值与 SHAP 值的依赖图
    shap_values_df = pd.DataFrame(shap_values[:, :, 1].values, columns=X_val.columns)
    features = shap_values_df.columns.tolist()
    # 设置画布和子图结构（根据特征数量动态调整）
    num_features = len(features)
    num_columns = 3
    num_rows = (num_features + num_columns - 1) // num_columns  # 向上取整
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(40, 10 * num_rows), dpi=300)
    axes = axes.flatten()
    # 循环绘制每个特征的散点图
    for i in range(num_features):
        feature = features[i]
        ax = axes[i]
        # 绘制散点图
        scatter = ax.scatter(X_val[feature][:n_interpret], shap_values_df[feature], 
                             s=20, alpha=0.7, c=X_val[feature][:n_interpret], cmap='viridis')
        ax.axhline(y=0, color='darkorange', linestyle='--', linewidth=1.5)  # 添加横线，颜色改为橙色，线型改为虚线
        # LOWESS 拟合
        lowess_fit = lowess(shap_values_df[feature], X_val[feature][:n_interpret], frac=0.4)  # frac 增大，平滑程度更高
        ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], color='darkred', linewidth=2.5, linestyle='-')  # 改变拟合曲线的颜色和线型
        # 添加标签
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel(f'SHAP value for\n{feature}', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Feature value', fontsize=10)
    # 隐藏多余的子图
    for i in range(num_features, num_rows * num_columns):
        axes[i].axis('off')
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f"{save_path}scatter_{model_name}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"所有 SHAP 解释性图像已保存在 {save_path}")

###################### 1. 决策树模型的可解释性 ##########################
shap_explain_model(model=tree_model, X_train=X_train, X_val=X_val, 
                   model_name="tree", n_interpret=150, obs_index=5)

###################### 2. 随机森林模型的可解释性 ##########################
shap_explain_model(model=rf_model, X_train=X_train, X_val=X_val, 
                   model_name="rf", n_interpret=150, obs_index=5)

###################### 3. XGBoost模型的可解释性 ##########################
shap_explain_model(model=xgb_model, X_train=X_train, X_val=X_val, 
                   model_name="xgb", n_interpret=150, obs_index=5)

###################### 4. LightGBM模型的可解释性 ##########################
shap_explain_model(model=lgb_model, X_train=X_train, X_val=X_val, 
                   model_name="lgb", n_interpret=150, obs_index=5)

###################### 5. SVM模型的可解释性 ##########################
shap_explain_model(model=svm_model, X_train=X_train, X_val=X_val, 
                   model_name="svm", n_interpret=150, obs_index=5)

###################### 6. ANN模型的可解释性 ##########################
shap_explain_model(model=ann_model, X_train=X_train, X_val=X_val, 
                   model_name="ann", n_interpret=150, obs_index=5)

####################################力图、瀑布图##############################################
save_path = "5.SHAP解释/"
def shap_explain_model(model, X_train, X_val, model_name, n_interpret=150):
    # 1. 建立 SHAP 核解释器（KernelExplainer 适用于任何模型）
    explainer = shap.KernelExplainer(lambda X: model.predict_proba(X), X_train)
    shap_values = explainer(X_val.iloc[:n_interpret, :])
    
    # 2.3 瀑布图（展示第1个观测的类别0和类别1）
    for class_index in [0, 1]:
        plt.figure(figsize=(15, 12))
        shap.plots.waterfall(shap_values[0][:, class_index], max_display=15, show=False)
        plt.title(f"SHAP Waterfall Plot for Class {class_index}")
        # 保存为 PDF 文件，路径为 save_path 下
        plt.savefig(f"{save_path}{model_name}_waterfall_1st_class{class_index}.pdf", format="pdf", bbox_inches='tight')
        plt.show()
        plt.close()

    # 2.4 力图（展示第3个观测的类别0和类别1）
    for class_index in [0, 1]:
        force_plot = shap.plots.force(shap_values[2][:, class_index])
        # 保存为 HTML 文件，路径为 save_path 下
        shap.save_html(f"{save_path}{model_name}_force_plot_3rd_class{class_index}.html", force_plot)
    
    print(f"{model_name} 的 SHAP 解释性图像已生成。")
###################### 1. 决策树模型的可解释性 ##########################
shap_explain_model(model=tree_model, X_train=X_train, X_val=X_val, 
                   model_name="tree", n_interpret=150)   
###################### 2. 随机森林模型的可解释性 ##########################
shap_explain_model(model=rf_model, X_train=X_train, X_val=X_val, 
                   model_name="rf", n_interpret=150)  
###################### 3. XGBoost模型的可解释性 ##########################
shap_explain_model(model=xgb_model, X_train=X_train, X_val=X_val, 
                   model_name="xgb", n_interpret=150)
###################### 4. LightGBM模型的可解释性 ##########################
shap_explain_model(model=lgb_model, X_train=X_train, X_val=X_val, 
                   model_name="lgb", n_interpret=150)
###################### 5. SVM模型的可解释性 ##########################
shap_explain_model(model=svm_model, X_train=X_train, X_val=X_val, 
                   model_name="svm", n_interpret=150)
###################### 6. ANN模型的可解释性 ##########################
shap_explain_model(model=ann_model, X_train=X_train, X_val=X_val, 
                   model_name="ann", n_interpret=150)


####################### 附: PR 曲线与 AUPRC (审稿意见 9) #######################
##############################################################################
# 低患病率下 ROC AUC 偏乐观, 补充 Precision-Recall 曲线与 AUPRC。
# 针对最终锁定的 Logistic 回归模型 (确定性算法, 重拟合结果与上文完全一致),
# 在内部验证集与外部测试集 (MIMIC) 上分别计算 AUPRC 并绘制 PR 曲线。
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression

pr_train = pd.read_csv("traindata.csv", encoding="GBK")
pr_val   = pd.read_csv("valdata.csv", encoding="GBK")
pr_test  = pd.read_csv("testdata.csv", encoding="GBK")

FEATS = [c for c in pr_train.columns if c != 'Outcome']
Xtr, ytr = pr_train[FEATS], pr_train['Outcome']
Xva, yva = pr_val[FEATS],   pr_val['Outcome']
Xte, yte = pr_test[FEATS],  pr_test['Outcome']

# 重拟合最终 LR (无随机性; 与上文 statsmodels Logit 等价)
lr_final = LogisticRegression(penalty=None, max_iter=5000)   # 无惩罚, 与 statsmodels Logit 一致
lr_final.fit(Xtr, ytr)

p_va = lr_final.predict_proba(Xva)[:, 1]
p_te = lr_final.predict_proba(Xte)[:, 1]

ap_va = average_precision_score(yva, p_va)
ap_te = average_precision_score(yte, p_te)
prev_va, prev_te = yva.mean(), yte.mean()
print(f"AUPRC 内部验证集 = {ap_va:.3f} (基线患病率 {prev_va:.3f})")
print(f"AUPRC 外部测试集 = {ap_te:.3f} (基线患病率 {prev_te:.3f})")

pd.DataFrame({
    'cohort': ['Internal validation', 'External test (MIMIC-IV)'],
    'AUPRC': [round(ap_va, 3), round(ap_te, 3)],
    'baseline_prevalence': [round(prev_va, 3), round(prev_te, 3)]
}).to_csv("auprc_summary.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y, p, ap, prev, name in [
    (axes[0], yva, p_va, ap_va, prev_va, 'Internal validation'),
    (axes[1], yte, p_te, ap_te, prev_te, 'External test (MIMIC-IV)')]:
    prec, rec, _ = precision_recall_curve(y, p)
    ax.plot(rec, prec, color='#C0392B', lw=2,
            label=f'Logistic (AUPRC = {ap:.3f})')
    ax.axhline(prev, ls='--', color='grey', lw=1,
               label=f'Baseline (prevalence = {prev:.3f})')
    ax.set_xlabel('Recall (Sensitivity)'); ax.set_ylabel('Precision (PPV)')
    ax.set_title(f'Precision-Recall Curve - {name}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("PR_curves_Logistic.pdf", bbox_inches='tight')
plt.show()
print("已输出: auprc_summary.csv, PR_curves_Logistic.pdf")

##############################################################################
############ 附2: 训练集 Youden 阈值重算分类指标 (审稿意见 9, 方案 A) ############
##############################################################################
# 分类阈值 = 训练集 ROC 曲线上 Youden 指数 (敏感度+特异度-1) 最大的切点,
# 在训练集内确定后锁定, 原样应用于内部验证集与外部测试集 (MIMIC-IV)。
# AUC 及其 95% CI 与阈值无关, 沿用原表; 其余分类指标按锁定阈值重算。

prob_train_dict = {
    'Logistic':      np.asarray(logist_model.predict(sm.add_constant(X_train))),
    'Decision Tree': tree_model.predict_proba(X_train)[:, 1],
    'Random Forest': rf_model.predict_proba(X_train)[:, 1],
    'XGBoost':       xgb_model.predict_proba(X_train)[:, 1],
    'LightGBM':      lgb_model.predict_proba(X_train)[:, 1],
    'SVM':           svm_model.predict_proba(X_train)[:, 1],
    'ANN':           ann_model.predict_proba(X_train)[:, 1],
}
prob_val_dict = {
    'Logistic': y_val_pred_prob_logist, 'Decision Tree': y_val_pred_prob_tree,
    'Random Forest': y_val_pred_prob_rf, 'XGBoost': y_val_pred_prob_xgb,
    'LightGBM': y_val_pred_prob_lgb, 'SVM': y_val_pred_prob_svm,
    'ANN': y_val_pred_prob_ann,
}
prob_test_dict = {
    'Logistic': y_test_pred_prob_logist, 'Decision Tree': y_test_pred_prob_tree,
    'Random Forest': y_test_pred_prob_rf, 'XGBoost': y_test_pred_prob_xgb,
    'LightGBM': y_test_pred_prob_lgb, 'SVM': y_test_pred_prob_svm,
    'ANN': y_test_pred_prob_ann,
}

# 1) 训练集内确定各模型 Youden 最佳阈值并锁定
youden_thresholds = {}
for name, p_tr in prob_train_dict.items():
    fpr_t, tpr_t, thr_t = roc_curve(y_train, p_tr)
    youden_thresholds[name] = float(thr_t[np.argmax(tpr_t - fpr_t)])

pd.DataFrame({'Model': list(youden_thresholds.keys()),
              'Youden_threshold': [round(v, 4) for v in youden_thresholds.values()]
              }).to_csv("youden_thresholds.csv", index=False)

# 2) 按锁定阈值重算分类指标 (AUC 与 CI 沿用原表)
def _metrics_at_thr(y_true, prob, thr):
    pred = (np.asarray(prob) >= thr).astype(int)
    cm = confusion_matrix(y_true, pred)
    acc, prec, sens, f1, spec, kap, j, ppv, npv = \
        calculate_acc_pre_sen_f1_spc(cm, y_true, pred)
    return acc, prec, sens, spec, f1, kap, j, ppv, npv

def _rebuild_table(csv_path, prob_dict, y_true, out_path):
    old = pd.read_csv(csv_path)
    auc_cols = [c for c in old.columns if ('AUC' in c or 'CI' in c)]
    rows = []
    for _, r in old.iterrows():
        name = r['Model']
        thr = youden_thresholds[name]
        acc, prec, sens, spec, f1, kap, j, ppv, npv = \
            _metrics_at_thr(y_true, prob_dict[name], thr)
        row = {'Model': name}
        for c in auc_cols:
            row[c] = r[c]
        row.update({'Threshold': round(thr, 3), 'Accuracy': acc,
                    'Precision': prec, 'Sensitivity': sens,
                    'Specificity': spec, 'F1 Score': f1, 'Kappa': kap,
                    "Youden's J": j, 'PPV': ppv, 'NPV': npv})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out

val_youden = _rebuild_table(
    "4.验证集模型评估/model_performance_validation.csv",
    prob_val_dict, y_val,
    "4.验证集模型评估/model_performance_validation_youden.csv")
test_youden = _rebuild_table(
    "4.外部测试集模型评估/model_performance_test.csv",
    prob_test_dict, y_test,
    "4.外部测试集模型评估/model_performance_test_youden.csv")

print("\n===== Youden 阈值 (训练集内确定并锁定) =====")
for k, v in youden_thresholds.items():
    print(f"  {k}: {v:.4f}")
print("\n===== 内部验证集 (Youden 阈值) =====")
print(val_youden.round(3).to_string(index=False))
print("\n===== 外部测试集 MIMIC-IV (Youden 阈值) =====")
print(test_youden.round(3).to_string(index=False))
print("\n已输出: youden_thresholds.csv, "
      "model_performance_validation_youden.csv, "
      "model_performance_test_youden.csv")

##############################################################################
############ 附3: 校准截距 / 校准斜率 / Brier (审稿意见 10) ############
##############################################################################
# 针对最终锁定的 Logistic 回归模型, 在内部验证集与外部测试集 (MIMIC-IV) 上
# 定量报告校准度: 校准截距 (理想值 0), 校准斜率 (理想值 1), Brier 分数, 均附 95% CI。
# 截距模型: logit(P(y=1)) = a + offset(logit(p))
# 斜率模型: logit(P(y=1)) = a + b * logit(p)

def _calibration_stats(y_true, prob, n_boot=1000, seed=123):
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))                       # logit(p)
    # 校准截距 (offset 模型)
    m_int = sm.GLM(y, np.ones((len(y), 1)),
                   family=sm.families.Binomial(), offset=lp).fit()
    ci_int = m_int.conf_int()[0]
    # 校准斜率
    m_slp = sm.GLM(y, sm.add_constant(lp),
                   family=sm.families.Binomial()).fit()
    ci_slp = m_slp.conf_int()[1]
    # Brier + bootstrap 95% CI
    brier = np.mean((p - y) ** 2)
    rng = np.random.default_rng(seed)
    boot = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot.append(np.mean((p[idx] - y[idx]) ** 2))
    b_lo, b_hi = np.percentile(boot, [2.5, 97.5])
    return {
        'intercept': m_int.params[0], 'intercept_lo': ci_int[0], 'intercept_hi': ci_int[1],
        'slope': m_slp.params[1], 'slope_lo': ci_slp[0], 'slope_hi': ci_slp[1],
        'brier': brier, 'brier_lo': b_lo, 'brier_hi': b_hi,
    }

cal_rows = []
for cohort, y_true, prob in [
    ('Internal validation', y_val, y_val_pred_prob_logist),
    ('External test (MIMIC-IV)', y_test, y_test_pred_prob_logist),
]:
    s = _calibration_stats(y_true, prob)
    cal_rows.append({
        'cohort': cohort,
        'calibration_intercept (95% CI)':
            f"{s['intercept']:.3f} ({s['intercept_lo']:.3f} to {s['intercept_hi']:.3f})",
        'calibration_slope (95% CI)':
            f"{s['slope']:.3f} ({s['slope_lo']:.3f} to {s['slope_hi']:.3f})",
        'brier (95% CI)':
            f"{s['brier']:.3f} ({s['brier_lo']:.3f} to {s['brier_hi']:.3f})",
    })

cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv("calibration_summary.csv", index=False)
print("\n===== 校准度 (最终 Logistic 模型) =====")
print(cal_df.to_string(index=False))
print("\n已输出: calibration_summary.csv")
