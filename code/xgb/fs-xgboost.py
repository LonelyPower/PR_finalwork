import pandas as pd
import xgboost as xgb
import shap


def process_data():
    df = pd.read_csv("../../data/UNSW_NB15_training-set.csv")

    # 删除不需要的列
    df = df.drop(columns=["service", "id"])

    # 将 'proto' 列转换为分类类型
    df["proto"] = df["proto"].astype("category")
    df["state"] = df["state"].astype("category")

    # 划分特征和标签
    X = df.drop(columns=["attack_cat", "label"])
    y = df["label"]
    return X, y


def train_model(X, y):
    # 加载数据

    # 将数据转换为DMatrix，并启用categorical
    dmatrix = xgb.DMatrix(X, label=y, enable_categorical=True)

    # 设置模型参数
    params = {
        "objective": "binary:logistic",  # 假设是二分类问题
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    # 使用 DMatrix 训练模型
    model = xgb.train(params, dmatrix, num_boost_round=100)

    model.save_model("../../model/xgboost.json")

    return model


def explain_model_with_xgb():

    model = xgb.Booster()

    # 加载模型
    model.load_model("../../model/xgboost.json")

    importance = model.get_score(importance_type="gain")

    # 将结果转换为 DataFrame 以便更易于查看
    xgb_df = pd.DataFrame(
        importance.items(), columns=["Feature", "XGBoost_Importance"]
    )
    xgb_df = xgb_df.sort_values(by="XGBoost_Importance", ascending=False)

    # print(xgb_df)
    return xgb_df


def explain_model_with_shap(X):
    # 确保X是分类变量时，使用 DMatrix 时启用 enable_categorical
    dmatrix = xgb.DMatrix(X, enable_categorical=True)
    model = xgb.Booster()

    # 加载模型
    model.load_model("../../model/xgboost.json")

    # 打印特征重要性（按平均 SHAP 值排序）

    # 使用 SHAP 解释模型
    explainer = shap.TreeExplainer(model)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(dmatrix)  # 这里传入的是 DMatrix

    shap_df = pd.DataFrame(shap_values, columns=X.columns)  # 假设 X 是你的特征数据

    # 计算每个特征的平均 SHAP 值
    mean_shap_values = shap_df.abs().mean().sort_values(ascending=False)
    # print(mean_shap_values)

    shap_df = pd.DataFrame(mean_shap_values).reset_index()
    shap_df.columns = ["Feature", "SHAP_Importance"]
    return shap_df
    # 可视化 SHAP summary plot
    # shap.summary_plot(shap_values, X)

    # # 可视化 SHAP dependence plot（以某个特征为例）
    # shap.dependence_plot("proto", shap_values, X)

    # # 可视化 SHAP force plot（以第一个样本为例）
    # shap.initjs()  # 初始化 JS 可视化
    # shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])


X, y = process_data()

# model = train_model(X, y)

# 使用 SHAP 解释模型
shap_df = explain_model_with_shap(X)
xgb_df = explain_model_with_xgb()
final_importance_df = pd.merge(xgb_df, shap_df, on="Feature", how="inner")

# 归一化处理：你可以选择将贡献值标准化，确保两个指标的影响力是平衡的
final_importance_df["Normalized_XGBoost_Importance"] = final_importance_df["XGBoost_Importance"] / final_importance_df["XGBoost_Importance"].max()
final_importance_df["Normalized_SHAP_Importance"] = final_importance_df["SHAP_Importance"] / final_importance_df["SHAP_Importance"].max()

# 求和得到最终的特征重要性
final_importance_df["Final_Importance"] = final_importance_df["Normalized_SHAP_Importance"]
# final_importance_df["Final_Importance"] = (final_importance_df["Normalized_XGBoost_Importance"] + final_importance_df["Normalized_SHAP_Importance"]) / 2

# 按照最终的特征重要性进行排序
final_importance_df = final_importance_df.sort_values(by="Final_Importance", ascending=False)

# 打印最终的特征重要性
print(final_importance_df)
# 假设 final_importance_df 是包含最终特征重要性的数据框
# 将最终结果保存为 CSV 文件
final_importance_df['Column_Number'] = final_importance_df.index
final_importance_df.to_csv('../../data/xgb_feature_importance4.csv', index=False)
# select_feature()
