from ctypes import CDLL
import os

dll_path = r"D:\\python\\Lib\\site-packages\\thundersvm\\thundersvm.dll"  # 替换为实际的 DLL 文件路
tem = CDLL(dll_path, winmode=0)

# import ctypes
# thundersvm_dll_path = r"D:\python\Lib\site-packages\thundersvm\thundersvm.dll"
# ctypes.CDLL(thundersvm_dll_path)

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import pickle
from itertools import product
from sklearn.preprocessing import LabelEncoder
from thundersvm import SVC


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper


def train_best_model(X_train, y_train_bin, param_grid):
    # 初始化最佳模型相关变量
    best_model = None
    best_accuracy = 0
    best_params = None

    # 遍历 param_grid 中的每一个参数组合
    for params in product(*param_grid.values()):  # 使用product生成参数组合
        # 创建当前组合的参数字典
        param_dict = dict(zip(param_grid.keys(), params))

        # 使用当前参数训练模型
        svm_model = SVC(**param_dict)
        svm_model.fit(X_train, y_train_bin)

        # 评估模型（这里我们使用训练集的准确度作为评估指标，你也可以改为交叉验证）
        y_pred = svm_model.predict(X_train)
        accuracy = accuracy_score(y_train_bin, y_pred)

        print(f"Trying parameters: {param_dict} | Accuracy: {accuracy:.4f}")

        # 如果当前模型性能更好，则更新最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = svm_model
            best_params = param_dict

    print(f"Best parameters: {best_params} | Best accuracy: {best_accuracy:.4f}")
    return best_model

    # # 保存最佳模型
    # with open("../model/best-xgb-thusvm-model.pkl", 'wb') as f:
    #     pickle.dump(best_model, f)

    # # 打印最佳模型的超参数
    # print(best_model.get_params())
    # return best_model


def convert_column_to_binary(df, column_name):
    # 为每个唯一单词分配一个从0开始的整数编号
    word_to_int = {word: idx for idx, word in enumerate(df[column_name].unique())}

    # 将每个单词转换为对应的整数编号
    df[column_name] = df[column_name].map(word_to_int)

    # 将整数编号转换为二进制，并去掉'0b'前缀
    binary_df = df[column_name].apply(lambda x: bin(x)[2:])

    # 找出最长的二进制串长度
    max_len = binary_df.apply(len).max()

    # 补齐二进制串，使其长度一致
    binary_df = binary_df.apply(lambda x: x.zfill(max_len))  # 用0填充前导零

    # 将二进制字符串拆分成每一位
    binary_columns = [f"bit_{i}" for i in range(max_len)]
    # binary_columns.append(f'bit_{max_len}')  # 添加最后一列
    binary_values = [list(x) for x in binary_df]  # 将每个二进制串拆分为单个字符

    # print(binary_values)
    # print(binary_columns)
    # 创建新的 DataFrame 来存储二进制列
    binary_df = pd.DataFrame(binary_values, columns=binary_columns)
    # print(binary_df)
    # 将原df与新的二进制列合并
    df = df.drop(columns=[column_name])
    df = pd.concat([df, binary_df], axis=1)

    return df


def process_data():
    # 读取数据
    train_df = pd.read_csv("../data/UNSW_NB15_testing-set.csv")
    test_df = pd.read_csv("../data/UNSW_NB15_training-set.csv")

    # 读取特征重要性文件并提取前20个特征
    feature_importance_df = pd.read_csv("../data/xgb_feature_importance.csv")
    top_20_features = feature_importance_df["Feature"].head(20).tolist()

    # 确保测试集和训练集包含相同的特征
    X_train = train_df[top_20_features]
    X_test = test_df[top_20_features]

    X_train = convert_column_to_binary(X_train, "proto")
    X_test = convert_column_to_binary(X_test, "proto")

    # X_train.to_csv('X_train.csv',index=False)

    # return

    # 对所有特征进行均值化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 二分类任务：预测label
    # y_train_bin = train_df["label"]
    # y_test_bin = test_df["label"]


    encoder = LabelEncoder()
    y_train_multi = encoder.fit_transform(train_df['attack_cat'])
    y_test_multi = encoder.transform(test_df['attack_cat'])

    return X_train, X_test, y_train_multi, y_test_multi
    # return X_train, X_test, y_train_bin, y_test_bin


@timer
def train_model(X_train, y_train_bin, param_grid):
    svm_model = SVC(C=param_grid['C'][0], gamma=param_grid['gamma'][0], kernel=param_grid['kernel'][0])
    svm_model.fit(X_train, y_train_bin)
    return svm_model


@timer
def predict(X_test, y_test_bin, model):
    # 读取模型
    # with open("../model/xgb-thusvm-model.pkl", 'rb') as f:
    #     svm_model = pickle.load(f)
    # print(svm_model.get_params())
    # 预测
    # svm_model=model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_bin, y_pred)
    print(f"SVM Model Accuracy: {accuracy:.4f}")


X_train, X_test, y_train_bin, y_test_bin = process_data()

# param_grid = {
#     'C': [0.1, 1, 10],  # 正则化参数
#     'gamma': [0.01, 0.1, 1],  # 核函数参数
#     'kernel': ['rbf', 'linear']  # 核函数类型
# }

param_grid = {
    "C": [10],  # 正则化参数
    "gamma": [0.1],  # 核函数参数
    "kernel": ["rbf"],  # 核函数类型
}


model = train_model(X_train, y_train_bin, param_grid)

predict(X_test, y_test_bin, model)


# return 1
# qew(X_train, y_train_bin)
