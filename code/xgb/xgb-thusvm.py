from ctypes import CDLL
import os

dll_path = r"D:\\python\\Lib\\site-packages\\thundersvm\\thundersvm.dll"  # 替换为实际的 DLL 文件路
tem = CDLL(dll_path, winmode=0)

# import ctypes
# thundersvm_dll_path = r"D:\python\Lib\site-packages\thundersvm\thundersvm.dll"
# ctypes.CDLL(thundersvm_dll_path)
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import pickle
from itertools import product
from sklearn.preprocessing import LabelEncoder
from thundersvm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns

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


def convert_column_to_binary(df, column_name, id):
    # 为每个唯一单词分配一个从0开始的整数编号
    
    # print(word_to_int)
    # print(df[column_name].unique())
    if id== 1:
        word_to_int = {
        "tcp": 12,
        "udp": 100,
        "arp": 2,
        "ospf": 3,
        "icmp": 4,
        "igmp": 5,
        "rtp": 6,
        "ddp": 7,
        "ipv6-frag": 8,
        "cftp": 9,
        "wsn": 10,
        "pvp": 11,
        "wb-expak": 0,
        "mtp": 13,
        "pri-enc": 14,
        "sat-mon": 15,
        "cphb": 16,
        "sun-nd": 17,
        "iso-ip": 18,
        "xtp": 19,
        "il": 20,
        "unas": 21,
        "mfe-nsp": 22,
        "3pc": 23,
        "ipv6-route": 24,
        "idrp": 25,
        "bna": 26,
        "swipe": 27,
        "kryptolan": 28,
        "cpnx": 29,
        "rsvp": 30,
        "wb-mon": 31,
        "vmtp": 32,
        "ib": 33,
        "dgp": 34,
        "eigrp": 35,
        "ax.25": 36,
        "gmtp": 37,
        "pnni": 38,
        "sep": 39,
        "pgm": 40,
        "idpr-cmtp": 41,
        "zero": 42,
        "rvd": 43,
        "mobile": 44,
        "narp": 45,
        "fc": 46,
        "pipe": 47,
        "ipcomp": 48,
        "ipv6-no": 49,
        "sat-expak": 50,
        "ipv6-opts": 51,
        "snp": 52,
        "ipcv": 53,
        "br-sat-mon": 54,
        "ttp": 55,
        "tcf": 56,
        "nsfnet-igp": 57,
        "sprite-rpc": 58,
        "aes-sp3-d": 59,
        "sccopmce": 60,
        "sctp": 61,
        "qnx": 62,
        "scps": 63,
        "etherip": 64,
        "aris": 65,
        "pim": 66,
        "compaq-peer": 67,
        "vrrp": 68,
        "iatp": 69,
        "stp": 70,
        "l2tp": 71,
        "srp": 72,
        "sm": 73,
        "isis": 74,
        "smp": 75,
        "fire": 76,
        "ptp": 77,
        "crtp": 78,
        "sps": 79,
        "merit-inp": 80,
        "idpr": 81,
        "skip": 82,
        "any": 83,
        "larp": 84,
        "ipip": 85,
        "micp": 86,
        "encap": 87,
        "ifmp": 88,
        "tp++": 89,
        "a/n": 90,
        "ipv6": 91,
        "i-nlsp": 92,
        "ipx-n-ip": 93,
        "sdrp": 94,
        "tlsp": 95,
        "gre": 96,
        "mhrp": 97,
        "ddx": 98,
        "ippc": 99,
        "visa": 1,
        "secure-vmtp": 101,
        "uti": 102,
        "vines": 103,
        "crudp": 104,
        "iplt": 105,
        "ggp": 106,
        "ip": 107,
        "ipnip": 108,
        "st2": 109,
        "argus": 110,
        "bbn-rcc": 111,
        "egp": 112,
        "emcon": 113,
        "igp": 114,
        "nvp": 115,
        "pup": 116,
        "xnet": 117,
        "chaos": 118,
        "mux": 119,
        "dcn": 120,
        "hmp": 121,
        "prm": 122,
        "trunk-1": 123,
        "xns-idp": 124,
        "leaf-1": 125,
        "leaf-2": 126,
        "rdp": 127,
        "irtp": 128,
        "iso-tp4": 129,
        "netblt": 130,
        "trunk-2": 131,
        "cbt": 132,
    }
    # 将每个单词转换为对应的整数编号
    else:
        word_to_int={'FIN': 0, 'INT': 1, 'CON': 2, 'ECO': 3, 'REQ': 4, 'RST': 5, 'PAR': 6, 'URN': 7, 'no': 8,'CLO': 9, 'ACC':10}
    # df[column_name] = df[column_name].map(word_to_int)
    
    word_to_int = {word: idx for idx, word in enumerate(df[column_name].unique())}
    df.loc[:, column_name] = df[column_name].map(word_to_int)

    # print(df[column_name].unique())
    # 将整数编号转换为二进制，并去掉'0b'前缀
    binary_df = df[column_name].apply(lambda x: bin(x)[2:])

    # 找出最长的二进制串长度
    max_len = binary_df.apply(len).max()

    # 补齐二进制串，使其长度一致
    binary_df = binary_df.apply(lambda x: x.zfill(max_len))  # 用0填充前导零

    # 将二进制字符串拆分成每一位
    binary_columns = [f"bit_{id}_{i}" for i in range(max_len)]
    # binary_columns.append(f'bit_{max_len}')  # 添加最后一列
    binary_values = [list(x) for x in binary_df]  # 将每个二进制串拆分为单个字符
    # binary_values = [[int(x) for x in list(x)] for x in binary_df]  # 将每个字符 '0' 和 '1' 转换为整数

    # print(binary_values)
    # print(binary_columns)
    # 创建新的 DataFrame 来存储二进制列
    binary_df = pd.DataFrame(binary_values, columns=binary_columns)
    # print(binary_df)
    # print(binary_df)
    # 将原df与新的二进制列合并
    df = df.drop(columns=[column_name])
    df = pd.concat([df, binary_df], axis=1)

    return df


def process_data():
    # 读取数据
    test_df = pd.read_csv("../../data/UNSW_NB15_testing-set.csv")
    train_df = pd.read_csv("../../data/UNSW_NB15_training-set.csv")

    # 读取特征重要性文件并提取前20个特征
    # feature_importance_df = pd.read_csv("../../data/2.csv")
    # feature_importance_df = pd.read_csv("../../data/gwo_feature_importance.csv")
    feature_importance_df = pd.read_csv("../../data/xgb_feature_importance4.csv")
    top_20_features = feature_importance_df["Feature"].head(20).tolist()
    print(top_20_features) 
    # 确保测试集和训练集包含相同的特征
    X_train = train_df[top_20_features]
    X_test = test_df[top_20_features]

    # X_train = convert_column_to_binary(X_train, "proto",1)
    # X_test = convert_column_to_binary(X_test, "proto",1)

    # X_train = convert_column_to_binary(X_train, "state",2)
    # X_test = convert_column_to_binary(X_test, "state",2)
#     # 检查数据类型
    # print(X_train.columns)
    # print(X_train.dtypes)
#     print(X_test.columns)
#     bit_columns = [
#     'bit_1_0', 'bit_1_1', 'bit_1_2', 'bit_1_3', 'bit_1_4', 'bit_1_5', 'bit_1_6', 'bit_1_7',
#     'bit_2_0', 'bit_2_1', 'bit_2_2', 'bit_2_3'
# ]

# # 将这些列转换为数值类型，如果无法转换为数字的值会变为 NaN
#     X_train[bit_columns] = X_train[bit_columns].apply(pd.to_numeric, errors='coerce')
#     bit_columns = [
#     'bit_1_0', 'bit_1_1', 'bit_1_2', 'bit_1_3', 'bit_1_4', 'bit_1_5', 'bit_1_6', 'bit_1_7',
#     'bit_2_0', 'bit_2_1', 'bit_2_2'
# ]
#     X_test[bit_columns] = X_test[bit_columns].apply(pd.to_numeric, errors='coerce')
    # print(X_train.dtypes)
    # X_train = pd.get_dummies(X_train, columns=["proto", "state"])
    # X_test = pd.get_dummies(X_test, columns=["proto", "state"])
    X_train = pd.get_dummies(X_train, columns=["proto"])
    X_test = pd.get_dummies(X_test, columns=["proto"])   
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # X_train.to_csv('X_train.csv',index=False)
    X_train = X_train + 1e-10  # 避免零值
    X_test = X_test + 1e-10
    # # return
    X_train = np.log10(X_train)
    X_test = np.log10(X_test)   

    # 对所有特征进行均值化
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # y_train_bin = train_df["label"]
    # y_test_bin = test_df["label"]

    # y_train_muti = train_df["attack_cat"]
    # y_test_muti = test_df["attack_cat"]


    encoder = LabelEncoder()

    y_train_cat = encoder.fit_transform(train_df['attack_cat'])
    y_test_cat = encoder.transform(test_df['attack_cat'])
    # encoder = LabelEncoder()
    class_names = encoder.classes_
    print(class_names)
    # y_train_multi = encoder.fit_transform(train_df['attack_cat'])
    # y_test_multi = encoder.transform(test_df['attack_cat'])

    # return X_train, X_test, y_train_multi, y_test_multi
    # return X_train, X_test, y_train_bin, y_test_bin
    # return X_train, X_test, y_train_muti, y_test_muti
    return X_train, X_test, y_train_cat, y_test_cat


def process_orgindata():
    # 读取数据
    test_df = pd.read_csv("../../data/UNSW_NB15_testing-set.csv")
    train_df = pd.read_csv("../../data/UNSW_NB15_training-set.csv")

    # 读取特征重要性文件并提取前20个特征
    # feature_importance_df = pd.read_csv("../../data/gwo_feature_importance.csv")
    # feature_importance_df = pd.read_csv("../../data/xgb_gwo_feature.csv")
    # feature_importance_df = pd.read_csv("../../data/xgb_feature_importance.csv")
    # top_20_features = feature_importance_df["Feature"].head(20).tolist()

    # # 确保测试集和训练集包含相同的特征
    # X_train = train_df[top_20_features]
    # X_test = test_df[top_20_features]

    X_train = train_df.drop(columns=['label', 'id', 'service', 'attack_cat'])  # 特征
    X_test = test_df.drop(columns=['label', 'id', 'service', 'attack_cat'])  # 特征
    print(X_test.shape[1])
    # X_train = train_df.drop(columns=['label', 'id', 'service', 'attack_cat','state'])  # 特征
    # X_test = test_df.drop(columns=['label', 'id', 'service', 'attack_cat','state'])  # 
    # 特征 
    # X_train = convert_column_to_binary(X_train, "proto", 1)
    # X_test = convert_column_to_binary(X_test, "proto", 1)

    # X_train = convert_column_to_binary(X_train, "state", 2)
    # X_test = convert_column_to_binary(X_test, "state", 2)

    X_train = pd.get_dummies(X_train, columns=["proto", "state"])
    X_test = pd.get_dummies(X_test, columns=["proto", "state"])

    # 对齐特征，确保训练集和测试集的特征一致
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)


    X_train = X_train + 1e-10  # 避免零值
    X_test = X_test + 1e-10
    # return
    X_train = np.log10(X_train)
    X_test = np.log10(X_test)   
    # 对所有特征进行均值化
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    encoder = LabelEncoder()

    y_train_cat = encoder.fit_transform(train_df['attack_cat'])
    y_test_cat = encoder.transform(test_df['attack_cat'])
    # 二分类任务：预测label
    # y_train_bin = train_df["label"].values
    # y_test_bin = test_df["label"].values

    # return X_train, X_test, y_train_bin, y_test_bin
    return X_train, X_test, y_train_cat, y_test_cat


@timer
def train_model(X_train, y_train_bin, param_grid):
    svm_model = SVC(C=param_grid['C'][0], gamma=param_grid['gamma'][0], kernel=param_grid['kernel'][0])
    svm_model.fit(X_train, y_train_bin)

    # 保存模型
    with open('../../model/42-xgb-thusvm-model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    return svm_model


# @timer


def predict(X_test, y_test_bin, model, save_path):
    # 使用模型进行预测
    y_pred = model.predict(X_test)
    
    # 保存预测结果到 CSV 文件
    df = pd.DataFrame({'True Labels': y_test_bin, 'Predicted Labels': y_pred})
    df.to_csv(save_path, index=False)
    # print(df)
    # print("Saving predictions to:", save_path)

    # 计算准确率
    accuracy = accuracy_score(y_test_bin, y_pred)
    print(f"SVM Model Accuracy: {accuracy:.4f}")
    
    # 返回预测结果
    return y_test_bin, y_pred


def load_predictions(file_path='predictions.csv'):
    # 从文件加载预测结果
    df = pd.read_csv(file_path)
    y_test_bin = df['True Labels'].values
    y_pred = df['Predicted Labels'].values
    return y_test_bin, y_pred



def predict1(X_test, y_test_bin, model):
    # 使用模型进行预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test_bin, y_pred)
    print(f"SVM Model Accuracy: {accuracy:.4f}")
    
    # 保存预测结果
    return y_test_bin, y_pred


def evaluate(y_test_bin, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_test_bin, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # 计算假阳性率 (FPR) 和假阴性率 (FNR)
    tn, fp, fn, tp = cm.ravel()  # 将混淆矩阵分解为四个部分
    
    # 假阳性率 (FPR)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    prc=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2*prc*recall/(prc+recall)


    fpr = fp / (fp + tn)
    
    # 假阴性率 (FNR)
    fnr = fn / (fn + tp)
    
    # 误报率 (FAR)
    far = fp / (fp + tp)

    # 打印计算结果
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {prc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"False Alarm Rate (FAR): {far:.4f}")
    
    # 计算 ROC 曲线
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test_bin, y_pred)
    roc_auc = auc(fpr_roc, tpr_roc)
    
    # 绘制 ROC 曲线
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_roc, tpr_roc, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    # 可选：可视化混淆矩阵
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test_bin), yticklabels=np.unique(y_test_bin))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    except ImportError:
        print("Seaborn or matplotlib not available for heatmap visualization.")

# print("start")
# X_train, X_test, y_train_bin, y_test_bin = process_orgindata()

# print("process_data done")

# param_grid = {
#     'C': [0.1, 1, 10],  # 正则化参数
#     'gamma': [0.01, 0.1, 1],  # 核函数参数
#     'kernel': ['rbf', 'linear']  # 核函数类型
# }
# X_train, X_test, y_train_bin, y_test_bin = process_orgindata()
X_train, X_test, y_train_bin, y_test_bin = process_data()
param_grid = {
    "C": [10],  # 正则化参数
    "gamma": [0.1],  # 核函数参数
    "kernel": ["rbf"],  # 核函数类型
}


# model = train_model(X_train, y_train_bin, param_grid)
# with open("../../model/xgb-thusvm-model.pkl", 'rb') as f:
#     model = pickle.load(f)
# 在首次运行时预测并保存结果
# y_test_bin, y_pred = predict(X_test, y_test_bin, model, 'm-42-20predictions.csv')

# 如果以后只想加载保存的预测结果
# y_test_bin, y_pred = load_predictions('m-20predictions.csv')
y_test_bin, y_pred = load_predictions('predictions.csv')
evaluate(y_test_bin, y_pred)

# return 1
# qew(X_train, y_train_bin)
