from ctypes import CDLL
import os
dll_path = r"D:\\python\\Lib\\site-packages\\thundersvm\\thundersvm.dll"  # 替换为实际的 DLL 文件路
tem = CDLL(dll_path, winmode=0)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from thundersvm import SVC  # 导入 ThunderSVM

# 加载数据
# train_df = pd.read_csv('../data/train-set-processed.csv')
# test_df = pd.read_csv('../data/test-set-processed.csv')

train_df = pd.read_csv('../data/train-set-nostring.csv')
test_df = pd.read_csv('../data/test-set-nostring.csv')

# 确保测试集和训练集包含相同的特征
X_train = train_df.drop(columns=['label', 'attack_cat'])
X_test = test_df.drop(columns=['label', 'attack_cat'])

# 二分类任务：预测label
y_train_bin = train_df['label']
y_test_bin = test_df['label']

# 多分类任务：预测attack_cat
# 需要先编码attack_cat列
# encoder = LabelEncoder()
# y_train_multi = encoder.fit_transform(train_df['attack_cat'])
# y_test_multi = encoder.transform(test_df['attack_cat'])

print("训练二分类SVM")
svm_binary = SVC(kernel='linear', gpu_id=0)  # 使用 ThunderSVM 的 SVC
svm_binary.fit(X_train, y_train_bin)
y_pred_bin = svm_binary.predict(X_test)

# 训练多分类SVM
# svm_multi = SVC(kernel='linear', gpu_id=0)  # 使用 ThunderSVM 的 SVC
# svm_multi.fit(X_train, y_train_multi)
# y_pred_multi = svm_multi.predict(X_test)

# 二分类的准确率
accuracy_binary = accuracy_score(y_test_bin, y_pred_bin)
print(f"二分类准确率: {accuracy_binary:.2f}")

# 多分类的准确率
# accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)
# print(f"多分类准确率: {accuracy_multi:.2f}")
