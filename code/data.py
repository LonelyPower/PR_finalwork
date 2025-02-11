import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("../data/UNSW_NB15_training-set.csv")

X_train = train_df.drop(columns=['label', 'id', 'service', 'attack_cat','proto','state'])  # 特征
max_values = np.max(X_train, axis=0)
min_values = np.min(X_train, axis=0)

# 打印结果
for i in range(X_train.shape[1]):
    print(f"Feature {i+1}: Min = {min_values[i]}, Max = {max_values[i]}")
X_train_np = X_train.to_numpy()
plt.figure(figsize=(10, 6))
for i in range(X_train_np.shape[1]):  # 遍历每个特征
    plt.hist(X_train_np[:, i], bins=50, alpha=0.5, label=f'Feature {i+1}')

plt.title("Distribution of Features in X_train")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()
# 绘制热力图
# plt.figure(figsize=(10, 6))
# sns.heatmap(X_train, cmap='viridis', cbar=True)
# plt.title("Heatmap of X_train")
# plt.xlabel("Feature Index")
# plt.ylabel("Sample Index")
# plt.show()
# print(X_train.m)
# plt.plot(X_train, bins=50)
# plt.title("Data Distribution")
# plt.show()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
# X_train_scaled=scaler.transform(X_train)
# print(X_train)
# print(X_train_scaled)

# 进行log变换
offset = 1e-10  # 偏移量
X_train_log = np.log10(X_train + offset)

# 计算每组数据的偏度
original_skew = skew(X_train, axis=0)
scaled_skew = skew(X_train_scaled, axis=0)
log_skew = skew(X_train_log, axis=0)

print("Skewness of Original Data:", original_skew)
print("Skewness of Standardized Data:", scaled_skew)
print("Skewness of Log Transformed Data:", log_skew)


# 绘制各组数据偏度的直方图
plt.plot(original_skew,  alpha=0.5, label='Original Data', color='blue')
# plt.plot(scaled_skew,  alpha=0.5, label='Standardized Data', color='green')
plt.plot(log_skew, alpha=0.5, label='Log Transformed Data', color='red')

# 添加标题和标签
plt.title("Comparison of Skewness Distributions")
plt.xlabel("Skewness")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('skewness.png')
# 显示图形
plt.show()


# # 打印每组数据的偏度
# print("Skewness of Original Data:", original_skew)
# print("Skewness of Standardized Data:", scaled_skew)
# print("Skewness of Log Transformed Data:", log_skew)