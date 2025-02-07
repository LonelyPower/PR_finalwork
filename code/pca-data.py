from sklearn.preprocessing import StandardScaler

# 假设你要对所有特征进行 PCA
# features = data_encoded.drop(columns=['proto'])  # 假设 'target' 是标签列T

# 标准化特征数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(new_data)

# 查看标准化后的数据
print(features_scaled[:5])

from sklearn.decomposition import PCA

# 设置降维后的维度数
n_components = 20  # 保留 2 个主成分

# 初始化 PCA
pca = PCA(n_components=n_components)

# 对数据进行降维
features_pca = pca.fit_transform(features_scaled)

# 查看降维后的数据
# print(features_pca[:5])
# 查看每个主成分的方差贡献率
# print(pca.explained_variance_ratio_)

# 查看累计方差贡献率
print(pca.explained_variance_ratio_.cumsum())
import matplotlib.pyplot as plt

# 可视化累计方差贡献率
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()