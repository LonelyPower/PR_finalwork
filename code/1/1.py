from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


# 读取CSV文件
df = pd.read_csv('../../data/attack.csv')
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
print(numeric_columns)
# 查看前几行数据，确认数据是否正确加载
# print(data['proto'])
# 打印 'proto' 列中每个类别的出现频次
# print(data['proto'].value_counts())
# 获取 'proto' 列中不同类别的数量
# print(data['proto'].nunique())

# 假设 data 是你的数据集，X 是特征，y 是标签
# X = data.drop('label', axis=1)
# y = data['label']
# print('X.head()')
# # 拆分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # 训练随机森林分类器
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # 获取特征重要性
# feature_importances = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)

# # 查看最重要的特征
# print(feature_importances.head())
