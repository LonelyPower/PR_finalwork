import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
csv_file = '../data/train-set-onehot.csv'
df = pd.read_csv(csv_file)

# 创建MinMaxScaler对象
scaler = MinMaxScaler()

# 选取数值型列
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# 只对数值型列进行标准化
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 检查标准化后的数据
print(df.head())

# 将修改后的数据另存为新的CSV文件
df.to_csv('../data/train-set-normalized.csv', index=False)
