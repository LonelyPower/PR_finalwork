import pandas as pd

csv_file = '../data/UNSW_NB15_testing-set.csv'
csv_file2 = '../data/UNSW_NB15_training-set.csv'

# 读取CSV文件
df = pd.read_csv(csv_file)
df = df.iloc[:, 1:]
# 删除最后两列
df = df.iloc[:, :-2]

# 将修改后的数据另存为新的CSV文件
df.to_csv('../data/train-set-without-lable.csv', index=False)
# df.to_csv('../data/train-set-without-lable.csv', index=False)
