import pandas as pd

# 读取CSV文件
test_csv = '../data/UNSW_NB15_testing-set.csv'
train_csv = '../data/train-set-normalized.csv'

# 加载数据
test_df = pd.read_csv(test_csv)
train_df = pd.read_csv(train_csv)

# 提取测试集中的最后两列
last_two_columns = test_df.iloc[:, -2:]

# 检查train_df的行数是否与last_two_columns的行数匹配
if train_df.shape[0] == last_two_columns.shape[0]:
    # 拼接列到训练集
    train_df = pd.concat([train_df, last_two_columns.reset_index(drop=True)], axis=1)
else:
    print("行数不匹配，请检查两个数据集!")

# 保存新的训练集
train_df.to_csv('../data/train-set-processed.csv', index=False)

# 查看新的训练集前几行确认是否成功
print(train_df.head())
