import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('../data/UNSW_NB15_training-set.csv')

# 提取 label 列为 1 的行
df_label_1 = df[df['label'] == 1]

# 打印结果
print(df_label_1)

# 可选择将结果保存为新的 CSV 文件
df_label_1.to_csv('../data/attack.csv', index=False)
