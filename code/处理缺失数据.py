import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


csv_file = '../data/train-set-without-lable.csv'
csv_file1 = '../data/train-set-without-lable.csv'

# 读取CSV文件
df = pd.read_csv(csv_file)
# 假设我们知道哪些列需要将'-'视为缺失值，我们可以指定这些列
columns_to_replace = ['service',]  # 填写需要替换 `-` 的列名列表

# 将指定列中的'-'替换为NaN
df[columns_to_replace] = df[columns_to_replace].replace('-', np.nan)

# 计算每列的缺失值数量和完整度
missing_data = df.isnull().sum()  # 每列缺失值的数量
total_data = df.shape[0]  # 总行数
complete_data = (df.notnull().sum())  # 每列非缺失值的数量
data_completeness = (complete_data / total_data) * 100  # 计算数据完整度

# 输出缺失值统计信息
print("缺失值统计：\n", missing_data)
print("\n数据完整度：\n", data_completeness)

# 可视化数据完整度
plt.figure(figsize=(10, 6))

# 根据是否缺失来设置颜色
colors = ['red' if missing_data[col] > 0 else 'skyblue' for col in df.columns]

# 创建柱状图，设置宽度为1.0，柱子之间没有间隔
data_completeness.plot(kind='bar', color=colors, width=1)

plt.title('数据完整度可视化')
plt.xlabel('列名')
plt.ylabel('完整度百分比')
plt.xticks(rotation=45)
plt.tight_layout()
plt.xticks(np.arange(len(df.columns)), df.columns, rotation=45, ha='right', fontsize=10)
plt.savefig('../output/数据完整度可视化.png')
plt.show()

df=df.drop('service',axis=1)
df.to_csv('../data/train-set-without-lable-service.csv', index=False)