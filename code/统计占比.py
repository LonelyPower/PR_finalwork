import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
train_df = pd.read_csv('../data/UNSW_NB15_training-set.csv')
test_df = pd.read_csv('../data/UNSW_NB15_testing-set.csv')

# 计算各攻击类别的数量
atla1 = train_df['attack_cat'].value_counts()
atla2 = test_df['attack_cat'].value_counts()
atla1_df = atla1.reset_index()
atla1_df.columns = ['Attack Category', 'Count']

atla2_df = atla2.reset_index()
atla2_df.columns = ['Attack Category', 'Count']

# 创建一个Excel文件并写入两个工作表
with pd.ExcelWriter('attack_categories_count.xlsx') as writer:
    atla1_df.to_excel(writer, sheet_name='Train Set', index=False)
    atla2_df.to_excel(writer, sheet_name='Test Set', index=False)
# 创建一个图形对象
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# 画训练集的饼图
ax[0].pie(atla1, labels=atla1.index, autopct='%1.1f%%', startangle=90)
ax[0].set_title('Train Set - Attack Categories')

# 画测试集的饼图
ax[1].pie(atla2, labels=atla2.index, autopct='%1.1f%%', startangle=90)
ax[1].set_title('Test Set - Attack Categories')

# 显示图形
plt.tight_layout()
plt.show()
