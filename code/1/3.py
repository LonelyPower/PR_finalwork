# import pandas as pd
# from scipy.stats import chi2_contingency

# # 假设 data 是包含协议和攻击类型的 DataFrame
# data = pd.read_csv('../data/UNSW_NB15_training-set.csv')

# # 计算协议和攻击类型的交叉表
# contingency_table = pd.crosstab(data['proto'], data['label'])
# print(contingency_table)
# # 进行卡方检验
# contingency_table.to_csv('contingency_table.csv', index=False)
# chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# # 输出结果
# print(f"Chi-squared Statistic: {chi2}")
# print(f"P-value: {p_value}")
# print(f"Degrees of freedom: {dof}")
# print(f"Expected frequencies: \n{expected}")

# # 判断是否存在显著关联
# if p_value < 0.05:
#     print("There is a significant association between protocol and attack type.")
# else:
#     print("There is no significant association between protocol and attack type.")

import pandas as pd
data = pd.read_csv('../data/UNSW_NB15_training-set.csv')
# 假设你有一个名为 data 的 DataFrame，其中包含 'proto' 和 'label' 列
contingency_table = pd.crosstab(data['proto'], data['label'])

# 打印交叉表，查看每个 proto 和 label 的频次
print(contingency_table)
