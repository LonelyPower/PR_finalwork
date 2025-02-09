import category_encoders as ce
import pandas as pd

csv_file = '../data/train-set-without-lable-service.csv'

# 读取CSV文件
import pandas as pd

# 假设数据

df = pd.read_csv(csv_file)

# 使用pandas的get_dummies函数进行独热编码
df_encoded = pd.get_dummies(df, columns=['proto', 'state'])

print(df_encoded)

df_encoded.to_csv('../data/train-set-onehot.csv', index=False)

