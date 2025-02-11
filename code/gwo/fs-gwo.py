import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
from tqdm import tqdm # 导入 tqdm



def convert_column_to_binary(df, column_name):
  # 为每个唯一单词分配一个从0开始的整数编号
  word_to_int = {word: idx for idx, word in enumerate(df[column_name].unique())}
  
  # 将每个单词转换为对应的整数编号
  df[column_name] = df[column_name].map(word_to_int)
  
  # 将整数编号转换为二进制，并去掉'0b'前缀
  binary_df = df[column_name].apply(lambda x: bin(x)[2:])
  
  # 找出最长的二进制串长度
  max_len = binary_df.apply(len).max()
  
  # 补齐二进制串，使其长度一致
  binary_df = binary_df.apply(lambda x: x.zfill(max_len)) # 用0填充前导零
  
  # 将二进制字符串拆分成每一位
  binary_columns = [f'bit_{i}' for i in range(max_len)]
  # binary_columns.append(f'bit_{max_len}') # 添加最后一列
  binary_values = [list(x) for x in binary_df] # 将每个二进制串拆分为单个字符
  df=df.drop(columns=[column_name])
  df = pd.concat([df, binary_df], axis=1)
  
  return df


def process_data():
# 加载数据
    train_df = pd.read_csv('../../data/UNSW_NB15_training-set.csv')

    # 假设目标列是 'label'，特征是其他列
    X = train_df.drop(columns=['label','id','service','attack_cat']) # 特征
    y = train_df['label'] # 目标变量

    # 如果有分类特征，需要进行编码
    # label_encoder = LabelEncoder()
    # y = label_encoder.fit_transform(y)

    X=convert_column_to_binary(X,'proto')
    X=convert_column_to_binary(X,'state')
    print(X)

    # 标准化特征（可选，但推荐）
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # print(X)
  # X.to_csv('X.csv',index=False)
    return X, y

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def fitness_function(feature_subset, X, y):
  """
  计算特征子集的适应度值（分类准确率）
  :param feature_subset: 二进制向量，表示选择的特征（1选择，0不选择）
  :param X: 原始特征矩阵
  :param y: 目标变量
  :return: 适应度值（分类准确率）
  """
  # 选择特征
  selected_features = X[:, feature_subset == 1]
  
  # 划分训练集和测试集
  X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=42)

  # 使用随机森林分类器进行训练
  classifier = RandomForestClassifier(random_state=42,n_jobs=-1)
  classifier.fit(X_train, y_train)

  # 预测并计算准确率
  y_pred = classifier.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)

  return accuracy


def fitness_function1(feature_subset, X, y):
  """
  计算特征子集的适应度值（分类准确率）
  :param feature_subset: 二进制向量，表示选择的特征（1选择，0不选择）
  :param X: 原始特征矩阵
  :param y: 目标变量
  :return: 适应度值（分类准确率）
  """
  # 选择特征
  selected_features = X[:, feature_subset == 1]
  print('a')
  # 使用随机森林分类器进行评估
  classifier = RandomForestClassifier(random_state=42)
  print('b')
  # 使用交叉验证计算准确率
  scores = cross_val_score(classifier, selected_features, y, cv=3, scoring='accuracy', n_jobs=-1)

  print('c')
  # 返回平均准确率
  return scores.mean()



def gwo_feature_selection(X, y, num_wolves=10, max_iter=100):
  """
  使用GWO算法进行特征选择
  :param X: 特征矩阵
  :param y: 目标变量
  :param num_wolves: 灰狼数量
  :param max_iter: 最大迭代次数
  :return: 最优特征子集
  """
  num_features = X.shape[1]
  print(num_features)
  # 初始化灰狼群体（二进制向量）
  wolves = np.random.randint(2, size=(num_wolves, num_features))
  
  # 初始化Alpha、Beta、Delta
  alpha = np.zeros(num_features)
  beta = np.zeros(num_features)
  delta = np.zeros(num_features)
  
  alpha_score = -np.inf
  beta_score = -np.inf
  delta_score = -np.inf
  
  # 迭代优化
  # print("GWO is optimizing...")
  for iteration in tqdm(range(max_iter), desc="GWO Progress"): # 添加进度条
    for i in range(num_wolves):
      # 计算适应度值
      fitness = fitness_function(wolves[i], X, y)
      print('fitness')
      # 更新Alpha、Beta、Delta
      if fitness > alpha_score:
        alpha_score = fitness
        alpha = wolves[i].copy()
      elif fitness > beta_score:
        beta_score = fitness
        beta = wolves[i].copy()
      elif fitness > delta_score:
        delta_score = fitness
        delta = wolves[i].copy()
    
    # 更新灰狼位置
    a = 2 - iteration * (2 / max_iter) # 线性递减参数
    for i in range(num_wolves):
      for j in range(num_features):
        r1 = np.random.random()
        r2 = np.random.random()
        
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        
        D_alpha = abs(C1 * alpha[j] - wolves[i][j])
        X1 = alpha[j] - A1 * D_alpha
        
        r1 = np.random.random()
        r2 = np.random.random()
        
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        
        D_beta = abs(C2 * beta[j] - wolves[i][j])
        X2 = beta[j] - A2 * D_beta
        
        r1 = np.random.random()
        r2 = np.random.random()
        
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        
        D_delta = abs(C3 * delta[j] - wolves[i][j])
        X3 = delta[j] - A3 * D_delta
        
        # 更新位置
        wolves[i][j] = np.round((X1 + X2 + X3) / 3)
    
    print(f"Iteration {iteration + 1}, Best Fitness: {alpha_score}")
  
  return alpha


# X, y = process_data()
# # 运行GWO进行特征选择
# best_feature_subset = gwo_feature_selection(X, y, num_wolves=10, max_iter=50)

# # 输出最优特征子集
# print("Best Feature Subset:", best_feature_subset)
# selected_features = X[:, best_feature_subset == 1]
# print("Selected Features Shape:", selected_features.shape)
train_df = pd.read_csv('../../data/UNSW_NB15_training-set.csv').drop(columns=['label','id','service','attack_cat','proto','state'])
col=train_df.columns.values
print(col.shape)
best=[0, 3 ,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,1,0,1,0,0,1,1,1,0,1,0,1,0,0,1,0,1,1,11,0]
# print(best[)
for i in range(len(best)-2):
  if best[i]!=0:
    print(col[i])