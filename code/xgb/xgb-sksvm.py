from sklearn.svm import SVC as skSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import pickle

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper


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
    binary_df = binary_df.apply(lambda x: x.zfill(max_len))  # 用0填充前导零
    
    # 将二进制字符串拆分成每一位
    binary_columns = [f'bit_{i}' for i in range(max_len)]
    # binary_columns.append(f'bit_{max_len}')  # 添加最后一列
    binary_values = [list(x) for x in binary_df]  # 将每个二进制串拆分为单个字符
    
    # print(binary_values)
    # print(binary_columns)
    # 创建新的 DataFrame 来存储二进制列
    binary_df = pd.DataFrame(binary_values, columns=binary_columns)
    # print(binary_df)
    # 将原df与新的二进制列合并
    df=df.drop(columns=[column_name])
    df = pd.concat([df, binary_df], axis=1)
    
    return df


def process_data():
    # 读取数据
    test_df = pd.read_csv('../../data/UNSW_NB15_testing-set.csv')
    train_df = pd.read_csv('../../data/UNSW_NB15_training-set.csv')

    # 读取特征重要性文件并提取前20个特征
    feature_importance_df = pd.read_csv('../../data/xgb_feature_importance.csv')
    top_20_features = feature_importance_df['Feature'].head(20).tolist()

    # 确保测试集和训练集包含相同的特征
    X_train = train_df[top_20_features]
    X_test = test_df[top_20_features]
    

    # X_train=convert_column_to_binary(X_train,'proto')
    # X_test=convert_column_to_binary(X_test,'proto')

    X_train = pd.get_dummies(X_train, columns=["proto"])
    X_test = pd.get_dummies(X_test, columns=["proto"])
    # X_train.to_csv('X_train.csv',index=False)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    # return

    # 对所有特征进行均值化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 二分类任务：预测label
    y_train_bin = train_df['label']
    y_test_bin = test_df['label']

    return X_train, X_test, y_train_bin, y_test_bin


@timer
def train_model(X_train, y_train_bin, param_grid):
    svm_model = skSVC(C=param_grid['C'][0], gamma=param_grid['gamma'][0], kernel=param_grid['kernel'][0])
    svm_model.fit(X_train, y_train_bin)
    return svm_model


def train_best_model(X_train, y_train_bin): 
    # 初始化 SVM 分类器
    svm_model = skSVC()
    
    # 网格搜索超参数
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
    
    # 训练模型
    grid_search.fit(X_train, y_train_bin)
    
    # 输出最佳参数
    print("Best parameters found:", grid_search.best_params_)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 保存最佳模型
    with open("../../model/best_sksvm_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    
    # 打印最佳模型的超参数
    print(best_model.get_params())


@timer
def predict(X_test, y_test_bin, model):
    # 读取模型
    # with open("../../model/xgb-thusvm-model.pkl", 'rb') as f:
    #     svm_model = pickle.load(f)
    # print(svm_model.get_params())
    # 预测
    # svm_model=model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_bin, y_pred)
    print(f"SVM Model Accuracy: {accuracy:.4f}")

X_train, X_test, y_train_bin, y_test_bin = process_data()

# train_model(X_train, y_train_bin)
# predict( X_test, y_test_bin)


di=[
    {"C": [0.1], "gamma": [0.1], "kernel": ["rbf"]},
    # {"C": [0.1], "gamma": [1], "kernel": ["rbf"]},
    # {"C": [1], "gamma": [0.01], "kernel": ["rbf"]},
    {"C": [1], "gamma": [0.1], "kernel": ["rbf"]},
    # {"C": [1], "gamma": [1], "kernel": ["rbf"]},
    {"C": [10], "gamma": [0.1], "kernel": ["rbf"]}
    # {"C": [10], "gamma": [1], "kernel": ["rbf"]}
]


for i in di:
    print(i)
    model = train_model(X_train, y_train_bin, i)

    predict(X_test, y_test_bin, model)


