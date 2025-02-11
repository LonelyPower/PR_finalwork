import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 定义多分类 SVM 模型（用Softmax激活）
class MultiClassSVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiClassSVM, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)  # 输出层调整为 num_classes

    def forward(self, x):
        return self.linear(x)
    

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


# 处理数据
def process_data():
    # 读取数据
    train_df = pd.read_csv("../data/UNSW_NB15_testing-set.csv")
    test_df = pd.read_csv("../data/UNSW_NB15_training-set.csv")

    # 读取特征重要性文件并提取前20个特征
    feature_importance_df = pd.read_csv("../data/xgb_feature_importance.csv")
    top_20_features = feature_importance_df["Feature"].head(20).tolist()

    # 确保测试集和训练集包含相同的特征
    X_train = train_df[top_20_features]
    X_test = test_df[top_20_features]

    # 对特征列进行处理，例如proto列的二进制化（根据之前的代码）
    X_train = convert_column_to_binary(X_train, "proto")
    X_test = convert_column_to_binary(X_test, "proto")

    # 对所有特征进行均值化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 多分类任务：预测 attack_cat
    # attack_cat_mapping = {
    #     "Normal": 0,
    #     "Fuzzers": 1,
    #     "Analysis": 2,
    #     "Backdoor": 3,
    #     "DoS": 4,
    #     "Generic": 5,
    #     "Reconnaissance": 6,
    #     "Shellcode": 7,
    #     "Worms": 8,
    #     "Exploits": 9
    # }
    # print("训练集中的唯一attack_cat类别:", train_df["attack_cat"].unique())
    # print("测试集中的唯一attack_cat类别:", test_df["attack_cat"].unique())

    # 使用映射将类别转换为数字，并检查是否有NaN值
    # y_train_cat = train_df["attack_cat"].map(attack_cat_mapping)
    # y_test_cat = test_df["attack_cat"].map(attack_cat_mapping)
    encoder = LabelEncoder()
    y_train_cat = encoder.fit_transform(train_df['attack_cat'])
    y_test_cat = encoder.transform(test_df['attack_cat'])
    # 使用映射将类别转换为数字
    # y_train_cat = train_df["attack_cat"].map(attack_cat_mapping).values
    # y_test_cat = test_df["attack_cat"].map(attack_cat_mapping).values

    # 转换为 Tensor，并确保标签是整型的
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    # print(y_train_cat)
    y_train_cat = torch.tensor(y_train_cat, dtype=torch.long)  # 标签类型需要是long
    y_test_cat = torch.tensor(y_test_cat, dtype=torch.long)

    return X_train, X_test, y_train_cat, y_test_cat

# 训练模型时，指定类别数
def train_model(X_train, y_train_cat, num_classes, lr=0.01, num_epochs=600):
    # 初始化模型
    input_dim = X_train.shape[1]
    model = MultiClassSVM(input_dim, num_classes)

    # 使用SGD优化器
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # 设置损失函数为 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(num_epochs):
        model.train()
        
        # 前向传播
        outputs = model(X_train)

        # 计算损失
        loss = criterion(outputs, y_train_cat)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    return model

# 预测函数
def predict(X_test, y_test_cat, model):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, 1)  # 预测类别

    accuracy = accuracy_score(y_test_cat.cpu().numpy(), predictions.cpu().numpy())
    print(f"Multi-Class SVM Model Accuracy: {accuracy:.4f}")

# 主程序
if __name__ == "__main__":
    X_train, X_test, y_train_cat, y_test_cat = process_data()

    # 获取类别数
    num_classes = len(np.unique(y_train_cat))  # attack_cat 列的类别数量

    # 训练模型
    model = train_model(X_train, y_train_cat, num_classes)

    # 预测并计算准确度
    predict(X_test, y_test_cat, model)
