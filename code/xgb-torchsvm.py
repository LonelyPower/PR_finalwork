import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 定义SVM的Hinge损失
class SVM(nn.Module):
    def __init__(self, input_dim, C=1.0):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 线性层，输出维度为1
        self.C = C  # 设置C超参数

    def forward(self, x):
        return self.linear(x)

    def hinge_loss(self, outputs, labels):
        # 计算hinge损失，加入C参数
        loss = torch.mean(torch.clamp(1 - labels * outputs, min=0))  # 误分类损失
        regularization = torch.sum(self.linear.weight ** 2)  # L2 正则化项
        return 0.5 * regularization + self.C * loss  # 综合损失
    

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
    train_df = pd.read_csv("../data/UNSW_NB15_testing-set.csv")
    test_df = pd.read_csv("../data/UNSW_NB15_training-set.csv")

    # 读取特征重要性文件并提取前20个特征
    feature_importance_df = pd.read_csv("../data/xgb_feature_importance.csv")
    top_20_features = feature_importance_df["Feature"].head(20).tolist()

    # 确保测试集和训练集包含相同的特征
    X_train = train_df[top_20_features]
    X_test = test_df[top_20_features]

    X_train = convert_column_to_binary(X_train, "proto")
    X_test = convert_column_to_binary(X_test, "proto")

    # 对所有特征进行均值化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 二分类任务：预测label
    y_train_bin = train_df["label"].values
    y_test_bin = test_df["label"].values

    # 转换标签为 -1 和 1，因为 SVM 需要这样
    y_train_bin = np.where(y_train_bin == 1, 1, -1)
    y_test_bin = np.where(y_test_bin == 1, 1, -1)

    # 转换为 Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train_bin = torch.tensor(y_train_bin, dtype=torch.float32).view(-1, 1)
    y_test_bin = torch.tensor(y_test_bin, dtype=torch.float32).view(-1, 1)

    return X_train, X_test, y_train_bin, y_test_bin

def train_model(X_train, y_train_bin, C=1.0):
    # 初始化模型
    input_dim = X_train.shape[1]
    model = SVM(input_dim, C)

    # 使用SGD优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 设置损失函数为 Hinge 损失
    criterion = model.hinge_loss

    # 训练
    num_epochs = 600
    for epoch in range(num_epochs):
        model.train()
        
        # 前向传播
        outputs = model(X_train)

        # 计算损失
        loss = criterion(outputs, y_train_bin)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if epoch % 10 == 0:
            # print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    return model


def predict(X_test, y_test_bin, model):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.sign(outputs).cpu().numpy().flatten()

    accuracy = accuracy_score(y_test_bin.cpu().numpy(), predictions)

    print(f"SVM Model Accuracy: {accuracy:.4f}")
    return accuracy
# 主程序
if __name__ == "__main__":
    X_train, X_test, y_train_bin, y_test_bin = process_data()

    C_value = 10  # 设置C值，通常需要调优
    acc=0
    for i in range(6):
        model = train_model(X_train, y_train_bin, C=C_value)
        acc=acc+predict(X_test, y_test_bin, model)
        # C_value *= 10
    print(acc/6)
    # model = train_model(X_train, y_train_bin, C=C_value)
    # predict(X_test, y_test_bin, model)
