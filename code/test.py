import pandas as pd

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
    
    print(binary_values)
    print(binary_columns)
    # 创建新的 DataFrame 来存储二进制列
    binary_df = pd.DataFrame(binary_values, columns=binary_columns)
    print(binary_df)
    # 将原df与新的二进制列合并
    df=df.drop(columns=[column_name])
    df = pd.concat([df, binary_df], axis=1)
    
    return df

# 示例使用
# data = {'words': ['apple', 'banana', 'cherry', 'apple', 'banana']}
# df = pd.DataFrame(data)
# print("原始数据：")
# print(df)

# df = convert_column_to_binary(df, 'words')
# print("\n转换后的数据：")
# print(df)

import sys
print(sys.version)
import torch
print(torch.cuda.is_available())  # 返回 True 表示 GPU 可用
print(torch.cuda.device_count())  # 返回可用 GPU 的数量
print(torch.cuda.get_device_name(0))  # 返回第一个 GPU 的名称RRRR

{'C': [0.1], 'gamma': [0.1], 'kernel': ['rbf']}
{'C': [1], 'gamma': [0.1], 'kernel': ['rbf']}
{'C': [10], 'gamma': [0.1], 'kernel': ['rbf']}