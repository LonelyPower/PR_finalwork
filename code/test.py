import pandas as pd


# feature_importance_df = pd.read_csv("../data/xgb_feature_importance.csv")
# feature_importance_df1 = pd.read_csv("../data/gwo_feature_importance.csv")
feature_importance_df1 = pd.read_csv("../data/2.csv")
feature_importance_df = pd.read_csv("../data/xgb_feature_importance.csv")
top_20_features1 = feature_importance_df["Feature"].head(20).tolist()
top_20_features2 = feature_importance_df1["Feature"].head(20).tolist()
# print(top_20_features1)
# print(len(top_20_features2))
print(set(top_20_features1).difference(set(top_20_features2)))
print(set(top_20_features1))
# print(top_20_features2)
common_features = list(set(top_20_features1) & set(top_20_features2))
# print(common_features)
df1 = pd.read_csv("../data/UNSW_NB15_testing-set.csv")
df2 = pd.read_csv("../data/UNSW_NB15_training-set.csv")

# a=df1['state'].unique().shape
# print(a)
# b=df2['state'].unique().shape
# print(set(df1['state'].unique()).difference(set(df2['state'].unique())))
# c=df1['proto'].unique().shape
# d=df2['proto'].unique().shape
# print(a)
# print(b)
{'udp': 0, 'arp': 1, 'tcp': 2, 'igmp': 3, 'ospf': 4, 'sctp': 5, 'gre': 6, 'ggp': 7, 'ip': 8, 'ipnip': 9, 'st2': 10, 'argus': 11, 'chaos': 12, 'egp': 13, 'emcon': 14, 'nvp': 15, 'pup': 16, 'xnet': 17, 'mux': 18, 'dcn': 19, 'hmp': 20, 'prm': 21, 'trunk-1': 22, 'trunk-2': 23, 'xns-idp': 24, 'leaf-1': 25, 'leaf-2': 26, 'irtp': 27, 'rdp': 28, 'netblt': 29, 'mfe-nsp': 30, 'merit-inp': 31, '3pc': 32, 'idpr': 33, 'ddp': 34, 'idpr-cmtp': 35, 'tp++': 36, 'ipv6': 37, 'sdrp': 38, 'ipv6-frag': 39, 'ipv6-route': 40, 'idrp': 41, 'mhrp': 42, 'i-nlsp': 43, 'rvd': 44, 'mobile': 45, 'narp': 46, 'skip': 47, 'tlsp': 48, 'ipv6-no': 49, 'any': 50, 'ipv6-opts': 51, 'cftp': 52, 'sat-expak': 53, 'ippc': 54, 'kryptolan': 55, 'sat-mon': 56, 'cpnx': 57, 'wsn': 58, 'pvp': 59, 'br-sat-mon': 60, 'sun-nd': 61, 'wb-mon': 62, 'vmtp': 63, 'ttp': 64, 'vines': 65, 'nsfnet-igp': 66, 'dgp': 
67, 'eigrp': 68, 'tcf': 69, 'sprite-rpc': 70, 'larp': 71, 'mtp': 72, 'ax.25': 73, 'ipip': 74, 'aes-sp3-d': 75, 'micp': 76, 'encap': 77, 'pri-enc': 78, 'gmtp': 79, 'ifmp': 80, 'pnni': 81, 'qnx': 82, 'scps': 83, 'cbt': 84, 'bbn-rcc': 85, 'igp': 86, 'bna': 87, 'swipe': 88, 'visa': 89, 'ipcv': 90, 'cphb': 91, 'iso-tp4': 92, 'wb-expak': 93, 'sep': 94, 'secure-vmtp': 95, 'xtp': 96, 'il': 97, 'rsvp': 98, 'unas': 99, 'fc': 100, 'iso-ip': 101, 'etherip': 102, 'pim': 103, 'aris': 104, 'a/n': 105, 'ipcomp': 106, 'snp': 107, 'compaq-peer': 108, 'ipx-n-ip': 109, 'pgm': 110, 'vrrp': 111, 'l2tp': 112, 'zero': 113, 'ddx': 114, 'iatp': 115, 'stp': 116, 'srp': 117, 'uti': 118, 'sm': 119, 'smp': 120, 'isis': 121, 'ptp': 122, 'fire': 123, 'crtp': 124, 'crudp': 125, 
'sccopmce': 126, 'iplt': 127, 'pipe': 128, 'sps': 129, 'ib': 130}

{'FIN': 0, 'INT': 1, 'CON': 2, 'ECO': 3, 'REQ': 4, 'RST': 5, 'PAR': 6, 'URN': 7, 'no': 8}


{'tcp': 0, 'udp': 1, 'arp': 2, 'ospf': 3, 'icmp': 4, 'igmp': 5, 'rtp': 6, 'ddp': 7, 'ipv6-frag': 8, 'cftp': 
9, 'wsn': 10, 'pvp': 11, 'wb-expak': 12, 'mtp': 13, 'pri-enc': 14, 'sat-mon': 15, 'cphb': 16, 'sun-nd': 17, 
'iso-ip': 18, 'xtp': 19, 'il': 20, 'unas': 21, 'mfe-nsp': 22, '3pc': 23, 'ipv6-route': 24, 'idrp': 25, 'bna': 26, 'swipe': 27, 'kryptolan': 28, 'cpnx': 29, 'rsvp': 30, 'wb-mon': 31, 'vmtp': 32, 'ib': 33, 'dgp': 34, 'eigrp': 35, 'ax.25': 36, 'gmtp': 37, 'pnni': 38, 'sep': 39, 'pgm': 40, 'idpr-cmtp': 41, 'zero': 42, 'rvd': 43, 'mobile': 44, 'narp': 45, 'fc': 46, 'pipe': 47, 'ipcomp': 48, 'ipv6-no': 49, 'sat-expak': 50, 'ipv6-opts': 51, 'snp': 52, 'ipcv': 53, 'br-sat-mon': 54, 'ttp': 55, 'tcf': 56, 'nsfnet-igp': 57, 'sprite-rpc': 58, 'aes-sp3-d': 59, 'sccopmce': 60, 'sctp': 61, 'qnx': 62, 'scps': 63, 'etherip': 64, 'aris': 65, 'pim': 66, 'compaq-peer': 67, 'vrrp': 68, 'iatp': 69, 'stp': 70, 'l2tp': 71, 'srp': 72, 'sm': 73, 'isis': 74, 'smp': 75, 'fire': 76, 'ptp': 77, 'crtp': 78, 'sps': 79, 'merit-inp': 80, 'idpr': 81, 'skip': 82, 'any': 83, 'larp': 84, 'ipip': 85, 'micp': 86, 'encap': 87, 'ifmp': 88, 'tp++': 89, 'a/n': 90, 'ipv6': 91, 'i-nlsp': 92, 'ipx-n-ip': 93, 'sdrp': 94, 'tlsp': 95, 'gre': 96, 'mhrp': 97, 'ddx': 98, 'ippc': 99, 'visa': 100, 'secure-vmtp': 101, 
'uti': 102, 'vines': 103, 'crudp': 104, 'iplt': 105, 'ggp': 106, 'ip': 107, 'ipnip': 108, 'st2': 109, 'argus': 110, 'bbn-rcc': 111, 'egp': 112, 'emcon': 113, 'igp': 114, 'nvp': 115, 'pup': 116, 'xnet': 117, 'chaos': 
118, 'mux': 119, 'dcn': 120, 'hmp': 121, 'prm': 122, 'trunk-1': 123, 'xns-idp': 124, 'leaf-1': 125, 'leaf-2': 126, 'rdp': 127, 'irtp': 128, 'iso-tp4': 129, 'netblt': 130, 'trunk-2': 131, 'cbt': 132}



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
    print(binary_df)
    df=df.drop(columns=[column_name])
    df = pd.concat([df, binary_df], axis=1)

    # 将二进制字符串拆分成每一位
    # binary_columns = [f'bit_{i}' for i in range(max_len)]
    # # binary_columns.append(f'bit_{max_len}')  # 添加最后一列
    # binary_values = [list(x) for x in binary_df]  # 将每个二进制串拆分为单个字符
    
    # print(binary_values)
    # print(binary_columns)
    # # 创建新的 DataFrame 来存储二进制列
    # binary_df = pd.DataFrame(binary_values, columns=binary_columns)
    # print(binary_df)
    # # 将原df与新的二进制列合并
    # df=df.drop(columns=[column_name])
    # df = pd.concat([df, binary_df], axis=1)
    
    return df

# # 示例使用
# data = {'words': ['apple', 'banana', 'cherry', 'apple', 'banana']}
# df = pd.DataFrame(data)
# print("原始数据：")
# print(df)

# df = convert_column_to_binary(df, 'words')
# print("\n转换后的数据：")
# print(df)

# import sys
# print(sys.version)
# import torch
# print(torch.cuda.is_available())  # 返回 True 表示 GPU 可用
# print(torch.cuda.device_count())  # 返回可用 GPU 的数量
# print(torch.cuda.get_device_name(0))  # 返回第一个 GPU 的名称RRRR

# {'C': [0.1], 'gamma': [0.1], 'kernel': ['rbf']}
# {'C': [1], 'gamma': [0.1], 'kernel': ['rbf']}
# {'C': [10], 'gamma': [0.1], 'kernel': ['rbf']}