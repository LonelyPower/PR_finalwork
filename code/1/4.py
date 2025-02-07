from collections import defaultdict
import pandas as pd
# 创建一个 defaultdict，默认值为另一个 defaultdict，用于统计每个 proto 下的 label 计数
proto_label_dict = defaultdict(lambda: defaultdict(int))
data = pd.read_csv('../data/UNSW_NB15_training-set.csv')
# 遍历数据统计
for _, row in data.iterrows():
    proto = row['proto']
    label = row['label']
    proto_label_dict[proto][label] += 1
proto_label_dict.to_csv('proto_label_dict.csv', index=False)
# 打印结果
for proto, label_counts in proto_label_dict.items():
    print(f"Protocol: {proto}")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count}")
