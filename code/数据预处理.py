import pandas as pd
from sklearn.preprocessing import MinMaxScaler




def process_train_data(train_file, test_file):

    df = pd.read_csv(train_file)
    df2 = pd.read_csv(train_file)

    df = df.iloc[:, 1:]
    df = df.iloc[:, :-2]
    # df.to_csv('../data/train-set-without-lable.csv', index=False)

    df=df.drop('service',axis=1)
    # df.to_csv('../data/train-set-without-lable-service.csv', index=False)

    df = pd.get_dummies(df, columns=['proto', 'state'])
    # df.to_csv('../data/train-set-onehot.csv', index=False)

    scaler = MinMaxScaler()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    # df.to_csv('../data/train-set-processed.csv', index=False)


    # test_df = pd.read_csv(test_file)
    train_df = df

    # 提取测试集中的最后两列
    last_two_columns = df2.iloc[:, -2:]

    print(train_df.shape[0], last_two_columns.shape[0])
    # 检查train_df的行数是否与last_two_columns的行数匹配
    if train_df.shape[0] == last_two_columns.shape[0]:
        # 拼接列到训练集
        train_df = pd.concat([train_df, last_two_columns.reset_index(drop=True)], axis=1)
        print(train_df[:-2])
    else:
        print("11行数不匹配，请检查两个数据集!")

    # 保存新的训练集
    train_df.to_csv('../data/train-set-processed.csv', index=False)

def process_test_data(train_file):
    df = pd.read_csv(train_file)
    df2 = pd.read_csv(train_file)

    df = df.iloc[:, 1:]
    # 删除最后两列
    df = df.iloc[:, :-2]

    # df.to_csv('../data/train-set-without-lable.csv', index=False)

    df=df.drop('service',axis=1)
    # df.to_csv('../data/train-set-without-lable-service.csv', index=False)

    df = pd.get_dummies(df, columns=['proto', 'state'])
    # df_encoded.to_csv('../data/train-set-onehot.csv', index=False)


    # 将修改后的数据另存为新的CSV文件
    # df.to_csv('../data/train-set-processed.csv', index=False)


    # 提取测试集中的最后两列
    last_two_columns = df2.iloc[:, -2:]
    print(last_two_columns)
    # 检查train_df的行数是否与last_two_columns的行数匹配
    if df.shape[0] == last_two_columns.shape[0]:
        # 拼接列到训练集
        df = pd.concat([df, last_two_columns.reset_index(drop=True)], axis=1)
        print(df[:-2])
    else:
        print("行数不匹配，请检查两个数据集!")

    # 保存新的训练集
    df.to_csv('../data/test-set-processed.csv', index=False)

def process_data_nostring(train,test):
    df1 = pd.read_csv(train)
    df2 = pd.read_csv(test)
    df1 = df1.iloc[:, 1:]
    df2 = df2.iloc[:, 1:]
    df1 = df1.drop('service', axis=1)
    df2 = df2.drop('service', axis=1)
    df1 = df1.drop('proto', axis=1)
    df2 = df2.drop('proto', axis=1)
    df1 = df1.drop('state', axis=1)
    df2 = df2.drop('state', axis=1)

    df1.to_csv('../data/train-set-nostring.csv', index=False)
    df2.to_csv('../data/test-set-nostring.csv', index=False)

csv_file1 = '../data/UNSW_NB15_testing-set.csv'
csv_file2 = '../data/UNSW_NB15_training-set.csv'

# process_train_data(csv_file1, csv_file2)
# process_test_data(csv_file2)
process_data_nostring(csv_file1,csv_file2)