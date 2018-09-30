import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

# 姓名特征提取
# cabin特征提取
# one-hot encoding（线性模型）
# 年龄离散化（线性模型）
# pclass与sex做特征交叉（线性模型）

# 处理缺失的舱位信息
def handle_cabin(df):
    df.loc[(df.Cabin.notnull(), 'Cabin')] = 'Yes'
    df.loc[(df.Cabin.isnull(), 'Cabin')] = 'No'
    return df


# 提取姓名信息
def handle_name(df):
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
    title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other",
                          "Mme": "Other",
                          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Ms": "Other",
                          "Dona": "Other"}
    df.replace({'Title': title_replacements}, inplace=True)
    return df


# one-hot编码一列类别型信息
def handle_one_catagory(origin_df, col_name):
    dummies = pd.get_dummies(origin_df[col_name], prefix=col_name)
    df = pd.concat([origin_df, dummies], axis=1)
    df.drop([col_name], axis=1, inplace=True)
    return df


# one-hot编码处理类别信息
def handle_catagory(origin_df):
    df = handle_one_catagory(origin_df, 'Cabin')
    df = handle_one_catagory(df, 'Embarked')
    df = handle_one_catagory(df, 'Sex')
    df = handle_one_catagory(df, 'Pclass')
    df = handle_one_catagory(df, 'Title')
    df = handle_one_catagory(df, 'SexAndClass')
    return df


# 处理缺失的年龄信息
def handle_age(df, regressor):
    # 选取数值型特征作为输入特征
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # 按照年龄已知和未知划分为两类
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    y = known_age[:, 0]
    X = known_age[:, 1:]

    if regressor == None:
        # 使用随机森林对缺失的年龄进行填充
        regressor = RandomForestRegressor(n_estimators=500, n_jobs=-1)
        regressor.fit(X, y)

    age_predict = regressor.predict(unknown_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = age_predict
    return df, regressor


# 处理家庭信息
def handle_family(df):
    df['fmlNum'] = df['SibSp'] + df['Parch'] + 1
    return df


# 数据归一化
def age_fare_scaling(df, age_scale_param=None, fare_scale_param=None):
    scaler = preprocessing.StandardScaler()
    if age_scale_param == None:
        age_scale_param = scaler.fit(df[['Age']])
    df['Age_scaled'] = scaler.fit_transform(df[['Age']])
    if fare_scale_param == None:
        fare_scale_param = scaler.fit(df[['Fare']])
    df['Fare_scaled'] = scaler.fit_transform(df[['Fare']])
    return df, age_scale_param, fare_scale_param

# 特征离散化
def feature_scatter(df):
    # TODO
    pass

# 特征交叉
def feature_crossing(df):
    # mother:女性+Parch
    # df['Mother'] = ((df['Sex_female'] == 1) & (df['Parch'] > 1)).astype(int)
    # sex+pclass 特征交叉
    df['SexAndClass'] = df['Sex']
    df.loc[(df["Sex"] == 'male') & (df['Pclass'] == 1), "SexAndClass"] = 0
    df.loc[(df["Sex"] == 'female') & (df['Pclass'] == 1), "SexAndClass"] = 1

    df.loc[(df["Sex"] == 'male') & (df['Pclass'] == 2), "SexAndClass"] = 2
    df.loc[(df["Sex"] == 'female') & (df['Pclass'] == 2), "SexAndClass"] = 3

    df.loc[(df["Sex"] == 'male') & (df['Pclass'] == 3), "SexAndClass"] = 4
    df.loc[(df["Sex"] == 'female') & (df['Pclass'] == 3), "SexAndClass"] = 5
    return df


def select_columns(df):
    return df.filter(regex='Survived|Age_.*|SibSp|Parch|fmlNum|Fare_.*|Cabin_.*|Embarked_.*|Title_.*|SexAndClass_.*')


def load_data_and_preprocessing():
    """
    读取数据，进行数据处理和特征选择后将处理好的训练数据和测试数据以numpy.array的形式返回
    :return: train_np, test_np
    """
    # 1.训练集数据读入和处理
    train_data = pd.read_csv('data/train.csv')
    train_data = handle_cabin(train_data)
    train_data = handle_name(train_data)
    train_data, age_regressor = handle_age(train_data, None)
    train_data = feature_crossing(train_data)
    train_data = handle_catagory(train_data)
    train_data = handle_family(train_data)
    # 数据标准化
    train_data, age_scale_param, fare_scale_param = age_fare_scaling(train_data)
    train_df = select_columns(train_data)
    train_df.to_csv("data/train_processed.csv", index=False)

    # 2.测试数据集处理
    test_data = pd.read_csv('data/test.csv')
    test_data.loc[(test_data.Fare.isnull()), 'Fare'] = 0
    test_data = handle_cabin(test_data)
    test_data = handle_name(test_data)
    test_data = handle_age(test_data, age_regressor)[0]
    test_data = feature_crossing(test_data)
    test_data = handle_catagory(test_data)
    test_data = handle_family(test_data)
    test_data = age_fare_scaling(test_data, age_scale_param, fare_scale_param)[0]
    test_df = select_columns(test_data)
    test_df.to_csv("data/test_processed.csv", index=False)
    return train_df, test_df.values


def load_data_train():
    train_df, test_np = load_data_and_preprocessing()
    train_np = train_df.values
    train_y = train_np[:, 0]
    train_X = train_df[:, 1:]
    return train_X, train_y


if __name__ == '__main__':
    train_np, test_np = load_data_and_preprocessing()
    print(train_np.shape)
    print(test_np.shape)
