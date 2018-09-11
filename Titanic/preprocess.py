import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


# 处理缺失的年龄信息
def handle_age(df):
    # 选取数值型特征作为输入特征
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # 按照年龄已知和未知划分为两类
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    y = known_age[:, 0]
    X = known_age[:, 1:]

    # 使用随机森林对缺失的年龄进行填充
    rf_reg = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    rf_reg.fit(X, y)

    age_predict = rf_reg.predict(unknown_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = age_predict
    return df, rf_reg


# 处理舱位信息
def handle_cabin(df):
    df.loc[(df.Cabin.notnull(), 'Cabin')] = 'Yes'
    df.loc[(df.Cabin.isnull(), 'Cabin')] = 'No'
    return df


# one-hot编码处理类别信息
def handle_catagory(origin_df):
    dummies_Cabin = pd.get_dummies(origin_df['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(origin_df['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(origin_df['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(origin_df['Pclass'], prefix='Pclass')
    df = pd.concat([origin_df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df


# 数据归一化
def scaling(df):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df[['Age']])
    df['Age_scaled'] = scaler.fit_transform(df[['Age']])
    fare_scale_param = scaler.fit(df[['Fare']])
    df['Fare_scaled'] = scaler.fit_transform(df[['Fare']])
    return age_scale_param, fare_scale_param


def load_data_and_preprocessing():
    """
    读取数据，进行数据处理和特征选择后将处理好的训练数据和测试数据以numpy.array的形式返回
    :return: train_np, test_np
    """
    train_data = pd.read_csv('data/train.csv')
    train_data, rfr = handle_age(train_data)
    train_data = handle_cabin(train_data)
    train_data = handle_catagory(train_data)
    age_scale_param, fare_scale_param = scaling(train_data)
    train_df = train_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values
    # 测试数据集处理
    test_data = pd.read_csv('data/test.csv')
    test_data.loc[(test_data.Fare.isnull()), 'Fare'] = 0
    # 接着我们对test_data做和train_data中一致的特征变换
    # 首先用同样的RandomForestRegressor模型填上丢失的年龄
    tmp_df = test_data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[test_data.Age.isnull()].values
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    test_data.loc[(test_data.Age.isnull()), 'Age'] = predictedAges
    data_test = handle_cabin(test_data)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    scaler = preprocessing.StandardScaler()
    df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']], age_scale_param)
    df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']], fare_scale_param)
    test_np = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*').values
    return train_np, test_np


if __name__ == '__main__':
    train_np, test_np = load_data_and_preprocessing()
    print(train_np.shape)
    print(test_np.shape)
