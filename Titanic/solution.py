import numpy as np
import pandas as pd
from Titanic.model.lr import LogisticClf
from Titanic.model.random_forest import RandomForestClf
from Titanic.model.xgboost import XgbClf
from Titanic.preprocess import load_data_train, load_data_and_preprocessing


def generate_csv_res(predict, clf_name):
    test_data = pd.read_csv('data/test.csv')
    pd_result = pd.DataFrame(
        {'PassengerId': test_data['PassengerId'].values, 'Survived': predict.astype(np.int32)})
    pd_result.to_csv("predict_" + clf_name + ".csv", index=False)


def generate_feature_importance(train_df, feature_importance, clf_name):
    cols = train_df.columns.values
    feature_dataframe = pd.DataFrame({'features': cols, clf_name + ' feature importances': feature_importance})
    return feature_dataframe


if __name__ == '__main__':
    # 1.训练和测试数据预处理
    train_df, test_np = load_data_and_preprocessing()
    train_np = train_df.values
    train_y = train_np[:, 0]
    train_X = train_df.iloc[:, 1:]

    # 2.各模型训练

    # 2.1 logistic regression
    logistic_clf = LogisticClf(train_X, train_y)
    logistic_clf.train()
    lr_feature_importance = generate_feature_importance(train_X, logistic_clf.feature_importance(), "lr")
    # lr_predict = logistic_clf.predict(test_np)
    # generate_csv_res(lr_predict, "lr")

    # 2.2 xgboost
    xgb_clf = XgbClf(train_X, train_y)
    xgb_clf.train()
    xgb_feature_importance = generate_feature_importance(train_X, xgb_clf.feature_importance(), "xgb")
    # xgb_predict = xgb_clf.predict(test_np)
    # generate_csv_res(xgb_predict, "xgb")

    # 2.3 random forest
    rf_clf = RandomForestClf(train_X, train_y)
    rf_clf.train()
    rf_feature_importance = generate_feature_importance(train_X, rf_clf.feature_importance(), "rf")
    # rf_predict = rf_clf.predict(test_np)
    # generate_csv_res(rf_predict, "rf")
