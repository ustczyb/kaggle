import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def gather_results():
    gather_result = pd.DataFrame({
        'PassengerId': pd.read_csv('predict_rf.csv').values[:, 0],
        'RandomForest': pd.read_csv('predict_rf.csv').values[:, 1],
        'LogisticRegression': pd.read_csv('predict_lr.csv').values[:, 1],
        # 'GBDT': pd.read_csv('predict_gbdt.csv').values[:, 1],
        # 'Xgboost': pd.read_csv('predict_xgb.csv').values[:, 1],
        'SVM': pd.read_csv('predict_svm.csv').values[:, 1]
    })
    gather_result.to_csv('gather_predict.csv', index=False)


def vote():
    """
    对多个分类器的结果投票
    :param train_X:
    :param train_y:
    :return:
    """
    # gather_results()
    gather_df = pd.read_csv('gather_predict.csv')
    vote_res_df = pd.DataFrame()
    vote_res_df['PassengerId'] = gather_df['PassengerId']
    gather_df.drop(['PassengerId'], axis=1, inplace=True)
    vote_res_df['Survived'] = gather_df.mean(axis=1).map(lambda s: 1 if s >= 0.5 else 0)
    vote_res_df.to_csv('vote_result.csv', index=False)


def stacking(train_X, train_y):
    """
    对多个分类器进行stacking
    :param train_X:
    :param train_y:
    :return:
    """
    pass


if __name__ == '__main__':
    vote()
