import warnings

from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from Titanic.preprocess import load_data_and_preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from xgboost import XGBClassifier


def load_data():
    train_np, test_np = load_data_and_preprocessing()
    train_y = train_np[:, 0]
    train_X = train_np[:, 1:]
    return train_X, train_y, test_np


def logistic_model(train_X, train_y):
    logistic_clf = LogisticRegression().fit(train_X, train_y)
    cv_score = cross_validation.cross_val_score(logistic_clf, train_X, train_y, cv=5)
    print("logistic cv score : ", cv_score)
    return logistic_clf


def logistic_predict(logistic_clf, test_np):
    y_predict = logistic_clf.predict(test_np)
    return y_predict


def xgboost_model(train_X, train_y):
    xgb_clf = XGBClassifier(n_jobs=-1)
    param_grid = {# 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
                  # 'n_estimators':[50, 100, 150, 200],
                  'max_depth': [2, 3, 4, 5, 6, 7]}
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(xgb_clf, param_grid, n_jobs=-1, cv=kfold)
    grid_search.fit(train_X, train_y)

    # predict_res = xgb_clf.predict(X_test)
    # y_predict = [round(v) for v in predict_res]
    # accuracy = accuracy_score(y_test, y_predict)
    print("xgboost score:", grid_search.best_score_)
    return grid_search


def xgboost_predict(xgb_clf, test_np):
    predict_res = xgb_clf.predict(test_np)
    y_predict = np.array([round(v) for v in predict_res])
    return y_predict


def generate_csv_res(test_np, predict):
    test_data = pd.read_csv('data/test.csv')
    pd_result = pd.DataFrame(
        {'PassengerId': test_data['PassengerId'].as_matrix(), 'Survived': predict.astype(np.int32)})
    pd_result.to_csv("predict.csv", index=False)


if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    train_X, train_y, test_np = load_data()

    # 使用logistic regression
    # logistic_clf = logistic_model(train_X, train_y)
    # y_predict = logistic_predict(logistic_clf, test_np)
    # generate_csv_res(test_np, y_predict)

    # 使用xgboost
    xgb_clf = xgboost_model(train_X, train_y)
    y_predict = xgboost_predict(xgb_clf, test_np)
    generate_csv_res(test_np, y_predict)
    # 使用svm
