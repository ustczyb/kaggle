import warnings

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from Titanic.preprocess import load_data_and_preprocessing


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
    param_grid = { 'learning_rate': [0.2],
        # 'n_estimators':[50, 100, 150, 200],
        'max_depth': [3]}
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


def random_forest(train_X, train_y):
    rf_clf = RandomForestClassifier(n_jobs=-1, oob_score=True)
    rf_clf.fit(train_X, train_y)
    print("random forest oob score:", rf_clf.oob_score_)
    return rf_clf


def rf_predict(rf, test_np):
    return rf.predict(test_np)


def generate_csv_res(test_np, predict):
    test_data = pd.read_csv('data/test.csv')
    pd_result = pd.DataFrame(
        {'PassengerId': test_data['PassengerId'].as_matrix(), 'Survived': predict.astype(np.int32)})
    pd_result.to_csv("predict.csv", index=False)


if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    train_X, train_y, test_np = load_data()

    # 使用logistic regression 0.75
    # logistic_clf = logistic_model(train_X, train_y)
    # y_predict = logistic_predict(logistic_clf, test_np)

    # 使用xgboost 0.74
    xgb_clf = xgboost_model(train_X, train_y)
    y_predict = xgboost_predict(xgb_clf, test_np)
    # 使用svm

    # 使用随机森林 0.74
    # rf_clf = random_forest(train_X, train_y)
    # y_predict = rf_predict(rf_clf, test_np)


    generate_csv_res(test_np, y_predict)
