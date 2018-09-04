from sklearn.metrics import accuracy_score
import pandas as pd
from Titanic.preprocess import load_data_and_preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
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


def xgboost_model(train_X, train_y):
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_y,random_state=66, test_size=0.25)
    train_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_test, label=y_test)
    watch_list = [(test_data, 'eval'), (train_data, 'train')]
    param={'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
    xgb_clf = XGBClassifier(n_jobs=-1)
    xgb_clf.fit(X_train, y_train)
    predict_res = xgb_clf.predict(X_test)
    y_predict = [round(v) for v in predict_res]
    accuracy = accuracy_score(y_test, y_predict)
    print("xgboost score:", accuracy)
    return xgb_clf


def generate_csv_res(test_np, predict):
    test_data = pd.read_csv('data/test.csv')
    pd_result = pd.DataFrame(
        {'PassengerId': test_data['PassengerId'].as_matrix(), 'Survived': predict})
    pd_result.to_csv("predict.csv", index=False)


if __name__ == '__main__':
    train_X, train_y, test_np = load_data()
    logistic_clf = logistic_model(train_X, train_y)
    # xgb_clf = xgboost_model(train_X, train_y)
    y_predict = logistic_clf.predict(test_np)
    # y_predict = [round(v) for v in predict_res]
    generate_csv_res(test_np, y_predict)
