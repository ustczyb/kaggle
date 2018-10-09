import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_oof(clf, x_train, y_train, x_test, n_train=891, n_test=418):
    SEED = 0  # for reproducibility
    NFOLDS = 5  # set folds for out-of-fold prediction
    kf = KFold(n_train, n_folds=NFOLDS, random_state=SEED)
    oof_train = np.zeros((n_train,))
    oof_test = np.zeros((n_test,))
    oof_test_skf = np.empty((NFOLDS, n_test))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te).astype(np.int32)
        oof_test_skf[i, :] = clf.predict(x_test).astype(np.int32)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def stacking_train():
    # 1.训练和测试数据预处理
    train_df = pd.read_csv('data/train_processed.csv')
    test_df = pd.read_csv('data/test_processed.csv')
    test_np = test_df.values
    train_np = train_df.values
    train_y = train_np[:, 0]
    train_X = train_df.iloc[:, 1:].values

    # 2.模型创建
    lr_clf = LogisticRegressionCV()
    xgb_clf = XGBClassifier(n_jobs=-1, learning_rate=0.2, max_depth=3)
    rf_clf = RandomForestClassifier(max_depth=6, min_samples_leaf=2, n_estimators=300)
    gbdt_clf = GradientBoostingClassifier(max_depth=3)
    svm_clf = SVC(kernel='linear', C=0.1)

    # 3.构建第二层的训练数据
    lr_oof_train, lr_oof_test = get_oof(lr_clf, train_X, train_y, test_np)
    # xgb_oof_train, xgb_oof_test = get_oof(xgb_clf, train_X, train_y, test_np)
    rf_oof_train, rf_oof_test = get_oof(rf_clf, train_X, train_y, test_np)
    gbdt_oof_train, gbdt_oof_test = get_oof(gbdt_clf, train_X, train_y, test_np)
    svm_oof_train, svm_oof_test = get_oof(svm_clf, train_X, train_y, test_np)
    base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
                                           'LogisticRegression': lr_oof_train.ravel(),
                                           # 'XGBoost': xgb_oof_train.ravel(),
                                           'GradientBoost': gbdt_oof_train.ravel(),
                                           'SVM': svm_oof_train.ravel(),
                                           'Survival': train_y
                                           })
    base_predictions_train.to_csv('stacking_train.csv', index=False)
    base_predictions_test = pd.DataFrame({'RandomForest': rf_oof_test.ravel(),
                                           'LogisticRegression': lr_oof_test.ravel(),
                                           # 'XGBoost': xgb_oof_test.ravel(),
                                           'GradientBoost': gbdt_oof_test.ravel(),
                                           'SVM': svm_oof_test.ravel(),
                                           })
    base_predictions_test.to_csv('stacking_test.csv', index=False)
    x_train = np.concatenate((rf_oof_train, lr_oof_train, svm_oof_train), axis=1)
    x_test = np.concatenate((rf_oof_test, lr_oof_test, svm_oof_test), axis=1)
    return x_train, train_y, x_test


if __name__ == '__main__':
    X_train, y_train, X_test = stacking_train()
    lr_clf = LogisticRegressionCV()
    lr_clf.fit(X_train, y_train)
    predict = lr_clf.predict(X_test)
    test_data = pd.read_csv('data/test.csv')
    pd_result = pd.DataFrame(
        {'PassengerId': test_data['PassengerId'].values, 'Survived': predict.astype(np.int32)})
    pd_result.to_csv("stacking.csv", index=False)
