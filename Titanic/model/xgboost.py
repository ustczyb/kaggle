import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

from Titanic.model.interface import BaseModel


class XgbClf(BaseModel):

    def __init__(self, train_df, train_label):
        """
        :param train_df:  训练样本，格式为pd.DataFrame
        :param train_label:  训练样本对应的标记，格式为np.array
        """
        self.train_df = train_df
        self.train_X = train_df.values
        self.train_y = train_label
        self.model = None

    def train(self):
        """
        生成并训练模型
        :param train_X np.array 训练数据集的样本
        :param train_y np.array 训练数据集的类别
        """
        xgb_clf = XGBClassifier(n_jobs=-1, learning_rate=0.2, max_depth=3)
        xgb_clf.fit(self.train_X, self.train_y)
        # param_grid = {'learning_rate': [0.2],
        #               'max_depth': [3]}
        # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        # grid_search = GridSearchCV(xgb_clf, param_grid, n_jobs=-1, cv=kfold)
        # grid_search.fit(self.train_X, self.train_y)
        # self.model = grid_search

    def predict(self, predict_X):
        """
        根据训练好的模型预测结果
        :param model: 训练好的模型
        :param predict_X: 预测数据
        :return: 预测结果 np.array
        """
        predict_res = self.model.predict(predict_X)
        y_predict = np.array([round(v) for v in predict_res])
        return y_predict

    def score(self, test_X, test_y):
        pass

    def feature_importance(self):
        """
        特征重要度评分
        :return: np.array
        """
        return self.model.feature_importances_
