from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from Titanic.model.interface import BaseModel


class GradientBoostingClf(BaseModel):

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
        gbdt_clf = GradientBoostingClassifier(max_depth=3)
        self.model = gbdt_clf.fit(self.train_X, self.train_y)
        # param_grid = {'n_estimators': [100, 200, 300, 500],
        #               'learning_rate': [0.1, 0.2, 0.3],
        #               'max_depth': [3, 4, 5, 6]}
        # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        # grid_search = GridSearchCV(gbdt_clf, param_grid, n_jobs=-1, cv=kfold)
        # self.model = grid_search.fit(self.train_X, self.train_y)
        # print('best params:', grid_search.best_params_)
        # print('best score:', grid_search.best_score_)

    def predict(self, predict_X):
        return self.model.predict(predict_X)

    def score(self, test_X, test_y):
        pass

    def feature_importance(self):
        return self.model.feature_importances_
