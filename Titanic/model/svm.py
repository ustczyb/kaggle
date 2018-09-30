from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC

from Titanic.model.interface import BaseModel


class SVMClf(BaseModel):

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
        svm_clf = SVC(kernel='linear', C=0.1)
        self.model = svm_clf.fit(self.train_X, self.train_y)
        # param_grid = {'C': [0.025, 0.05, 0.1, 0.2, 0.5],
        #               'kernel': ['rbf', 'linear']}
        # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        # grid_search = GridSearchCV(svm_clf, param_grid, n_jobs=-1, cv=kfold)
        # self.model = grid_search.fit(self.train_X, self.train_y)

    def predict(self, predict_X):
        return self.model.predict(predict_X)

    def score(self, test_X, test_y):
        pass

    def feature_importance(self):
        pass
