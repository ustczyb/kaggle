from sklearn.ensemble import GradientBoostingClassifier

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
        self.model = GradientBoostingClassifier().fit(self.train_X, self.train_y)

    def predict(self, predict_X):
        return self.model.predict(predict_X)

    def score(self, test_X, test_y):
        pass

    def feature_importance(self):
        return self.model.feature_importances_
