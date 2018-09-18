from sklearn.linear_model import LogisticRegressionCV

from Titanic.model.interface import BaseModel


class LogisticClf(BaseModel):

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
        训练模型
        """
        self.model = LogisticRegressionCV().fit(self.train_X, self.train_y)

    def predict(self, predict_X):
        """
        根据训练好的模型预测结果
        :param model: 训练好的模型
        :param predict_X: 预测数据
        :return: 预测结果 np.array
        """
        return self.model.predict(predict_X)

    def score(self, test_X, test_y):
        """
        模型评分
        :param test_X: 测试数据集样本
        :param test_y: 测试数据集结果
        :return:
        """
        pass

    def feature_importance(self):
        """
        特征重要度评分
        :return: np.array
        """
        return self.model.coef_.reshape(-1)
