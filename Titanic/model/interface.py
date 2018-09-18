from abc import abstractmethod, ABCMeta


class BaseModel(metaclass=ABCMeta):

    @abstractmethod
    def train(self):
        """
        生成并训练模型
        :return: 返回训练完成的模型
        """

    @abstractmethod
    def predict(self, predict_X):
        """
        根据训练好的模型预测结果
        :param predict_X: 预测数据
        :return: 预测结果 np.array
        """
        pass

    @abstractmethod
    def score(self, test_X, test_y):
        """
        模型评分
        :param test_X: 测试数据集样本
        :param test_y: 测试数据集结果
        :return:
        """
        pass

    @abstractmethod
    def feature_importance(self):
        pass
