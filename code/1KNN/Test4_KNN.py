#-*-coding:utf-8-*-

# 一个KNN算法解决分类问题。 主要的模块包括欧式距离的计算以及投票环节。
# 对于投票的部分，可以使用Counter object的most_common函数来实现。
if __name__ == '__main__':
    from sklearn import datasets
    from collections import Counter  # 为了做投票
    from sklearn.model_selection import train_test_split
    import numpy as np

    # 导入iris数据
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)

    def euc_dis(instance1, instance2):
        """
        计算两个样本instance1和instance2之间的欧式距离
        instance1: 第一个样本， array型
        instance2: 第二个样本， array型
        """
        # TODO
        dist = np.sqrt(sum((instance1 - instance2) ** 2))
        return dist


    def knn_classify(X, y, testInstance, k):
        """
        给定一个测试数据testInstance, 通过KNN算法来预测它的标签。
        X: 训练数据的特征
        y: 训练数据的标签
        testInstance: 测试数据，这里假定一个测试数据 array型
        k: 选择多少个neighbors?
        """
        # TODO  返回testInstance的预测标签 = {0,1,2}
        distances = [euc_dis(x, testInstance) for x in X]
        kneighbors = np.argsort(distances)[:k]
        count = Counter(y[kneighbors])
        return count.most_common()[0][0]


    # 预测结果。
    predictions = [knn_classify(X_train, y_train, data, 3) for data in X_test]
    correct = np.count_nonzero((predictions == y_test) == True)
    print("Accuracy is: %.3f" % (correct / len(X_test)))