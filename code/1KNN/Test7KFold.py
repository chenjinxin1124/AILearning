# -*-coding:utf-8-*-

if __name__ == '__main__':
    from sklearn.model_selection import GridSearchCV # 通过网格方式来搜索参数
    from sklearn import datasets
    from sklearn.neighbors import KNeighborsClassifier

    # 导入iris数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 设置需要搜索的k值，‘n_neighbors’是sklearn中的KNN参数
    parameters = {'n_neighbors':[1,3,5,7,9,11,13,15]}
    knn=KNeighborsClassifier() # 注意：在这里不用指定参数

    # 通过GridSearchCV来搜索最好的K值，这个模块的内部其实就是对于每一个k值做了评估
    clf=GridSearchCV(knn,parameters,cv=5)
    clf.fit(X,y)

    # 输出最好的参数以及对应的准确率
    print("best score is: %.2f"%clf.best_score_," best param: ",clf.best_params_)