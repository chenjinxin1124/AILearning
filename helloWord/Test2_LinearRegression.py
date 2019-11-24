# -*-coding:utf-8-*-

if __name__ == '__main__':
    from sklearn import datasets, linear_model  # 引用 sklearn库，主要为了使用其中的线性回归模块

    # 创建数据集，把数据写入到numpy数组
    import numpy as np  # 引用numpy库，主要用来做科学计算
    import matplotlib.pyplot as plt  # 引用matplotlib库，主要用来画图

    data = np.array([[152, 51], [156, 53], [160, 54], [164, 55],
                     [168, 57], [172, 60], [176, 62], [180, 65],
                     [184, 69], [188, 72]])

    # 打印出数组的大小
    print(data.shape)
    x, y = data[:, 0].reshape(-1, 1), data[:, 1]

    # TODO 1. 实例化一个线性回归的模型
    regr = linear_model.LinearRegression()

    # TODO 2. 在x,y上训练一个线性回归模型。 如果训练顺利，则regr会存储训练完成之后的结果模型
    regr.fit(x, y)

    # TODO 3. 画出身高与体重之间的关系
    plt.scatter(x, y, color='red')

    # 画出已训练好的线条
    plt.plot(x, regr.predict(x), color='blue')

    # 画x,y轴的标题
    plt.xlabel('height (cm)')
    plt.ylabel('weight (kg)')
    plt.show()  # 展示

    # 利用已经训练好的模型去预测身高为163的人的体重
    print("Standard weight for person with 163 is %.2f" % regr.predict([[163]]))

'''
1.实例化的过程跟创建object的过程是类似的，这里需要调用sklearn的线性模型模块即可，暂时不需要指定任何的输入参数。并把实例化后的线性回归模型命名为regr
2.对于模型的训练，可以直接使用模型自带的fit函数
3.针对于所有给定的x，计算出在这个模型下的预测值，并把x和预测值通过matplotlib库来去可视化。
'''

'''
1.在sklearn里线性回归模型在linear_model模块里。可以参考https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
2.对于训练数据（x,y），通常通过sklearn中的fit(x,y)来实现模型的训练。这里x可以看做是特征部分，y可以看做是真实的值或者标签。
3.调用plt.plot()函数即可以画出一条线。
'''