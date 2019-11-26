# -*-coding:utf-8-*-

if __name__ == '__main__':
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns  # 用来画一个heatmap

    # 读取数据
    # 利用pandas读取数据并打印 可以看得到具体数据了，每一列都是一个特征值
    df = pd.read_csv('./data.csv')
    print(df)  # data frame

    '''
    做了特征的处理，把类别型特征转换成了独热编码的形式。
    这里针对Color和Type做了独热编码操作。
    而对于Brand没有做任何操作，因为在给定的数据里Brand都是一样的，可以去掉了. 
    可以看出来表格中多了几个列，分别以Color:和Type:开头，表示的就是转换成独热编码之后的结果。
    '''
    # 特征处理
    # 把颜色独热编码
    df_colors = df['Color'].str.get_dummies().add_prefix('Color: ')
    # 把类型独热编码
    df_type = df['Type'].apply(str).str.get_dummies().add_prefix('Type: ')
    # 添加独热编码数据列
    df = pd.concat([df, df_colors, df_type], axis=1)
    # 去除独热编码对应的原始列
    df = df.drop(['Brand', 'Type', 'Color'], axis=1)

    print(df)

    # 数据转换
    matrix = df.corr()
    f, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, square=True)
    plt.title('Car Price Variables')
    plt.show()
    '''
    第一步先看一下数据特征之间的相关性，这里使用了corr()函数来计算特征之间的相关性。
    之后通通过sns模块来可视化相关性。 颜色越深的代表相关性越大。
    '''

    # 忽略警告信息
    import warnings

    warnings.filterwarnings('ignore')

    '''导入KNN相关的库以及其他相关的库。
    这里StandardScaler用来做特征的归一化，把原始特征转换成均值为0方差为1的高斯分布。'''
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    '''主要用来做特征的归一化。
    这里需要注意的一点是特征的归一化的标准一定要来自于训练数据，之后再把它应用在测试数据上。
    因为实际情况下，测试数据是我们看不到的，也就是统计不到均值和方差。'''
    # 先将标签取出来
    y = df['Ask Price'].values.reshape(-1, 1)
    # 删除标签列作为特征数据
    X = df.drop(['Ask Price'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

    X_normalizer = StandardScaler()  # N(0,1)
    # 将需要归一化处理的三个列值替换为处理后的
    X_train.loc[:, ['Construction Year', 'Days Until MOT', 'Odometer']] = X_normalizer.fit_transform(
        X_train[['Construction Year', 'Days Until MOT', 'Odometer']])
    X_test.loc[:, ['Construction Year', 'Days Until MOT', 'Odometer']] = X_normalizer.transform(
        X_test[['Construction Year', 'Days Until MOT', 'Odometer']])

    y_normalizer = StandardScaler()
    y_train = y_normalizer.fit_transform(y_train)
    y_test = y_normalizer.transform(y_test)

    '''
    这部分主要用来训练KNN模型，以及用KNN模型做预测，并把结果展示出来。
    这里我们使用了y_normalizer.inverse_transform，
    因为我们在训练的时候把预测值y也归一化了，所以最后的结论里把之前归一化的结果重新恢复到原始状态。 
    在结果图里，理想情况下，假如预测值和实际值一样的话，所有的点都会落在对角线上，但实际上现在有一些误差。
    '''
    knn = KNeighborsRegressor(n_neighbors=2)
    knn.fit(X_train, y_train.ravel())

    # Now we can predict prices:
    y_pred = knn.predict(X_test)
    y_pred_inv = y_normalizer.inverse_transform(y_pred)
    y_test_inv = y_normalizer.inverse_transform(y_test)

    # Build a plot
    plt.scatter(y_pred_inv, y_test_inv)
    plt.xlabel('Prediction')
    plt.ylabel('Real value')

    # Now add the perfect prediction line
    diagonal = np.linspace(500, 1500, 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.xlabel('Predicted ask price')
    plt.ylabel('Ask price')
    plt.show()

    # 打印最终的结果，也就是预测的值 可以看到实际预测的值
    print(y_pred_inv)

    print(knn)