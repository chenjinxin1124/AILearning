# -*-coding:utf-8-*-

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # 读取图片的数据，存放在img
    img = plt.imread('../../result/myplot2.png')
    print(img.shape) # 打印图片的大小

    plt.imshow(img) # 展示图片
