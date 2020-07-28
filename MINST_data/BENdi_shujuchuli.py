'''
    使用python解析二进制文件
'''
import numpy as np
import struct   #导入struck模块，
from sklearn import preprocessing   #标准化处理
def process():
    file1 = 'D:/pycharmss/pycharm CNN/tensorflow-mnist-cnn-master/data/train-images.idx3-ubyte'  # 训练集数据
    file2 = 'D:/pycharmss/pycharm CNN/tensorflow-mnist-cnn-master/data/train-labels.idx1-ubyte'  # 训练集标签

    file3 = 'D:/pycharmss/pycharm CNN/tensorflow-mnist-cnn-master/data/t10k-images.idx3-ubyte'  # 训练集数据
    file4 = 'D:/pycharmss/pycharm CNN/tensorflow-mnist-cnn-master/data/t10k-labels.idx1-ubyte'  # 训练集标签

    train_imgs, train_data_head = loadImageSet(file1)
    # print('data_head:', train_data_head)
    # print(type(train_imgs))  #数组类型
    # #print('imgs_array:', imgs)
    # size = np.reshape(train_imgs[1, :], [28, 28])
    # #print(np.reshape(imgs[1, :], [28, 28]))  # 取出其中一张图片的像素，转型为28*28，大致就能从图像上看出是几啦
    # print((np.reshape(train_imgs[1, :], [28, 28])).shape)  # 数据形状

    print('----------我是分割线-----------')

    train_labels, train_labels_head = loadLabelSet(file2)
    # print('labels_head:', train_labels_head)
    # print(type(train_labels))
    # print(train_labels.shape)

    test_imgs, test_data_head = loadImageSet(file3)
    test_labels, test_labels_head = loadLabelSet(file4)
    TA_labels , TE_labels = one_hot(train_labels, test_labels)
    return train_imgs, TA_labels, test_imgs, TE_labels

def loadImageSet(filename):
    binfile = open(filename, 'rb')  # 打开文件
    buffers = binfile.read()        #读取二进制文件

    head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置 ，用来计算fmt格式所描述的机构的大小
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])  # reshape为[60000,784]型数组

    return imgs, head


def loadLabelSet(filename):
    binfile = open(filename, 'rb')  # 读二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)  # 取label文件前2个整形数

    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    numString = '>' + str(labelNum) + "B"  # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

    binfile.close()
    labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)

    return labels, head

def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])  # 将列表转换为数组形式，在进行重组，-1表示自动匹配，即自动识别7000
        Test_Y = np.array(Test_Y).reshape([-1, 1])  # 将列表转换为数组形式，在进行重组，-1表示自动匹配，即自动识别3000

        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()

        Train_Y = np.asarray(Train_Y, dtype=np.int32)  # 使用asarray创建array变量,都可以将数据结构转换为数组
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

if __name__ == "__main__":
    train, T_bel, test, t_bel = process()