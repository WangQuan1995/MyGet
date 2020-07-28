from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import os
import gzip  #使用gzip模块完成对文件的压缩，使用gzip模块完成对文件的解压
import numpy as np
import struct   #导入struck模块，
# labels = np.empty((20, 44))   #创建空的数组
#
# file1 = 'D:/pycharmss/pycharm CNN/tensorflow-mnist-cnn-master/data/train-images.idx3-ubyte'  # 训练集数据
# file2 = 'D:/pycharmss/pycharm CNN/tensorflow-mnist-cnn-master/data/train-labels.idx1-ubyte'  # 训练集标签
#
# # 读取二进制数据
# bin_data = open(file1, 'rb').read()   #读取二进制文件，返回字节数据
# head = struct.unpack_from('>IIII', bin_data, 0)  # 取前4个整数，返回一个元组
# offset = struct.calcsize('>IIII')  # 定位到data开始的位置 ，用来计算fmt格式所描述的机构的大小
# imgNum = head[1]
# width = head[2]
# height = head[3]
#
# bits = imgNum * width * height  # data一共有60000*28*28个像素值
# bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
#
# imgs = struct.unpack_from(bitsString, bin_data, offset)  # 取data数据，返回一个元组
#
# imgs = np.reshape(imgs, [imgNum, width * height])  # reshape为[60000,784]型数组
#
# d_path = r'data\okh\kkk'
# train_data_filename = os.path.join(d_path, 'train-images.idx3-ubyte')
# print(train_data_filename)
# x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 4, 5, 6, 2, 3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 4, 5, 6, 2, 3, 3]])
# c = x.reshape([-1, 4, 4])  #数据形状转变
# cc = np.reshape(x, [-1, 4, 4])
#
# x = 1
# if not x:
#     print('zhengque')
# else:
#     print('false')
#
# filepath = os.path.join('data', 'train-images-idx3-ubyte.gz')  #引号为字符串格式
#
# print(filepath)

face = misc.face()   ##face是测试图像之一
plt.figure()         #创建图形

plt.imshow(face)     #绘制测试图像
plt.show()           #显示测试图像

plt.figure()         #创建图形
blurred_face = ndimage.gaussian_filter(face, sigma=7)   #高斯滤波
plt.imshow(blurred_face)     #绘制测试图像
plt.show()           #显示测试图像

plt.figure()         #创建图形
blurred_face1 = ndimage.gaussian_filter(face, sigma=1)   #边缘锐化
blurred_face3 = ndimage.gaussian_filter(face, sigma=3)   #
sharp_face = blurred_face3 + 6*(blurred_face3 - blurred_face1)
plt.imshow(sharp_face)     #绘制测试图像
plt.show()           #显示测试图像

plt.figure()         #创建图形
median_face = ndimage.median_filter(face, 7)   #中值吕滤波
plt.imshow(median_face)     #绘制测试图像
plt.show()           #显示测试图像
