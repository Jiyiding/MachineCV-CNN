import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

file_dir = 'gen_inputData/train'
# 创建各样本和标签列表
AoKeng = []
lable_Aokeng = []

LieWen = []
lable_LieWen = []

WeiHantou = []
lable_WeiHantou = []

WeiRonghe = []
lable_WeiRonghe = []

WuQuexian = []
lable_WuQuexian = []

moi_all = []
lable_all = []


# 所有的操作都是对图像名字和标签的字符串，而不是图像存储信息
def get_files(file_dir, ratio):
    # file_dir:图片地址
    # ratio:验证集所占比例
    for file_name in os.listdir(file_dir + '/AoKeng'):
        AoKeng.append(file_dir + '/AoKeng/' + file_name)
        lable_Aokeng.append(0)
    for file_name in os.listdir(file_dir + '/LieWen'):
        LieWen.append(file_dir + '/LieWen/' + file_name)
        lable_LieWen.append(1)
    for file_name in os.listdir(file_dir + '/WeiHantou'):
        WeiHantou.append(file_dir + '/WeiHantou/' + file_name)
        lable_WeiHantou.append(2)
    for file_name in os.listdir(file_dir + '/WeiRonghe'):
        WeiRonghe.append(file_dir + '/WeiRonghe/' + file_name)
        lable_WeiRonghe.append(3)
    for file_name in os.listdir(file_dir + '/WuQuexian'):
        WuQuexian.append(file_dir + '/WuQuexian/' + file_name)
        lable_WuQuexian.append(4)
    # 将各样本和标签列表组合成总的数组形式
    all_moi = np.hstack((AoKeng, LieWen, WeiHantou, WeiRonghe, WuQuexian))
    all_lables = np.hstack((lable_Aokeng, lable_LieWen, lable_WeiHantou, lable_WeiRonghe, lable_WuQuexian))
    # 将两个一维列表转化为对应的一个二维列表
    temp = np.array([all_moi, all_lables])
    # all_train转置并打乱顺序
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 取出样本和标签，并转化为list列表形式
    all_moi_list = list(temp[:, 0])
    all_lables_list = list(temp[:, 1])
    # 分配训练集和测试集比重
    n_moi = len(all_moi_list)
    n_val = int(math.ceil(n_moi * ratio))
    n_train = n_moi - n_val
    tra_moi = all_moi_list[0:n_train]
    tra_lables = all_lables_list[0:n_train]
    # 标签数据转为整数类型 ???
    tra_lables = [int(float(i)) for i in tra_lables]
    val_moi = all_moi_list[n_train:-1]
    val_lables = all_lables_list[n_train:-1]
    val_lables = [int(float(i)) for i in val_lables]
    return tra_moi, tra_lables, val_moi, val_lables


def get_batch(image, lable, image_W, image_H, batch_size, capacity):
    # 转换list中数据类型，但传进来的是文件名及标签所以应该是string,int。可以试一下
    image = tf.cast(image, tf.string)
    lable = tf.cast(lable, tf.int32)
    #print("get_batch.image.type", type(image))

    # 将image、lable列表生成文件名队列,默认生成无限次队列，并打乱顺序
    # 一个tensor生成器，每次从一个tensor列表[image, lable]中随机抽取出一个tensor放入队列。
    input_queue = tf.train.slice_input_producer([image, lable])

    lable = input_queue[1]
    # 从队列中读取图片
    image_contents = tf.read_file(input_queue[0])
    # 将图片解码成RGB三维矩阵类型,解码为tf中的图像格式
    image_contents = tf.image.decode_jpeg(image_contents, channels=3)  # Tensor
    # 图像处理，旋转、缩放、裁剪、归一化等，此步骤暂时不做
    image_contents = tf.image.resize_image_with_crop_or_pad(image_contents, image_W, image_H)
    image_contents = tf.image.per_image_standardization(image_contents)  # 归一化

    # 生成batch,生成文件名队列时已经打乱顺序了，所以不用tf.train.shuffle_batch函数再次打乱顺序。
    # batch_size:设置每次从队列中获取出数据的数量
    # 其中num_threads表示线程数量
    # capacity为队列中最多可以存储的样例个数
    image_batch, lable_batch = tf.train.batch([image_contents, lable],
                                              batch_size=batch_size, num_threads=32, capacity=capacity)
    # （重新排列label，shape为[batch_size)）需要吗？？？
    lable_batch = tf.reshape(lable_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, lable_batch



