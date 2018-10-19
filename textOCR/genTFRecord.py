# 将图片保存成 TFRecord
import tensorflow as tf
import numpy as np
import os
from PIL import Image

savedir = "data.tfrecords"  # 希望在data/文件夹中生成“data.tfrecords"的TFRecord格式文件


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load(filedir):
    fileLists = []
    # 读取所有文件
    for root, dirs, files in os.walk(filedir):
        print(files)  # 当前路径下所有非目录子文件
        print(len(files))  # 当前路径下所有非目录子文件
        fileLists = files

    # 准备一个 writer 用来写 TFRecord 文件
    writer = tf.python_io.TFRecordWriter(savedir)

    with tf.Session() as sess:
        for line in fileLists:  # open 打开的文件返回对象是一个可迭代对象，直接用 for 迭代访问
            # 获得图片的路径和类型
            if (line == '.DS_Store'):
                continue
            tmp = line.strip().split('.')  # str.strip([chars])用于去除头尾的字符chars,为空时默认删除空白符；
            # str.split(' ')通过指定一个空格对字符串进行切片，返回分割后的字符串列表tmp
            imgpath = filedir + '/' + line  # 字符串列表tmp中tmp[0]代表该图像的路径
            label = tmp[0]  # 字符串列表tmp中tmp[0]代表该图像的标签

            # 读取图片
            img = Image.open(imgpath)
            # 解码图片（如果是 png 格式就使用tf.image.decode_png)）

            # 将其图片矩阵转换成string
            img_raw = img.tobytes()  # 将图片转化为二进制格式

            # 将数据整理成 TFRecord 需要的数据结构
            example = tf.train.Example(features=tf.train.Features(feature= \
                                                                      {'img_raw': _bytes_feature(img_raw),
                                                                       'label': _bytes_feature(label.encode())}))

            # 写 TFRecord
            writer.write(
                example.SerializeToString())  # SerializeToString()作用:把example序列化为一个字符串,因为在写入到TFRcorde的时候,write方法的参数是字符串的.
    writer.close()


if __name__ == '__main__':
    load("data")
