# -*- coding:utf-8 -*-
"""
 @Time: 2020/12/16 上午10:18
 @Author: LiuHe
 @File: net_structure.py
 @Describe: tensorflow resnet18网络结构
"""
import tensorflow as tf
from tfTest.layer_blocks import *
from tfTest.resnet18 import ResNet


tf.set_random_seed(7)


print(tf.__version__)

img_size = [224, 224]
num_classes = 49

with tf.Session() as sess:
    x_rgb = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
    y = tf.placeholder(tf.float32, [None, num_classes])

    x_rgb = tf.reshape(x_rgb, [-1, img_size[0], img_size[1], 3])
    y = tf.reshape(y, [-1, num_classes])

    model_rgb = ResNet(x_rgb, num_classes, mode='rgb')

    summary_writer = tf.summary.FileWriter('./log/', sess.graph)