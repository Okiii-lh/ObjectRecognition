# -*- coding:utf-8 -*-
"""
 @Time: 2020/12/12 下午9:08
 @Author: LiuHe
 @File: train_and_eval.py
 @Describe: 训练验证
"""
import sys
import shutil
import random
from datetime import datetime
from image_data_handler_joint_multimodal import ImageDataHandler
from utils import flat_shape, count_params
from imgaug import augmenters as iaa
# from keras.objectives import categorical_crossentropy
# from keras.layers import LSTM, GRU, Dense
# from keras.metrics import categorical_accuracy
# from keras import backend as K
import os
import torch


# 定义路径
dataset_root_dir = "/home/liuh/PycharmProjectsFYQ/mnt/datasets/ocid_dataset"
params_root_dir = "/home/liuh/PycharmProjectsFYQ/mnt/params/models"

dataset_train_dir_rgb = dataset_root_dir + '/ARID20_crops/squared_rgb/'
dataset_val_dir_rgb = dataset_root_dir + '/ARID10_crops/squared_rgb/'
params_dir_rgb = params_root_dir + '/resnet18_ocid_rgb++_params.npy'

dataset_train_dir_depth = dataset_root_dir + '/ARID20_crops/surfnorm++/'
dataset_val_dir_depth = dataset_root_dir + '/ARID10_crops/surfnorm++/'
params_dir_depth = params_root_dir + '/resnet18_ocid_surfnorm++_params.npy'

train_file = dataset_root_dir + \
             '/split_files_and_labels/arid20_clean_sync_instances.txt'
val_file = dataset_root_dir + \
           '/split_files_and_labels/arid10_clean_sync_instances.txt'


# 设置超参数
learning_rate = [[0.0001]]
num_epoch = 50
batch_size = [[32]]
num_neurons = [[100]]
l2_factor = [[0.0]]
maximum_norm = [[4]]
dropout_rate = [[0.4]]

depth_transf = [[256]]
# TODO block块的定义
#transf_block = transformation_block_v1

num_classes = 49
img_size = [224, 224]
num_channels = 3


def data_aug(batch, batch_depth):
    """
    数据集扩充
    :param batch:
    :param batch_depth:
    :return:
    """
    num_img = batch.shape[0]
    list = []
    list_depth = []
    for i in range(num_img):
        # 水平反转
        val_fliplr = random.randrange(0, 2, 1)
        list.extend([iaa.Fliplr(val_fliplr)])
        list_depth.extend([iaa.Fliplr(val_fliplr)])

        # 垂直反转
        val_fliplr = random.randrange(0, 2, 1)
        list.extend([iaa.Fliplr(val_fliplr)])
        list_depth.extend([iaa.Fliplr(val_fliplr)])

        # 仿射变换
        val_scala = random.randrange(5, 11, 1)
        val = float(val_scala/10.0)
        list.extend([iaa.Affine(val, mode='edge')])
        list.extend([iaa.Affine(10.0/val_scala, mode='edge')])
        list_depth.extend([iaa.Affine(val, mode='edge')])
        list_depth.extend([iaa.Affine(10.0/val_scala, mode='edge')])

        # 不断旋转90度
        val_rotation = random.randrange(-180, 181, 90)
        list.extend([iaa.Affine(rotate=val_rotation, mode='edge')])
        list_depth.extend([iaa.Affine(rotate=val_rotation, mode='edge')])

        augseq = iaa.Sequential(list)
        batch[i] = augseq.augment_image(batch[i])
        augseq_depth = iaa.Sequential(list)
        batch_depth[i] = augseq_depth.augment_image(batch_depth[i])

        list = []
        list_depth = []


set_params = [lr+nn+bs+aa+mn+do+dt for lr in learning_rate
              for nn in num_neurons
              for bs in batch_size
              for aa in l2_factor
              for mn in maximum_norm
              for do in dropout_rate
              for dt in depth_transf]
print("222")
for hp in set_params:
    lr = hp[0]
    nn = hp[1]
    bs = hp[2]
    aa = hp[3]
    mn = hp[4]
    do = hp[5]
    dt = hp[6]

    dataset_train_dir = [dataset_train_dir_rgb, dataset_train_dir_depth]
    dataset_val_dir = [dataset_val_dir_rgb, dataset_val_dir_depth]

    0




# # 对数据集进行处理 并生成训练集和验证集
# dataset_train_dir = [dataset_train_dir_rgb, dataset_train_dir_depth]
# dataset_val_dir = [dataset_val_dir_rgb, dataset_val_dir_depth]



