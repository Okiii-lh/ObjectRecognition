# -*- coding:utf-8 -*-
"""
 @Time: 2020/12/12 下午9:07
 @Author: LiuHe
 @File: image_data_handler_joint_multimodal.py
 @Describe:  包含用于图像输入管道的帮助器类
"""
import numpy as np
from random import randint, choice
import torch
from torch.utils.data import DataLoader
from torch.nn import functional
from torchvision import transforms
import cv2


class ImageDataHandler(object):
    """
    处理图片输入的类
    """

    def __init__(self, txt_file, data_dir, params_dir, image_size, batch_size,
                 num_classes, num_channels=3, shuffle=False,
                 random_crops=False, buffer_size=1000):
        """
        初始化处理类
        :param txt_file: 数据对应标签文本
        :param data_dir: 数据地址
        :param params_dir: 参数地址 神经网络模型参数
        :param image_size:
        :param batch_size: 字面意思
        :param num_classes: 种类数量
        :param num_channels: 通道数 默认为3
        :param shuffle: 是否打乱样本数据集 默认为False
        :param random_crops: 是否随机裁剪 默认为False
        :param buffer_size: tensorflow Dataset中的方法 可能直接弃用
        """
        self.num_classes = num_classes
        self.img_size = image_size
        self.img_mean_rgb = np.load(params_dir + '/ocid_mean_rgb++.npy')
        self.img_mean_depth = np.load(params_dir + '/ocid_mean_surfnorm++.npy')
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.random_crops = random_crops
        self.img_path_rgb, self.img_paths_depth, self.labels, self.data_size \
            = self._read_text_file(txt_file, data_dir)
        self.data = self._create_dataset()

    def _read_text_file(self, txt_file, data_dir):
        """
        读取文件
        :param txt_file: 数据集标签文本
        :param data_dir: 数据集路径
        :return: 从txt_file返回图像路径和相应标签的列表作为张量
        """
        img_paths = []
        labels = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                img_paths.append(items[0])
                labels.append(int[items[1]])
        # TODO 源代码说shuffling的初始化非常重要 查看下列地址
        # https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
        data_size = len(labels)
        img_paths, labels = self._shuffle_lists(img_paths, labels, data_size)

        img_paths_rgb = []
        img_paths_depth = []

        for path in img_paths:
            img_paths_rgb.append(data_dir[0] + path)
            img_paths_depth.append(data_dir[1] + path)

        # TODO torch 将numpy转为tensor tf有类型标注
        # TODO tf.convert_to_tensor torch.from_numpy
        img_paths_rgb = torch.from_numpy(img_paths_rgb)
        img_paths_depth = torch.from_numpy(img_paths_depth)
        labels = torch.from_numpy(labels)

        return img_paths_rgb, img_paths_depth, labels, data_size

    def _create_dataset(self):
        """
        创建数据集
        :return:
        """
        data = []

        return data

    def _prepare_input(self, filename_rgb, filename_depth, label):
        """
        数据预处理
        :param filename_rgb: 色彩数据
        :param filename_depth: 深度数据
        :param label:  标签
        :return:
        """

        one_hot = functional.one_hot(filename_rgb, num_classes=self.num_classes)

        # 使用opencv代替tensorflow读取图片
        img_rgb = cv2.imread(filename_rgb)
        img_string_rgb = torch.from_numpy(img_rgb)
        img_depth = cv2.imread(filename_depth)
        img_string_depth = torch.from_numpy(img_depth)

        # TODO pytorch踩坑 https://zhuanlan.zhihu.com/p/180020358 要加一维度
        img_string_rgb.unsqueeze_(0)
        img_string_depth.unsqueeze_(0)
        img_resized_rgb = functional.interpolate(img_string_rgb,
                                                 size=(256, 256))
        img_resized_depth = functional.interpolate(img_string_depth,
                                                   size=(256, 256))

        img_bgr_rgb = img_resized_rgb[:, :, ::-1]
        img_bgr_depth = img_resized_depth[:, :, ::-1]

        # TODO tensorflow.substract() 转为 torch.sub
        img_centered_rgb = torch.sub(img_bgr_rgb.float(),
                                     self.img_mean_rgb[:, :, ::-1])
        img_centered_depth = torch.sub(img_bgr_depth.float(),
                                       self.img_mean_depth[:, :, ::-1])
        img_resized_rgb = functional.interpolate(img_centered_rgb,
                                                 self.img_size)
        img_resized_depth = functional.interpolate(img_centered_depth,
                                                   self.img_size)

        """
        下面是数据扩充
        """
        rot_param = choice([0, 1, 2, 3])
        vert_flip = choice([True, False])
        horiz_flip = choice([True, False])
        delta_brightness = choice([-1, +1]) * randint(0, 25)

        # TODO 图像增强 使用pytorch代替tensorflow 随机旋转 可能会出问题
        img_resized_rgb = transforms.RandomRotation()\
            .__call__(img_resized_rgb)
        img_resized_depth = transforms.RandomRotation()\
            .__call__(img_resized_depth)

        adjust = transforms.ColorJitter(brightness=delta_brightness)
        img_resized_rgb = adjust.forward(img_resized_rgb)
        img_resized_depth = adjust.forward(img_resized_depth)

        return img_resized_rgb, img_resized_depth, one_hot

    def _shuffle_lists(self, img_paths, labels, data_size):
        """
        将数据集序列打乱
        :param img_paths: 图片路径
        :param labels: 图片标签
        :param data_size: 数据大小
        :return:
        """
        tmp_img_paths = img_paths
        tmp_labels = labels
        permutation = np.random.permutation(data_size)
        img_paths = []
        labels = []

        for i in permutation:
            img_paths.append(tmp_img_paths[i])
            labels.append(tmp_labels[i])

        return img_paths, labels