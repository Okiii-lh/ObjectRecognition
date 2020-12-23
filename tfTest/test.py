# -*- coding:utf-8 -*-
"""
 @Time: 2020/12/14 下午8:25
 @Author: LiuHe
 @File: test.py
 @Describe:
"""
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import torch

# img = tf.io.read_file("wallhaven-mdkv1m.jpg")
# print(img)
# img_rgb = tf.image.decode_png(img, channels=3)
# print(img_rgb)

# img_2 = cv2.imread("wallhaven-mdkv1m.jpg")
# print(img_2)
# print(type(img_2))
# print(img_2.shape)
# print(img_2.transpose(2, 0, 1))

img2 = plt.imread("wallhaven-mdkv1m.jpg")
print(img2)
img_rgb = torch.from_numpy(img2)
print(img_rgb)
print(type(256))

img_rgb.unsqueeze_(0)
print(img_rgb)
resize = torch.nn.functional.interpolate(img_rgb, size=(256, 256))

print("======")
print(resize)

