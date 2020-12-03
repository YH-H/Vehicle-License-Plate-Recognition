# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy.linalg import norm

MAX_WIDTH = 1000
Min_Area = 2000
SZ = 20
PROVINCE_START = 1000

# 使用方向梯度直方图Histogram of Oriented Gradients （HOG）作为特征向量

#对训练图片（灰度图）进行抗扭曲处理，摆正
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    # 图像的平移，参数:输入图像、变换矩阵、变换后的大小
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

# 来自opencv的sample，用于svm训练
#用于从图片中抽取特征向量
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)#笛卡尔坐标转极坐标
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel  特征数据的归一化
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)
"""
计算图像 X 方向和 Y 方向的 Sobel 导数
计算得到每个像素的梯度角度angle和梯度大小magnitude
把这个梯度的角度转换成 0至16 之间的整数
将图像分为4个小的方块，对每一个小方块计算它们梯度角度的直方图histogram（16个 bin），使用梯度的大小做权重。
每一个小方块都会得到一个含有16个值的向量，4 个小方块的4个向量就组成了这个图像的特征向量（包含64个值）。
"""

provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "青",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]



