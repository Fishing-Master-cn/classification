# -*- coding: utf-8 -*-
# @File  : preprocessing.py
# @Author: 汪畅
# @Time  : 2022/4/21  19:22
# 包括数据读取程序和预处理

import matplotlib.pyplot as plt
import xlrd
import torch
import numpy as np
import os
import math
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)


# 读取力.xls 文件
def __DataLoader_of_forces(root, features):
    X_excel = xlrd.open_workbook(root)
    sheet = X_excel.sheet_by_index(0)  # 获取工作薄
    if features[0] != 'time':
        features.insert(0, 'time')
    X = []
    for index in features:
        index_name = features.index(index)
        X.append(sheet.col_values(index_name))
    X = np.array(X).transpose(1, 0)
    index = 0
    for i in X[:, 0]:
        if not __is_number(i):
            index += 1
        else:
            break
    X = X[index:, :]
    X = X.astype(float)
    return X


# 读取mark点轨迹 .xls 文件
def __DataLoader_of_marks(root, features):
    X_excel = xlrd.open_workbook(root)
    sheet = X_excel.sheet_by_index(0)  # 获取工作薄

    if features[0] != 'time':
        features.insert(0, 'time')
    X = []
    for index in features:
        index_name = features.index(index)
        if index_name == 0:
            X.append(sheet.col_values(index_name))
        elif index_name != 0:
            for i in range(3):
                X.append(sheet.col_values((index_name - 1) * 3 + i + 1))
    X = np.array(X).transpose(1, 0)
    index = 0
    for i in X[:, 0]:
        if not __is_number(i):
            index += 1
        else:
            break
    X = X[index:, :]
    X = X.astype(float)
    return X


# 画图
def __draw(X, X_index_name, picture_name, is_draw, is_save=True, pic_number=0):
    x_arrow_size = 5
    y_arrow_size = int(X.shape[1] / x_arrow_size + 1)
    plt.figure(figsize=(50, 32))
    print(X.shape)
    for i in range(1, X.shape[1]):
        plt.subplot(x_arrow_size, y_arrow_size, i)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.9, hspace=1)
        plt.title("index {}: ".format(i) + X_index_name[i])
        plt.plot(X[:, 0], X[:, i])
    if is_save:
        plt.savefig(os.path.join(r"D:\force_predict\plt_result", picture_name + f"{pic_number}"))
    if is_draw:
        plt.show()
    else:
        plt.close()


# 取一个周期
def __get_a_period(X, y, t_0, T):
    """
    :param t_0: 周期开始时间 T: 周期长度
    """
    X, y = X[X[:, 0] < t_0 + T, :], y[y[:, 0] < t_0 + T, :]
    return X[t_0 < X[:, 0], :], y[t_0 < y[:, 0], :]


def __is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False


# 取时刻对应的输入特征和输出特征为一组
def __to_same_size(X, y):
    y_ = []
    for x_index in range(X.shape[0]):
        first = x_index
        for y_index in range(first, y.shape[0]):
            if math.fabs(X[x_index, 0] - y[y_index, 0]) < 1e-3:
                y_.append(y[y_index, :].tolist())
                break
    # TODO: 有些数据存在最后一个值多余的情况，按情况取消注释
    # X = X[:-1, :]
    y = np.array(y_)[:, :]
    return X, y


# 数据预处理程序
def __Compose(X_data, y_data):
    X_data = X_data.reshape(X_data.shape[0], -1, 3)
    y_data = y_data.reshape(y_data.shape[0], -1, 3)
    X_data = trans_square_1(X_data)
    y_data = trans_square_1(y_data)

    # plt.imshow(X_data)
    # plt.show()
    # plt.imshow(y_data)
    # plt.show()

    return X_data.transpose(2, 0, 1), y_data.transpose(2, 0, 1)


def trans_square_1(img):
    """
    图片转正方形，边缘使用0填充
    :param img: np.ndarray
    :return: np.ndarray
    """
    img_h, img_w, img_c = img.shape
    if img_h != img_w:
        long_side = max(img_w, img_h)
        short_side = min(img_w, img_h)
        loc = abs(img_w - img_h) // 2
        img = img.transpose((1, 0, 2)) if img_w < img_h else img  # 如果高是长边则换轴，最后再换回来
        background = np.zeros((long_side, long_side, img_c), dtype=np.uint8)  # 创建正方形背景
        background[loc: loc + short_side] = img[...]  # 数据填充在中间位置
        img = background.transpose((1, 0, 2)) if img_w < img_h else background
    return img


def trans_square_2(image):
    r"""transform square.
    :return PIL image
    """
    img = transforms.ToTensor()(image)
    C, H, W = img.shape
    pad_1 = int(abs(H - W) // 2)  # 一侧填充长度
    pad_2 = int(abs(H - W) - pad_1)  # 另一侧填充长度
    img = img.unsqueeze(0)  # 加轴
    if H > W:
        img = nn.ZeroPad2d((pad_1, pad_2, 0, 0))(img)  # 左右填充，填充值是0
        # img = nn.ConstantPad2d((pad_1, pad_2, 0, 0), 127)(img)  # 左右填充，填充值是127
    elif H < W:
        img = nn.ZeroPad2d((0, 0, pad_1, pad_2))(img)  # 上下填充，填充值是0
        # img = nn.ConstantPad2d((0, 0, pad_1, pad_2), 127)(img)  # 上下填充，填充值是127
    img = img.squeeze(0)  # 减轴
    img = transforms.ToPILImage()(img)
    return img


# 文件读取主程序
def data_loader(X_root, y_root, input_feature, output_feature, picture_name="input_data", is_draw=False, is_save=False,
                is_get_a_period=False, T=None,
                t_0=None):
    """
    :param X_root: 输入特征的路径
    :param y_root: 标签的路径
    :param input_feature: 输入特征的索引
    :param output_feature: 输出特征的索引
    :param picture_name: 输入特征-时间 图像的命名
    :param is_draw: 是否画输入特征-时间 图像
    :param is_save: 是否保存输入特征-时间 图像
    :param is_get_a_period: 是否取数据的一个周期
    :param T: 周期大小
    :param t_0: 周期开始时间
    :return: 输入 输出特征
    """
    if is_save is True:
        assert picture_name is not None, "please set picture_name."

    if is_get_a_period is True:
        assert T is not None, "please set the T."

    # 为了方便获得某个周期的数据，保留了时间特征
    X_data = __DataLoader_of_marks(X_root, input_feature)
    y_data = __DataLoader_of_forces(y_root, output_feature)

    # 获取一个周期
    if is_get_a_period:
        if not isinstance(t_0, float):
            t_0 = np.random.rand()
        X_data, y_data = __get_a_period(X_data, y_data, t_0, T)

    X_data, y_data = __to_same_size(X_data, y_data)

    # 是否画图
    if is_draw or is_save:
        pic_number = 0
        __draw(X_data, input_feature, picture_name=picture_name, is_draw=is_draw, is_save=is_save,
               pic_number=pic_number)
        pic_number = 1
        __draw(y_data, output_feature, picture_name=picture_name, is_draw=is_draw, is_save=is_save,
               pic_number=pic_number)

    # 格式转变&去掉时间特征
    X_data = X_data[:, 1:]
    y_data = y_data[:, 1:]

    # TODO: 是否使用三维化
    X_data, y_data = __Compose(X_data, y_data)

    return X_data, y_data
