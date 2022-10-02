# -*- coding: utf-8 -*-
# @File  : data_pipeline.py
# @Author: 汪畅
# @Time  : 2022/4/21  10:05

import os
import numpy as np
from torch.utils.data import Dataset


class FORCE_Dataset(Dataset):
    def __init__(self, data_dir, label_dir, subject_list, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.subject_list = subject_list
        self.transform = transform
        self.file_list = self.get_file_list()

    def get_file_list(self):
        file_list = []
        for file in os.listdir(self.data_dir):
            person_id = file.split('_')[1]  # person_id为1.xls格式
            if person_id[:-4] in self.subject_list:  # 如果标号在训练集标号/验证集标号里就添加，真正意义上的分割数据集
                file_list.append(file)
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        data_filename = os.path.join(self.data_dir, self.file_list[i])
        label_filename = os.path.join(self.label_dir, self.file_list[i])

        X, y = data_loader(data_filename,
                           label_filename,
                           )

        sample = {'X': X, 'y': y}
        if self.transform:
            sample = self.transform(sample)

        return sample


def Loader_class_label(root):
    if "JUMP" in root:
        label = 0
    elif "RUN" in root:
        label = 1
    else:
        label = 2
    return label


# 文件读取主程序
def data_loader(X_root, y_root):
    """
    :param X_root: 输入特征的路径
    :param y_root: 标签的路径
    :return: 输入 输出特征
    """

    X_data = np.load(X_root)
    X_data = np.transpose(X_data, (2, 0, 1))

    y_data = Loader_class_label(y_root)

    return X_data, y_data
