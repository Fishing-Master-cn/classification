# -*- coding: utf-8 -*-
# @File  : loader.py
# @Author: 汪畅
# @Time  : 2022/4/21  10:05

from torch.utils.data import Dataset
from utils.preprocessing import *

input_feature = ['RFHD', 'RBHD', 'C7',
                 'T10', 'CLAV', 'RBAK',
                 'LSHO', 'LUPA', 'LELB',
                 'LFRM', 'LWRA', 'LWRB',
                 'LFIN', 'RSHO', 'RUPA',
                 'RELB', 'RFRM', 'RWRA',
                 'RWRB', 'RFIN', 'LASI',
                 'RASI', 'LPSI', 'RPSI',
                 'LTHI', 'LKNE', 'LTIB',
                 'LANK', 'LHEE', 'LTOE',
                 'RTHI', 'RKNE', 'RTIB',
                 'RANK', 'RHEE', 'RTOE',
                 'LFHD', 'LBHD', 'STRN']
'''
input_feature = ['RFHD', 'RBHD', 'C7',
                 'T10', 'CLAV', 'RBAK',
                 'LSHO', 'LUPA', 'LELB',
                 'LFRM', 'LWRA', 'LWRB',
                 'LFIN', 'RSHO', 'RUPA',
                 'RELB', 'RFRM', 'RWRA',
                 'RWRB', 'RFIN', 'LASI',
                 'RASI', 'LPSI', 'RPSI',
                 'LTHI', 'LKNE', 'LTIB',
                 'LANK', 'LHEE', 'LTOE',
                 'RTHI', 'RKNE', 'RTIB',
                 'RANK', 'RHEE', 'RTOE',
                 'LFHD', 'LBHD', 'STRN']
'''  # 输入参考

# 定义输出特征的索引
output_feature = ['ground_force_vx', 'ground_force_vy', 'ground_force_vz',
                  'ground_force_px', 'ground_force_py', 'ground_force_pz',
                  '1_ground_force_vx', '1_ground_force_vy', '1_ground_force_vz',
                  '1_ground_force_px', '1_ground_force_py', '1_ground_force_pz',
                  'ground_torque_x', 'ground_torque_y', 'ground_torque_z',
                  '1_ground_torque_x', '1_ground_torque_y', '1_ground_torque_z']
'''
 output_feature = ['ground_force_vx', 'ground_force_vy', 'ground_force_vz',
                  'ground_force_px', 'ground_force_py', 'ground_force_pz',
                  '1_ground_force_vx', '1_ground_force_vy', '1_ground_force_vz',
                  '1_ground_force_px', '1_ground_force_py', '1_ground_force_pz',
                  'ground_torque_x', 'ground_torque_y', 'ground_torque_z',
                  '1_ground_torque_x', '1_ground_torque_y', '1_ground_torque_z']
'''  # 参考输出


class FORCE_Dataset(Dataset):
    def __init__(self, data_dir, label_dir, subject_list):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.subject_list = subject_list
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
                           input_feature=input_feature,
                           output_feature=output_feature,
                           is_get_a_period=True, T=1.2, t_0=0)

        sample = {'X': X, 'y': y}

        return sample
