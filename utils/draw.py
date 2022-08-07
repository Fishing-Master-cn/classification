# -*- coding: utf-8 -*-
# @File  : draw.py
# @Author: 汪畅
# @Time  : 2022/5/25  13:05
import xlrd
import numpy as np
import matplotlib.pyplot as plt


def __DataLoader(root):
    X_excel = xlrd.open_workbook(root)
    sheet = X_excel.sheet_by_index(0)  # 获取工作薄

    col: list = sheet.col_values(0)  # 获取第一行的表头内容
    X = []
    for index in col:
        index_name = col.index(index)
        X.append(sheet.row_values(index_name))
    index_name = sheet.row_values(0)
    X = np.array(X)
    X = X[1:, :]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] == "       -nan(ind)":
                X[i, j] = 0

    X = X.astype(float)
    return X, index_name


root = "../inside_force_pre_data/inverse_dynamics.xls"
X, name = __DataLoader(root)

print(X, name)

plt.plot(X[:, 0], X[:, 19],label=name[19])
plt.xlabel("time")
plt.ylabel("knee_moment")
plt.legend()
plt.show()
