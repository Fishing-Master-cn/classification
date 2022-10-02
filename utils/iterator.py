# -*- coding: utf-8 -*-
# @File  : iterator.py
# @Author: 汪畅
# @Time  : 2022/5/11  18:55
import os
import random
from typing import List, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_confusion_matrix(trues, preds):
    labels = [0, 1, 2]
    conf_matrix = confusion_matrix(trues, preds, labels)
    return conf_matrix


def plot_confusion_matrix(conf_matrix):
    plt.figure(dpi=500)
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = [0, 1, 2]
    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig('heatmap_confusion_matrix.png')
    # plt.show()


def train_one_epoch(model, device, train_loader, criterion, optimizer, idx, verbose=True, mixed=False):
    """
    在train_data_batchloader上完成一轮完整的迭代
    :param model: 网络模型
    :param device: cuda或cpu
    :param train_loader: 训练数据loader
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param idx: 迭代轮数
    :param verbose: 是否打印进度条
    :return: training loss
    """
    model.train()

    # tqdm用于显示进度条
    # loader = tqdm(train_loader)

    tot_loss = 0.0
    tot_acc = 0.0
    train_preds = []
    train_trues = []

    # 用于混合精度训练,可以加快运算速率
    scaler = GradScaler()

    for i, sample in enumerate(train_loader):
        train_data_batch = sample['X'].to(device).double()
        train_label_batch = sample['y'].to(device).double()

        if mixed:
            with autocast():  # 半精度加速训练
                output = model(train_data_batch)
                loss = criterion(output, train_label_batch.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            output = model(train_data_batch)
            loss = criterion(output, train_label_batch.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        tot_loss += loss.data
        train_outputs = output.argmax(dim=1)

        train_preds.extend(train_outputs.detach().cpu().numpy())
        train_trues.extend(train_label_batch.detach().cpu().numpy())

        tot_acc += (output.argmax(dim=1) == train_label_batch).sum().item()

    sklearn_accuracy = accuracy_score(train_trues, train_preds)
    sklearn_precision = precision_score(train_trues, train_preds, average='micro')
    sklearn_recall = recall_score(train_trues, train_preds, average='micro')
    sklearn_f1 = f1_score(train_trues, train_preds, average='micro')

    torch.cuda.empty_cache()
    if not verbose:
        print(
            "[sklearn_metrics] Epoch:{} Lr:{:.8f} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
                idx, optimizer.param_groups[0]['lr'], tot_loss, sklearn_accuracy, sklearn_precision, sklearn_recall,
                sklearn_f1))


# 调用torch.no_grad装饰器，验证阶段不进行梯度计算
@torch.no_grad()
def evaluate(model, device, test_loader, criterion):
    """
    模型评估
    :param model: 网络模型
    :param device: cuda或cpu
    :param test_loader: 测试数据loader
    :param criterion: 损失函数
    :param metric_list: 评估指标列表
    :return: test loss，评估指标
    """
    model.eval()  # 指定是模型evaluate而不是train,BN和DropOut不会取平均值

    test_preds = []
    test_trues = []
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            test_data_batch = sample['X'].to(device).double()
            test_label_batch = sample['y'].to(device).double()
            output = model(test_data_batch)
            test_output = output.argmax(dim=1)
            # print(test_label_batch, test_output)
            test_preds.extend(test_output.detach().cpu().numpy())
            test_trues.extend(test_label_batch.detach().cpu().numpy())
            loss = criterion(output, test_label_batch.long())

        sklearn_accuracy = accuracy_score(test_trues, test_preds)
        sklearn_precision = precision_score(test_trues, test_preds, average='micro')
        sklearn_recall = recall_score(test_trues, test_preds, average='micro')
        sklearn_f1 = f1_score(test_trues, test_preds, average='micro')
        print(classification_report(test_trues, test_preds))
        conf_matrix = get_confusion_matrix(test_trues, test_preds)
        print(conf_matrix)
        plot_confusion_matrix(conf_matrix)
        print("[sklearn_metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(sklearn_accuracy,
                                                                                                  sklearn_precision,
                                                                                                  sklearn_recall,
                                                                                                  sklearn_f1))


def set_random_seed(seed=512, benchmark=True):
    """
    设定训练随机种子
    :param benchmark:
    :param seed: 随机种子
    :return: None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    if not benchmark:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
