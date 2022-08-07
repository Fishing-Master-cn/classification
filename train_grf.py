# -*- coding: utf-8 -*-
# @File  : preprocessing.py
# @Author: 汪畅
# @Time  : 2022/4/21  19:22

from loader import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.iterator import *
from net.net import *
import os
import yaml
import tqdm
import re


def main():
    torch.set_default_tensor_type(torch.DoubleTensor)

    model_save_dir = "./checkpoints"  # 模型保存路径
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    data_dir = r"grf_force_pre_data/input_feature"  # 输入特征读取路径
    label_dir = r"grf_force_pre_data/output_feature"  # 输出特征读取路径

    model_list = {"Neural": NeuralNetwork, "Lstm": LstmRNN, "FCNs": FCNs}

    # 定义相关参数
    with open('./config.cfg', 'r') as f:
        cfg = yaml.safe_load(f)
        print("successfully loaded config file: ", cfg)

    epochs = cfg['TRAIN']['EPOCHS']  # 训练轮数
    model_name = cfg['MODEL']['TYPE']  # 使用网络名称
    batch_size = cfg['TRAIN']['BATCH_SIZE']  # 每批次训练个数
    is_mixed = cfg['TRAIN']['IS_MIXED']  # 是否使用混合精度训练
    interval = cfg['TRAIN']['INTERVALS']  # 验证的间隔数
    seed = cfg['TRAIN']['SEED']  # 随机种子
    data_seed = cfg['TRAIN']['DATASEED']
    decay = cfg['TRAIN']['DECAY']  # 学习率迭代参数
    lr = cfg['TRAIN']['LR']  # 学习率
    new = cfg['TRAIN']['NEW']  # 是否重新训练网络
    model_path = cfg['TRAIN']['MODEL_PATH']  # 模型读取路径

    # 设置随机数
    set_random_seed(seed=seed, benchmark=True)

    # 定义运行设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # 分割数据集
    subject_list = [re.findall('(\d+)', file)[0] for file in os.listdir('grf_force_pre_data/input_feature')]
    train_list, val_list = train_test_split(subject_list, test_size=0.2, random_state=data_seed)

    print(train_list, val_list)
    # 加载数据
    train_set = FORCE_Dataset(data_dir, label_dir, train_list)
    val_set = FORCE_Dataset(data_dir, label_dir, val_list)

    # 获得输入、输出大小
    input_dim = 3
    hidden_dim = 256
    output_dim = 3

    trainDataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valDataLoader = DataLoader(val_set, batch_size=1, shuffle=False)

    # 初始化模型
    model = model_list[model_name](output_dim)
    print(model)
    model.to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': 1e-3}], lr=lr, weight_decay=decay)
    criterion = torch.nn.MSELoss(reduction='mean')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[epochs // 3, 2 * epochs // 3],
                                                     gamma=0.1,
                                                     last_epoch=0)  # 学习率动态变化，随epoch变化到milestones的值时，lr变为gamma倍

    if not new:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(epochs):
        train_one_epoch(model, device, trainDataLoader, criterion, optimizer, epoch + 1, True, is_mixed)
        scheduler.step()
        if (epoch + 1) % interval == 0:
            # 执行验证
            evaluate(model, device, valDataLoader, criterion)
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": epoch + 1
                }
            # 保存训练checkpoint
            torch.save(checkpoint, os.path.join(model_save_dir, '{}_checkpoint_{}.pth'.format(model_name, epoch + 1)))
        # 最终checkpoint
        final_checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            "epoch": epochs
            }
        torch.save(final_checkpoint, os.path.join(model_save_dir, '{}_checkpoint_final.pth'.format(model_name)))


if __name__ == '__main__':
    main()
