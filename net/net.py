# -*- coding: utf-8 -*-
# @File  : net.py
# @Author: 汪畅
# @Time  : 2022/5/11  18:29
from torch import nn
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.output = nn.Linear(in_features=64 * 25 * 25, out_features=3)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # [batch, 32,7,7]
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
        output = self.output(x)  # 输出[batch,10]
        output = self.softmax(output)
        return output


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 3, 1, 1),  # 3*100*100 -> 6*100*100
            nn.ReLU(),
            nn.MaxPool2d(2),  # 6*100*100 -> 6*50*50
            nn.Conv2d(6, 16, 3, 1, 1),  # 6*50*50 -> 16*50*50
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16*50*50 -> 16*25*25
            nn.Conv2d(16, 32, 6, ),  # 16*25*25 -> 32*20*20
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32*20*20 -> 32*10*10
            nn.Conv2d(32, 64, 3),  # 32*10*10 -> 64*8*8
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64*8*8 -> 64*4*4
            nn.Conv2d(64, 120, 4),  # 32*10*10 -> 120*1*1
            )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Linear(84, 3),
            nn.Softmax(dim=-1)
            )
        # init
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
