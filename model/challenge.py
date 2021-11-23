"""
EECS 445 - Introduction to Machine Learning
Fall 2021 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Inception(nn.Module):
    def __init__(self, in_channel, ch1, ch2, ch3):
        super(Inception, self).__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channel, ch1, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channel, ch2[0], kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(ch2[0], ch2[1], kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(ch2[1], ch2[2], kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        self.path3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channel, ch3, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.change_channel = nn.Conv2d(in_channel, ch1 + ch2[2] + ch3, kernel_size=(1, 1))

    def forward(self, x):
        residual = x
        residual = self.change_channel(residual)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out4 = torch.cat((out1, out2, out3), dim=1)
        out4 += residual
        out = F.relu(out4)
        return out


class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer of your network
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            Inception(16, 8, (8, 12, 16), 8),
            nn.Dropout(p=0.3),
            Inception(32, 16, (16, 24, 32), 16),
            nn.Dropout(p=0.3)
        )
        self.layer3 = nn.Sequential(
            Inception(64, 16, (32, 48, 64), 16),
            Inception(96, 16, (32, 48, 64), 16),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            Inception(96, 32, (64, 96, 128), 32),
            Inception(192, 32, (64, 96, 192), 32),
        )
        self.fc1 = nn.Linear(256, 2)

        self.init_weights()

    def init_weights(self):
        ## TODO: initialize the parameters for your network
        fc_in = self.fc1.weight.size(1)
        nn.init.normal_(self.fc1.weight, 0.0, 1 / sqrt(fc_in))
        nn.init.constant_(self.fc1.bias, 0.0)
        ##

    def forward(self, x):
        N, C, H, W = x.shape

        ## TODO: forward pass
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        x = F.adaptive_avg_pool2d(out3, (1, 1))
        z = self.fc1(x.view(x.size()[0], -1))
        ##
        return z
