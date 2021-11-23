"""
EECS 445 - Introduction to Machine Learning
Fall 2021 - Project 2
Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import target
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Target(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5),
                               stride=(2, 2), padding=(2, 2))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5),
                               stride=(2, 2), padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(5, 5),
                               stride=(2, 2), padding=(2, 2))
        self.fc_2 = nn.Linear(in_features=32, out_features=2)
        ##

        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc_1]
        fc_in = self.fc_2.weight.size(1)
        nn.init.normal_(self.fc_2.weight, 0.0, 1 / sqrt(fc_in))
        nn.init.constant_(self.fc_2.bias, 0.0)
        ##

    def forward(self, x):
        N, C, H, W = x.shape

        ## TODO: forward pass

        r1 = F.relu(self.conv1(x))
        r2 = self.pool(r1)
        r3 = F.relu(self.conv2(r2))
        r4 = self.pool(r3)
        r5 = F.relu(self.conv3(r4))
        z = self.fc_2(r5.view(r5.size()[0], -1))
        ##

        return z
