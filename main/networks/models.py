import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from networks.layers import conv_block, convT_block, linear_block, UPSoftmax


class disc(nn.Module):
    # Conv2D(in_channels, out_channels, kernel size, stride, padding)

    # conv1 (1, 150, 150)   ->  (16, 76, 76)
    # conv2 (16, 76, 76)    ->  (32, 38, 38)
    # conv3 (32, 38, 40)    ->  (64, 20, 20)
    # conv4 (64, 20, 20)    ->  (128, 10, 10)
    # conv5 (128, 10, 10)   ->  (256, 5, 5)

    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(1, n_df, 4, 2, 2, batchnorm=False)

        self.conv2 = conv_block(n_df, n_df * 2, 4, 2, 1)

        self.conv3 = conv_block(n_df * 2, n_df * 4, 4, 2, 2)

        self.conv4 = conv_block(n_df * 4, n_df * 8, 4, 2, 1)

        self.conv5 = conv_block(n_df * 8, n_df * 16, 4, 2, 1)

        self.linear1 = linear_block(n_df * 16 * 5 * 5, n_df * 16 * 5)

        self.linear2 = linear_block(n_df * 16 * 5, n_df * 16)

        self.linear3 = linear_block(n_df * 16, 2, activation=None, dropout=False)

    def forward(self, x, logit=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, n_df * 16 * 5 * 5)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        logits = x
        y = F.softmax(x, dim=1)
        if logit:
            return logits
        else:
            return logits, y


class Tang(nn.Module):
    def __init__(self):
        super(Tang, self).__init__()
        # Conv2D(in_channels, out_channels, kernel size, stride, padding)

        # conv1 (1, 28, 28)  ->  (32, 14, 14)
        # conv2 (32, 14, 14) ->  (64, 7, 7)
        # conv3 (64, 7, 7) ->  (128, 4, 4)
        # conv4 (128, 4, 4)   ->  (1, 1, 1 )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 11, 1, 1), nn.ReLU(), nn.MaxPool2d(2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 24, 3, 1, 1), nn.BatchNorm2d(24), nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1), nn.BatchNorm2d(24), nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(24, 16, 3, 1, 1),
            nn.BatchNorm2d(
                16,
            ),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=4),
        )

        # 8192 -> 2048
        # 2048 -> 512
        # 512  -> 512
        # 512  -> 3
        self.linear1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.linear2 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, logit=False, features=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 256)
        f = x
        x = self.linear1(x)
        x = self.linear2(x)
        if logit:
            return x
        elif features:
            return f
        else:
            return x, self.softmax(x)
