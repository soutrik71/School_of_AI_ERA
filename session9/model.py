import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import (ExponentialLR, OneCycleLR,
                                      ReduceLROnPlateau, StepLR)
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
from torchsummary import summary
from torchview import draw_graph
from tqdm import tqdm


class DialatedClassificationTransition(nn.Module):
    """This class is used to create a model with dialated convolution and transition block for CIFAR10 dataset"""

    def __init__(
        self,
        in_channels,
        hidden_units,
        out_channels,
        multiplier_1=1,
        multiplier_2=2,
        dropout=0.1,
    ):
        super(DialatedClassificationTransition, self).__init__()
        # block1
        self.conv_bl1 = nn.Sequential(
            # set1
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=multiplier_1 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(multiplier_1 * hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            # set2
            nn.Conv2d(
                in_channels=multiplier_1 * hidden_units,
                out_channels=multiplier_2 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(multiplier_2 * hidden_units),
            nn.Dropout(dropout),
            # set3 -dialation
            nn.Conv2d(
                in_channels=multiplier_2 * hidden_units,
                out_channels=multiplier_2 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                dilation=2,  # application of dialated convolution
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(multiplier_2 * hidden_units),
            nn.Dropout(dropout),
        )
        # transition block
        self.transition = nn.Sequential(
            nn.Conv2d(
                in_channels=multiplier_2 * hidden_units,
                out_channels=hidden_units,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
        )
        # block2
        self.conv_bl2 = nn.Sequential(
            # set1
            nn.Conv2d(
                in_channels=(hidden_units),
                out_channels=multiplier_1 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(multiplier_1 * hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            # set2
            nn.Conv2d(
                in_channels=multiplier_1 * hidden_units,
                out_channels=multiplier_2 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(multiplier_2 * hidden_units),
            nn.Dropout(dropout),
            # set3 -dialation
            nn.Conv2d(
                in_channels=multiplier_2 * hidden_units,
                out_channels=multiplier_2 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                dilation=2,  # application of dialated convolution
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(multiplier_2 * hidden_units),
            nn.Dropout(dropout),
        )
        # block3
        self.conv_bl3 = nn.Sequential(
            # set1
            nn.Conv2d(
                in_channels=(hidden_units),
                out_channels=multiplier_1 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(multiplier_1 * hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            # set2
            nn.Conv2d(
                in_channels=multiplier_1 * hidden_units,
                out_channels=multiplier_2 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(multiplier_2 * hidden_units),
            nn.Dropout(dropout),
            # set3 -dialation
            nn.Conv2d(
                in_channels=multiplier_2 * hidden_units,
                out_channels=multiplier_2 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                dilation=2,  # application of dialated convolution
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(multiplier_2 * hidden_units),
            nn.Dropout(dropout),
        )
        # block4
        self.conv_bl4 = nn.Sequential(
            # set1
            nn.Conv2d(
                in_channels=(hidden_units),
                out_channels=multiplier_1 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(multiplier_1 * hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            # set2
            nn.Conv2d(
                in_channels=multiplier_1 * hidden_units,
                out_channels=multiplier_2 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(multiplier_2 * hidden_units),
            nn.Dropout(dropout),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=multiplier_2 * hidden_units,
                out_channels=out_channels,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            )
        )

    def forward(self, x):
        x = self.conv_bl1(x)
        x = self.transition(x)
        x = self.conv_bl2(x)
        x = self.transition(x)
        x = self.conv_bl3(x)
        x = self.transition(x)
        x = self.conv_bl4(x)
        x = self.gap(x)
        x = self.classifier(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class DepthwiseSeparableClassifierSync(nn.Module):
    """This class is used to create a model with depthwise separable convolution and transition block for CIFAR10 dataset"""
    def __init__(
        self,
        in_channels,
        hidden_units,
        out_channels,
        dropout=0.1,
    ):
        super(DepthwiseSeparableClassifierSync, self).__init__()
        # block1
        self.conv_bl1 = nn.Sequential(
            # set1
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(2 * hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            # set2
            nn.Conv2d(
                in_channels=2 * hidden_units,
                out_channels=4 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(4 * hidden_units),
            nn.Dropout(dropout),
            # set3 -dialation
            nn.Conv2d(
                in_channels=4 * hidden_units,
                out_channels=4 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                dilation=2,  # application of dialated convolution
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(4 * hidden_units),
            nn.Dropout(dropout),
        )
        # transition block
        self.transition = nn.Sequential(
            nn.Conv2d(
                in_channels=4 * hidden_units,
                out_channels=hidden_units,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
        )
        # block2
        self.conv_bl2 = nn.Sequential(
            # set1
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=2 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(2 * hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            # set2
            nn.Conv2d(
                in_channels=2 * hidden_units,
                out_channels=4 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(4 * hidden_units),
            nn.Dropout(dropout),
            # set3 -dialation
            nn.Conv2d(
                in_channels=4 * hidden_units,
                out_channels=4 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                dilation=2,  # application of dialated convolution
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(4 * hidden_units),
            nn.Dropout(dropout),
        )
        # block3
        self.conv_bl3 = nn.Sequential(
            # set1
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=2 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(2 * hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            # set2
            nn.Conv2d(
                in_channels=2 * hidden_units,
                out_channels=4 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(4 * hidden_units),
            nn.Dropout(dropout),
            # set3 -dialation
            nn.Conv2d(
                in_channels=4 * hidden_units,
                out_channels=4 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                dilation=2,  # application of dialated convolution
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(4 * hidden_units),
            nn.Dropout(dropout),
        )
        # block4 - Depthwise Separable Convolution **
        self.conv_bl4 = nn.Sequential(
            # set1
            # depthwise
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=2 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
                groups=hidden_units,
            ),
            # pointwise
            nn.Conv2d(
                in_channels=2 * hidden_units,
                out_channels=2 * hidden_units,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(2 * hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            # set2
            # depthwise
            nn.Conv2d(
                in_channels=(2 * hidden_units),
                out_channels=4 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
                groups=2 * hidden_units,
            ),
            # pointwise
            nn.Conv2d(
                in_channels=4 * hidden_units,
                out_channels=4 * hidden_units,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(4 * hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            # set3 -non-dialation - depthwise
            # depthwise
            nn.Conv2d(
                in_channels=(4 * hidden_units),
                out_channels=4 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
                groups=4 * hidden_units,
            ),
            # pointwise
            nn.Conv2d(
                in_channels=4 * hidden_units,
                out_channels=4 * hidden_units,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(4 * hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            # set4- expansion
            # depthwise
            nn.Conv2d(
                in_channels=(4 * hidden_units),
                out_channels=8 * hidden_units,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
                groups=4 * hidden_units,
            ),
            # pointwise
            nn.Conv2d(
                in_channels=8 * hidden_units,
                out_channels=8 * hidden_units,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(8 * hidden_units),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=8 * hidden_units,
                out_channels=out_channels,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            )
        )

    def forward(self, x):
        x = self.conv_bl1(x)
        x = self.transition(x)
        x = self.conv_bl2(x)
        x = self.transition(x)
        x = self.conv_bl3(x)
        x = self.transition(x)
        x = self.conv_bl4(x)
        x = self.gap(x)
        x = self.classifier(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
