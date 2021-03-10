import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class CheckersNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(CheckersNNet, self).__init__()
        self.conv1 = nn.Conv2d(5, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)
        self.bn5 = nn.BatchNorm2d(args.num_channels)
        self.bn6 = nn.BatchNorm2d(args.num_channels)
        self.bn7 = nn.BatchNorm2d(args.num_channels)
        self.bn8 = nn.BatchNorm2d(args.num_channels)
        self.bn9 = nn.BatchNorm2d(args.num_channels)
        self.bn10 = nn.BatchNorm2d(args.num_channels)
        self.bn11 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x)*(self.board_y), 512)
        self.fc_bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(args.num_channels*(self.board_x)*(self.board_y), 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        # Rectified batch normalized conv. layer
        s = s.view(-1, 5, self.board_x, self.board_y)                           # batch_size x 5 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                                     # batch_size x num_channels x board_x x board_y

        # Residual blocks (4, instead of 19)
        s = s + F.relu(self.bn3(self.conv3(F.relu(self.bn2(self.conv2(s))))))   # batch_size x num_channels x board_x x board_y
        s = s + F.relu(self.bn5(self.conv5(F.relu(self.bn4(self.conv4(s))))))   # batch_size x num_channels x board_x x board_y
        s = s + F.relu(self.bn7(self.conv6(F.relu(self.bn6(self.conv6(s))))))   # batch_size x num_channels x board_x x board_y
        s = s + F.relu(self.bn9(self.conv9(F.relu(self.bn8(self.conv8(s))))))   # batch_size x num_channels x board_x x board_y

        # Policy head
        pi = F.relu(self.bn10(self.conv10(s)))
        pi = pi.view(-1, self.args.num_channels*(self.board_x)*(self.board_y))
        pi = F.dropout(F.relu(self.fc_bn1(self.fc1(pi))), p=self.args.dropout, training=self.training)  # batch_size x 512
        pi = self.fc3(pi)                                                                               # batch_size x action_size

        # Value head
        v = F.relu(self.bn11(self.conv11(s)))
        v = v.view(-1, self.args.num_channels*(self.board_x)*(self.board_y))
        v = F.dropout(F.relu(self.fc_bn2(self.fc2(v))), p=self.args.dropout, training=self.training)    # batch_size x 512
        v = self.fc4(v)                                                                                 # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
