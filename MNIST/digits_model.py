import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image  
    

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1)
        self.average1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1)
        self.average2 = nn.AvgPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 128, kernel_size=4, stride=1)
        
        self.flatten = Flatten()
        
        self.fc1 = nn.Linear(128, 10)
    def forward(self, xb):
        xb = xb.view(-1, 1, 20, 20)
        xb = F.relu(self.conv1(xb))
        xb = self.average1(xb)
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = xb.view(-1, xb.shape[1])
        xb = self.fc1(xb)
#         xb = F.relu(self.fc2(xb))
        return xb
    