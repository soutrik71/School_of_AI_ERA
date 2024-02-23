import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)  # ip: -1,1,28,28  op: -1,32,26,26  rf:3
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # ip: -1,32,26,26  op1: -1,64,24,24 op2: -1,64,12,12  rf:6
        x = F.relu(self.conv3(x), 2) # ip: -1,64,12,12 op : -1,128,10,10 rf: 10
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # ip: -1,128,10,10  op1: -1,256,8,8  op2: -1,256,4,4  rf:16
        x = x.view(-1, 4096) # ip: -1,256,4,4  op: -1, 4*4*256 ie flatten
        x = F.relu(self.fc1(x)) # ip: -1,4096  op: -1,50
        x = self.fc2(x) # ip: -1,50 op: -1,10
        return F.log_softmax(x, dim=1)
