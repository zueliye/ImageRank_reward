import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 2.构建网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,stride=1,padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 5, 3,1,1)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2560, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        # print('x.shape',x.shape)     #64,2560
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features











