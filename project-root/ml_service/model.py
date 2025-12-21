import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # 64 * 5 * 5

        self.fc1 = nn.Linear(1600, 800)
        self.fc2 = nn.Linear(800, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 28*28
        x = F.max_pool2d(x, 2) # 14 * 14
        x = F.dropout(x, 0.25)

        x = F.relu(self.conv2(x)) # 14*14
        x = F.max_pool2d(x, 1) # 7*7
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.15)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.1)
        x = self.fc3(x)

        return x