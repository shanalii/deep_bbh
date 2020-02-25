import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import numpy as np
import torch

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class deepFilter(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=8, dilation=4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=8, dilation=4)
        #self.conv3_drop = nn.Dropout(0.8)
        self.fc1 = nn.Linear(7616, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 4))
        x = F.relu(F.max_pool1d(self.conv2(x), 4))
        x = F.relu(F.max_pool1d(self.conv3(x), 4))
        x = x.view(-1, 7616)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class deeperFilter(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=16)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=16, dilation=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=16, dilation=2)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=32, dilation=2)
        self.fc1 = nn.Linear(7168, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 4))
        x = F.relu(F.max_pool1d(self.conv2(x), 4))
        x = F.relu(F.max_pool1d(self.conv3(x), 4))
        x = F.relu(F.max_pool1d(self.conv4(x), 4))
        x = x.view(-1, 7168)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class linclass(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.layer = torch.nn.Linear(8192,num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x