import torch
import torch.nn as nn


class VisionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # fully connected
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.adaptive = nn.AdaptiveAvgPool2d((3, 3))
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.outpout = nn.Linear(512, 4)

    def forward(self, image):
        x = self.pool(self.relu(self.conv1(image)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.conv4(x))
        x = self.adaptive(x)

        # fully connected

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.outpout(x)

        return x
