import torch
import torch.nn as nn


class VisionCNN(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Common layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.2)

        self.adaptive = nn.AdaptiveAvgPool2d((3, 3))

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 512)

        self.output = nn.Linear(512, num_classes)


    def forward(self, image):

        # Block 1
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        # Adaptive pooling
        x = self.adaptive(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Fully connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Classification
        x = self.output(x)

        return x
