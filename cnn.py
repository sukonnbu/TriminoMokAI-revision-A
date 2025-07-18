import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()

        # Input: 14x19x19
        # Convolutional Layers
        self.conv1 = nn.Conv2d(14, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 4)

        # Pooling Layers
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 5 * 5, 4 * 19 * 19)

        self.relu = nn.ReLU()

    def forward(self, n):
        n = self.relu(self.conv1(n)) # 19 - 4 + 1, 32 x 16 x 16
        n = self.pool(n)  # 32 x 8 x 8
        n = self.relu(self.conv2(n)) # 8 - 4 + 1, 64 x 5 x 5

        n = n.view(-1, 64 * 5 * 5)

        n = self.fc1(n)
        n = n.view(-1, 19, 19, 4)

        return n

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        # Input: 13x19x19
        # Convolutional Layers
        self.conv1 = nn.Conv2d(13, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 4)

        # Pooling Layers
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 5 * 5, 1)

        self.relu = nn.ReLU()

    def forward(self, n):
        n = self.relu(self.conv1(n)) # 19 - 4 + 1, 32 x 16 x 16
        n = self.pool(n) # 32 x 8 x 8
        n = self.relu(self.conv2(n)) # 64 x 5 x 5

        n = n.view(-1, 64 * 5 * 5)

        n = self.fc1(n)

        return n
