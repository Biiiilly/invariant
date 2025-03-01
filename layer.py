import torch
import torch.nn as nn
import torch.nn.functional as F


class InvariantNet(nn.Module):
    """
    Reduced-parameter network (~14K params) for input dimension n=100.
    Architecture:
      fc1: 100 -> 64
      fc2: 64  -> 64
      fc3: 64  -> 40
      fc4: 40  -> 10   (output)
      bn1, bn2, bn3 for the three hidden layers
      dropout = 0.5
    """

    def __init__(self, n=100):
        super(InvariantNet, self).__init__()
        self.n = n

        # Fully connected layers
        self.fc1 = nn.Linear(self.n, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 40)
        self.fc4 = nn.Linear(40, 10)

        # BatchNorm layers for the 3 hidden layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(40)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
        
