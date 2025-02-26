import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input images are 8x8, with one channel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Output: 16 x 8 x 8
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # Output: 32 x 8 x 8
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces each spatial dim by a factor of 2

        # After one pooling, size becomes 4x4. After a second pooling, size becomes 2x2.
        # Thus, the number of features entering the FC layer is 32 * 2 * 2 = 128.
        self.fc1 = nn.Linear(32 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        # x should have shape (batch, 8, 8); add channel dimension if needed
        if x.ndim == 3:  
            x = x.unsqueeze(1)  # shape becomes (batch, 1, 8, 8)

        x = F.relu(self.bn1(self.conv1(x)))  # shape: (batch, 16, 8, 8)
        x = self.pool(x)                     # shape: (batch, 16, 4, 4)
        
        x = F.relu(self.bn2(self.conv2(x)))  # shape: (batch, 32, 4, 4)
        x = self.pool(x)                     # shape: (batch, 32, 2, 2)
        
        x = x.view(x.size(0), -1)            # Flatten, shape: (batch, 128)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
