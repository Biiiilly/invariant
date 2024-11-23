import torch
import torch.nn as nn
import torch.functional as F

'''
Construct the invariant MLP model for image classification. (e.g. Mnist test)

First step is to tranform the image into a list of vectors (v1, ..., vn).
Then find the dot product of vi and vj and use them as the input set.

As a test, we consider the image with size 2x2 first.
'''


class InvariantNet(nn.Module):

    def __init__(self):

        super(InvariantNet, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        

    def forward(self, x):
