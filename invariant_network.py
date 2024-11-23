import torch
import torch.nn as nn
import torch.functional as F

'''
Construct the invariant MLP model for image classification. (e.g. Mnist test)

First step is to tranform the image into a list of vectors (v1, ..., vn).
Then find the dot product of vi and vj and use them as the input set.

As a test, we consider the image with size 2x2 first.
We assume the input size would be [batch_size, 1, 2, 2].
'''

def transform2x2(x):

    basis = [1, -1]
    basis_list = []
    result_list= []
 
    for i in basis:
        for j in basis:
            b = torch.tensor([i, j])
            basis_list.append(b)

    for i in basis_list:
        for j in basis_list:
            result = torch.dot(basis_list[i], basis_list[j])
            



    




class InvariantNet(nn.Module):

    def __init__(self):

        super(InvariantNet, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

