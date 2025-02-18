import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from eval import compute_from_file
import random


'''
def TransformingLayer(x, n):

    output = compute_from_file(x, "output.txt")
    print(len(output))
    print(output[0])
    output = [x for x in output if x != 0]
    random_output = random.sample(output, n)

    return torch.tensor(random_output)

class TransformingLayer(nn.Module):

    def __init__(self, n):
        super(TransformingLayer, self).__init__()
        self.n = n
    
    def forward(self, x):
        
        output_list = []
        a, b, c, d = x.shape
        for i in range(a):
            image = x[i][0].flatten()
            print(image)
            output = compute_from_file(image, "output.txt")
            output = [x for x in output if x != 0]
            random_output = random.sample(output, self.n)
            output_list.append(random_output)

        output_list = torch.tensor(output_list)
        print(output_list.shape)
        return output_list
'''

class InvariantNet(nn.Module):

    def __init__(self, n=64):

        super(InvariantNet, self).__init__()

        self.n = n
        self.fc1 = nn.Linear(self.n, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 10)


    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
        
