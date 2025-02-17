import numpy as np
import torch
import torch.nn as nn
from eval import compute_from_file
import random


def TransformingLayer(x, n):

    output = compute_from_file(x, "output.txt")
    output = [x for x in output if x != 0]
    random_output = random.sample(output, n)

    return random_output
    

class InvariantNet(nn.Module):

    def __init__(self, n):

        super(InvariantNet, self).__init__()

        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 10)
        self.n = n

    def forward(self, x):

        x = TransformingLayer(x, self.n)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
