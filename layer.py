import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, n=5200):

        super(InvariantNet, self).__init__()

        self.n = n
        self.fc1 = nn.Linear(self.n, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = F.relu(self.bn1(self.fc1(x)))
        #x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        #x = F.dropout(x, training=self.training, p=0.3)  # 30% Dropout
        x = self.fc5(x)

        return F.log_softmax(x, dim=1)
        
