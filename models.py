import torch
import torch.nn as nn
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Model definition 
class MultiLabelNN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, n_outputs)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x