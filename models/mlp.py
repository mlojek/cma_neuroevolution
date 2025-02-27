'''
Simple multi-layer perceptron model. Default layer dimensionality matches the Iris dataset.
'''

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=10, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)