import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
from copy import deepcopy

import torch_pruning as tp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepFCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.add_module('first_relu', nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU()
        )
        self.fc3 = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU()) for i in range(3)
            ]
        )
        self.fc4 = nn.ModuleDict({
            'fc4-1': nn.Linear(64,32),
            'relu': nn.ReLU()
        })
        self.fc5 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.first_relu(x)
        x = self.fc2(x)
        for i, l in enumerate(self.fc3):
            x = l(x)
        x = self.fc4['fc4-1'](x)
        x = self.fc4['relu'](x)
        y_hat = self.fc5(x)
        return y_hat

model = DeepFCN(128, 10)
print(model)

planner = tp.planner.GlobalRandomPlanner()
pruning_plans = planner(model, amount=0.2, target_type=nn.Linear, static_layers=[model.fc5], example_inputs=torch.randn(1,128))

print(model)
for plan in pruning_plans:
        plan.exec()
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< before')
print('after >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print(model)
