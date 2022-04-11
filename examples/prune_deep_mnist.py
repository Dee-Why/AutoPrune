import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import time, math
from copy import deepcopy

import torch_pruning as tp


# 定义模型
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
            # x = self.fc3[i//2](x) + l(x)
            x = l(x)
        x = self.fc4['fc4-1'](x)
        x = self.fc4['relu'](x)
        y_hat = self.fc5(x)
        return y_hat

model = DeepFCN(128, 10)
print(model)

# 确定要修剪的各项idxs
strategy = tp.strategy.RandomStrategy()
module_to_idxs = {}
def init_strategy(m):
    if isinstance(m, nn.Linear):
        print('[linear]', m, end='\n')
        module_to_idxs[m] = strategy(m.weight, amount=0.2)
        print(module_to_idxs[m])
        print()
    else:
        print(m, end='\n\n')
# 利用apply的递归探索性质 逐层调用init_strategy
model.apply(init_strategy)

# 搞出依赖图
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,128))

# 搞出一组 【剪枝计划】
pruning_plans = []
def get_pruning_plans(m):
    if m in module_to_idxs:
        pruning_plans.append(DG.get_pruning_plan(m, tp.prune_linear, idxs=module_to_idxs[m]))
model.apply(get_pruning_plans)
for plan in pruning_plans:
    print(plan)

# 打印剪枝前后的模型，进行对比
print(model)
pruning_plans.pop() # 不要处理最后一层（输出层）
for plan in pruning_plans:
    print('<<<<<<<<<<<<<<<<<<<< before')
    print('after >>>>>>>>>>>>>>>>>>>>')
    plan.exec()
    print(model)


# 看一些模型细节
for item in model.fc2.children():
    try:
        print(item.weight)
    except:
        print('nope')