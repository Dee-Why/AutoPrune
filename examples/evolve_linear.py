import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
import torch_pruning as tp
from torch_pruning import experiment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeepFCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.add_module('first_relu', nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.fc3 = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU()) for i in range(3)
            ]
        )
        self.fc4 = nn.ModuleDict({
            'fc4-1': nn.Linear(64, 32),
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


if __name__ == '__main__':
    base_model = DeepFCN(225, 10)
    population = 5
    static_layers = [base_model.fc5]
    example_inputs = torch.randn(1, 225)
    # 算法需要的专属标签【do_not_prune】
    for layer in static_layers:
        layer.do_not_prune = True

    # 预训练模型
    import torch_pruning.experiment as experiment
    experiment.fast_train_dense(base_model, 5)
    experiment_history = [[base_model.performance]]
    # 创建模型池 并产生第一代
    model_pool = tp.ModelPool(base_model, population, example_inputs)
    model_pool.spawn_first_generation()
    # 训练第一代
    for model in model_pool.pool:
        # if not hasattr(model, 'performance'):
        experiment.fast_train_dense(model, 2)
    # 保存第一代成果
    experiment_history.append([model.performance for model in model_pool.pool])
    for generation in range(5):
        # 产生第二代 继承率0.3 交叉互换率0.55 变异率0.15
        model_pool.evolve(0.3, 0.55, 0.15)
        # 训练第二代
        for model in model_pool.pool:
            # if not hasattr(model, 'performance'):
            experiment.fast_train_dense(model, 2)
        # 所有子代成果
        experiment_history.append([model.performance for model in model_pool.pool])

        model_pool.elimination()
        # 筛选后留下的部分
        experiment_history.append([model.performance for model in model_pool.pool])

    print(experiment_history)
