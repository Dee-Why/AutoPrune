import random
from abc import ABC
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict
import torch_pruning as tp

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


def get_module_to_idxs(model, amount, target_type):
    module_to_idxs = OrderedDict()

    def init_strategy(m):
        strategy = tp.prune.strategy.RandomStrategy()
        if hasattr(m, 'do_not_prune'):
            return
        elif isinstance(m, target_type):
            module_to_idxs[m] = strategy(m.weight, amount=amount)

    model.apply(init_strategy)
    return module_to_idxs


def get_pruning_plans(model, example_inputs):
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)
    pruning_plans = []

    def get_pruning_plan(m):
        if m in model.module_to_idxs:
            pruning_plans.append(DG.get_pruning_plan(m, tp.prune.prune_linear,
                                                     idxs=model.module_to_idxs[m]))
    model.apply(get_pruning_plan)
    return pruning_plans


class ModulePool(ABC):
    def __init__(self, base_model, population, example_inputs, strategy=None):
        super(ModulePool, self).__init__()
        self.base_model = base_model
        self.pool = []
        self.population = population
        self.example_inputs = example_inputs
        self.strategy = strategy
        self.selection_mark = [0 for i in range(self.population)]
        self.fitness = []
        self.num_parameter_base = 0
        for para in base_model.parameters():
            self.num_parameter_base += para.size().numel()

    def spawn_first_generation(self):
        for i in range(self.population):
            child = deepcopy(self.base_model)
            child.module_to_idxs = get_module_to_idxs(child, 0.2, nn.Linear)
            pruning_plans = get_pruning_plans(child, self.example_inputs)
            for plan in pruning_plans:
                plan.exec()
            self.pool.append(child)
        return

    def calculate_fitness(self):
        for model in self.pool:
            assert hasattr(model, 'performance'), 'model do not have performance(not been trained)'
            num_parameter = 0
            for para in model.parameters():
                num_parameter += para.size().numel()
            model.fitness = max(0, model.performance - 0.05 * (num_parameter / self.num_parameter_base))

    def inherit(self):
        # 每次迭代的第一步，在这里做循环初始化
        self.fitness = []
        self.selection_mark = [0 for i in range(self.population)]
        for i in range(self.population):
            if not hasattr(self.pool[i], 'fitness'):
                self.calculate_fitness()
            self.fitness.append(self.pool[i].fitness)

        incumbent = self.pool[0].fitness
        incumbent_flag = 0
        for i in range(1, self.population):
            if self.pool[i].fitness > incumbent:
                incumbent = self.pool[i].fitness
                incumbent_flag = i
        self.selection_mark[incumbent_flag] = 1
        return

    def selection(self):
        index = [i for i in range(self.population)]
        weights = [0 if self.selection_mark[i] == 1 else self.fitness[i] for i in range(self.population)]
        choice = random.choices(index, weights=weights, k=1)[0]
        self.selection_mark[choice] = 1
        return

    def crossover(self):
        index = [i for i in range(self.population)]
        weights = [self.fitness[i] for i in range(self.population)]
        choices = random.choices(index, weights=weights, k=1)
        weights[choices[0]] = 0
        choices.extend(random.choices(index, weights=weights, k=1))
        vec1 = deepcopy(self.pool[choices[0]].module_to_idxs)
        vec2 = deepcopy(self.pool[choices[1]].module_to_idxs)
        s = random.randint(0, len(vec1) - 1)
        e = random.randint(s + 1, len(vec1))
        indicate_vector = [1 if s <= i < e else 0 for i in range(len(vec1))]
        for i, ((k1, v1), (k2, v2)) in enumerate(zip(*[vec1.items(), vec2.items()])):
            if indicate_vector[i] == 1:
                tmp = vec1[k1]
                vec1[k1] = vec2[k2]
                vec2[k2] = tmp

        for vec in [vec1, vec2]:
            child = deepcopy(self.base_model)
            child.module_to_idxs = get_module_to_idxs(child, 0, nn.Linear)
            for i, ((k1, v1), (k2, v2)) in enumerate(zip(*[child.module_to_idxs.items(), vec.items()])):
                child.module_to_idxs[k1] = vec[k2]  # 对应位置赋值
            pruning_plans = get_pruning_plans(child, self.example_inputs)
            for plan in pruning_plans:
                plan.exec()
            self.pool.append(child)
        return

    def mutation(self):
        index = [i for i in range(self.population)]
        weights = [self.fitness[i] for i in range(self.population)]
        choice = random.choices(index, weights=weights, k=1)[0]
        vec = deepcopy(self.pool[choice].module_to_idxs)
        s = random.randint(0, len(vec) - 1)
        e = random.randint(s + 1, len(vec))
        indicate_vector = [1 if s <= i < e else 0 for i in range(len(vec))]
        child = deepcopy(self.base_model)
        child.module_to_idxs = get_module_to_idxs(child, 0.2, nn.Linear)
        for i, ((k1, v1), (k2, v2)) in enumerate(zip(*[child.module_to_idxs.items(), vec.items()])):
            if indicate_vector[i] == 0:
                child.module_to_idxs[k1] = vec[k2]

        pruning_plans = get_pruning_plans(child, self.example_inputs)
        for plan in pruning_plans:
            plan.exec()
        self.pool.append(child)
        return

    def evolve(self, s1, s2):
        for i in range(self.population):
            if i == 0:
                print(i, 'inherit')
                self.inherit()
                continue
            dice = random.random()
            if dice < s1:
                print(i, 'selection')
                self.selection()
            elif dice < s1 + s2:
                print(i, 'crossover')
                self.crossover()
            else:
                print(i, 'mutation')
                self.mutation()

    def elimination(self):
        for i in range(self.population-1, -1, -1):
            if self.selection_mark[i] == 0:
                del self.pool[i]

        self.calculate_fitness()

        while len(self.pool) > self.population:
            weak_flag = 0
            weak = self.pool[0].fitness
            for i in range(1, len(self.pool)):
                if self.pool[i].fitness < weak:
                    weak = self.pool[i].fitness
                    weak_flag = i
            del self.pool[weak_flag]
        return


if __name__ == '__main__':
    base_model = DeepFCN(225, 10)
    population = 10
    target_type = nn.Linear  # 目前无用
    static_layers = [base_model.fc5]
    example_inputs = torch.randn(1, 225)
    # 算法需要的专属标签【do_not_prune】
    for layer in static_layers:
        layer.do_not_prune = True

    # 预训练模型
    import torch_pruning.experiment as experiment
    experiment.fast_train(base_model, 20)
    experiment_history = [[base_model.performance]]
    # 创建模型池 并产生第一代
    model_pool = ModulePool(base_model, population, example_inputs)
    model_pool.spawn_first_generation()
    # 训练第一代
    for model in model_pool.pool:
        # if not hasattr(model, 'performance'):
        experiment.fast_train(model, 20)
    # 保存第一代成果
    experiment_history.append([model.performance for model in model_pool.pool])
    for generation in range(20):
        # 产生第二代 继承率0.3 交叉互换率0.55 变异率0.15
        model_pool.evolve(0.3, 0.55)
        # 训练第二代
        for model in model_pool.pool:
            # if not hasattr(model, 'performance'):
            experiment.fast_train(model, 20)
        # 所有子代成果
        experiment_history.append([model.performance for model in model_pool.pool])

        model_pool.elimination()
        # 筛选后留下的部分
        experiment_history.append([model.performance for model in model_pool.pool])
