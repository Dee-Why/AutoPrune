import random

import torch
import torch.nn as nn
import typing
import collections
from abc import ABC, abstractclassmethod
from . import prune
from .dependency import *
from copy import deepcopy


def prune_global(
        model: nn.Module, 
        amount: float, 
        target_type: nn.Module, 
        static_layers: typing.Sequence[nn.Module], 
        example_inputs)-> typing.Sequence[PruningPlan]:
    # pruning
    module_to_idxs = {}
    def init_strategy(m):
        strategy = prune.strategy.RandomStrategy()
        if isinstance(m, target_type) and m not in static_layers:
            module_to_idxs[m] = strategy(m.weight, amount=amount)
    model.apply(init_strategy)
    DG = DependencyGraph()
    DG.build_dependency(model,example_inputs=example_inputs)
    pruning_plans = []
    def get_pruning_plans(m):
        if m in module_to_idxs:
            pruning_plans.append(DG.get_pruning_plan(m, prune.prune_linear, idxs=module_to_idxs[m]))
    model.apply(get_pruning_plans)
    return pruning_plans


class BaseGloalPlanner(ABC):
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @abstractclassmethod
    def apply(self,model, amount, target_type, static_layers, example_inputs):
        raise NotImplementedError


class GlobalRandomPlanner(BaseGloalPlanner):

    def apply(self,model, amount, target_type, static_layers, example_inputs):
        module_to_idxs = {}
        def init_strategy(m):
            strategy = prune.strategy.RandomStrategy()
            if isinstance(m, target_type) and m not in static_layers:
                module_to_idxs[m] = strategy(m.weight, amount=amount)
        model.apply(init_strategy)
        DG = DependencyGraph()
        DG.build_dependency(model,example_inputs=example_inputs)
        pruning_plans = []
        def get_pruning_plans(m):
            if m in module_to_idxs:
                pruning_plans.append(DG.get_pruning_plan(m, prune.prune_linear, idxs=module_to_idxs[m]))
        model.apply(get_pruning_plans)
        return pruning_plans


# TODO: 要搞出来一个evolve过程，输入一个模型，输出一个训练过的模型池
# 理论基础是LTH，也就是说用同样的初始化，利用遗传算法找winning ticket
# 首先，我们要能记录一次剪枝的具体细节，并移植到另一个网络去
# 我认为最简单的方法就是使用orderded dict。这样我们的字典不光有键值对儿，而且还可以借助顺序相同来进行idxs的迁移
# 对于全连接网络，交叉互换的最小单位是层

def get_ordered_module_to_idxs(model, amount, target_type, static_layers, example_inputs):
    module_to_idxs = collections.OrderedDict()
    def init_strategy(m):
        strategy = prune.strategy.RandomStrategy()
        if isinstance(m, target_type) and m not in static_layers:
            module_to_idxs[m] = strategy(m.weight, amount=amount)
    model.apply(init_strategy)
    return module_to_idxs


def get_module_to_idxs(model, amount, target_type, _strategy='random'):
    module_to_idxs = collections.OrderedDict()

    def init_strategy(m):
        if _strategy == 'random':
            strategy = prune.strategy.RandomStrategy()
        elif _strategy == 'L1':
            strategy = prune.strategy.L1Strategy()
        if hasattr(m, 'do_not_prune'):
            return
        elif isinstance(m, target_type):
            module_to_idxs[m] = strategy(m.weight, amount=amount)


    model.apply(init_strategy)
    return module_to_idxs


def get_pruning_plans(model, example_inputs):
    DG = DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)
    pruning_plans = []

    def get_pruning_plan(m):
        if m in model.module_to_idxs:
            if isinstance(m, nn.Linear):
                pruning_plans.append(DG.get_pruning_plan(m, prune.prune_linear,
                                                     idxs=model.module_to_idxs[m]))
            elif isinstance(m, nn.Conv2d):
                pruning_plans.append(DG.get_pruning_plan(m, prune.prune_conv,
                                                     idxs=model.module_to_idxs[m]))
    model.apply(get_pruning_plan)
    return pruning_plans


class ModelPool(ABC):
    def __init__(self, base_model, population, example_inputs, strategy=None):
        super(ModelPool, self).__init__()
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

    def calculate_parameter_num(self):
        res = []
        for model in self.pool:
            num = 0
            for para in model.parameters():
                num += para.size().numel()
            res.append(num)
        return res


    def spawn_first_generation(self, _strategy='random'):
        for i in range(self.population):
            child = deepcopy(self.base_model)
            if hasattr(child, 'performance'):
                delattr(child, "performance")
            child.module_to_idxs = get_module_to_idxs(child, random.uniform(0,1), (nn.Linear, nn.Conv2d), _strategy=_strategy)
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
            if hasattr(child, 'performance'):
                delattr(child, "performance")
            child.module_to_idxs = get_module_to_idxs(child, 0, (nn.Linear, nn.Conv2d))
            for i, ((k1, v1), (k2, v2)) in enumerate(zip(*[child.module_to_idxs.items(), vec.items()])):
                child.module_to_idxs[k1] = vec[k2]  # 对应位置赋值
            pruning_plans = get_pruning_plans(child, self.example_inputs)
            for plan in pruning_plans:
                plan.exec()
            self.pool.append(child)
        return

    def mutation(self, _strategy='random'):
        index = [i for i in range(self.population)]
        weights = [self.fitness[i] for i in range(self.population)]
        choice = random.choices(index, weights=weights, k=1)[0]
        vec = deepcopy(self.pool[choice].module_to_idxs)
        s = random.randint(0, len(vec) - 1)
        e = random.randint(s + 1, len(vec))
        indicate_vector = [1 if s <= i < e else 0 for i in range(len(vec))]
        child = deepcopy(self.base_model)
        if hasattr(child, 'performance'):
                delattr(child, "performance")
        child.module_to_idxs = get_module_to_idxs(child, random.uniform(0,1), (nn.Linear, nn.Conv2d), _strategy=_strategy)
        for i, ((k1, v1), (k2, v2)) in enumerate(zip(*[child.module_to_idxs.items(), vec.items()])):
            if indicate_vector[i] == 0:
                child.module_to_idxs[k1] = vec[k2]

        pruning_plans = get_pruning_plans(child, self.example_inputs)
        for plan in pruning_plans:
            plan.exec()
        self.pool.append(child)
        return

    def evolve(self, s1, s2, s3, _strategy='random'):
        assert s1 >= 0 and s2 >= 0 and s3 >= 0
        for i in range(self.population):
            if i == 0:
                print(i, 'inherit')
                self.inherit()
                continue
            dice = random.uniform(0, s1+s2+s3)
            if dice < s1:
                print(i, 'selection')
                self.selection()
            elif dice < s1 + s2:
                print(i, 'crossover')
                self.crossover()
            else:
                print(i, 'mutation')
                self.mutation(_strategy=_strategy)

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
