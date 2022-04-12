import torch
import torch.nn as nn
import typing
import collections
from abc import ABC, abstractclassmethod
from . import prune
from .dependency import *


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


def crossover(module_to_idxs1, module_to_idxs2, indicate_vector):
    # 两个训练过的剪枝模型的idxs进行局部交换，然后从原始模型里prune出来，然后train from scratch
    # 交换idxs
    for i, ((k1, v1), (k2, v2)) in enumerate(zip(*[module_to_idxs1.items(), module_to_idxs2.items()])):
        if indicate_vector[i] == 1:
            tmp = module_to_idxs1[k1]
            module_to_idxs1[k1] = module_to_idxs2[k2]
            module_to_idxs2[k2] = tmp
    return module_to_idxs1, module_to_idxs2

def inherit():
    # 只需要维护模型池，也就是存档的 文件名称 和 他们performance指标
    return

def mutation(module_to_idx, indicate_vector):
    # 利用crossover，让原有模型和随机模型进行交叉互换
    return