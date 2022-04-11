import torch
import torch.nn as nn
import typing
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