"""
对比对于同一个任务而言，是否引入阶段性剪枝 prune or not-prune
实验设计： 证明我们的算法是实际有帮助的(放弃剪枝过程中的finetune从而接近zero-cost)
训练模型INIT_RUN次 记为origin
以origin为基spawn-(evolve-evalue-elimin)*gen次, 选出最好的 记为best1
训练 origin和best1 INIT_RUN次
以best1为基spawn-(evolve-evalue-elimin)*gen次, 选出最好的 记为best2
训练 origin和best1和best2 INIT_RUN次
以best2为基spawn-(evolve-evalue-elimin)*gen次, 选出最好的 记为best3

在某一次训练INIT_RUN之后结束 保存origin模型和best1-4 一共得到五个参数量依次递减的模型

之后试图减少的参数 gen 争取不是到gen结束 而是收敛了就结束 收敛的标准定为多少代里没有首位变化
"""
import sys
import os
import time
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
import torch_pruning as tp
from torch_pruning import ModelPool
import torch_pruning.experiment as experiment
import pickle
import argparse
import matplotlib.pyplot as plt

torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, choices=['alex', 'dense'])
parser.add_argument('--s', type=str, choices=['random', 'L1'])
parser.add_argument('--i', type=int, help='base_model init epoch')
parser.add_argument('--f', type=int, help='finetune epoch')
parser.add_argument('--g', type=int, help='evolve generation')
parser.add_argument('--p', type=int, help='evolve population')
parser.add_argument('--m', type=int, help='MAX_STAGE: compression iteration num')

args = parser.parse_args()
MODEL_TYPE = args.type
STRATEGY = args.s
INIT_RUN = args.i
FINETUNE = args.f
GENERATION = args.g
POPULATION = args.p
MAX_STAGE = args.m

def compress_alex(base_model, s1, s2, s3):
    """Example experiment.
    
    Args:
        base_model (AlexNetMnist): The base_model.
        s1 (float): probability weight of selection.
        s2 (float): probability weight of crossover.
        s3 (float): probability weight of mutation.

    Returns:
        res (dict): keys--'compressed_model(model)', 'objs(tuple)', 'history(list)'
    """
    # Init run
    sum = s1 + s2 + s3
    s1, s2, s3 = s1/sum, s2/sum, s3/sum
    experiment.fast_train_alex(base_model, INIT_RUN)
    experiment_history = [[base_model.performance]]

    # Spawn first generation
    example_inputs = torch.randn(1, 1, 224, 224)    # AlexNet input-size
    model_pool = ModelPool(base_model, POPULATION, example_inputs)
    model_pool.spawn_first_generation(_strategy=STRATEGY)
    for model in model_pool.pool:
        if not hasattr(model, 'performance'):
            experiment.fast_train_alex(model, FINETUNE)
    experiment_history.append([model.performance for model in model_pool.pool])
    # Evolve
    for generation in range(GENERATION):
        print('[generation', generation,']')
        model_pool.evolve(s1,s2,s3,_strategy=STRATEGY)
        for model in model_pool.pool:
            if not hasattr(model, 'performance'):
                experiment.fast_train_alex(model, FINETUNE)
        model_pool.elimination()
        experiment_history.append([model.performance for model in model_pool.pool])

    print(experiment_history)
    last_run= experiment_history[-1]
    last_run.sort()
    res = last_run[-1]
    print('incumbent: ', res)
    print('loss', 1-res)

    perf_list = [model.performance for model in model_pool.pool]
    index = perf_list.index(max(perf_list))

    return {'objs': (1-res,), 'history': experiment_history, 'compressed_model': model_pool.pool[index], 'model_pool': model_pool}  # 返回最后一代里的最优值对应的loss，因为openbox默认为最小化任务



if __name__ == '__main__':
    base_model = experiment.DeepFCN(225, 10)
    base_model.fc5.do_not_prune = True
    for i in range(MAX_STAGE):
        experiment.fast_train_dense(base_model, INIT_RUN)
        model_pool = ModelPool(base_model, POPULATION, example_inputs=torch.randn(1,225))
        model_pool.spawn_first_generation(_strategy=STRATEGY)
        