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

TO RUN:
python examples/pnp_experiment.py --type le --s L1 --i 3 --g 3 --p 5 --m 2
python examples/pnp_experiment.py --type alex --s L1 --i 20 --g 20 --p 20 --m 5
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

torch.set_num_threads(4)

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, choices=['alex', 'dense', 'le'])
parser.add_argument('--s', type=str, choices=['random', 'L1'])
parser.add_argument('--i', type=int, help='base_model init epoch')
parser.add_argument('--g', type=int, help='evolve generation')
parser.add_argument('--p', type=int, help='evolve population')
parser.add_argument('--m', type=int, help='MAX_STAGE: compression iteration num')
parser.add_argument('--l', type=str, help='logger name, use string like May5 is recommended')

args = parser.parse_args()
MODEL_TYPE = args.type
STRATEGY = args.s
INIT_RUN = args.i
GENERATION = args.g
POPULATION = args.p
MAX_STAGE = args.m
LOGGER_SUFFIX = args.l


class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("pnp_log_"+LOGGER_SUFFIX+".log", sys.stdout)
sys.stderr = Logger("pnp_err_"+LOGGER_SUFFIX+".log", sys.stderr)       # redirect std err, if necessary


def fork_le(base_model, s1=0.52, s2=0.99, s3=0.97):
    """Example experiment.
    
    Args:
        base_model (DeepFCN): The base_model.
        s1 (float): probability weight of selection. default=0.52
        s2 (float): probability weight of crossover. default=0.99
        s3 (float): probability weight of mutation. default=0.97

    Returns:
        compressed_model (model): best_model when evolution algo end
    """
    sum = s1 + s2 + s3
    s1, s2, s3 = s1/sum, s2/sum, s3/sum
    # Spawn first generation and evaluate them
    model_pool = ModelPool(base_model, POPULATION, torch.randn(1, 1, 28, 28))
    model_pool.spawn_first_generation(_strategy=STRATEGY, preserve_origin=False)
    for model in model_pool.pool:
        if not hasattr(model, 'performance'):
            experiment.fast_evaluate_le(model)
    # Evolve
    for generation in range(GENERATION):
        model_pool.evolve(s1,s2,s3,_strategy=STRATEGY)
        for model in model_pool.pool:
            if not hasattr(model, 'performance'):
                experiment.fast_evaluate_le(model)
        model_pool.elimination()
        perf_list = [model.performance for model in model_pool.pool]
        print('[generation', generation,']', perf_list)

    index = perf_list.index(max(perf_list))

    return model_pool.pool[index]

def fork_alex(base_model, s1=0.52, s2=0.99, s3=0.97):
    """Example experiment.
    
    Args:
        base_model (AlexNetMnist): The base_model.
        s1 (float): probability weight of selection. default=0.52
        s2 (float): probability weight of crossover. default=0.99
        s3 (float): probability weight of mutation. default=0.97

    Returns:
        compressed_model (model): best_model when evolution algo end
    """
    sum = s1 + s2 + s3
    s1, s2, s3 = s1/sum, s2/sum, s3/sum
    # Spawn first generation and evaluate them
    model_pool = ModelPool(base_model, POPULATION, torch.randn(1, 1, 224, 224))
    model_pool.spawn_first_generation(_strategy=STRATEGY, preserve_origin=False)
    for model in model_pool.pool:
        if not hasattr(model, 'performance'):
            experiment.fast_evaluate_alex(model)
    # Evolve
    for generation in range(GENERATION):
        model_pool.evolve(s1,s2,s3,_strategy=STRATEGY)
        for model in model_pool.pool:
            if not hasattr(model, 'performance'):
                experiment.fast_evaluate_alex(model)
        model_pool.elimination()
        perf_list = [model.performance for model in model_pool.pool]
        print('[generation', generation,']', perf_list)

    index = perf_list.index(max(perf_list))

    return model_pool.pool[index]

if __name__ == '__main__':
    if MODEL_TYPE == 'le':
        base_model = experiment.LeNet()
        base_model.fc[4].do_not_prune = True
    elif MODEL_TYPE == 'alex':
        base_model = experiment.AlexNetMnist()  # Alex Version
        base_model.fc[6].do_not_prune = True  # AlexNet static Layer
    all_model = [base_model]
    # plot_data = [[] for i in range(MAX_STAGE+1)]
    for i in range(MAX_STAGE):
        for model in all_model:
            if MODEL_TYPE == 'le':
                experiment.fast_train_le(model, INIT_RUN)
            elif MODEL_TYPE == 'alex':
                experiment.fast_train_alex(model, INIT_RUN)
        # perfs = [model.performance for model in all_model]
        # plot_data.append(perfs)
        # 这里有一个变种，就是选择perfs最好的座位fork的基模型, 先不实现
        print("FORK STAGE", i)
        if MODEL_TYPE == 'le':
            new_model = fork_le(all_model[i])
        elif MODEL_TYPE == 'alex':
            new_model = fork_alex(all_model[i])
        all_model.append(new_model)
    
    for model in all_model:
        if MODEL_TYPE == 'le':
            experiment.fast_train_le(model, INIT_RUN)
        elif MODEL_TYPE == 'alex':
            experiment.fast_train_alex(model, INIT_RUN)
    
    perfs = [model.performance for model in all_model]
    paras = [tp.planner.count_parameters(model) for model in all_model]
    z = zip(paras, perfs)
    for i in z:
        print(i)

    timestring = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    expstring = './pnp_'+MODEL_TYPE+'_'+STRATEGY+'_i'+str(INIT_RUN)+'p'+str(POPULATION)+'g'+str(GENERATION)+'m'+str(MAX_STAGE)+'_'+timestring

    for i in range(MAX_STAGE+1):
        torch.save(base_model, expstring+'_'+str(i)+'.model')