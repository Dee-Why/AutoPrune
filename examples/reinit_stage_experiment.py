from email.mime import base
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
import torch_pruning as tp
from torch_pruning import ModelPool
import torch_pruning.experiment as experiment
import pickle
import argparse

torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, choices=['alex', 'dense'])
parser.add_argument('--i', type=int, help='base_model init epoch')
parser.add_argument('--f', type=int, help='finetune epoch')
parser.add_argument('--g', type=int, help='evolve generation')
parser.add_argument('--p', type=int, help='evolve population')
parser.add_argument('--m', type=int, help='MAX_STAGE: compression iteration num')
parser.add_argument('--s', type=str, choices=['random', 'L1'])
args = parser.parse_args()
MODEL_TYPE = args.type
INIT_RUN = args.i
FINETUNE = args.f
GENERATION = args.g
POPULATION = args.p
MAX_STAGE = args.m
STRATEGY = args.s


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

def compress_dense(base_model, s1, s2, s3):
    """Example experiment.
    
    Args:
        base_model (DenseNet): The base_model.
        s1 (float): probability weight of selection.
        s2 (float): probability weight of crossover.
        s3 (float): probability weight of mutation.

    Returns:
        res (dict): keys--'compressed_model(model)', 'objs(tuple)', 'history(list)'
    """
    # Init run
    sum = s1 + s2 + s3
    s1, s2, s3 = s1/sum, s2/sum, s3/sum
    experiment.fast_train_dense(base_model, INIT_RUN)
    experiment_history = [[base_model.performance]]

    # Spawn first generation
    example_inputs = torch.randn(1, 225)  # DenseNet input size
    model_pool = ModelPool(base_model, POPULATION, example_inputs)
    model_pool.spawn_first_generation(_strategy=STRATEGY)
    for model in model_pool.pool:
        if not hasattr(model, 'performance'):
            experiment.fast_train_dense(model, FINETUNE)
    experiment_history.append([model.performance for model in model_pool.pool])
    # Evolve
    for generation in range(GENERATION):
        print('[generation', generation,']')
        model_pool.evolve(s1,s2,s3,_strategy=STRATEGY)
        for model in model_pool.pool:
            if not hasattr(model, 'performance'):
                experiment.fast_train_dense(model, FINETUNE)
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

# Run
if __name__ == '__main__':
    if MODEL_TYPE == 'alex':
        base_model = experiment.AlexNetMnist()  # Alex Version
        base_model.fc[6].do_not_prune = True  # AlexNet static Layer
    elif MODEL_TYPE == 'dense':
        base_model = experiment.DeepFCN(225,10) # Dense Version
        base_model.fc5.do_not_prune = True  # Dense static Layer

    # 阶段性遗传算法迭代
    experiment_history = []
    parameter_num_history = []
    s1, s2, s3 = 0.3, 0.55, 0.15
    for stage in range(MAX_STAGE):
        if MODEL_TYPE == 'alex':
            res = compress_alex(base_model, s1, s2, s3)
        elif MODEL_TYPE == 'dense':
            res = compress_dense(base_model, s1, s2, s3)
        base_model = res['compressed_model']
        experiment_history.extend(res['history'])
        parameter_num_history.append(res['model_pool'].calculate_parameter_num())
        

    timestring = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    expstring = './reinit_'+MODEL_TYPE+'_'+STRATEGY+'_i'+str(INIT_RUN)+'f'+str(FINETUNE)+'p'+str(POPULATION)+'g'+str(GENERATION)+'m'+str(MAX_STAGE)+'_'+timestring
    
    # TODO: save the compressed model
    print(base_model)
    torch.save(base_model, expstring+'.model')
    
    with open(expstring+'_history.pkl','wb') as p:
        pickle.dump(experiment_history, p)
    with open(expstring+'_parameter_num.pkl','wb') as p:
        pickle.dump(parameter_num_history, p)