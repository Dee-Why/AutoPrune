"""
对比对于同一个任务而言，是否引入阶段性剪枝 prune or not-prune
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

if __name__ == '__main__':
    base_model = experiment.DeepFCN(225, 10)
    base_model.fc5.do_not_prune = True
    for i in range(MAX_STAGE):
        experiment.fast_train_dense(base_model, INIT_RUN)
        model_pool = ModelPool(base_model, POPULATION, example_inputs=torch.randn(1,225))
        model_pool.spawn_first_generation()