"""
给定一个已经训练好的模型 给定时间约束和目标压缩比 看何样的超参数可以在约束内达到最好的解

实验设计：
读取训练好的模型 记为origin
以origin为基spawn-(evolve-evalue-elimin)*gen次, 选出最好的 记为best1
以best1为基spawn-(evolve-evalue-elimin)*gen次, 选出最好的 记为best2
以best2为基spawn-(evolve-evalue-elimin)*gen次, 选出最好的 记为best3

当某一次选出的最佳模型参数量达到目标压缩比 则返回loss 交给openbox优化


TO RUN:
python examples/hp_experiment.py --m ./experiment/alex_mnist.model --t 1800 --r 25 --l May10hp2
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
import torch_pruning as tp
from torch_pruning import ModelPool
import torch_pruning.experiment as experiment
from openbox import Advisor, Observation, sp
import matplotlib.pyplot as plt
import pickle
import argparse

torch.set_num_threads(4)

parser = argparse.ArgumentParser()

parser.add_argument('--m', type=str, help='origin model filename, should save the model inside folder /experiment/')
parser.add_argument('--t', type=int, help='time limit for each experiment, for fairness')
parser.add_argument('--r', type=int, help='MAX_RUNS for openbox, 50 is elegant')
parser.add_argument('--l', type=str, help='logger name, such as May8hp')
# Parse args
args = parser.parse_args()

MODEL = args.m
TIME = args.t
MAX_RUNS = args.r
LOGGER = args.l

class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(LOGGER+"_log.log", sys.stdout)
sys.stderr = Logger(LOGGER+"_err.log", sys.stderr)       # redirect std err, if necessary

# Define Search Space
space = sp.Space()
strategy = sp.Categorical("strategy", choices=['random','L1'], default_value='random')
generation = sp.Int("generation", 5, 40, default_value=20)
population = sp.Int("population", 5, 20, default_value=10)
s1 = sp.Real("s1", 0, 1, default_value=0.53)
s2 = sp.Real("s2", 0, 1, default_value=0.99)
s3 = sp.Real('s3', 0, 1, default_value=0.97)
space.add_variables([strategy, generation, population, s1, s2, s3])

def fork_model(base_model, strategy='random', generation=10, population=10, s1=0.52, s2=0.99, s3=0.97, example_inputs=torch.randn(1, 1, 224, 224)):
    """Example experiment.
    
    Args:
        base_model (AlexNetMnist): The base_model.
        strategy (str): 'random' or 'L1'
        generation (int): generation
        population (int): population
        s1 (float): probability weight of selection. default=0.52
        s2 (float): probability weight of crossover. default=0.99
        s3 (float): probability weight of mutation. default=0.97

    Returns:
        compressed_model (model): best_model when evolution algo end
    """
    sum = s1 + s2 + s3
    s1, s2, s3 = s1/sum, s2/sum, s3/sum
    # Spawn first generation and evaluate them
    model_pool = ModelPool(base_model, population, torch.randn(1, 1, 224, 224))
    model_pool.spawn_first_generation(_strategy=strategy, preserve_origin=False) # 不保留原模型，一定要起到压缩作用
    for model in model_pool.pool:
        if not hasattr(model, 'performance'):
            experiment.fast_evaluate_alex(model)
    # Evolve
    for generation in range(generation):
        model_pool.evolve(s1,s2,s3,_strategy=strategy)
        for model in model_pool.pool:
            if not hasattr(model, 'performance'):
                experiment.fast_evaluate_alex(model)
        model_pool.elimination()
        perf_list = [model.performance for model in model_pool.pool]
        print('[generation', generation,']', perf_list)

    index = perf_list.index(max(perf_list))

    return model_pool.pool[index]

# define objective function
def exp(config, ratio=0.1):
    strategy, generation, population = config["strategy"], config["generation"], config["population"]
    s1, s2, s3 = config['s1'], config['s2'], config['s3']
    sum = s1 + s2 + s3
    s1, s2, s3 = s1/sum, s2/sum, s3/sum
    example_inputs = torch.randn(1, 1, 224, 224)

    model = torch.load(MODEL)
    model.fc[6].do_not_prune = True
    num_param = tp.count_parameters(model)
    threshold = ratio * num_param

    start_time = time.time()
    stage = 0

    while time.time()-start_time < TIME:
        stage += 1
        print("STAGE", stage)
        model = fork_model(model, strategy, generation, population, s1, s2, s3, example_inputs)
        if tp.count_parameters(model) < threshold:
            # 完成任务 返回最优值对应的loss，因为openbox默认为最小化任务
            return {'objs': (1 - model.performance,)} 
    # 如果没有完成任务，则默认loss为1，并且加上惩罚项，参数量越多惩罚越大，惩罚项最大为1
    return {'objs': (1 + tp.count_parameters(model) / num_param,)} 

# Run
if __name__ == '__main__':
    advisor = Advisor(
        space,
        surrogate_type='auto',
        task_id='evolve_hp_turn'
    )
    for i in range(MAX_RUNS):
        print('SAMPLE', i+1)
        config = advisor.get_suggestion()
        ret = exp(config, ratio=0.1)
        observation = Observation(config=config, objs=ret['objs'])
        advisor.update_observation(observation)
        print('===== ITER %d/%d: %s.' % (i+1, MAX_RUNS, observation))
        print('^'*60)

    history = advisor.get_history()
    print(history)
    history.plot_convergence()
    timestring = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    expstring = LOGGER+'_r'+str(MAX_RUNS)+'_'+timestring
    plt.savefig(expstring)
    with open(expstring+'_perfs.pkl','wb') as p:
        pickle.dump(history.perfs, p)
    with open(expstring+'_confs.pkl', 'wb') as p:
        pickle.dump(history.configurations, p)