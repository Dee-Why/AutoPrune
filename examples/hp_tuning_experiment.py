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

torch.set_num_threads(1)

parser = argparse.ArgumentParser()

parser.add_argument('--i', type=int, help='base_model init epoch')
parser.add_argument('--f', type=int, help='finetune epoch')
parser.add_argument('--g', type=int, help='evolve generation')
parser.add_argument('--p', type=int, help='evolve population')
parser.add_argument('--r', type=int, help='MAX_RUNS for openbox')
parser.add_argument('--s', type=str, choices=['random', 'L1'])
# Parse args
args = parser.parse_args()

INIT_RUN = args.i
FINETUNE = args.f
GENERATION = args.g
POPULATION = args.p
MAX_RUNS = args.r
STRATEGY = args.s

# Define Search Space
space = sp.Space()
s1 = sp.Real("s1", 0, 1, default_value=0.3)
s2 = sp.Real("s2", 0, 1, default_value=0.55)
s3 = sp.Real('s3', 0, 1, default_value=0.15)
space.add_variables([s1, s2, s3])


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


# define objective function
def exp(config):
    s1, s2, s3 = config['s1'], config['s2'], config['s3']
    sum = s1 + s2 + s3
    s1, s2, s3 = s1/sum, s2/sum, s3/sum
    base_model = AlexNet()
    experiment.fast_train_alex(base_model, INIT_RUN)
    experiment_history = [[base_model.performance]]
    # 设置遗传算法超参数
    population = POPULATION
    # static_layers = [base_model.fc[4]]
    static_layers = [base_model.fc[6]]
    example_inputs = torch.randn(1, 1, 224, 224)
    for layer in static_layers:
        layer.do_not_prune = True
    # 创建模型池
    model_pool = ModelPool(base_model, population, example_inputs)
    model_pool.spawn_first_generation(_strategy=STRATEGY)
    for model in model_pool.pool:
        if not hasattr(model, 'performance'):
            experiment.fast_train_alex(model, FINETUNE)
    experiment_history.append([model.performance for model in model_pool.pool])
    # 进化部分
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
    return {'objs': (1-res,)}  # 返回最后一代里的最优值对应的loss，因为openbox默认为最小化任务

# Run
if __name__ == '__main__':
    advisor = Advisor(
        space,
        surrogate_type='auto',
        task_id='evolve_hp_turn'
    )
    for i in range(MAX_RUNS):
        config = advisor.get_suggestion()
        ret = exp(config)
        observation = Observation(config=config, objs=ret['objs'])
        advisor.update_observation(observation)
        print('===== ITER %d/%d: %s.' % (i+1, MAX_RUNS, observation))
        print('^'*60)

    history = advisor.get_history()
    print(history)

    history.plot_convergence()
    timestring = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    expstring = './hp_'+STRATEGY+'_i'+str(INIT_RUN)+'f'+str(FINETUNE)+'p'+str(POPULATION)+'g'+str(GENERATION)+'r'+str(MAX_RUNS)+'_'+timestring
    plt.savefig(expstring)
    with open(expstring+'_perfs.pkl','wb') as p:
        pickle.dump(history.perfs, p)
    with open(expstring+'_confs.pkl', 'wb') as p:
        pickle.dump(history.configurations, p)