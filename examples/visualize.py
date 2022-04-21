import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pickle
from torch_pruning import experiment
from tensorboardX import SummaryWriter

# /Users/dee_why/Documents/python_code/pytorch_code/AutoPrune/experiment/evolve_dair2/experiment_history_b30p10g40t5.pkl
with open('experiment/evolve_dair2/experiment_history_b30p10g40t5.pkl', 'rb') as r:
    data = pickle.load(r)

experiment.visualize_incumbent(data)
