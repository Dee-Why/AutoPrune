import numpy as np
from openbox import Optimizer, sp, Advisor, Observation
import matplotlib.pyplot as plt
import time
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--r', type=int, help='MAX_RUNS for openbox')

# Parse args
args = parser.parse_args()
MAX_RUNS = args.r

# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", -5, 10, default_value=0)
x2 = sp.Real("x2", 0, 15, default_value=0)
space.add_variables([x1, x2])


# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2+10*(1-1/(8*np.pi))*np.cos(x1)+10
    ret = {'objs':(y,)}
    return ret
    

# Run
if __name__ == '__main__':
    advisor = Advisor(
        space,
        surrogate_type='auto',
        task_id='branin_test'
    )
    for i in range(MAX_RUNS):
        config = advisor.get_suggestion()
        ret = branin(config)
        observation = Observation(config=config, objs=ret['objs'])
        advisor.update_observation(observation)
        print('===== ITER %d/%d: %s.' % (i+1, MAX_RUNS, observation))

    history = advisor.get_history()
    print(history)

    history.plot_convergence()
    timestring = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    plt.savefig('branin_'+timestring)
    with open('branin_perfs_'+timestring+'.pkl','wb') as p:
        pickle.dump(history.perfs, p)
    with open('branin_configs_'+timestring+'.pkl', 'wb') as p:
        pickle.dump(history.configurations, p)
