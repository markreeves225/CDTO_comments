import collections
import json
from random import randint
import random
from algorithm.CDTO.base_ergodic import  Simulator_ergodic

import time
import os
import numpy as np
__version__ = '0.1'

PLOT_STYLE = 'darkgrid'

LOG_CONFIG = {
    'clear_files': False,
    'streams': [
        print,
        'output.log'
    ],
    'log_dir': '~/logs/default'
}


def set_log_dir(path):
    expanded_path = os.path.expanduser(path)
    if not os.path.exists(expanded_path):
        os.makedirs(expanded_path)

    LOG_CONFIG['log_dir'] = path

    if LOG_CONFIG['clear_files'] or not os.path.exists(os.path.join(path, 'output.log')):
        for stream in LOG_CONFIG['streams']:
            if isinstance(stream, str):
                with open(stream, 'w') as f:
                    f.write('')


def get_log_dir():
    return os.path.expanduser(LOG_CONFIG['log_dir'])


set_log_dir(LOG_CONFIG['log_dir'])


def log(*strings, end='\n', sep=''):
    for stream in LOG_CONFIG['streams']:
        if isinstance(stream, str):
            with open(os.path.expanduser(os.path.join(LOG_CONFIG['log_dir'], stream)), 'a') as f:
                timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
                string = sep.join(str(s) for s in strings)
                f.write(f'{timestamp}  {string}{end}')
        else:
            stream(*strings, end=end, sep=sep)


def prefix_heuristic(prefix_length=None, delimiter=None):
    def should_colocate(a, b):
        if prefix_length:
            return a[:prefix_length] == b[:prefix_length]
        if delimiter:
            return a.split(delimiter)[0] == b.split(delimiter)

    return should_colocate


def create_colocation_groups(layer_names, colocation_heuristic):
    groups = []
    for layer in layer_names:
        for group in groups:
            if colocation_heuristic(layer, group[0]):
                group.append(layer)
                break
        else:
            groups.append([layer])
    return groups

#Function call @ generate_random_placement( [127 16 26 59...16 59], 35, true)
## The loop range should be the sum of task dimensions minus 2 times the length of task_dim
'''
import numpy as np
from random import randint

def generate_random_placement(task_dim, n_devices, allow_device_0=True):
    placement = []
    
    # The loop range should be the sum of task dimensions minus 2 times the length of task_dim
    for i in range(np.sum(task_dim) - 2 * len(task_dim)):
        if allow_device_0:
            placement.append(randint(0, n_devices - 1))
        else:
            placement.append(randint(1, n_devices - 1))

    return placement

# Example usage
task_dimensions = [4,5,6]

result_placement = generate_random_placement(task_dimensions, 35, True)
print(result_placement)
##First and Last layer of each job is not counted, if DNN have 5 layer then only 5-2=3 devices will be allocated
#-------------------------->[25, 0, 5, 19, 33, 5, 16, 13, 23]
'''
def generate_random_placement(task_dim, n_devices, allow_device_0=True):
    placement = []
    for i in range(np.sum(task_dim)-len(task_dim)*2):
        if allow_device_0:
            placement.append(randint(0, n_devices - 1))
        else:
            placement.append(randint(1, n_devices - 1))
    # print(placement)
    return placement


def evaluate_placement(net, task_iner_priority, device_graph, task_unit, batch_size=128, batches=1, pipeline_batches=1, memory_penalization_factor=1,
                       noise_std=0, comp_penalty=1, comm_penalty=1, device_memory_utilization=1):
    # print("util.py 116")
    # print('1',net)
    '''
    task_inner_priority=[[data, conv1-1, conv1-2, pool1, conv2-1, fc7, dropout7, fc8, softmax, output],
                            [data, conv1, pool1, res2a_branch2a, res4c, res5c, pool5, fc1000, loss, output],
                            [data, conv1, pool1, res2a_branch2a,  fc1000, loss, output]]
    '''

    '''
    import numpy as np

    task_unit = 25

    # Create a dummy net matrix
    net = np.array([
        [2, 3, 8, 5, 4],  
        [3, 1, 7, 2, 1],  
        [4, 1, 2, 1, 1],  
        
    ])


    for i in range(5):
        if i == 0 or i == 4:
            # Compute device assignments using task indices and task_unit
            device = [int(task / task_unit) for task in range(5)]
        else:
            # Use the assignment from the previous iteration
            device = net[i-1]

        print(f"i={i}, device={device}")

    i=0, device=[0, 0, 0, 0, 0]
    i=1, device=[2 3 8 5 4]
    i=2, device=[3 1 7 2 1]
    i=3, device=[4 1 2 1 1]
    i=4, device=[0, 0, 0, 0, 0]

    '''
    simulator2 = Simulator_ergodic(task_iner_priority, device_graph, task_unit)
    for i in range(127):
        # print(i)
        if i == 0 or i == 126:
            device = [int(task / task_unit) for task in range(len(task_iner_priority))]
            # print('device',device)

        else:
            # print(i)
            device = net[i-1]
            # print(device)
            # device = devices[i-1]
        task_finish_time3 = simulator2.simulate(device)

    # print(max(task_finish_time3))
    return max(task_finish_time3)


def apply_placement(task_dim,placement):
    # print("##############################")
    # print("util.py 138")
    # print(task_dim) #size of each task, that is number of layers in each tasks and there are total 250 tasks
    # print(len(task_dim))
    # print(placement) #placement for all layers
    # print(len(placement))  #12890 values
    net = np.ones((max(task_dim) - 2, len(task_dim)), dtype=int)
    j = 0
    for i in range(len(task_dim)):
        net[:task_dim[i] - 2, i] = placement[j:j + task_dim[i] - 2]
        j += task_dim[i] - 2



    return net

'''
#-------------------------net returns 127-2 X 250 matrix where each coloumns defines separate tasks and rows is equal to max sub layer in any tasks
import numpy as np

def apply_placement(task_dim, placement):
    net = np.ones((max(task_dim) - 2, len(task_dim)), dtype=int)
    j = 0
    for i in range(len(task_dim)):
        if task_dim[i] - 2 > 0:
            net[:task_dim[i] - 2, i] = placement[j:j + task_dim[i] - 2]
            j += task_dim[i] - 2
    return net

# Example usage
task_dim = [6, 3, 5,2,1]
placement = [1, 2, 3, 4, 5,8,7,100]

result = apply_placement(task_dim, placement)
print(result)


[[  1   5   8   1   1]
 [  2   1   7   1   1]
 [  3   1 100   1   1]
 [  4   1   1   1   1]]


Max Value is 127 and length of task_dim is 250, they are creating matrix of size 125X250
In above example max subtask of for any task is 6-2=4 and there are 5 such task

'''
def get_device_assignment(net_dict):
    device_assignment = {}

    for layer_name in net_dict['layers'].keys():
        layer = net_dict['layers'][layer_name]

        try:
            device_assignment[layer_name] = layer['device']
        except KeyError:
            device_assignment[layer_name] = 0

        if layer['type'] == 'Block':
            for sublayer_name in layer['layers'].keys():
                sublayer = layer['layers'][sublayer_name]
                try:
                    device_assignment[f'{layer_name}/{sublayer_name}'] = sublayer['device']
                except KeyError:
                    device_assignment[f'{layer_name}/{sublayer_name}'] = 0
    return device_assignment


def flatten(l, depth=-1, current_depth=0, include_none=True):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)) \
                and (depth == -1 or current_depth < depth):
            yield from flatten(el, current_depth=current_depth+1, depth=depth, include_none=include_none)
        else:
            if include_none or el is not None:
                yield el
