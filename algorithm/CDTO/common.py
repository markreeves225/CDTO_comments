import numpy as np
import copy


import random
from flopsprofiler import FlopsProfiler
from algorithm import  ComputationGraph


def random_task(number,user):
    task_unit=int(number/user)

    list1 = {task: [] for task in range(0,user)}
    for i in range(0,user):
        for j in range(0,task_unit):
          list1[i].append(random.randint(0, 3))

    return list1

def calculate_tensor_size(shape, dtype='float32'):
    return np.prod(shape) * np.dtype(dtype).itemsize

def generate_diff_task(user,num,task_num,base_exc_time,net_graph_path1,net_graph_path2,net_graph_path3,net_graph_path4):
    task = []
    with open(net_graph_path1) as f:
        net_string1 = f.read()
    with open(net_graph_path2) as f:
        net_string2 = f.read()
    with open(net_graph_path3) as f:
        net_string3 = f.read()
    with open(net_graph_path4) as f:
        net_string4 = f.read()
    graph1 = ComputationGraph()
    graph1.load_from_string(net_string1)
    graph2 = ComputationGraph()
    graph2.load_from_string(net_string2)
    graph3 = ComputationGraph()
    graph3.load_from_string(net_string3)
    graph4 = ComputationGraph()
    graph4.load_from_string(net_string4)
    for i in range(0,user):
        #10 users
        for j in task_num[i]:
            #each user have got 25 task with value [0,3]
            if j==0:
                    task.append(copy.deepcopy(graph1))
            elif j==1:
                    task.append(copy.deepcopy(graph2))
            elif j==2:
                    task.append(copy.deepcopy(graph3))
            else:
                    task.append(copy.deepcopy(graph4))
    # print(np.sum(task_num))
    task_dl=[i*10+base_exc_time for i in range(num)]
    return task,task_dl

def generate_same_task(task_num,base_exc_time,net_graph_path1):
    task=[]
    with open(net_graph_path1) as f:
        net_string = f.read()
    graph = ComputationGraph()
    graph.load_from_string(net_string)
    for i in range(task_num[0]):
        task.append(copy.deepcopy(graph))
    task_dl = [random.random() * 10 + base_exc_time for i in range(np.sum(task_num))]
    return task, task_dl

def  task_sort(device,task, task_dl):
    task_iner_priority=[]
    peak_gflops=0
    for i in device.devices:
        peak_gflops+=i.device.peak_gflops
    aver_peak_gflops=peak_gflops/len(device.devices)
    band = 0
    for i in device.comm_channels:
        band += i.bandwidth
    aver_bandwidth = band  / len(device.comm_channels)/8

    ''' -------------------------------------------------Output S T A R T WORK HERE '''
    '''for user_tasks in task:
        print(type(user_tasks))
        for layer_spec in user_tasks.topological_order:
            print(f"Layer Name: {layer_spec.name}")
            print(f"Layer Type: {layer_spec['type']}")
            print(f"Layer Parameters: {layer_spec.params}")
            print(f"Layer Parents: {layer_spec.parents}")
            print(f"Layer Inbounds: {layer_spec.inbounds}")
            print(f"Layer Outbounds: {layer_spec.outbounds}")
            print(f"Layer Device: {layer_spec['device']}")
            print("------")
        print("#######")

        # Layer Name: output
        # Layer Type: Output
        # Layer Parameters: {'parents': ['softmax'], 'type': 'Output', 'tensor': [128, 1000], 'device': 1}
        # Layer Parents: [softmax]
        # Layer Inbounds: [softmax]
        # Layer Outbounds: []
        # Layer Device: 1
        # ------
        # #######


        # Layer Name: fc8
        # Layer Type: Convolution
        # Layer Parameters: {'parents': ['dropout7'], 'type': 'Convolution', 'filter': [1, 1, 4096, 1000], 'padding': 'SAME', 'strides': [1, 1, 1, 1], 'activation_fn': None, 'device': 1}
        # Layer Parents: [dropout7]
        # Layer Inbounds: [dropout7]
        # Layer Outbounds: [softmax]
        # Layer Device: 1
        # ------


    print("999999999999999999999999999999999999999999999999999999999999999999999999999999999999")'''
    ''' -------------------------------------------------E N D WORK HERE '''
    task_priority = np.array(task)[np.argsort(np.array(task_dl))]
    # print(task_priority)
    # print(aver_peak_gflops)
    # print(aver_bandwidth)
    ''' -------------------------------------------------Output S T A R T WORK HERE '''
    '''
    for user_tasks in task_priority:
        print(type(user_tasks))
        for layer_spec in user_tasks.topological_order:
            print(f"Layer Name: {layer_spec.name}")
            print(f"Layer Type: {layer_spec['type']}")
            print(f"Layer Parameters: {layer_spec.params}")
            print(f"Layer Parents: {layer_spec.parents}")
            print(f"Layer Inbounds: {layer_spec.inbounds}")
            print(f"Layer Outbounds: {layer_spec.outbounds}")
            print(f"Layer Device: {layer_spec['device']}")
            print(layer_spec) #layer_spec.name
            print(layer_spec.operation) #output [1, 1000]
            print(layer_spec.operation.name) #output
            print(layer_spec.operation.outputs) #[1, 1000]
            print("------")
        print("#######")
    '''
    ''' -------------------------------------------------E N D WORK HERE '''
    for graph in task_priority :
        task_iner_priority.append(iner_sorting(graph,aver_peak_gflops,aver_bandwidth))
    return task_iner_priority

def iner_sorting(graph,aver_peak_gflops,aver_bandwidth):

    incoming = {}
    for layer_spec in graph.topological_order:
        incoming[layer_spec] = 0
        # print(incoming[layer_spec])
    # print(*incoming)
    for node in reversed(graph.topological_order):
        '''
        print('node',node)
        print(node.operation.outputs)
            node pool1
            [1, 55, 55, 64]
        '''
        d = calculate_tensor_size(node.operation.outputs, dtype='float32')
        gflops=FlopsProfiler.profile(node)
        ''' -------------------------------------------------Output S T A R T WORK HERE '''
        '''
        print(gflops)
        print(d/ 2 ** 30 /aver_bandwidth)
        print(gflops/aver_peak_gflops)
        print((gflops/aver_peak_gflops)/(d/ 2 ** 30 /aver_bandwidth))
        print("^^^^^^^^^^^^^^")

        # ^^^^^^^^^^^^^^
        # 0.236027904
        # 0.006202980324074074
        # 6.752418763946673e-05
        # 0.010885765246973622
        # ^^^^^^^^^^^^^^

        '''
        ''' -------------------------------------------------E N D WORK HERE '''
        
        # print(d/ 2 ** 30 /aver_bandwidth)
        # print(gflops/aver_peak_gflops)
        theta=(gflops/aver_peak_gflops)/(d/ 2 ** 30 /aver_bandwidth)
        # print('theta',theta)

        '''
        theta 0.04265442709018236
        theta 0.04265442709018236
        theta 0.0
        '''
        if 1-pow(1000,-theta)<np.random.rand():
          prob=0
        else:
          prob=1
        # print('prob',prob)
        max=0
        for m in node.outbounds:
               grade=incoming[m]+prob*(d/2 ** 30 /aver_bandwidth)
               if grade>=max:
                   max=grade
        incoming[node]=max+gflops/aver_peak_gflops
        # print('grade',incoming[node])
    a1 = sorted(incoming.items(), key=lambda x: x[1],reverse=True)
    task_iner_priority = [ k for k, v in dict(a1).items()]
    # print(task_iner_priority)
    return task_iner_priority


