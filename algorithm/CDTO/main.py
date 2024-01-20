import argparse
import random
import numpy as np
from algorithm import DeviceGraph
import time
from algorithm.CDTO.common import generate_diff_task,generate_same_task,task_sort,random_task
from algorithm.CDTO.cdto import  Simulator_greedy
from algorithm.CDTO.base_local import  Simulator_local
from algorithm.CDTO.base_cloud import  Simulator_cloud
from algorithm.CDTO.local_cloud import  Simulator_local_cloud
from algorithm.CDTO.base_ergodic import  Simulator_ergodic
from genetic_algorithm import GAOptimizer
def four_schemes(parse):

    local_time=[]
    cloud_time=[]
    cdto_time=[]
    local_cloud_time=[]
    ergodic_time=[]
    user=10
    time_start = time.time()  # 记录开始时间
    for i in range(1):
        task=random_task(parse.num,user) 
        # print(task)
        '''
        list of 10 users with there random task ranging from 0to3
        {0: [0, 1, 0, 2, 2, 0, 0, 3, 0, 2, 0, 2, 0, 3, 2, 3, 0, 3, 1, 2, 3, 2, 2, 0, 2], 1: [2, 1, 2, 3, 0, 0, 3, 1, 3, 3, 3, 0, 0, 3, 0, 3, 0, 0, 1, 2, 0, 3, 3, 3, 2], 2: [1, 2, 2, 0, 0, 2, 1, 2, 3, 1, 2, 3, 0, 1, 0, 2, 2, 2, 0, 1, 1, 1, 3, 1, 2], 3: [2, 2, 0, 3, 1, 1, 2, 0, 1, 1, 2, 1, 3, 2, 2, 3, 3, 0, 3, 3, 3, 1, 1, 2, 3], 4: [3, 3, 3, 3, 1, 2, 1, 1, 0, 2, 3, 2, 3, 2, 2, 1, 3, 2, 1, 2, 1, 1, 2, 2, 3], 5: [3, 3, 1, 3, 3, 2, 0, 3, 0, 2, 0, 2, 1, 3, 2, 0, 1, 0, 3, 3, 2, 3, 2, 2, 3], 6: [0, 2, 3, 1, 3, 1, 0, 3, 3, 0, 3, 1, 1, 0, 2, 3, 0, 1, 0, 1, 1, 3, 1, 3, 1], 7: [2, 2, 0, 0, 3, 2, 1, 2, 3, 2, 1, 2, 2, 3, 0, 3, 2, 3, 0, 1, 3, 2, 3, 0, 2], 8: [2, 3, 0, 1, 3, 2, 3, 2, 1, 1, 1, 2, 3, 3, 1, 2, 3, 0, 0, 1, 1, 0, 1, 3, 2], 9: [2, 1, 1, 2, 0, 0, 0, 0, 1, 3, 1, 2, 3, 0, 3, 1, 0, 1, 0, 1, 1, 3, 2, 0, 0]}
        '''
        task_unit=int(parse.num/user)

        #print(task_num)
        device_graph = DeviceGraph.load_from_file( parse.device_graph_path)
        # print(len(device_graph.devices)) #35 ranges from 0 to 34

        ''' -------------------------------------------------Output S T A R T WORK HERE '''
        
        # for device_node in device_graph.devices:
        #     device = device_node.device
        #     print(f"Device ID: {device_node.id}")
        #     print(f"Model: {device.model}")
        #     print(f"Clock: {device.clock} MHz")
        #     print(f"Peak GFLOPS: {device.peak_gflops}")
        #     print(f"Memory: {device.memory} GB")
        #     print(f"Memory Bandwidth: {device.mem_bandwidth} GB/s")
        #     print(f"Type: {device.type}")
        #     print(f"Hardware ID: {device.hardware_id}")
        #     print(f"Is GPU: {device.is_gpu}")
        #     print(f"Relate: {device.relate}")
        #     print("Neighbours:")
        #     for neighbor, channel in device_node.neighbours.items():
        #         print(f"  - Neighbour ID: {neighbor.id}, Communication Channel ID: {channel.id}, Hop: {device_node.neighbourshop[neighbor]}")
        #     print("------------------------------")

        # for comm_channel in device_graph.comm_channels:
        #     print(f"Communication Channel ID: {comm_channel.id}")
        #     print(f"Type: {comm_channel.type}")
        #     print(f"Bandwidth: {comm_channel.bandwidth} Gb/s")
        #     print("------------------------------")


        ''' -------------------------------------------------E N D WORK HERE '''
        if parse.diff_task:#是否是相同任务
            task, task_dl=generate_diff_task(user,parse.num,task,parse.task_release_time,parse.net_graph_path1,parse.net_graph_path2,parse.net_graph_path3,parse.net_graph_path4)
        else:
            task, task_dl=generate_same_task(user,parse.num,task,parse.task_release_time,parse.net_graph_path1)

        '''
        task_dl is [500, 510, 520, 530, 540, 550, 560, ..., 2980, 2990], this list is having 250 elements
        '''

        ''' -------------------------------------------------Output S T A R T WORK HERE '''
        # print(type(task))
        # print(len(task)) #250
        # print(dir(task)) #listFunction is printed
        # print(type(task[0])) #--------------------------------algorithm.graph.ComputationGraph
        # print(*task, sep='\n')
        # for user_tasks in task:
        #     print(type(user_tasks))
        #     for layer_spec in user_tasks.topological_order:
        #         print(f"Layer Name: {layer_spec.name}")
        #         print(f"Layer Type: {layer_spec['type']}")
        #         print(f"Layer Parameters: {layer_spec.params}")
        #         print(f"Layer Parents: {layer_spec.parents}")
        #         print(f"Layer Inbounds: {layer_spec.inbounds}")
        #         print(f"Layer Outbounds: {layer_spec.outbounds}")
        #         print(f"Layer Device: {layer_spec['device']}")  #********Result is either 0 or 1 like device 0 and device 1
        #         print("------")
        #     print("#######")
        
        # for layer_spec in task[0].topological_order:
        #     print(layer_spec.name)

        '''Layer Name: res5a_branch1
        Layer Type: Convolution
        Layer Parameters: {'type': 'Convolution', 'parents': ['res4f'], 'filter': [1, 1, -1, 512], 'strides': [1, 2, 2, 1], 'padding': 'VALID', 'device': 0}
        Layer Parents: [res4f]
        Layer Inbounds: [res4f]
        Layer Outbounds: [res5a]
        Layer Device: 0
        ------
        '''

        # print(len(task_dl))
        # print(*task_dl, sep='\n')
        
        ''' -------------------------------------------------E N D WORK HERE '''


        '''
        we calculate the priority of each subtask using a probabilistic b-level algorithm
        Since it is uncertain to which device the subtask is offloaded, _ B and _ p represent the average computing 
        capability of all devices and the average data rate of all links, respectively.

        '''
        task_iner_priority=task_sort(device_graph,task, task_dl)
        # print(dir(task_iner_priority))  #listFunction is printed
        # for graph in task_iner_priority:
        #    print(type(graph))
        #    print("^^^^^^^^^^^^^^^^^^^^^^") 
        # print(len(task_iner_priority[0]))
        # print(type(task_iner_priority[0])) #--------------------------------LIST
        # print(len(task_iner_priority[1]))
        # print(len(task_iner_priority[2]))
        # print(len(task_iner_priority[30]))
        # print(len(task_iner_priority[60]))
        # print(len(task_iner_priority[10]))
        # print(len(task_iner_priority[249]))
        # print(len(task_iner_priority)) #250 JOBS

        # print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        # for graph in task_iner_priority[0]:
        #    print(graph)

        # https://www.diffchecker.com/j8sRleww/  DiffCheck
        '''
        Sorting is Done till here
        task_iner_priority: Store tasks list
        device_graph: Store status of ARPANET with device and its neighbour device along with communication channels
        task_unit: num/user = 250/10 = 25
        '''

        '''
        tasks_inner_priority have list of layer for each taks, there are 250 tasks and each tasks have its sublayer.
        task_inner_priority have list of each layer as there name

        task_inner_priority=[[data, conv1-1, conv1-2, pool1, conv2-1, fc7, dropout7, fc8, softmax, output],
                            [data, conv1, pool1, res2a_branch2a, res4c, res5c, pool5, fc1000, loss, output],
                            [data, conv1, pool1, res2a_branch2a,  fc1000, loss, output]]
        '''
        task_dim=[]
        task_dim=[]
        '''
        for i in range(len(task_iner_priority)):
            task_dim.append(len(task_iner_priority[i]))
        print(*task_dim)
        print(len(task_dim))

        '''
        time_end1 = time.time()  # 记录结束时间
        time_sum1= time_end1 - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum1)
        ##local scheme#######################################################

        simulator1 = Simulator_local(task_iner_priority, device_graph,task_unit)
        task_finish_time2=simulator1.simulate()
        # print(task_finish_time2)
        local_time.append(max(task_finish_time2))
        print("##############################")
        time_end2 = time.time()  # 记录结束时间
        time_sum2= time_end2- time_end1  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum2)

        #cloud  scheme#######################################################
        simulator2 = Simulator_cloud(task_iner_priority, device_graph,task_unit)
        for i in range(127):
            # print(i)
            if i==0 or i==126:
                device = [int(task/ task_unit) for task in range(len(task_iner_priority))]

            else:
                 # print(i)
                 device = [len(device_graph.devices)-1 for task in range(len(task_iner_priority))]

                 # device = devices[i-1]
            task_finish_time3=simulator2.simulate(device)

        cloud_time.append(max(task_finish_time3))
        print("##############################")
        time_end3 = time.time()  # 记录结束时间
        time_sum3= time_end3- time_end2  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum3)

        #CDTO scheme#######################################################
        simulator3 = Simulator_greedy(task_iner_priority, device_graph,task_unit)
        task_finish_time4 = simulator3.simulate()
        # print(task_finish_time4)
        cdto_time.append(max(task_finish_time4))
        time_end4 = time.time()  # 记录结束时间
        time_sum4= time_end4- time_end3  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum4)
        print("##############################")
        #local_cloud scheme#######################################################
        simulator4 = Simulator_local_cloud(task_iner_priority, device_graph,task_unit)
        task_finish_time5 = simulator4.simulate()
        local_cloud_time.append(max(task_finish_time5))
        time_end5 = time.time()  # 记录结束时间
        time_sum5= time_end5- time_end4  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum5)
         ########################################################

        #ergodic  scheme#######################################################
        #'''
        optimizer = GAOptimizer(plot_fitness_history=True,
                                generations= 5,
                                population_size= 100,
                                mutation_rate=0.8,
                                zone_mutation_rate=0.2,
                                mutation_sharding_rate=0,
                                crossover_rate=0.8,
                                crossover_type="1-point",
                                parent_selection_mechanism="rank",
                                evolve_mutation_rate=True,
                                verbose=5,
                                elite_size=5,
                                max_mutation_rate=0.9,
                                min_mutation_rate=0.05,
                                print_diversity=True,
                                include_trivial_solutions_in_initialization=False,
                                allow_cpu= True,
                                pipeline_batches=2,
                                batches=10,
                                n_threads=-1,
                                checkpoint_period=5,
                                simulator_comp_penalty=0.9,
                                simulator_comm_penalty=0.25)
        best_solution, best_placement =optimizer.optimize(net_len=127-2,
                                        task_num=parse.num,
                                        n_devices=len(device_graph.devices),
                                        task_iner_priority=task_iner_priority,
                                        device_graph=device_graph,
                                        task_unit=task_unit)
        # print(best_solution)
        # print(best_placement)
        

        print("##############################")

        #'''

    print('local_time',np.average(local_time))
    print('cloud_time',np.average(cloud_time))
    print('local_cloud_time', np.average(local_cloud_time))
    print('network_time', np.average(cdto_time))
    # # print('ergodic_time1', ergodic_time)
    # print('ergodic_time', min(ergodic_time))
if __name__ == '__main__':
    np.random.seed(3)
    random.seed(3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_graph_path", type=str, default='configs/device_graphs/arpanet.json')
    parser.add_argument("--diff_task", type=bool, default=True)
    parser.add_argument("--num", type=list, default=250)
    parser.add_argument("--task_release_time", type=int, default=500,help='ms')
    parser.add_argument("--net_graph_path1", type=str, default='nets/alex_v2.json')
    parser.add_argument("--net_graph_path2", type=str, default='nets/inception_v3.json')
    parser.add_argument("--net_graph_path3", type=str, default='nets/resnet34.json')
    parser.add_argument("--net_graph_path4", type=str, default='nets/vgg16.json')

    args = parser.parse_args()
    four_schemes(args)









