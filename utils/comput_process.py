"""Contains an experiment class for running simulations."""

import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random
import collections
import logging
import datetime
import numpy as np
import time
import traci
#from utils.rl_tools import *
import pickle
import networkx as nx
import dgl
from torch.utils.data import DataLoader
import sys
import torch
import copy
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix
import pylab as pl
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from statistics import mean 

def comput(request_ends, comput_model_queue,comput_model2_queue, baseline_reward_queue = None, greedy_reward_queue = None,  tested_learner_ends = None, tested = None, reward_queues = None):
    # COMMON
    #print("reward_queues", reward_queues)
    #print("START COMPUTE PROCESS")
    device = "cpu"
    #print("WAITING FOR MODEL")
    env_params = comput_model_queue.get() # BEFORE BEGINNING, WAIT FOR A FIRST VERSION OF THE MODEL 
    env_params2 = comput_model2_queue.get() # BEFORE BEGINNING, WAIT FOR A FIRST VERSION OF THE MODEL 

    #print("MODEL LOADED")
    if env_params.additional_params['mode'] == 'train' or not env_params.additional_params['sequential_computation']:
        ## confuse
        model = comput_model_queue.get() # BEFORE BEGINNING, WAIT FOR A FIRST VERSION OF THE MODEL 
        model2 = comput_model2_queue.get() # BEFORE BEGINNING, WAIT FOR A FIRST VERSION OF THE MODEL 
        model.to(device) # GET THE MODEL 
        model2.to(device) # GET THE MODEL 
        if env_params.additional_params['Policy_Type'] == "Q_Learning":
            model.eval()
        #model = nn.DataParallel(model)
    else:
        models = collections.OrderedDict()
        for idx, request_end in request_ends.items():
            model = request_end.recv()
            if idx == 0:
                env_params = env_params
            if type(model) != str :
                model.to(device)
                if env_params.additional_params['Policy_Type'] == "Q_Learning":
                    model.eval()
                models[idx] = model

        models2 = collections.OrderedDict()
        for idx, request_end in request_ends.items():
            model = request_end.recv()
            if idx == 0:
                env_params = env_params
            if type(model) != str :
                model.to(device)
                if env_params.additional_params['Policy_Type'] == "Q_Learning":
                    model.eval()
                models2[idx] = model2               
                
        #for param1,param2 in zip(models[0].conv_layers[0].layers['message_module'].parameters(), models[1].conv_layers[0].layers['message_module'].parameters()):
            #print("diff", param1 == param2)
    counter = 0 
    
    #print("COMPUTE PROCESS STARTED SUCCESSFULY")

    

    #if env_params.additional_params['mode'] == 'test':
        #resquest_ends = request_ends[0]
    """
        request_ends.pop(0)
        if memory_queues:
            memory_queues.pop(0)
        if reward_queues:
            reward_queues.pop(0)
    """


    
    
    if env_params.additional_params['Policy_Type'] == "Q_Learning":
        #model.eval()
        with torch.no_grad():    
            while True:
                
                #print('START LOAD COMPUTE')
                if env_params.additional_params['mode'] == "train":
                    # UPDATE MODEL
                    if counter % env_params.additional_params['update_comput_model_frequency'] == 0 :
                        if not comput_model_queue.empty():
                            try:
                                model = comput_model_queue.get_nowait() # UPDATE MODEL EVERY TIME 
                                model.eval()
                                model.to(device)
                                #print("UPDATE MODEL")
                            except:
                                pass



                # GET THE REQUEST
                compute_idx = []
                random_idx = []
                graphs_list = []
                lengths_list = []
                batched_state = []
                actions_sizes = []
                
                
                
                #print('START COMPUTATION')
                
                 
                if env_params.additional_params['mode'] == 'train' or not env_params.additional_params['sequential_computation']:   
                    
                    #while len(lengths_list) < env_params.additional_params['n_workers'] :
                    for idx, request_end in request_ends.copy().items():
                        #if request_end.poll():
                        request = request_end.recv()
                        #print("request", request)
                        if type(request) == str :
                            if request == 'Done':
                                del request_ends[idx]
                            else:
                                random_idx.append(idx)
                        else:
                            if env_params.additional_params['GCN']:                        
                                graphs_list.append(request[0])
                            elif not env_params.additional_params['GCN']:
                                batched_state.append(request[0])
                            lengths_list.append(request[1])
                            actions_sizes.append(request[2])
                            compute_idx.append(idx)


                    if graphs_list or batched_state: # CHECK THAT IT'S NOT EMPTY

                        if env_params.additional_params['GCN']:
                            mode='boltzmann_multi' 
                            n_graphs = len(graphs_list)
                            batched_graph = dgl.batch(graphs_list, node_attrs = ['state', 'node_type'], edge_attrs = ['rel_type', 'norm'])
                            graphs_list = []
                            def average_prob(q1,q2):
                                '''
                                Takes three different probability vectors in and outputs a randomly
                                sampled action from n_action with probability equals the average \
                                probability of the input vectors
                                '''
                                Q = (q1 + q2)/2
                                return Q

                            def boltzmann_prob(q1, q2, T=5):
                                '''
                                Takes three different probability vectors in and outputs a randomly
                                Sampled action from n_action with probability equals the average
                                probability of the normalized exponentiated input vectors, with a
                                temperature T controlling the degree of spread for the out vector
                                '''
                                boltz_ps = [np.exp((prob/T).cpu().numpy())/sum(np.exp((prob/T).cpu().numpy())) for prob in [q1,q2]]
                                # print(sum(q2))
                                # a=0
                                # for prob in [q1,q2]:
                                #     a=a+np.exp(prob/T)

                                # a1=np.exp((q1/T))
                                # # print(a.size())
                                # a2=np.exp((q2/T))
                                # print((q1/a1+q2/a2)/a)

                                q1=torch.Tensor(boltz_ps[0])
                                q2=torch.Tensor(boltz_ps[1])
                                # print('q1',q1)
                                # print('q2',q2)
                                #print('botlze',q1.size())
                                # Q = (boltz_ps[0] + boltz_ps[1])/2
                                Q=(q1+q2)/2
                                print('Q',Q.size())
                                return Q

                            def boltzmann_multi(q1, q2, T=5):
                                '''
                                Takes three different probability vectors in and outputs a randomly
                                Sampled action from n_action with probability equals the average
                                probability of the normalized exponentiated input vectors, with a
                                temperature T controlling the degree of spread for the out vector
                                '''
                                boltz_ps = [np.exp((prob/T).cpu().numpy())/sum(np.exp((prob/T).cpu().numpy())) for prob in [q1,q2]]
                                # print(sum(q2))
                                # a=0
                                # for prob in [q1,q2]:
                                #     a=a+np.exp(prob/T)

                                # a1=np.exp((q1/T))
                                # # print(a.size())
                                # a2=np.exp((q2/T))
                                # print((q1/a1+q2/a2)/a)

                                q1=torch.Tensor(boltz_ps[0])
                                q2=torch.Tensor(boltz_ps[1])
                                # print('q1',q1)
                                # print('q2',q2)
                                #print('botlze',q1.size())
                                # Q = (boltz_ps[0] + boltz_ps[1])/2
                                Q=q1*q2
                                # print('Q',Q.size())
                                return Q
                            def weight_add(q1, q2, T=5):
                                '''
                                Takes three different probability vectors in and outputs a randomly
                                Sampled action from n_action with probability equals the average
                                probability of the normalized exponentiated input vectors, with a
                                temperature T controlling the degree of spread for the out vector
                                '''
                                boltz_ps = [sum(np.exp((prob/T).cpu().numpy())) for prob in [q1,q2]]
                                # print(sum(q2))
                                # a=0
                                # for prob in [q1,q2]:
                                #     a=a+np.exp(prob/T)

                                # a1=np.exp((q1/T))
                                # # print(a.size())
                                # a2=np.exp((q2/T))
                                # print((q1/a1+q2/a2)/a)

                                q1w=torch.Tensor(boltz_ps[0])/(torch.Tensor(boltz_ps[0])+torch.Tensor(boltz_ps[1]))
                                q2w=torch.Tensor(boltz_ps[1])/(torch.Tensor(boltz_ps[0])+torch.Tensor(boltz_ps[1]))
                                # print(q1w+q2w)
                                # print('q1',q1)
                                # print('q2',q2)
                                #print('botlze',q1.size())
                                # Q = (boltz_ps[0] + boltz_ps[1])/2
                                Q=q1w*q1+q2w*q2
                                # print('Q',Q)
                                return Q




                            s = time.time()
                            hid, Q_values = model.forward(batched_graph, device, testing = True if env_params.additional_params['mode'] == 'test' else False, actions_sizes = torch.cat(tuple(actions_sizes),dim=0))
                            hid2, Q_values2 = model2.forward(batched_graph, device, testing = True if env_params.additional_params['mode'] == 'test' else False, actions_sizes = torch.cat(tuple(actions_sizes),dim=0))
                            mode=env_params.additional_params['combination']
                            ## add up
                            if mode == 'addup':
                                print('mode: add up')
                                #print('q value',Q_values.size())
                                #print('q value2',Q_values2.size())
                                Q_values= average_prob(Q_values,Q_values2)
                                #print('new q value',Q_values.size())

                            ## boltzman
                            if mode == 'boltzman':
                                print('mode: boltzman')
                                #print('q value',Q_values.size())
                                #print('q value2',Q_values2.size())
                                Q_values= boltzmann_prob(Q_values,Q_values2)

                            ## weight add
                            if mode == 'weight_add':
                                print('mode: weight_add')
                                #print('q value',Q_values.size())
                                #print('q value2',Q_values2.size())
                                Q_values= weight_add(Q_values,Q_values2)

                            if mode == 'boltzmann_multi':
                                print('mode: boltzmann_multi')
                                #print('q value',Q_values.size())
                                #print('q value2',Q_values2.size())
                                Q_values= boltzmann_multi(Q_values,Q_values2)

                                #print('new q value',Q_values.size())                           
                            # print(Q_values.size())
                            # print(Q_values2.size())
                            # print(Q_values==Q_values2)
                            # # hid=hid.mean(dim=1)
                            # Q_values=Q_values.mean(dim=1)
                            ##calculate avg n_tau, Q value
                            ##it should have correct Q values
                            ##why Q_value dim is still [159, 2], hid is [159, 8, 32]
                            # print('hid size',hid.size())
                            # print("Q_values size", Q_values.size())
                            #print("Q_values", Q_values[0])
                            if env_params.additional_params['gaussian_mixture']:
                                Q_values = sample(Q_values[0],Q_values[1],Q_values[2])
                            if env_params.additional_params['policy'] == 'binary':
                                _ , Q_values = torch.max(Q_values, dim =1)
                                # print('action',Q_values)
                                _2 , Q_values2 = torch.max(Q_values2, dim =1)

                                # print('Q_values1',Q_values)
                                # print('Q_values2',Q_values2)
                                
                                Q_values = Q_values.split(lengths_list, dim = 0)
                            else:
                                if len(lengths_list) == 1:
                                    Q_values = [Q_values]
                                else:
                                    Q_values = Q_values.split(lengths_list, dim = 0)
                            #Q_values = Q_values.chunk(n_graphs, dim = 0)

                        elif not env_params.additional_params['GCN']:
                            n = len(batched_state)
                            _ , Q_values = model.forward(batched_state, device, testing = True if env_params.additional_params['mode'] =='test' else False, actions_sizes = torch.cat(tuple(actions_sizes),dim=0))
                            
                            
                            if env_params.additional_params['policy'] == 'binary':
                                #print("Q_VALUES OUTPUT", Q_values.size())
                                _ , Q_values = torch.max(Q_values, dim =1)
                            Q_values = Q_values.view(n, model.n_tls)           
                            #Q_values = Q_values.view(model.n_tls,n)
                            #Q_values = torch.cat(tuple([Q_values[:,idx].view(1,model.n_tls,1) for idx in range(n)]),0)

                            #print("Q_values", Q_values)


                    # SEND RANDOM FOR OTHERS
                    for idx in random_idx:
                        request_ends[idx].send('N/A')
                        #answer_queues[idx].put(torch.randint(low=0, high=2, size=(env_params.additional_params['row_num']*env_params.additional_params['col_num'],1)).squeeze())


                    #print('Q_values', Q_values.size())

                    # SEND COMPUTED RESULT
                    for result_idx, request_idx in enumerate(compute_idx):
                        #print('result_idx', result_idx)
                        #print('Q_values[result_idx]', Q_values[result_idx].size())
                        request_ends[request_idx].send(Q_values[result_idx].squeeze().cpu().numpy())             



                            
                 
                # WHEN USING SEQUENTIAL COMPUTATION 
                else : 
                    for idx, request_end in request_ends.copy().items():
                        #if request_end.poll():
                        request = request_end.recv()
                        #print("request", request)
                        if type(request) == str :
                            if request == 'Done':
                                del request_ends[idx]
                            else:
                                request_end.send('N/A')
                        else:                                          
                            s = time.time()
                            if env_params.additional_params['GCN']:
                                hid, Q_values = models[idx].forward(request[0], device, testing = True if env_params.additional_params['mode'] == 'test' else False, actions_sizes = request[2])
                                # print("Q_values", Q_values)
                                #print("Q_values", Q_values[0])
                                
                                if env_params.additional_params['policy'] == 'binary':
                                    # print('q value',Q_values)
                                    _ , Q_values = torch.max(Q_values, dim =1)
                                #Q_values = Q_values.split(lengths_list, dim = 0)
                                #Q_values = Q_values.chunk(n_graphs, dim = 0)

                            elif not env_params.additional_params['GCN']:
                                #n = len(batched_state)
                                _ , Q_values = models[idx].forward(request[0], device, testing = True if env_params.additional_params['mode'] =='test' else False, actions_sizes = request[2])
                                if env_params.additional_params['policy'] == 'binary':
                                    _ , Q_values = torch.max(Q_values, dim =1)
                                #Q_values = Q_values
                                
                                
                            #print("Q values size", Q_values.size())
                                
                                
                            request_end.send(Q_values.view(1,-1).squeeze().cpu().numpy())
                            #print("GPU TIME", time.time() -s, " sec")
                            #time.sleep(120)
                                #Q_values = Q_values.view(model.n_tls,n)
                                #Q_values = torch.cat(tuple([Q_values[:,idx].view(1,model.n_tls,1) for idx in range(n)]),0)

                                #print("Q_values", Q_values)



        

    elif 'critic' in env_params.additional_params['Policy_Type'].lower() and 'actor' in env_params.additional_params['Policy_Type'].lower():       

        
        
         
    ######################################################################################################################################3           
        
        
        
        def compute_gae(rewards, values, gamma=0.99, tau=0.95):
            gae = 0
            returns = []
            for step in reversed(range(len(rewards))):
                #print("rewards[step]", rewards[step])
                #print("values[step][0]", values[step][0])
                delta = rewards[step] + gamma * values[step + 1] - values[step]
                gae = delta + gamma * tau * gae
                #print("gae", gae.size())
                #print("gae[0]", gae[0])
                #print("values[step]", values[step].size())
                returns.insert(0, gae + values[step])
            return torch.stack(returns).to("cpu")

        def ppo_iter(gcn, mini_batch_size, states, actions, log_probs, returns, advantage):
            if gcn:
                #print("len(states)", len(states))
                idxs = list(range(len(states)-mini_batch_size)) 
            else:
                idxs = list(range(states.size()[0]-mini_batch_size))
            #batch_size = len(states)
            #print("b_s", batch_size)
            #print("mini_batch_size", mini_batch_size)
            #print("states", states.size())
            idxs = np.random.choice(idxs, size = 1, replace = False).tolist()
            #idxs = np.random.choice(idxs, size = batch_size // mini_batch_size, replace = False).tolist()
            for idx in idxs:
                #print("a")
                #yield states[idx:idx+mini_batch_size] , actions[idx:idx+mini_batch_size].to(device), log_probs[idx:idx+mini_batch_size].to(device), returns[idx:idx+mini_batch_size].to(device), advantage[idx:idx+mini_batch_size].to(device)        
                for idx2 in range(mini_batch_size):   
                    #print("b")
                    
                    if gcn:
                        #print([states[i] for i in range(idx, idx+idx2)])
                        #b_graph = dgl.batch([states[i] for i in range(idx, idx+idx2)])
                        b_graph = dgl.batch([states[idx+idx2]])
                        if idx2 == 0:
                            b_graph = model.init_hidden(b_graph, device)
                        yield b_graph , actions[idx+idx2].to(device), log_probs[idx+idx2].to(device), returns[idx+idx2].to(device), advantage[idx+idx2].to(device)    
                    else:
                        b_state = states[idx+idx2]
                        if idx2 == 0:
                            model.init_hidden(b_state, device)
                        yield states[idx+idx2], actions[idx+idx2].to(device), log_probs[idx+idx2].to(device), returns[idx+idx2].to(device), advantage[idx+idx2].to(device) 

            if False:
                batch_size = len(states)-1
                rand_ids_ts = np.random.choice(range(batch_size), size =(batch_size // mini_batch_size, mini_batch_size), replace = False)
                for rand_ids in rand_ids_ts:
                    rand_ids = list(rand_ids)
                    yield dgl.batch([states[idx] for idx in rand_ids]) if gcn else states[rand_ids, :], actions[rand_ids, :].to(device), log_probs[rand_ids, :].to(device), returns[rand_ids, :].to(device), advantage[rand_ids, :].to(device)


        def ppo_update(policy, gcn, ppo_epochs, accumulation_steps, mini_batch_size, states, actions, actions_sizes, log_probs, returns, advantages, clip_param=0.2):
            Critic_Loss = 0 
            Actor_Loss = 0 
            loss = 0
            model.optimizer.zero_grad()
            for ppo_epoch in range(ppo_epochs):
                for idx, (state, action, old_log_prob, return_, advantage) in enumerate(ppo_iter(gcn, mini_batch_size, states, actions, log_probs, returns, advantages)):
                    #print("state size",state.size())
                    _, dist, value = model.forward(state, device, learning =  False, testing = False, actions_sizes = torch.cat(tuple(actions_sizes),dim=0))
                    if policy != 'binary':
                        actions_sizes_list = actions_sizes.tolist()[0]
                        new_log_probs = torch.zeros(len(actions_sizes_list), dtype=torch.float32, device=device)  #      [0]*len(actions_sizes_list)
                        #print("actions", actions)
                        for dim in list(set(actions_sizes_list)):
                            positions = [i for i, n in enumerate(actions_sizes_list) if n == dim] 
                            dist_ = torch.cat(tuple(dist[i].view(1,-1) for i in positions),dim=0)
                            actions_ = torch.cat(tuple(action[i].view(1,-1) for i in positions),dim=0)
                            probs_init = F.softmax(dist_, dim = 1)#.view(mini_batch_size,-1,2)
                            log_probs_init = F.log_softmax(dist_, dim = 1)#.view(mini_batch_size,-1,2)
                            #print("log_probs_init", log_probs_init.size())
                            #print("actions_", actions_.size())
                            new_log_probs_ = log_probs_init.gather(1, Variable(actions_).view(-1,1).to(device))#.view(mini_batch_size, -1, 1)))                    
                            for idx,position in enumerate(positions):
                                new_log_probs[position] = new_log_probs_[idx]    
                    else:
                        probs_init = F.softmax(dist, dim = 1)#.view(mini_batch_size,-1,2)
                        log_probs_init = F.log_softmax(dist, dim = 1)#.view(mini_batch_size,-1,2)
                        #print("lg size", log_probs_init.size())
                        #print("action", action.size())
                        new_log_probs = log_probs_init.gather(1, Variable(action).view(-1,1))#.view(mini_batch_size, -1, 1)))

                    ratio = (new_log_probs.squeeze() - old_log_prob.type(new_log_probs.dtype)).exp()
                    surr1 = ratio * advantage.type(ratio.dtype)
                    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                    actor_loss  = - torch.min(surr1, surr2).mean()
                    #print("return", return_.size())
                    #print("return[0]", return_[0])
                    #print("value[0]", value.view(-1)[0])
                
                    critic_loss = (return_ - value.view(-1)).pow(2).mean()
                    #print("critic_loss", critic_loss)
                    #sys.exit()
                    Actor_Loss += actor_loss.item()
                    Critic_Loss += critic_loss.item()
                    loss += 1 * critic_loss + actor_loss #- 0.01 * entropy
                    loss /= env_params.additional_params['n_workers']
                    Actor_Loss /= env_params.additional_params['n_workers']
                    Critic_Loss /=env_params.additional_params['n_workers']
                    if (idx+1) % env_params.additional_params['BPTT_steps'] == 0 :
                        loss /= env_params.additional_params['BPTT_steps']
                        loss.backward()
                        loss.detach()
                        loss = 0
                        model.hid = Variable(model.hid.detach())
                    
                    if env_params.additional_params['GCN']:
                        pass
                        #print("grads before", model.conv_layers[0].layers['message_module'].weight_message_input.grad)
                    else:
                        pass
                        #print("grads before", model.weight_inp.grad)                
                #loss /= mini_batch_size
                Actor_Loss /= mini_batch_size
                Critic_Loss /= mini_batch_size
                #loss.backward()
                if env_params.additional_params['GCN']:
                    pass
                    #print("grads after", model.conv_layers[0].layers['message_module'].weight_message_input.grad)
                else:
                    pass
                    #print("grads after", model.weight_inp.grad)
                #print("EPOCH")
                if (ppo_epoch+1) % accumulation_steps == 0:
                    #print("EPOCH")
                    #sys.exit()
                    model.optimizer.step()
                    model.optimizer.zero_grad()
                #loss = 0
                
                    
            Actor_Loss /= ppo_epochs
            Critic_Loss /= ppo_epochs

            return Actor_Loss, Critic_Loss

        
        
        
    ######################################################################################################################################3    

    
        train_counter = 0
        writer = SummaryWriter(env_params.additional_params["tb_foldername"])
        while True:
            reward_to_save = None
            log_probs_list = []
            values_list    = []
            states_list    = []
            actions_list   = []
            rewards_list  = []
            entropy = 0   
            counter = 0
            

            step_counter = 0
            for _ in range(env_params.additional_params['num_steps_per_experience']) :

                #if counter % env_params.additional_params["clear_after_n_epoch"] == 0:  
                    #clear_output(wait=True)   
                exp = []
                random_idx = []
                para_rewards_list = []
                para_states_list = []
                para_actions_sizes_list = []
                sizes = []
                graphs_list = []
                lengths_list = []
                batched_state = []
                actions_sizes = []
                batched_rewards = []
                actions_s_list = []
                #RECEIVE REQUESTS
                compute_idx = []
                #while len(lengths_list) < env_params.additional_params['n_workers'] :
                for idx, request_end in request_ends.items():
                    #if request_end.poll():
                    request = request_end.recv()
                    if type(request) == str :
                        random_idx.append(idx)
                    else:
                        if env_params.additional_params['GCN']:                        
                            graphs_list.append(request[0])
                        elif not env_params.additional_params['GCN']:
                            batched_state.append(request[0])
                        lengths_list.append(request[1])
                        actions_sizes.append(request[2])
                        batched_rewards.append(torch.FloatTensor(list(request[3].values())))
                        #print("br", batched_rewards)
                        #print("b_r", batched_rewards)
                        compute_idx.append(idx)

                                
                #print("comput_idx", compute_idx)
                sizes.append(len(batched_rewards))
                para_rewards_list.append(torch.cat((batched_rewards),0))  
                para_actions_sizes_list.append(torch.cat((actions_sizes),0))
                if not env_params.additional_params['GCN']:
                    para_states_list.append(torch.FloatTensor(batched_state))

                                
                #print(para_states_list)
                #print("5")
                # TIME FOR COMPUT
                if graphs_list or batched_state: # CHECK THAT IT'S NOT EMPTY
                    with torch.no_grad():
                        if env_params.additional_params['GCN']:
                            #print(graphs_list)
                            n_graphs = len(graphs_list)
                            batched_graph = dgl.batch(graphs_list, node_attrs = ['state', 'node_type'], edge_attrs = ['rel_type', 'norm'])
                            if step_counter == 0:
                                model.init_hidden(batched_graph, device)
                            #print("model.hid1", model.hid)
                            _, dist, values = model.forward(batched_graph, device, learning =  False, testing = True if env_params.additional_params['mode'] == 'test' else False, actions_sizes = torch.cat(tuple(actions_sizes),dim=0))
                            
                            #print("model.hid", model.hid)
                            #print(graphs_list)
                            graphs_list = []
                            states_list.append(batched_graph)


                        elif not env_params.additional_params['GCN']:
                            n = len(batched_state)
                            if step_counter == 0:
                                model.init_hidden(batched_state, device)
                            #print("model.hid1", model.hid)
                            _, dist, values = model.forward(batched_state, device, learning =  False, testing = True if env_params.additional_params['mode'] == 'test' else False, actions_sizes = torch.cat(tuple(actions_sizes),dim=0)) 

                            #states_list.append(batched_state)
                            batched_state = torch.cat((para_states_list),0)
                            states_list.append(batched_state.detach().cpu().squeeze())
                            #print(states_list)
                        #print("model.hid2", model.hid)
                        #print("states_list", states_list)
                        #print("sizes", sizes)
                        #print(values.size())

                        #print("dist", dist)

                        if env_params.additional_params['policy'] != 'binary':
                            actions_sizes_list = tuple(torch.cat(tuple(actions_sizes),dim=0).tolist())  
                            log_probs = torch.zeros(len(actions_sizes_list), dtype=torch.float64, device=device)  #      [0]*len(actions_sizes_list)
                            actions = torch.zeros(len(actions_sizes_list), dtype=torch.int8, device=device)       # [0]*len(actions_sizes_list)
                            for dim in list(set(actions_sizes_list)):
                                positions = [i for i, n in enumerate(actions_sizes_list) if n == dim] 
                                dist_ = torch.cat(tuple(dist[i].view(1,-1) for i in positions),dim=0)
                                probs_init_ = F.softmax(dist_, dim = 1) 
                                log_probs_init_ = F.log_softmax(dist_, dim = 1)
                                actions_ = probs_init_.multinomial(num_samples=1).data                    
                                actions_ = actions_.type(torch.LongTensor).to(device)
                                log_probs_ = log_probs_init_.gather(1, Variable(actions_))
                                for idx,position in enumerate(positions):
                                    log_probs[position] = log_probs_[idx]
                                    actions[position] = actions_[idx]            

                        else:
                            probs_init = F.softmax(dist, dim = 1) 
                            log_probs_init = F.log_softmax(dist, dim = 1)
                            actions = probs_init.multinomial(num_samples=1).data                    
                            actions = actions.type(torch.LongTensor).to(device)
                            log_probs = log_probs_init.gather(1, Variable(actions))
                            #entropies = -(log_probs_init * probs_init)
                            #actions = actions.squeeze().chunk(n_graphs, dim = 0)



                        #print("actions", actions)

                        if env_params.additional_params['mode'] == 'train':    
                            batched_rewards = torch.cat((para_rewards_list), 0) 
                            actions_sizes = torch.cat((para_actions_sizes_list), 0) 

                            actions_s_list.append(actions_sizes.cpu().detach().squeeze())
                            actions_list.append(actions.cpu().detach().squeeze())
                            log_probs_list.append(log_probs.cpu().detach().squeeze())
                            values_list.append(values.cpu().detach().squeeze())
                            rewards_list.append(batched_rewards.detach().cpu().squeeze())

                        #print("sizes", sizes)
                        #print("actions", actions.squeeze().size())
                        #if env_params.additional_params['policy'] == 'binary':
                            #print("actions", actions)
                        actions = actions.squeeze().split(lengths_list, dim = 0)
                            #print("actions", actions)
                        #for result_idx, request_idx in enumerate(compute_idx):
                            #request_ends[request_idx].send(actions[result_idx].squeeze().cpu().numpy())             
                        #else:
                            #if len(lengths_list) == 1:
                                #Q_values = [Q_values]
                            #else:
                                #Q_values = Q_values.split(lengths_list, dim = 0)


                        for result_idx, request_idx in enumerate(compute_idx):
                            #print("actions[result_idx]",actions[result_idx])
                            request_ends[request_idx].send(actions[result_idx].squeeze().cpu())       
                            
                            
                step_counter +=1
                """
                probs_init = F.softmax(dist, dim = 1) # generating a distribution of probabilities of the Q-values according to 
                log_probs_init = F.log_softmax(dist, dim = 1) # generating a distribution of log probabilities of the Q-values according to the 
                actions = probs_init.multinomial(num_samples=1).data                    
                actions = actions.type(torch.LongTensor).to(device)
                log_probs = log_probs_init.gather(1, Variable(actions))
                #entropies = -(log_probs_init * probs_init)
                actions = actions.squeeze().chunk(n_graphs, dim = 0)
                log_probs = log_probs.squeeze().chunk(n_graphs, dim = 0)
                #entropies = entropies.squeeze().chunk(n_graphs, dim = 0)
                values = values.squeeze().chunk(n_graphs, dim = 0)

                
                # SEND COMPUTED RESULT
                for result_idx, answer_idx in enumerate(compute_idx):
                    answer_queues[answer_idx].put((actions[result_idx].squeeze().cpu(), log_probs[result_idx].squeeze().cpu(), entropies[result_idx].squeeze().cpu(), values[result_idx].squeeze().cpu()))
                """
            if env_params.additional_params['mode'] == 'train': 
                # TIME FOR TRAIN 
                #next_state = torch.FloatTensor(next_state).to(device)
                #_, next_value = model(next_state)
                #print("rewards_list[0]", rewards_list[0])
                #print("rewards size", rewards.size())
                #print("rewards_list", rewards_list)
                #print("values_list", values_list)
                values = torch.stack(values_list)
                rewards = torch.stack(rewards_list, 0).view(values.size()) 
                #print("rewards size", rewards.size())
                #print("rewards", rewards)
                #print("values", values.size())

                returns = compute_gae(rewards[:-1].to(device), values.to(device), env_params.additional_params['time_gamma'], env_params.additional_params['tau'])
                #returns = torch.cat(returns).detach()
                log_probs = torch.stack(log_probs_list)
                #values    = torch.cat(values_list)
                if env_params.additional_params['GCN']:
                    states = states_list
                else:
                    states = torch.stack(states_list)
                #states    = torch.cat(states_list)
                actions_sizes = torch.stack(actions_s_list)
                actions = torch.stack(actions_list).type(torch.LongTensor)
                advantages = (returns - values[:-1]).detach()
                
                #print("state1", states.size())
                #print("actions", actions.size())
                #print("advantages", advantages.size())
                #print("log_probs", log_probs.size())
                #print("values", values.size())

                Actor_Loss, Critic_Loss = ppo_update(env_params.additional_params['policy'], env_params.additional_params['GCN'], env_params.additional_params['ppo_epochs'], env_params.additional_params['accumulation_steps'],env_params.additional_params['mini_batch_size'], states, actions, actions_sizes, log_probs, returns, advantages, env_params.additional_params['ppo_clip_param'])

                """            
                value_loss = 0
                policy_loss = 0
                Value_loss = 0 
                Policy_loss = 0 
                # GET EXPERIENCES FOR DISTRIBUTED TREATMENT
                #print("exp", exp)
                rewards, log_probs, entropies, values = map(collate, zip(*exp))
                norm = rewards.numel()

                #writer.add_scalar('General_Value_Loss', (Value_loss/norm).item() , train_counter)
                #writer.add_scalar('General_Policy_Loss', (Policy_loss/norm).item(), train_counter)


                # TRAINING STEP WITH LAST GENERATED EXPERIENCES 
                R = values[-1].detach()
                gae = torch.zeros(1, 1)
                R = Variable(R) 
                for reverse_step in reversed(range(len(rewards)-1)): 
                    R = env_params.additional_params['time_gamma'] * R + rewards[reverse_step] 
                    advantage = R - values[reverse_step] 
                    value_loss = value_loss + 1 * advantage.pow(2)
                    Value_loss += value_loss.sum()   
                    TD = rewards[reverse_step] + env_params.additional_params['time_gamma'] * values[reverse_step+1].detach() - values[reverse_step].detach() 
                    gae = gae * env_params.additional_params['time_gamma'] * env_params.additional_params['tau'] + TD 
                    policy_loss = (policy_loss - log_probs[reverse_step] * Variable(gae) - 0.01 * entropies[reverse_step].mean()) # mean(dim = 2)
                    Policy_loss += policy_loss.sum()


                model.optimizer.zero_grad()
                ((Value_loss+Policy_loss)/norm).backward(retain_graph = False) 
                model.optimizer.step()
                model.optimizer.zero_grad()                

                writer.add_scalar('General_Value_Loss', (Value_loss/norm).item() , train_counter)
                writer.add_scalar('General_Policy_Loss', (Policy_loss/norm).item(), train_counter)

                """
            
                print("Epoch {:05d} | ".format(counter) + "Actor " +  "Loss: {:.4f} | ".format(Actor_Loss) + "Critic " +  "Loss: {:.4f} | ".format(Critic_Loss) )

                
                
                writer.add_scalar('Actor_Loss', Actor_Loss , train_counter)
                writer.add_scalar('Critic_Loss', Critic_Loss, train_counter)                
                
                
                #reward_to_save = ((rewards.sum().item())/env_params.additional_params['n_workers'])/(env_params.additional_params['num_steps_per_experience']
                if reward_to_save is not None:
                    if max_reward is None:
                        max_reward = reward_to_save

                    if reward_to_save >= max_reward:
                        max_reward = reward_to_save
                        #torch.save(model, model.env_params.additional_params['trained_model_path'] + "/model.pt")
                        torch.save(model.state_dict(), env_params.additional_params['save_model_path'] + "/params.pt")

                if counter% env_params.additional_params['save_params_frequency']==0:
                    torch.save(model.state_dict(), env_params.additional_params['save_model_path'] + "/params_checkpoint.pt")

            train_counter +=1
            counter +=1
            if (counter) % env_params.additional_params["clear_after_n_epoch"] == 0:  
                clear_output(wait=True)   

