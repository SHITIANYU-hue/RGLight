"""Contains an experiment class for running simulations."""
import os 
import sys
from sumolib import checkBinary
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import collections
import logging
import datetime
import numpy as np
import time
import random
import traci
import math
from AGENT.Class import Agent
from LANE.Class import Lane
from EDGE.Class import Edge
from flow.core.util import emission_to_csv
from utils.new_functions import *
from utils.gen_model import * 
import pickle
import networkx as nx
import dgl
from torch.utils.data import DataLoader
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
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import shutil
import signal



def init_exp(self):
    
    signal.signal(signal.SIGALRM, timeout_handler)
    
    if self.env.env_params.additional_params['save_render'] :
        suffix = '/' + str(tested) + '/'
        self.env.rendering_path = str("sumo_rendering" + "/" + self.env.env_params.additional_params['tb_foldername'] + "/" + self.env.env_params.additional_params['tb_filename'] + suffix)
    
    np.random.seed(seed)
    torch.manual_seed(seed)        
    self.env.tested = tested
    self.env.tested_end = tested_end        
    self.env.seed = seed
    self.env.eps_threshold = 1
    self.env.capture_counter = 0
    self.env.global_reward = 0
    self.env.env_params.additional_params['mode'] = mode
    self.env.n_workers = n_workers
    self.env.memory_queue = memory_queue
    self.env.request_end = request_end
    self.env.learn_model_queue = learn_model_queue
    self.env.comput_model_queue = comput_model_queue
    self.env.reward_queue = reward_queue
    self.env.baseline_reward_queue = baseline_reward_queue
    self.env.greedy_reward_queue = greedy_reward_queue            
    self.env.baseline = False
    self.env.greedy = False



    if self.env.tested == 'greedy':
        self.env.greedy = True
    elif self.env.tested != None:
        self.env.baseline = True
    if type(self.env.tested) == int:
        self.env.env_params.additional_params["min_time_between_actions"] = self.env.tested
        self.env.env_params.additional_params["yellow_duration"] = self.env.tested

    if seed == 0:      
        self.env = init(self.env) # TAKES CARE OF SAVE/LOAD    TRAIN/TEST
        comput_model_queue.put(self.env.model) # SEND THE ENV/MODEL TO THE COMPUT
        learn_model_queue.put(self.env.model) # SEND THE ENV/MODEL TO THE LEARN       

    # EVERY RUN     
    self.env.graph_of_interest = self.env.env_params.additional_params["graph_of_interest"]
    graph_of_interest = self.env.graph_of_interest
    self.actions_counts = collections.OrderedDict()           
    for i in range(self.env.env_params.additional_params['n_actions']):
        self.actions_counts[i] = 0

    self.env.step_counter = 0
    self.env.all_graphs = []

    self.env.last_lane = {}
    for idx,tl_id in enumerate(self.env.Agents):
        try:
            self.env.traci_connection.trafficlight.unsubscribe(tl_id)
        except:
            pass
        self.env.traci_connection.trafficlight.subscribe(tl_id, [traci.constants.TL_CURRENT_PHASE, traci.constants.TL_RED_YELLOW_GREEN_STATE]) 
        self.env.Agents[tl_id].current_phase_idx = self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[traci.constants.TL_CURRENT_PHASE]
        self.env.Agents[tl_id].current_phase = self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[traci.constants.TL_RED_YELLOW_GREEN_STATE]    
        self.env.Agents[tl_id].max_idx = len(self.env.traci_connection.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]._phases) 
        self.env.Agents[tl_id].cycle_duration = self.env.Agents[tl_id].max_idx -1               

        self.env.Agents[tl_id].time_since_last_action = 0
        self.env.Agents[tl_id].reset_rewards()

    for lane_id in self.env.Lanes:
        try:
            self.env.traci_connection.lane.unsubscribe(lane_id)
        except:
            pass
        self.env.traci_connection.lane.subscribe(lane_id, [traci.constants.LAST_STEP_VEHICLE_ID_LIST, traci.constants.LANE_LINKS])    
        self.env.Lanes[lane_id].reset_rewards()   


        
        
def step(self):# INITIALIZE A NEW STEP 
    
    # RESET TRAFFIC LIGHTS 
    reset_agents()
    reset_lanes()
    self.env.reward_vector = [0] * len(self.env.Lanes)
    
    create_current_graphs() # COPIES THE ORIGINAL GRAPHS IN ORDER TO CREATE GRAPHS SPECIFIC TO THE CURRENT TIMESTEP TO WHICH WE WILL ADD VEHICLES
    
    gen_vehicles() # GENERATE CURRENT TRAFFIC/VEHICLES
    add_vehicles() # UPDATE STATE/GRAPHS BY ADDING GENERATED VEHICLES
    
    # INITIALIZE NODE REPRESENTATIONS
    init_nodes_reps()
    
    # UPDATES VEH AND LANE DATA     AND   COMPUTES REWARDS
    update_veh_lane_reward()
    
    # ACCUMULATE CURRENT REWARD ON EXPERIENCE REWARD
    if self.env.step_counter > self.env.env_params.additional_params["wait_n_steps"]:  
        current_reward = self.env.reward_vector.sum()
        self.env.global_reward += current_reward
        
    update_tl_edge_connection_phase() # UPDATE OTHER NODES DATA
        
    update_agents() #UPDATE TIME SINCE LAST ACTION AND INFO REQUIRED FOR BENCHMARK POLICIES

    if self.env.env_params.additional_params['print_graph_state'] and self.env.seed == 0 and (self.env.step_counter +1) % 100 ==0 : 
        print_graph_state()
        
        
    # SELECT ACTIONS AND UPDATE SIMULATOR / SUMO
    _ = self.env.step(self.rl_actions())

    
    # IF TIME TO SEND A BATCH 
    if (self.env.step_counter - self.env.env_params.additional_params["wait_n_steps"]) % self.env.env_params.additional_params["num_steps_per_experience"] == 0  and self.env.step_counter >= (self.env.env_params.additional_params["wait_n_steps"] + self.env.env_params.additional_params["num_steps_per_experience"]):                
        if self.env.env_params.additional_params['mode'] == 'test' :
            self.env.tested_end.send(self.env.global_reward)
        else:
            send_experience_batch()
            

        self.env.global_reward = 0 
        self.env.all_graphs = []
    
    
    self.env.step_counter +=1
    
def reset_agents(self):
    self.env.ignored_vehicles = collections.OrderedDict()  

    for tl_id in self.env.Agents:
        self.env.Agents[tl_id].nb_stop_inb = 0
        self.env.Agents[tl_id].nb_mov_inb = 0                
        self.env.Agents[tl_id].current_phase_idx = self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[traci.constants.TL_CURRENT_PHASE]
        self.env.Agents[tl_id].current_phase = self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[traci.constants.TL_RED_YELLOW_GREEN_STATE] 

        # LAST BECOMES FIRST ("ROTATION IN THE CYCLE")

        if self.env.env_params.additional_params["graph_of_interest"] == "tl_connection_lane_graph" or self.env.env_params.additional_params["graph_of_interest"] == 'full_graph':
            while self.env.Agents[tl_id].phases_defs[-1] != self.env.Agents[tl_id].current_phase:
                self.env.Agents[tl_id].phases_defs.insert(0,self.env.Agents[tl_id].phases_defs.pop())

    if not self.env.tested == 'classic':
        self.env.traci_connection.trafficlight.setPhaseDuration(tl_id, 10000)
        
        
def create_current_graphs(self):
    self.env.current_graphs = {}
    for graph_name, graph in self.env.original_graphs.items():
        if graph_name == self.env.env_params.additional_params['graph_of_interest']:
            self.env.current_graphs[graph_name] = copy.deepcopy(graph)


def reset_lanes(self):
    for lane_id in self.env.Lanes:
        self.env.Lanes[lane_id].reset()
    self.env.busy_lanes = set()    
    
            
def gen_vehicles(self):
    if (self.env.step_counter % (self.env.env_params.additional_params["demand_duration"])) == 0 or self.env.step_counter == 1:
        update_gen_probs() # PERIODICALLY UPDATES THE PROBABILITIES OF TRAFFIC GENERATION TO ENSURE VARIABILITY IN DEMAND 
                
    for i in range(self.env.env_params.additional_params["N_VEH_SAMPLES"]):
        sample = np.random.uniform()
        if sample <= (self.env.env_params.additional_params["PROB_VEH"] *(1+self.env.demand_adjuster)):
            self.env.number_of_vehicles+=1
            veh_id = str("veh_" + str(self.env.number_of_vehicles))

            # RANDOMLY SAMPLE TRAJECTORY UNTIL WE GET A VALID ONE 
            valid = False
            while not valid:
                entering_edge = np.random.choice(list(self.env.entering_edges), p = self.env.new_entering_edges_probs)
                leaving_edge = np.random.choice(list(self.env.leaving_edges), p = self.env.new_leaving_edges_probs)
                if entering_edge != str('-' + leaving_edge) and leaving_edge != str('-' + entering_edge):
                    valid = True


                # IF NOT VALID WE CHANGE THE PROBS (CANNOT GET OUT OF THE SOFTMAX LOOP OTHERWISE)
                else:
                    self.env.entering_adjuster = np.absolute(np.random.normal(loc=0, scale = self.env.env_params.additional_params["lane_demand_variance"], size=len(self.env.entering_edges)))
                    self.env.leaving_adjuster = np.absolute(np.random.normal(loc=0, scale = self.env.env_params.additional_params["lane_demand_variance"], size=len(self.env.leaving_edges)))
                    self.env.new_entering_edges_probs =  self.env.entering_edges_probs * (1+self.env.entering_adjuster) 
                    self.env.new_entering_edges_probs = self.env.new_entering_edges_probs -  self.env.new_entering_edges_probs.max()
                    self.env.new_leaving_edges_probs = self.env.leaving_edges_probs * (1+self.env.leaving_adjuster)
                    self.env.new_leaving_edges_probs = self.env.new_leaving_edges_probs - self.env.new_leaving_edges_probs.max()
                    self.env.new_entering_edges_probs = np.exp(self.env.new_entering_edges_probs)/np.exp(self.env.new_entering_edges_probs).sum()
                    self.env.new_leaving_edges_probs = np.exp(self.env.new_leaving_edges_probs)/np.exp(self.env.new_leaving_edges_probs).sum()                            



            trip_name = str("route_" + entering_edge + "_" + leaving_edge)

            self.env.traci_connection.vehicle.add(vehID = veh_id, routeID = str(trip_name + "_" + str(np.random.choice(range(len(self.env.shortest_paths[trip_name]))))))
            self.env.traci_connection.vehicle.subscribe(veh_id, [traci.constants.VAR_LANE_ID, traci.constants.VAR_NEXT_TLS,
                traci.constants.VAR_LANE_INDEX, traci.constants.VAR_LANEPOSITION, traci.constants.VAR_ROAD_ID, traci.constants.VAR_MAXSPEED,
                traci.constants.VAR_SPEED, traci.constants.VAR_EDGES, traci.constants.VAR_POSITION, traci.constants.VAR_SIGNALS])
            self.env.traci_connection.vehicle.subscribeLeader(veh_id, 2000)



        
def update_gen_probs(self):

    self.env.demand_adjuster = np.absolute(np.random.normal(loc=0, scale=self.env.env_params.additional_params["demand_variance"], size=None))
    self.env.entering_adjuster = np.absolute(np.random.normal(loc=0, scale = self.env.env_params.additional_params["lane_demand_variance"], size=len(self.env.entering_edges)))
    self.env.leaving_adjuster = np.absolute(np.random.normal(loc=0, scale = self.env.env_params.additional_params["lane_demand_variance"], size=len(self.env.leaving_edges)))
    self.env.new_entering_edges_probs =  self.env.entering_edges_probs * (1+self.env.entering_adjuster) 
    self.env.new_entering_edges_probs = self.env.new_entering_edges_probs -  self.env.new_entering_edges_probs.max()
    self.env.new_leaving_edges_probs = self.env.leaving_edges_probs * (1+self.env.leaving_adjuster)
    self.env.new_leaving_edges_probs = self.env.new_leaving_edges_probs - self.env.new_leaving_edges_probs.max()
    self.env.new_entering_edges_probs = np.exp(self.env.new_entering_edges_probs)/np.exp(self.env.new_entering_edges_probs).sum()
    self.env.new_leaving_edges_probs = np.exp(self.env.new_leaving_edges_probs)/np.exp(self.env.new_leaving_edges_probs).sum()



def add_vehicles(self):

    for veh_id in self.env.traci_connection.vehicle.getIDList():
        lane_id = self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[traci.constants.VAR_LANE_ID]
        self.env.ignored_vehicles[veh_id] = False 
        if lane_id in self.env.lanes :
            self.env.last_lane[veh_id]= lane_id                 
        elif self.env.env_params.additional_params['ignore_central_vehicles']:
            self.env.ignored_vehicles[veh_id] = True


    if self.env.env_params.additional_params["veh_as_nodes"]:
        for idx, (graph_name, graph) in enumerate(self.env.current_graphs.items()):
            if graph_name == self.env.env_params.additional_params['graph_of_interest']:
                count = 0
                if "lane" in graph_name:
                    src = []
                    dst = []
                    tp = []
                    norm = []
                    node_type = []
                    for veh_id in self.env.traci_connection.vehicle.getIDList():


                        lane_id = self.env.last_lane[veh_id]

                        if veh_id not in graph.adresses_in_graph and lane_id in graph.adresses_in_graph:

                            #REGISTER CAR IN GRAPHS
                            veh_graph_id = graph.number_of_nodes()+count
                            current_length = len(graph.norms[lane_id])
                            graph.adresses_in_graph[veh_id] = veh_graph_id
                            graph.norms[veh_id] = [0]*current_length
                            graph.adresses_in_sumo[str(veh_graph_id)] = veh_id
                            node_type.append(len(graph.nodes_types))

                            #ADD SELF LOOP WITH TYPE AT THE END 
                            src.append(graph.adresses_in_graph[veh_id])
                            dst.append(graph.adresses_in_graph[veh_id])
                            graph.norms[veh_id][-1] += 1     
                            tp.append(len(graph.norms[veh_id])-1)   
                            #CAR TO LANE - EDGE
                            src.append(int(veh_graph_id))
                            dst.append(int(graph.adresses_in_graph[lane_id]))
                            #LANE TO CAR - EDGE
                            src.append(int(graph.adresses_in_graph[lane_id]))
                            dst.append(int(veh_graph_id))
                            graph.norms[lane_id][-3] +=1
                            tp.append(int(current_length-3))
                            graph.norms[veh_id][-2] +=1
                            tp.append(int(current_length-2))


                        count +=1         



                    if count>0:
                        for destination, t in zip(dst,tp):
                            norm.append([(1/(graph.norms[graph.adresses_in_sumo[str(destination)]][t]))])            
                        self.env.number_of_cars = count
                        src = torch.LongTensor(src)
                        dst = torch.LongTensor(dst)
                        edge_type = torch.LongTensor(tp)
                        edge_norm = torch.FloatTensor(norm)

                        graph.add_nodes(self.env.number_of_cars)
                        graph.add_edges(src,dst) #, {'rel_type':edge_type, 'norm':edge_norm})

                        graph.edata['rel_type'] = torch.cat((graph.edata['rel_type'][:-len(src)],torch.LongTensor(edge_type).squeeze()),0).squeeze()
                        graph.edata['norm'] = torch.cat((graph.edata['norm'][:-len(src)],torch.FloatTensor(edge_norm).squeeze()),0).squeeze()
                        graph.ndata['id'] = torch.eye(graph.number_of_nodes())    

                        graph.ndata['node_type'] = torch.cat((graph.ndata['node_type'][:-self.env.number_of_cars],torch.LongTensor(node_type)),0) 

                    new_filt = partial(filt, identifier = graph.nodes_types['veh'])
                    graph.nodes_lists['veh'] = graph.filter_nodes(new_filt)






def init_nodes_reps(self):
    for graph_name, graph in self.env.current_graphs.items():
        if graph_name == self.env.env_params.additional_params['graph_of_interest']:
            graph.ndata.update({'state' : torch.zeros(graph.number_of_nodes(),self.env.node_state_size, dtype = torch.float32)})

            
            
def update_veh_lane_reward(self):

    for veh_id in self.env.traci_connection.vehicle.getIDList():


        
        # GET RELEVANT  VEHICLE VARIABLES
        lane_id = self.env.last_lane[veh_id]
        veh_speed = self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[traci.constants.VAR_SPEED]
        veh_max_speed = self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[traci.constants.VAR_MAXSPEED]
        veh_position = self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[traci.constants.VAR_LANEPOSITION]


        # UPDATE REWARD PER LANE AND LANE INFORMATION
        self.env.Lanes[lane_id].update_lane_state_and_reward(veh_speed, veh_max_speed, veh_position, self.env.env_params.additional_params["veh_state"], self.env.ignored_vehicles[veh_id])
        self.env.busy_lanes.add(lane_id)                     

        # UPDATE VEH NODE DATA
        if self.env.env_params.additional_params['veh_as_nodes']:# and not self.env.ignored_vehicles[veh_id]:                 
            counter = 0           
            for var_name, var_dim in self.env.env_params.additional_params['veh_vars'].items():
                var = self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[var_name]                        
                if var_name == traci.constants.VAR_LANEPOSITION :
                    if self.env.ignored_vehicles[veh_id]:
                        var = var + self.env.Lanes[lane_id].length
                    var/= self.env.env_params.additional_params['max_lane_length']
                elif var_name == traci.constants.VAR_SPEED:
                    var /= self.env.env_params.additional_params['Max_Speed']

                for graph_name, graph in self.env.current_graphs.items():
                    if graph_name == self.env.env_params.additional_params['graph_of_interest']:
                        if 'lane' in graph_name:

                            # CONTINUOUS VARIABLE

                            if var_dim == 1 :
                                graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter] = var
                            # DUMMY VARIABLE
                            elif var_dim > 1: 
                                if var_name == traci.constants.VAR_SIGNALS:
                                    if var == 2 or var == 10: # 2 = LEFT # 10 = LEFT+BRAKE
                                        graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + 0] = 1                                                
                                        if var == 10:
                                            graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + 2] = 1      
                                    elif var == 1 or var == 9: # 1 = RIGHT # 9 = RIGHT + BRAKE
                                        graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + 1] = 1
                                        if var == 9:
                                            graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + 2] = 1
                                    elif var == 8: # 8 = BRAKE 
                                        graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + 2] = 1
                                else:
                                    graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + var] = 1


                counter += var_dim






    # UPDATE LANE NODE DATA
    for lane_id in self.env.Lanes:

        if self.env.env_params.additional_params['lane_node_state']:  
            for graph_name, graph in self.env.current_graphs.items():
                if graph_name == self.env.env_params.additional_params['graph_of_interest']:
                    if 'x' in self.env.env_params.additional_params['lane_vars']:
                        graph.ndata['state'][graph.adresses_in_graph[lane_id]][-3] = float(lane_id[-3]) #X  
                    if 'y' in self.env.env_params.additional_params['lane_vars']:                
                        graph.ndata['state'][graph.adresses_in_graph[lane_id]][-2]  = float(lane_id[-5]) #Y
                    if 'which_lane' in self.env.env_params.additional_params['lane_vars']:
                        graph.ndata['state'][graph.adresses_in_graph[lane_id]][-1] = float(lane_id[-1]) # WHICH LANE

                if 'type' in self.env.env_params.additional_params['lane_vars']:
                    if 'bot' in lane_id:
                        graph.ndata['state'][graph.adresses_in_graph[lane_id]][-4]  = 1                      
                    if 'top' in lane_id:
                        graph.ndata['state'][graph.adresses_in_graph[lane_id]][-5]  = 1    
                    if 'left' in lane_id:
                        graph.ndata['state'][graph.adresses_in_graph[lane_id]][-6]  = 1    
                    if 'right' in lane_id:
                        graph.ndata['state'][graph.adresses_in_graph[lane_id]][-7]  = 1                                                                 
                    if 'lane' in graph_name or graph_name == 'full_graph':                           
                        for idx, (var_name, lane_state_var) in enumerate(self.env.env_params.additional_params['lane_vars'].items()):
                            if var_name == 'nb_veh':
                                graph.ndata['state'][graph.adresses_in_graph[lane_id]][idx] = self.env.Lanes[lane_id].state[0]                                          
                            elif var_name =='avg_speed':
                                graph.ndata['state'][graph.adresses_in_graph[lane_id]][idx] = self.env.Lanes[lane_id].state[1]                                                                                 
                            elif var_name =='length':
                                graph.ndata['state'][graph.adresses_in_graph[lane_id]][idx] = self.env.Lanes[lane_id].std_length                                                                               


        # UPDATE REWARD VECTOR WITH REWARD PER LANE AND CORRESPONDING DISCOUNT VECTORS
        self.env.reward_vector[list(self.env.Lanes.keys()).index(lane_id)] = -self.env.Lanes[lane_id].delay


    
    self.env.reward_vector = np.asarray(self.env.reward_vector)


def update_tl_edge_connection_phase(self):


    for tl_id in self.env.Agents:
        counter = 0
        for var_name, var_dim in self.env.env_params.additional_params['tl_vars'].items():
            if type(var_name) is int:
                var = self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[var_name]
                for graph_name, graph in self.env.current_graphs.items():  
                    if graph_name == self.env.env_params.additional_params['graph_of_interest']:
                        if len(graph.nodes_lists['tl']) > 0 : 
                            # CONTINUOUS VARIABLE
                            if var_dim == 1 :
                                graph.ndata['state'][graph.adresses_in_graph[tl_id]][counter] = var
                            # DUMMY VARIABLE
                            elif var_dim > 1: 
                                graph.ndata['state'][graph.adresses_in_graph[tl_id]][counter + var] = 1



                counter += var_dim


    for tl_idx, tl_id in enumerate(self.env.Agents):
        for graph_name, graph in self.env.current_graphs.items():  
            if graph_name == self.env.env_params.additional_params['graph_of_interest']:

                if 'time_since_last_action' in self.env.env_params.additional_params['tl_vars']:
                    graph.ndata['state'][graph.adresses_in_graph[tl_id]][-1] = self.env.Agents[tl_id].time_since_last_action 

                if 'x' in self.env.env_params.additional_params['tl_vars']:
                    graph.ndata['state'][graph.adresses_in_graph[tl_id]][-2] = int(tl_id[6:])%self.env.env_params.additional_params["row_num"]   
                if 'y' in self.env.env_params.additional_params['tl_vars']:
                    graph.ndata['state'][graph.adresses_in_graph[tl_id]][-3] = int(tl_id[6:])//self.env.env_params.additional_params["col_num"]  

                if graph_name == 'tl_connection_lane_graph' or graph_name == "full_graph":
                    for link_idx,link in enumerate(self.env.Agents[tl_id].unordered_connections_trio):
                        link_name = str(tl_id+"_link_"+str(link_idx))

                        if self.env.Agents[tl_id].current_phase[link_idx].lower() == 'g':   
                            if 'open' in self.env.env_params.additional_params['connection_vars']:
                                graph.ndata['state'][graph.adresses_in_graph[link_name]][-1] = 1
                        if self.env.Agents[tl_id].current_phase[link_idx] == 'G':  
                            if 'current_priority' in self.env.env_params.additional_params['connection_vars']:
                                graph.ndata['state'][graph.adresses_in_graph[link_name]][-3] = 1
                            # else 0

                        elif 'nb_switch_to_open' in self.env.env_params.additional_params['connection_vars']:
                            for idx,phase in enumerate(self.env.Agents[tl_id].phases_defs):
                                if phase[link_idx].lower() == 'g':                      
                                    graph.ndata['state'][graph.adresses_in_graph[link_name]][-2] = (idx+1)  
                                    if 'priority_next_open' in self.env.env_params.additional_params['connection_vars']:
                                        graph.ndata['state'][graph.adresses_in_graph[link_name]][-4] = 1       

                                    break

                        if self.env.env_params.additional_params['phase_state']:

                            for idx,phase in enumerate(self.env.Agents[tl_id].phases_defs):
                                if (idx+1) <= self.env.env_params.additional_params['num_observed_next_phases']:
                                    #print("tl_id", tl_id, "link_idx", link_idx, "link", link, "phase", phase)
                                    if phase[link_idx].lower() == 'g':
                                        graph.ndata['state'][graph.adresses_in_graph[link_name]][4*idx] = 1                

                                    if phase[link_idx] == 'G':
                                        graph.ndata['state'][graph.adresses_in_graph[link_name]][4*idx+1] = 1          
                                    elif phase[link_idx].lower() == 'y':
                                        graph.ndata['state'][graph.adresses_in_graph[link_name]][4*idx+2] = 1
                                    elif phase[link_idx].lower() == 'r':
                                        graph.ndata['state'][graph.adresses_in_graph[link_name]][4*idx+3] = 1                      


def update_agents(self):
    for tl_id in self.env.Agents:              
        self.env.Agents[tl_id].time_since_last_action +=1
    for lane_id in self.env.Agents[tl_id].inb_lanes:
        self.env.Agents[tl_id].nb_stop_inb += self.env.Lanes[lane_id].nb_stop_veh
        self.env.Agents[tl_id].nb_mov_inb += self.env.Lanes[lane_id].nb_mov_veh          


def send_experience_batch(self):
    # Q LEARNING
    if self.env.env_params.additional_params['Policy_Type'] == "Q_Learning":
        for idx, (g, labels, actions, choices, forced_0, forced_1, tl_graph) in enumerate(reversed(self.env.all_graphs)): # choices
            labels = np.asarray(labels) 
            if idx == 0:
                reward = labels
            elif idx < len(self.env.all_graphs):
                reward = labels + (self.env.env_params.additional_params['time_gamma'] * reward)

                # extend with FUTURE REWARD, NEXT STATE (GRAPH), IMMEDIATE REWARD t-1, NEXT_CHOICES, NEXT_FORCED_0, NEXT_FORCED_1

                self.env.all_graphs[-(idx)].extend([list(reward), self.env.all_graphs[-(idx-1)][0], self.env.all_graphs[-(idx+1)][1],self.env.all_graphs[-(idx-1)][3], self.env.all_graphs[-(idx-1)][4],self.env.all_graphs[-(idx-1)][5]])

        if self.env.env_params.additional_params['mode'] == "train":
            signal.alarm(60)  
            try:    
                if self.env.greedy:
                    self.env.greedy_reward_queue.put(float(self.env.global_reward))
                else:
                    self.env.memory_queue.send(self.env.all_graphs[1:-1])
                    self.env.reward_queue.send(float(self.env.global_reward))
            except TimeoutException:
                print("TIMEOUT !!!!!! ")
            signal.alarm(0)