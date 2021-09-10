#from flow.envs.base_env import Env
#from AGENT.Connection_Value_NN import Connection_Value_NN
#from AGENT.Connections_Mixing_NN import Connections_Mixing_NN
import collections
import traci
import numpy as np
import torch

class Agent():
    # CREATE THE NNs HERE TO BE SHARED BY ALL AGENTS
    connections_state = []
    # CONNECTION NN 
     #Connection_Value_NN = Connection_Value_NN(input_size = , nb_hidden_layers, output_size, dropout = False)
    # MIX NN 
     #Connections_Mixing_NN = Connections_Mixing_NN(input_size, nb_hidden_layers, output_size, dropout = False)
    # POLICY NN 
    # MEMORY BUFFER
    #memory = Memory(m, max_buffer_size)
    
    
    def __init__(self, tl_id):
        #self, tl_id, discount_vector, inbound_lanes, outbound_lanes, connections): # OPTIONNAL : DISTANCE VECTOR 
        #1) INIT 
        self.reward = 0
        self.state = 0
        self.connection_rewards = collections.OrderedDict()
        self.connection_values = collections.OrderedDict()
        self.agent_id = tl_id
        self.inb_lanes = []
        self.outb_lanes = []
        self.connections_trio = []
        self.connections_info = []
        self.complete_controlled_lanes = []
        self.is_time_to_choose = False    
        
        
    def reset(self):
        return 0
    
    def reset_rewards(self):
        self.reward = 0 
        self.connection_rewards = collections.OrderedDict()
        for connection in self.connections_trio:
            self.connection_rewards[connection[0][0]] = 0 
            
    def get_reward(self, reward_vector):
        self.reward =  np.matmul(self.discount_vector,reward_vector)
        return self.reward
