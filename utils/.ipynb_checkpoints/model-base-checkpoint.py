import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch
import numpy as np 
import dgl
from functools import partial
from torch.autograd import Variable
import copy
from utils.mdn import *
from utils.graph_tools import *
from IPython.display import clear_output
import glob
import atexit
import time
import traceback

  
    
class Convolutional_Message_Passing_Framework(nn.Module):
    def __init__(self, num_attention_heads, bias_before_aggregation , nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, state_first_dim_only, gaussian_mixture, n_gaussians, value_model_based, use_attention,  separate_actor_critic, n_actions, rl_learner_type, std_attention, state_vars, n_convolutional_layers, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = 1, rel_num_bases = -1, norm = True, is_bias = True, activation = F.relu, use_message_module = True, use_aggregation_module = False, dropout = False, resnet = False, dueling = False):
        
        
        
        """
        use_prediction_module : if False, embeddings obtained after a given propagation are used as predicion values (ie, they are not transformed by additional layers)
        use_attention : if True, attention element-wise attention mechanisms are used to weight messages
        separate_actor_critic : if True, two separate architectures (one for the actor and one for the critic) perform propagations, with their own sets of parameters
        n_pred : number of values to predict 
        learner_type : if "actor" or "critic" or "actor-critic" specific behaviour will be specified in the fields between ############ Start and ############### End
        std_attention : if True, attention coefficients corresponding to messages of the same type leading to the same node will be normalized to sum up to one
        state_vars : list containing the names of variables representing the state (RL)
        n_convolutional_layers : number of layers with different parameters we want to use.
        num_nodes_types : number of different types of nodes which exist in the studied graphs 
        nodes_types_num_bases : if < 0 no basis decomposition is used for the nodes parameters (Gating/GRU). If > 0 there will be as many bases for node parameters as indicated (up to a maximum of 'num_nodes_types')
        node_state_dim : the size of the complete node representation (state_size + embedding_size)
        node_embedding_dim : the size of the node embedding (computed representation)
        num_rels : number of different types of edges which exist in the studied graphs
        rel_num_bases : if < 0 no basis decomposition is used for the edges parameters (message/request/attention). If > 0 there will be as many bases for edges parameters as indicated (up to a maximum of 'num_rels')
        n_hidden_message : number of hidden layers of neural networks used in the message module (to compute :message, request, attention) 
        n_hidden_aggregation : number of hidden GRU layers used to update node embeddings 
        n_hidden_prediction : number of hidden layers used to make a prediction in the prediction module from the embeddings of the nodes 
        hidden_layers_size : deprecated
        prediction_size : deprecated
        num_propagations : default number of propagations which are performed ON EVERY DIFFERENT LAYER during a forward pass if no 'num_propagations' argument is passed to the forward function.
        norm : if True, messages of the same type leading to the same node will be averaged 
        is_bias : if True, a bias will be included with every layer (message/request/attention/GRU/prediction...)
        activation: The activation function which will be used to add non linearity in the architecture
        use_message_module : deprecated
        use_aggregation_module : if False, the embedding is simply the aggregation of the received messages. if True, a GRU/Gate is used to compute the new embedding using the aggregation of the received messages and the previous embedding.
        dropout : if True, dropout will be used with the specified probability ( not yet properly implemented )
        resnet : if True, a residual connection is added between the previous embedding of a node and its new embedding (ie. the aggregation module only has to learn to output the relevant difference between the previous and new embedding)


        """        
        super(Convolutional_Message_Passing_Framework, self).__init__() 
        self.num_attention_heads = num_attention_heads
        self.bias_before_aggregation = bias_before_aggregation
        self.nonlinearity_before_aggregation = nonlinearity_before_aggregation
        self.share_initial_params_between_actions = share_initial_params_between_actions
        self.multidimensional_attention = multidimensional_attention
        self.state_first_dim_only = state_first_dim_only, 
        self.gaussian_mixture = gaussian_mixture
        self.n_gaussians = n_gaussians
        self.value_model_based = value_model_based
        self.use_attention = use_attention
        self.separate_actor_critic = separate_actor_critic
        self.n_actions = n_actions
        self.rl_learner_type = rl_learner_type
        self.std_attention = std_attention
        self.state_vars = state_vars 
        self.state_vars = state_vars
        self.n_convolutional_layers = n_convolutional_layers
        self.num_nodes_types = num_nodes_types 
        self.nodes_types_num_bases = nodes_types_num_bases
        self.node_state_dim = node_state_dim
        self.node_embedding_dim = node_embedding_dim
        self.num_rels = num_rels
        self.n_hidden_message = n_hidden_message 
        self.n_hidden_aggregation = n_hidden_aggregation 
        self.n_hidden_prediction = n_hidden_prediction
        self.hidden_layers_size = hidden_layers_size
        self.n_hidden_aggregation = n_hidden_aggregation
        self.prediction_size = prediction_size
        self.num_propagations = num_propagations
        self.rel_num_bases = rel_num_bases
        self.norm = norm 
        self.is_bias = is_bias 
        self.activation = activation
        self.use_message_module = use_message_module
        self.use_aggregation_module = use_aggregation_module
        self.dropout = dropout
        self.resnet = resnet 
        self.dueling = dueling
        
        
        self.conv_layers = nn.ModuleList()
        
        if self.dueling:
            self.rl_learner_type = 'actor_critic'
        if self.value_model_based:
            self.value_transition_model = nn.ModuleList()
        
        if 'actor' in self.rl_learner_type.lower() and 'critic' in self.rl_learner_type.lower() and self.separate_actor_critic :
            self.critic_conv_layers = nn.ModuleList()
            self.actor_conv_layers = nn.ModuleList()
        
            for i in range(self.n_convolutional_layers):
                
                
                if i < (self.n_convolutional_layers - 1):
                    critic_conv = Relational_Message_Passing_Framework(num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers,True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, is_bias = is_bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = False, dropout = dropout, resnet = resnet)
                    actor_conv = Relational_Message_Passing_Framework(num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers,True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, is_bias = is_bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = False, dropout = dropout, resnet = resnet)
                else:
                    critic_conv = Relational_Message_Passing_Framework(num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers, True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, n_actions, 'critic', std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, is_bias = is_bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = True, dropout = dropout, resnet = resnet)
                    actor_conv = Relational_Message_Passing_Framework(num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers, True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, n_actions, 'actor', std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, is_bias = is_bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = True, dropout = dropout, resnet = resnet)

                self.critic_conv_layers.append(critic_conv)
                self.actor_conv_layers.append(actor_conv)
                
    
    
    
        else: 
            for i in range(self.n_convolutional_layers):
                if i < (self.n_convolutional_layers - 1):
                    conv = Relational_Message_Passing_Framework(num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers, True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, n_actions, self.rl_learner_type, std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, is_bias = is_bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = False, dropout = dropout, resnet = resnet)
                else:
                    conv = Relational_Message_Passing_Framework(num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers, True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, n_actions, self.rl_learner_type, std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, is_bias = is_bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = True, dropout = dropout, resnet = resnet)

                self.conv_layers.append(conv)
            
            
    def forward(self, graph, device):
        # SEND TENSORS TO DEVICE 
        graph.ndata['state'].to(device)
        graph.edata['rel_type'].to(device)
        if self.norm:
            graph.edata['norm'].to(device)
        graph.ndata["node_type"].to(device)


        # INITIALIZATION OF NODE HIDDEN AND MEMORY HIDDEN

        graph.ndata.update({"hid":torch.zeros(graph.number_of_nodes(), self.node_embedding_dim, dtype = torch.float32)}) # requires_grad = True
        graph.ndata["hid"].to(device)

        graph.ndata.update({"memory_input":torch.zeros(graph.number_of_nodes(), self.node_embedding_dim, dtype = torch.float32)}) # requires_grad = True
        graph.ndata["memory_input"].to(device)    


        if self.n_hidden_aggregation > 0 :

            for idx in range(self.n_hidden_aggregation-1):
                graph.ndata.update({str('memory_' + str(idx)): torch.zeros(graph.number_of_nodes(),self.node_embedding_dim, dtype = torch.float32)})             
                graph.ndata[str('memory_' + str(idx))].to(device)

            graph.ndata.update({"memory_output":torch.zeros(graph.number_of_nodes(), self.node_embedding_dim, dtype = torch.float32)}) # requires_grad = True
            graph.ndata["memory_output"].to(device)
        
        
        
        
        if 'actor' in self.rl_learner_type.lower() and 'critic' in self.rl_learner_type.lower() and self.separate_actor_critic :  
            critic_graph = copy.deepcopy(graph)
            actor_graph = copy.deepcopy(graph)
            for critic_layer, actor_layer in zip(self.critic_conv_layers[:-1], self.actor_conv_layers[:-1]):
                critic_layer.forward(critic_graph, device)            
                actor_layer.forward(actor_graph, device)   
                
            if self.dueling:
                v = self.critic_conv_layers[-1].forward(critic_graph, device)[1]
                a = self.actor_conv_layers[-1].forward(actor_graph, device)[0]

                return (v.unsqueeze(1) + (a - torch.mean(a, dim = 1).unsqueeze(1))).squeeze()
            
            else:
                return self.actor_conv_layers[-1].forward(actor_graph, device), self.critic_conv_layers[-1].forward(critic_graph, device)
        
        
        
        
        elif 'actor' in self.rl_learner_type.lower() and 'critic' in self.rl_learner_type.lower() and not self.separate_actor_critic :  
            
            for layer in self.conv_layers[:-1]:
                layer.forward(graph, device)
            
            if self.dueling:
                hid, a, v = self.conv_layers[-1].forward(graph, device)
                if self.gaussian_mixture:
                    #print("pi v size",v[0].size(), "pi a size", a[0].size(), "pi mean a", torch.mean(a[0],dim=1).size())
                    #print("sigma v size",v[1].size(), "sigma a size", a[1].size())
                    #print("mu v size",v[2].size(), "mu a size", a[2].size())
                    pi = v[0] + (a[0] - torch.mean(a[0], dim = 1).unsqueeze(1))
                    if self.n_gaussians > 1 :
                        pi = F.softmax(pi, dim=1)
                    #print("softm pi :", pi)
                    sigma = v[1] + (a[1] - torch.mean(a[1], dim = 1).unsqueeze(1))
                    mu = v[2] + (a[2] - torch.mean(a[2], dim =1).unsqueeze(1))
                    return hid, [pi,sigma,mu]
                elif not self.gaussian_mixture:
                    return hid, (v.unsqueeze(1) + (a - torch.mean(a, dim = 1).unsqueeze(1))).squeeze()
                    
            else:
                
                print("NORMAL ACTOR CRITIC (NOT DUELING) HAS TO BE RE-IMPLEMENTEND")
                sys.exit()
        else:
            for layer in self.conv_layers[:-1]:
                layer.forward(graph, device)

            hid, Q = self.conv_layers[-1].forward(graph, device)
            return hid, Q 

            
        
            
    
    
    
    
    
    
class Relational_Message_Passing_Framework(nn.Module):
    def __init__(self, num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, state_first_dim_only, n_convolutional_layers, is_first_layer, gaussian_mixture, n_gaussians, value_model_based, use_attention, n_actions, rl_learner_type, std_attention, state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = 1, rel_num_bases = -1, norm = True, is_bias = True, activation = F.relu, use_message_module = True, use_aggregation_module = False, is_final_convolutional_layer = False, dropout = False, resnet = False):
        
        
        super(Relational_Message_Passing_Framework, self).__init__()
        self.is_first_layer = is_first_layer
        self.state_first_dim_only = state_first_dim_only
        self.n_convolutional_layers = n_convolutional_layers
        self.rl_learner_type = rl_learner_type
        self.state_vars = state_vars
        self.is_final_convolutional_layer = is_final_convolutional_layer
        self.prediction_size = prediction_size
        self.num_nodes_types = num_nodes_types
        self.nodes_types_num_bases = nodes_types_num_bases
        self.use_aggregation_module = use_aggregation_module
        self.node_state_dim = node_state_dim
        self.node_embedding_dim = node_embedding_dim
        self.norm = norm
        self.hidden_layers_size = hidden_layers_size
        self.num_propagations = num_propagations 
        self.layers = nn.ModuleDict()
        self.num_rels = num_rels
        self.rel_num_bases = rel_num_bases
        self.is_bias = is_bias
        self.activation = activation
        
        data = []
        
        if is_first_layer:
            data.append('state')
            in_feat = self.node_state_dim
            if n_convolutional_layers == 1 or not state_first_dim_only:
                in_feat += self.node_embedding_dim
                data.append('hid')
        else:
            if n_convolutional_layers == 1 or not state_first_dim_only:
                in_feat += self.node_state_dim
                data.append('state')
            in_feat = self.node_embedding_dim
            data.append('hid')

            
        
        self.layers['message_module'] = Message_Module(num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, multidimensional_attention, data, use_attention, std_attention, resnet, use_message_module, self.state_vars, self.num_rels, self.rel_num_bases, n_hidden_message, self.hidden_layers_size, in_feat = in_feat , out_feat = self.node_embedding_dim, is_bias = is_bias, activation = F.relu, norm = norm, dropout = dropout)

        self.layers['aggregation_module'] = Aggregation_Module(num_attention_heads, multidimensional_attention, resnet, use_aggregation_module, self.num_nodes_types, self.nodes_types_num_bases, n_hidden_aggregation, self.hidden_layers_size, in_feat = self.node_embedding_dim, out_feat = self.node_embedding_dim, is_bias = is_bias, activation = F.relu, norm = False, dropout = dropout)
        
        if self.is_final_convolutional_layer:
            self.layers['prediction_module'] = Prediction_Module(share_initial_params_between_actions, gaussian_mixture, n_gaussians, value_model_based, n_actions, rl_learner_type, use_message_module, self.num_nodes_types, self.nodes_types_num_bases, n_hidden_prediction, self.hidden_layers_size, in_feat = self.node_embedding_dim, out_feat = self.prediction_size, is_bias = is_bias, activation = F.relu, norm = False, dropout = dropout)


        
    def forward(self, graph, device, num_propagations = None):
        if not num_propagations :
            num_propagations = self.num_propagations

        for _ in range(num_propagations):
            self.layers['message_module'].propagate(graph, device)
            self.layers['aggregation_module'].aggregate(graph, device)
                
                
                
        # AUTRE FONCTION POUR CLASSIFY ? 
        if self.is_final_convolutional_layer:
            new_filt = partial(filt, identifier = 1)
            subgraph = graph.subgraph(list(graph.filter_nodes(new_filt)))
            #subgraph = graph.subgraph(list(graph.filter_nodes(is_tl)))
            subgraph.copy_from_parent()
            
            #print("subgraph nodes :", subgraph.nodes())
            #print("subgraph ndata :", subgraph.ndata)

            return self.layers['prediction_module'].predict(subgraph, device)
    
    
    
    
    

class Message_Module(nn.Module):
    def __init__(self, num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, multidimensional_attention, data, use_attention, std_attention, resnet, use_message_module, state_vars, num_rels, num_bases, n_hidden, hidden_layers_size, in_feat , out_feat , is_bias = True, activation = F.relu, norm = True, dropout = False):
        super(Message_Module, self).__init__()    
        
        #print("in_feat", in_feat)
        self.num_attention_heads = num_attention_heads
        self.bias_before_aggregations = bias_before_aggregation
        self.nonlinearity_before_aggregation = nonlinearity_before_aggregation
        self.multidimensional_attention = multidimensional_attention
        self.data = data
        self.use_attention = use_attention
        self.std_attention = std_attention
        self.resnet = resnet
        self.use_message_module = use_message_module
        self.dropout = dropout
        self.state_vars = state_vars
        self.norm = norm
        self.hidden_layers_size = hidden_layers_size
        self.n_hidden = n_hidden
        self.message_weights = nn.ParameterList()  
        self.message_w_comps = nn.ParameterList()   
        self.message_biases = nn.ParameterList()    
        self.request_weights = nn.ParameterList()  
        self.request_w_comps = nn.ParameterList()   
        self.request_biases = nn.ParameterList()   
        self.attention_weights = nn.ParameterList()  
        self.attention_w_comps = nn.ParameterList()   
        self.attention_biases = nn.ParameterList()   
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_bias = is_bias
        self.activation = activation

        
        if self.use_message_module:
            
                
            # sanity check
            if self.num_bases <= 0 or self.num_bases > self.num_rels:
                self.num_bases = self.num_rels        
 
                    
            
            # CREATE INFORMATION - MODULE
            
            
            
            if self.n_hidden == 0 :
                #UNIQUE LAYER 

                # weight bases in equation (3)

                self.weight_message_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                        self.out_feat))


                #print("weight size", self.weight_message_input.size())
                if self.num_bases < self.num_rels:
                    # linear combination coefficients in equation (3)
                    self.w_comp_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                #INITS
                nn.init.xavier_uniform_(self.weight_message_input,
                                        gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_rels:
                    nn.init.xavier_uniform_(self.w_comp_message_input,
                                            gain=nn.init.calculate_gain('relu'))            
                if self.is_bias:
                    
                    self.bias_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))                
                    nn.init.uniform_(self.bias_message_input)   


                
                
            else:

                #INPUT LAYER 

                # weight bases in equation (3)

                self.weight_message_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                        self.hidden_layers_size))


                #print("weight size", self.weight_message_input.size())
                if self.num_bases < self.num_rels:
                    # linear combination coefficients in equation (3)
                    self.w_comp_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                #INITS
                nn.init.xavier_uniform_(self.weight_message_input,
                                        gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_rels:
                    nn.init.xavier_uniform_(self.w_comp_message_input,
                                            gain=nn.init.calculate_gain('relu'))            
                if self.is_bias:
                    self.bias_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))                
                    nn.init.uniform_(self.bias_message_input)   




                #HIDDEN LAYERS

                for _ in range(self.n_hidden -1):
                    # weight bases in equation (3)
                    weight_message = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.hidden_layers_size))
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    if self.is_bias:
                        bias_message = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))          
                        nn.init.uniform_(bias_message)              
                    # init trainable parameters
                    nn.init.xavier_uniform_(weight_message,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(w_comp,
                                                gain=nn.init.calculate_gain('relu'))            

                    self.message_weights.append(weight_message)
                    if self.num_bases < self.num_rels:            
                        self.message_w_comps.append(w_comp_message)
                    if self.is_bias:
                        self.message_biases.append(bias_message)



                #OUTPUT LAYER 
                # weight bases in equation (3)

                self.weight_message_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                        self.out_feat))
                if self.num_bases < self.num_rels:
                    # linear combination coefficients in equation (3)
                    self.w_comp_message_output = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                #INITS
                nn.init.xavier_uniform_(self.weight_message_output,
                                        gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_rels:
                    nn.init.xavier_uniform_(self.w_comp_message_output,
                                            gain=nn.init.calculate_gain('relu'))            
                if self.is_bias:
                    self.bias_message_output = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))               
                    nn.init.uniform_(self.bias_message_output)     



            if self.use_attention : 
                # CREATE REQUEST - MODULE


                if self.n_hidden == 0 :


                    #UNIQUE LAYER 

                    # weight bases in equation (3)

                    self.weight_request_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                            self.out_feat))


                    #print("weight size", self.weight_request_input.size())
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_request_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_request_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.bias_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))                
                        nn.init.uniform_(self.bias_request_input)   




                else:


                    #INPUT LAYER 

                    # weight bases in equation (3)

                    self.weight_request_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                            self.hidden_layers_size))


                    #print("weight size", self.weight_request_input.size())
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_request_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_request_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.bias_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))                
                        nn.init.uniform_(self.bias_request_input)   




                    #HIDDEN LAYERS

                    for _ in range(self.n_hidden -1):
                        # weight bases in equation (3)
                        weight_request = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                                self.hidden_layers_size))
                        if self.num_bases < self.num_rels:
                            # linear combination coefficients in equation (3)
                            w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                        if self.is_bias:
                            bias_request = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))          
                            nn.init.uniform_(bias_request)              
                        # init trainable parameters
                        nn.init.xavier_uniform_(weight_request,
                                                gain=nn.init.calculate_gain('relu'))
                        if self.num_bases < self.num_rels:
                            nn.init.xavier_uniform_(w_comp,
                                                    gain=nn.init.calculate_gain('relu'))            

                        self.request_weights.append(weight_request)
                        if self.num_bases < self.num_rels:            
                            self.request_w_comps.append(w_comp_request)
                        if self.is_bias:
                            self.request_biases.append(bias_request)



                    #OUTPUT LAYER 
                    # weight bases in equation (3)

                    self.weight_request_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.out_feat))
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_request_output = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_request_output,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_request_output,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.bias_request_output = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))               
                        nn.init.uniform_(self.bias_request_output)     





                # CREATE ATTENTION - MODULE




                if self.n_hidden == 0 :


                    #UNIQUE LAYER 

                    # weight bases in equation (3)
                    self.weight_attention_input = nn.Parameter(torch.Tensor(self.num_bases, 2*self.hidden_layers_size,
                                                            self.out_feat if self.multidimensional_attention else self.num_attention_heads))
           


                    #print("weight size", self.weight_request_input.size())
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_attention_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_attention_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_attention_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.bias_attention_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat if self.multidimensional_attention else 1))  
                        nn.init.uniform_(self.bias_attention_input)   




                else:

                    #INPUT LAYER 

                    # weight bases in equation (3)

                    self.weight_attention_input = nn.Parameter(torch.Tensor(self.num_bases, 2*self.hidden_layers_size,
                                                            self.hidden_layers_size))


                    #print("weight size", self.weight_attention_input.size())
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_attention_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_attention_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_attention_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.bias_attention_input = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size))                
                        nn.init.uniform_(self.bias_attention_input)   




                    #HIDDEN LAYERS

                    for _ in range(self.n_hidden  -1):
                        # weight bases in equation (3)
                        weight_attention = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                                self.hidden_layers_size))
                        if self.num_bases < self.num_rels:
                            # linear combination coefficients in equation (3)
                            w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                        if self.is_bias:
                            bias_attention = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))          
                            nn.init.uniform_(bias_attention)              
                        # init trainable parameters
                        nn.init.xavier_uniform_(weight_attention,
                                                gain=nn.init.calculate_gain('relu'))
                        if self.num_bases < self.num_rels:
                            nn.init.xavier_uniform_(w_comp,
                                                    gain=nn.init.calculate_gain('relu'))            

                        self.attention_weights.append(weight_attention)
                        if self.num_bases < self.num_rels:            
                            self.attention_w_comps.append(w_comp_attention)
                        if self.is_bias:
                            self.attention_biases.append(bias_attention)



                    #OUTPUT LAYER 
                    # weight bases in equation (3)
                    self.weight_attention_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.out_feat if self.multidimensional_attention else self.num_attention_heads))
         

                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_attention_output = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_attention_output,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_attention_output,
                                                gain=nn.init.calculate_gain('relu'))   
                        
                        
                    if self.is_bias:
                        self.bias_attention_output = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat if self.multidimensional_attention else 1))                       
                        
                        nn.init.uniform_(self.bias_attention_output)   
                        





            
            
            
        
    def propagate(self, graph, device):
           
        def message_func(edges):
            #INPUT LAYER 

            
            if self.use_message_module:
                

                # FORWARD MESSAGE 
                

            

                if self.n_hidden == 0 :
                

                    if self.num_bases < self.num_rels:
                        # generate all weights from bases (equation (3))
                        weight_message_input = self.weight_message_input.view(self.in_feat, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_message_input = torch.matmul(self.w_comp_message_input, weight_message_input).view(self.num_rels,
                                                                    self.in_feat, self.out_feat)#.to("cuda")     
                    else:
                        weight_message_input = self.weight_message_input#.to("cuda")    
                        #print("weight size", weight_message_input.size())

                    w_message_input = weight_message_input[edges.data['rel_type']]#.to("cuda")   
                    bias_message_input = self.bias_message_input[edges.data['rel_type']]


                    #msg = torch.bmm(torch.cat((edges.src['state'].to(device), edges.src['hid'].to(device)),1).unsqueeze(1), w_message_input).squeeze()
                    msg = torch.bmm(torch.cat((tuple(edges.src[var].to(device) for var in self.data)),1).unsqueeze(1), w_message_input).squeeze()
                    

                        #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_message_input).squeeze()

                    if self.is_bias:
                        msg = msg + bias_message_input

                    if self.nonlinearity_before_aggregation:
                        msg = self.activation(msg) 
                        
                    if self.dropout :
                        msg = torch.nn.functional.dropout(msg, p=0.5, training=True, inplace=False) 
                    
                        
                    
                    
                    
                    
                else:

                    if self.num_bases < self.num_rels:
                        # generate all weights from bases (equation (3))
                        weight_message_input = self.weight_message_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                        weight_message_input = torch.matmul(self.w_comp_message_input, weight_message_input).view(self.num_rels,
                                                                    self.in_feat, self.hidden_layers_size)#.to("cuda")     
                    else:
                        weight_message_input = self.weight_message_input#.to("cuda")    
                        #print("weight size", weight_message_input.size())

                    w_message_input = weight_message_input[edges.data['rel_type']]#.to("cuda")   
                    bias_message_input = self.bias_message_input[edges.data['rel_type']]

                    #print("edges.src['h'].size()", edges.src['h'].size())
                    #msg = torch.bmm(torch.cat((edges.src['state'].to(device), edges.src['hid'].to(device)),1).unsqueeze(1), w_message_input).squeeze()
                    msg = torch.bmm(torch.cat((tuple(edges.src[var].to(device) for var in self.data)),1).unsqueeze(1), w_message_input).squeeze()

                    #print("msg", msg.size())
                        #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_message_input).squeeze()

                    if self.is_bias:
                        msg = msg + bias_message_input

                    msg = self.activation(msg)            
                    if self.dropout :
                        msg = torch.nn.functional.dropout(msg, p=0.5, training=True, inplace=False) 

                    #HIDDEN LAYERS            

                    for idx in range(self.n_hidden  -1):

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_message_hid = self.message_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_message_hid = torch.matmul(self.message_w_comps[idx], weight_message_hid).view(self.num_rels,
                                                                        self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                        else:
                            weight_message_hid = self.message_weights[idx]#.to("cuda")   

                        bias_message = self.message_biases[idx][edges.data['rel_type']]


                        w_message_hid = weight_message_hid[edges.data['rel_type']]#.to("cuda")   
                        msg = torch.bmm(msg.unsqueeze(1), w_message_hid).squeeze()

                        if self.is_bias:
                            msg = msg + bias_message

                        msg = self.activation(msg)      
                        if self.dropout :
                            msg = torch.nn.functional.dropout(msg, p=0.5, training=True, inplace=False)             
                    #OUTPUT LAYER 

                    if self.num_bases < self.num_rels:
                        # generate all weights from bases (equation (3))
                        weight_message_output = self.weight_message_output.view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                        weight_message_output = torch.matmul(self.w_comp_message_output, weight_message_output).view(self.num_rels,
                                                                    self.hidden_layers_size, self.out_feat)#.to("cuda")     

                    else:
                        weight_message_output = self.weight_message_output#.to("cuda")    



                    w_message_output = weight_message_output[edges.data['rel_type']]#.to("cuda")   
                    bias_message_output = self.bias_message_output[edges.data['rel_type']]
                    msg = torch.bmm(msg.unsqueeze(1), w_message_output).squeeze()



                    if self.is_bias:
                        msg = msg + bias_message_output            

                    if self.norm:
                        msg = msg * edges.data['norm']

                    if self.nonlinearity_before_aggregation:
                        msg = self.activation(msg)          
                        
                        
                    if self.dropout :
                        msg = torch.nn.functional.dropout(msg, p=0.5, training=True, inplace=False)  


                
            
                if self.use_attention : 
                    #TESTING START HERE

                    # FORWARD REQUEST


                    if self.n_hidden == 0 :

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_request_input = self.weight_request_input.view(self.in_feat, self.num_bases, self.out_feat)#.to("cuda")     
                            weight_request_input = torch.matmul(self.w_comp_request_input, weight_request_input).view(self.num_rels,
                                                                        self.in_feat, self.self.out_feat)#.to("cuda")     
                        else:
                            weight_request_input = self.weight_request_input#.to("cuda")    
                            #print("weight size", weight_request_input.size())


                        w_request_input = weight_request_input[edges.data['rel_type']]#.to("cuda")   
                        bias_request_input = self.bias_request_input[edges.data['rel_type']]


                        #rqst = torch.bmm(torch.cat((edges.dst['state'].to(device), edges.dst['hid'].to(device)),1).unsqueeze(1), w_request_input).squeeze()
                        rqst = torch.bmm(torch.cat((tuple(edges.src[var].to(device) for var in self.data)),1).unsqueeze(1), w_request_input).squeeze()

                            #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_request_input).squeeze()

                        if self.is_bias:
                            rqst = rqst + bias_request_input

                        rqst = self.activation(rqst)            
                        if self.dropout :
                            rqst = torch.nn.functional.dropout(rqst, p=0.5, training=True, inplace=False) 



                    else:

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_request_input = self.weight_request_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_request_input = torch.matmul(self.w_comp_request_input, weight_request_input).view(self.num_rels,
                                                                        self.in_feat, self.hidden_layers_size)#.to("cuda")     
                        else:
                            weight_request_input = self.weight_request_input#.to("cuda")    
                            #print("weight size", weight_request_input.size())

                        w_request_input = weight_request_input[edges.data['rel_type']]#.to("cuda")   
                        bias_request_input = self.bias_request_input[edges.data['rel_type']]

                        #rqst = torch.bmm(torch.cat((edges.dst['state'].to(device), edges.dst['hid'].to(device)),1).unsqueeze(1), w_request_input).squeeze()
                        rqst = torch.bmm(torch.cat((tuple(edges.src[var].to(device) for var in self.data)),1).unsqueeze(1), w_request_input).squeeze()

                            #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_request_input).squeeze()

                        if self.is_bias:
                            rqst = rqst + bias_request_input

                        rqst = self.activation(rqst)            
                        if self.dropout :
                            rqst = torch.nn.functional.dropout(rqst, p=0.5, training=True, inplace=False) 

                        #HIDDEN LAYERS            

                        for idx in range(self.n_hidden -1):

                            if self.num_bases < self.num_rels:
                                # generate all weights from bases (equation (3))
                                weight_request_hid = self.request_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                                weight_request_hid = torch.matmul(self.request_w_comps[idx], weight_request_hid).view(self.num_rels,
                                                                            self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                            else:
                                weight_request_hid = self.request_weights[idx]#.to("cuda")   

                            bias_request = self.request_biases[idx][edges.data['rel_type']]


                            w_request_hid = weight_request_hid[edges.data['rel_type']]#.to("cuda")   
                            rqst = torch.bmm(rqst.unsqueeze(1), w_request_hid).squeeze()

                            if self.is_bias:
                                rqst = rqst + bias_request

                            rqst = self.activation(rqst)      
                            if self.dropout :
                                rqst = torch.nn.functional.dropout(rqst, p=0.5, training=True, inplace=False)             
                        #OUTPUT LAYER 

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_request_output = self.weight_request_output.view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_request_output = torch.matmul(self.w_comp_request_output, weight_request_output).view(self.num_rels,
                                                                        self.hidden_layers_size, self.out_feat)#.to("cuda")     

                        else:
                            weight_request_output = self.weight_request_output#.to("cuda")    



                        w_request_output = weight_request_output[edges.data['rel_type']]#.to("cuda")   
                        bias_request_output = self.bias_request_output[edges.data['rel_type']]
                        rqst = torch.bmm(rqst.unsqueeze(1), w_request_output).squeeze()



                        if self.is_bias:
                            rqst = rqst + bias_request_output            

                        if self.norm:
                            rqst = rqst * edges.data['norm']

                        rqst = self.activation(rqst)            
                        if self.dropout :
                            rqst = torch.nn.functional.dropout(rqst, p=0.5, training=True, inplace=False)  




                    # FORWARD ATTENTION
                    if self.n_hidden == 0 :

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_attention_input = self.weight_attention_input.view(2*self.hidden_layers_size, self.num_bases, self.out_feat if self.multidimensional_attention else self.num_attention_heads)#.to("cuda")     
                            weight_attention_input = torch.matmul(self.w_comp_attention_input, weight_attention_input).view(self.num_rels, 2*self.hidden_layers_size, self.out_feat if self.multidimensional_attention else self.num_attention_heads)#.to("cuda")     
                        else:
                            weight_attention_input = self.weight_attention_input#.to("cuda")    
                            #print("weight size", weight_attention_input.size())

                        w_attention_input = weight_attention_input[edges.data['rel_type']]#.to("cuda")   
                        bias_attention_input = self.bias_attention_input[edges.data['rel_type']]
                        #print("msg size", msg.size())
                        #print("rqst size", rqst.size())
                        #print("cat size", torch.cat((msg, rqst),1).size())
                        #print("self w_attention_input size", self.weight_attention_input.size())
                        #print("w_attention_input size", w_attention_input.size())
                        att = torch.bmm(torch.cat((msg, rqst),1).unsqueeze(1), w_attention_input).squeeze()
                        #print("att", att.size())
                            #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_attention_input).squeeze()

                        if self.is_bias:
                            att = att + bias_attention_input

                        #print("att2", att.size())
                        att = self.activation(att)            
                        #if self.dropout :
                            #att = torch.nn.functional.dropout(att, p=0.5, training=True, inplace=False)   






                    else:
                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_attention_input = self.weight_attention_input.view(2*self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_attention_input = torch.matmul(self.w_comp_attention_input, weight_attention_input).view(self.num_rels,
                                                                        2*self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                        else:
                            weight_attention_input = self.weight_attention_input#.to("cuda")    
                            #print("weight size", weight_attention_input.size())

                        w_attention_input = weight_attention_input[edges.data['rel_type']]#.to("cuda")   
                        bias_attention_input = self.bias_attention_input[edges.data['rel_type']]
                        #print("msg size", msg.size())
                        #print("rqst size", rqst.size())
                        #print("cat size", torch.cat((msg, rqst),1).size())
                        #print("w_attention_input size", w_attention_input.size())
                        att = torch.bmm(torch.cat((msg, rqst),1).unsqueeze(1), w_attention_input).squeeze()

                            #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_attention_input).squeeze()

                        if self.is_bias:
                            att = att + bias_attention_input

                        att = self.activation(att)            
                        if self.dropout :
                            att = torch.nn.functional.dropout(att, p=0.5, training=True, inplace=False) 

                        #HIDDEN LAYERS            

                        for idx in range(self.n_hidden -1):

                            if self.num_bases < self.num_rels:
                                # generate all weights from bases (equation (3))
                                weight_attention_hid = self.attention_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                                weight_attention_hid = torch.matmul(self.attention_w_comps[idx], weight_attention_hid).view(self.num_rels,
                                                                            self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                            else:
                                weight_attention_hid = self.attention_weights[idx]#.to("cuda")   

                            bias_attention = self.attention_biases[idx][edges.data['rel_type']]


                            w_attention_hid = weight_attention_hid[edges.data['rel_type']]#.to("cuda")   
                            att = torch.bmm(att.unsqueeze(1), w_attention_hid).squeeze()

                            if self.is_bias:
                                att = att + bias_attention

                            att = self.activation(att)      
                            if self.dropout :
                                att = torch.nn.functional.dropout(att, p=0.5, training=True, inplace=False)             
                        #OUTPUT LAYER 

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_attention_output = self.weight_attention_output.view(self.hidden_layers_size, self.num_bases, self.out_feat if self.multidimensional_attention else 1)#.to("cuda")     
                            weight_attention_output = torch.matmul(self.w_comp_attention_output, weight_attention_output).view(self.num_rels, self.hidden_layers_size, self.out_feat if self.multidimensional_attention else 1)#.to("cuda")     

                        else:
                            weight_attention_output = self.weight_attention_output#.to("cuda")    



                        w_attention_output = weight_attention_output[edges.data['rel_type']]#.to("cuda")   
                        bias_attention_output = self.bias_attention_output[edges.data['rel_type']]
                        att = torch.bmm(att.unsqueeze(1), w_attention_output).squeeze()



                        if self.is_bias:
                            att = att + bias_attention_output            

                        if self.norm:
                            att = att * edges.data['norm']


                        att = self.activation(att)            
                        #if self.dropout :
                        #    att = torch.nn.functional.dropout(att, p=0.5, training=True, inplace=False)  



                    return {'msg': msg, 'att' : att}            


                else: 
                    return {'msg' : msg}
                #TESTING START HERE
                
    
        
        
            else:
                msg = edges.src['short_cycle_durations']
                return {'msg': -msg}

        """
        if not self.use_message_module:
            def reduce_func(nodes):
                hid , _ = nodes.mailbox['msg'].min(0)
                print("hid size", hid.size())
                return {'hid', hid}
        """
        def reduce_func(nodes):
            # AGGREGATION OF ATTENTION 
            # REDUCE FUNCTION BATCHES NODES OF SAME IN-DEGREES TOGETHER 
            
            
            if self.use_attention : 
                att_w = nodes.mailbox['att']
                #print("att_w before", att_w.size(), att_w)
                if self.std_attention:
                    att_w = F.softmax(att_w,dim = 1)
                mailbox = nodes.mailbox['msg']
                    
                #print("att_w", att_w.size(), att_w, "nodes.mailbox[msg]",mailbox.size(), mailbox)
                if not self.multidimensional_attention:
                    mailbox = mailbox.unsqueeze(2).repeat(1,1,self.num_attention_heads,1)
                    #print('new message size', mailbox.size(), mailbox)
                    att_w = att_w.unsqueeze(3).expand_as(mailbox)
                    #print("new att size", att_w.size(), att_w)
                agg_msg = torch.sum( att_w * mailbox, dim = 1)
                #print("agg_msg", agg_msg.size(), agg_msg)
                
            
            else:
                agg_msg = torch.sum(mailbox, dim = 1)

            
            if not self.nonlinearity_before_aggregation:
                agg_msg = self.activation(agg_msg)
                
            return {'agg_msg' : agg_msg}
            #return{'agg_msg' : torch.mean(nodes.mailbox['msg'], dim = 1)}

        def apply_func(nodes):
            #print("1st agg msg size :", nodes.data["agg_msg"].size())
            pass
        
        
        if self.use_message_module: 
            graph.update_all(message_func = message_func, reduce_func = reduce_func, apply_node_func = apply_func)     

            
        else:
            graph.update_all(message_func = message_func, reduce_func = fn.max(msg='msg',out='hid'), apply_node_func = apply_func)  
        #print("went through message layer")
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class Aggregation_Module(nn.Module):

    def __init__(self,num_attention_heads, multidimensional_attention, resnet, use_aggregation_module, num_nodes_types, num_bases, n_hidden, hidden_layers_size, in_feat , out_feat , is_bias = True, activation = F.relu, norm = False, dropout = False):

        super(Aggregation_Module, self).__init__()   
        self.num_attention_heads = num_attention_heads
        self.multidimensional_attention = multidimensional_attention
        self.use_aggregation_module = use_aggregation_module
        self.resnet = resnet
        self.dropout = dropout
        self.norm = norm
        self.hidden_layers_size = hidden_layers_size
        self.n_hidden = n_hidden
        
        # HIDDEN
        self.aggregation_weights = nn.ParameterList()  
        self.aggregation_w_comps = nn.ParameterList()   
        self.aggregation_biases = nn.ParameterList()   
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_nodes_types = num_nodes_types
        self.num_bases = num_bases
        self.is_bias = is_bias
        self.activation = activation
        
        # RNN _ HIDDEN
        self.weights_input_aggregation = nn.ParameterList()  
        self.weights_hidden_aggregation = nn.ParameterList()  
        if self.num_bases < self.num_nodes_types:       
            self.w_comps_input_aggregation = nn.ParameterList()  
            self.w_comps_hidden_aggregation = nn.ParameterList()  
        if self.is_bias:
            self.biases_input_aggregation = nn.ParameterList()  
            self.biases_hidden_aggregation = nn.ParameterList()  

        
        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_nodes_types:
            self.num_bases = self.num_nodes_types    

        # RNN - GRU 
        
        """
        r=σ(Wir​x+bir+Whr​h+bhr)
        z=σ(Wiz​x+biz+Whz​h+bhz)
        n=tanh(Win​x+bin+r∗(Whn​h+bhn))
        h′=(1−z)∗n+z∗h
        """
        
        
        # MAPPING FROM CONCATENATED MULTI-HEAD ATTENTION MECHANISMS RESULTS TO THE ORIGINAL EMBEDDING SIZE 
        if not self.multidimensional_attention:
                self.weight_att_head_aggregation = nn.Parameter(torch.Tensor(self.num_bases, self.num_attention_heads*self.in_feat, self.in_feat))  
                nn.init.xavier_uniform_(self.weight_att_head_aggregation, gain=nn.init.calculate_gain('relu')) 
                if self.num_bases < self.num_nodes_types:
                    self.w_comp_att_head_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))   
                    nn.init.xavier_uniform_(self.w_comp_att_head_aggregation, gain=nn.init.calculate_gain('relu'))       
                if self.is_bias:
                    self.bias_att_head_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, self.out_feat))
                    nn.init.uniform_(self.bias_att_head_aggregation)   
            # CREATE AND INIT WEIGHTS
            
        if self.use_aggregation_module : 
            if self.n_hidden == 0 : 


                # UNIQUE LAYER 


                # BETTER PARRAL
                self.weight_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, 3*self.out_feat))
                self.weight_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_bases, self.out_feat, 3*self.out_feat))            
                nn.init.xavier_uniform_(self.weight_input_aggregation_input, gain=nn.init.calculate_gain('relu'))            
                nn.init.xavier_uniform_(self.weight_hidden_aggregation_input, gain=nn.init.calculate_gain('relu'))            


                if self.num_bases < self.num_nodes_types:            
                    self.w_comp_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    self.w_comp_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.self.w_comp_input_aggregation_input, gain=nn.init.calculate_gain('relu'))            
                    nn.init.xavier_uniform_(self.self.w_comp_hidden_aggregation_input, gain=nn.init.calculate_gain('relu'))                

                if self.is_bias:
                    self.bias_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.out_feat))
                    self.bias_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.out_feat))
                    nn.init.uniform_(self.bias_input_aggregation_input)   
                    nn.init.uniform_(self.bias_hidden_aggregation_input)   



            else:


                # INPUT LAYER 


                # BETTER PARRAL
                self.weight_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, 3*self.hidden_layers_size))
                self.weight_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size, 3*self.hidden_layers_size))            
                nn.init.xavier_uniform_(self.weight_input_aggregation_input, gain=nn.init.calculate_gain('relu'))            
                nn.init.xavier_uniform_(self.weight_hidden_aggregation_input, gain=nn.init.calculate_gain('relu'))            


                if self.num_bases < self.num_nodes_types:            
                    self.w_comp_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    self.w_comp_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.self.w_comp_input_aggregation_input, gain=nn.init.calculate_gain('relu'))            
                    nn.init.xavier_uniform_(self.self.w_comp_hidden_aggregation_input, gain=nn.init.calculate_gain('relu'))                

                if self.is_bias:
                    self.bias_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.hidden_layers_size))
                    self.bias_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.hidden_layers_size))
                    nn.init.uniform_(self.bias_input_aggregation_input)   
                    nn.init.uniform_(self.bias_hidden_aggregation_input)   


                #HIDDEN LAYERS


                for _ in range(self.n_hidden -1):

                    # BETTER PARRAL
                    weight_input_aggregation = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size, 3*self.hidden_layers_size))
                    weight_hidden_aggregation = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size, 3*self.hidden_layers_size))            
                    nn.init.xavier_uniform_(weight_input_aggregation, gain=nn.init.calculate_gain('relu'))            
                    nn.init.xavier_uniform_(weight_hidden_aggregation, gain=nn.init.calculate_gain('relu'))    
                    self.weights_input_aggregation.append(weight_input_aggregation)
                    self.weights_hidden_aggregation.append(weight_hidden_aggregation)                


                    if self.num_bases < self.num_nodes_types:            
                        w_comp_input_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                        w_comp_hidden_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                        nn.init.xavier_uniform_(w_comp_input_aggregation, gain=nn.init.calculate_gain('relu'))            
                        nn.init.xavier_uniform_(w_comp_hidden_aggregation, gain=nn.init.calculate_gain('relu')) 
                        self.w_comps_input_aggregation.append(w_comp_input_aggregation)
                        self.w_comps_hidden_aggregation.append(w_com_hidden_aggregation)

                    if self.is_bias:
                        bias_input_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.hidden_layers_size))
                        bias_hidden_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.hidden_layers_size))
                        nn.init.uniform_(bias_input_aggregation)   
                        nn.init.uniform_(bias_hidden_aggregation)   
                        self.biases_input_aggregation.append(bias_input_aggregation)
                        self.biases_hidden_aggregation.append(bias_hidden_aggregation)


                # OUTPUT LAYER 


                # BETTER PARRAL
                self.weight_input_aggregation_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size, 3*self.out_feat))
                self.weight_hidden_aggregation_output = nn.Parameter(torch.Tensor(self.num_bases, self.out_feat, 3*self.out_feat))            
                nn.init.xavier_uniform_(self.weight_input_aggregation_output, gain=nn.init.calculate_gain('relu'))            
                nn.init.xavier_uniform_(self.weight_hidden_aggregation_output, gain=nn.init.calculate_gain('relu'))            


                if self.num_bases < self.num_nodes_types:            
                    self.w_comp_input_aggregation_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    self.w_comp_hidden_aggregation_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.self.w_comp_input_aggregation_output, gain=nn.init.calculate_gain('relu'))            
                    nn.init.xavier_uniform_(self.self.w_comp_hidden_aggregation_output, gain=nn.init.calculate_gain('relu'))                

                if self.is_bias:
                    self.bias_input_aggregation_output = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.out_feat))
                    self.bias_hidden_aggregation_output = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.out_feat))
                    nn.init.uniform_(self.bias_input_aggregation_output)   
                    nn.init.uniform_(self.bias_hidden_aggregation_output)   


                
                
                
                
                
                
                
                
                

        
    def aggregate(self, graph, device):
        #print("yololo2")
        def message_func(edges):
            pass
        def reduce_func(nodes):
            pass
            
        def apply_func(nodes):
            
            
            agg_msg = nodes.data['agg_msg']
            
            #print("agg_msg size :", agg_msg.size())

            # MAPPING FROM CONCATENATED RESULTS OF MULTIPLE ATTENTION HEAD MECHANISMS TO ORIGINAL EMBEDDING DIM
            if not self.multidimensional_attention:
                if self.num_bases < self.num_nodes_types:
                    weight_att_head_aggregation = self.weight_att_head_aggregation.view(self.num_attention_heads * self.in_feat, self.num_bases, self.in_feat) 
                    weight_att_head_aggregation = torch.matmul(self.w_comp_att_head_aggregation, weight_att_head_aggregation).view(self.num_nodes_types, self.num_attention_heads * self.in_feat, self.in_feat)  
                else:
                    weight_att_head_aggregation = self.weight_att_head_aggregation 

                weight_att_head_aggregation = weight_att_head_aggregation[nodes.data['node_type']-1]
                bias_att_head_aggregation = self.bias_att_head_aggregation[nodes.data['node_type']-1]
                agg_msg = agg_msg.view(-1,self.num_attention_heads*self.in_feat)
                agg_msg = torch.bmm(agg_msg.unsqueeze(1), weight_att_head_aggregation).squeeze()
                if self.is_bias:
                    agg_msg = agg_msg + bias_att_head_aggregation
                agg_msg = self.activation(agg_msg)            
                if self.dropout :
                    agg_msg = torch.nn.functional.dropout(agg_msg, p=0.5, training=True, inplace=False) 



            
            
            
            
            # OPTIONNAL GRU LAYER (OPTION TO MAKE RESIDUAL CONNECTIONS <- self.resnet)

            if self.use_aggregation_module: 
                if self.n_hidden == 0 : 


                    
                    
                    # FORWARD UNIQUE
                    if self.num_bases < self.num_nodes_types:

                        weight_input_aggregation_input = self.weight_input_aggregation_input.view(self.in_feat, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_input_aggregation_input = torch.matmul(self.w_comp_input_aggregation_input, weight_input_aggregation_input).view(self.num_nodes_types,
                                                                    self.in_feat, 3*self.out_feat)#.to("cuda")     

                        weight_hidden_aggregation_input = self.weight_hidden_aggregation_input.view(self.out_feat, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_hidden_aggregation_input = torch.matmul(self.w_comp_hidden_aggregation_input, weight_hidden_aggregation_input).view(self.num_nodes_types,
                                                                    self.out_feat, 3*self.out_feat)#.to("cuda")

                    else:
                        weight_input_aggregation_input = self.weight_input_aggregation_input     
                        weight_hidden_aggregation_input = self.weight_hidden_aggregation_input



                    w_input_aggregation_input = weight_input_aggregation_input[nodes.data['node_type']-1]#.to("cuda")   
                    w_hidden_aggregation_input = weight_hidden_aggregation_input[nodes.data['node_type']-1]

                    if self.is_bias:
                        bias_input_aggregation_input = self.bias_input_aggregation_input[nodes.data['node_type']-1]
                        bias_hidden_aggregation_input = self.bias_hidden_aggregation_input[nodes.data['node_type']-1]

    
                    #print("agg msg size :", nodes.data["agg_msg"].size())
                    #print("w size :", weight_input_aggregation_input[nodes.data['node_type']-1].size())
                    
                    gate_x = torch.bmm(agg_msg.to(device).unsqueeze(1), w_input_aggregation_input).squeeze()
                    gate_h = torch.bmm(nodes.data['hid'].to(device).unsqueeze(1), w_hidden_aggregation_input).squeeze()

                    if self.is_bias:
                        #print("gate x size :", gate_x.size())
                        #print("bias size :", bias_input_aggregation_input.size())
                        gate_x += bias_input_aggregation_input
                        gate_h += bias_hidden_aggregation_input

                    #print("new gate x size :", gate_x.size())

                    i_r, i_i, i_n = gate_x.chunk(3, 1)
                    h_r, h_i, h_n = gate_h.chunk(3, 1)                

                    resetgate = torch.sigmoid(i_r + h_r)
                    inputgate = torch.sigmoid(i_i + h_i)
                    newgate = torch.tanh(i_n + (resetgate * h_n))



                    if self.resnet:
                        hy = nodes.data['hid'].to(device) + (newgate + inputgate * (nodes.data['hid'].to(device) - newgate))    
                    else:
                        hy = newgate + inputgate * (nodes.data['hid'].to(device) - newgate)

                else:


                    # FORWARD INPUT
                    if self.num_bases < self.num_nodes_types:

                        weight_input_aggregation_input = self.weight_input_aggregation_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                        weight_input_aggregation_input = torch.matmul(self.w_comp_input_aggregation_input, weight_input_aggregation_input).view(self.num_nodes_types,
                                                                    self.in_feat, 3*self.hidden_layers_size)#.to("cuda")     

                        weight_hidden_aggregation_input = self.weight_hidden_aggregation_input.view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                        weight_hidden_aggregation_input = torch.matmul(self.w_comp_hidden_aggregation_input, weight_hidden_aggregation_input).view(self.num_nodes_types,
                                                                    self.hidden_layers_size, 3*self.hidden_layers_size)#.to("cuda")

                    else:
                        weight_input_aggregation_input = self.weight_input_aggregation_input     
                        weight_hidden_aggregation_input = self.weight_hidden_aggregation_input



                    w_input_aggregation_input = weight_input_aggregation_input[nodes.data['node_type']-1]#.to("cuda")   
                    w_hidden_aggregation_input = weight_hidden_aggregation_input[nodes.data['node_type']-1]
                    if self.is_bias:
                        bias_input_aggregation_input = self.bias_input_aggregation_input[nodes.data['node_type']-1]
                        bias_hidden_aggregation_input = self.bias_hidden_aggregation_input[nodes.data['node_type']-1]


                    gate_x = torch.bmm(agg_msg.to(device).unsqueeze(1), w_input_aggregation_input).squeeze()
                    gate_h = torch.bmm(nodes.data['memory_input'].to(device).unsqueeze(1), w_hidden_aggregation_input).squeeze()
                    if self.is_bias:
                        gate_h += bias_hidden_aggregation_input
                        gate_x += bias_input_aggregation_input

                    i_r, i_i, i_n = gate_x.chunk(3, 1)
                    h_r, h_i, h_n = gate_h.chunk(3, 1)                

                    resetgate = torch.sigmoid(i_r + h_r)
                    inputgate = torch.sigmoid(i_i + h_i)
                    newgate = torch.tanh(i_n + (resetgate * h_n))

                    if self.resnet:
                        hy = nodes.data['memory_input'].to(device) + (newgate + inputgate * (nodes.data['memory_input'].to(device)- newgate))    
                    else:
                        hy = newgate + inputgate * (nodes.data['memory_input'].to(device) - newgate)


                    graph.ndata.update({'memory_input': hy})                



                    # FORWARD HIDDEN

                    for idx in range(self.n_hidden-1):
                        if self.num_bases < self.num_nodes_types:


                            weight_input_aggregation = self.weights_input_aggregation[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_input_aggregation = torch.matmul(self.w_comps_input_aggregation[idx], weight_input_aggregation).view(self.num_nodes_types,
                                                                        self.hidden_layers_size, 3*self.hidden_layers_size)#.to("cuda")     

                            weight_hidden_aggregation = self.weights_hidden_aggregation[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_hidden_aggregation = torch.matmul(self.w_comps_hidden_aggregation[idx], weight_hidden_aggregation).view(self.num_nodes_types,
                                                                        self.hidden_layers_size, 3*self.hidden_layers_size)#.to("cuda")

                        else:
                            weight_input_aggregation = self.weight_input_aggregation_input     
                            weight_hidden_aggregation = self.weight_hidden_aggregation_input



                        w_input_aggregation = weight_input_aggregation[nodes.data['node_type']-1]#.to("cuda")   
                        w_hidden_aggregation = weight_hidden_aggregation[nodes.data['node_type']-1]
                        if self.is_bias:
                            bias_input_aggregation = self.bias_input_aggregation[nodes.data['node_type']-1]
                            bias_hidden_aggregation = self.bias_hidden_aggregation[nodes.data['node_type']-1]


                        gate_x = torch.bmm(hy.unsqueeze(1), w_input_aggregation).squeeze()
                        gate_h = torch.bmm(nodes.data[str('memory_' + str(idx))].unsqueeze(1), w_hidden_aggregation).squeeze()
                        if self.is_bias:
                            gate_x += bias_input_aggregation
                            gate_h += bias_hidden_aggregation

                        i_r, i_i, i_n = gate_x.chunk(3, 1)
                        h_r, h_i, h_n = gate_h.chunk(3, 1)                

                        resetgate = torch.sigmoid(i_r + h_r)
                        inputgate = torch.sigmoid(i_i + h_i)
                        newgate = torch.tanh(i_n + (resetgate * h_n))


                        if self.resnet:
                            hy = nodes.data[str('memory_' + str(idx))] + (newgate + inputgate * (nodes.data[str('memory_' + str(idx))] - newgate))   
                        else:
                            hy = newgate + inputgate * (nodes.data[str('memory_' + str(idx))] - newgate)

                        graph.ndata.update({str('memory_' + str(idx)): hy})



                    # FORWARD OUTPUT
                    if self.num_bases < self.num_nodes_types:

                        weight_input_aggregation_output = self.weight_input_aggregation_output.view(self.hidden_layers_size, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_input_aggregation_output = torch.matmul(self.w_comp_input_aggregation_output, weight_input_aggregation_output).view(self.num_nodes_types,
                                                                    self.hidden_layers_size, 3*self.out_feat)#.to("cuda")     

                        weight_hidden_aggregation_output = self.weight_hidden_aggregation_output.view(self.out_feat, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_hidden_aggregation_output = torch.matmul(self.w_comp_hidden_aggregation_output, weight_hidden_aggregation_output).view(self.num_nodes_types,
                                                                    self.out_feat, 3*self.out_feat)#.to("cuda")

                    else:
                        weight_input_aggregation_output = self.weight_input_aggregation_output     
                        weight_hidden_aggregation_output = self.weight_hidden_aggregation_output



                    w_input_aggregation_output = weight_input_aggregation_output[nodes.data['node_type']-1]#.to("cuda")   
                    w_hidden_aggregation_output = weight_hidden_aggregation_output[nodes.data['node_type']-1]
                    if self.is_bias:
                        bias_input_aggregation_output = self.bias_input_aggregation_output[nodes.data['node_type']-1]
                        bias_hidden_aggregation_output = self.bias_hidden_aggregation_output[nodes.data['node_type']-1]


                    gate_x = torch.bmm(hy.unsqueeze(1), w_input_aggregation_output).squeeze()
                    gate_h = torch.bmm(nodes.data['hid'].unsqueeze(1), w_hidden_aggregation_output).squeeze()
                    if self.is_bias:
                        gate_x += bias_input_aggregation_output
                        gate_h += bias_hidden_aggregation_output

                    i_r, i_i, i_n = gate_x.chunk(3, 1)
                    h_r, h_i, h_n = gate_h.chunk(3, 1)                

                    resetgate = torch.sigmoid(i_r + h_r)
                    inputgate = torch.sigmoid(i_i + h_i)
                    newgate = torch.tanh(i_n + (resetgate * h_n))
             

                    if self.resnet:
                        hy = nodes.data[str('hid')] + (newgate + inputgate * (nodes.data[str('hid')] - newgate))   
                    else:
                        hy = newgate + inputgate * (nodes.data[str('hid')] - newgate)
              

            else:
                if self.resnet:
                    hy = nodes.data['hid'].to(device) + agg_msg.to(device)
                else:
                    hy = agg_msg.to(device)
            return {'hid' : hy}      


        
        graph.update_all(message_func = message_func, reduce_func = reduce_func, apply_node_func = apply_func)             



            
            
            
class Prediction_Module(nn.Module):

    def __init__(self, share_initial_params_between_actions, gaussian_mixture, n_gaussians, value_model_based, n_actions, rl_learner_type, use_message_module, num_nodes_types, num_bases, n_hidden, hidden_layers_size, in_feat , out_feat , is_bias = True, activation = F.relu, norm = True, dropout = False):
        super(Prediction_Module, self).__init__() 
        self.share_initial_params_between_actions = share_initial_params_between_actions
        self.gaussian_mixture = gaussian_mixture
        self.n_gaussians = n_gaussians
        self.value_model_based = value_model_based
        self.n_actions = n_actions
        self.rl_learner_type = rl_learner_type
        self.use_message_module = use_message_module
        self.dropout = dropout
        self.norm = norm
        self.hidden_layers_size = hidden_layers_size
        self.n_hidden = n_hidden
        self.prediction_weights = nn.ParameterList()  
        self.prediction_w_comps = nn.ParameterList()   
        self.prediction_biases = nn.ParameterList()     
        self.in_feat = in_feat
        self.out_feat = out_feat
        if self.rl_learner_type == "Q_Learning":
            self.out_feat = self.n_actions
        if 'critic' in self.rl_learner_type.lower():
            self.critic_out_feat = 1
        if 'actor' in self.rl_learner_type.lower():
            self.actor_out_feat = self.n_actions
        
        self.num_nodes_types = num_nodes_types
        self.num_bases = num_bases
        self.is_bias = is_bias
        self.activation = activation

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_nodes_types:
            self.num_bases = self.num_nodes_types
            
        
        if not self.gaussian_mixture:


            if self.n_hidden == 0:


                if self.rl_learner_type == "Q_Learning":                
                    #UNIQUE Q_LEARNER LAYER 

                    # weight bases in equation (3)

                    self.weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                            self.out_feat))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        self.w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        nn.init.xavier_uniform_(self.w_comp_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.out_feat))               
                        nn.init.uniform_(self.bias_prediction_input)   


                if 'critic' in self.rl_learner_type.lower():            
                    #UNIQUE CRITIC LAYER 

                    # weight bases in equation (3)

                    self.critic_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                            self.critic_out_feat))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        self.critic_w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.critic_weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        nn.init.xavier_uniform_(self.critic_w_comp_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.critic_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.critic_out_feat))               
                        nn.init.uniform_(self.critic_bias_prediction_input)   


                if 'actor' in self.rl_learner_type.lower(): 
                    #UNIQUE ACTOR LAYER 

                    # weight bases in equation (3)

                    self.actor_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                            self.actor_out_feat))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        self.actor_w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.actor_weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        nn.init.xavier_uniform_(self.actor_w_comp_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.actor_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.actor_out_feat))               
                        nn.init.uniform_(self.actor_bias_prediction_input)   





            else:
                if self.rl_learner_type == "Q_Learning":      
                    #INPUT Q LEARNER LAYER 

                    # weight bases in equation (3)

                    self.weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                            self.hidden_layers_size))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        self.w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        nn.init.xavier_uniform_(self.w_comp_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size))               
                        nn.init.uniform_(self.bias_prediction_input)   



                    #HIDDEN Q LEARNER LAYERS

                    for _ in range(self.n_hidden -1):
                        # weight bases in equation (3)
                        weight_prediction = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                                self.hidden_layers_size))
                        if self.num_bases < self.num_nodes_types:
                            # linear combination coefficients in equation (3)
                            w_comp = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                        if self.is_bias:
                            bias_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size))            
                            nn.init.uniform_(bias_prediction)              
                        # init trainable parameters
                        nn.init.xavier_uniform_(weight_prediction,
                                                gain=nn.init.calculate_gain('relu'))
                        if self.num_bases < self.num_nodes_types:
                            nn.init.xavier_uniform_(w_comp,
                                                    gain=nn.init.calculate_gain('relu'))            

                        self.prediction_weights.append(weight_prediction)
                        if self.num_bases < self.num_nodes_types:            
                            self.prediction_w_comps.append(w_comp_prediction)
                        if self.is_bias:
                            self.prediction_biases.append(bias_prediction)



                    #OUTPUT Q LEARNER LAYER 
                    # weight bases in equation (3)

                    self.weight_prediction_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.out_feat))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        self.w_comp_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_prediction_output,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        nn.init.xavier_uniform_(self.w_comp_output,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.bias_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.out_feat))               
                        nn.init.uniform_(self.bias_prediction_output)     

                if 'critic' in self.rl_learner_type.lower():     
                    #INPUT CRITIC LAYER 

                    # weight bases in equation (3)

                    self.critic_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                            self.hidden_layers_size))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        self.critic_w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.critic_weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        nn.init.xavier_uniform_(self.critic_w_comp_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.critic_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size))               
                        nn.init.uniform_(self.critic_bias_prediction_input)   



                    #HIDDEN CRITIC LAYERS

                    for _ in range(self.n_hidden -1):
                        # weight bases in equation (3)
                        critic_weight_prediction = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                                self.hidden_layers_size))
                        if self.num_bases < self.num_nodes_types:
                            # linear combination coefficients in equation (3)
                            critic_w_comp = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                        if self.is_bias:
                            critic_bias_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size))            
                            nn.init.uniform_(critic_bias_prediction)              
                        # init trainable parameters
                        nn.init.xavier_uniform_(critic_weight_prediction,
                                                gain=nn.init.calculate_gain('relu'))
                        if self.num_bases < self.num_nodes_types:
                            nn.init.xavier_uniform_(critic_w_comp,
                                                    gain=nn.init.calculate_gain('relu'))            

                        self.critic_prediction_weights.append(critic_weight_prediction)
                        if self.num_bases < self.num_nodes_types:            
                            self.critic_prediction_w_comps.append(critic_w_comp_prediction)
                        if self.is_bias:
                            self.critic_prediction_biases.append(critic_bias_prediction)



                    #OUTPUT CRITIC LAYER 
                    # weight bases in equation (3)

                    self.critic_weight_prediction_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.critic_out_feat))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        self.critic_w_comp_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.critic_weight_prediction_output,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        nn.init.xavier_uniform_(self.critic_w_comp_output,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.critic_bias_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.out_feat))               
                        nn.init.uniform_(self.critic_bias_prediction_output)     


                if 'actor' in self.rl_learner_type.lower():    

                    #INPUT ACTOR LAYER 

                    # weight bases in equation (3)

                    self.actor_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                            self.hidden_layers_size))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        self.actor_w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.actor_weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        nn.init.xavier_uniform_(self.actor_w_comp_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.actor_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size))               
                        nn.init.uniform_(self.actor_bias_prediction_input)   



                    #HIDDEN ACTOR LAYERS

                    for _ in range(self.n_hidden -1):
                        # weight bases in equation (3)
                        actor_weight_prediction = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                                self.hidden_layers_size))
                        if self.num_bases < self.num_nodes_types:
                            # linear combination coefficients in equation (3)
                            actor_w_comp = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                        if self.is_bias:
                            actor_bias_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size))            
                            nn.init.uniform_(actor_bias_prediction)              
                        # init trainable parameters
                        nn.init.xavier_uniform_(actor_weight_prediction,
                                                gain=nn.init.calculate_gain('relu'))
                        if self.num_bases < self.num_nodes_types:
                            nn.init.xavier_uniform_(actor_w_comp,
                                                    gain=nn.init.calculate_gain('relu'))            

                        self.actor_prediction_weights.append(actor_weight_prediction)
                        if self.num_bases < self.num_nodes_types:            
                            self.actor_prediction_w_comps.append(actor_w_comp_prediction)
                        if self.is_bias:
                            self.actor_prediction_biases.append(actor_bias_prediction)



                    #OUTPUT ACTOR LAYER 
                    # weight bases in equation (3)

                    self.actor_weight_prediction_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.actor_out_feat))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        self.actor_w_comp_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.actor_weight_prediction_output,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        nn.init.xavier_uniform_(self.actor_w_comp_output,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.is_bias:
                        self.actor_bias_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.out_feat))               
                        nn.init.uniform_(self.actor_bias_prediction_output)     

        elif self.gaussian_mixture:
            if self.rl_learner_type == 'Q_Learning' :
                if self.n_hidden == 0:
                    self.q_learner = nn.Sequential(
                        MDN(self.in_feat, self.out_feat, self.n_gaussians)
                    )                
                else:
                    mdn_hidden_layers = nn.ModuleList()
                    for _ in range(self.n_hidden_layers-1):
                        mdn_hidden_layers.append(nn.Linear(self.hidden_layers_size, self.hidden_layers_size, bias = self.bias))
                    self.q_learner = nn.Sequential(
                        nn.Linear(self.in_feat, self.hidden_layers_size, bias = self.bias),
                        mdn_hidden_layers,
                        MDN(self.hidden_layers_size, self.out_feat, self.n_gaussians)
                    )
                    
            if 'critic' in self.rl_learner_type.lower():
                if self.n_hidden == 0:
                    self.critic = nn.Sequential(
                        MDN(self.in_feat, self.out_feat, self.n_gaussians)
                    )                
                else:
                    mdn_hidden_layers = nn.ModuleList()
                    for _ in range(self.n_hidden_layers-1):
                        mdn_hidden_layers.append(nn.Linear(self.hidden_layers_size, self.hidden_layers_size, bias = self.bias))
                    self.critic = nn.Sequential(
                        nn.Linear(self.in_feat, self.hidden_layers_size, bias = self.bias),
                        mdn_hidden_layers,
                        MDN(self.hidden_layers_size, self.out_feat, self.n_gaussians)
                    )                
            if 'actor' in self.rl_learner_type.lower():           
                if self.n_hidden == 0:
                    self.actor = nn.Sequential(
                        MDN(self.in_feat, self.out_feat, self.n_gaussians)
                    )                
                else:
                    mdn_hidden_layers = nn.ModuleList()
                    for _ in range(self.n_hidden_layers-1):
                        mdn_hidden_layers.append(nn.Linear(self.hidden_layers_size, self.hidden_layers_size, bias = self.bias))
                    self.actor = nn.Sequential(
                        nn.Linear(self.in_feat, self.hidden_layers_size, bias = self.bias),
                        mdn_hidden_layers,
                        MDN(self.hidden_layers_size, self.out_feat, self.n_gaussians)
                    )                
            
    def predict(self, graph, device):
        
        def message_func(edges):
            pass
        def reduce_func(nodes):  
            pass    
        
        
        def apply_func(nodes):

            
            if not self.gaussian_mixture:
                if self.n_hidden == 0:
                    if self.rl_learner_type == "Q_Learning":                    
                        #UNIQUE Q LEARNER LAYER 
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            weight_prediction_input = self.weight_prediction_input.view(self.in_feat, self.num_bases, self.out_feat)#.to("cuda")     
                            weight_prediction_input = torch.matmul(self.w_comp_input, weight_prediction_input).view(self.num_nodes_types,
                                                                        self.in_feat, self.out_feat)#.to("cuda")     
                        else:
                            weight_prediction_input = self.weight_prediction_input#.to("cuda")    

                        w_prediction_input = weight_prediction_input[nodes.data['node_type']-1]#.to("cuda")   
                        bias_prediction_input = self.bias_prediction_input[nodes.data['node_type']-1]

                        if self.use_message_module:
                            #print("hid size", nodes.data['hid'].size())
                            pred = torch.bmm(nodes.data['hid'].unsqueeze(1), w_prediction_input).squeeze()
                        else:
                            pred = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), w_prediction_input).squeeze()


                        pred = pred.view(-1, self.out_feat)

                        if self.is_bias:
                            pred = pred + bias_prediction_input

                        pred = pred.squeeze()


                        if self.dropout:
                            pred = torch.nn.functional.dropout(pred, p=0.5, training=True, inplace=False)

                    if 'critic' in self.rl_learner_type.lower(): 
                        #UNIQUE CRITIC LAYER 
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            critic_weight_prediction_input = self.critic_weight_prediction_input.view(self.in_feat, self.num_bases, self.critic_out_feat)#.to("cuda")     
                            critic_weight_prediction_input = torch.matmul(self.critic_w_comp_input, critic_weight_prediction_input).view(self.num_nodes_types,
                                                                        self.in_feat, self.critic_out_feat)#.to("cuda")     
                        else:
                            critic_weight_prediction_input = self.critic_weight_prediction_input#.to("cuda")    

                        critic_w_prediction_input = critic_weight_prediction_input[nodes.data['node_type']-1]#.to("cuda")   
                        critic_bias_prediction_input = self.critic_bias_prediction_input[nodes.data['node_type']-1]
                        if self.use_message_module:
                            value = torch.bmm(nodes.data['hid'].unsqueeze(1), critic_w_prediction_input).squeeze()
                        else:
                            value = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), critic_w_prediction_input).squeeze()

                        value = value.view(-1, self.critic_out_feat)
                        if self.is_bias:
                            value = value + critic_bias_prediction_input

                        value = value.squeeze()


                    if 'actor' in self.rl_learner_type.lower(): 
                        #UNIQUE ACTOR LAYER 
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            actor_weight_prediction_input = self.actor_weight_prediction_input.view(self.in_feat, self.num_bases, self.actor_out_feat)#.to("cuda")     
                            actor_weight_prediction_input = torch.matmul(self.actor_w_comp_input, actor_weight_prediction_input).view(self.num_nodes_types,
                                                                        self.in_feat, self.actor_out_feat)#.to("cuda")     
                        else:
                            actor_weight_prediction_input = self.actor_weight_prediction_input#.to("cuda")    

                        actor_w_prediction_input = actor_weight_prediction_input[nodes.data['node_type']-1]#.to("cuda")   
                        actor_bias_prediction_input = self.actor_bias_prediction_input[nodes.data['node_type']-1]
                        if self.use_message_module:
                            actions = torch.bmm(nodes.data['hid'].unsqueeze(1), actor_w_prediction_input).squeeze()
                        else:
                            actions = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), actor_w_prediction_input).squeeze()

                        actions = actions.view(-1, self.actor_out_feat)
                        if self.is_bias:
                            actions = actions + actor_bias_prediction_input

                        actions = actions.squeeze()       






                else:
                    if self.rl_learner_type == "Q_Learning":      
                        #INPUT Q LEARNER LAYER 
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            weight_prediction_input = self.weight_prediction_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_prediction_input = torch.matmul(self.w_comp_input, weight_prediction_input).view(self.num_nodes_types,
                                                                        self.in_feat, self.hidden_layers_size)#.to("cuda")     
                        else:
                            weight_prediction_input = self.weight_prediction_input#.to("cuda")    

                        w_prediction_input = weight_prediction_input[nodes.data['node_type']-1]#.to("cuda")   
                        bias_prediction_input = self.bias_prediction_input[nodes.data['node_type']-1]
                        if self.use_message_module:
                            pred = torch.bmm(nodes.data['hid'].unsqueeze(1), w_prediction_input).squeeze()
                        else:
                            pred = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), w_prediction_input).squeeze()


                        pred = pred.view(-1, self.hidden_layers_size)    

                        if self.is_bias:
                            pred = pred + bias_prediction_input

                        pred = pred.squeeze()
                        pred = self.activation(pred)   
                        if self.dropout:
                            pred = torch.nn.functional.dropout(pred, p=0.5, training=True, inplace=False)





                    if 'critic' in self.rl_learner_type.lower():        
                        #INPUT CRITIC LAYER 
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            critic_weight_prediction_input = self.weight_prediction_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            critic_weight_prediction_input = torch.matmul(self.critic_w_comp_input, critic_weight_prediction_input).view(self.num_nodes_types,
                                                                        self.in_feat, self.hidden_layers_size)#.to("cuda")     
                        else:
                            critic_weight_prediction_input = self.critic_weight_prediction_input#.to("cuda")    

                        critic_w_prediction_input = critic_weight_prediction_input[nodes.data['node_type']-1]#.to("cuda")   
                        critic_bias_prediction_input = self.critic_bias_prediction_input[nodes.data['node_type']-1]
                        if self.use_message_module:
                            value = torch.bmm(nodes.data['hid'].unsqueeze(1), critic_w_prediction_input).squeeze()
                        else:
                            value = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), critic_w_prediction_input).squeeze()
                        value = value.view(-1, self.hidden_layers_size)    
                        if self.is_bias:
                            value = value + critic_bias_prediction_input
                        value = value.squeeze()
                        value = self.activation(value)   
                        if self.dropout:
                            value = torch.nn.functional.dropout(value, p=0.5, training=True, inplace=False)


                    if 'actor' in self.rl_learner_type.lower(): 
                        #INPUT ACTOR LAYER 
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            actor_weight_prediction_input = self.weight_prediction_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            actor_weight_prediction_input = torch.matmul(self.actor_w_comp_input, actor_weight_prediction_input).view(self.num_nodes_types,
                                                                        self.in_feat, self.hidden_layers_size)#.to("cuda")     
                        else:
                            actor_weight_prediction_input = self.actor_weight_prediction_input#.to("cuda")    

                        actor_w_prediction_input = actor_weight_prediction_input[nodes.data['node_type']-1]#.to("cuda")   
                        actor_bias_prediction_input = self.actor_bias_prediction_input[nodes.data['node_type']-1]
                        if self.use_message_module:
                            actions = torch.bmm(nodes.data['hid'].unsqueeze(1), actor_w_prediction_input).squeeze()
                        else:
                            actions = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), actor_w_prediction_input).squeeze()
                        actions = actions.view(-1, self.hidden_layers_size)    
                        if self.is_bias:
                            actions = actions + actor_bias_prediction_input
                        actions = actions.squeeze()
                        actions = self.activation(actions)   
                        if self.dropout:
                            actions = torch.nn.functional.dropout(actions, p=0.5, training=True, inplace=False)





                    #HIDDEN Q LEARNER LAYERS            

                    for idx in range(self.n_hidden -1):
                        if self.rl_learner_type == "Q_Learning":    
                            if self.num_bases < self.num_nodes_types:
                                # generate all weights from bases (equation (3))
                                weight_prediction_hid = self.prediction_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                                weight_prediction_hid = torch.matmul(self.prediction_w_comps[idx], weight_prediction_hid).view(self.num_nodes_types,
                                                                            self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                            else:
                                weight_prediction_hid = self.prediction_weights[idx]#.to("cuda")    
                            w_prediction_hid = weight_prediction_hid[nodes.data['node_type']-1]#.to("cuda")  
                            bias_prediction_hid = self.prediction_biases[idx][nodes.data['node_type']-1]
                            pred = torch.bmm(pred.unsqueeze(1), w_prediction_hid).squeeze()
                            pred = pred.view(-1, self.hidden_layers_size)
                            if self.is_bias:
                                pred = pred + bias_prediction_hid
                            pred = pred.squeeze()
                            pred = self.activation(pred)
                            if self.dropout:
                                pred = torch.nn.functional.dropout(pred, p=0.5, training=True, inplace=False)                    



                        if 'critic' in self.rl_learner_type.lower():                                

                        #HIDDEN CRITIC LEARNER LAYERS            
                            if self.num_bases < self.num_nodes_types:
                                # generate all weights from bases (equation (3))
                                critic_weight_prediction_hid = self.prediction_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                                critic_weight_prediction_hid = torch.matmul(self.critic_prediction_w_comps[idx], critic_weight_prediction_hid).view(self.num_nodes_types,
                                                                            self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                            else:
                                critic_weight_prediction_hid = self.critic_prediction_weights[idx]#.to("cuda")    
                            critic_w_prediction_hid = critic_weight_prediction_hid[nodes.data['node_type']-1]#.to("cuda")  
                            critic_bias_prediction_hid = self.critic_prediction_biases[idx][nodes.data['node_type']-1]
                            value = torch.bmm(value.unsqueeze(1), critic_w_prediction_hid).squeeze()
                            value = value.view(-1, self.hidden_layers_size)
                            if self.is_bias:
                                value = value + critic_bias_prediction_hid
                            value = value.squeeze()
                            value = self.activation(value)
                            if self.dropout:
                                value = torch.nn.functional.dropout(value, p=0.5, training=True, inplace=False)                               


                        if 'actor' in self.rl_learner_type.lower(): 
                        #HIDDEN ACTOR LEARNER LAYERS            
                            if self.num_bases < self.num_nodes_types:
                                # generate all weights from bases (equation (3))
                                actor_weight_prediction_hid = self.prediction_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                                actor_weight_prediction_hid = torch.matmul(self.actor_prediction_w_comps[idx], actor_weight_prediction_hid).view(self.num_nodes_types,
                                                                            self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                            else:
                                actor_weight_prediction_hid = self.actor_prediction_weights[idx]#.to("cuda")    
                            actor_w_prediction_hid = actor_weight_prediction_hid[nodes.data['node_type']-1]#.to("cuda")  
                            actor_bias_prediction_hid = self.actor_prediction_biases[idx][nodes.data['node_type']-1]
                            actions = torch.bmm(actions.unsqueeze(1), actor_w_prediction_hid).squeeze()
                            actions = actions.view(-1, self.hidden_layers_size)
                            if self.is_bias:
                                actions = actions + actor_bias_prediction_hid
                            actions = actions.squeeze()
                            actions = self.activation(actions)
                            if self.dropout:
                                actions = torch.nn.functional.dropout(actions, p=0.5, training=True, inplace=False)                               



                    if self.rl_learner_type == "Q_Learning":  

                        #OUTPUT Q LEARNER LAYER 
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            weight_prediction_output = self.weight_prediction_output.view(self.hidden_layers_size, self.num_bases, self.out_feat)#.to("cuda")     
                            weight_prediction_output = torch.matmul(self.w_comp_output, weight_prediction_output).view(self.num_nodes_types,
                                                                        self.hidden_layers_size, self.out_feat)#.to("cuda")     
                        else:
                            weight_prediction_output = self.weight_prediction_output#.to("cuda")    
                        w_prediction_output = weight_prediction_output[nodes.data['node_type']-1]#.to("cuda")   
                        bias_prediction_output = self.bias_prediction_output[nodes.data['node_type']-1]
                        pred = torch.bmm(pred.unsqueeze(1), w_prediction_output).squeeze()
                        pred = pred.view(-1, self.out_feat)                
                        if self.is_bias:
                            pred = pred + bias_prediction_output      
                        pred = pred.squeeze()
                        if self.norm:
                            pred = pred * nodes.data['norm']




                    if 'critic' in self.rl_learner_type.lower() and 'actor' in self.rl_learner_type.lower():         

                        # OUTPUT CRITIC LAYER
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            critic_weight_prediction_output = self.critic_weight_prediction_output.view(self.hidden_layers_size, self.num_bases, self.critic_out_feat)#.to("cuda")     
                            critic_weight_prediction_output = torch.matmul(self.critic_w_comp_output, critic_weight_prediction_output).view(self.num_nodes_types,
                                                                        self.hidden_layers_size, self.critic_out_feat)#.to("cuda")     
                        else:
                            critic_weight_prediction_output = self.critic_weight_prediction_output#.to("cuda")    
                        critic_w_prediction_output = critic_weight_prediction_output[nodes.data['node_type']-1]#.to("cuda")   
                        critic_bias_prediction_output = self.critic_bias_prediction_output[nodes.data['node_type']-1]
                        value = torch.bmm(value.unsqueeze(1), critic_w_prediction_output).squeeze()
                        value = value.view(-1, self.critic_out_feat)                
                        if self.is_bias:
                            value = value + critic_bias_prediction_output      
                        value = value.squeeze()
                        if self.norm:
                            value = value * nodes.data['norm']                        


                    if 'actor' in self.rl_learner_type.lower(): 
                        #OUTPUT ACTOR LAYER
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            actor_weight_prediction_output = self.actor_weight_prediction_output.view(self.hidden_layers_size, self.num_bases, self.actor_out_feat)#.to("cuda")     
                            actor_weight_prediction_output = torch.matmul(self.actor_w_comp_output, actor_weight_prediction_output).view(self.num_nodes_types,
                                                                        self.hidden_layers_size, self.actor_out_feat)#.to("cuda")     
                        else:
                            actor_weight_prediction_output = self.actor_weight_prediction_output#.to("cuda")    
                        actor_w_prediction_output = actor_weight_prediction_output[nodes.data['node_type']-1]#.to("cuda")   
                        actor_bias_prediction_output = self.actor_bias_prediction_output[nodes.data['node_type']-1]
                        actions = torch.bmm(actions.unsqueeze(1), actor_w_prediction_output).squeeze()
                        actions = actions.view(-1, self.actor_out_feat)                
                        if self.is_bias:
                            actions = actions + actor_bias_prediction_output      
                        actions = actions.squeeze()
                        if self.norm:
                            actions = actions * nodes.data['norm']                         


                # RETURN RESULTS            
                r = {}
                if self.rl_learner_type == "Q_Learning":
                    r['pred'] =  pred
                if 'critic'in self.rl_learner_type.lower():
                    r['value'] = value
                if 'actor' in self.rl_learner_type.lower():
                    r['actions_values'] = actions
            
                        
                        
            elif self.gaussian_mixture:        
                if self.rl_learner_type == 'Q_Learning':
                    pred_pi, pred_sigma, pred_mu  = self.q_learner.forward(graph.ndata['hid'])
                if 'critic' in self.rl_learner_type.lower():
                    value_pi, value_sigma, value_mu = self.critic.forward(graph.ndata['hid'])
                if 'actor' in self.rl_learner_type.lower(): 
                    actions_pi, actions_sigma, actions_mu  = self.actor.forward(graph.ndata['hid'])
                        
                # RETURN RESULTS            
                r = {}
                if self.rl_learner_type == "Q_Learning":
                    r['pred_pi'] =  pred_pi
                    r['pred_sigma'] = pred_sigma
                    r['pred_mu'] = pred_mu
                if 'critic'in self.rl_learner_type.lower():
                    r['value_pi'] =  value_pi
                    r['value_sigma'] = value_sigma
                    r['value_mu'] = value_mu
                if 'actor' in self.rl_learner_type.lower():
                    r['actions_pi'] =  actions_pi
                    r['actions_sigma'] = actions_sigma
                    r['actions_mu'] = actions_mu

            return r



        graph.update_all(message_func, reduce_func, apply_node_func = apply_func)  
        r = []
        #if self.value_model_based:
        r.append(graph.ndata['hid'])
            
        if not self.gaussian_mixture:
            if self.rl_learner_type == "Q_Learning":
                r.append(graph.ndata['pred'])
            if 'actor' in self.rl_learner_type.lower():
                r.append(graph.ndata['actions_values'])
            if 'critic'in self.rl_learner_type.lower():
                r.append(graph.ndata['value'])
                
        elif self.gaussian_mixture:
            if self.rl_learner_type == "Q_Learning":
                r.append([graph.ndata['pred_pi'], graph.ndata['pred_sigma'], graph.ndata['pred_mu']])
            if 'actor' in self.rl_learner_type.lower():
                r.append([graph.ndata['value_pi'], graph.ndata['value_sigma'], graph.ndata['value_mu']])
            if 'critic'in self.rl_learner_type.lower():
                r.append([graph.ndata['actions_pi'], graph.ndata['actions_sigma'], graph.ndata['actions_mu']])         
        return r

    
        """
        if self.rl_learner_type == "Q_Learning":        
            return graph.ndata["pred"] 
        elif 'critic' in self.rl_learner_type.lower() and 'actor' in self.rl_learner_type.lower():
            return graph.ndata['actions_values'] , graph.ndata['value']    
        """