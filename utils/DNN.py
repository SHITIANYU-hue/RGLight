import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from torch.autograd import Variable
import collections


class DNN_IQL(nn.Module):
    def __init__(self, resnet, rl_learner_type, policy, noisy, double, target_model,n_actions, dueling, tl_input_dims, activation, n_workers, batch_size, mini_batch_size, sigma_init=0.017, max_phase_len = None):
        super(DNN_IQL, self).__init__()
        self.resnet = resnet
        self.rl_learner_type = rl_learner_type
        self.policy = policy
        self.noisy = noisy
        self.double = double
        self.target_model = target_model
        if policy == 'binary':
            self.out_dim = n_actions
        else:
            self.out_dim = max_phase_len
            assert type(max_phase_len) == int, 'max_phase_len must be an integer'
        self.dueling = dueling
        if self.rl_learner_type != 'Q_Learning':
            self.recurrent = True
        else:
            self.recurrent = False
        if self.dueling or self.rl_learner_type != "Q_Learning":
            self.out_dim += 1
        self.input_models = nn.ModuleDict()
        self.tl_input_dims = tl_input_dims
        self.n_tls = len(tl_input_dims)
        self.activation = activation
        self.n_workers = n_workers
        self.batch_size = batch_size
        l_comput = []
        l_train_simple = []
        l_train_double = []
        l_test = []
        # PER INTERSECTION PARA
        
        self.gath_idx_comput = torch.tensor(list(range(self.n_tls)),dtype = torch.long).repeat(n_workers)
        self.gath_idx_train_simple = torch.tensor(list(range(self.n_tls)),dtype = torch.long).repeat(batch_size)
        self.gath_idx_mini_batch = torch.tensor(list(range(self.n_tls)),dtype = torch.long).repeat(mini_batch_size)
        self.gath_idx_train_double = torch.tensor(list(range(self.n_tls)),dtype = torch.long).repeat(2*batch_size)
        self.gath_idx_test = torch.tensor(list(range(self.n_tls)),dtype = torch.long).repeat(1)

        """
        for idx, (tl_id, input_dim) in enumerate(tl_input_dims.items()):
            #self.input_models[tl_id] = DNN(input_dim = input_dim, activation = F.elu)
            #GLOBAL PARA
            l_comput.append(torch.ones(n_workers, dtype = torch.long)*idx)
            l_train_simple.append(torch.ones(batch_size,dtype = torch.long)*idx)
            l_train_double.append(torch.ones(2*batch_size, dtype = torch.long)*idx)
            l_test.append(torch.ones(1, dtype = torch.long)*idx)

        self.gath_idx_comput = torch.cat((tuple(l_comput)),0)
        self.gath_idx_train_simple = torch.cat((tuple(l_train_simple)),0)
        self.gath_idx_train_double = torch.cat((tuple(l_train_double)),0)
        self.gath_idx_test = torch.cat((tuple(l_test)),0) 
        """
        
   
        #L
        self.weight_inp = nn.Parameter(torch.Tensor(self.n_tls, max(self.tl_input_dims.values()),
                                                        256))
        nn.init.xavier_uniform_(self.weight_inp,
                                gain=nn.init.calculate_gain('relu'))
        self.bias_inp = nn.Parameter(torch.Tensor(self.n_tls, 256))                
        nn.init.uniform_(self.bias_inp)  
        
        
        if self.noisy and False:
            self.sigma_weight_inp = nn.Parameter(torch.Tensor(self.n_tls, max(self.tl_input_dims.values()),
                                                        256).fill_(sigma_init))
            self.register_buffer("epsilon_weight_inp", torch.zeros(self.n_tls, max(self.tl_input_dims.values()),
                                                        256))
            self.sigma_bias_inp = nn.Parameter(torch.Tensor(self.n_tls, 256).fill_(sigma_init))
            self.register_buffer("epsilon_bias_inp", torch.zeros(self.n_tls, 256))
            std = math.sqrt(3 / max(self.tl_input_dims.values()))
            #nn.init.uniform(self.weight_inp, -std, std)
            #nn.init.uniform(self.bias_inp, -std, std)              

        
        
        #L1
        self.weight = nn.Parameter(torch.Tensor(self.n_tls, 256,
                                                        128))
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        self.bias = nn.Parameter(torch.Tensor(self.n_tls, 128))                
        nn.init.uniform_(self.bias)   
        
        if self.noisy and False:
            self.sigma_weight = nn.Parameter(torch.Tensor(self.n_tls, 256,
                                                        128).fill_(sigma_init))
            self.register_buffer("epsilon_weight", torch.zeros(self.n_tls, 256,
                                                        128))
            self.sigma_bias = nn.Parameter(torch.Tensor(self.n_tls, 128).fill_(sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(self.n_tls, 128))
            std = math.sqrt(3 / 256)
        
        #L2
        if self.recurrent:
            # BETTER PARRAL
            self.weight_input_aggregation_input = nn.Parameter(torch.Tensor(self.n_tls, 128, 3*64))
            self.weight_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.n_tls, 64, 3*64))            
            nn.init.xavier_uniform_(self.weight_input_aggregation_input, gain=nn.init.calculate_gain('relu'))            
            nn.init.xavier_uniform_(self.weight_hidden_aggregation_input, gain=nn.init.calculate_gain('relu'))            
            self.bias_input_aggregation_input = nn.Parameter(torch.Tensor(self.n_tls, 3*64))
            self.bias_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.n_tls, 3*64))
            nn.init.uniform_(self.bias_input_aggregation_input)   
            nn.init.uniform_(self.bias_hidden_aggregation_input)   


            
        else:
            #L2
            self.weight_1 = nn.Parameter(torch.Tensor(self.n_tls, 128,64))
            nn.init.xavier_uniform_(self.weight_1,
                                    gain=nn.init.calculate_gain('relu'))
            self.bias_1 = nn.Parameter(torch.Tensor(self.n_tls, 64))                
            nn.init.uniform_(self.bias_1)   
            
            if self.noisy and False:
                self.sigma_weight_1 = nn.Parameter(torch.Tensor(self.n_tls, 128, 64).fill_(sigma_init))
                self.register_buffer("epsilon_weight_1", torch.zeros(self.n_tls, 128, 64))
                self.sigma_bias_1 = nn.Parameter(torch.Tensor(self.n_tls, 64).fill_(sigma_init))
                self.register_buffer("epsilon_bias_1", torch.zeros(self.n_tls, 64))
                std = math.sqrt(3 / 128)
                #nn.init.uniform(self.weight_1, -std, std)
                #nn.init.uniform(self.bias_1, -std, std)             
            
        #L3
        self.weight_2 = nn.Parameter(torch.Tensor(self.n_tls, 64,
                                                        self.out_dim))
        nn.init.xavier_uniform_(self.weight_2,
                                    gain=nn.init.calculate_gain('relu'))
        self.bias_2 = nn.Parameter(torch.Tensor(self.n_tls, self.out_dim))     
        nn.init.uniform_(self.bias_2)  
        
        if self.noisy:
            self.sigma_weight_2 = nn.Parameter(torch.Tensor(self.n_tls, 64, self.out_dim).fill_(sigma_init))
            self.register_buffer("epsilon_weight_2", torch.zeros(self.n_tls, 64, self.out_dim))
            self.sigma_bias_2 = nn.Parameter(torch.Tensor(self.n_tls, self.out_dim).fill_(sigma_init))
            self.register_buffer("epsilon_bias_2", torch.zeros(self.n_tls, self.out_dim))
            std = math.sqrt(3 / 64)         


        
    def forward(self, batched_state,device, learning = False, joint = False, testing = False, actions_sizes = None):
        n = len(batched_state)
        #print("batched_state", batched_state)
        #print("n", n)
    
        """
        y=[]
        for idx, tl_id in enumerate(self.tl_input_dims):
            l = []
            #print("tl_id", tl_id)
            for instance in range(n):
                #print("instance", instance)
                #print("l elem", torch.FloatTensor(batched_state[instance][idx]).view(1,-1))
                l.append(torch.FloatTensor(batched_state[instance][idx]).view(1,-1))
                #print("batched_state[instance][idx]", batched_state[instance][idx])
                #print("l list", l)
            l = torch.cat(tuple(l),0)
            #print("l cat", l)
            k = self.input_models[tl_id].forward(l.to(device))
            y.append(k)
            #print("y elem", k)
            #print("y list", y)
            
        y = torch.cat(tuple(y),0)
        if self.rl_learner_type == 'Q_Learning':
            if learning:
                if ((self.target_model is None or self.double) and joint):
                    l = self.gath_idx_train_double
                else:
                    l = self.gath_idx_train_simple
            else:
                l = self.gath_idx_comput
        else:
            if learning:
                l = self.gath_idx_mini_batch
            else:
                l = self.gath_idx_comput
        """
        
        
        l = torch.tensor(list(range(self.n_tls)),dtype = torch.long).repeat(n)
        
            
        #print("gath size", l.size())
        #print("gath", l)
        #L_inp
        w = self.weight_inp[l]
        b = self.bias_inp[l]
        y = torch.tensor(batched_state).view(n*self.n_tls,self.weight_inp.size()[1]).to(device)
        if self.noisy and not testing and False:
            e_w = torch.cuda.FloatTensor(self.sigma_weight_inp[l].size(), device=device).normal_()              
            w += self.sigma_weight_inp[l] * Variable(e_w)
            e_b = torch.cuda.FloatTensor(self.sigma_bias_inp[l].size(), device=device).normal_()  
            b += self.sigma_bias_inp[l] * Variable(e_b)
        y = torch.bmm(y.unsqueeze(1),w).squeeze()    
        y = y + b
        y = self.activation(y)
        
        
        #L1
        w = self.weight[l]
        b = self.bias[l]
        if self.noisy and not testing and False:
            e_w = torch.cuda.FloatTensor(self.sigma_weight[l].size(), device=device).normal_()              
            w += self.sigma_weight[l] * Variable(e_w)
            e_b = torch.cuda.FloatTensor(self.sigma_bias[l].size(), device=device).normal_()  
            b += self.sigma_bias[l] * Variable(e_b)        
        y = torch.bmm(y.unsqueeze(1),w).squeeze()    
        y = y + b
        y = self.activation(y)
        
        
        
        # L2
        
        if self.recurrent:
             # FORWARD UNIQUE
            weight_input_aggregation_input = self.weight_input_aggregation_input     
            weight_hidden_aggregation_input = self.weight_hidden_aggregation_input
            w_input_aggregation_input = weight_input_aggregation_input[l]#.to("cuda")   
            w_hidden_aggregation_input = weight_hidden_aggregation_input[l]
            bias_input_aggregation_input = self.bias_input_aggregation_input[l]
            bias_hidden_aggregation_input = self.bias_hidden_aggregation_input[l]
            gate_x = torch.bmm(y.unsqueeze(1), w_input_aggregation_input).squeeze()
            gate_h = torch.bmm(self.hid.to(device).unsqueeze(1), w_hidden_aggregation_input).squeeze()
            gate_x += bias_input_aggregation_input
            gate_h += bias_hidden_aggregation_input
            i_r, i_i, i_n = gate_x.chunk(3, 1)
            h_r, h_i, h_n = gate_h.chunk(3, 1)                
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + (resetgate * h_n))
            y = newgate + inputgate * (self.hid.to(device) - newgate)
            if self.resnet:
                y += self.hid.to(device)
            
            self.hid = y

        else:
            #print("y", y.size())
            #print("y unsqueeze", y.unsqueeze(1).size())
            #L2
            w = self.weight_1[l]
            b = self.bias_1[l]
            if self.noisy and not testing and False:
                e_w = torch.cuda.FloatTensor(self.sigma_weight_1[l].size(), device=device).normal_()              
                w += self.sigma_weight_1[l] * Variable(e_w)
                e_b = torch.cuda.FloatTensor(self.sigma_bias_1[l].size(), device=device).normal_()  
                b += self.sigma_bias_1[l] * Variable(e_b)    



            y = torch.bmm(y.unsqueeze(1),w).squeeze()
            y = y + b
            y = self.activation(y)

        
        
        
        
        
        #L3
        w = self.weight_2[l]
        b = self.bias_2[l]
        if self.noisy and not testing:
            #torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
            #torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
            #w += self.sigma_weight[l] * Variable(self.epsilon_weight[l])
            #b += self.sigma_bias[l] * Variable(self.epsilon_bias[l])
            # FOR MORE VARIABILITY 
            e_w = torch.cuda.FloatTensor(self.sigma_weight_2[l].size(), device=device).normal_()              
            w += self.sigma_weight_2[l] * Variable(e_w)
            #e_w = torch.randn(self.epsilon_weight[l].size()).to(device)
            e_b = torch.cuda.FloatTensor(self.sigma_bias_2[l].size(), device=device).normal_()  
            b += self.sigma_bias_2[l] * Variable(e_b)
                #e_b = torch.randn(self.epsilon_bias[l].size()).to(device)
            #w += self.sigma_weight[l] * Variable(e_w)
            #b += self.sigma_bias[l] * Variable(e_b)
            
        
        y = torch.bmm(y.unsqueeze(1),w).squeeze()
        y = y + b




                
        if self.rl_learner_type != 'Q_Learning':
            if self.policy != 'binary':
                a, v = y[:,1:],y[:,0]  
                actions_sizes_list = tuple(actions_sizes.tolist())    
                l_a = []
                for idx,i in enumerate(actions_sizes):
                    l_a.append(a[idx][1:i])
                return None, l_a, v
            else:
                a, v = y[:,1:],y[:,0]  
                return None, a,v 
 
        else:
            if self.policy != 'binary':
                actions_sizes_list = tuple(actions_sizes.tolist())    
                l_q = []
                for idx,i in enumerate(actions_sizes):
                    l_q.append(y[idx][:i+(1 if self.dueling else 0)])

            if self.dueling:
                if self.policy != 'binary':
                    if not learning:
                        l_Q = torch.zeros(actions_sizes.size(), dtype=torch.int8, device=device)                       
                    else:
                        l_Q = [0]*len(l_q)

                    for dim in list(set(actions_sizes_list)):
                        positions = [i for i, n in enumerate(actions_sizes_list) if n == dim]
                        a_ = torch.cat(tuple(l_q[i].view(1,-1) for i in positions),dim=0)
                        #print("a_", a_)
                        v_ = a_[:,0]
                        a_ = a_[:,1:]
                        #print('a_',a_.size())
                        #print('v_',v_.size())
                        #print("a_ - torch.mean(a_, dim = 1).unsqueeze(1)",a_ - torch.mean(a_, dim = 1).unsqueeze(1))
                        q_ = (v_.view(-1,1) + (a_ - torch.mean(a_, dim = 1).unsqueeze(1))).squeeze()
                        #print('q_',q_)
                        q_ = q_.view(-1,dim)
                        for idx,position in enumerate(positions):
                            if not learning:
                                _ , l_Q[position] = torch.max(q_[idx],dim = 0)             
                            else:
                                l_Q[position] = q_[idx]    

                    return None, l_Q


                else:
                    a, v = y[:,1:],y[:,0]
                    y = (v.unsqueeze(1) + (a - torch.mean(a, dim = 1).unsqueeze(1))).squeeze()
                    return None , y


            else :
                if self.policy != 'binary':
                    if not learning:
                        l_Q = torch.zeros(actions_sizes.size(), dtype=torch.int8, device=device)                       
                    else:
                        l_Q = [0]*len(l_q)

                    for dim in list(set(actions_sizes_list)):
                        positions = [i for i, n in enumerate(actions_sizes_list) if n == dim]
                        q_ = torch.cat(tuple(l_q[i].view(1,-1) for i in positions),dim=0)
                        q_ = q_.view(-1,dim)
                        for idx,position in enumerate(positions):
                            if not learning:
                                _ , l_Q[position] = torch.max(q_[idx],dim = 0)             
                            else:
                                l_Q[position] = q_[idx]                 

                    #print(l_Q)
                    return None, l_Q

                else:
                    return None , y



    def init_hidden(self, batched_state, device):
        n = len(batched_state)
        #y = torch.tensor(batched_state).view(n*self.n_tls,64).to(device)
        self.hid = torch.zeros((n*self.n_tls, 64), dtype = torch.float32, device = device)

                
class DNN(nn.Module):
    def __init__(self, input_dim, activation):
        super(DNN, self).__init__()
        self.activation = activation
        self.input_dim = input_dim
        self.input_layer = nn.Linear(self.input_dim, 256)
        """
        self.wave_input_layer = nn.Linear(self.input_dim, 128)
        self.wait_input_layer = nn.Linear(self.input_dim, 32)
        self.hidden_layer = nn.Linear(128+32, 64)
        self.output_layer = nn.Linear(64,1)
        """
        
    def forward(self,x):
        return self.activation(self.input_layer(x))
        """
        y1 = self.activation(self.wave_input_layer(wave))
        y2 = self.activation(self.wait_input_layer(wait))
        return y1,y2
        y = torch.cat((y1,y2),1) #2 ? 
        y = self.activation(self.hidden_layer(y))
        return torch.sigmoid(self.output_layer(y))
        """

