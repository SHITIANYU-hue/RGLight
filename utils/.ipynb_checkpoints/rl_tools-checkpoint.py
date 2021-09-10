import torch
import numpy as np
import dgl
import torch.nn.functional as F

def gaussian_probability(sigma, mu, data):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    data = data.unsqueeze(1).unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)
    
def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    #print("pi",pi.size(),"sigma", sigma.size(), "mu", mu.size())
    categorical = Categorical(pi)
    #print("pi", pi)
    #print("categorical", categorical)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    #print("pis", pis)
    #print("len pis",len(pis))
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample
    

def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    #print("prob", prob)
    nll = - torch.logsumexp(prob, dim = 1, keepdim=False, out=None)
    #-torch.log(torch.sum(prob, dim=1))
    #print("nll", nll)
    return torch.mean(nll)

def compute_gae(rewards, values, gamma=0.99, tau=0.95):
    #print("rewards", rewards.size())
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        #print("rewards[step]", rewards[step])
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * tau * gae
        returns.insert(0, gae + values[step])
    #print("returns", returns)
    #print("\n\n\n\n\n\nfinal returns", torch.stack(returns).to("cpu"))
    return torch.stack(returns).to("cpu")


def ppo_iter(gcn, mini_batch_size, states, actions, log_probs, returns, advantage, device):
    
    
    idxs = list(range(states.size()[0]))  
    batch_size = len(states)
    print("b_s", batch_size)
    print("mini_batch_size", mini_batch_size)
    print("states", states.size())
    idxs = np.random.choice(idxs, size = batch_size // mini_batch_size, replace = False).tolist()
    for idx in idxs:
        #yield states[idx:idx+mini_batch_size] , actions[idx:idx+mini_batch_size].to(device), log_probs[idx:idx+mini_batch_size].to(device), returns[idx:idx+mini_batch_size].to(device), advantage[idx:idx+mini_batch_size].to(device)        
        for idx2 in range(mini_batch_size):   
            if gcn:
                if idx2 == 0:
                    yield dgl.batch([states[idx] for idx in range(start = idx, end = idx+idx2)]) , actions[idx+idx2].to(device), log_probs[idx+idx2].to(device), returns[idx+idx2].to(device), advantage[idx+idx2].to(device)    
                else:
                    pass
            else:
                if idx2 == 0:
                    yield states[idx+idx2] , actions[idx+idx2].to(device), log_probs[idx+idx2].to(device), returns[idx+idx2].to(device), advantage[idx+idx2].to(device) 
                else:
                    pass
                    
            yield states[idx+idx2] , actions[idx+idx2].to(device), log_probs[idx+idx2].to(device), returns[idx+idx2].to(device), advantage[idx+idx2].to(device)        
        
    if False:

        #print("states[0]", states[0].ndata['state'])
        #print("rewards")
        batch_size = len(states)-1
        rand_ids_ts = np.random.choice(range(batch_size), size =(batch_size // mini_batch_size, mini_batch_size), replace = False)
        #for _ in range(batch_size // mini_batch_size):
        #print("actions", actions.size())
        for rand_ids in rand_ids_ts:
            rand_ids = list(rand_ids)
            #rand_ids = np.random.randint(0, batch_size-1, mini_batch_size)
            #print(rand_ids)
            #print("states", [states[idx] for idx in rand_ids])
            #print("actions", actions[rand_ids, :].size())
            #print("log_probs", log_probs[rand_ids, :].size())
            #print("returns", returns[rand_ids, :].size())
            #print("advantage", advantage[rand_ids, :].size())
            yield dgl.batch([states[idx] for idx in rand_ids]) if gcn else states[rand_ids, :], actions[rand_ids, :].to(device), log_probs[rand_ids, :].to(device), returns[rand_ids, :].to(device), advantage[rand_ids, :].to(device)
        
        
def ppo_update(policy, gcn, ppo_epochs, accumulation_steps, mini_batch_size, states, actions, actions_sizes, log_probs, returns, advantages, clip_param=0.2, device = 'cuda', model = None):
    Critic_Loss = 0 
    Actor_Loss = 0 
    for ppo_epoch in range(ppo_epochs):
        for idx, (state, action, old_log_prob, return_, advantage) in enumerate(ppo_iter(model, gcn, mini_batch_size, states, actions, log_probs, returns, advantages, device)):
            print("state size",state.size())
            _, dist, values = model.forward(state, device, learning =  False, testing = False, actions_sizes = torch.cat(tuple(actions_sizes),dim=0))
            if policy != 'binary':
                actions_sizes_list = actions_sizes.tolist()
                new_log_probs = torch.zeros(len(actions_sizes_list), dtype=torch.float64, device=device)  #      [0]*len(actions_sizes_list)
                for dim in list(set(actions_sizes_list)):
                    positions = [i for i, n in enumerate(actions_sizes_list) if n == dim] 
                    dist_ = torch.cat(tuple(dist[i].view(1,-1) for i in positions),dim=0)
                    actions_ = torch.cat(tuple(actions[i].view(1,-1) for i in positions),dim=0)
                    probs_init = F.softmax(dist, dim = 1)#.view(mini_batch_size,-1,2)
                    log_probs_init = F.log_softmax(dist, dim = 1)#.view(mini_batch_size,-1,2)
                    new_log_probs = log_probs_init.gather(2, Variable(actions))#.view(mini_batch_size, -1, 1)))                    
                    for idx,position in enumerate(positions):
                        new_log_probs[position] = new_log_probs_[idx]    
            else:
                probs_init = F.softmax(dist, dim = 1)#.view(mini_batch_size,-1,2)
                log_probs_init = F.log_softmax(dist, dim = 1)#.view(mini_batch_size,-1,2)
                new_log_probs = log_probs_init.gather(2, Variable(action))#.view(mini_batch_size, -1, 1)))
            
            
            
            #print("new_log", new_log_probs.size())
            #entropy = dist.entropy().mean()
            #new_log_probs = dist.log_prob(actions)
            #entropy = (-(log_probs_init * probs_init)).mean()
            ratio = (new_log_probs.squeeze() - old_log_prob).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value.view(mini_batch_size,-1)).pow(2).mean()

            Actor_Loss += actor_loss.item()
            Critic_Loss += critic_loss.item()
            
            loss = 1 * critic_loss + actor_loss #- 0.01 * entropy
            #print("grads before", model.conv_layers[0].layers['message_module'].weight_message_input.grad)
            loss.backward()
           
            if (idx+1) % accumulation_steps == 0:
                #print("grads after", model.conv_layers[0].layers['message_module'].weight_message_input.grad)
                model.optimizer.step()
                model.optimizer.zero_grad()
            
    return model, Actor_Loss, Critic_Loss
            
            

class Memory():
    def __init__(self, manager, max_size):
        self.manager = manager
        self.buffer = self.manager.list()
        self.max_size = max_size
        
    def add(self, experience):
        if len(self.buffer)>=self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]

