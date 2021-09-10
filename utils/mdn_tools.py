import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from utils.new_functions import *
from utils.auto_initialize import *
import torch
import numpy as np 
import dgl
from functools import partial
from torch.autograd import Variable
import copy
from utils.mdn import *
from IPython.display import clear_output
import glob
import atexit
import time
import traceback

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
