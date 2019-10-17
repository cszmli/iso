# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import os
from collections import defaultdict
import logging
from torch.autograd import Variable
import torch.nn as nn
import json
import torch.nn.functional as F
import copy
import sys
from torch.distributions import MultivariateNormal,  Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INT = 0
LONG = 1
FLOAT = 2
def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var

class BaseModel(nn.Module):
    def __init__(self, config):
        self.config = config
        self.use_gpu = config.use_gpu
    
    def cast_gpu(self, var):
        if self.use_gpu:
            return var.cuda()
        else:
            return var

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        if type(inputs)==list:
            return cast_type(Variable(torch.Tensor(inputs)), dtype,
                         self.use_gpu)
        elif type(inputs)==torch.Tensor:
            return cast_type(inputs, dtype, self.use_gpu)
        return cast_type(Variable(torch.from_numpy(inputs)), dtype,
                         self.use_gpu)
    def forward(self, *input):
        raise NotImplementedError

    def get_optimizer(self):
        if self.config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=self.config.init_lr, betas=(0.5, 0.999))
        elif self.config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(self.parameters(), lr=self.config.init_lr,
                                   momentum=self.config.momentum)
        elif self.config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(self.parameters(), lr=self.config.init_lr,
                                       momentum=self.config.momentum)
    def clip_gradient(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip)



class ActorCriticContinuous(BaseModel):
    def __init__(self, config):
        super(ActorCriticContinuous, self).__init__(config)
        state_dim = config.state_dim
        action_dim =config.action_dim
        action_std = config.action_std
        # action mean range -1 to 1
        # std is fixed
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self, state):
        return self.actor(state)
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = torch.squeeze(self.actor(state))
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class ActorCriticDiscrete(BaseModel):
    def __init__(self, config):
        super(ActorCriticDiscrete, self).__init__(config)
        state_dim = config.state_dim
        action_dim =config.action_dim
        action_std = config.action_std
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self, x):
        raise torch.argmax(self.actor(x), -1)
    
    def act(self, state, memory):
        action_weights = self.actor(state)
        a_probs = torch.softmax(a_weights, -1)
        dist = Categorical(a_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_weights = self.actor(state)
        a_probs = torch.softmax(a_weights, -1)
        dist = Categorical(a_probs)
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        

class RewardModule(BaseModel):
    """
    label: 1 for real, 0 for generated
    """
    def __init__(self, config):
        super(RewardModule, self).__init__(config)
        
        self.gamma = config.gamma
        self.g = nn.Sequential(nn.Linear(config.state_dim+config.action_dim, 100),
                               nn.ReLU(),
                               nn.Linear(config.hidden_dim, 1))
        self.h = nn.Sequential(nn.Linear(config.state_dim, config.hidden_dim),
                               nn.ReLU(),
                               nn.Linear(config.hidden_dim, 1))
    
    def forward(self, s, a, next_s):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :param next_s: [b, s_dim]
        :return:  [b, 1]
        """
        weights = self.g(torch.cat([s,a], -1)) + self.gamma * self.h(next_s) - self.h(s)
        return weights
    

##########################################################
##########################################################
##########################################################
##########################################################


class DiscretePolicy(nn.Module):
    def __init__(self, config):
        super(DiscretePolicy, self).__init__()

        self.net = nn.Sequential(nn.Linear(config.s_dim, config.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(config.h_dim, config.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(config.h_dim, config.a_dim))

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.net(s)

        return a_weights

    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [1]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.softmax(a_weights, 0)

        # randomly sample from normal distribution, whose mean and variance come from policy network.
        # [a_dim] => [1]
        a = a_probs.multinomial(1) if sample else a_probs.argmax(0, True)

        return a

    def get_log_prob(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, 1]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.softmax(a_weights, -1)

        # [b, a_dim] => [b, 1]
        trg_a_probs = a_probs.gather(-1, a)
        log_prob = torch.log(trg_a_probs)

        return log_prob
        

class ContinuousPolicy(nn.Module):
    def __init__(self, config):
        super(ContinuousPolicy, self).__init__()

        self.net = nn.Sequential(nn.Linear(config.s_dim, config.h_dim),
                                 nn.ReLU(),
                                 nn.Linear(config.h_dim, config.h_dim),
                                 nn.ReLU())
        self.net_mean = nn.Linear(config.h_dim, config.a_dim)
        self.net_std = nn.Linear(config.h_dim, config.a_dim)

    def forward(self, s):
        # [b, s_dim] => [b, h_dim]
        h = self.net(s)

        # [b, h_dim] => [b, a_dim]
        a_mean = self.net_mean(h)
        a_log_std = self.net_std(h)

        return a_mean, a_log_std

    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action mean and log_std
        # [s_dim] => [a_dim]
        a_mean, a_log_std = self.forward(s)

        # randomly sample from normal distribution, whose mean and variance come from policy network.
        # [a_dim]
        a = torch.normal(a_mean, a_log_std.exp()) if sample else a_mean

        return a

    def get_log_prob(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        def normal_log_density(x, mean, log_std):
            """
            x ~ N(mean, std)
            this function will return log(prob(x)) while x belongs to guassian distrition(mean, std)
            :param x:       [b, a_dim]
            :param mean:    [b, a_dim]
            :param log_std: [b, a_dim]
            :return:        [b, 1]
            """
            std = log_std.exp()
            var = std.pow(2)
            log_density = - (x - mean).pow(2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std
        
            return log_density.sum(-1, keepdim=True)
        
        # forward to get action mean and log_std
        # [b, s_dim] => [b, a_dim]
        a_mean, a_log_std = self.forward(s)

        # [b, a_dim] => [b, 1]
        log_prob = normal_log_density(a, a_mean, a_log_std)

        return log_prob
    
    
class Value(nn.Module):
    def __init__(self, config):
        super(Value, self).__init__()

        self.net = nn.Sequential(nn.Linear(config.s_dim, config.hv_dim),
                                 nn.ReLU(),
                                 nn.Linear(config.hv_dim, config.hv_dim),
                                 nn.ReLU(),
                                 nn.Linear(config.hv_dim, 1))

    def forward(self, s):
        """
        :param s: [b, s_dim]
        :return:  [b, 1]
        """
        value = self.net(s)

        return value

