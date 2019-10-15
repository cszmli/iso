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


class MlpDiscrete(BaseModel):
    def __init__(self, config, input_size, output_size):
        super(MlpDiscrete, self).__init__(config)
        self.input_size = input_size
        self.output_size = output_size
        self.mid_size = (input_size + output_size)//2
        self.body = nn.Sequential(
            nn.Linear(self.input_size, self.mid_size)
            nn.ReLU(),
            nn.Linear(self.mid_size, self.output_size)
        )

    def forward(self, x):
        h = self.body(self.cast_gpu(x))
        prob = F.softmax(h, -1)
        return h, prob, prob.log()

    def predict(self, x):
        prob = self.forward(x)
        return torch.argmax(prob, -1)

class MlpContinuous(BaseModel):
    def __init__(self, config, input_size, output_size):
        super(MlpContinuous, self).__init__(config)
        self.input_size = input_size
        self.output_size = output_size
        self.mid_size = (input_size + output_size)//2
        self.body = nn.Sequential(
            nn.Linear(self.input_size, self.mid_size)
            nn.ReLU(),
            nn.Linear(self.mid_size, self.output_size)
        )

    def forward(self, x):
        h = self.body(self.cast_gpu(x))
        prob = F.softmax(h, -1)
        return h, prob, prob.log()



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
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