import torch
import random
import numpy as np 
import copy 

class AIRL(object):
    def __init__(self, config=None, user_policy=None, user_reward=None, system_policy=None):
        # the parameters of user policy and user reward will not be loaded. The modules are just copies to keep the structure of 
        # the modules.
        self.config = config
        self.user_policy = user_policy
        self.reward_module = user_reward
        self.system_policy = system_policy

class InteractAgent(object):
    def __init__(self, config=None, user_policy=None, user_reward=None, system_policy=None):
        self.config = config
        self.user_policy = user_policy  # this should be a PPO agent with discrete action space
        self.reward_module = user_reward  # this is an MLP taking as input the state representation
        self.system_policy = system_policy # this should be a PPO agent with continuous action space
        self.airl = AIRL(config=config, 
                         user_policy=copy.deepcopy(user_policy), 
                         user_reward=user_reward,
                         system_policy=system_policy
                         )
    
    def irl_train(self, system_policy, expert_data):
        #TODO: feed system_policy and expert_data to AIRL to get the new user_policy and reward_function.
        raise NotImplementedError("not finsihed yet")

    def system_train(self, ):
        #TODO: update the system_policy (PPO) in the second MDP  
        raise NotImplementedError("not finsihed yet")
