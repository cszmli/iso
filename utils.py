from __future__ import print_function
import numpy as np
import torch
import os



class cfg():
    def __init__(self):
        ppo_config = cfg_ppo()
        state_dim = 10
        action_dim = 5
        action_std = 0.5
        op = 'adam'
        clip = 10.0
        gamma = 0.99
        max_step = 20




class cfg_ppo():
    def __init__(self):
        render = False
        solved_reward = 300         # stop training if avg_reward > solved_reward
        log_interval = 20           # print avg reward in the interval
        max_episodes = 10000        # max training episodes
        max_timesteps = 1500        # max timesteps in one episode
        
        update_timestep = 4000      # update policy every n timesteps
        action_std = 0.5            # constant std for action distribution (Multivariate Normal)
        K_epochs = 80               # update policy for K epochs
        eps_clip = 0.2              # clip parameter for PPO
        gamma = 0.99                # discount factor
        
        lr = 0.0003                 # parameters for Adam optimizer
        betas = (0.9, 0.999)
