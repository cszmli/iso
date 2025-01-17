from __future__ import print_function
import numpy as np
import torch
import os
import logging
import time
import argparse

def init_logging_handler(log_dir, extra=''):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(log_dir, current_time+extra))
    logging.basicConfig(handlers=[stderr_handler, file_handler], format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_epochs', type=int, default=10 )
    parser.add_argument('--state_dim', type=int, default=10, )
    parser.add_argument('--action_dim', type=int, default=10,)
    parser.add_argument('--max_episodes', type=int, default=20000,)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    # parser.add_argument('--gamma', type=float, default=0.99, help='Discounted factor')
    # parser.add_argument('--epsilon', type=float, default=0.2, help='Clip epsilon of ratio r(theta)')
    parser.add_argument('--kl_factor', type=float, default=0.01, help='constraint factor between init system and updated system')
    # parser.add_argument('--clip', type=float, default=0.03, help='Clipping parameter on WGAN')
    parser.add_argument('--use_airl_user', dest='use_airl_user', action='store_true', default=False, help='')    
    parser.add_argument('--use_airl_reward', dest='use_airl_reward', action='store_true', default=False, help='')
    return parser

class cfg():
    def __init__(self):
        self.ppo_config = cfg_ppo()
        self.state_dim = 10
        self.action_dim = 5
        self.action_std = 0.5
        self.op = 'adam'
        self.master_epochs = 10
        self.clip = 10.0
        self.clamp = 0.03
        self.gamma = 0.99
        self.max_step = 20
        self.use_gpu = False
        self.data_size = {'train':50000, 'valid':5000, 'test':5000}
        self.max_episodes = 30000
        self.anneal = 10
        self.batch_size = 64
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.K_epochs = 80               # update policy for K epochs
        self.eps_clip = 0.2              # clip parameter for PPO
        self.log_dir = 'logs'
        self.kl_factor = 0.01
        self.use_airl_user = False
        self.use_airl_reward = False



class cfg_ppo():
    def __init__(self):
        self.render = False
        self.solved_reward = 300         # stop training if avg_reward > solved_reward
        self.log_interval = 20           # print avg reward in the interval
        self.max_episodes = 10000        # max training episodes
        self.max_timesteps = 1500        # max timesteps in one episode
        self.use_gpu = False
        
        self.update_timestep = 4000      # update policy every n timesteps
        self.action_std = 0.5            # constant std for action distribution (Multivariate Normal)
        self.K_epochs = 80               # update policy for K epochs
        self.eps_clip = 0.2              # clip parameter for PPO
        self.gamma = 0.99                # discount factor
        
        self.lr = 0.0003                 # parameters for Adam optimizer
        self.betas = (0.9, 0.999)
