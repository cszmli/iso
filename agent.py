import torch
import random
import numpy as np 
import copy 
from ppo import PPOEngine, ENV, PPO
from utils import cfg
from network import RewardModule, ActorCriticContinuous, ActorCriticDiscrete
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AIRL(object):
    def __init__(self, config=None, user_agent=None, user_reward=None, system_agent=None):
        # the parameters of user policy and user reward will not be loaded. The modules are just copies to keep the structure of 
        # the modules.
        # This is for the alternative training between the reward model and the PPO agent
        # TODO: the expert data should be loaded here
        self.config = config
        self.user_policy = user_agent
        self.reward_module = user_reward
        self.system_policy = system_agent
    
    def update(self):
        # rewrite the PPOEngine function and add IRL training step
        #  
        raise NotImplementedError
        
class InteractAgent(object):
    # this is for the whole trianing process: reward shaping, system optimization
    def __init__(self, config=None, user_agent=None, user_reward=None, system_agent=None):
        self.config = config
        self.user_agent = user_agent # this should be a PPO agent with discrete action space
        self.reward_module = user_reward  # this is an MLP taking as input the state representation
        self.system_agent = system_agent # this should be a PPO agent with continuous action space
        self.airl = AIRL(config=config, 
                         user_agent=copy.deepcopy(self.user_agent), 
                         user_reward=self.reward_module,
                         system_agent=self.system_agent
                         )
        self.env1 = ENV(system_policy=self.system_agent, stopping_judger=None)
        # TODO: env 2 is for the second MDP
        self.env2 = None

    
    def irl_train(self, system_policy, expert_data):
        #TODO: feed system_policy and expert_data to AIRL to get the new user_policy and reward_function.
        raise NotImplementedError("not finsihed yet")

    def system_train(self, ):
        #TODO: update the system_policy (PPO) in the second MDP  
        env2 = None
        ppo = PPOEngine(ppo=self.system_policy, env=env2)
        raise NotImplementedError("not finsihed yet")


def main(config):
    irl_model = RewardModule(config).to(device=device)   # this is the reward model only, which will be fed to RewardEstimator.
    user_policy = ActorCriticDiscrete(config).to(device=device)
    system_policy = ActorCriticContinuous(config).to(device=device)
    user_ppo = PPO(config, user_policy)
    system_ppo = PPO(config, system_policy)





if __name__ == "__main__":
    config = cfg()
    main(config)
