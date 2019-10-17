import torch
import random
import numpy as np 
import copy 
from ppo import PPOEngine, ENV, PPO, Memory
from utils import cfg
from network import RewardModule, ActorCriticContinuous, ActorCriticDiscrete
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from reward import RewardEstimator

class AIRL(object):
    def __init__(self, config=None, user_agent=None, user_reward=None, system_env=None, manager=None):
        # the parameters of user policy and user reward will not be loaded. The modules are just copies to keep the structure of 
        # the modules.
        # This is for the alternative training between the reward model and the PPO agent
        # TODO: the expert data should be loaded here
        # TODO: there should be a function to create the expert data with initial system and the ground truth reward
        self.config = config
        self.user_agent = user_agent
        self.reward_module = user_reward
        self.env = system_env
        self.manager = manager
        self.expert_data_train = self.creat_expert_data()


    def creat_expert_data(self):
        #raise NotImplementedError("TODO: creat the expert data for the reward function learning")
        self.data_train = self.manager.create_dataset_irl('train', self.config)
        self.data_valid = self.manager.create_dataset_irl('valid', self.config)
        self.data_test = self.manager.create_dataset_irl('test', self.config)
        self.irl_iter = iter(self.data_train)
        self.irl_iter_valid = iter(self.data_valid)
        self.irl_iter_test = iter(self.data_test)
        return self.irl_iter, self.data_train

    def train(self):
        # rewrite the PPOEngine function and add IRL training step
        render = False
        solved_reward = 300         # stop training if avg_reward > solved_reward
        log_interval = 20           # print avg reward in the interval
        max_episodes = 10000        # max training episodes
        max_timesteps = 1500        # max timesteps in one episode
        
        update_timestep = 4000      # update policy every n timesteps
        update_timestep_reward = 4000
        action_std = 0.5            # constant std for action distribution (Multivariate Normal)
        K_epochs = 80               # update policy for K epochs
        eps_clip = 0.2              # clip parameter for PPO
        gamma = 0.99                # discount factor
        
        lr = 0.0003                 # parameters for Adam optimizer
        betas = (0.9, 0.999)
        
        
        memory = Memory()
        policy_buffer = Memory()
        env = self.env
        ppo = copy.deepcopy(self.user_agent)
        ############### Training ####################
        # logging variables
        running_reward = 0
        avg_length = 0
        time_step = 0
        
        # training loop
        for i_episode in range(1, max_episodes+1):
            state = env.reset()
            for t in range(max_timesteps):
                time_step +=1
                # Running policy_old:
                action = ppo.select_action(state, memory)
                state, reward, done, _ = env.step(state, action)
                reward = None
                # TODO: retrieve reward value by feeding state to RewardEstimator
                
                # Saving reward and is_terminals:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                # update if its time
                if time_step % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()

                if time_step % update_timestep_reward == 0:
                    self.reward_module.train_irl(policy_buffer, time_step)
                    policy_buffer.clear_memory()

                
                running_reward += reward
                if render:
                    env.render()
                if done:
                    break
            
            avg_length += t

        return ppo
        
class InteractAgent(object):
    # this is for the whole trianing process: reward shaping, system optimization
    def __init__(self, config=None, user_agent=None, user_reward=None, system_agent=None):
        self.config = config
        self.user_agent = user_agent # this should be a PPO agent with discrete action space
        self.reward_agent = user_reward  # this is an Reward Estimator taking as input the state representation
        self.system_agent = system_agent # this should be a PPO agent with continuous action space
        self.env1 = ENV(system_agent=self.system_agent, reward_agent=self.reward_agent, stopping_judger=None)
        self.airl = AIRL(config=config, 
                         user_agent=copy.deepcopy(self.user_agent), 
                         user_reward=self.reward_agent,
                         system_env=self.env1
                        )
        # TODO: env 2 is for the second MDP
        self.env2 = ENV(user_agent=self.user_agent, reward_agent=self.reward_agent, stopping_judger=None)

    
    def irl_train(self, system_policy, expert_data):
        #TODO: feed system_policy and expert_data to AIRL to get the new user_policy and reward_function.
        self.airl.train()
        # raise NotImplementedError("not finsihed yet")

    def system_train(self, ):
        #TODO: update the system_policy (PPO) in the second MDP  
        env2 = self.env2
        ppo = PPOEngine(ppo=self.system_agent, env=env2)
        # raise NotImplementedError("not finsihed yet")


def main(config):
    irl_model = RewardModule(config).to(device=device)   # this is the reward model only, which will be fed to RewardEstimator.
    reward_agent = RewardEstimator(config=config, irl_model=irl_model)
    user_policy = ActorCriticDiscrete(config).to(device=device)
    system_policy = ActorCriticContinuous(config).to(device=device)
    user_ppo = PPO(config, user_policy)
    system_ppo = PPO(config, system_policy)
    
    main_agent = InteractAgent(config=config,
                               user_agent=user_ppo,
                               user_reward=reward_agent,
                               system_agent=system_ppo)




if __name__ == "__main__":
    config = cfg()
    main(config)
