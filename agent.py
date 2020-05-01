import torch
import random
import numpy as np 
import copy 
import logging
from ppo import PPOEngine, ENV, PPO, Memory, Data_Generator
from utils import cfg
from network import RewardModule, RewardTruth, ActorCriticContinuous, ActorCriticDiscrete
from reward import RewardEstimator
from utils import init_logging_handler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AIRL(object):
    def __init__(self, config=None, user_agent=None, user_reward=None, system_env=None, manager=None):
        # the parameters of user policy and user reward will not be loaded. The modules are just copies to keep the structure of 
        # the modules.
        # This is for the alternative training between the reward model and the PPO agent
        # TODO: the expert data should be loaded here
        # TODO: there should be a function to create the expert data with initial system and the ground truth reward
        self.config = config
        self.user_agent = user_agent
        self.reward_agent = user_reward
        self.env = system_env
        self.manager = manager


    def load_expert_data(self, expert_data):
        self.reward_agent.irl_iter = iter(expert_data['train'])
        self.reward_agent.irl_iter_valid = iter(expert_data['valid'])
        self.reward_agent.irl_iter_test = iter(expert_data['test'])
    def clean_expert_data(self):
        del self.reward_agent.irl_iter
        del self.reward_agent.irl_iter_valid
        del self.reward_agent.irl_iter_test
        self.reward_agent.irl_iter, self.reward_agent.irl_iter_valid, self.reward_agent.irl_iter_test = None, None, None

    def train_warmup(self):
        raise NotImplementedError


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
                    self.reward_agent.train_irl(policy_buffer, time_step)
                    policy_buffer.clear_memory()

                
                running_reward += reward
                if render:
                    env.render()
                if done:
                    break
            
            avg_length += t

        return ppo, self.reward_agent
        
class InteractAgent(object):
    # this is for the whole trianing process: reward shaping, system optimization
    def __init__(self, config=None, user_agent=None, user_reward=None, system_agent=None, reward_groundtruth=None):
        self.config = config
        self.user_agent = user_agent # this should be a PPO agent with discrete action space
        self.reward_agent = user_reward  # this is an Reward Estimator taking as input the state representation
        self.system_agent = system_agent # this should be a PPO agent with continuous action space
        self.reward_oracle = reward_groundtruth
        self.env1 = ENV(system_agent=self.system_agent, reward_agent=self.reward_agent, stopping_judger=None, config=config)
        self.airl = AIRL(config=config, 
                         user_agent=copy.deepcopy(self.user_agent), 
                         user_reward=self.reward_agent,
                         system_env=self.env1
                        )
        # TODO: env 2 is for the second MDP
        self.env2 = ENV(user_agent=self.user_agent, reward_agent=self.reward_agent, stopping_judger=None, config=config)

    def ready_for_irl(self):
        # TODO: load reward_agent; load Env with system policy; load (generate) expert data
        raise NotImplementedError

    def ready_for_system_opt(self):
        # TODO: load reward_agent; load Env with user policy
        raise NotImplementedError

    def optimizer_user_policy(self, reward_truth=None, system_agent=None):
        # This function is used to optimize the user policy given ground truth reward function and a system policy (env)
        # reward_truth is not the reward agent and it is just a randomly initialized MLP which takes state as input and output reward value.
        env = ENV(system_agent=system_agent, reward_agent=reward_truth, stopping_judger=None, config=self.config)
        ppo = PPOEngine(ppo=self.user_agent, env=env, config=self.config)
        return ppo, env

    def generate_load_expert_data(self):
        logging.info("Training user policy with true reward and the initial system")
        ppo, env = self.optimizer_user_policy(reward_truth=self.reward_oracle, system_agent=self.system_agent)
        logging.info("Generating expert data and load them")
        train = Data_Generator(ppo, env, self.config, 'train')
        valid = Data_Generator(ppo, env, self.config, 'valid')
        test = Data_Generator(ppo, env, self.config, 'test')
        memory = {'train':train, 'valid':valid, 'test':test}
        self.airl.load_expert_data(memory)
        logging.info("Generating data and loading data finished")

        

    def irl_train(self, system_policy, expert_data):
        #TODO: feed system_policy and expert_data to AIRL to get the new user_policy and reward_function.
        user_agent, reward_module = self.airl.train()
        self.user_agent.policy.load_state_dict(user_agent.policy.state_dict())
        self.reward_agent.irl.load_state_dict(reward_module.irl.state_dict())

    def system_train(self, ):
        #TODO: update the system_policy (PPO) in the second MDP  
        user_ppo, env = self.optimizer_user_policy(reward_truth=self.reward_oracle, system_agent=self.system_agent)
        logging.info("user policy training stop here")
        env2 = ENV(user_agent=user_ppo, reward_agent=self.reward_oracle, reward_truth=self.reward_oracle, stopping_judger=None, config=config)
        ppo = PPOEngine(ppo=self.system_agent, env=env2, config=self.config)
        logging.info("system policy training stop here")
        user_ppo, env = self.optimizer_user_policy(reward_truth=self.reward_oracle, system_agent=ppo)
        
        self.system_agent = ppo



def main(config):
    init_logging_handler(config.log_dir)
    logging.info("Start initializing")
    irl_model = RewardModule(config).to(device=device)   # this is the reward model only, which will be fed to RewardEstimator.
    reward_agent = RewardEstimator(config=config, irl_model=irl_model)
    
    user_policy = ActorCriticDiscrete(config).to(device=device)
    user_ppo = PPO(config, user_policy)

    system_policy = ActorCriticContinuous(config).to(device=device)

    init_system_policy = ActorCriticContinuous(config).to(device=device)
    init_system_policy.load_state_dict(system_policy.state_dict())

    system_ppo = PPO(config, system_policy, init_policy=init_system_policy)

    reward_true = RewardTruth(config).to(device=device)  # this is the ground truth which will not be updated once randomly initialized.
    logging.info("Finish building module: reward agent, user ppo, system ppo")

    main_agent = InteractAgent(config=config,
                               user_agent=user_ppo,
                               user_reward=reward_agent,
                               system_agent=system_ppo,
                               reward_groundtruth=reward_true
                               )
    
    # main_agent.generate_load_expert_data()
    for _ in range(3):
        main_agent.system_train()
    raise ValueError("stop here")





if __name__ == "__main__":
    config = cfg()
    main(config)
