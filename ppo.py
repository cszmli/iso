import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from network import ActorCriticContinuous, ActorCriticDiscrete
import copy
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.states_next = []


    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.states_next[:]



class ExpertMemory(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.states_next = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.states_next[:]



class PPO(object):
    # def __init__(self, config, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
    def __init__(self, config, policy_):
        self.lr = config.lr
        self.betas = config.betas
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.K_epochs = config.K_epochs
        
        self.policy = policy_
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        
        self.policy_old = policy_
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory)
    
    def infer_action(self, state):
        return self.policy(state)
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

def Data_Generator(ppo, env, config, data_id):
     ############## Hyperparameters ##############

    data_size = config.data_size[data_id]        # expert data size
    max_timesteps = config.max_step        # max timesteps in one episode

    memory = []
    ############### Training ####################
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    while time_step<data_size:
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.infer_action(state)
            state, action, next_state, reward, done = env.step(state=state, action=action)
            state = next_state
            # memory.states.append(state)
            # memory.actions.append(action)
            # memory.states_next.append(next_state)
            # memory.rewards.append(reward)
            # memory.is_terminals.append(done)
            memory.append((state, action, next_state))
            
            if done:
                break
    return memory

            


def PPOEngine(ppo, env, config):
    ############## Hyperparameters ##############
    log_interval = 100           # print avg reward in the interval
    max_episodes = 100000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    
    update_timestep = 4000      # update policy every n timesteps
    
    memory = Memory()
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
            log_prob = memory.logprobs[-1]
            state_ori = state[:,:config.state_dim]
            state, action, next_state, reward, done = env.step(state=state,
                                                               action=action, 
                                                               log_prob=log_prob,
                                                               state_ori=state_ori)
            # Saving reward and is_terminals:
            # print("###################")
            # print(state, action, next_state, reward)
            memory.states_next.append(next_state)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            state = next_state
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if done:
                break
        
        avg_length += t
        # logging
        if i_episode % log_interval == 0:
            avg_length = avg_length/log_interval
            running_reward = running_reward/log_interval
            
            logging.info('Episode {}, Avg length: {},  Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
    return ppo
            

class ENV(object):
    def __init__(self, user_agent=None, system_agent=None, reward_agent=None, reward_truth=None, stopping_judger=None, config=None):
        self.reward_agent = reward_agent
        self.reward_truth = reward_truth
        self.stopping_judger = stopping_judger
        self.config = config
        self.max_step = config.max_step
        self.counter = 0
        self.env_type = None
        if user_agent is not None and system_agent is not None:
            raise ValueError("user_agent and system agent can not be both None or NotNone")
        if user_agent is not None:
            self.policy = user_agent.policy
            if reward_truth is not None:
                self.env_type = 'user_step_real_r'
            else:
                self.env_type = 'user_step'
        if system_agent is not None:
            self.policy = system_agent.policy
            self.env_type = 'system_step'


    def reset(self):
        self.counter = 0
        state_mean = torch.FloatTensor(1,self.config.state_dim).uniform_(-1, 1).to(device)
        if self.env_type == 'user_step' or self.env_type == 'user_step_real_r':
            action_init = self.policy(state_mean)
            action_init = self.embedding_act(action_init)
            state_mean = torch.cat([state_mean, action_init], -1)
        return state_mean

    def load_user(self, new_policy):
        self.policy.load_state_dict(new_policy.state_dict())

    def step(self, state=None, action=None, log_prob=None, state_ori=None):
        # print("step")
        if self.env_type == 'user_step' or self.env_type == 'user_step_real_r':
            return self.step_user(state=state, action=action, state_ori=state_ori)
        elif self.env_type == 'system_step':
            return self.step_system(state=state, action=action, log_prob=log_prob)
        else:
            raise ValueError("No such env_type: {}".format(self.env_type))

    def step_system(self, state=None, action=None, log_prob=None):
        # TODO: feed state+action to self.system_policy and get state_next
        # TODO: feed state + action + state_next to a judger and get if the tuple is linkable: 
        # TODO: if yes, return state_next, reward(state), Terminal
        # TODO: if not, return state_false, reward, Terminal
        # Q: how to decide if two states are linkable and how to select a linkable state from the state space?
        # This is just a simplified version without state transition constraints.

        # state = torch.from_numpy(np.stack(state)).to(device=device)        
        # action = torch.from_numpy(np.stack(action)).to(device=device)     
        action_embed = self.embedding_act(action)        
        state_action = torch.cat([state, action_embed], -1)
        if self.counter<self.max_step:
            next_state = self.policy(state_action)
            is_terminal = False
        else:
            next_state = state
            is_terminal = True
        self.counter += 1
        reward = self.reward_agent(state, action, next_state, log_prob)  # this is for the first MDP, with log_prob
        # reward = self.reward_tryth(state)
        return (state, action, next_state, reward, is_terminal)

    def step_user(self, state=None, action=None, state_ori=None):
        # TODO: feed (state, action, state_next) to self.user_policy and get state_next' (state_next'=(state', action'))
        # TODO: feed state + action + state_next to a judger and get if the tuple is linkable: 
        # TODO: if yes, return state_next, reward(state), Terminal
        # TODO: if not, return state_false, reward, Terminal
        # Q: how to decide if two states are linkable and how to select a linkable state from the state space?
        
        # This is just a simplified version without state transition constraints.
        # state=<s, a>, action=s', state_ori = <s>
        if self.counter<self.max_step:
            next_state_a = self.policy(action)
            # TODO: convert a to onehot_embedding with dim=action_num
            next_state_a = self.embedding_act(next_state_a)
            next_state = torch.cat([action, next_state_a], -1)
            is_terminal = False
        else:
            next_state = state
            is_terminal = True
        self.counter += 1
        if self.env_type=='user_step':
            reward = self.reward_agent.irl(s=state_ori, a=action, next_s=action)    # this is for the second MDP, without log_prob
        else:
            reward = self.reward_truth(state_ori)
        return state, action, next_state, reward, is_terminal

    def embedding_act(self, labels):
        num_classes = self.config.action_dim
        if type(labels)==list:
            labels = torch.LongTensor(labels)
        y = torch.eye(num_classes) 
        return y[labels]




    
