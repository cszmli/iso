import torch
import random
import numpy as np 
import copy 
import logging
from ppo import PPOEngine, ENV, PPO, Memory, Data_Generator, ExpertMemory
from utils import cfg, get_parser
from network import RewardModule, RewardTruth, ActorCriticContinuous, ActorCriticDiscrete, RewardTruthSampled
from reward import RewardEstimator
from utils import init_logging_handler
import torch.utils.data as data
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_device(data):
    if type(data) == dict:
        for k, v in data.items():
            data[k] = v.to(device=device)
    else:
        for idx, item in enumerate(data):
            data[idx] = item.to(device=device)
    return data


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
        self.chunk_size = config.batch_size
        self.gamma = config.gamma
        self.optim_batchsz = config.batch_size


    def load_expert_data(self, expert_data):
        self.reward_agent.data_train = data.DataLoader(expert_data['train'], self.chunk_size)
        self.reward_agent.data_valid = data.DataLoader(expert_data['valid'], self.chunk_size)
        self.reward_agent.data_test = data.DataLoader(expert_data['test'], self.chunk_size)
        self.reward_agent.irl_iter = iter(self.reward_agent.data_train)
        self.reward_agent.irl_iter_valid = iter(self.reward_agent.data_valid)
        self.reward_agent.irl_iter_test = iter(self.reward_agent.data_test)
    def clean_expert_data(self):
        del self.reward_agent.irl_iter
        del self.reward_agent.irl_iter_valid
        del self.reward_agent.irl_iter_test
        self.reward_agent.irl_iter, self.reward_agent.irl_iter_valid, self.reward_agent.irl_iter_test = None, None, None

    def init_net(self,net):
        for p in net.parameters():
            p.data.uniform_(-0.08, 0.08)
        return net

    def train_warmup(self, epochs):
        self.user_agent.policy = self.init_net(self.user_agent.policy)
        self.user_agent.policy_old.load_state_dict(self.user_agent.policy.state_dict())

        self.imitate_policy(epochs)
        for epoch_id in range(4 * epochs):
            sampled_batch = self.sample_run(4096)
            self.reward_agent.train_irl(sampled_batch, epoch_id)
            self.imitate_value(epoch_id, 4096)

    def v_target(self, memory):
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
        return rewards
    
    def imitate_value(self, epoch=0, batchsz=1024, sampled_batch=None):
        self.user_agent.policy.critic.train()
        if sampled_batch is None:
            batch = self.sample_run(batchsz)
        else:
            batch = sampled_batch
        s = torch.stack(batch.states).detach().to(device=device)
        a = torch.stack(batch.actions).detach().to(device=device)
        next_s = torch.stack(batch.states_next).detach().to(device=device)
        mask = torch.Tensor(batch.is_terminals).detach().to(device=device)
        batchsz = s.size(0)
        
        v_target = self.v_target(batch).detach()
        
        for i in range(10):
            perm = torch.randperm(batchsz)
            v_target_shuf, s_shuf = v_target[perm], s[perm]
            optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
            v_target_shuf, s_shuf = torch.chunk(v_target_shuf, optim_chunk_num), torch.chunk(s_shuf, optim_chunk_num)
            
            value_loss = 0.
            for v_target_b, s_b in zip(v_target_shuf, s_shuf):
                self.user_agent.optimizer_critic.zero_grad()
                v_b = self.user_agent.policy.critic(s_b).squeeze(-1)
                loss = (v_b - v_target_b).pow(2).mean()
                value_loss += loss.item()
                loss.backward()
                self.user_agent.optimizer_critic.step()
        
            value_loss /= optim_chunk_num
        if epoch%2==0:
            logging.debug('<<Value net>> epoch {}, iteration {}, loss {}'.format(epoch, i, value_loss))
        self.user_agent.policy.critic.eval()


    def imitate_loop(self, epoch):
        loss_log = 0
        batch_count =0
        train_flag = True
        
        while train_flag:
            try:
                self.user_agent.optimizer_actor.zero_grad()
                # self.user_agent.optimizer.zero_grad()
                data = self.reward_agent.irl_iter.next()
                # s, a, _ = to_device(data)
                s, a, _ = data
                # s = torch.from_numpy(np.stack(s)).to(device=device)
                # a = torch.from_numpy(np.stack(a)).to(device=device)

                loss = self.user_agent.policy.imitate(s.detach(), a.detach())
                loss_log += loss.item()
                loss.backward()
                self.user_agent.optimizer_actor.step()
                # self.user_agent.optimizer.step()
                batch_count += 1
                data = self.reward_agent.irl_iter.next()
            except StopIteration:  
                # logging.info('reload')
                self.reward_agent.irl_iter = iter(self.reward_agent.data_train)
                train_flag = False
        if epoch%10==0:
            logging.info("Epoch: {}, Avg batch loss: {}".format(epoch, loss_log/batch_count))

    def imitate_policy(self, epochs):
        self.user_agent.policy.train()
        for i in range(epochs):
            self.imitate_loop(epoch=i)
        self.user_agent.policy.eval()
        

    
    def sample_run(self, sample_size):
        memory = Memory()
        env = self.env
        ppo = self.user_agent
        ############### sampling ####################
        time_step = 0  # frames
        stop_collect = False
        reward_cul_true = 0
        traj_counter = 0
        # training loop
        while not stop_collect:
            state = env.reset()
            for t in range(100):
                # Running policy_old:
                action = ppo.select_action(state, memory)

                log_prob = memory.logprobs[-1]
                state_ori = state[:,:config.state_dim]
                state, action, next_state, reward_true, done = env.step(state=state,
                                                                action=action, 
                                                                log_prob=log_prob,
                                                                state_ori=state_ori)
                action_embed = env.embedding_act(action)
                reward = env.reward_agent.estimate(state, action_embed, next_state, log_prob).view(-1).detach()
                
                # Saving reward and is_terminals:
                memory.states_next.append(next_state)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                time_step += 1
                state = next_state
                if done:
                    traj_counter += 1
                    break
                reward_cul_true += reward_true.item()
            if time_step>=sample_size:
                stop_collect = True
        logging.info("Avg true reward in sample_run: {}".format(reward_cul_true/traj_counter))
        return memory


    def train(self, warmup_epochs, max_epoch):
        logging.info("******** Start AIRL WarmUp ********")
        self.train_warmup(warmup_epochs)
        logging.info("******** Start AIRL Training ********")
        # rewrite the PPOEngine function and add IRL training step
        log_interval = 10        # print avg reward in the interval
        max_timesteps = 150        # max timesteps in one episode
        
        update_timestep = 1000      # update policy every n timesteps
        update_reward_policy_ratio = 1
        update_reward_counter = 0
        update_policy_counter = 0

        action_std = 0.5            # constant std for action distribution (Multivariate Normal)
        eps_clip = 0.2              # clip parameter for PPO
        gamma = 0.96                # discount factor
        
        lr = 0.001                 # parameters for Adam optimizer
        betas = (0.9, 0.999)
        
        
        memory = Memory()
        policy_buffer = ExpertMemory()
        env = self.env
        # ppo = copy.deepcopy(self.user_agent)
        ppo = self.user_agent
        ############### Training ####################
        # logging variables
        running_reward = 0       # log the accumulated estimated reward 
        running_reward_true = 0  # log the accumulated true reward 
        avg_length = 0
        time_step = 0  # frames
        time_step_temp =0
        traj_counter = 0
        log_flag = True
        # training loop
        while update_policy_counter < max_epoch:
            state = env.reset()
            for t in range(max_timesteps):
                time_step +=1
                time_step_temp += 1
                # Running policy_old:
                action = ppo.select_action(state, memory)

                log_prob = memory.logprobs[-1]
                state_ori = state[:,:config.state_dim]
                state, action, next_state, reward_true, done = env.step(state=state,
                                                                action=action, 
                                                                log_prob=log_prob,
                                                                state_ori=state_ori)

                action_embed = env.embedding_act(action)
                reward = env.reward_agent.estimate(state, action_embed, next_state, log_prob).view(-1).detach()
                
                # Saving reward and is_terminals:
                memory.states_next.append(next_state)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                policy_buffer.states.append(memory.states[-1])
                policy_buffer.actions.append(memory.actions[-1])
                policy_buffer.states_next.append(memory.states_next[-1])
                    
                if time_step % update_timestep == 0:
                    if update_reward_counter==update_reward_policy_ratio:
                        ppo.update(memory)
                        update_reward_counter=0
                        update_policy_counter+=1
                        log_flag=True
                    memory.clear_memory()

                    self.reward_agent.update_irl(policy_buffer, update_timestep, update_policy_counter, True)
                    policy_buffer.clear_memory()
                    update_reward_counter += 1


                state = next_state
                running_reward += reward
                running_reward_true += reward_true
                if done:
                    traj_counter+=1
                    break

            avg_length += t

            if update_policy_counter % log_interval == 0 and log_flag:
                avg_length = avg_length/traj_counter
                running_reward = running_reward/traj_counter
                running_reward_true = running_reward_true/traj_counter
                
                logging.info('Epochs {}, Avg length: {}, Avg estimated reward: {}, Avg reward: {}'.format(update_policy_counter, avg_length, running_reward.item(), running_reward_true.item()))
                running_reward = 0
                running_reward_true = 0
                avg_length = 0
                log_flag = False
                traj_counter, time_step_temp =0, 0
        return ppo, self.reward_agent
        
class InteractAgent(object):
    # this is for the whole trianing process: reward shaping, system optimization
    def __init__(self, config=None, user_agent=None, user_reward=None, system_agent=None, reward_groundtruth=None):
        self.config = config
        self.user_agent_oracle = copy.deepcopy(user_agent)
        self.user_agent = user_agent # this should be a PPO agent with discrete action space
        self.reward_agent = user_reward  # this is an Reward Estimator taking as input the state representation
        self.system_agent = system_agent # this should be a PPO agent with continuous action space
        self.reward_oracle = reward_groundtruth
        self.env1 = ENV(system_agent=self.system_agent, reward_agent=self.reward_agent, reward_truth=reward_groundtruth, stopping_judger=None, config=config)
        self.airl = AIRL(config=config, 
                         user_agent=copy.deepcopy(self.user_agent), 
                         user_reward=self.reward_agent,
                         system_env=None
                        )
        # TODO: env 2 is for the second MDP
        self.env2 = ENV(user_agent=self.user_agent, reward_agent=self.reward_agent, stopping_judger=None, config=config)

    def ready_for_irl(self):
        # TODO: load reward_agent; load Env with system policy; load (generate) expert data
        raise NotImplementedError

    def ready_for_system_opt(self):
        # TODO: load reward_agent; load Env with user policy
        raise NotImplementedError

    def init_agent_policy(self, agent):
        for p in agent.policy.parameters():
            p.data.uniform_(-0.08, 0.08)
        agent.policy_old.load_state_dict(agent.policy.state_dict())
        return agent
    
    def optimizer_user_policy(self, reward_truth=None, system_agent=None):
        # This function is used to optimize the user policy given ground truth reward function and a system policy (env)
        # reward_truth is not the reward agent and it is just a randomly initialized MLP which takes state as input and output reward value.
        env = ENV(system_agent=system_agent, reward_truth=reward_truth, stopping_judger=None, config=self.config)
        self.user_agent = self.init_agent_policy(self.user_agent)
        ppo = PPOEngine(ppo=self.user_agent, env=env, config=self.config)
        return ppo, env

    def generate_load_expert_data(self, ppo, env):
        logging.info("Generating expert data and load them")
        train = Data_Generator(ppo, env, self.config, 'train')
        valid = Data_Generator(ppo, env, self.config, 'valid')
        test = Data_Generator(ppo, env, self.config, 'test')
        memory = {'train':train, 'valid':valid, 'test':test}
        self.airl.load_expert_data(memory)
        logging.info("Generating data and loading data finished")
        return ppo
        

    def irl_train(self):
        #TODO: feed system_policy and expert_data to AIRL to get the new user_policy and reward_function.
        warmup_epochs, max_epochs = 5, 100
        user_agent, reward_module = self.airl.train(warmup_epochs, max_epochs)
        logging.info('finished airl train')
        self.user_agent.policy.load_state_dict(user_agent.policy.state_dict())
        self.reward_agent.irl.load_state_dict(reward_module.irl.state_dict())
        logging.info("load learned user policy and reward module to agents")
        logging.info("user policy training stop here")
        return user_agent, reward_module

    def system_train(self, ):
        #TODO: update the system_policy (PPO) in the second MDP  
        user_ppo, env = self.optimizer_user_policy(reward_truth=self.reward_oracle, system_agent=self.system_agent)
        logging.info("user policy training stop here")
        env2 = ENV(user_agent=user_ppo, reward_truth=self.reward_oracle, stopping_judger=None, config=self.config)
        ppo = PPOEngine(ppo=self.system_agent, env=env2, config=self.config)
        logging.info("system policy training stop here")
        user_ppo, env = self.optimizer_user_policy(reward_truth=self.reward_oracle, system_agent=ppo)
        
        self.system_agent = ppo

    def keep_oracle_user(self, ppo):
        self.user_agent_oracle.policy.load_state_dict(ppo.policy.state_dict())
        self.user_agent_oracle.policy_old.load_state_dict(ppo.policy_old.state_dict())
    
    def master_train(self, steps):
        # 1. generate data
        # 2. airl train to get user_policy and user_reward
        # 3. system train to get update system_agent
        # 4. load updated system_agent to the env and retrain the user policy with gold reward
        logging.info("\n******** ************************ ********")
        logging.info("******** Master Train Epoch: {} ********".format(steps))
        logging.info("Training user policy with true reward and the initial system")
        ppo, env = self.optimizer_user_policy(reward_truth=self.reward_oracle, system_agent=self.system_agent)
        self.keep_oracle_user(ppo)
        logging.info("\n******** Generating Expert Data ********")
        user_ppo = self.generate_load_expert_data(ppo, env)  # ppo_oracle is the agent trained with true reward
        logging.info("\n******** Launch AIRL Module ********")
        self.airl.env = ENV(system_agent=self.system_agent, reward_agent=self.reward_agent, reward_truth=self.reward_oracle, stopping_judger=None, config=self.config)
        user_agent, reward_module = self.irl_train()    # the trained user agent and reward function by running AIRL; replaced already
        
        logging.info("\n******** Launch MDP-2 ********")
        if self.config.use_airl_reward and not self.config.use_airl_user:
            env2 = ENV(user_agent=self.user_agent_oracle, reward_agent=reward_module, reward_truth=self.reward_oracle, stopping_judger=None, config=self.config)
        elif not self.config.use_airl_reward and not self.config.use_airl_user:
            env2 = ENV(user_agent=self.user_agent_oracle, reward_truth=self.reward_oracle, stopping_judger=None, config=self.config)
        elif self.config.use_airl_reward and self.config.use_airl_user:
            env2 = ENV(user_agent=user_agent, reward_agent=reward_module, reward_truth=self.reward_oracle, stopping_judger=None, config=self.config)
        else:
            raise ValueError("NO such ENV setup for second MDP")
        logging.info("\n******** Launch System Updating Module ********")
        sys_ppo = PPOEngine(ppo=self.system_agent, env=env2, config=self.config)
        self.system_agent = sys_ppo
        logging.info("system policy training stop here")
        logging.info("\n******** one turn of master_train ends here ********")
        logging.info("******** ************************ ********\n")


        # logging.info("\n******** Updating User Policy with True Reward and New System ********")
        # user_ppo, env = self.optimizer_user_policy(reward_truth=self.reward_oracle, system_agent=sys_ppo)
        
def init_net(net):
    # for p in net.parameters():
        # p.data.uniform_(-0.08, 0.08)
    return net

def update_cfg(cfg, args):
    cfg.master_epochs = args.master_epochs
    cfg.state_dim = args.state_dim
    cfg.action_dim = args.action_dim
    cfg.kl_factor = args.kl_factor
    cfg.max_episodes = args.max_episodes
    cfg.use_airl_user = args.use_airl_user
    cfg.use_airl_reward = args.use_airl_reward
    return cfg

def main(config):
    parser = get_parser()
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)

    init_logging_handler(config.log_dir)
    logging.info(args)
    config = update_cfg(config, args)

    logging.info("Start initializing")
    irl_model = RewardModule(config).to(device=device)   # this is the reward model only, which will be fed to RewardEstimator.
    reward_agent = RewardEstimator(config=config, irl_model=irl_model)
    
    user_policy = ActorCriticDiscrete(config).to(device=device)
    user_policy = init_net(user_policy)
    user_ppo = PPO(config, user_policy)

    system_policy = ActorCriticContinuous(config).to(device=device)
    system_policy = init_net(system_policy)

    init_system_policy = ActorCriticContinuous(config).to(device=device)
    init_system_policy.load_state_dict(system_policy.state_dict())

    system_ppo = PPO(config, system_policy, init_policy=init_system_policy)

    # reward_true = RewardTruth(config).to(device=device)  # this is the ground truth which will not be updated once randomly initialized.
    reward_true = RewardTruthSampled(config).to(device)
    reward_true = init_net(reward_true)
    logging.info("Finish building module: reward agent, user ppo, system ppo")

    main_agent = InteractAgent(config=config,
                               user_agent=user_ppo,
                               user_reward=reward_agent,
                               system_agent=system_ppo,
                               reward_groundtruth=reward_true
                               )
    
    for e_id in range(config.master_epochs):
        main_agent.master_train(e_id)
    # for _ in range(3):
        # main_agent.system_train()
    # raise ValueError("stop here")
    logging.info("@@@@@@@@@@  Finished  @@@@@@@@@@@")





if __name__ == "__main__":
    config = cfg()
    main(config)
