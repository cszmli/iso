# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import os
from collections import defaultdict
from network import RewardModule
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RewardEstimator(object):
    def __init__(self, args=None, manager=None, config=None, irl_model=None, pretrain=False, inference=False):
        
        self.irl = irl_model
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.step = 0
        self.anneal = config.anneal
        self.irl_params = self.irl.parameters()
        self.irl_optim = self.irl.get_optimizer()
        self.weight_cliping_limit = config.clamp
        self.config = config
        self.optim_batchsz = config.batch_size
        self.irl.eval()
        self.irl_iter, self.irl_iter_test, self.irl_iter_valid = None, None, None  # these three data pools will be loaded later.

    def embedding_act(self, labels):
        num_classes = self.config.action_dim
        if type(labels)==list:
            labels = torch.LongTensor(labels)
        y = torch.eye(num_classes) 
        return y[labels]

    def kl_divergence(self, mu, logvar, istrain):
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        beta = min(self.step/self.anneal, 1) if istrain else 1
        return beta*klds
    
    def irl_loop(self, data_real, data_gen):
        s_real, a_real, next_s_real = data_real
        s, a, next_s = data_gen
            
        a = self.embedding_act(a)
        a_real = self.embedding_act(a_real)

        weight_real = self.irl(s_real.detach(), a_real.float().detach(), next_s_real.detach())
        loss_real = -weight_real.mean()

        # train with generated data
        weight = self.irl(s, a, next_s)
        loss_gen = weight.mean()
        return loss_real, loss_gen
    
    def train_irl(self, batch, epoch):
        self.irl.train()
        input_s = torch.stack(batch.states).detach().to(device=device)
        input_a = torch.stack(batch.actions).detach().to(device=device)
        # input_a = self.embedding_act(input_a)
        input_next_s = torch.stack(batch.states_next).detach().to(device=device)
        batchsz = input_s.size(0)
        
        real_loss, gen_loss = 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a, turns)
        next_s_chunk = torch.chunk(input_next_s, turns)
        
        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            try:
                data = self.irl_iter.next()
            except StopIteration:
                self.irl_iter = iter(self.data_train)
                data = self.irl_iter.next()
            
            self.irl_optim.zero_grad()
            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            loss = loss_real + loss_gen
            loss.backward()
            self.irl_optim.step()
            
            for p in self.irl_params:
                p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
            
        real_loss /= turns
        gen_loss /= turns
        if epoch%2==0:
            logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}'.format(
                epoch, real_loss, gen_loss))
        self.irl.eval()
    
    def test_irl(self, batch, epoch, best):
        input_s = torch.stack(batch.states).detach().to(device=DEVICE)
        input_a = torch.stack(batch.actions).detach().to(device=DEVICE)
        # input_a = self.embedding_act(input_a)
        input_next_s = torch.stack(batch.states_next).detach().to(device=DEVICE)
        batchsz = input_s.size(0)
        
        real_loss, gen_loss = 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a, turns)
        next_s_chunk = torch.chunk(input_next_s, turns)
        
        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            try:
                data = self.irl_iter_valid.next()
            except StopIteration:
                self.irl_iter_valid = iter(self.data_valid)
                data = self.irl_iter_valid.next()
            
            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            
        real_loss /= turns
        gen_loss /= turns
        logging.debug('<<reward estimator>> validation, epoch {}, loss_real:{}, loss_gen:{}'.format(
                epoch, real_loss, gen_loss))
        loss = real_loss + gen_loss

            
        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            try:
                data = self.irl_iter_test.next()
            except StopIteration:
                self.irl_iter_test = iter(self.data_test)
                data = self.irl_iter_test.next()
            
            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            
        real_loss /= turns
        gen_loss /= turns
        logging.debug('<<reward estimator>> test, epoch {}, loss_real:{}, loss_gen:{}'.format(
                epoch, real_loss, gen_loss))
        return best
    
    def update_irl(self, inputs, batchsz, epoch, backward):
        """
        train the reward estimator (together with encoder) using cross entropy loss (real, mixed, generated)
        Args:
            inputs: (s, a, next_s)
        """
        # backward = True if best is None else False
        if backward:
            self.irl.train()
        input_s, input_a, input_next_s = torch.stack(inputs.states), torch.stack(inputs.actions), torch.stack(inputs.states_next)
        # input_a = self.embedding_act(input_a)
        
        real_loss, gen_loss = 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s.detach(), turns)
        a_chunk = torch.chunk(input_a.detach(), turns)
        next_s_chunk = torch.chunk(input_next_s.detach(), turns)
        
        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            if backward:
                try:
                    data = self.irl_iter.next()
                except StopIteration:
                    self.irl_iter = iter(self.data_train)
                    data = self.irl_iter.next()
            else:
                try:
                    data = self.irl_iter_valid.next()
                except StopIteration:
                    self.irl_iter_valid = iter(self.data_valid)
                    data = self.irl_iter_valid.next()
            
            if backward:
                self.irl_optim.zero_grad()
            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            if backward:
                loss = loss_real + loss_gen
                loss.backward()
                self.irl_optim.step()
                
                for p in self.irl_params:
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
        
        real_loss /= turns
        gen_loss /= turns
        if backward:
            if epoch%10==0:
                logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}'.format(
                        epoch, real_loss, gen_loss))
            self.irl.eval()
        else:
            if epoch%10==0:
                logging.debug('<<reward estimator>> validation, epoch {}, loss_real:{}, loss_gen:{}'.format(
                        epoch, real_loss, gen_loss))
            loss = real_loss + gen_loss

            # return best
        
    def save_irl(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.irl.state_dict(), directory + '/' + str(epoch) + '_estimator.mdl')
        logging.info('<<reward estimator>> epoch {}: saved network to mdl'.format(epoch))
        
    def load_irl(self, filename):
        irl_mdl = filename + '_estimator.mdl'
        if os.path.exists(irl_mdl):
            self.irl.load_state_dict(torch.load(irl_mdl))
            logging.info('<<reward estimator>> loaded checkpoint from file: {}'.format(irl_mdl))
    
    def estimate(self, s, a, next_s, log_pi):
        """
        infer the reward of state action pair with the estimator
        """
        # logging.info("{}, {}, {}".format(s.shape, a.shape, log_pi.shape))
        weight = self.irl(s, a.float(), next_s)
        # logging.debug('<<reward estimator>> weight {}'.format(weight.mean().item()))
        # logging.debug('<<reward estimator>> log pi {}'.format(log_pi.mean().item()))
        # see AIRL paper
        # r = f(s, a, s') - log_p(a|s)
        # logging.info(weight.shape)
        reward = (weight - log_pi).view(-1)
        return reward
    def forward(self, s, a, next_s, log_pi):  # remove this func later
        return self.estimate(s, a, next_s, log_pi)