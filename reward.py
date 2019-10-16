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
    def __init__(self, args, manager, config, irl_model, pretrain=False, inference=False):
        
        self.irl = irl_model
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.step = 0
        self.anneal = args.anneal
        self.irl_params = self.irl.parameters()
        self.irl_optim = self.irl.get_optimizer()
        self.weight_cliping_limit = args.clip
        
        self.save_dir = args.save_dir
        self.save_per_epoch = args.save_per_epoch
        self.optim_batchsz = args.batchsz
        self.irl.eval()
        
        db = None
        if pretrain:
            self.print_per_batch = args.print_per_batch
            self.data_train = manager.create_dataset_irl('train', args.batchsz, config, db)
            self.data_valid = manager.create_dataset_irl('valid', args.batchsz, config, db)
            self.data_test = manager.create_dataset_irl('test', args.batchsz, config, db)
            self.irl_iter = iter(self.data_train)
            self.irl_iter_valid = iter(self.data_valid)
            self.irl_iter_test = iter(self.data_test)
        elif not inference:
            self.data_train = manager.create_dataset_irl('train', args.batchsz, config, db)
            self.data_valid = manager.create_dataset_irl('valid', args.batchsz, config, db)
            self.irl_iter = iter(self.data_train)
            self.irl_iter_valid = iter(self.data_valid)
        
    def kl_divergence(self, mu, logvar, istrain):
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        beta = min(self.step/self.anneal, 1) if istrain else 1
        return beta*klds
    
    def irl_loop(self, data_real, data_gen):
        s_real, a_real, next_s_real = data_real
        s, a, next_s = data_gen
            
        # train with real data
        weight_real = self.irl(s_real, a_real, next_s_real)
        loss_real = -weight_real.mean()

        # train with generated data
        weight = self.irl(s, a, next_s)
        loss_gen = weight.mean()
        return loss_real, loss_gen
    
    def train_irl(self, batch, epoch):
        self.irl.train()
        input_s = torch.from_numpy(np.stack(batch.state)).to(device=device)
        input_a = torch.from_numpy(np.stack(batch.action)).to(device=device)
        input_next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=device)
        batchsz = input_s.size(0)
        
        real_loss, gen_loss = 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
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
        logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}'.format(
                epoch, real_loss, gen_loss))
        if (epoch+1) % self.save_per_epoch == 0:
            self.save_irl(self.save_dir, epoch)
        self.irl.eval()
    
    def test_irl(self, batch, epoch, best):
        input_s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        input_a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        input_next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        batchsz = input_s.size(0)
        
        real_loss, gen_loss = 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
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
        if loss < best:
            logging.info('<<reward estimator>> best model saved')
            best = loss
            self.save_irl(self.save_dir, 'best')
            
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
    
    def update_irl(self, inputs, batchsz, epoch, best=None):
        """
        train the reward estimator (together with encoder) using cross entropy loss (real, mixed, generated)
        Args:
            inputs: (s, a, next_s)
        """
        backward = True if best is None else False
        if backward:
            self.irl.train()
        input_s, input_a, input_next_s = inputs
        
        real_loss, gen_loss = 0., 0.
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)
        
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
            logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}'.format(
                    epoch, real_loss, gen_loss))
            self.irl.eval()
        else:
            logging.debug('<<reward estimator>> validation, epoch {}, loss_real:{}, loss_gen:{}'.format(
                    epoch, real_loss, gen_loss))
            loss = real_loss + gen_loss
            if loss < best:
                logging.info('<<reward estimator>> best model saved')
                best = loss
                self.save_irl(self.save_dir, 'best')
            return best
        
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
        weight = self.irl(s, a.float(), next_s)
        logging.debug('<<reward estimator>> weight {}'.format(weight.mean().item()))
        logging.debug('<<reward estimator>> log pi {}'.format(log_pi.mean().item()))
        # see AIRL paper
        # r = f(s, a, s') - log_p(a|s)
        reward = (weight - log_pi).squeeze(-1)
        return reward