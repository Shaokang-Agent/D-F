import argparse
import pickle
from collections import namedtuple
from itertools import count
import os, time
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import numpy as np
import torch
from models.ppo_actor_critic import PPO_Actor, PPO_Critic


class PPO():
    def __init__(self, args, action_num):
        super(PPO, self).__init__()
        self.action_num = action_num
        self.args = args
        self.actor_net = PPO_Actor(self.args, self.action_num)
        self.old_actor_net = PPO_Actor(self.args, self.action_num)
        self.critic_net = PPO_Critic(self.args)
        self.actor_net.load_state_dict(self.old_actor_net.state_dict())
        self.optimizer = optim.Adam([{'params': self.actor_net.parameters(), 'lr': self.args.actor_lr},
                                     {'params': self.critic_net.parameters(), 'lr': self.args.critic_lr}])
        if self.args.cuda:
            self.actor_net.cuda()
            self.old_actor_net.cuda()
            self.critic_net.cuda()

    def choose_action(self, obs, episolon):
        obsnp = np.array(obs)
        obs = torch.from_numpy(obsnp.copy()).unsqueeze(dim=0).unsqueeze(dim=0)
        obs = obs.float()
        if self.args.cuda:
            obs = obs.cuda()
        with torch.no_grad():
            action_prob = self.old_actor_net(obs)
            dist = Categorical(action_prob)
            if np.random.rand() <= episolon:
                action = dist.sample()
                action_logprob = dist.log_prob(action)
            else:
                action = torch.randint(0, self.action_num, (1,))
                if self.args.cuda:
                    action = action.cuda()
                action_logprob = dist.log_prob(action)
            # action = dist.sample()
            # action_logprob = dist.log_prob(action)
            action = action.squeeze().cpu().data.numpy()
            action_logprob = action_logprob.squeeze().squeeze().cpu().data.numpy()
        return int(action), action_logprob

    def learn(self, episode_data, agent_id):
        batch_state = torch.from_numpy(np.array(episode_data['o'])).float()
        batch_action = torch.from_numpy(np.array(episode_data['u'])).long()
        batch_action_prob = torch.from_numpy(np.array(episode_data['u_probability'])).float()
        batch_reward = torch.from_numpy(np.array(episode_data['r'])).float().unsqueeze(dim=3)

        if self.args.cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_action_prob = batch_action_prob.cuda()
            batch_reward = batch_reward.cuda()

        value_loss = 0
        action_loss = 0
        batch_return = torch.zeros(batch_state.shape[0], batch_state.shape[1], 1)
        if self.args.cuda:
            batch_return = batch_return.cuda()
        for step in range(self.args.num_steps - 1, -1, -1):
            if step == self.args.num_steps - 1:
                batch_return[:, step, ...] = batch_reward[:, step, agent_id, ...]
            if step < self.args.num_steps - 1:
                batch_return[:, step, ...] = batch_reward[:, step, agent_id, ...] + self.args.gamma * batch_return[:, step + 1, ...]
        batch_return = (batch_return - batch_return.mean()) / (batch_return.std() + 1e-5)
        V = self.critic_net(batch_state[:, :, agent_id, ...])
        advantage = (batch_return - V).detach()
        for _ in range(self.args.training_times):
            action_probs = self.actor_net(batch_state[:, :, agent_id, ...]) # new policy
            dist = Categorical(action_probs)
            action_prob = dist.log_prob(batch_action[:, :, agent_id, ...].squeeze()).squeeze()
            entropy = dist.entropy()
            ratio = torch.exp(action_prob - batch_action_prob[:, :, agent_id, ...]).unsqueeze(dim=1)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.args.clip_param, 1 + self.args.clip_param) * advantage
            action_loss = -torch.min(surr1, surr2).mean()
            V = self.critic_net(batch_state[:, :, agent_id, ...])
            value_loss = F.mse_loss(batch_return, V)
            loss = action_loss + value_loss + 0.01 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.args.grad_clip)
            nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.args.grad_clip)
            self.optimizer.step()

        self.old_actor_net.load_state_dict(self.actor_net.state_dict())
        return value_loss, action_loss
