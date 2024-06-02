import torch
import os
from models.ma_actor_critic import Actor, Critic
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

class MADDPG:
    def __init__(self, args, action_num):
        self.args = args
        self.action_num = action_num
        self.train_step = 0
        # create the network
        self.actor_network = Actor(self.args, self.action_num)
        self.critic_network = Critic(self.args, self.action_num)

        # build up the target network
        self.actor_target_network = Actor(self.args, self.action_num)
        self.critic_target_network = Critic(self.args, self.action_num)

        if self.args.cuda:
            self.actor_network.cuda()
            self.actor_target_network.cuda()
            self.critic_network.cuda()
            self.critic_target_network.cuda()

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.optimization = torch.optim.Adam(
            [{"params": self.actor_network.parameters(), "lr": self.args.actor_lr},
             {"params": self.critic_network.parameters(), "lr": self.args.critic_lr}])

        self.loss_func = torch.nn.MSELoss()
    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def init_hidden(self):
        self.h0, self.c0 = torch.randn(1, 1, 32),torch.randn(1, 1, 32)

    def choose_action(self, obs, episolon):
        if np.random.rand() <= episolon:
            obsnp = np.array(obs)
            obs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(obsnp.copy()), 0), 0)
            obs = obs.float()
            if self.args.cuda:
                obs = obs.cuda()
            with torch.no_grad():
                if self.args.cuda:
                    self.h0 = self.h0.cuda()
                    self.c0 = self.c0.cuda()
                action_prob, self.h0, self.c0 = self.actor_network(obs, self.h0, self.c0)
                dist = Categorical(action_prob)
                action = dist.sample().squeeze().cpu().data.numpy()
        else:
            action = np.random.randint(0, self.action_num)
        return int(action)

    # update the network
    def learn(self, episode_data, agent_id, agent_maddpg):
        batch_state = torch.from_numpy(episode_data['s']).float()
        batch_observation = torch.from_numpy(episode_data['o']).float()
        batch_action = torch.from_numpy(episode_data['u']).long()
        batch_reward = torch.from_numpy(episode_data['r']).float()
        batch_next_state = torch.from_numpy(episode_data['s_next']).float()
        batch_next_observation = torch.from_numpy(episode_data['o_next']).float()

        if self.args.cuda:
            batch_state = batch_state.cuda()
            batch_observation = batch_observation.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_observation = batch_next_observation.cuda()
            batch_next_state = batch_next_state.cuda()

        a_next = []
        with torch.no_grad():
            for i in range(self.args.num_agents):
                if i == agent_id:
                    a_prob = self.actor_target_network(batch_next_observation[:,:,i,...])
                else:
                    a_prob = agent_maddpg[i].actor_target_network(batch_next_observation[:, :, i, ...])
                dist = Categorical(a_prob)
                a_next_ind = dist.sample()
                a_next.append(a_next_ind)
            a_next = torch.stack(a_next, dim=2).unsqueeze(dim=3)
            if self.args.cuda:
                a_next = a_next.cuda()
        q_next = self.critic_target_network(batch_next_state, a_next).detach()
        q_value = self.critic_network(batch_state, batch_action)
        q_target = (batch_reward[:,:,agent_id,...] + self.args.gamma * q_next).detach()
        critic_loss = self.loss_func(q_value, q_target)

        a_prob = self.actor_network(batch_observation[:, :, agent_id, ...])
        dist = Categorical(a_prob)
        a_i = dist.sample().unsqueeze(dim=2)
        batch_action[:, :, agent_id, ...] = a_i
        actor_loss = -self.critic_network(batch_state, batch_action).mean()

        self.optimization.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()

        self.optimization.step()

        if self.train_step % self.args.replace_param == 0:
            self._soft_update_target_network()
        self.train_step += 1

        return critic_loss, actor_loss
