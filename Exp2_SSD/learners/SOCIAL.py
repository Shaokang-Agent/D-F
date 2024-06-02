import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import gym
import matplotlib.pyplot as plt
import copy

class QNet(nn.Module):
    """docstring for Net"""
    def __init__(self, action_num, args):
        super(QNet, self).__init__()
        self.args = args
        self.cnn = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(169 * 6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32, action_num)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        Batch, seq_len = x.shape[0], x.shape[1]
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = F.leaky_relu(self.cnn(x))
        x = x.reshape(Batch, seq_len, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.out(x)
        return x

class PNet(nn.Module):
    """docstring for Net"""
    def __init__(self, action_num, args):
        super(PNet, self).__init__()
        self.args = args
        self.action_num = action_num
        self.cnn = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(169*6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32 + self.action_num, (self.args.num_agents - 1)*self.action_num)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x, a):
        Batch, seq_len = x.shape[0], x.shape[1]
        x = x.permute(0,1,4,2,3)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = F.leaky_relu(self.cnn(x))
        x = x.reshape(Batch,seq_len, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        a_onehot = F.one_hot(a.unsqueeze(dim=2), self.action_num).squeeze(dim=2).squeeze(dim=2)
        x = torch.cat((x, a_onehot), dim=2)
        x = F.gumbel_softmax(self.out(x), dim=2)
        return x


class DQN_SOCIAL():
    """docstring for DQN"""
    def __init__(self, args, action_num):
        super(DQN_SOCIAL, self).__init__()
        self.eval_net, self.target_net = QNet(action_num, args).float(), QNet(action_num, args).float()
        self.PNet = PNet(action_num, args).float()
        if args.cuda:
            self.eval_net.cuda()
            self.target_net.cuda()
            self.PNet.cuda()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.args = args
        self.action_num = action_num
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.optimizer_list = list(self.eval_net.parameters()) + list(self.PNet.parameters())
        self.loss_func1 = nn.MSELoss()
        self.loss_func2 = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.optimizer_list, lr=self.args.lr)

    def choose_action(self, state, episolon):
        obsnp = np.array(state)
        obs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(obsnp.copy()), 0),0)
        obs = obs.float()
        if self.args.cuda:
            obs = obs.cuda()
        with torch.no_grad():
            action_value = self.eval_net(obs)
            action_prob = F.gumbel_softmax(action_value, dim=2).squeeze()
            if np.random.rand() <= episolon:
                action = torch.max(action_value, 2)[1].cpu().data.numpy()[0][0]
            else:
                action = np.random.randint(0, self.action_num)
        return action, action_prob.cpu().numpy()

    def KL_divergence(self, p1, p2):
        kl = torch.zeros((self.args.batch_size, self.args.num_steps, 1))
        if self.args.cuda:
            kl = kl.cuda()
        for i in range(p1.shape[2]):
            kl[:,:,0] += p1[:,:,i]*(torch.log(p1[:,:,i]) - torch.log(p2[:,:,i]))
        return kl

    def learn(self, episode_data, agent_id):
        if self.learn_step_counter % self.args.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        batch_state = torch.from_numpy(episode_data['o']).float()
        batch_action = torch.from_numpy(episode_data['u']).long()
        batch_action_prob = torch.from_numpy(episode_data['u_probability_all']).float()
        batch_reward = torch.from_numpy(episode_data['r']).float()
        batch_next_state = torch.from_numpy(episode_data['o_next']).float()
        batch_next_action = torch.from_numpy(episode_data['u_next']).long()
        batch_reward_in = torch.Tensor(np.zeros((self.args.batch_size, self.args.num_steps, 1))).long()

        if self.args.cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_action_prob = batch_action_prob.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()
            batch_next_action = batch_next_action.cuda()
            batch_reward_in = batch_reward_in.cuda()

        state = batch_state[:,:,agent_id,...]
        action = batch_action[:,:,agent_id,...]
        reward = batch_reward[:,:,agent_id,...]
        next_state = batch_next_state[:,:,agent_id,...]
        other_next_action = torch.cat((batch_next_action[:,:,:agent_id,...], batch_next_action[:,:,agent_id+1:,...]), dim=2)
        other_action_prob = torch.cat((batch_action_prob[:,:,:agent_id,...], batch_action_prob[:,:,agent_id+1:,...]), dim=2)

        q_eval = self.eval_net(state).gather(2, action)
        q_next = self.target_net(next_state).detach()

        for j in range(self.args.num_agents - 1):
            p_other = other_action_prob[:,:,j,...]
            p_condition_other = self.PNet(state, action)
            p_other_pred = p_condition_other[:, :, j*self.action_num:j*self.action_num+self.action_num]
            r_in = self.KL_divergence(p_other_pred, p_other)
            batch_reward_in = batch_reward_in + r_in

        batch_reward_in = torch.clamp(batch_reward_in, 0, 10)
        r_in_scale = self.args.r_in_scale
        ENV_ALPHA = np.min([self.args.env_alpha_final, self.args.env_alpha_initial + (self.learn_step_counter / self.args.env_alpha_steplen) * (
                                        self.args.env_alpha_final - self.args.env_alpha_initial)])

        reward = reward + (1-ENV_ALPHA) * r_in_scale * batch_reward_in
        reward = (reward - reward.mean() ) / (reward.std() + 1e-4)
        if self.args.double_dqn:
            q_target = reward + self.args.gamma * q_next.gather(2, self.eval_net(next_state).max(2)[1].unsqueeze(dim=2)).view(self.args.batch_size, self.args.num_steps, 1)
            q_target = q_target.detach()
        else:
            q_target = reward + self.args.gamma * q_next.max(2)[0].view(self.args.batch_size, self.args.num_steps, 1)
            q_target = q_target.detach()

        loss1 = self.loss_func1(q_eval, q_target)

        p_prediction = self.PNet(state, action)
        loss2 = 0
        for j_other in range(self.args.num_agents - 1):
            other_next_action_prob_prediction = p_prediction[:, :, j_other*self.action_num:j_other*self.action_num+self.action_num]
            loss2 += self.loss_func2(other_next_action_prob_prediction.reshape(self.args.batch_size*self.args.num_steps,-1), other_next_action[:, :, j_other, ...].reshape(self.args.batch_size*self.args.num_steps))

        loss = loss1 + loss2

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer_list, self.args.grad_norm_clip)
        self.optimizer.step()
        return loss
