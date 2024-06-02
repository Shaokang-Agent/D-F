import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class QNet(nn.Module):
    """docstring for Net"""
    def __init__(self, action_num, args):
        super(QNet, self).__init__()
        self.args = args
        self.cnn = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(169*6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32, action_num)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        Batch, seq_len = x.shape[0], x.shape[1]
        x = x.permute(0,1,4,2,3)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = F.leaky_relu(self.cnn(x))
        x = x.reshape(Batch,seq_len, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.out(x)
        return x

class DQN():
    """docstring for DQN"""
    def __init__(self, args, action_num):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = QNet(action_num, args).float(), QNet(action_num, args).float()
        if args.cuda:
            self.eval_net.cuda()
            self.target_net.cuda()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.args = args
        self.action_num = action_num
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.args.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, obs, episolon):
        if np.random.rand() <= episolon:
            obsnp = np.array(obs)
            obs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(obsnp.copy()), 0),0)
            obs = obs.float()
            if self.args.cuda:
                obs = obs.cuda()
            with torch.no_grad():
                action_value = self.eval_net(obs)
                # action_value = action_value.squeeze(dim=0)
                # action = torch.max(action_value, 1)[1].cpu().data.numpy()
                # action = action[0]
                action_value = action_value.squeeze()
                action_value_max = torch.max(action_value)
                actions = torch.nonzero(torch.eq(action_value, action_value_max)).squeeze(dim=0)
                action = int(actions[np.random.randint(0, actions.shape[0])].cpu().numpy())
        else:
            action = np.random.randint(0, self.action_num)
        return action

    def learn(self, episode_data, agent_id):
        if self.learn_step_counter % self.args.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter+=1

        batch_state = torch.from_numpy(episode_data['o'][:,:,agent_id,...]).float()
        batch_action = torch.from_numpy(episode_data['u'][:,:,agent_id,...]).long()
        if self.args.algorithm == "DQN":
            batch_reward = torch.from_numpy(episode_data['r'][:, :, agent_id, ...]).float()
        if self.args.algorithm == "DQN-MIN":
            batch_reward = torch.from_numpy(episode_data['r']).float()
            batch_reward = torch.min(batch_reward, dim=2)[0]
        if self.args.algorithm == "DQN-RMF":
            batch_reward = torch.from_numpy(episode_data['r']).float()
            batch_reward = batch_reward.min(dim=2)[0] + self.args.RMF_alpha * batch_reward.mean(dim=2)
        if self.args.algorithm == "DQN-AVG":
            batch_reward = torch.from_numpy(episode_data['r']).float()
            batch_reward = batch_reward.mean(dim=2)
        if self.args.algorithm == "DQN-IA":
            reward = torch.from_numpy(episode_data['r']).float()
            batch_reward = torch.from_numpy(episode_data['r'][:, :, agent_id, ...]).float()
            for ba in range(reward.shape[0]):
                for st in range(reward.shape[1]):
                    r = copy.deepcopy(reward[ba,st])
                    r = r - r[agent_id]
                    batch_reward[ba,st] = batch_reward[ba,st] - self.args.IA_alpha * r[torch.gt(r, 0)].sum() + self.args.IA_beta * r[torch.le(r, 0)].sum()

        batch_next_state = torch.from_numpy(episode_data['o_next'][:,:,agent_id,...]).float()

        if self.args.cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

        q_eval = self.eval_net(batch_state).gather(2, batch_action)
        q_next = self.target_net(batch_next_state).detach()

        if self.args.double_dqn:
            q_target = batch_reward + self.args.gamma * q_next.gather(2, self.eval_net(batch_next_state).max(2)[1].unsqueeze(dim=2))
            q_target = q_target.detach()
        else:
            q_target = batch_reward + self.args.gamma * q_next.max(2)[0].unsqueeze(dim=2)
            q_target = q_target.detach()
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()
        return loss
