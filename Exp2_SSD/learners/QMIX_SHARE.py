import torch
import os
from models.qmix_net_share import QMixNet, QNet
import torch.nn.functional as F
import numpy as np

class QMIX_SHARE:
    def __init__(self, args, action_num):
        self.args = args
        self.action_num = action_num
        self.train_step = 0
        self.eval_q = QNet(self.args, self.action_num)
        self.target_q = QNet(self.args, self.action_num)
        self.eval_qmix_net = QMixNet(self.args)
        self.target_qmix_net = QMixNet(self.args)
        if self.args.cuda:
            self.eval_q.cuda()
            self.target_q.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()

        self.target_q.load_state_dict(self.eval_q.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_q.parameters()) + list(self.eval_qmix_net.parameters())

        # self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)

    def choose_action(self, obs, episolon, agent_id):
        if np.random.rand() <= episolon:
            obsnp = np.array(obs)
            obs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(obsnp.copy()), 0),0)
            obs = obs.float()
            if self.args.cuda:
                obs = obs.cuda()
            with torch.no_grad():
                action_value = self.eval_q(obs)
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

    def learn(self, episode_data):
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

        q_evals = []
        q_evals_next = []
        q_targets = []
        for i in range(self.args.num_agents):
            q_evals.append(self.eval_q(batch_observation[:,:,i,...]))
            q_evals_next.append(self.eval_q(batch_next_observation[:,:,i,...]).detach())
            q_targets.append(self.target_q(batch_next_observation[:,:,i,...]).detach())
        q_evals = torch.stack(q_evals, dim=2)
        q_evals_next = torch.stack(q_evals_next, dim=2)
        q_targets = torch.stack(q_targets, dim=2)

        q_evals = q_evals.gather(dim=3, index=batch_action)
        a_ = torch.max(q_evals_next, dim=3)[1].unsqueeze(dim=3)
        q_targets = q_targets.gather(dim=3, index=a_)

        q_total_eval = self.eval_qmix_net(q_evals, batch_state)
        q_total_target = self.target_qmix_net(q_targets, batch_next_state).detach()

        targets = (batch_reward.mean(dim=2) + self.args.gamma * q_total_target).detach()
        loss = (targets - q_total_eval).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if self.train_step % self.args.replace_param == 0:
            self.target_q.load_state_dict(self.eval_q.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        self.train_step += 1

        return loss
