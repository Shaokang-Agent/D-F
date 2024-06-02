import torch
from models.DF_net import QMixNet, QNet
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class D_agent():
    """docstring for DQN"""
    def __init__(self, args, action_num):
        super(D_agent, self).__init__()
        self.eval_net, self.target_net = QNet(args, action_num).float(), QNet(args, action_num).float()
        if args.cuda:
            self.eval_net.cuda()
            self.target_net.cuda()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.args = args
        self.action_num = self.args.action_num
        self.learn_step_counter = 0
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
                action_value = action_value.squeeze()
                action_value_max = torch.max(action_value)
                actions = torch.nonzero(torch.eq(action_value, action_value_max)).squeeze(dim=0)
                action = int(actions[np.random.randint(0, actions.shape[0])].cpu().numpy())
        else:
            action = np.random.randint(0, self.action_num)
        return action

    def learn(self, episode_data, agent_id):
        # update the parameters
        if self.learn_step_counter % self.args.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        batch_state = torch.from_numpy(episode_data['o'][:, :, agent_id, ...]).float()
        batch_action = torch.from_numpy(episode_data['u'][:, :, agent_id, ...]).long()
        batch_reward = torch.from_numpy(episode_data['r'][:, :, agent_id, ...]).float()
        batch_next_state = torch.from_numpy(episode_data['o_next'][:, :, agent_id, ...]).float()

        if self.args.cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

        q_eval = self.eval_net(batch_state).gather(2, batch_action)
        q_next = self.target_net(batch_next_state).detach()

        if self.args.double_dqn:
            q_target = batch_reward + self.args.gamma * q_next.gather(2, self.eval_net(batch_next_state).max(2)[
                1].unsqueeze(dim=2))
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

    def new_model(self, target_model):
        model = QNet(self.args, self.action_num)
        for param, policy_param in zip(model.parameters(), target_model.parameters()):
            param.data = policy_param.data
        return model

    def upload_paremeters(self, omega_0_model):
        omega_up = self.new_model(omega_0_model)
        for local_param, fed_param in zip(self.eval_net.parameters(), omega_up.parameters()):
            fed_param.data = fed_param.data - self.args.alpha * self.args.lamdaw * (fed_param.data - local_param.data)
        return omega_up

    def learn_regularization(self, omega_0_model, omega_fair_model):
        for local_param, fed_param in zip(self.eval_net.parameters(), omega_0_model.parameters()):
            local_param.data = local_param.data - self.args.alpha * self.args.lamdaw * (local_param.data - fed_param.data)
        for local_param, fed_param in zip(self.eval_net.parameters(), omega_fair_model.parameters()):
            local_param.data = local_param.data - self.args.alpha * self.args.lamdae * (local_param.data - fed_param.data)


class F_agent:
    def __init__(self, args, action_num, str):
        self.args = args
        self.str = str
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

    def choose_action(self, obs, episolon):
        if np.random.rand() <= episolon:
            obsnp = np.array(obs)
            obs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(obsnp.copy()), 0),0)
            obs = obs.float()
            if self.args.cuda:
                obs = obs.cuda()
            with torch.no_grad():
                action_value = self.eval_q(obs)
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

        if self.str == "T":
            targets = (batch_reward.sum(dim=2) + self.args.gamma * q_total_target).detach()
        if self.str == "F":
            std_reward = -batch_reward.std(dim=2)
            std_reward = (std_reward - std_reward.mean()) / ( std_reward.std() + 1e-5)
            targets = (std_reward + self.args.gamma * q_total_target).detach()
        loss = (targets - q_total_eval).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if self.train_step % self.args.target_update_iter == 0:
            self.target_q.load_state_dict(self.eval_q.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        self.train_step += 1

        return loss

    def new_model(self, target_model):
        model = QNet(self.args, self.action_num)
        for param, policy_param in zip(model.parameters(), target_model.parameters()):
            param.data = policy_param.data
        return model

    def boardcast_omega_to_agents(self):
        return self.new_model(self.eval_q)

    def download_update_parameters(self, omega_ups, omega_fair_model):
        new_model = self.new_model(self.eval_q)
        for new_param, old_param in zip(new_model.parameters(), self.eval_q.parameters()):
            new_param.data = (1-self.args.beta) * old_param.data
        for i in range(self.args.num_agents):
            for fed_param, upload_param in zip(new_model.parameters(), omega_ups[i].parameters()):
                fed_param.data = fed_param.data + self.args.beta * upload_param.data / self.args.num_agents
        for fed_param, fair_param in zip(new_model.parameters(), omega_fair_model.parameters()):
            fed_param.data = fed_param.data - self.args.alpha * self.args.lamdae * (fed_param.data - fair_param.data)
        self.update_parameters(new_model.parameters())

    def update_parameters(self, target_parameters):
        for param, target_param in zip(self.eval_q.parameters(), target_parameters):
            param.data = target_param.data


