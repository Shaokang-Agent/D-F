import torch
import os
from model.actor_critic import Actor, Critic
import numpy as np

class DDPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args, agent_id)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args, agent_id)

        if args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())


        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.alpha)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.alpha)

        if self.args.load_policy and self.agent_id == self.args.n_agents:
            self.model_path = self.args.model_path + "/" + self.args.scenario_name
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id, self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id, self.model_path + '/critic_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions):
        # for key in transitions.keys():
        #     transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]
        o = transitions['o_%d' % self.agent_id]
        u = transitions['u_%d' % self.agent_id]
        o_next = transitions['o_next_%d' % self.agent_id]

        r = torch.from_numpy(np.array(r)).float()
        o = torch.from_numpy(np.array(o)).float()
        u = torch.from_numpy(np.array(u)).float()
        o_next = torch.from_numpy(np.array(o_next)).float()
        if self.args.cuda:
            r = r.cuda()
            o = o.cuda()
            u = u.cuda()
            o_next = o_next.cuda()
        # calculate the target Q value function
        u_next = self.actor_target_network(o_next).detach()
        q_next = self.critic_target_network(o_next, u_next).detach()
        target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        u_new = self.actor_network(o)
        actor_loss = - self.critic_network(o, u_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.train_step % self.args.udpate_target_step == 0:
            self._soft_update_target_network()

        self.train_step += 1

    def save_model(self, save_para_dir):
        torch.save(self.actor_network.state_dict(), save_para_dir + '/' + 'actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  save_para_dir + '/' + 'critic_params.pkl')


