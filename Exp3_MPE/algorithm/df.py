import torch
import os
from model.Fma_actor_critic import Actor, Critic
from model.Fma_actor_critic import FMA_Actor, FMA_Critic
import numpy as np
import copy

class D_policy:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args,self.agent_id)
        self.critic_network = Critic(args,self.agent_id)

        # build up the target network
        self.actor_target_network = Actor(args,self.agent_id)
        self.critic_target_network = Critic(args,self.agent_id)

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

        # if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
        #     self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        model_path = os.path.join(model_path, self.args.algorithm)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')

class F_policy:
    def __init__(self, args):
        self.args = args
        self.train_step = 0

        # create the network
        self.F_actor_network = FMA_Actor(args)
        self.F_critic_network = FMA_Critic(args)

        # build up the target network
        self.F_actor_target_network = FMA_Actor(args)
        self.F_critic_target_network = FMA_Critic(args)

        if args.cuda:
            self.F_actor_network.cuda()
            self.F_critic_network.cuda()
            self.F_actor_target_network.cuda()
            self.F_critic_target_network.cuda()

        # load the weights into the target networks
        self.F_actor_target_network.load_state_dict(self.F_actor_network.state_dict())
        self.F_critic_target_network.load_state_dict(self.F_critic_network.state_dict())


        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.F_actor_network.parameters(), lr=self.args.alpha)
        self.critic_optim = torch.optim.Adam(self.F_critic_network.parameters(), lr=self.args.alpha)

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.F_actor_target_network.parameters(), self.F_actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.F_critic_target_network.parameters(), self.F_critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions):
        if self.args.scenario_name == "simple_world_comm":
            o, u, r, o_next = [], [], [], []
            for agent_id in range(self.args.n_agents):
                o.append(transitions['o_%d' % agent_id])
                r.append(transitions['r_%d' % agent_id])
                a = torch.from_numpy(np.array(transitions['u_%d' % agent_id])).float()
                if self.args.cuda:
                    a = a.cuda()
                u.append(a)
                o_next.append(transitions['o_next_%d' % agent_id])

            r = torch.from_numpy(np.array(r)).float()
            o = torch.from_numpy(np.array(o)).float()
            o_next = torch.from_numpy(np.array(o_next)).float()
            if self.args.cuda:
                r = r.cuda()
                o = o.cuda()
                o_next = o_next.cuda()

            u_next = []
            with torch.no_grad():
                for agent_id in range(self.args.n_agents):
                    u_next.append(self.F_actor_target_network(o_next[agent_id]))
                q_next = self.F_critic_target_network(o_next, u_next).detach()
                target_q = (r.mean(dim=1) + self.args.gamma * q_next).detach()
        else:
            o, u, r, o_next = [], [], [], []
            for agent_id in range(self.args.n_agents):
                o.append(transitions['o_%d' % agent_id])
                u.append(transitions['u_%d' % agent_id])
                r.append(transitions['r_%d' % agent_id])
                o_next.append(transitions['o_next_%d' % agent_id])

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

            u_next = []
            with torch.no_grad():
                for agent_id in range(self.args.n_agents):
                    u_next.append(self.F_actor_target_network(o_next[agent_id]))
                u_next = torch.stack(u_next)
                if self.args.cuda:
                    u_next = u_next.cuda()
                q_next = self.F_critic_target_network(o_next, u_next).detach()
                target_q = (r.mean(dim=1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.F_critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean() * self.args.n_agents
        actor_loss = 0
        # the actor loss
        for agent_id in range(self.args.n_agents):
            uc = copy.deepcopy(u)
            uc[agent_id] = self.F_actor_network(o[agent_id])
            actor_loss += - self.F_critic_network(o, u).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.train_step % self.args.udpate_target_step == 0:
            self._soft_update_target_network()
        # if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
        #     self.save_model(self.train_step)
        self.train_step += 1






