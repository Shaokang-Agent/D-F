import numpy as np
import torch
from algorithm.df import D_policy, F_policy
import copy
from model.Fma_actor_critic import Actor, Critic
from model.Fma_actor_critic import FMA_Actor, FMA_Critic

class D_Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = D_policy(args, agent_id)

    def select_action(self, o, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                inputs = inputs.cuda()
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    def new_model(self, target_model):
        model = FMA_Actor(self.args)
        for param, policy_param in zip(model.parameters(), target_model.parameters()):
            param.data = policy_param.data
        return model

    def upload_paremeters(self, omega_federation_model):
        omega_up = self.new_model(omega_federation_model)
        for local_param, fed_param in zip(self.policy.actor_network.parameters(), omega_up.parameters()):
            fed_param.data = fed_param.data - self.args.alpha * self.args.lamda * (fed_param.data - local_param.data)
        return omega_up

    def learn(self, transitions):
        self.policy.train(transitions)

    def learn_regularization(self, omega_federation_model):
        for local_param, fed_param in zip(self.policy.actor_network.parameters(), omega_federation_model.parameters()):
            local_param.data = local_param.data - self.args.alpha * self.args.lamda * (local_param.data - fed_param.data)

class F_Agent:
    def __init__(self, args):
        self.args = args
        self.policy = F_policy(args)
    def new_model(self, target_model):
        model = FMA_Actor(self.args)
        for param, policy_param in zip(model.parameters(), target_model.parameters()):
            param.data = policy_param.data
        return model

    def boardcast_omega_to_agents(self):
        return self.new_model(self.policy.F_actor_network)

    def download_update_parameters(self, models):
        new_model = self.new_model(self.policy.F_actor_network)
        for new_param, old_param in zip(new_model.parameters(), self.policy.F_actor_network.parameters()):
            new_param.data = (1-self.args.beta) * old_param.data
        for i in range(self.args.n_agents):
            for fed_param, upload_param in zip(new_model.parameters(), models[i].parameters()):
                fed_param.data = fed_param.data + self.args.beta * upload_param.data / self.args.n_agents
        self.update_parameters(new_model.parameters())

    def update_parameters(self, target_parameters):
        for param, target_param in zip(self.policy.F_actor_network.parameters(), target_parameters):
            param.data = target_param.data

    def learn(self, transitions):
        self.policy.train(transitions)
