import numpy as np
import torch
from algorithm.maddpg import MADDPG


class Maddpg_Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

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

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

