import numpy as np
class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.args = args
        print('Init RolloutWorker')

    def generate_episode(self, episode_num, evaluate=False):
        epi_s, epi_s_next, epi_o, epi_u, epi_r, epi_o_next, epi_terminate = [], [], [], [], [], [], []
        state, observation = self.env.reset()
        state = state / 256
        for i in range(self.args.num_agents):
            observation["agent-" + str(i)] = observation["agent-" + str(i)] / 256
        terminated = False
        step = 0
        episode_reward = np.zeros(self.args.num_agents)
        # epsilon
        if evaluate:
            epsilon = 1
        else:
            epsilon = np.min([1, self.args.epsilon_init + (self.args.epsilon_final - self.args.epsilon_init) * episode_num / self.args.epsilon_steplen])

        while not terminated and step < self.args.num_steps:
            o, u, r, o_next, terminate = [], [], [], [], []
            actions_dict = {}
            for i in range(self.args.num_agents):
                o.append(observation["agent-" + str(i)])
                action = self.agents.choose_action(o[i], epsilon, i)
                u.append(action)
                actions_dict["agent-" + str(i)] = action
            state_next, observation_next, reward, dones, infos = self.env.step(actions_dict)
            state_next = state_next / 256
            for i in range(self.args.num_agents):
                observation_next["agent-" + str(i)] = observation_next["agent-" + str(i)] / 256
                o_next.append(observation_next["agent-" + str(i)])
                r.append(reward["agent-"+str(i)])
                terminate.append(dones["agent-" + str(i)])
            episode_reward += np.array(r)
            epi_o.append(o)
            epi_u.append(u)
            epi_r.append(r)
            epi_o_next.append(o_next)
            epi_s.append(state)
            epi_s_next.append(state_next)
            epi_terminate.append(terminate)

            state = state_next
            observation = observation_next
            step += 1

        episode = dict(o=epi_o.copy(),
                       u=epi_u.copy(),
                       r=epi_r.copy(),
                       o_next=epi_o_next.copy(),
                       terminate=epi_terminate.copy(),
                       s=epi_s.copy(),
                       s_next=epi_s_next.copy()
                       )
        return episode, episode_reward
