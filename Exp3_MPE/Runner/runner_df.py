from tqdm import tqdm
from Agent.df_agent import D_Agent, F_Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.env = env
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm + '-' + str(self.args.lamda)
        self.tensorboard_dir = self.args.tensorboard_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm + '-' + str(self.args.lamda)
        self.return_list = np.empty([self.args.round, int(self.args.max_episode_num / self.args.evaluate_rate_epi) - 1])
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        d_agents = []
        for i in range(self.args.n_agents):
            d_agent = D_Agent(i, self.args)
            d_agents.append(d_agent)
        f_agent = F_Agent(self.args)
        return d_agents, f_agent

    def run(self, run_number):
        self.d_agents, self.f_agent = self._init_agents()
        self.buffer = Buffer(self.args)
        self.writer = SummaryWriter(self.tensorboard_dir + "/" + str(run_number))
        for epi in tqdm(range(self.args.max_episode_num)):
            # reset the environment
            s = self.env.reset()
            for step in range(self.args.max_episode_step):
                u = []
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.d_agents):
                        if self.args.scenario_name == "simple_tag" or self.args.scenario_name == "simple_spread" or self.args.scenario_name == "simple_world_comm":
                            action = agent.select_action(s[agent_id], self.epsilon)
                        if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_crypto" or self.args.scenario_name == "simple_push":
                            action = agent.select_action(s[self.args.num_adversaries+agent_id], self.epsilon)
                        u.append(action)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    if self.args.scenario_name == "simple_world_comm":
                        actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                        actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                    if self.args.scenario_name == "simple_tag" or self.args.scenario_name == "simple_spread":
                        actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                    if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_push":
                        actions.insert(0, [0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                    if self.args.scenario_name == "simple_crypto":
                        actions.insert(0, np.array(
                            [np.random.rand() * 2 - 1, np.random.rand() * 2 - 1, np.random.rand() * 2 - 1,
                             np.random.rand() * 2 - 1]))
                s_next, r, done, info = self.env.step(actions)
                if self.args.scenario_name == "simple_tag" or self.args.scenario_name == "simple_spread" or self.args.scenario_name == "simple_world_comm":
                    self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents],
                                              s_next[:self.args.n_agents])
                if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_crypto" or self.args.scenario_name == "simple_push":
                    self.buffer.store_episode(s[self.args.num_adversaries:self.args.n_players], u,
                                              r[self.args.num_adversaries:self.args.n_players],
                                              s_next[self.args.num_adversaries:self.args.n_players])
                s = s_next
                if self.buffer.current_size >= self.args.batch_size:
                    if (step + 1) % self.args.fed_steps == 0:
                        omega_ups = []
                        omega_0_model = self.f_agent.boardcast_omega_to_agents()
                    for agent in self.d_agents:
                        transitions = self.buffer.sample(self.args.batch_size)
                        agent.learn(transitions)
                        if (step + 1) % self.args.fed_steps == 0:
                            agent.learn_regularization(omega_0_model)
                            omega_up = agent.upload_paremeters(omega_0_model)
                            omega_ups.append(omega_up)
                    transitions = self.buffer.sample(self.args.batch_size)
                    if (step + 1) % self.args.fed_steps == 0:
                        self.f_agent.download_update_parameters(omega_ups)
                    self.f_agent.learn(transitions)
                self.epsilon = max(0.05, self.epsilon - 0.0000005)
            if epi > 0 and epi % self.args.evaluate_rate_epi == 0:
                rewards = self.evaluate(run_number)
                self.return_list[run_number, int(epi / self.args.evaluate_rate_epi) - 1] = rewards
                self.writer.add_scalar("Total_reward", rewards, int(epi / self.args.evaluate_rate_epi) - 1)
                np.save(self.save_path + '/returns.pkl', self.return_list)

    def evaluate(self, run_number):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.d_agents):
                        if self.args.scenario_name == "simple_tag" or self.args.scenario_name == "simple_spread" or self.args.scenario_name == "simple_world_comm":
                            action = agent.select_action(s[agent_id], 0)
                        if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_crypto" or self.args.scenario_name == "simple_push":
                            action = agent.select_action(s[self.args.num_adversaries + agent_id], 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    if self.args.scenario_name == "simple_world_comm":
                        actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                        actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                    if self.args.scenario_name == "simple_tag" or self.args.scenario_name == "simple_spread":
                        actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                    if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_push":
                        actions.insert(0, [0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                    if self.args.scenario_name == "simple_crypto":
                        actions.insert(0, np.array([np.random.rand() * 2 - 1, np.random.rand() * 2 - 1, np.random.rand() * 2 - 1,
                                           np.random.rand() * 2 - 1]))
                s_next, r, done, info = self.env.step(actions)
                if self.args.scenario_name == "simple_tag" or self.args.scenario_name == "simple_spread":
                    rewards += r[0]
                if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_crypto":
                    rewards += r[1]
                if self.args.scenario_name == "simple_push":
                    rewards += np.mean(r[1:])
                if self.args.scenario_name == "simple_world_comm":
                    rewards += np.mean(r[:4])
                s = s_next
            returns.append(rewards)
            print('Algorithm {}, Run {}, Returns is {}'.format(self.args.algorithm + '-' + str(self.args.lamda), run_number, rewards))
        return sum(returns) / self.args.evaluate_episodes
