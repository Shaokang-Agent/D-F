from tqdm import tqdm
from Agent.maddpg_agent import Maddpg_Agent
from Agent.ddpg_agent import ddpg_Agent
from common.replay_buffer_vsadv import Buffer_VS
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.env = env
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm
        self.save_para_path = './parameters_save/' + self.args.scenario_name
        self.tensorboard_dir = self.args.tensorboard_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm
        self.return_list = np.empty([self.args.round, int(self.args.max_episode_num / self.args.evaluate_rate_epi) - 1])
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.save_para_path):
            os.makedirs(self.save_para_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Maddpg_Agent(i, self.args)
            agents.append(agent)
        adv_agent = ddpg_Agent(self.args.n_agents, self.args)
        return agents, adv_agent

    def run(self, run_number):
        self.agents, self.adv_agent = self._init_agents()
        self.buffer = Buffer_VS(self.args)
        self.writer = SummaryWriter(self.tensorboard_dir + "/" + str(run_number))
        for epi in tqdm(range(self.args.max_episode_num)):
            # reset the environment
            s = self.env.reset()
            for step in range(self.args.max_episode_step):
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        if self.args.scenario_name == "simple_tag":
                            action = agent.select_action(s[agent_id], self.epsilon)
                        if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_crypto" or self.args.scenario_name == "simple_push":
                            action = agent.select_action(s[self.args.num_adversaries+agent_id], self.epsilon)
                        actions.append(action)
                if self.args.scenario_name == "simple_tag":
                    adv_action = self.adv_agent.select_action(s[self.args.n_agents], self.epsilon)
                    actions.append(adv_action)
                if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_push" or self.args.scenario_name == "simple_crypto":
                    adv_action = self.adv_agent.select_action(s[0], self.epsilon)
                    actions.insert(0, adv_action)
                s_next, r, done, info = self.env.step(actions)
                if self.args.scenario_name == "simple_tag":
                    buffer_s = s
                    buffer_s_next = s_next
                    buffer_r = r
                    buffer_a = actions
                if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_crypto" or self.args.scenario_name == "simple_push":
                    buffer_s = s[1:self.args.n_players].copy()
                    buffer_s = np.append(buffer_s, s[0])
                    buffer_a = actions[1:self.args.n_players].copy()
                    buffer_a = np.append(buffer_a, actions[0])
                    buffer_r = r[1:self.args.n_players].copy()
                    buffer_r = np.append(buffer_r, r[0])
                    buffer_s_next = s_next[1:self.args.n_players].copy()
                    buffer_s_next = np.append(buffer_s_next, s_next[0])
                self.buffer.store_episode(buffer_s, buffer_a, buffer_r, buffer_s_next)
                s = s_next
                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)
                    if self.args.save_policy:
                        self.adv_agent.learn(transitions)
                self.epsilon = max(0.05, self.epsilon - 0.0000005)
            if epi > 0 and epi % self.args.evaluate_rate_epi == 0:
                rewards = self.evaluate(run_number)
                self.return_list[run_number, int(epi / self.args.evaluate_rate_epi) - 1] = rewards
                self.writer.add_scalar("Total_reward", rewards, int(epi / self.args.evaluate_rate_epi) - 1)
                np.save(self.save_path + '/returns.pkl', self.return_list)
        if self.args.save_policy:
            self.adv_agent.policy.save_model(self.save_para_path)

    def evaluate(self, run_number):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        if self.args.scenario_name == "simple_tag":
                            action = agent.select_action(s[agent_id], self.epsilon)
                        if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_crypto" or self.args.scenario_name == "simple_push":
                            action = agent.select_action(s[self.args.num_adversaries+agent_id], self.epsilon)
                        actions.append(action)
                if self.args.scenario_name == "simple_tag":
                    adv_action = self.adv_agent.select_action(s[self.args.n_agents], self.epsilon)
                    actions.append(adv_action)
                if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_push" or self.args.scenario_name == "simple_crypto":
                    adv_action = self.adv_agent.select_action(s[0], self.epsilon)
                    actions.insert(0, adv_action)
                s_next, r, done, info = self.env.step(actions)
                if self.args.scenario_name == "simple_tag" or self.args.scenario_name == "simple_spread":
                    rewards += r[0]
                if self.args.scenario_name == "simple_adversary" or self.args.scenario_name == "simple_crypto":
                    rewards += r[1]
                if self.args.scenario_name == "simple_push":
                    rewards += np.mean(r[1:])
                s = s_next
            returns.append(rewards)
            print('Run {}, Returns is {}'.format(run_number, rewards))
        return sum(returns) / self.args.evaluate_episodes
