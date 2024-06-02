import sys
sys.path.append("../")
import numpy as np
import os
from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from ray.tune.registry import register_env
from torch.utils.tensorboard import SummaryWriter
from learners.MAPPO import MAPPO
from tqdm import tqdm

def make_env(args):
    if args.env == "Harvest":
        single_env = HarvestEnv(num_agents=args.num_agents)
        env_name = "HarvestEnv"
        def env_creator(_):
            return HarvestEnv(num_agents=args.num_agents)
    elif args.env == "Cleanup":
        single_env = CleanupEnv(num_agents=args.num_agents)
        env_name = "CleanupEnv"
        def env_creator(_):
            return CleanupEnv(num_agents=args.num_agents)
    else:
        return 0
    register_env(env_name, env_creator)
    if env_name == "HarvestEnv":
        action_num = 8
    else:
        action_num = 9
    return single_env, action_num


class Runner_mappo:
    def __init__(self, args):
        env, action_num = make_env(args)
        self.env = env
        self.args = args
        self.args.action_num = action_num
        self.episode_rewards = np.empty([self.args.round, self.args.num_agents, int(self.args.num_episodes/self.args.evaluate_cycle)])
        self.save_data_path = './data/' + self.args.env + str(self.args.num_agents) + '/' + self.args.algorithm
        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)
        file = sorted(os.listdir(self.save_data_path))
        if file == []:
            self.next_num = 1
        else:
            self.next_num = int(file[-1].split('.')[0][-1]) + 1

    def run(self, num):
        self.agents = MAPPO(self.args, self.args.action_num)
        self.writer = SummaryWriter("./runs/" + self.args.env + str(self.args.num_agents) + "/" + self.args.algorithm + "/" + str(num))
        train_steps = 0
        for epi in tqdm(range(self.args.num_episodes)):
            print('Env {}, Run {}, train episode {}'.format(self.args.env, num, epi))
            self.agents.init_hidden()
            epi_s, epi_s_next, epi_o, epi_u, epi_u_probability, epi_r, epi_o_next, epi_terminate = [], [], [], [], [], [], [], []
            state, observation = self.env.reset()
            state = state / 256
            for i in range(self.args.num_agents):
                observation["agent-" + str(i)] = observation["agent-" + str(i)] / 256
            terminated = False
            step = 0
            # epsilon
            epsilon = np.min([1, self.args.epsilon_init + (
                        self.args.epsilon_final - self.args.epsilon_init) * epi / self.args.epsilon_steplen])

            while not terminated and step < self.args.num_steps:
                o, u, u_probability, r, o_next, terminate = [], [], [], [], [], []
                actions_dict = {}
                for i in range(self.args.num_agents):
                    o.append(observation["agent-" + str(i)])
                    action, action_logprobability = self.agents.choose_action(o[i], epsilon, i)
                    u.append(action)
                    u_probability.append(action_logprobability)
                    actions_dict["agent-" + str(i)] = action
                state_next, observation_next, reward, dones, infos = self.env.step(actions_dict)
                state_next = state_next / 256
                for i in range(self.args.num_agents):
                    observation_next["agent-" + str(i)] = observation_next["agent-" + str(i)] / 256
                    o_next.append(observation_next["agent-" + str(i)])
                    r.append(reward["agent-" + str(i)])
                    terminate.append(dones["agent-" + str(i)])

                epi_o.append(o)
                epi_u.append(u)
                epi_u_probability.append(u_probability)
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
                           u_probability=epi_u_probability.copy(),
                           r=epi_r.copy(),
                           o_next=epi_o_next.copy(),
                           s=epi_s.copy(),
                           s_next=epi_s_next.copy(),
                           terminate=epi_terminate.copy()
                           )
            for _ in range(self.args.training_times):
                train_steps += 1
                closs, aloss = self.agents.learn(episode)
                self.writer.add_scalar("Agent_total_CLoss", closs, train_steps)
                self.writer.add_scalar("Agent_total_ALoss", aloss, train_steps)


            # evaulate
            episode_reward = np.zeros(self.args.num_agents)
            _, observation = self.env.reset()
            for i in range(self.args.num_agents):
                observation["agent-" + str(i)] = observation["agent-" + str(i)] / 256
            for istep in range(self.args.num_steps):
                r = []
                actions_dict = {}
                for i in range(self.args.num_agents):
                    action, action_logprobability = self.agents.choose_action(observation["agent-" + str(i)], 1, i)
                    actions_dict["agent-" + str(i)] = action
                _, observation_next, reward, dones, infos = self.env.step(actions_dict)
                for i in range(self.args.num_agents):
                    observation_next["agent-" + str(i)] = observation_next["agent-" + str(i)] / 256
                    r.append(reward["agent-" + str(i)])
                observation = observation_next
                episode_reward += np.array(r)
            self.episode_rewards[num, :, epi] = episode_reward
            for i in range(self.args.num_agents):
                self.writer.add_scalar("Agent_{}_reward".format(str(i)), episode_reward[i], epi)
            self.writer.add_scalar("Total_reward", episode_reward.sum(), epi)
            print("training episode {}, total_reward {}, algorithm {}, agent_num {}".format(epi, episode_reward.sum(),
                                                                                            self.args.algorithm,
                                                                                            self.args.num_agents))

            np.save(self.save_data_path + '/epi_total_reward_{}'.format(str(self.next_num)), self.episode_rewards)












