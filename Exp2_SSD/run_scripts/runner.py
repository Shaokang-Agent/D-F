import sys
sys.path.append("../")
import numpy as np
import os
from replay_buffer.replay_buffer_episode import ReplayBuffer
from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from ray.tune.registry import register_env
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from learners.DQN import DQN
from learners.SOCIAL import DQN_SOCIAL
from learners.PPO import PPO
from learners.DDPG import DDPG
from learners.MADDPG import MADDPG
from learners.QMIX_SHARE import QMIX_SHARE
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


class Runner:
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
        self.buffer = ReplayBuffer(self.args)
        if self.args.algorithm == "DQN" or self.args.algorithm == "DQN-AVG" or self.args.algorithm == "DQN-MIN" or self.args.algorithm == "DQN-RMF" or self.args.algorithm == "DQN-IA":
            self.agents = [DQN(self.args, self.args.action_num) for _ in range(self.args.num_agents)]
            from run_scripts.rollout import RolloutWorker
        elif self.args.algorithm == "SOCIAL":
            self.agents = [DQN_SOCIAL(self.args, self.args.action_num) for _ in range(self.args.num_agents)]
            from run_scripts.rollout_social import RolloutWorker
        elif self.args.algorithm == "DDPG":
            self.agents = [DDPG(self.args, self.args.action_num) for _ in range(self.args.num_agents)]
            from run_scripts.rollout import RolloutWorker
        elif self.args.algorithm == "MADDPG":
            self.agents = [MADDPG(self.args, self.args.action_num) for _ in range(self.args.num_agents)]
            from run_scripts.rollout_maddpg import RolloutWorker
        elif self.args.algorithm == "QMIX":
            self.agents = QMIX_SHARE(self.args, self.args.action_num)
            from run_scripts.rollout_qmix import RolloutWorker
        else:
            return None

        self.rolloutWorker = RolloutWorker(self.env, self.agents, self.args)
        self.writer = SummaryWriter("./runs/" + self.args.env + str(self.args.num_agents) + "/" + self.args.algorithm + "/" + str(num))
        train_steps = 0
        for epi in tqdm(range(self.args.num_episodes)):
            print('Env {}, Run {}, train episode {}'.format(self.args.env, num, epi))
            if epi % self.args.evaluate_cycle == 0:
                episode_individual_reward = self.evaluate()
                episode_reward = np.sum(episode_individual_reward)
                self.episode_rewards[num, :, int(epi/self.args.evaluate_cycle)] = episode_individual_reward
                for i in range(self.args.num_agents):
                    self.writer.add_scalar("Agent_{}_reward".format(str(i)), episode_individual_reward[i], epi)
                self.writer.add_scalar("Total_reward", episode_reward, epi)
                print("training episode {}, total_reward {}, algorithm {}, agent_num {}".format(epi, episode_reward, self.args.algorithm, self.args.num_agents))
            episode_data, _ = self.rolloutWorker.generate_episode(epi)
            self.buffer.add(episode_data)
            if self.args.batch_size < self.buffer.__len__():
                for train_step in range(self.args.train_steps):
                    if self.args.algorithm == "QMIX":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss, train_steps)
                    elif self.args.algorithm == "QMIX_SHARE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss, train_steps)
                    elif self.args.algorithm == "QMIX_SHARE_STATE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss, train_steps)
                    elif self.args.algorithm == "VDN_SHARE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss, train_steps)
                    elif self.args.algorithm == "VDN_SHARE_STATE":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        loss = self.agents.learn(mini_batch)
                        self.writer.add_scalar("Agent_Total_Loss", loss, train_steps)
                    elif self.args.algorithm == "MADDPG":
                        mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                        for i in range(self.args.num_agents):
                            closs, aloss = self.agents[i].learn(mini_batch, i, self.agents)
                            self.writer.add_scalar("Agent_{}_CLoss".format(str(i)), closs, train_steps)
                            self.writer.add_scalar("Agent_{}_ALoss".format(str(i)), aloss, train_steps)
                    elif self.args.algorithm == "DDPG":
                        for i in range(self.args.num_agents):
                            mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                            closs, aloss = self.agents[i].learn(mini_batch, i)
                            self.writer.add_scalar("Agent_{}_CLoss".format(str(i)), closs, train_steps)
                            self.writer.add_scalar("Agent_{}_ALoss".format(str(i)), aloss, train_steps)
                    elif self.args.algorithm == "DQN" or self.args.algorithm == "DQN-AVG" or self.args.algorithm == "DQN-MIN" or self.args.algorithm == "DQN-RMF" or self.args.algorithm == "DQN-IA" or self.args.algorithm == "SOCIAL":
                        for i in range(self.args.num_agents):
                            mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                            loss = self.agents[i].learn(mini_batch, i)
                            self.writer.add_scalar("Agent_{}_Loss".format(str(i)), loss, train_steps)
                    else:
                        return None
                    train_steps += 1
            np.save(self.save_data_path + '/epi_total_reward_{}'.format(str(self.next_num)), self.episode_rewards)

    def evaluate(self):
        episode_rewards = 0
        for epi in range(self.args.evaluate_epi):
            _, episode_reward = self.rolloutWorker.generate_episode(epi, evaluate=True)
            episode_rewards += episode_reward
        return episode_rewards / self.args.evaluate_epi










