import sys

sys.path.append("../")
import numpy as np
import os
from replay_buffer.replay_buffer_episode import ReplayBuffer
from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from ray.tune.registry import register_env
import torch
from torch.utils.tensorboard import SummaryWriter
from learners.DF import D_agent, F_agent
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


class Runner_df:
    def __init__(self, args):
        env, action_num = make_env(args)
        self.env = env
        self.args = args
        self.args.action_num = action_num
        self.episode_rewards = np.empty(
            [self.args.round, self.args.num_agents, int(self.args.num_episodes / self.args.evaluate_cycle)])
        self.save_data_path = './data/' + self.args.env + str(self.args.num_agents) + '/' + self.args.algorithm + "-w" + str(self.args.lamdaw) + "-e" + str(self.args.lamdae)

        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)

        file = sorted(os.listdir(self.save_data_path))
        if file == []:
            self.next_num = 1
        else:
            self.next_num = int(file[-1].split('.')[0][-1]) + 1

    def run(self, num):
        self.DAgent = [D_agent(self.args, self.args.action_num) for _ in range(self.args.num_agents)]
        self.FAgent = F_agent(self.args, self.args.action_num, "T")
        self.FFair = F_agent(self.args, self.args.action_num, "F")
        self.buffer = ReplayBuffer(self.args)
        from run_scripts.rollout_df import RolloutWorker
        self.rolloutWorker = RolloutWorker(self.env, self.DAgent, self.args)
        self.writer = SummaryWriter("./runs/" + self.args.env + str(self.args.num_agents) + "/" + self.args.algorithm + "-w" + str(self.args.lamdaw) + "-e" + str(self.args.lamdae) + "/" + str(num))
        train_steps = 0
        for epi in tqdm(range(self.args.num_episodes)):
            epi_loss = np.zeros(self.args.num_agents)
            print('Env {}, Run {}, train episode {}'.format(self.args.env, num, epi))
            if epi % self.args.evaluate_cycle == 0:
                episode_individual_reward = self.evaluate()
                episode_reward = np.sum(episode_individual_reward)
                self.episode_rewards[num, :, int(epi / self.args.evaluate_cycle)] = episode_individual_reward
                for i in range(self.args.num_agents):
                    self.writer.add_scalar("Agent_{}_reward".format(str(i)), episode_individual_reward[i], epi)
                self.writer.add_scalar("Total_reward", episode_reward, epi)
                print("training episode {}, total_reward {}, algorithm {}, agent_num {}".format(epi, episode_reward,
                                                                                                self.args.algorithm + "-w" + str(
                                                                                                    self.args.lamdaw) + "-e" + str(
                                                                                                    self.args.lamdae),
                                                                                                self.args.num_agents))
            episode_data, _ = self.rolloutWorker.generate_episode(epi)
            self.buffer.add(episode_data)
            if self.args.batch_size < self.buffer.__len__():
                if epi % self.args.fed_epis == 0:
                    omega_ups = []
                    omega_0_model = self.FAgent.boardcast_omega_to_agents()
                    omega_fair_model = self.FFair.boardcast_omega_to_agents()
                for i in range(self.args.num_agents):
                    sample = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                    loss = self.DAgent[i].learn(sample, i)
                    self.writer.add_scalar("Agent_{}_Loss".format(i), loss, epi)
                    epi_loss[i] += loss
                    if epi % self.args.fed_epis == 0:
                        self.DAgent[i].learn_regularization(omega_0_model, omega_fair_model)
                        self.DAgent[i].learn_regularization(omega_0_model, omega_0_model)
                        omega_up = self.DAgent[i].upload_paremeters(omega_0_model)
                        omega_ups.append(omega_up)
                sample = self.buffer.sample(min(self.buffer.__len__(), 32))
                if epi % self.args.fed_epis == 0:
                    self.FAgent.download_update_parameters(omega_ups, omega_fair_model)
                    self.FAgent.download_update_parameters(omega_ups, omega_0_model)
                self.FAgent.learn(sample)
                self.FFair.learn(sample)
                train_steps += 1
        np.save(self.save_data_path + '/epi_total_reward_{}'.format(str(self.next_num)), self.episode_rewards)


    def evaluate(self):
        episode_rewards = 0
        for epi in range(self.args.evaluate_epi):
            _, episode_reward = self.rolloutWorker.generate_episode(epi, evaluate=True)
            episode_rewards += episode_reward
        return episode_rewards / self.args.evaluate_epi










