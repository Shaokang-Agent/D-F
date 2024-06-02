import numpy as np
import pandas as pd
import argparse
import sys
sys.path.append('./')
import torch
from run_scripts.runner import Runner
from run_scripts.runner_ppo import Runner_ppo
from run_scripts.runner_mappo import Runner_mappo
from run_scripts.runner_df import Runner_df


parser = argparse.ArgumentParser()

parser.add_argument("--algorithm", type=str, default="DF") #DQN,DQN-AVG,DQN-MIN,DQN-RMF,DQN-IA,SOCIAL,QMIX,DF,PPO,MAPPO,DDPG,MADDPG
parser.add_argument("--env", type=str, default="Cleanup") #Cleanup, Harvest
parser.add_argument("--num_agents", type=int, default=5)
parser.add_argument("--num_episodes", type=int, default=10000)
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--evaluate_cycle", type=int, default=1)
parser.add_argument("--train_steps", type=int, default=1)
parser.add_argument("--evaluate_epi", type=int, default=1)
parser.add_argument("--buffer_size", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--lamdaw", type=float, default=1)
parser.add_argument("--lamdae", type=float, default=0.1)
parser.add_argument("--cuda", type=bool, default=False)
parser.add_argument("--round", type=int, default=5)
args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = True

if args.algorithm == "DQN":
    args.lr = 5e-4
    args.epsilon_init = 0.5
    args.epsilon_steplen = args.num_episodes/5
    args.epsilon_final = 0.99
    args.target_update_iter = 100
    args.double_dqn = False
    args.grad_norm_clip = 5
    print("Initialized: DQN")
elif args.algorithm == "DQN-AVG" or args.algorithm == "DQN-MIN" or args.algorithm == "DQN-RMF" or args.algorithm == "DQN-IA":
    args.lr = 5e-4
    args.epsilon_init = 0.5
    args.epsilon_steplen = args.num_episodes/5
    args.epsilon_final = 0.99
    args.target_update_iter = 100
    args.double_dqn = False
    args.grad_norm_clip = 5
    args.IA_alpha = 0.5
    args.IA_beta = 0.05
    args.RMF_alpha = 0.1
    print("Initialized: " + str(args.algorithm))
elif args.algorithm == "SOCIAL":
    args.lr = 5e-4
    args.epsilon_init = 0.5
    args.epsilon_steplen = args.num_episodes/5
    args.epsilon_final = 0.99
    args.target_update_iter = 100
    args.double_dqn = False
    args.grad_norm_clip = 10
    args.r_in_scale = 0.03
    args.env_alpha_steplen = args.num_episodes / 1.5
    args.env_alpha_initial = 0.5
    args.env_alpha_final = 0.95
    print("Initialized: SOCIAL")
elif args.algorithm == "DDPG":
    args.actor_lr = 5e-4
    args.critic_lr = 1e-3
    args.epsilon_init = 0.5
    args.epsilon_final = 0.99
    args.epsilon_steplen = args.num_episodes/5
    args.tau = 0.99
    args.replace_param = 200
    args.grad_norm_clip = 10
    print("Initialized: DDPG")
elif args.algorithm == "MADDPG":
    args.actor_lr = 5e-4
    args.critic_lr = 1e-3
    args.epsilon_init = 0.5
    args.epsilon_final = 0.99
    args.epsilon_steplen = args.num_episodes/5
    args.tau = 0.99
    args.replace_param = 200
    args.grad_norm_clip = 10
    print("Initialized: MADDPG")
elif args.algorithm == "QMIX":
    args.lr = 1e-3
    args.epsilon_init = 0.5
    args.epsilon_steplen = args.num_episodes/5
    args.epsilon_final = 0.99
    args.two_hyper_layers = True
    args.qmix_hidden_dim = 32
    args.hyper_hidden_dim = 32
    args.tau = 0.99
    args.replace_param = 200
    args.grad_norm_clip = 15
    print("Initialized: QMIX")
else:
    print("None")

if args.algorithm != "PPO" and args.algorithm != "MAPPO" and args.algorithm != "DF":
    print("Environment: {}, agent-number: {}, algorithm: {}, cuda: {}".format(args.env, args.num_agents, args.algorithm, args.cuda))
    runner = Runner(args)
    for i in range(args.round):
        runner.run(i)

if args.algorithm == "PPO":
    args.actor_lr = 5e-4
    args.critic_lr = 1e-3
    args.clip_param = 0.05
    args.epsilon_init = 0.5
    args.epsilon_final = 0.99
    args.epsilon_steplen = args.num_episodes/5
    args.training_times = 10
    args.training_epi_gap = 5
    args.grad_clip = 10
    print("Initialized: PPO")
    print("Environment: {}, agent-number: {}, algorithm: {}, cuda: {}".format(args.env, args.num_agents, args.algorithm, args.cuda))
    runner = Runner_ppo(args)
    for i in range(args.round):
        runner.run(i)

if args.algorithm == "MAPPO":
    args.actor_lr = 5e-4
    args.critic_lr = 1e-3
    args.clip_param = 0.05
    args.epsilon_init = 0.5
    args.epsilon_final = 0.99
    args.epsilon_steplen = args.num_episodes/5
    args.training_times = 10
    args.training_epi_gap = 5
    args.grad_clip = 10
    print("Initialized: MAPPO")
    print("Environment: {}, agent-number: {}, algorithm: {}, cuda: {}".format(args.env, args.num_agents, args.algorithm, args.cuda))
    runner = Runner_mappo(args)
    for i in range(args.round):
        runner.run(i)

if args.algorithm == "DF":
    args.gamma = 0.95
    args.lr = 1e-3
    args.epsilon_init = 0.5
    args.epsilon_steplen = args.num_episodes/5
    args.epsilon_final = 0.99
    args.two_hyper_layers = True
    args.qmix_hidden_dim = 32
    args.hyper_hidden_dim = 32
    args.alpha = 1e-3
    args.beta = 1.5
    args.target_update_iter = 100
    args.grad_norm_clip = 15
    args.fed_epis = 10
    args.double_dqn = True
    print("Initialized: DF")
    print("Environment: {}, agent-number: {}, algorithm: {}, cuda: {}".format(args.env, args.num_agents, args.algorithm, args.cuda))
    runner = Runner_df(args)
    for i in range(args.round):
        runner.run(i)
