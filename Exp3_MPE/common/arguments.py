import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--algorithm", type=str, default="MADDPG")
    parser.add_argument("--scenario_name", type=str, default="simple_tag", help="name of the scenario script: simple_tag, simple_world_comm, simple_spread, simple_adversary, simple_crypto, simple_push")
    parser.add_argument("--max_episode_step", type=int, default=100, help="maximum episode length")
    parser.add_argument("--max_episode_num", type=int, default=10000, help="number of time steps")
    parser.add_argument("--fed_steps", type=int, default=25, help="number of time steps")
    parser.add_argument("--num_adversaries", type=int, default=1, help="number of adversaries")
    parser.add_argument("--alpha", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--actor_lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--lamda", type=float, default=5, help="personlized rate")
    parser.add_argument("--beta", type=float, default=1, help="update rate")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.99, help="parameter for updating the target network")
    parser.add_argument("--udpate_target_step", type=int, default=100, help="number of time steps")
    parser.add_argument("--buffer_size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=64, help="number of episodes to optimize at the same time")

    parser.add_argument("--save_dir", type=str, default="./data", help="directory in which training state and model should be saved")
    parser.add_argument("--tensorboard_dir", type=str, default="./runs", help="directory in which training state and model should be saved")
    parser.add_argument("--save_rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model_dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--load_policy", type=bool, default=True)
    parser.add_argument("--save_policy", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="./parameters_save")

    parser.add_argument("--evaluate_episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate_episode_len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate_rate_epi", type=int, default=10, help="how often to evaluate model")
    parser.add_argument("--round", type=int, default=5)
    parser.add_argument("--cuda", type=bool, default=False)
    args = parser.parse_args()

    return args
