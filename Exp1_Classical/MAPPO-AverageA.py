import numpy as np
from keras.utils import to_categorical
import copy
from common.utils import eligibility_traces, default_config, make_env, str2bool, get_omega, get_more_obs_com, discount_rewards
from common.ppo_independent import PPOPolicyNetwork, ValueNetwork
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
render = False
normalize_inputs = True

config = default_config()
LAMBDA = float(config['agent']['lambda'])
lr_actor = float(config['agent']['lr_actor'])
twophase_proportion = float(config['agent']['twophase_proportion'])


env = make_env(config, normalize_inputs)
env.toggle_compute_neighbors()

n_agent = env.n_agent
T = env.T
GAMMA = env.GAMMA
if config['agent']['env_name'] == "matthew":
   n_episode = 40000
   state_input_size = 130
elif config['agent']['env_name'] == "plant":
    n_episode = 20000
    if config['agent']['plant_ohe']:
        state_input_size = 5*5+3*12*12
    else:
        state_input_size = 5*5+12*12
else:
    n_episode = 10000
    state_input_size = 74
max_steps = env.max_steps
n_actions = env.n_actions
round = 5
reward_step = np.zeros([round, n_episode, n_agent])
if not os.path.exists("./" + config['agent']['env_name']):
    os.makedirs("./" + config['agent']['env_name'])

for round_num in range(round):
    writter = SummaryWriter("./" + config['agent']['env_name'] + "/runs/MAPPO-AverageA/" + str(round_num))
    Pi = []
    V = []

    for i in range(n_agent):
        Pi.append(PPOPolicyNetwork(num_features=env.input_size, num_actions=n_actions, layer_size=256, epsilon=0.1,
                                   learning_rate=lr_actor))
        V.append(ValueNetwork(num_features=state_input_size+n_agent+n_agent*n_actions, hidden_size=256, learning_rate=0.001))

    memory_ep_rewards = [deque() for _ in range(n_agent)]
    average_jpi = np.zeros(n_agent)

    for i_episode in tqdm(range(n_episode)):
        memory_ep_rewards = [deque() for _ in range(n_agent)]
        average_jpi = np.zeros(n_agent)

        avg = [0.] * n_agent

        ep_actions = [[] for _ in range(n_agent)]
        ep_rewards = [[] for _ in range(n_agent)]
        ep_obss = [[] for _ in range(n_agent)]
        ep_states = []
        ep_states_h = []

        score = 0
        steps = 0
        su = [0.] * n_agent
        su = np.array(su)

        env_state, obs = env.reset()

        done = False
        while steps < max_steps and not done:
            steps += 1
            action = []
            ep_states.append(env_state)
            state_h = copy.deepcopy(env_state)
            more_return = average_jpi
            more_return = (more_return - np.mean(more_return)) / (np.std(more_return) + 0.0000000001)
            state_h.extend(more_return)
            for i in range(n_agent):
                # add more information
                more_action = Pi[i].get_dist(np.array([obs[i]]))[0]
                state_h.extend(more_action)
                action.append(np.random.choice(range(n_actions), p=more_action))
                ep_actions[i].append(to_categorical(action[i], n_actions))
                ep_obss[i].append(obs[i])

            ep_states_h.append(state_h)
            env_state, obs, rewards, done = env.step(action)

            su += np.array(rewards)
            score += sum(rewards)

            for i in range(n_agent):
                ep_rewards[i].append(rewards[i])
                memory_ep_rewards[i].append(rewards[i])
                average_jpi[i] += rewards[i]
                if len(memory_ep_rewards[i]) > max_steps * 5:
                    average_jpi[i] -= memory_ep_rewards[i].popleft()

            if steps % T == 0:
                all_ep_advantages=[]
                ep_actions = np.array(ep_actions)
                ep_rewards = np.array(ep_rewards, dtype=np.float_)
                ep_states = np.array(ep_states)
                ep_states_h = np.array(ep_states_h)
                ep_rewards_mean = ep_rewards.mean(axis=0)

                nstate = copy.deepcopy(env_state)
                more_return = average_jpi
                more_return = (more_return - np.mean(more_return)) / (np.std(more_return) + 0.0000000001)
                nstate.extend(more_return)
                for i in range(n_agent):
                    more_action = Pi[i].get_dist(np.array([obs[i]]))[0]
                    nstate.extend(more_action)

                for i in range(n_agent):
                    if LAMBDA < -0.1:
                        targets = discount_rewards(ep_rewards_mean, GAMMA)
                        V[i].update(ep_states_h, targets)
                        vs = V[i].get(ep_states_h)
                    else:
                        vs = V[i].get(ep_states_h)
                        targets = eligibility_traces(np.mean(ep_rewards,axis=0), vs, V[i].get([nstate]), GAMMA, LAMBDA)
                        V[i].update(ep_states_h, targets)

                    ep_advantages = targets - vs
                    ep_advantages = (ep_advantages - np.mean(ep_advantages)) / (np.std(ep_advantages) + 0.0000000001)
                    all_ep_advantages.append(ep_advantages)

                all_ep_advantages = np.array(all_ep_advantages)
                all_ep_advantages = np.sum(all_ep_advantages, axis=0)

                for i in range(n_agent):
                    Pi[i].update(ep_obss[i], ep_actions[i], all_ep_advantages)

                ep_actions = [[] for _ in range(n_agent)]
                ep_rewards = [[] for _ in range(n_agent)]
                ep_obss = [[] for _ in range(n_agent)]
                ep_states = []
                ep_states_h = []
            if render:
                env.render()

        print(config['agent']['env_name'], "MAPPO-AverageA", round_num, i_episode)
        print(su, su.sum())
        writter.add_scalar("Total Rewards", su.sum(), i_episode)
        reward_step[round_num, i_episode] = su
        np.save("./" + config['agent']['env_name'] + "/MAPPO-AverageA.npy", reward_step)



