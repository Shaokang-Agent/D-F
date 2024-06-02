import numpy as np
from keras.utils import to_categorical
import copy
from common.utils import default_config, make_env, eligibility_traces, discount_rewards
from common.ppo_independent import PPOPolicyNetwork, ValueNetwork

render = False
normalize_inputs = True

config = default_config()
env = make_env(config, normalize_inputs)
LAMBDA = float(config['agent']['lambda'])
lr_actor = float(config['agent']['lr_actor'])

n_agent = env.n_agent
T = env.T
GAMMA = env.GAMMA
n_episode = env.n_episode
max_steps = env.max_steps
n_actions = env.n_actions

i_episode = 0
Pi = []
V = []
for i in range(n_agent):
    Pi.append(PPOPolicyNetwork(num_features=env.input_size, num_actions=n_actions, layer_size=256, epsilon=0.1, learning_rate=lr_actor))
    V.append(ValueNetwork(num_features=env.input_size, hidden_size=256, learning_rate=0.001))

while i_episode < n_episode:
    i_episode += 1

    avg = [0] * n_agent

    ep_actions = [[] for _ in range(n_agent)]
    ep_rewards = [[] for _ in range(n_agent)]
    ep_states = [[] for _ in range(n_agent)]

    score = 0
    steps = 0
    su = [0.] * n_agent
    su = np.array(su)

    obs = env.reset()

    done = False
    while steps < max_steps and not done:
        steps += 1
        action = []
        for i in range(n_agent):
            h = copy.deepcopy(obs[i])
            p = Pi[i].get_dist(np.array([h]))[0]
            action.append(np.random.choice(range(n_actions), p=p))
            ep_states[i].append(h)
            ep_actions[i].append(to_categorical(action[i], n_actions))

        obs, rewards, done = env.step(action)

        su += np.array(rewards)
        score += sum(rewards)

        for i in range(n_agent):
            ep_rewards[i].append(rewards[i])

        if steps % T == 0:
            for i in range(n_agent):
                ep_actions[i] = np.array(ep_actions[i])
                ep_rewards[i] = np.array(ep_rewards[i], dtype=np.float_)
                ep_states[i] = np.array(ep_states[i])
                if LAMBDA < -0.1:
                    targets = discount_rewards(ep_rewards[i], GAMMA)
                    V[i].update(ep_states[i], targets)
                    vs = V[i].get(ep_states[i])
                else:
                    vs = V[i].get(ep_states[i])
                    targets = eligibility_traces(ep_rewards[i], vs, V[i].get(copy.deepcopy([obs[i]])), GAMMA, LAMBDA)
                    V[i].update(ep_states[i], targets)
                ep_advantages = targets - vs
                ep_advantages = (ep_advantages - np.mean(ep_advantages)) / (np.std(ep_advantages) + 0.0000000001)
                Pi[i].update(ep_states[i], ep_actions[i], ep_advantages)

            ep_actions = [[] for _ in range(n_agent)]
            ep_rewards = [[] for _ in range(n_agent)]
            ep_states = [[] for _ in range(n_agent)]

        if render:
            env.render()

    print(i_episode)
    print(score / max_steps, steps)
    print(su)

    print(env.rinfo.flatten())
    env.end_episode()
