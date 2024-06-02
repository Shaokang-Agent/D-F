import numpy as np
# from scipy.stats import norm


# adapted from OpenAI baselines/common/running_mean_std.py
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.clipob = 10.
        self.epsilon = 1e-8

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def obs_filter(self, obs):
        self.update(obs)
        obs = np.clip((obs - self.mean) / np.sqrt(self.var + self.epsilon), -self.clipob, self.clipob)
        return obs

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def discount_rewards(rewards,gamma):
    running_total = 0
    discounted = np.zeros_like(rewards)
    for r in reversed(range(len(rewards))):
        running_total = running_total *gamma + rewards[r]
        discounted[r] = running_total
    return discounted

def eligibility_traces(rewards, vf_values, last_value, gamma, lam):
    last_gae_lam = 0
    discounted = np.zeros_like(rewards)
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            nextvalues = last_value
        else:
            nextvalues = vf_values[step + 1]
        delta = rewards[step] + gamma * nextvalues - vf_values[step]
        discounted[step] = last_gae_lam = delta + gamma * lam * last_gae_lam
    return discounted + vf_values

def eligibility_traces_vtraces(rewards, vf_values, last_value, proba, gamma, lam):
    last_gae_lam = 0
    discounted = np.zeros_like(rewards)
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            nextvalues = last_value
        else:
            nextvalues = vf_values[step + 1]
        delta = rewards[step] + gamma * nextvalues - vf_values[step]
        discounted[step] = last_gae_lam = min(1, proba[step])*delta + gamma * min(1, proba[step]) * lam * last_gae_lam
    return discounted + vf_values

def eligibility_traces_mask(rewards, vf_values, last_value, gamma, lam, mask):
    last_gae_lam = 0
    discounted = np.zeros_like(rewards)
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            nextvalues = last_value
        else:
            nextvalues = vf_values[step + 1]
        delta = rewards[step] + gamma * nextvalues - vf_values[step]
        discounted[step] = last_gae_lam = delta + gamma * lam * last_gae_lam
        if not mask[step]:
            last_gae_lam = 0
    return discounted + vf_values

def make_env(config, normalize_inputs):
    env_name=config['agent']['env_name']
    if env_name == 'matthew':
        from common.env_matthew import Env
        env = Env(normalize_inputs, config['agent']['resource'], str2bool(config['agent']['matthew_neighbors_obs']))
    elif env_name == 'plant':
        from common.env_plant import Env
        env = Env(normalize_inputs, int(config['agent']['plant_T']), str2bool(config['agent']['plant_ohe']))
    elif env_name == 'job':
        from common.env_job import Env
        env = Env(normalize_inputs, config['agent']['resource'])
    else:
        raise("Not supported")
    return env

def default_config():
    default_tensorflow()
    import configparser
    config = configparser.ConfigParser()
    config.add_section('agent')
    config['agent']['env_name'] = 'job'
    config['agent']['resource'] = 'all' # in (all, rr, nrr)

    # env
    config['agent']['matthew_neighbors_obs'] = 'false'
    config['agent']['plant_T'] = '50'
    config['agent']['plant_ohe'] = 'true'

    config['agent']['lambda'] = '0.97'
    config['agent']['cliprange'] = '0.1'
    config['agent']['entropy_coeff'] = '0.03'
    config['agent']['lr_actor'] = '0.0001'
    config['agent']['lr_critic'] = '0.001'
    config['agent']['stdev'] = '0.2'
    config['agent']['if_tanh'] = 'false'

    # fixed strategy
    config['agent']['fixed_steps'] = '1'
    config['agent']['cfixed_val'] = '1'

    # ggi
    config['agent']['ggi_constant'] = '2'
    config['agent']['ggi_type'] = 'xpowerminusn' #xpowerminusn, dedicated

    # alpha-fairness
    config['agent']['alpha_fairness'] = '0.9'

    # ggicom
    config['agent']['twophase_proportion'] = '0.5'

    #fen
    config['agent']['meta_skip_etrace'] = 'true' #should be kept true because of the reward form
    config['agent']['fen_communication_round'] = '10'

    config.read('config.ini')
    return config

def default_tensorflow():
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True,
                            device_count={'CPU': 1})
    session = tf.Session(config=config)
    KTF.set_session(session)

def str2bool(v):
    return v.lower() in ("yes", "true", "1")

def argmin_rand(v):
    index = v.argsort()
    all_min=[index[0]]
    for i in range(1, len(v)):
        if v[index[i]] <= v[index[0]]:
            all_min.append(index[i])
        else:
            break

    return all_min[np.random.randint(len(all_min))]

def get_omega(config, n_agent):
    if config['agent']['ggi_type'] == 'xpowerminusn':
        ggi_constant = float(config['agent']['ggi_constant'])
        return np.array([1 / (ggi_constant ** i) for i in range(n_agent)])
    elif config['agent']['ggi_type'] == 'xpowerminusnrescaled':
        ggi_constant = float(config['agent']['ggi_constant'])
        omega = np.array([1 / (ggi_constant ** i) for i in range(n_agent)])
        omega = omega/omega.sum()
        return omega
    elif config['agent']['ggi_type'] == 'dedicated':
        return np.array([0.6, 2/15 + 0.01,  2/15,  2/15 - 0.01])[:n_agent]
    elif config['agent']['ggi_type'] == 'min':
        omega = np.zeros((n_agent))
        omega[0] = 1
        return omega
    elif config['agent']['ggi_type'] == 'minlin':
        ggi_constant = float(config['agent']['ggi_constant'])
        linw = np.array([1. / float(i+1) for i in range(n_agent-1)])
        omega = np.zeros(n_agent)
        omega[0] = ggi_constant
        omega[1:] = ((1-ggi_constant)/linw.sum())*linw
        return omega
    else:
        raise("not supported")

def get_more_obs_com(add_own_value, neighbors_, average_jpi, i, more_obs_size):
    (neigh, index) = neighbors_[0][i], neighbors_[1][i]
    neigh = neigh[0:len(index)]
    more_obs = np.zeros((more_obs_size))
    more_obs[index] = average_jpi[neigh]
    if add_own_value:
        more_obs[-1] = average_jpi[i]
    more_obs = (more_obs - np.mean(more_obs)) / (np.std(more_obs) + 0.0000000001)
    return more_obs

def get_more_obs_comac(neighbors_, ep_actions, i, more_obs_size, n_action):
    (neigh, index) = neighbors_[0][i], neighbors_[1][i]
    neigh = neigh[0:len(index)]
    more_obs = np.zeros((more_obs_size))
    if len(ep_actions[i]) > 0:
        index = np.array(index) * n_action
        nindex = index + n_action
        for i in range(len(index)):
            more_obs[index[i]:nindex[i]] = ep_actions[neigh[i]][-1]
        more_obs = (more_obs - np.mean(more_obs)) / (np.std(more_obs) + 0.0000000001)
    return more_obs


