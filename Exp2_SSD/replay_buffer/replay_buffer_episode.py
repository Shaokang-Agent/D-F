import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, args):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.args = args
        if self.args.env == "Harvest":
            self._storage = {
                            's': np.empty([self.args.buffer_size, self.args.num_steps, 16, 38, 3]),
                            's_next': np.empty([self.args.buffer_size, self.args.num_steps, 16, 38, 3]),
                            'o': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 15, 15, 3]),
                            'u': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1]),
                            'u_probability': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1]),
                            'u_probability_all': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 8]),
                            'r': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents,  1]),
                            'o_next': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 15, 15, 3]),
                            'u_next': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1]),
                            'terminate': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1])
                            }
        else:
            self._storage = {
                            's': np.empty([self.args.buffer_size, self.args.num_steps, 25, 18, 3]),
                            's_next': np.empty([self.args.buffer_size, self.args.num_steps, 25, 18, 3]),
                            'o': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 15, 15, 3]),
                            'u': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1]),
                            'u_probability': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1]),
                            'u_probability_all': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 9]),
                            'r': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents,  1]),
                            'o_next': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 15, 15, 3]),
                            'u_next': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1]),
                            'terminate': np.empty([self.args.buffer_size, self.args.num_steps, self.args.num_agents, 1])
                            }
        self.actual_length = 0
        self.index = 0

    def __len__(self):
        return self.actual_length

    def add(self, episode_data):
        self.actual_length = min(self.args.buffer_size, self.actual_length+1)
        if "s" in episode_data.keys():
            self._storage['s'][self.index] = np.array(episode_data['s'])
        if "s_next" in episode_data.keys():
            self._storage['s_next'][self.index] = np.array(episode_data['s_next'])
        if "o" in episode_data.keys():
            self._storage['o'][self.index] = np.array(episode_data['o'])
        if "u" in episode_data.keys():
            self._storage['u'][self.index] = np.expand_dims(np.array(episode_data['u']), axis=2)
        if "r" in episode_data.keys():
            self._storage['r'][self.index] = np.expand_dims(np.array(episode_data['r']), axis=2)
        if "o_next" in episode_data.keys():
            self._storage['o_next'][self.index] = np.array(episode_data['o_next'])
        if "terminate" in episode_data.keys():
            self._storage['terminate'][self.index] = np.expand_dims(np.array(episode_data['terminate']), axis=2)
        if "u_next" in episode_data.keys():
            self._storage['u_next'][self.index] = np.expand_dims(np.array(episode_data['u_next']), axis=2)
        if "u_probability" in episode_data.keys():
            self._storage['u_probability'][self.index] = np.expand_dims(np.array(episode_data['u_probability']), axis=2)
        self.index = (self.index + 1) % self.args.buffer_size

    def _encode_sample(self, idxes):
        temp_buffer = {}
        for key in self._storage.keys():
            temp_buffer[key] = self._storage[key][idxes]
        return temp_buffer

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        sample_list = [i for i in range(self.actual_length)]
        if self.actual_length < 1000:
            idxes = random.sample(sample_list, batch_size)
        else:
            idxes = [random.randint(0, self.actual_length-1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
