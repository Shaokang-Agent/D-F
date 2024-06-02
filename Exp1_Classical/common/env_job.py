import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from copy import deepcopy
from common.utils import RunningMeanStd

class Env():

    def __init__(self, normalize, resource_type):
        # shared parameters
        self.neighbors_size = 8
        self.T = 25
        self.max_steps = 200
        self.n_signal = 4
        self.resource_type = resource_type
        if self.resource_type != 'all':
            self.n_agent = 3
            self.n_actions = 2
            self.n_episode = 10000
            self.max_u = 1/3
            self.n_neighbors = 2
        else:
            self.n_agent = 4
            self.n_actions = 5
            self.n_episode = 10000
            self.max_u = 0.25
            self.n_neighbors = 3
        self.input_size = 13
        self.state_size = 74
        self.nD = self.n_agent
        self.GAMMA = 0.98

        self.normalize = normalize
        self.compute_neighbors = False
        if normalize:
            self.obs_rms = [RunningMeanStd(shape=self.input_size) for _ in range(self.n_agent)]
            self.state_rms = RunningMeanStd(shape=self.state_size)

    def toggle_compute_neighbors(self):
        self.compute_neighbors = True

    def neighbors(self):
        assert self.compute_neighbors
        return self.compute_neighbors_last, self.compute_neighbors_last_index

    def reset(self):
        self.env = np.zeros((8, 8))
        self.target = np.random.randint(2, 5, 2)
        self.ant = []
        for i in range(self.n_agent):
            candidate = list(np.random.randint(1, 6, 2))
            while candidate in self.ant:
                candidate = list(np.random.randint(1, 6, 2))
            self.ant.append(candidate)
            self.env[self.ant[i][0]][self.ant[i][1]] = 1
        self.rinfo = np.array([0.] * self.n_agent)

        return self._get_state(), self._get_obs()

    def _get_obs(self):
        if self.compute_neighbors:
            distances = distance_matrix(self.ant, self.ant, p=float('+inf'))
            distances = np.array(distances).astype(np.float)
            for i in range(len(self.ant)):
                distances[i,i]=float('+inf')
            distances = np.argsort(distances)[:,:self.n_neighbors]
            self.compute_neighbors_last = distances

            self.compute_neighbors_last_index=[[] for _ in range(self.n_agent)]
            for k in range(len(self.ant)):
                index = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i != 0 or j != 0:
                            if self.env[self.ant[k][0] + i][self.ant[k][1] + j] == 1:
                                self.compute_neighbors_last_index[k].append(index)
                            index += 1

        h = []
        for k in range(self.n_agent):
            state = []
            state.append(self.ant[k][0])
            state.append(self.ant[k][1])
            state.append(self.target[0] - self.ant[k][0])
            state.append(self.target[1] - self.ant[k][1])
            for i in range(-1, 2):
                for j in range(-1, 2):
                    state.append(self.env[self.ant[k][0] + i][self.ant[k][1] + j])
            h.append(state)

        if self.normalize:
            for i in range(self.n_agent):
                h[i] = list(self.obs_rms[i].obs_filter(np.array(h[i])))
        return h

    def _get_state(self):
        state=[]
        for k in range(self.n_agent):
            state.append(self.ant[k][0])
            state.append(self.ant[k][1])
        state.append(self.target[0])
        state.append(self.target[1])
        for i in range(8):
            for j in range(8):
                state.append(self.env[i][j])
        if self.normalize:
            state = list(self.state_rms.obs_filter(np.array(state)))
        return state


    def step(self, action):
        if self.resource_type != 'all':
            action = list(deepcopy(action))
            for i in range(self.n_agent):
                if action[i] != 0:
                    if self.target[0] < self.ant[i][0]:
                        action[i]=1
                    elif self.target[0] > self.ant[i][0]:
                        action[i]=2
                    elif self.target[1] < self.ant[i][1]:
                        action[i]=3
                    elif self.target[1] > self.ant[i][1]:
                        action[i]=4
                    else:
                        action[i]=0
                else:
                    action[i] = np.random.randint(1, 5)

        next_ant = []
        for i in range(self.n_agent):
            x = self.ant[i][0]
            y = self.ant[i][1]
            if action[i] == 0:
                next_ant.append([x, y])
            if action[i] == 1:
                x = x - 1
                if x == 0:
                    next_ant.append([x + 1, y])
                    continue
                if self.env[x][y] != 1:
                    self.env[x][y] = 1
                    next_ant.append([x, y])
                else:
                    next_ant.append([x + 1, y])
            if action[i] == 2:
                x = x + 1
                if x == 6:
                    next_ant.append([x - 1, y])
                    continue
                if self.env[x][y] != 1:
                    self.env[x][y] = 1
                    next_ant.append([x, y])
                else:
                    next_ant.append([x - 1, y])
            if action[i] == 3:
                y = y - 1
                if y == 0:
                    next_ant.append([x, y + 1])
                    continue
                if self.env[x][y] != 1:
                    self.env[x][y] = 1
                    next_ant.append([x, y])
                else:
                    next_ant.append([x, y + 1])
            if action[i] == 4:
                y = y + 1
                if y == 6:
                    next_ant.append([x, y - 1])
                    continue
                if self.env[x][y] != 1:
                    self.env[x][y] = 1
                    next_ant.append([x, y])
                else:
                    next_ant.append([x, y - 1])
        self.ant = next_ant
        self.env *= 0
        re = [0.] * self.n_agent
        for i in range(self.n_agent):
            self.env[self.ant[i][0]][self.ant[i][1]] = 1
            if (self.ant[i][0] == self.target[0]) & (self.ant[i][1] == self.target[1]):
                re[i] = 1

        self.rinfo += re
        return self._get_state(), self._get_obs(), re, False

    def render(self):
        for i in range(self.n_agent):
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = self.ant[i][0] + 0.05 * np.cos(theta)
            y = self.ant[i][1] + 0.05 * np.sin(theta)
            plt.plot(x, y)

        plt.scatter(self.target[0], self.target[1], color='green')
        plt.axis("equal")
        plt.xlim(-1, 7)
        plt.ylim(-1, 7)
        plt.pause(0.1)
        plt.cla()
