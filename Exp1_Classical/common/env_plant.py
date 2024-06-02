import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from common.utils import RunningMeanStd

class Env():

    def __init__(self, normalize, T=50, one_hot_encoding=False):
        # shared parameters
        self.neighbors_size = 24
        self.T = T
        self.max_steps = 200
        self.n_signal = 4
        self.n_agent = 5
        self.nD = self.n_agent
        self.n_actions = 5
        self.n_episode = 10000
        self.max_u = 0.003
        self.n_neighbors = 4
        if one_hot_encoding:
            self.input_size = 80
            self.state_size = 5*5+3*12*12
        else:
            self.input_size = 30
            self.state_size = 5*5+12*12
        self.GAMMA = 0.98
        self.n_resource = 8
        self.one_hot_encoding = one_hot_encoding

        self.normalize = normalize
        self.compute_neighbors = False
        if normalize:
            self.obs_rms = [RunningMeanStd(shape=self.input_size) for _ in range(self.n_agent)]
            self.state_rms = RunningMeanStd(shape=self.state_size)

        self.requirement = [[2, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 2]]

    def toggle_compute_neighbors(self):
        self.compute_neighbors = True

    def neighbors(self):
        assert self.compute_neighbors
        return self.compute_neighbors_last, self.compute_neighbors_last_index

    def reset(self):
        self.env = np.zeros((12,12))
        self.possession = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.resource=[]
        self.resource_type=[]
        self.ant = []
        self.number = 0
        for i in range(self.n_agent):
            candidate = list(np.random.randint(2, 10, 2))
            while candidate in self.ant:
                candidate = list(np.random.randint(2, 10, 2))
            self.ant.append(candidate)
            self.env[self.ant[i][0]][self.ant[i][1]] = 1

        for i in range(self.n_resource):
            candidate = list(np.random.randint(3, 9, 2))
            while candidate in self.resource:
                candidate = list(np.random.randint(3, 9, 2))
            self.resource.append(candidate)
            self.resource_type.append(np.random.randint(3))

        self.rinfo = np.array([0.] * self.n_agent)
        self.rinfo2 = np.array([0.] * self.n_agent)

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
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        if i != 0 or j != 0:
                            if self.env[self.ant[k][0] + i][self.ant[k][1] + j] == 1:
                                self.compute_neighbors_last_index[k].append(index)
                            index += 1

        h = []
        if self.one_hot_encoding:
            re_map = np.zeros((3, 12, 12))
            for i in range(self.n_resource):
                re_map[self.resource_type[i]][self.resource[i][0]][self.resource[i][1]] = 1
        else:
            re_map = np.zeros((12, 12))
            for i in range(self.n_resource):
                re_map[self.resource[i][0]][self.resource[i][1]] = self.resource_type[i] + 1
        for k in range(self.n_agent):
            state = []
            state.append(self.ant[k][0])
            state.append(self.ant[k][1])
            for i in range(3):
                state.append(self.possession[k][i])
            if self.one_hot_encoding:
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        state.append(re_map[0][self.ant[k][0] + i][self.ant[k][1] + j])
                        state.append(re_map[1][self.ant[k][0] + i][self.ant[k][1] + j])
                        state.append(re_map[2][self.ant[k][0] + i][self.ant[k][1] + j])
            else:
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        state.append(re_map[self.ant[k][0] + i][self.ant[k][1] + j])
            h.append(state)

        if self.normalize:
            for i in range(self.n_agent):
                h[i] = list(self.obs_rms[i].obs_filter(np.array(h[i])))
        return h

    def _get_state(self):
        h = []
        if self.one_hot_encoding:
            re_map = np.zeros((3, 12, 12))
            for i in range(self.n_resource):
                re_map[self.resource_type[i]][self.resource[i][0]][self.resource[i][1]] = 1
        else:
            re_map = np.zeros((12, 12))
            for i in range(self.n_resource):
                re_map[self.resource[i][0]][self.resource[i][1]] = self.resource_type[i] + 1
        for k in range(self.n_agent):
            h.append(self.ant[k][0])
            h.append(self.ant[k][1])
            for i in range(3):
                h.append(self.possession[k][i])
        if self.one_hot_encoding:
            for i in range(12):
                for j in range(12):
                    h.append(re_map[0][i][j])
                    h.append(re_map[1][i][j])
                    h.append(re_map[2][i][j])
        else:
            for i in range(12):
                for j in range(12):
                    h.append(re_map[i][j])
        if self.normalize:
            h = list(self.state_rms.obs_filter(np.array(h)))
        return h

    def step(self, action):
        next_ant = []
        for i in range(self.n_agent):
            x = self.ant[i][0]
            y = self.ant[i][1]
            if action[i] == 0:
                next_ant.append([x, y])
            if action[i] == 1:
                x = x - 1
                if x == 1:
                    next_ant.append([x + 1, y])
                    continue
                if self.env[x][y] != 1:
                    self.env[x][y] = 1
                    next_ant.append([x, y])
                else:
                    next_ant.append([x + 1, y])
            if action[i] == 2:
                x = x + 1
                if x == 10:
                    next_ant.append([x - 1, y])
                    continue
                if self.env[x][y] != 1:
                    self.env[x][y] = 1
                    next_ant.append([x, y])
                else:
                    next_ant.append([x - 1, y])
            if action[i] == 3:
                y = y - 1
                if y == 1:
                    next_ant.append([x, y + 1])
                    continue
                if self.env[x][y] != 1:
                    self.env[x][y] = 1
                    next_ant.append([x, y])
                else:
                    next_ant.append([x, y + 1])
            if action[i] == 4:
                y = y + 1
                if y == 10:
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

        for j in range(self.n_resource):
            for i in range(self.n_agent):
                if (self.ant[i][0] == self.resource[j][0]) & (self.ant[i][1] == self.resource[j][1]):
                    self.resource[j] = np.random.randint(3, 9, 2)
                    self.possession[i][self.resource_type[j]] += 1
                    re[i] += 0.01
                    self.resource_type[j] = np.random.randint(3)
                    self.number += 1
                    break
        self.rinfo2 += re
        re2 = [0] * self.n_agent
        for i in range(self.n_agent):
            x = 1000
            for j in range(3):
                if self.requirement[i][j] == 0:
                    continue
                else:
                    t = int(self.possession[i][j] / self.requirement[i][j])
                    if t < x:
                        x = t
            re[i] += (x)
            re2[i] += (x)
            for j in range(3):
                self.possession[i][j] -= self.requirement[i][j] * x

        self.rinfo += re2
        return self._get_state(), self._get_obs(), re, False

    def render(self):
        for i in range(self.n_agent):
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = self.ant[i][0] + 0.05 * np.cos(theta)
            y = self.ant[i][1] + 0.05 * np.sin(theta)
            plt.plot(x, y)

        for i in range(self.n_resource):
            plt.scatter(self.resource[i][0], self.resource[i][1], color='green')

        plt.axis("equal")
        plt.xlim(-1, 13)
        plt.ylim(-1, 13)
        plt.pause(0.1)
        plt.cla()
