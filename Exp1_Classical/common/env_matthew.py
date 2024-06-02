import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from common.utils import RunningMeanStd

class Env():
    # resource_type in (all, rr, nrr)
    def __init__(self, normalize, resource_type, obs3neighbors):
        # shared parameters
        self.n_neighbors = 3
        self.neighbors_size = self.n_neighbors
        self.T = 50
        self.max_steps = 200
        self.n_actions = 5
        self.n_signal = 4
        self.n_agent = 10
        self.nD = self.n_agent
        self.n_resource = 3
        self.n_episode = 10000
        self.max_u = 0.15
        self.GAMMA = 0.98
        if obs3neighbors:
            self.input_size = 6 + self.n_neighbors*4
        else:
            self.input_size = 6
        self.state_size = 130
        if resource_type != 'all':
            self.input_size += 1
            if resource_type == 'rr':
                self.n_episode = 20000

        self.normalize = normalize
        self.resource_type = resource_type
        self.obs3neighbors = obs3neighbors
        self.compute_neighbors = self.obs3neighbors
        self.compute_neighbors_last = np.zeros((self.n_agent, self.n_neighbors), dtype=int)
        self.compute_neighbors_last_index = [[i for i in range(self.n_neighbors)] for _ in range(self.n_agent)]
        if normalize:
            self.obs_rms = [RunningMeanStd(shape=self.input_size) for _ in range(self.n_agent)]
            self.state_rms = RunningMeanStd(shape=self.state_size)

    def toggle_compute_neighbors(self):
        self.compute_neighbors = True

    def neighbors(self):
        assert self.compute_neighbors
        return (self.compute_neighbors_last, self.compute_neighbors_last_index)

    def reset(self):
        self.resource = []
        for i in range(self.n_resource):
            self.resource.append(np.random.rand(2))
        self.ant = []
        self.size = []
        self.speed = []
        for i in range(self.n_agent):
            self.ant.append(np.random.rand(2))
            self.size.append(0.01 + np.random.rand() * 0.04)
            self.speed.append(0.01 + self.size[i])
        self.rinfo = np.array([[0.] * self.n_resource] * self.n_agent)

        return self._get_state(), self._get_obs()

    def _get_obs(self):
        if self.compute_neighbors:
            distances = distance_matrix(self.ant, self.ant, p=2)
            for i in range(len(self.ant)):
                distances[i,i] = float('+inf')
            distances = distances.argsort()[:,:self.n_neighbors]
            self.compute_neighbors_last = distances

        state=[]
        for i in range(self.n_agent):
            h=[]
            h.append(self.ant[i][0])
            h.append(self.ant[i][1])
            h.append(self.size[i])
            h.append(self.speed[i])
            if self.obs3neighbors:
                for j in range(self.n_neighbors):
                    h.append(self.ant[distances[i][j]][0])
                    h.append(self.ant[distances[i][j]][1])
                    h.append(self.size[distances[i][j]])
                    h.append(self.speed[distances[i][j]])
            j=0
            mi = 10
            last_resource = False
            for k in range(len(self.resource)):
                t = (self.resource[k][0] - self.ant[i][0]) ** 2 + (self.resource[k][1] - self.ant[i][1]) ** 2
                condition = t < mi
                if self.resource_type != 'all':
                    condition = (condition and (i <= 1 or k != (self.n_resource-1)))

                if condition:
                    j = k
                    mi = t
                    last_resource = (k == self.n_resource-1)
            h.append(self.resource[j][0])
            h.append(self.resource[j][1])
            if self.resource_type != 'all':
                h.append(float(last_resource))
            state.append(h)

        if self.normalize:
            for i in range(self.n_agent):
                state[i] = list(self.obs_rms[i].obs_filter(np.array(state[i])))

        return state

    def _get_state(self):
        state=[]
        for i in range(self.n_agent):
            state.append(self.ant[i][0])
            state.append(self.ant[i][1])
            state.append(self.size[i])
            state.append(self.speed[i])
            for k in range(len(self.resource)):
                state.append(self.resource[k][0] - self.ant[i][0])
                state.append(self.resource[k][1] - self.ant[i][1])
                state.append((self.resource[k][0]-self.ant[i][0])**2+(self.resource[k][1]-self.ant[i][1])**2)

        if self.normalize:
            state = list(self.state_rms.obs_filter(np.array(state)))
        return state


    def step(self, action):
        re=[0.]*self.n_agent
        rinfo_=np.array([[0]*self.n_resource]*self.n_agent)
        for i in range(self.n_agent):
            if action[i]==1:
                self.ant[i][0]-=self.speed[i]
                if self.ant[i][0]<0:
                    self.ant[i][0]=0
            if action[i]==2:
                self.ant[i][0]+=self.speed[i]
                if self.ant[i][0]>1:
                    self.ant[i][0]=1
            if action[i]==3:
                self.ant[i][1]-=self.speed[i]
                if self.ant[i][1]<0:
                    self.ant[i][1]=0
            if action[i]==4:
                self.ant[i][1]+=self.speed[i]
                if self.ant[i][1]>1:
                    self.ant[i][1]=1
        for i in range(self.n_resource):
            for j in range(self.n_agent):
                condition = (self.resource[i][0]-self.ant[j][0])**2+(self.resource[i][1]-self.ant[j][1])**2<self.size[j]**2
                if self.resource_type == 'rr':
                    condition = (condition and (j <= 1 or i != (self.n_resource-1)))
                if condition:
                    re[j]=1
                    rinfo_[j][i]=1
                    self.resource[i]=np.random.rand(2)
                    self.size[j]=min(self.size[j]+0.005,0.15)
                    self.speed[j]=0.01+self.size[j]
                    break

        self.rinfo += rinfo_
        return self._get_state(), self._get_obs(), re, False

    def render(self):
        for i in range(self.n_agent):
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = self.ant[i][0] + self.size[i] * np.cos(theta)
            y = self.ant[i][1] + self.size[i] * np.sin(theta)
            plt.plot(x, y)
        for i in range(self.n_resource):
            plt.scatter(self.resource[i][0], self.resource[i][1], color='green')
        plt.axis("off")
        plt.axis("equal")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ion()
        plt.pause(0.1)
        plt.close()
