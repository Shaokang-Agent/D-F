import torch
import torch.nn as nn
import torch.nn.functional as F

class MAPPO_Actor(nn.Module):
    def __init__(self, args, action_num):
        super(MAPPO_Actor, self).__init__()
        self.args = args
        self.cnn = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(169 * 6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(32, action_num)

    def forward(self, obs):
        batch, step = obs.shape[0], obs.shape[1]
        x = obs.permute(0, 1, 4, 2, 3)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = F.leaky_relu(self.cnn(x))
        x = x.reshape(batch, step, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class MAPPO_Critic(nn.Module):
    def __init__(self, args):
        super(MAPPO_Critic, self).__init__()
        self.args = args
        self.state_shape = 256
        if self.args.env == "Harvest":
            self.fc1 = nn.Linear(16*38*3, self.state_shape)
        else:
            self.fc1 = nn.Linear(25*18*3, self.state_shape)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(self.state_shape, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(32, 32)
        self.fc3.weight.data.normal_(0, 0.1)
        self.q_out = nn.Linear(32, 1)

    def forward(self, s):
        Batch, step = s.shape[0], s.shape[1]
        x = s.reshape(Batch, step, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
