import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, args, action_num):
        super(Actor, self).__init__()
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


class Critic(nn.Module):
    def __init__(self, args, action_num):
        super(Critic, self).__init__()
        self.agent_num = args.num_agents
        self.action_num = action_num
        self.args = args
        self.cnn = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.cnn.weight.data.normal_(0, 0.1)
        if self.args.env == "Harvest":
            self.fc1 = nn.Linear(14*36*6, 128)
        else:
            self.fc1 = nn.Linear(23*16*6, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        self.fc3 = nn.Linear(32, 32)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc4 = nn.Linear(self.action_num*self.agent_num, 8)
        self.fc4.weight.data.normal_(0, 0.1)
        self.q_out = nn.Linear(40, 1)

    def forward(self, s, a):
        Batch, seq_len = s.shape[0], s.shape[1]
        x = s.permute(0,1,4,3,2)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = F.leaky_relu(self.cnn(x))
        x = x.reshape(Batch,seq_len, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.out(x)
        return x
