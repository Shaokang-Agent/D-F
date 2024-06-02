import torch.nn as nn
import torch
import torch.nn.functional as F

class QNet(nn.Module):
    """docstring for Net"""
    def __init__(self, args, action_num):
        super(QNet, self).__init__()
        self.args = args
        self.action_num = action_num
        self.cnn = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.cnn.weight.data.normal_(0, 0.1)
        self.fc1 = nn.Linear(169*6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32, action_num)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, obs):
        Batch, seq_len = obs.shape[0], obs.shape[1]
        x = obs.permute(0,1,4,3,2)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = F.leaky_relu(self.cnn(x))
        x = x.reshape(Batch,seq_len, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.out(x)
        return x

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        self.state_shape = 256
        if self.args.env == "Harvest":
            self.fc1 = nn.Linear(16*38*3, self.state_shape)
        else:
            self.fc1 = nn.Linear(25*18*3, self.state_shape)

        if args.two_hyper_layers:
            self.hyper_w1_1 = nn.Linear(self.state_shape, args.hyper_hidden_dim)
            self.hyper_w1_1.weight.data.normal_(0, 0.1)
            self.hyper_w1_2 = nn.Linear(args.hyper_hidden_dim, self.args.num_agents*args.qmix_hidden_dim)
            self.hyper_w1_2.weight.data.normal_(0, 0.1)
            self.hyper_w1 = nn.Sequential(self.hyper_w1_1,
                                          nn.LeakyReLU(),
                                          self.hyper_w1_2)

            self.hyper_w2_1 = nn.Linear(self.state_shape, args.hyper_hidden_dim)
            self.hyper_w2_1.weight.data.normal_(0, 0.1)
            self.hyper_w2_2 = nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim)
            self.hyper_w2_2.weight.data.normal_(0, 0.1)
            self.hyper_w2 = nn.Sequential(self.hyper_w2_1,
                                          nn.LeakyReLU(),
                                          self.hyper_w2_2)
        else:
            self.hyper_w1 = nn.Linear(self.state_shape, self.args.num_agents*args.qmix_hidden_dim)
            self.hyper_w1.weight.data.normal_(0, 0.1)
            self.hyper_w2 = nn.Linear(self.state_shape, args.qmix_hidden_dim)
            self.hyper_w2.weight.data.normal_(0, 0.1)

        self.hyper_b1 = nn.Linear(self.state_shape, args.qmix_hidden_dim)
        self.hyper_b1.weight.data.normal_(0, 0.1)

        self.hyper_b2_1 = nn.Linear(self.state_shape, args.qmix_hidden_dim)
        self.hyper_b2_1.weight.data.normal_(0, 0.1)
        self.hyper_b2_2 = nn.Linear(args.qmix_hidden_dim, 1)
        self.hyper_b2_2.weight.data.normal_(0, 0.1)
        self.hyper_b2 = nn.Sequential(self.hyper_b2_1,
                                     nn.LeakyReLU(),
                                     self.hyper_b2_2)

    def forward(self, q_values, s):
        Batch, len_step = s.shape[0], s.shape[1]
        x = s.reshape(Batch, len_step, -1)
        x = F.leaky_relu(self.fc1(x))
        q_values = q_values.reshape(Batch,len_step, -1, self.args.num_agents)
        w1 = torch.abs(self.hyper_w1(x))
        b1 = self.hyper_b1(x)
        w1 = w1.view(Batch,len_step, self.args.num_agents, self.args.qmix_hidden_dim)
        b1 = b1.view(Batch,len_step, 1, self.args.qmix_hidden_dim)

        hidden = F.elu(torch.matmul(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(x)).unsqueeze(dim=3)
        b2 = self.hyper_b2(x)
        q_total = torch.matmul(hidden, w2).squeeze(dim=2) + b2
        q_total = q_total.reshape(Batch, len_step, 1)
        return q_total
