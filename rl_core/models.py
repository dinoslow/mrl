import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cell_net import CellNet

class QNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        conv_out = self.conv(obs)
        advantage = self.fc_advantage(conv_out)
        value = self.fc_value(conv_out)
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q
    
class QNetSpatial(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetSpatial, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.zeros(1, *shape))
        o = torch.cat((conv_out, conv_out), dim=1)
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        top_down = s["top_down"]
        obs_conv_out = self.conv(obs)
        top_down_conv_out = self.conv(top_down)
        # breakpoint()
        conv_out = torch.cat((obs_conv_out, top_down_conv_out), dim=1)
        advantage = self.fc_advantage(conv_out)
        value = self.fc_value(conv_out)
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q

class QNetPositionSpatial(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetPositionSpatial, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.pos_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.zeros(1, *shape))
        pos_conv_out = self.pos_conv(torch.zeros(1, 1, 4, 4))
        o = torch.cat((conv_out, conv_out, pos_conv_out), dim=1)
        return int(np.prod(o.size()))

    def forward(self, s):
        obs = s["obs"]
        top_down = s["top_down"]
        pos = s['position'].unsqueeze(1)
        obs_conv_out = self.conv(obs)
        top_down_conv_out = self.conv(top_down)
        pos_conv_out = self.pos_conv(pos)
        conv_out = torch.cat((obs_conv_out, top_down_conv_out, pos_conv_out), dim=1)
        advantage = self.fc_advantage(conv_out)
        value = self.fc_value(conv_out)
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q

class QNetRNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetRNN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.flat = nn.Flatten(start_dim=-3, end_dim=-1)
        self.rnn = nn.LSTM(input_size=conv_out_size, hidden_size=256, batch_first=True)
        self.fc_advantage = nn.Linear(512, n_actions)
        self.fc_value = nn.Linear(512, 1)
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s, hidden=None):
        obs = s["obs"]
        batch_size, seq_size = obs.shape[0], obs.shape[1]
        conv_out = self.conv(obs.reshape(-1,*self.input_shape))
        feature_flat = self.flat(conv_out)
        feature_seq = feature_flat.reshape(batch_size, seq_size, feature_flat.shape[-1])
        
        if hidden is not None:
            out, hidden = self.rnn(feature_seq, hidden)
        else:
            out, hidden = self.rnn(feature_seq)

        advantage = self.fc_advantage(out)
        value = self.fc_value(out)
        q = value + advantage - advantage.mean(2, keepdim=True)
        return q, hidden

class QNetRNNDebug(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetRNNDebug, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.flat = nn.Flatten(start_dim=-3, end_dim=-1)
        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s, hidden=None):
        obs = s["obs"]
        batch_size, seq_size = obs.shape[0], obs.shape[1]
        conv_out = self.conv(obs.reshape(-1,*self.input_shape))
        conv_out = self.flat(conv_out)
        conv_out = conv_out.reshape(batch_size, seq_size, conv_out.shape[-1])
        advantage = self.fc_advantage(conv_out)
        value = self.fc_value(conv_out)
        q = value + advantage - advantage.mean(2, keepdim=True)
        
        return q, hidden

class QNetRNNCell(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size=256):
        super(QNetRNNCell, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.flat = nn.Flatten(start_dim=-3, end_dim=-1)
        self.rnn = nn.GRUCell(input_size=conv_out_size, hidden_size=hidden_size)
        self.fc_advantage = nn.Linear(hidden_size, n_actions)
        self.fc_value = nn.Linear(hidden_size, 1)
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s, hidden=None):
        obs = s["obs"]
        batch_size, seq_size = obs.shape[0], obs.shape[1]
        conv_out = self.conv(obs.reshape(-1,*self.input_shape))
        feature_flat = self.flat(conv_out)
        feature_seq = feature_flat.reshape(batch_size, seq_size, feature_flat.shape[-1])
        
        if hidden is None:
            hidden = torch.zeros(obs.shape[0], self.hidden_size).to(obs.get_device())

        hidden_list = []
        for i in range(obs.shape[1]):
            hidden = self.rnn(feature_seq[:,i,...], hidden)
            hidden_list.append(hidden.unsqueeze(1))

        out = torch.cat(hidden_list, 1)
        advantage = self.fc_advantage(out)
        value = self.fc_value(out)
        q = value + advantage - advantage.mean(2, keepdim=True)
        return q, hidden

class QNetFRMQN(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size=256, emb_size=64):
        super(QNetFRMQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.flat = nn.Flatten(start_dim=-3, end_dim=-1)

        # Memory Module
        self.weights_key = nn.Linear(conv_out_size, self.hidden_size, bias=False)
        self.weights_val = nn.Linear(conv_out_size, self.hidden_size, bias=False)
        
        # Context Recurrent Module
        self.rnn = nn.GRUCell(input_size=conv_out_size+self.hidden_size, hidden_size=hidden_size)
        
        # Duel Network 
        self.weights_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_advantage = nn.Linear(hidden_size, n_actions)
        self.fc_value = nn.Linear(hidden_size, 1)
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, s, hidden_info=None):
        if hidden_info is None:
            hidden, eps_memory, memory_read = None, None, None
        else:
            hidden, eps_memory, memory_read = hidden_info["hidden"], hidden_info["eps_memory"], hidden_info["memory_read"]
        
        # Convolutional Feature Extraction
        obs = s["obs"]
        batch_size, seq_size = obs.shape[0], obs.shape[1]
        conv_out = self.conv(obs.reshape(-1,*self.input_shape))
        feature_flat = self.flat(conv_out)
        feature_seq = feature_flat.reshape(batch_size, seq_size, feature_flat.shape[-1])

        # Memory Extraction
        m_key = self.weights_key(feature_seq) # (batch, seq_size, hidden_size)
        m_val = self.weights_val(feature_seq) # (batch, seq_size, hidden_size)

        # Memory Initialization
        if hidden is None:
            hidden = torch.zeros(obs.shape[0], self.hidden_size).to(obs.get_device())
        if memory_read is None:
            memory_read = torch.zeros(obs.shape[0], self.hidden_size).to(obs.get_device())
        if eps_memory == None:
            start_idx = 0
            eps_memory = {"key":m_key, "val":m_val}
        else:
            # Memory Writing
            start_idx = eps_memory["key"].shape[1]
            eps_memory["key"] = torch.cat((eps_memory["key"], m_key), 1)
            eps_memory["val"] = torch.cat((eps_memory["val"], m_val), 1)

        hidden_list = []
        memory_read_list = []
        # Memory Step Iteration
        for i in range(obs.shape[1]):
            # Context RNN
            input = torch.cat((feature_seq[:,i,...], memory_read), 1)
            hidden = self.rnn(input, hidden) # (batch, hidden_size)
            hidden_list.append(hidden.unsqueeze(1))
            
            ## Memory Reading
            eps_memory_key_temp = eps_memory["key"][:, :start_idx+i+1:, :]
            eps_memory_val_temp = eps_memory["val"][:, :start_idx+i+1:, :]
            p = torch.bmm(eps_memory_key_temp, hidden.unsqueeze(2))
            p = torch.nn.Softmax(dim=1)(p) # (batch, seq_size, 1)
            memory_read = torch.bmm(eps_memory_val_temp.permute(0,2,1), p).squeeze(-1)
            memory_read_list.append(memory_read.unsqueeze(1))

        h = torch.cat(hidden_list, 1)
        o = torch.cat(memory_read_list, 1)
        g = torch.relu(self.weights_hidden(h) + o)
        advantage = self.fc_advantage(g)
        value = self.fc_value(g)
        q = value + advantage - advantage.mean(2, keepdim=True)
        return q, {"hidden":hidden, "eps_memory":eps_memory, "memory_read":memory_read}
    
class EMQNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(EMQNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )

        self.cnet = CellNet(256, 256, 32)

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.fc_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.zeros(1, *shape))
        # o = torch.cat((conv_out, conv_out.repeat(1,2,1,1)), dim=1)
        o = torch.cat((conv_out, conv_out), dim=1)
        return int(np.prod(o.size()))

    def forward(self, s, features):
        obs = s["obs"]
        keys = features["key"] #(b, m, 12)
        values = features["value"] #(b, m, 256)
        # (b, 256, 1, 1)
        cell_feature = self.cnet(values, keys).unsqueeze(2).unsqueeze(3)
        # (b, 256, 1, 1)
        conv_out = self.conv(obs)
        # (b, 512, 1, 1)
        fuse_feature = torch.cat([conv_out, cell_feature],1)

        advantage = self.fc_advantage(fuse_feature)
        value = self.fc_value(fuse_feature)
        q = value + advantage - advantage.mean(1, keepdim=True)
        return q