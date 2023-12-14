
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import replay_memory
import episodic_memory
import vae

class EMDQNAgent():
    def __init__(
        self,
        n_actions,
        input_shape,
        qnet,
        device,
        learning_rate = 2e-4,
        reward_decay = 0.99,
        replace_target_iter = 1000,
        memory_size = 10000,
        batch_size = 32,
    ):
        # initialize parameters
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = device
        self.learn_step_counter = 0
        self.memory = replay_memory.Memory(memory_size)
        self.episodic_memory = episodic_memory.EpisodicMemory(memory_size)

        # Network
        self.qnet_eval = qnet(self.input_shape, self.n_actions).to(self.device)
        self.qnet_target = qnet(self.input_shape, self.n_actions).to(self.device)
        self.qnet_target.eval()
        self.vae = vae.Model(3, device=self.device).to(self.device)
        self.optimizer = optim.RMSprop(self.qnet_eval.parameters(), lr=self.lr)

        self.vae.load_state_dict(torch.load("/home/mislab711-50/Desktop/winston/Deep-Q-Network-Memory-3D/rl_core/checkpoint/vae.pth", map_location=self.device))

    def choose_action(self, s, epsilon=0):

        b_s = {}
        for key in s:
            b_s[key] = torch.FloatTensor(np.expand_dims(s[key],0)).to(self.device)

        pose = b_s['key']
        b_f = self.episodic_memory.fatch_features(pose, self.device)
        
        with torch.no_grad():
            actions_value = self.qnet_eval.forward(b_s, b_f)
        if np.random.uniform() > epsilon:   # greedy
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:   # random
            action = np.random.randint(0, self.n_actions)
        return action
    
    def init_episodic_memory(self, s):
        self.episodic_memory.reset()
        key = s['key']
        value = torch.FloatTensor(s['value'] / 255.).unsqueeze(0).to(self.device)
        value = self.vae.gen_latent(value).squeeze(0).detach().cpu()
        self.episodic_memory.push(key, value)

    def store_transition(self, s, a, r, s_, d, store_episodic=False):
        s_k = torch.FloatTensor(np.expand_dims(s['key'], 0)).to(self.device)
        s_k_ = torch.FloatTensor(np.expand_dims(s_['key'], 0)).to(self.device)
        s['K'] = self.episodic_memory.fatch_features(s_k, self.device, True)["key"].detach().cpu()
        s["V"] = self.episodic_memory.fatch_features(s_k, self.device, True)["value"].detach().cpu()
        s_['K'] = self.episodic_memory.fatch_features(s_k_, self.device, True)["key"].detach().cpu()
        s_["V"] = self.episodic_memory.fatch_features(s_k_, self.device, True)["value"].detach().cpu()
        self.memory.push(s, a, r, s_, d)
        if store_episodic:
            key = s['key']
            value = torch.FloatTensor(s['value'] / 255.).unsqueeze(0).to(self.device)
            value = self.vae.gen_latent(value).squeeze(0).detach().cpu()
            self.episodic_memory.push(key, value)

    def get_closest_pose(self, pose):
        return self.episodic_memory.get_closest_pose(pose)
    
    def get_all_poses(self):
        return self.episodic_memory.get_all_poses()

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.qnet_target.load_state_dict(self.qnet_eval.state_dict())

        # sample batch memory from all memory
        b_s, b_a, b_r, b_s_, b_d = self.memory.sample_torch(self.batch_size, self.device) 
        b_f = {"key": b_s['K'].squeeze(1), "value": b_s['V'].squeeze(1)}
        b_f_ = {"key": b_s_['K'].squeeze(1), "value": b_s_['V'].squeeze(1)}

        q_curr_eval = self.qnet_eval(b_s, b_f)
        # (32, 3), (32, 1)
        q_curr_eval_action = q_curr_eval.gather(1, b_a)
        q_next_target = self.qnet_target(b_s_, b_f_).detach()

        #next_state_values = q_next_target.max(1)[0].view(-1, 1)   # DQN
        q_next_eval = self.qnet_eval(b_s_, b_f_).detach()
        next_state_values = q_next_target.gather(1, q_next_eval.max(1)[1].unsqueeze(1))   # DDQN

        q_curr_recur = b_r + (1-b_d) * self.gamma * next_state_values
        self.loss = F.smooth_l1_loss(q_curr_eval_action, q_curr_recur).mean()
        
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1

        return float(self.loss.detach().cpu().numpy())
    
    def save_model(self, path, step):
        if not os.path.exists(path):
            os.makedirs(path)
        qnet_path = os.path.join(path, f"{step}_qnet.pt")
        torch.save(self.qnet_eval.state_dict(), qnet_path)
    
    def load_model(self, path, step):
        qnet_path = os.path.join(path, f"{step}_qnet.pt")
        self.qnet_eval.load_state_dict(torch.load(qnet_path, map_location=self.device))
        self.qnet_target.load_state_dict(torch.load(qnet_path, map_location=self.device))
