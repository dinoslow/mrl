import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import replay_memory_episodic as replay_memory

class DRQNAgent():
    def __init__(
        self,
        n_actions,
        input_shape,
        qnet,
        device,
        learning_rate = 2e-4,
        reward_decay = 0.99,
        replace_target_iter = 1000,
        memory_size = 1000,
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
        self.memory = replay_memory.Memory(memory_size=memory_size)

        # Network
        self.qnet_eval = qnet(self.input_shape, self.n_actions).to(self.device)
        self.qnet_target = qnet(self.input_shape, self.n_actions).to(self.device)
        self.qnet_target.eval()
        self.optimizer = optim.RMSprop(self.qnet_eval.parameters(), lr=self.lr)

    def choose_action(self, s, hidden, epsilon=0):
        b_s = {}
        for key in s:
            ts = torch.FloatTensor(s[key]).unsqueeze(0).unsqueeze(0).to(self.device)
            b_s[key] = ts
        with torch.no_grad():
            actions_value, hidden = self.qnet_eval.forward(b_s, hidden)
        if np.random.uniform() > epsilon:   # greedy
            action = torch.max(actions_value, 2)[1].data.cpu().numpy()[0,0]
        else:   # random
            action = np.random.randint(0, self.n_actions)
        return action, hidden

    def store_transition(self, s, a, r, s_, d, end_episode=False):
        self.memory.push(s, a, r, s_, d, end_episode=end_episode)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.qnet_target.load_state_dict(self.qnet_eval.state_dict())
        
        # sample batch memory from all memory
        b_s, b_a, b_r, b_s_, b_d = self.memory.sample_torch(self.batch_size, self.device) 
        
        # Burn in
        burn_in_len = 2
        b_a = b_a[:,burn_in_len:,...]
        b_r = b_r[:,burn_in_len:,...]
        b_d = b_d[:,burn_in_len:,...]
        for k in b_s:
            b_s[k] = b_s[k][:,burn_in_len:,...]
            b_s_[k] = b_s_[k][:,burn_in_len:,...]

        q_curr_eval, _ = self.qnet_eval(b_s)
        q_curr_eval_action = q_curr_eval.gather(2, b_a)
        q_next_target, _ = self.qnet_target(b_s_)
        q_next_target = q_next_target.detach()

        #next_state_values = q_next_target.max(2)[0].unsqueeze(-1)   # DQN
        q_next_eval, _ = self.qnet_eval(b_s_)
        q_next_eval = q_next_eval.detach()
        next_state_values = q_next_target.gather(2, q_next_eval.max(2)[1].unsqueeze(2))   # DDQN
        
        q_curr_recur = b_r + (1-b_d) * self.gamma * next_state_values
        self.loss = F.smooth_l1_loss(q_curr_eval_action, q_curr_recur).mean()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
        
        return float(self.loss.detach().cpu().numpy())
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        qnet_path = os.path.join(path, "qnet.pt")
        torch.save(self.qnet_eval.state_dict(), qnet_path)
    
    def load_model(self, path):
        qnet_path = os.path.join(path, "qnet.pt")
        self.qnet_eval.load_state_dict(torch.load(qnet_path, map_location=self.device))
        self.qnet_target.load_state_dict(torch.load(qnet_path, map_location=self.device))
