import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import replay_memory

class DQNAgent():
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

        # Network
        self.qnet_eval = qnet(self.input_shape, self.n_actions).to(self.device)
        self.qnet_target = qnet(self.input_shape, self.n_actions).to(self.device)
        self.qnet_target.eval()
        self.optimizer = optim.RMSprop(self.qnet_eval.parameters(), lr=self.lr)

    def choose_action(self, s, epsilon=0):
        b_s = {}
        for key in s:
            b_s[key] = torch.FloatTensor(np.expand_dims(s[key],0)).to(self.device)
        with torch.no_grad():
            actions_value = self.qnet_eval.forward(b_s)
        if np.random.uniform() > epsilon:   # greedy
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:   # random
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_, d):
        self.memory.push(s, a, r, s_, d)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.qnet_target.load_state_dict(self.qnet_eval.state_dict())

        # sample batch memory from all memory
        b_s, b_a, b_r, b_s_, b_d = self.memory.sample_torch(self.batch_size, self.device) 
        q_curr_eval = self.qnet_eval(b_s)
        # print(q_curr_eval.shape, b_a.shape)
        q_curr_eval_action = q_curr_eval.gather(1, b_a)
        q_next_target = self.qnet_target(b_s_).detach()

        #next_state_values = q_next_target.max(1)[0].view(-1, 1)   # DQN
        q_next_eval = self.qnet_eval(b_s_).detach()
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
