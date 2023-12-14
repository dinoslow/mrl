import numpy as np
import torch 
import random

class Memory(object):
    def __init__(self, memory_size=100000):
        self.memory_size = memory_size
        self.reset()

    def reset(self):
        self.next_idx = 0
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)
        
    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size: # buffer not full
            self.buffer.append(data)
        else: # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = {}, [], [], {}, []
        for i in range(batch_size):
            idx = random.randint(0, len(self.buffer) - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done = data
            #
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            # state
            for key in state:
                if key in states:
                    states[key].append(state[key])
                else:
                    states[key] = [state[key]]
            # next state
            for key in next_state:
                if key in next_states:
                    next_states[key].append(next_state[key])
                else:
                    next_states[key] = [next_state[key]]

        return states, actions, rewards, next_states, dones
    
    def sample_torch(self, batch_size, device, discrete_control=True):
        states, actions, rewards, next_states, dones = self.sample(batch_size)
        
        if discrete_control:
            b_a = torch.LongTensor(np.array(actions).reshape(-1,1)).to(device)
        else:
            b_a = torch.FloatTensor(np.array(actions)).to(device)
        
        b_r = torch.FloatTensor(np.array(rewards).reshape(-1,1)).to(device)
        b_d = torch.FloatTensor(np.array(dones).reshape(-1,1)).to(device)
        b_s, b_s_ = {}, {}
        for key in states:
            b_s[key] = torch.FloatTensor(np.stack(states[key], 0)).to(device)
            b_s_[key] = torch.FloatTensor(np.stack(next_states[key], 0)).to(device)
        
        return b_s, b_a, b_r, b_s_, b_d

class EpisodicMemory():
    pass
