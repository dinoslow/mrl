import numpy as np
import torch 
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

class EpisodicMemory(object):
    def __init__(self, memory_size=100000):
        self.memory_size = memory_size
        self.reset()

    def reset(self):
        self.next_idx = 0
        self.key_buffer = []
        self.value_buffer = []
    
    def __len__(self):
        return len(self.key_buffer)
    
    def push(self, key, value):
        # check if nearset key is similar
        dis, idx_to_remove = self.calculate_distance(key)
        if dis != None:
            if dis < 10.: # similar
                self.key_buffer[idx_to_remove] = key
                self.value_buffer[idx_to_remove] = value
                return
        
        if len(self.key_buffer) <= self.memory_size: # buffer not full
            self.key_buffer.append(key)
            self.value_buffer.append(value)
        else: # buffer is full
            self.key_buffer[self.next_idx] = key
            self.value_buffer[self.next_idx] = value
        self.next_idx = (self.next_idx + 1) % self.memory_size
    
    def calculate_distance(self, key):
        if len(self.key_buffer) == 0:
            return None, 0
        key = np.expand_dims(key, 0)
        keys = np.stack(self.key_buffer, 0)
        distance = euclidean_distances(key, keys)[0]
        min_dis = np.min(distance)
        min_idx = np.argmin(distance)
        return min_dis, min_idx
    
    def get_closest_pose(self, pose):
        dis, idx = self.calculate_distance(pose)
        return dis, self.key_buffer[idx]
    
    def get_all_poses(self):
        return self.key_buffer.copy()
        
    def create_pose_matrix(self, pose, device):
        pose = pose.reshape(-1, 3, 4)
        vec_affine = torch.tensor([0., 0., 0., 1.]).to(device)
        vec_affine = vec_affine.unsqueeze(0).repeat((pose.shape[0], 1, 1))
        transition_mat = torch.cat([pose, vec_affine], dim=1).reshape(-1, 4, 4)
        return transition_mat
    
    def transition(self, pose, device):
        # pose => agent pose (batch, 1, 3, 4)
        # self.key_buffer => memory pose (memory_size, 3, 4)

        # (batch, 1, 3, 4) => (batch, 1, 4, 4)
        pose_matrix = self.create_pose_matrix(pose, device).unsqueeze(1)
        pose_mat_inv = torch.linalg.inv(pose_matrix)

        # (batch, memory_size, 3, 4) => (batch, memory_size, 4, 4)
        abs_poses = torch.FloatTensor(np.stack(self.key_buffer)).to(device)
        abs_poses = self.create_pose_matrix(abs_poses, device)
        abs_poses = abs_poses.unsqueeze(0).repeat((pose_matrix.shape[0], 1, 1, 1))
        
        # (batch, memory_size, 4, 4)
        transformed_poses = torch.matmul(pose_mat_inv, abs_poses)
        # (batch, memory_size, 12)
        transformed_poses = transformed_poses[:, :, :3, :].reshape(transformed_poses.shape[0], -1, 12)
        
        return transformed_poses

    def fatch_features(self, pose, device, train=False):
        with torch.no_grad():
            transition_keys = self.transition(pose, device).to(device)

            # (1, memory_size, 256)
            image_features = np.expand_dims(np.stack(self.value_buffer), 0)
            image_features = torch.FloatTensor(image_features).to(device)
            # (batch, memory_size, 256)
            image_features = image_features.repeat((transition_keys.shape[0], 1, 1))

            if train:
                if transition_keys[0].shape[0] < 32:
                    repeat = 32 // transition_keys[0].shape[0] + 1
                    transition_keys = transition_keys.repeat(1, repeat, 1)[:,:32,:]
                    image_features = image_features.repeat(1, repeat, 1)[:,:32,:]
                else:
                    seq = np.random.randint(0, transition_keys[0].shape[0], 32)
                    transition_keys = transition_keys[:,seq,:]
                    image_features = image_features[:,seq,:]
            
            return {'key': transition_keys, 'value': image_features}
