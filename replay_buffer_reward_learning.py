import numpy as np
import torch
import utils

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, window=1):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

    # def use_existing_replay_buffer(self, replay_buffer, num):
    #     for i in range(0, num):
    #         obs = replay_buffer.obses[i]
    #         action = replay_buffer.actions[i]
    #         reward = replay_buffer.rewards[i]
    #         next_obs = replay_buffer.next_obses[i]
    #         not_dones = replay_buffer.not_dones[i]
    #         done_no_max = replay_buffer.not_dones_no_max[i]
    #         np.copyto(self.obses[self.idx], obs)
    #         np.copyto(self.actions[self.idx], action)
    #         np.copyto(self.rewards[self.idx], reward)
    #         np.copyto(self.next_obses[self.idx], next_obs)
    #         np.copyto(self.not_dones[self.idx], not_dones)
    #         np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

    #         self.idx = (self.idx + 1) % self.capacity
    #         self.full = self.full or self.idx == 0

    # def use_existing_replay_buffer2(self, replay_buffer):

    #     """In this version of the replay buffer, I end up gradually removing the old data"""
    #     num = len(replay_buffer.obses)
    #     for i in range(0, num):
    #         obs = replay_buffer.obses[i]
    #         action = replay_buffer.actions[i]
    #         reward = replay_buffer.rewards[i]
    #         next_obs = replay_buffer.next_obses[i]
    #         not_dones = replay_buffer.not_dones[i]
    #         done_no_max = replay_buffer.not_dones_no_max[i]
    #         np.copyto(self.obses[i], obs)
    #         np.copyto(self.actions[i], action)
    #         np.copyto(self.rewards[i], reward)
    #         np.copyto(self.next_obses[i], next_obs)
    #         np.copyto(self.not_dones[i], not_dones)
    #         np.copyto(self.not_dones_no_max[i], not done_no_max)
        
    #     self.idx = 0
    #     self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    # def updated_add(self, obs, action, reward, next_obs, done, done_no_max):

    #     np.copyto(self.obses[self.add_idx], obs)
    #     np.copyto(self.actions[self.add_idx], action)
    #     np.copyto(self.rewards[self.add_idx], reward)
    #     np.copyto(self.next_obses[self.add_idx], next_obs)
    #     np.copyto(self.not_dones[self.add_idx], not done)
    #     np.copyto(self.not_dones_no_max[self.add_idx], not done_no_max)

    #     self.add_idx = (self.add_idx + 1) % self.capacity
    #     self.full = self.full or self.add_idx == 0
    #     #print('self.add_idx inside replay buffer add func', self.add_idx)

    #     if self.add_idx > self.idx:
    #         self.idx = self.add_idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, obs, action, reward, next_obs, done, done_no_max):
        
        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.next_obses[self.idx:self.capacity], next_obs[:maximum_index])
            np.copyto(self.not_dones[self.idx:self.capacity], done[:maximum_index] <= 0)
            np.copyto(self.not_dones_no_max[self.idx:self.capacity], done_no_max[:maximum_index] <= 0)
            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.next_obses[0:remain], next_obs[maximum_index:])
                np.copyto(self.not_dones[0:remain], done[maximum_index:] <= 0)
                np.copyto(self.not_dones_no_max[0:remain], done_no_max[maximum_index:] <= 0)
            self.idx = remain
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.next_obses[self.idx:next_index], next_obs)
            np.copyto(self.not_dones[self.idx:next_index], done <= 0)
            np.copyto(self.not_dones_no_max[self.idx:next_index], done_no_max <= 0)
            self.idx = next_index
        
    def relabel_with_predictor(self, predictor, relabel_prior_data=True, num_random_steps=0):
        batch_size = 200
        total_iter = int(self.idx/batch_size)
        if self.idx > batch_size*total_iter:
            total_iter += 1
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx

            if relabel_prior_data == False:
                if last_index < num_random_steps or index*batch_size<num_random_steps:
                    pass
            obses = self.obses[index*batch_size:last_index]
            actions = self.actions[index*batch_size:last_index]
            inputs = np.concatenate([obses, actions], axis=-1)
            pred_reward   = predictor.r_hat_batch(inputs)
            self.rewards[index*batch_size:last_index] = pred_reward
            
    def sample(self, batch_size):        
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device)
        
        return obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max