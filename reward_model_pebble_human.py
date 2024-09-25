import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time
import utils
from scipy.stats import norm
import get_human_preferences
device = 'cpu'

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, last_activation='tanh', init='default'):
    net = []
    for i in range(n_layers):
        layer = nn.Linear(in_size, H)
        if init == 'zeros':
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        elif init == 'normal':
            nn.init.normal_(layer.weight)
            nn.init.normal_(layer.bias)

        elif init == 'glorot':
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        elif init == 'neg_normal':
            nn.init.normal_(layer.weight, -1, 1)
            nn.init.normal_(layer.bias, -1, 1)  

        net.append(layer)
        activation = nn.LeakyReLU()
        net.append(activation)
        in_size = H

    layer = nn.Linear(in_size, out_size)
    if init == 'zeros':
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)

    elif init == 'normal':
        nn.init.normal_(layer.weight)
        nn.init.normal_(layer.bias)

    elif init == 'glorot':
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

    elif init == 'neg_normal':
        nn.init.normal_(layer.weight, -1, 1)
        nn.init.normal_(layer.bias, -1, 1)  

    net.append(layer)

    if last_activation == 'tanh':
        net.append(nn.Tanh())
    elif last_activation == 'sig':
        net.append(nn.Sigmoid())
    elif last_activation == 'relu':
        net.append(nn.ReLU())
    elif last_activation == 'leaky_relu':
        #print('using last activation of leaky_relu')
        net.append(nn.LeakyReLU())

    return net


class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, reward_batch = 128, size_segment=1, train_batch_size=128,
                 max_size=100, activation='tanh', init='default', capacity=5e5,  
                 device = 'device', use_human_labels=True, max_feedback=48, reload_preferences=False, prior_human_study_data=None):
        
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.last_activation = activation
        self.init = init
        self.size_segment = size_segment
        self.device = device
        self.capacity = int(capacity)
        self.reward_batch = reward_batch
        self.train_batch_size = train_batch_size
        self.use_human_labels = use_human_labels
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
        self.number_feedback_sessions = int(max_feedback/reward_batch)
        self.current_feedback_session = 1
        self.reload_preferences = reload_preferences
                
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.frames = []



        self.CEloss = nn.CrossEntropyLoss()
        self.human_study_data = {
            'sampled_trajectory1': [],
            'sampled_trajectory2': [], 
            'frames_sampled_trajectory1': [],
            'frames_sampled_trajectory2': [],
            'human_labels': [], 
            'ground_truth_labels': [],
            'time': []
        }
        self.prior_human_study_data = prior_human_study_data



     
    def construct_ensemble(self):

        for i in range(self.de):

            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           last_activation=self.last_activation, init=self.init)).float().to(self.device)

            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rew, done, frame):
    
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        frame = np.array(frame)
        flat_frame = frame.reshape(1, int(84*84*3*1.5)) #frame.reshape(1, 1) #
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
            self.frames.append(flat_frame)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            self.frames[-1] = np.concatenate([self.frames[-1], flat_frame])

            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
                self.frames = self.frames[1:]

            self.inputs.append([])
            self.targets.append([])
            self.frames.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
                self.frames[-1] = flat_frame
    

            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                self.frames[-1] = np.concatenate([self.frames[-1], flat_frame])


                
    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
    
    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        value = self.ensemble[member](torch.from_numpy(x).float().to(self.device))
        return self.ensemble[member](torch.from_numpy(x).float().to(self.device))

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member), map_location=str(self.device))
            )

    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(self.device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    

    def get_queries(self, reward_batch=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)

        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
        frames =  np.array(self.frames[:max_len])


        batch_index_2 = np.random.choice(max_len, size=reward_batch, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a

        r_t_2 = train_targets[batch_index_2] # Batch x T x 1
        frames_t_2 = frames[batch_index_2] # Batch x T x 1
        
        batch_index_1 = np.random.choice(max_len, size=reward_batch, replace=True)
        sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1] # Batch x T x 1
        frames_t_1 = frames[batch_index_1] # Batch x T x 1
  

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (Batch x T) x 1
        frames_t_1 = frames_t_1.reshape(-1, frames_t_1.shape[-1]) # (Batch x T) x 1

        
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (Batch x T) x 1
        frames_t_2 = frames_t_2.reshape(-1, frames_t_2.shape[-1]) # (Batch x T) x 1

        # Generate time index 
        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(reward_batch)])
        
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=reward_batch, replace=True).reshape(-1,1)
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=reward_batch, replace=True).reshape(-1,1)

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
        frames_t_1 =  np.take(frames_t_1, time_index_1, axis=0)
        frames_t_2 =  np.take(frames_t_2, time_index_2, axis=0)          

    
        return sa_t_1, sa_t_2, r_t_1, r_t_2, frames_t_1, frames_t_2

    def get_all_human_labels(self, frames_t_1, frames_t_2):
        print("---------------------------------------------------------")
        input(f'Feedback session {self.current_feedback_session} is about to begin!')
        print('\n')
        labels = []
        start_time = time.time()
        for trajectory_index in range(0, self.reward_batch):
            segment1 = frames_t_1[trajectory_index]
            segment2 = frames_t_2[trajectory_index]

            labels.append(self.get_single_human_label(segment1, segment2, trajectory_index))

        end_time = time.time() 
        duration = end_time - start_time
        self.human_study_data['time'].append(duration)

        print(f'Feedback session is over. You have {self.number_feedback_sessions-self.current_feedback_session} feedback sessions remaining.')
        print(f'It took you {duration} seconds to give feedback')
        print("---------------------------------------------------------")
        self.current_feedback_session +=1
        return labels

    def get_single_human_label(self, segment1, segment2, trajectory_id):
        segment1 = segment1.reshape(self.size_segment, 84, int(84*1.5),3)
        segment2 = segment2.reshape(self.size_segment, 84, int(84*1.5),3)

        print(f'Getting ready to show two trajectories (A and B) for batch # {trajectory_id}!')
        input("Press Enter if you are ready to give preferences to Trajectory A")
        print("\n")
        get_human_preferences.render_trajectory(segment1, trajectory_id, 'A')
        input("Press Enter if you are ready to give preferences to Trajectory B")
        print("\n")

        get_human_preferences.render_trajectory(segment2, trajectory_id, 'B')
        get_human_preferences.render_trajectory_again(segment1, segment2, trajectory_id)

        
        while(True):
            print('Which trajectory did you prefer?')
            print(f'Press A for Trajectory A')
            print(f'Press B for Trajectory B')
            human_label = input("If you prefer the trajectories equally, press C: \n")
            print('\n')

            try:
                if human_label == 'B' or human_label == 'b':
                    label = 1
                    print(f'You preferred trajectory B')
                    print('\n')
                    return label
                elif human_label == 'A' or human_label == 'a':
                    label = 0
                    print(f'You preferred trajectory A')
                    print('\n')
                    return label
                elif human_label == 'C' or human_label == 'c':
                    label = 0.5
                    print(f'You preferred trajectory A and B equally.')
                    print('\n')
                    return label
                else:
                    print('You pressed an invalid key')
                    print('\n')
                    continue
      
            except:
                print('You pressed an invalid key')
                print('\n')



    def put_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
            
    def get_all_artifical_labels(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
      
            
        labels = 1*(sum_r_t_1 < sum_r_t_2)
      
        return  labels
    
       
  
    def uniform_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, frames_t_1, frames_t_2 =  self.get_queries(
            reward_batch=self.reward_batch)
            

        """We always want to get the artifical labels to see how many times the human labels agree
        with them"""
        ground_truth_labels = self.get_all_artifical_labels(sa_t_1, sa_t_2, r_t_1, r_t_2)

        if self.use_human_labels:
            human_labels = self.get_all_human_labels(frames_t_1, frames_t_2)
            labels = human_labels

        elif self.reload_preferences:
            human_labels = self.prior_human_study_data[self.current_feedback_session-1]
        else:
            human_labels = None
            labels = ground_truth_labels

        labels = np.array(labels)
        labels = np.reshape(labels, (self.reward_batch, 1))
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        """Saving the data"""
        self.human_study_data['sampled_trajectory1'].append(sa_t_1)
        self.human_study_data['sampled_trajectory2'].append(sa_t_2)
        self.human_study_data['frames_sampled_trajectory1'].append(frames_t_1)
        self.human_study_data['frames_sampled_trajectory2'].append(frames_t_2)
        self.human_study_data['human_labels'].append(human_labels)
        self.human_study_data['ground_truth_labels'].append(ground_truth_labels)
        return len(labels)


    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc
