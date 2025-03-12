
from hmac import new
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle 
import tqdm
import time

from logger import Logger
from basic_replay_buffer import ReplayBuffer

from reward_model_regression import RewardModel
from collections import deque

import utils
import hydra
import wandb


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.random_data = True
        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)
        print('finished setting logger')
        utils.set_seed_everywhere(cfg.seed)
        print('finished setting seed')
        self.device = torch.device(cfg.device)
        print('finished setting device, device =', cfg.device)
        self.log_success = False
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)
        
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.device, 
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            init = cfg.init, 
            train_batch_size = cfg.rm_train_batch_size,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal, 
            std_white_noise = cfg.std_white_noise,
            white_noise_label = cfg.white_noise_label,
            round = cfg.round
        )




        self.relabel_prior_data = 1
        self.agent_pretrain = 1
  
    def evaluate(self):
        average_predicted_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            print(f'eval episode {episode}')
            obs = self.env.reset()
            self.agent.reset()
            done = False
            pred_episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                reward_hat  = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
                
                pred_episode_reward += reward_hat
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
            print(f'predicted eval episode {episode} return = {pred_episode_reward}')
            print(f'true eval episode return = {true_episode_reward}')
            average_predicted_episode_reward += pred_episode_reward

            average_true_episode_reward += true_episode_reward

            if self.log_success:
                success_rate += episode_success
            
        average_predicted_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        print(f'average eval episode return = {average_predicted_episode_reward}')
        print(f'true average eval episode return = {average_true_episode_reward}')

        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_predicted_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
        self.logger.dump(self.step)
    
    def get_feedback(self):
        # update schedule
        if self.cfg.reward_schedule == 1:
            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
            if frac == 0:
                frac = 0.01
        elif self.cfg.reward_schedule == 2:
            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
        else:
            frac = 1
        self.reward_model.change_batch(frac)
        
        # update margin --> not necessary / will be updated soon
        '''skip threshold will just be 0 for now, so these 3 lines don't actually do anything'''
        new_margin = np.mean(self.avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
        self.reward_model.set_teacher_thres_skip(new_margin)
        self.reward_model.set_teacher_thres_equal(new_margin)
        
        # first learn reward
        self.learn_reward()
        print('total feedback so far', self.total_feedback)
        print(f"Got real feedback at step {self.step}")
        
        # relabel buffer
        self.replay_buffer.relabel_with_predictor(self.reward_model, relabel_prior_data=self.relabel_prior_data, num_random_steps=self.cfg.num_seed_steps + self.cfg.num_included_unsup_steps)

    def learn_reward(self, first_flag=0):

        # get feedbacks

        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling(self.random_data)
            #print('using uniform sampling')
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling(self.random_data)
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()

            else:
                raise NotImplementedError
        
        if first_flag != 1:
            self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            print('going to update the reward', self.cfg.reward_update)
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    
                    train_loss = self.reward_model.train_reward()
                if train_loss < .03:
                    break
        print("Reward function is updated!! TRAIN LOSS: " + str(train_loss))

    
    def run(self):
        utils.save_model(self.agent,self.work_dir, self.step)
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        self.avg_train_true_return = deque([], maxlen=10) 
        start_time = time.time()

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            print(f'step = {self.step}')
            if done:
                print('episode', episode)
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))
                    if self.step > self.cfg.num_seed_steps:
                        print('dumping train data')

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()


                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    print('train/true_episode_success', episode_success, self.step)
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                self.avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0

                self.logger.log('train/episode', episode, self.step)
                    
                print('episode', episode, 'step', self.step)
                print('total feedback', self.total_feedback)
                #input('wait')
                episode += 1  
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)


            #This should be getting the "free labels" for the random data we initally collected
            if self.cfg.leverage_prior_data and self.cfg.transfer_reward_model:  
                if self.step % self.cfg.reward_batch == 0 and self.step > self.cfg.reward_batch and self.step <= self.cfg.num_seed_steps + self.cfg.num_included_unsup_steps and self.step > self.reward_model.size_segment:
                    
                    # first learn reward
                    self.learn_reward(first_flag=1)

                    # relabel buffer
                    self.replay_buffer.relabel_with_predictor(self.reward_model)
                    print(f'pretraining my reward model with the free random data {self.step}')


            if self.step == (self.cfg.num_seed_steps) and self.cfg.include_unsup_steps == False:
                #I am done with with the reward model pretraining phase. If I am in this condition, it means I am not 
                #treating the unsupervised exploration data as sub-optimal 
                assert self.cfg.num_included_unsup_steps == 0
                self.random_data = False
                
                print('Not using the unsupervised exploration as sub-optimal data')

                if self.cfg.transfer_reward_model:
                    #This condition means, I am using this file to run experiments with scalar based FB
                    if self.cfg.num_train_steps >=500000:
                        print('Running a scalar based feedback experiment')
                        

                        print('Going to do the agent update phase, therefore I must first reinit reward model dataset')
                        #This conditions means I do want to do the agent update phase. Therefore I am going to 
                        #reinit the reward model dataset. 
                        self.reward_model.reset_reward_model()
                    else:
                        #This condition means, I am using this file to do sdp, but then I am going to plug in preference based RL
                        print("I am just getting the reward model and replay buffer for my pref learning agent. ")
                utils.save_model(self.reward_model, self.work_dir, self.step)
                utils.save_model(self.agent,self.work_dir, self.step)
    

            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                print(f'step = {self.cfg.num_seed_steps + self.cfg.num_unsup_steps}')
    
                if self.cfg.include_unsup_steps:
                    print('Using the unsupervised exploration as sub-optimal data')
                #I am done with with the reward model pretraining phase. If I am in this condition, it means I am  
                #treating the unsupervised exploration data as sub-optimal 
                    assert self.cfg.num_included_unsup_steps > 0 

                    #This condition means, I am using this file to run experiments with scalar based FB
                    if self.cfg.num_train_steps >=500000:
                        print('Running a scalar based feedback experiment')
                        #This conditions means I do want to do the agent update phase. Therefore I am going to 
                        #reinit the reward model dataset. 
                        self.reward_model.reset_reward_model()


                    #This condition means, I am using this file to do sdp, but then I am going to plug in preference based RL
                    else:
                        print("I am just getting the reward model and replay buffer for my pref learning agent. ")
                
                #This condition means, I am using this file to do sdp, but then I am going to plug in preference based RL
                #More specifically, I am not considering the unsupervised exploration data as sub-optimal
                elif (self.cfg.transfer_reward_model or self.cfg.transfer_replay_buffer) and self.cfg.include_unsup_steps == False and self.cfg.num_unsup_steps == 0:
                    print('This is the case where I want to pretrain my reward model only on the random data. NOT on the unsupervised exploration data.')
                    print('After this, I want to save my reward models and my program should end')
                else:
                    #This condition occurs if I don't want to use the unsupervised exploration as sub-optimal data, 
                    #but as data to get actual feedback on.
                    print('This condition occurs if I dont want to use the unsupervised exploration as sub-optimal data, but as data to get actual feedback on.')

                    self.get_feedback()
                

                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0

                utils.save_model(self.reward_model, self.work_dir, self.step)
                utils.save_model(self.agent,self.work_dir, self.step)

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:

                #This random_data variable is only used if we want to run noisy teacher experiments. 
                #It makes sure we don't treat the sub-optimal data as noisy data.
                self.random_data = False
                print(f'step {self.step} is greater than {self.cfg.num_seed_steps + self.cfg.num_unsup_steps}')

                # update reward function
                if self.total_feedback < self.cfg.max_feedback:

                    if interact_count == self.cfg.num_interact:
                        print('about to get feedback and update my reward')
                        print('total feedback so far', self.total_feedback)
                        self.get_feedback()


                        utils.save_model(self.reward_model, self.work_dir, self.step)
                        utils.save_model(self.agent,self.work_dir, self.step)

                        interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                print('just passed self.agent.update')
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                print('unsupervised exploration')
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                                            


            
            next_obs, reward, done, extra = self.env.step(action)
            true_reward = reward

            
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
 
            episode_reward += reward_hat
            true_episode_reward += reward


            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                

            #In this condition, I want to do the agent update phase. Therefore I add the sub-opt transitions to my agent's replay buffer with r=0
            if self.cfg.leverage_prior_data and self.cfg.transfer_replay_buffer and self.step < self.cfg.num_seed_steps + self.cfg.num_included_unsup_steps:
                reward_hat = 0
                print(f'labeling data for agent replay buffer with 0 at time step {self.step}')

            #In this condition, I want to do the pretrain phase. Therefore I add the sub-opt transitions to my reward model's replay buffer with r=0
            if self.cfg.leverage_prior_data and self.cfg.transfer_reward_model and self.step < self.cfg.num_seed_steps + self.cfg.num_included_unsup_steps:
                reward = 0
                print(f'labeling data for reward model with 0 at time step {self.step}')

        
            self.reward_model.add_data(obs, action, reward, done)

            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)



            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

            
        utils.save_model(self.reward_model, self.work_dir, self.step)
        utils.save_model(self.agent,self.work_dir, self.step)
        utils.save_data(f'replay_buffer', self.replay_buffer)
        utils.save_data(f'reward_model_inputs', self.reward_model.inputs)
        utils.save_data(f'reward_model_targets', self.reward_model.targets)



@hydra.main(config_path='config/sdp_regression.yaml', strict=True)
def main(cfg):
    entity = ''
    workspace = Workspace(cfg)
    wandb.init(project="pbrl", config=utils.flatten_dict(dict(cfg)), name=f'sdp_pretraining_{cfg.env}_{cfg.seed}',
               entity=f"{entity}")
    workspace.run()

if __name__ == '__main__':
    main()


