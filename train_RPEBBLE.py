
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
#from replay_buffer import ReplayBuffer
from basic_replay_buffer import ReplayBuffer
import utils 
from reward_model import RewardModel
from collections import deque

import utils
import hydra

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.random_data = False
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
        
        print('finished setting env')
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        print('reward range', self.env.reward_range)
        #input('wait')
        print('finished setting obs dim')
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        print('finished setting action dim')
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        print('finished setting action rang')
        self.agent = hydra.utils.instantiate(cfg.agent)
        print('finished setting agent')

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        print('finished setting replay buffer')
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        print('num_seed_steps', cfg.num_seed_steps)
        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.device, 
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            init = cfg.init, 
            train_batch_size = cfg.train_batch_size,
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

        self.reward_data = False
    def evaluate(self):
        print('Inside evaluation')
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
            # self.logger.log('train/true_episode_success', success_rate,
            #             self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
        print('inside learn reward')
        # get feedbacks
        print('self.reward_model.mb_size', self.reward_model.mb_size)
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
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        #print('labeled_queries', labeled_queries)
        #print('reward update', self.cfg.reward_update)
        train_acc = 0
        update_start_time = time.time()
        if self.labeled_feedback > 0:
            # update reward
            print('going to update the reward', self.cfg.reward_update)
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    
                    train_loss = self.reward_model.train_reward()
                #total_acc = np.mean(train_acc)
                if train_loss < .03:
                    break

                
        utils.save_model(self.reward_model, self.work_dir, self.step)
        utils.save_model(self.agent,self.work_dir, self.step)
        end_time = time.time()
        print('duration of training = ', end_time-update_start_time)
        print("Reward function is updated!! TRAIN LOSS: " + str(train_loss))

    def run(self):
        utils.save_model(self.agent,self.work_dir, self.step)
        #print('inside run')
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
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

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    print('finished evaluating')
                print('train/episode_reward and step ', episode_reward, self.step)
                print('train/true_episode_reward and step ', true_episode_reward, self.step)
                print('train/total_feedback and step', self.total_feedback, self.step)
                print('train/labeled_feedback and step ', self.labeled_feedback, self.step)

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
                avg_train_true_return.append(true_episode_reward)
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

            # run training update    
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                ##print('run training update')
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
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                #print('new margin inside first if statement = ', new_margin)
                
                # first learn reward
                self.learn_reward(first_flag=1)
                print('total feedback so far', self.total_feedback)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                print('reset critic')
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                print('update agent')
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                #print('updating reward function')
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    #print('shouldnt be here')
                    #print('max feedback',  self.cfg.max_feedback, 'total feedback', self.total_feedback)
                    #print('interact_count', interact_count)
                    if interact_count == self.cfg.num_interact:
                        print('about to get feedback and update my reward')
                        print('total feedback so far', self.total_feedback)
                        #print('interact_count == self.interact_count')
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
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)

                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        #print('finished learn_reward')

                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                #print('unsupervised exploration')
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                                            


            
            next_obs, reward, done, extra = self.env.step(action)
     
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done

            episode_reward += reward_hat
            true_episode_reward += reward


            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                


            self.reward_model.add_data(obs, action, reward, done)
            
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

    
            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
            if self.step % 150000 == 0:
                utils.save_model(self.agent,self.work_dir, self.step)

 
        utils.save_model(self.reward_model, self.work_dir, self.step)
        utils.save_model(self.agent,self.work_dir, self.step)

@hydra.main(config_path='config/train_reward_learning.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()