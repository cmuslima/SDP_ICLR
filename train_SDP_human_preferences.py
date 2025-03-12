
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from logger import Logger
from basic_replay_buffer import ReplayBuffer
from reward_model_pebble_human import RewardModel
import utils
import hydra
import wandb
LOGGING = False
class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        if LOGGING:
            print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)
        
        self.all_frames = []

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
        use_human_labels = True

       # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            reward_batch=cfg.reward_batch,
            device = cfg.device, 
            use_human_labels = use_human_labels, 
            train_batch_size = cfg.rm_train_batch_size, 
            max_feedback = self.cfg.max_feedback)
        
        
            
        self.reward_model.load(f'{self.cfg.base_dir}/{self.cfg.prior_data_path}_seed{self.cfg.seed}', self.cfg.prior_data_amount)
        self.replay_buffer = utils.get_data(f'{self.cfg.base_dir}/{self.cfg.prior_data_path}_seed{self.cfg.seed}/replay_buffer')
        self.replay_buffer.device = 'cpu'

        if self.cfg.load_agent:
            self.agent.load(f'{self.cfg.base_dir}/{self.cfg.prior_data_path}_seed{self.cfg.seed}', self.cfg.prior_data_amount)   
        self.relabel_prior_data = 1 
        self.agent_pretrain = 1

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if LOGGING:
            print(f'Eval return = {average_true_episode_reward}')
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
        
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
                if LOGGING:
                    print('Got human preferences')
            else:
                raise NotImplementedError
        
        self.total_feedback += self.reward_model.reward_batch
        self.labeled_feedback += labeled_queries
        if LOGGING:
            print('self.total_feedback', self.total_feedback)
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break;

            if LOGGING:
                print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        start_time = time.time()

        interact_count = 0
        frames_single_episode = []
        while self.step < self.cfg.num_train_steps: 
            if done:
                print(f'Agent trained for {self.step} steps')
                if self.step > 0:
                    assert len(frames_single_episode) == self.env._max_episode_steps
                    self.all_frames.append(frames_single_episode)
                
                frames_single_episode = []
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                if LOGGING:
                    print(f'train/total_feedback = {self.total_feedback}')
                    print(f'train/labeled_feedback = {self.labeled_feedback}')
                    print(f'train/true_episode_reward = {true_episode_reward}')   

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
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
            # run training update           
            #
            if self.step == self.cfg.num_seed_steps and self.cfg.num_unsup_steps > 0:
                if LOGGING:
                    print('Not using the unsupervised exploration as sub-optimal data')
                    print('Going to do the agent update phase, therefore I am not getting feedback this early')

            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                
                if LOGGING:
                    print(f' step = {(self.cfg.num_seed_steps + self.cfg.num_unsup_steps)}')
                if self.cfg.num_unsup_steps > 0:
                    # first learn reward
                    self.learn_reward(first_flag=1)
                    
                    # relabel buffer
                    self.replay_buffer.relabel_with_predictor(self.reward_model,self.relabel_prior_data, self.cfg.prior_data_amount)
                
                else:
                    if LOGGING:
                        print('Used the unsupervised exploration as sub-optimal data, not getting anymore')
                        print('Going to do the agent update phase, therefore I am not getting feedback this early')
                    # relabel buffer
                    self.replay_buffer.relabel_with_predictor(self.reward_model,self.relabel_prior_data, self.cfg.prior_data_amount)
                #reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                #reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                 
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model, self.relabel_prior_data, self.cfg.prior_data_amount)
                        interact_count = 0
            
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            next_obs, reward, done, extra = self.env.step(action)
            if self.total_feedback < self.cfg.max_feedback:
                frame = self.env.render(mode='rgb_array')
            else:
                frame = None #no longer want to save the frames
            frames_single_episode.append(frame)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            if self.total_feedback < self.cfg.max_feedback:
                # adding data to the reward training data
                self.reward_model.add_data(obs, action, reward, done, frame)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1


        utils.save_model(self.reward_model, self.work_dir, self.step)
        utils.save_model(self.agent,self.work_dir, self.step)
        utils.save_data('human_study_preference_data', self.reward_model.human_study_data)

@hydra.main(config_path='config/train_sdp_pebble.yaml', strict=True)
def main(cfg):
    entity_name = ''
    workspace = Workspace(cfg)
    wandb.init(project="sdp", config=utils.flatten_dict(dict(cfg)), name=f'human_sdp_pebble_{cfg.env}_{cfg.max_feedback}_{cfg.seed}',
               entity=f"{entity_name}", mode=cfg.wandb_mode)
    workspace.run()

if __name__ == '__main__':
    main()