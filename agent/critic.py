import numpy as np
import torch
import torch.nn.functional as F
import utils

from torch import nn


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, device):
        super().__init__()
        self.device = device
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        #print('inside critic for')
        #print(obs, action)
        #print(np.shape(obs)[0], np.shape(action)[0])
        #print(action)
        #print('shape of obs', np.shape(obs))
        #print('shape of action', np.shape(action))
        #input('wait')
        # assert (obs.size(0) == action.size(0) or np.shape(obs)[0] == np.shape(action)[0])
        # print('passed')
        # input('wait')
        try:
            assert obs.size(0) == action.size(0)
            obs_action = torch.cat([obs, action], dim=-1)
        except:

            assert np.shape(obs)[0] == np.shape(action)[0]
            obs_action = torch.cat([torch.tensor(obs, dtype=torch.double), torch.tensor(action, dtype = torch.double)], 1)
            obs_action = torch.tensor(obs_action, dtype=torch.float, device=self.device)
        #print(obs_action)
        #print(np.shape(obs_action))
        #input('wait')
        #print('obs_action.get_device()', obs_action.device)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        #print('finsihed logger.log_histogram in ciritic')

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)

class TripleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, device):
        super().__init__()
        self.device = device
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q3 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        #print('inside critic for')
        #print(obs, action)
        #print(np.shape(obs)[0], np.shape(action)[0])
        #print(action)
        #print('shape of obs', np.shape(obs))
        #print('shape of action', np.shape(action))
        #input('wait')
        # assert (obs.size(0) == action.size(0) or np.shape(obs)[0] == np.shape(action)[0])
        # print('passed')
        # input('wait')
        try:
            assert obs.size(0) == action.size(0)
            obs_action = torch.cat([obs, action], dim=-1)
        except:

            assert np.shape(obs)[0] == np.shape(action)[0]
            obs_action = torch.cat([torch.tensor(obs, dtype=torch.double), torch.tensor(action, dtype = torch.double)], 1)
            obs_action = torch.tensor(obs_action, dtype=torch.float, device=self.device)
        #print(obs_action)
        #print(np.shape(obs_action))
        #input('wait')
        #print('obs_action.get_device()', obs_action.device)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        q3 = self.Q3(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2
        self.outputs['q3'] = q3
        return q1, q2, q3

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        #print('finsihed logger.log_histogram in ciritic')

        assert len(self.Q1) == len(self.Q2) == len(self.Q3)
        for i, (m1, m2, m3) in enumerate(zip(self.Q1, self.Q2, self.Q3)):
            assert type(m1) == type(m2)==type(m3)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
                logger.log_param(f'train_critic/q3_fc{i}', m3, step)