import numpy as np
import torch
import torch.nn.functional as F
import gym
import os
import random
import math
import pickle
from collections import deque
from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv
from collections import deque
from torch import nn
from torch import distributions as pyd
import scipy.stats as stats
import numpy as np
import json
import heapq
from omegaconf import OmegaConf, DictConfig

def get_norm(model_params):
    return np.linalg.norm(model_params)

def normalize(value, min_value, max_value):

    normalized_value = (value-min_value)/(max_value-min_value)
    return normalized_value
    
def find_indices_n_largest_elements(lst, N):
    res = [lst.index(i) for i in heapq.nlargest(N, lst)]
    return res
def get_json_data(file_name):
    with open(file_name) as data:
        file_contents = json.load(data)
    return file_contents
def welsh_t_test(data1, data2, alternative):
    print(data1)
    print(data2)
    result = stats.ttest_ind(data1, data2, alternative=alternative, equal_var = False)
    return result

def make_gym_env(cfg):
    #import gymnasium as gym

    env = gym.make(cfg.env)
    return env

def make_gym_env(cfg):
    import gym
    env = gym.make(cfg.env)
    env.seed(cfg.seed)
    return env
def make_env(cfg):
    print(cfg.env)
    #input('wait')

    import dmc2gym

    """Helper function to create dm_control environment"""

    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'

    if cfg.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'

    elif cfg.env == 'point_mass_hard':
        domain_name = 'point_mass'
        task_name = 'hard'

    elif cfg.env == 'point_mass_v2_easy':
        domain_name = 'point_mass_v2'
        task_name = 'easy'
    elif cfg.env == 'point_mass_v3_easy':
        domain_name = 'point_mass_v3'
        task_name = 'easy'
    elif cfg.env == 'point_mass_v3_strict_easy':
        domain_name = 'point_mass_v3_strict'
        task_name = 'easy'
    elif cfg.env == 'reacher_v2_easy':
        domain_name = 'reacher_v2'
        task_name = 'easy'
    elif cfg.env == 'reacher_v2_hard':
        domain_name = 'reacher_v2'
        task_name = 'hard'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])


    print('task name', task_name)
    print('domain name', domain_name)
    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=False)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

def ppo_make_env(env_id, seed):
    """Helper function to create dm_control environment"""
    if env_id == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = env_id.split('_')[0]
        task_name = '_'.join(env_id.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=seed,
                       visualize_reward=True)
    env.seed(seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias
    
def make_metaworld_env(cfg):
    import metaworld
    import metaworld.envs.mujoco.env_dict as _env_dict
    print('imported metaworld')
    env_name = cfg.env.replace('metaworld_','')
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]
    
    env = env_cls()
    
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(cfg.seed)
    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)

def ppo_make_metaworld_env(env_id, seed):
    env_name = env_id.replace('metaworld_','')
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]
    
    env = env_cls()
    
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    
    return TimeLimit(env, env.max_path_length)



class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def get_target(matched):
    if matched:
        return 1
    else:
        return -1 
def match(x,y, epilson):
    dim = np.shape(x)[0]
    for i in range(0, dim):
        print(x[i], y[i])
        print('i', i)
        if (x[i] <= y[i] + epilson) and (x[i] >= y[i] - epilson):
            continue
        else:
            return False
    return True
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
    
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
    
class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)

def get_data(file):
    with open(file,'rb') as input:
        data = pickle.load(input)
    return data

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def save_data(filename, data):
    pickle.dump(data, open(filename, 'wb'))

def save_model(model, dir, step):
    model.save(dir, step)
def flatten_dict(cfg, parent_key=''):
    if isinstance(cfg, DictConfig):
        # Convert DictConfig to a regular dictionary
        cfg = OmegaConf.to_container(cfg, resolve=True)

    items = []
    for key, value in cfg.items():
        new_key = f"{parent_key}_{key}" if parent_key else key
        if isinstance(value, DictConfig):
            # If the value is a DictConfig, convert it and then extend
            value = OmegaConf.to_container(value)  # , resolve=True)  # resolve gave an error here, for the ??? values
            items.extend(flatten_dict(value, new_key).items())
        elif isinstance(value, dict):
            # Recursively call flatten_dict if the value is a dictionary
            items.extend(flatten_dict(value, new_key).items())
        else:
            # Add key-value pair directly to items
            items.append((new_key, value))
    return dict(items)


class MetaOptim(torch.optim.Adam):
    def __init__(self, net, *args, **kwargs):
        super(MetaOptim, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads):
        group = self.param_groups[0]
        lr = group['lr']
        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            parameter.detach_()
            self.set_parameter(self.net, name, parameter.add(grad, alpha=-lr))