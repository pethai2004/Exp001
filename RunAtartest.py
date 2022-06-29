#export PYTHONPATH=~/general_wrap/algorithm
#export PYTHONPATH=/root/general_wrap/algorithm
from algorithm.PPO import ProximalPG
from algorithm.VG import VanillaPG
from algorithm.REINFORCE import ReinforcePG
from algorithm.base_agent import BaseAgent, ActorCritic
from base_wrap import ActionSpec, ActionTYPE, GymWrapper, ObservationSpec
from nets import create_simple_conv, create_simple_conv_low
from policies import SimplePolicy
from runner import GymRunner
from utils import *

import gym 
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils.layer_utils import count_params  

from copy import deepcopy

gym_env_exp = ["ALE/AirRaid-v5", "ALE/IceHockey-v5", 'ALE/JourneyEscape-v5', "ALE/Breakout-v5",
              "ALE/PrivateEye-v5", "ALE/Tennis-v5", "ALE/MsPacman-v5", "ALE/Boxing-v5"]

# for env_name in gym_env_exp:

#     env0 = gym.make(env_name)
#     assert isinstance(env0.action_space,  gym.spaces.Discrete), 'env must have discrete_action_space'

def run_atari_ppo(env_names : list, epoches=40, policy=None, value_network=None, latent_obs_dim=(100, 100, 1), dir_='Run'):
    
    hist_model = []
    hist_algorithm = []

    for env_name in env_names:
        env0 = gym.make(env_name)
        assert isinstance(env0.action_space,  gym.spaces.Discrete), 'env must have discrete_action_space'

        obs_dim = env0.observation_space.shape
        act_dim = env0.action_space.n
        obs_spec = ObservationSpec(obs=latent_obs_dim)
        act_spec = ActionSpec(dis=(act_dim, ))

        ENV = GymWrapper(env=env0, obs_spec=obs_spec, act_spec=act_spec, rescale_image=latent_obs_dim[:-1])
        ENV.random_self_action()
        print('random_self_action_successful')

        if policy is None:
            policy_network = create_simple_conv(latent_obs_dim, act_dim, actv='relu', output_actv='sigmoid', seed_i=555)
            policy = SimplePolicy(policy_network, obs_spec, act_spec)

        if value_network is None:
            value_network = create_simple_conv_low(latent_obs_dim, 1, actv='relu', output_actv='sigmoid', seed_i=555)
        
        ALG = ProximalPG(ENV, policy, value_network, gamma=0.99, max_trj = 4000, clip_ratio=0.3, kl_beta=0, kl_target=0.001, kl_tol=2, kl_low=1.5, kl_cutoff=0.0,
        grad_clip_norm=2., policy_lr=0.0002, value_lr=0.01
    )
        ALG.entropy_rate = 0.5
        ALG._value_reg = 0.1
        ALG.value_loss_const = 0.7
        ALG.pol_iters = 80
        ALG.model_save_dir = 'Models/{}/{}'.format(env_name, type(ALG).__name__)

        summarizer = tf.summary.create_file_writer('{}/{}/{}'.format(dir_, env_name, type(ALG).__name__))

        with summarizer.as_default():
            rs = ALG.train(epoches)

        print('finish_training_env', env_name)

        hist_model.append(deepcopy(ALG.policy.network))
        hist_algorithm.append(deepcopy(ALG))

    return hist_model, hist_algorithm

def run_atari_vg(env_names : list, epoches=40, policy=None, value_network=None, latent_obs_dim=(100, 100, 1), dir_='Run', ):
    
    hist_model = []
    hist_algorithm = []

    for env_name in env_names:
        env0 = gym.make(env_name)
        assert isinstance(env0.action_space,  gym.spaces.Discrete), 'env must have discrete_action_space'

        obs_dim = env0.observation_space.shape
        act_dim = env0.action_space.n
        obs_spec = ObservationSpec(obs=latent_obs_dim)
        act_spec = ActionSpec(dis=(act_dim, ))

        ENV = GymWrapper(env=env0, obs_spec=obs_spec, act_spec=act_spec, rescale_image=latent_obs_dim[:-1])
        ENV.random_self_action()
        print('random_self_action_successful')

        if policy is None:
            policy_network = create_simple_conv(latent_obs_dim, act_dim, actv='relu', output_actv='sigmoid', seed_i=555)
            policy = SimplePolicy(policy_network, obs_spec, act_spec)

        if value_network is None:
            value_network = create_simple_conv_low(latent_obs_dim, 1, actv='relu', output_actv='sigmoid', seed_i=555)
        
        ALG = VanillaPG(ENV, policy, value_network,  gamma=0.99, max_trj=4000, summary=True, optimizer=None, grad_clip_norm=2., policy_lr=0.0002, value_lr=0.01, use_gae=True, lam=0.95
   )
        ALG.entropy_rate = 0.5
        ALG._value_reg = 0.1
        ALG.value_loss_const = 0.7
        ALG.pol_iters = 70
        ALG.model_save_dir = 'Models/{}/{}'.format(env_name, type(ALG).__name__)

        summarizer = tf.summary.create_file_writer('{}/{}/{}'.format(dir_, env_name, type(ALG).__name__))

        with summarizer.as_default():
            rs = ALG.train(epoches)

        print('finish_training_env', env_name)

        hist_model.append(deepcopy(ALG.policy.network))
        hist_algorithm.append(deepcopy(ALG))

    return hist_model, hist_algorithm


def run_atari_rf(env_names : list, epoches=40, policy=None, value_network=None, latent_obs_dim=(100, 100, 1), dir_='Run' ):
    
    hist_model = []
    hist_algorithm = []

    for env_name in env_names:
        env0 = gym.make(env_name)
        assert isinstance(env0.action_space,  gym.spaces.Discrete), 'env must have discrete_action_space'

        obs_dim = env0.observation_space.shape
        act_dim = env0.action_space.n
        obs_spec = ObservationSpec(obs=latent_obs_dim)
        act_spec = ActionSpec(dis=(act_dim, ))

        ENV = GymWrapper(env=env0, obs_spec=obs_spec, act_spec=act_spec, rescale_image=latent_obs_dim[:-1])
        ENV.random_self_action()
        print('random_self_action_successful')

        if policy is None:
            policy_network = create_simple_conv(latent_obs_dim, act_dim, actv='relu', output_actv='sigmoid', seed_i=555)
            policy = SimplePolicy(policy_network, obs_spec, act_spec)

        if value_network is None:
            value_network = create_simple_conv_low(latent_obs_dim, act_dim, actv='relu', output_actv='sigmoid', seed_i=555)
        
        ALG = ReinforcePG(ENV, policy, gamma=0.99, max_steps=10000, optimizer=None, grad_clip_norm=2., policy_lr=0.0002)
        ALG.entropy_rate = 0.5
        ALG._value_reg = 0.1
        ALG.model_save_dir = 'Models/{}/{}'.format(env_name, type(ALG).__name__)

        summarizer = tf.summary.create_file_writer('{}/{}/{}'.format(dir_, env_name, type(ALG).__name__))

        with summarizer.as_default():
            rs = ALG.train(epoches)
        ALG.policy.save_model()

        print('finish_training_env', env_name)

        hist_model.append(deepcopy(ALG.policy.network))
        hist_algorithm.append(deepcopy(ALG))

    return hist_model, hist_algorithm

with tf.device('/GPU:0'):
    N2 = run_atari_rf(env_names=gym_env_exp[:2], epoches=30, policy=None, value_network=None, latent_obs_dim=(80, 80, 4), dir_='Run')
    
    N0 = run_atari_ppo(env_names=gym_env_exp[:2], epoches=30, policy=None, value_network=None, latent_obs_dim=(80, 80, 4), dir_='Run')

    N1 = run_atari_vg(env_names=gym_env_exp[:2], epoches=30, policy=None, value_network=None, latent_obs_dim=(80, 80, 4), dir_='Run')

    
