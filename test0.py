#export PYTHONPATH=/Users/pethai/Desktop/general_wrap/algorithm

from algorithm.VG import VanillaPG
from base_wrap import ActionSpec, ActionTYPE, GymWrapper, ObservationSpec
from nets import create_simple_conv
from policies import SimplePolicy
from runner import GymRunner
from utils import Timer

import gym 
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils.layer_utils import count_params  

# gym_env = gym.make("ALE/Adventure-v5")
# origin_obs_space = gym_env.observation_space.shape
# repres_obs_space = [120, 75] #rescale image
# act_spec = ActionSpec(dis=(18, ), cont=0, action_type=ActionTYPE.DISCRETE)
# obs_spec = ObservationSpec(obs=[120, 75, 3])
# wrapped = GymWrapper(env=gym_env, obs_spec=obs_spec, act_spec=act_spec)

# network0 = create_simple_conv(input_dim=obs_spec.OBSERVATION_SPACE, output_dim=18)
# v_net = create_simple_conv(input_dim=obs_spec.OBSERVATION_SPACE, output_dim=1)

gym_env = gym.make('CartPole-v1')
network0 = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(85, activation='relu'),
    keras.layers.Dense(2, activation='relu')
])
network0.build((None, 4))

value_net  = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='relu')
])
value_net.build((None, 4))

obs_spec = ObservationSpec(obs=[4, ])
act_spec = ActionSpec(dis=(2, ))


policy0 = SimplePolicy(network0, obs_spec, act_spec)
env = GymWrapper(env=gym_env, obs_spec=obs_spec, act_spec=act_spec)
env.random_self_action()

test_VG = VanillaPG(env, policy0, value_net, gamma=0.97, lam=0.92, max_trj=3000)

test_summary_writer = tf.summary.create_file_writer('run_VG/test0')

with test_summary_writer.as_default():
    rs = test_VG.train(30)
test_VG.policy.save_model()
print('yes ok!')

