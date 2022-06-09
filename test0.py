#export PYTHONPATH=/Users/pethai/Desktop/general_wrap/algorithm

from algorithm.VG import VanillaPG
from base_wrap import ActionSpec, ActionTYPE, GymWrapper, ObservationSpec
from nets import create_simple_conv
from policies import SimplePolicy
from runner import GymRunner
from utils import Timer

import gym 
import tensorflow as tf
import numpy as np

gym_env = gym.make("ALE/Adventure-v5")
act_spec = ActionSpec(dis=(18, ), cont=0, action_type=ActionTYPE.DISCRETE)
obs_spec = ObservationSpec(obs=gym_env.observation_space.shape)
wrapped = GymWrapper(env=gym_env, obs_spec=obs_spec, act_spec=act_spec)

network0 = create_simple_conv(input_dim=obs_spec.OBSERVATION_SPACE, output_dim=18)
v_net = create_simple_conv(input_dim=obs_spec.OBSERVATION_SPACE, output_dim=1)
policy0 = SimplePolicy(network0, obs_spec, act_spec)

print('succes')

times = Timer()
# with times:
#     results = GymRunner(wrapped, policy0, max_steps=10)

test_VG = VanillaPG(wrapped, policy0, v_net, gamma=0.98, lam=0.95)
print('build ok')

test_summary_writer = tf.summary.create_file_writer('run_VG/test0')
with test_summary_writer.as_default() and times:
    rs = test_VG.train(20)

print('yes ok!')

