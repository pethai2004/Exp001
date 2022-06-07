from base_wrap import ActionSpec, ActionTYPE, GymWrapper, ObservationSpec
from nets import create_simple_conv
from policies import SimplePolicy

import gym 
import tensorflow as tf
import numpy as np

gym_env = gym.make("ALE/Adventure-v5")
act_spec = ActionSpec(dis=(18, ), cont=0, action_type=ActionTYPE.DISCRETE)
obs_spec = ObservationSpec(obs=gym_env.observation_space.shape)
wrapped = GymWrapper(env=gym_env, obs_spec=obs_spec, act_spec=act_spec)

network0 = create_simple_conv(input_dim=obs_spec.OBSERVATION_SPACE, output_dim=18)

policy0 = SimplePolicy(network0, obs_spec, act_spec)

print('succes')

for i in range(3):
    s = wrapped.reset()
    for k in range(100):
        a = policy0.forward_policy(s)
        s, r, d, _ = wrapped.step(a)
        if d:
            break

print('yes ok!')