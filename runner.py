from base_wrap import *
from policies import *
from trajectory import *

import tensorflow as tf
import numpy as np

def GymRunner(env:GymWrapper, policy:BasePolicy, max_steps:int=10000):
    assert isinstance(env, GymWrapper), 'env must be subclass of BaseWrap'
    state_b, action_b, reward_b, stype_b = [], [], [], [] #TODO : try using numpy append if it is faster
    S0 = env.reset()
    step_type = StepType.FIRST

    for T in range(max_steps):

        A = policy.forward_policy(S=S0, training=False)

        S1, R, is_done, _ = env.step(A)
        
        stype_b.append(step_type)

        if is_done:
            step_type = StepType.LAST
        elif T == max_steps - 1:
            step_type = StepType.CUTT
        else:
            step_type = StepType.MID
        
        state_b.append(S0)
        action_b.append(A)
        reward_b.append([R])

        S0 = S1
        if (is_done or T == max_steps - 1):
            S0 = env.reset()
            stype_b[-1] = step_type
            step_type = StepType.FIRST
    
    stype_b = tf.convert_to_tensor(stype_b, dtype=tf.int32)
    state_b = tf.convert_to_tensor(state_b, dtype=tf.float32)
    action_b = tf.convert_to_tensor(action_b, dtype=tf.float32) # use float both for continuous and discrete case
    reward_b = tf.convert_to_tensor(reward_b, dtype=tf.float32)

    return FullBuffer(stype_b, state_b, reward_b, action_b)
    

        