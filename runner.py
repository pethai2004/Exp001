from base_wrap import *
from policies import *
from trajectory import *

import tensorflow as tf
import numpy as np
from collections import namedtuple
from queue import Queue

def GymRunner(env:GymWrapper, policy:BasePolicy, max_steps:int=10000, global_step=None):
    assert isinstance(env, GymWrapper), 'env must be subclass of BaseWrap'
    state_b, action_b, reward_b, stype_b = [], [], [], [] #TODO : try using numpy append if it is faster
    S0 = env.reset()
    step_type = StepType.FIRST
    RH = 0
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
        RH+=R
        S0 = S1

        if (is_done or T == max_steps - 1):

            S0 = env.reset()
            stype_b[-1] = step_type
            step_type = StepType.FIRST

            if global_step:
                global_step.assign_add(1)
                tf.summary.scalar('raw_reward_trj', RH, global_step)
            RH=0
            
    stype_b = tf.convert_to_tensor(stype_b, dtype=tf.int32)
    state_b = tf.convert_to_tensor(state_b, dtype=tf.float32)
    action_b = tf.convert_to_tensor(action_b, dtype=tf.float32) # use float both for continuous and discrete case
    reward_b = tf.convert_to_tensor(reward_b, dtype=tf.float32)

    return FullBuffer(stype_b, state_b, reward_b, action_b)

RESULT_EPS = namedtuple('RESULT_EPS', ['observations', 'rewards', 'actions'])

#TODO : fix summary on raw_reward, since global_step is reset everytime for REINFORCE algorithm
def GymRunner_Episodic(env:GymWrapper, policy:BasePolicy, max_steps:int=10000, global_step=None):
    '''use for REINFORCE'''
    assert isinstance(env, GymWrapper), 'env must be subclass of BaseWrap'
    assert global_step is not None

    state_b, action_b, reward_b = [], [], [] #TODO : try using numpy append if it is faster
    S0 = env.reset()

    RH = 0
    for T in range(max_steps):

        A = policy.forward_policy(S=S0, training=False)
        S1, R, is_done, _ = env.step(A)
        
        state_b.append(S0)
        action_b.append(A)
        reward_b.append([R])

        RH+=R
        S0 = S1

        global_step.assign_add(1)
        if (is_done or T == max_steps - 1):
            tf.summary.scalar('raw_reward_trj', RH, global_step)
            break

    state_b = tf.convert_to_tensor(state_b, dtype=tf.float32)
    action_b = tf.convert_to_tensor(action_b, dtype=tf.float32) # use float both for continuous and discrete case
    reward_b = tf.convert_to_tensor(reward_b, dtype=tf.float32)

    return RESULT_EPS(state_b, reward_b, action_b)

def GymRunner_EpisodicStack(env:GymWrapper, policy:BasePolicy, max_steps:int=10000, global_step=None, stack=4):
    '''use for REINFORCE'''
    assert isinstance(env, GymWrapper), 'env must be subclass of BaseWrap'
    assert global_step is not None

    state_b, action_b, reward_b = [], [], [] #TODO : try using numpy append if it is faster
    S0 = env.reset()

    RH = 0

    Q = Queue(maxsize=stack)
    for i in range(stack):
        Q.put(S0)

    for T in range(max_steps):
        Q_input = tf.concat(list(Q.queue), -1)

        A = policy.forward_policy(S=Q_input, training=False)
        S1, R, is_done, _ = env.step(A)
        
        state_b.append(Q_input)
        action_b.append(A)
        reward_b.append([R])

        RH+=R
        S0 = S1
        Q.get()
        Q.put(S0)
        
        global_step.assign_add(1)
        if (is_done or T == max_steps - 1):
            tf.summary.scalar('raw_reward_trj', RH, global_step)
            break

    state_b = tf.convert_to_tensor(state_b, dtype=tf.float32)
    action_b = tf.convert_to_tensor(action_b, dtype=tf.float32) # use float both for continuous and discrete case
    reward_b = tf.convert_to_tensor(reward_b, dtype=tf.float32)

    return RESULT_EPS(state_b, reward_b, action_b)
    
def GymRunnerStack(env:GymWrapper, policy:BasePolicy, max_steps:int=10000, global_step=None, stack=4):
    assert isinstance(env, GymWrapper), 'env must be subclass of BaseWrap'
    state_b, action_b, reward_b, stype_b = [], [], [], [] #TODO : try using numpy append if it is faster
    S0 = env.reset()
    step_type = StepType.FIRST
    RH = 0

    Q = Queue(maxsize=stack)
    for i in range(stack):
        Q.put(S0)

    for T in range(max_steps):
        Q_input = tf.concat(list(Q.queue), -1)
        A = policy.forward_policy(S=Q_input, training=False)
        S1, R, is_done, _ = env.step(A)
        
        stype_b.append(step_type)

        if is_done:
            step_type = StepType.LAST
        elif T == max_steps - 1:
            step_type = StepType.CUTT
        else:
            step_type = StepType.MID
        
        state_b.append(Q_input)
        action_b.append(A)
        reward_b.append([R])
        RH+=R
        S0 = S1

        Q.get()
        Q.put(S0)

        if (is_done or T == max_steps - 1):

            S0 = env.reset()

            Q = Queue(maxsize=stack)
            for i in range(stack):
                Q.put(S0)

            stype_b[-1] = step_type
            step_type = StepType.FIRST

            if global_step:
                global_step.assign_add(1)
                tf.summary.scalar('raw_reward_trj', RH, global_step)
            RH=0
            
    stype_b = tf.convert_to_tensor(stype_b, dtype=tf.int32)
    state_b = tf.convert_to_tensor(state_b, dtype=tf.float32)
    action_b = tf.convert_to_tensor(action_b, dtype=tf.float32) # use float both for continuous and discrete case
    reward_b = tf.convert_to_tensor(reward_b, dtype=tf.float32)

    return FullBuffer(stype_b, state_b, reward_b, action_b)

RESULT_EPS = namedtuple('RESULT_EPS', ['observations', 'rewards', 'actions'])