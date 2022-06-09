import tensorflow as tf
import gym
import numpy as np

from trajectory import *
def create_gym_action_spec(self):
        
    if isinstance(self.env.action_space, gym.spaces.Discrete):
        return ActionSpec(dis=(self.env.action_space.n, ), cont=0)
    if isinstance(self.env.action_space, gym.spaces.Continuous):
        return ActionSpec(dis=(), cont=self.env.action_space[0])
    if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
        return ActionSpec(dis=self.env.action_space.shape, cont=0) 
    else:
        raise NotImplementedError

def uniform_dis_action(branches, sum_axis=-1):
    max_n = max(branches)
    logits = np.ones((len(branches), max_n))
    for i, b in enumerate(branches): # assign -inf prob to every non valid action branches
        logits[i, b:] = -np.inf
    logits = tf.math.softmax(logits, axis=sum_axis)
    draw = tf.transpose(tf.random.categorical(logits, 1))[0]
    
    return tf.cast(draw, dtype=tf.float32)

def uniform_con_action(branches):
    return tf.random.normal((branches, ))


