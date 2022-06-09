from typing import NamedTuple
import tensorflow as tf
import numpy as np 

class ActionTYPE(object):
    
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'
    MULTI_DISCRETE = 'multi_discrete'
    MIXED = 'mixed'

class ActionSpec:

	def __init__(self, dis:list=[], cont:int=0, action_type:ActionTYPE=None):
    		
		self.DISCRETE_SPACE = dis
		self.CONTINUOUS_SPACE = cont
		self.action_type = action_type if action_type is not None else self.set_type()
	
	def set_type(self):
		
		dic = len(self.DISCRETE_SPACE)
		con = self.CONTINUOUS_SPACE

		if con >= 1 and dic == 0:
			self.action_type = ActionTYPE.CONTINUOUS
		elif dic == 1 and con == 0:
			self.action_type = ActionTYPE.DISCRETE
		elif dic > 1 and con == 0:
			self.action_type = ActionTYPE.MULTI_DISCRETE
		elif con >= 1 and dic >= 1:
			self.action_type = ActionTYPE.MIXED

	def is_discrete(self):
		'''This only check whether action has discrete space, not that it is purely discrete'''
		if len(self.DISCRETE_SPACE) >=1:
			return True
		else:
			return False

	def is_continuous(self):
		'''This only check whether action has continuous space, not that it is purely continuous'''
		if self.CONTINUOUS_SPACE >=1:
			return True
		else:
			return False

	def __len__(self):
		return len(self.DISCRETE_SPACE) + self.CONTINUOUS_SPACE

	def __repr__(self):
		return 'discrete_space :' + self.DISCRETE_SPACE + 'continuous_space :' + self.CONTINUOUS_SPACE

class ObservationSpec:

    def __init__(self, obs, _max=None, _min=None):
        self.OBSERVATION_SPACE = obs
        self.max = _max
        self.min = _min
    def __repr__(self):
        return f"ObsSpec{self.OBSERVATION_SPACE}"

class StepType(object):
    '''Describe the state of time step'''
    FIRST = 0 # starting state
    MID = 1 # mid state
    LAST = 2 # terminal state
    CUTT = 3 # mid state that had been cut

    def __new__(cls, value):
        if value == cls.FIRST:
            return cls.FIRST
        elif value == cls.MID:
            return cls.MID
        elif value == cls.LAST:
            return cls.LAST
        elif value == cls.CUTT:
            return cls.CUTT
        else:
            raise ValueError(
                'No known conversion for `%r` into a StepType' % value)

class FullBuffer(NamedTuple("FullBuffer", [('step_type', tf.Tensor),
                                           ('observations', tf.Tensor),
                                           ('rewards', tf.Tensor),
                                           ('actions', tf.Tensor)])):
    def __len__(self):
        return len(self.step_type)

class FullDatasetPG(NamedTuple("FullBuffer", [('step_type', tf.Tensor),
                                               ('observations', tf.Tensor),
                                               ('returns', tf.Tensor),
                                               ('actions', tf.Tensor),
                                               ('log_prob0', tf.Tensor),
                                               ('advantages', tf.Tensor),
                                               ('value_pred', tf.Tensor)])):
    def __len__(self):
        return len(self.step_type)

    def return_shuffle(self):
        seed = np.random.randint(0, 1000)
        step_type = tf.random.shuffle(self.step_type, seed=seed)
        observations = tf.random.shuffle(self.observations, seed=seed)
        returns = tf.random.shuffle(self.rewards, seed=seed)
        actions = tf.random.shuffle(self.actions, seed=seed)
        log_prob0 = tf.random.shuffle(self.log_prob, seed=seed)
        advantage = tf.random.shuffle(self.advantage, seed=seed)
        value_pred = tf.random.shuffle(self.value_pred, seed=seed)

        return FullDatasetPG(step_type, observations, returns, actions, log_prob0, advantage, value_pred)

