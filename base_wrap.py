from env_utils import * 
from utils import *
from trajectory import *

import tensorflow as tf
import numpy as np
import gym

class BaseWrap:

	def __init__(self, env=None, obs_spec:ObservationSpec=None, act_spec:ActionSpec=None):
		
		self.env = env
		self.observation_spec = obs_spec
		self.action_spec = act_spec
		self.clip_action = 1
		self._id = None
		self._seed = 555 
		self._max_episode_step = None

	def reset(self):
		'''Reset the environment
		return: state with shape of list of observations
		'''
		s = self._reset()
		return self._prepS_fn(s)

	def step(self, x, validate_action=True):
		'''Take one step in an environment, preprocess action A and output state S at once
		inputs:
			x: Array of shape (optional) [discrete_branch+continous_branch], note that I use float for discrete case too
				assume that discrete action is put first
		'''
		x = self._prepA_fn(x)

		if validate_action:
			x = self.validate_action(x)

		S, R, is_done, info = self._step(x)
		S = self._prepS_fn(S)

		return S, R, is_done, info

	def close(self):
		'''Close an env'''
		self._close()

	def validate_action(self, x):
		'''validate that action A has true dim and dtype'''

		if not len(x) == len(self.action_spec):
			x = flatten_shape(x)
			assert len(x) == len(self.action_spec), 'input length do not match the given action spaces'
		
		if self.action_spec.is_discrete(): # check valid point since I use discrete action as float not int
			x_dic = x[:len(self.action_spec.DISCRETE_SPACE)]
			assert tf.experimental.numpy.allclose(
				tf.cast(x_dic, dtype=tf.int32), x_dic, rtol=1e-09), 'discrete action do not "all close" ' # accept np array not tensor
		
		if self.clip_action != 0 and self.action_spec.is_continuous(): # clip action value for continuous case
			x[self.action_spec.DISCRETE_SPACE:] = tf.clip_by_value(x[self.action_spec.DISCRETE_SPACE:], 
				self.clip_action, -self.clip_action)
		return x

	def random_self_action(self, validate_action=True, rs=10):
		'''Use for trivial testing environment run correctly'''
		_ = self.reset()
		for i in range(rs):
			dis_a = uniform_dis_action(self.action_spec.DISCRETE_SPACE)
			con_a = uniform_con_action(self.action_spec.CONTINUOUS_SPACE)
			rand_a = tf.concat([dis_a, con_a], axis=-1)
			_ = self.step(rand_a, validate_action)

	def _reset(self):
		raise NotImplementedError

	def _step(self, x):
		raise NotImplementedError

	def _close(self):
		raise NotImplementedError

	def _prepS_fn(self, x):
		'''preprocess input state S after taking action A'''
		raise NotImplementedError

	def _prepA_fn(self, x):
		'''preprocess action and feed to reset method
		Return:
			numpy array of shape [discrete_branches + continous_branches]
		'''
		raise NotImplementedError

class GymWrapper(BaseWrap):

	def __init__(self, env, obs_spec=None, act_spec=None, rescale_image=None):

		super().__init__(env=env, obs_spec=obs_spec, act_spec=act_spec)
		self.rescale_image = rescale_image

	def _reset(self):
		s = self.env.reset()
		return s

	def _step(self, actions):
    	
		assert not self.action_spec.is_continuous(), 'action is not discrete, cannot transform it'
		disc_action = actions.numpy().astype(np.int32)[0]
		S_t, R_t, is_done_t, info = self.env.step(disc_action)
		
		return S_t, R_t, is_done_t, info

	def _close(self):
		self.env.close()
	
	def _prepA_fn(self, actions):
		return actions

	def _prepS_fn(self, state):
		if self.rescale_image:
			# env need to output image
			#state = state / 255.
			state = tf.image.resize(state, size=self.rescale_image)
			state = tf.image.rgb_to_grayscale(state) / 255
		return state 


class UnityWrapper(BaseWrap):
	'''Wrapper for Unity based environment'''
	pass

class MultiAgentWrapper(BaseWrap):
	'''Using multi-agent setting, wrap from BaseWrap'''
	pass