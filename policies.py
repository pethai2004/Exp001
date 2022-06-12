from re import A
import tensorflow as tf 
from tensorflow import keras
import numpy as np

from utils import *

class BasePolicy:

	def __init__(self, network, obs_spec, act_spec, epsilon=1., clip_act=1., clip_obs=3., distribution=None):
		'''Base Policy class
		Input:
			network : the model for policy
			epsilon : determine the randomness of action, 0 correspond to fully deterministic.
		'''
		self.network = network
		self.obs_spec = obs_spec
		self.act_spec = act_spec
		self.epsilon = epsilon
		self.distribution = distribution if distribution is not None else create_distribution(self.act_spec)
		assert isinstance(self.distribution, BaseDistribution), 'distribution must subclass the BaseDistribution'
		self.clip_obs = clip_obs
		self.clip_act = clip_act

	def forward_policy(self, S, training=False):
		'''Generate action A given state input S
		Inputs:
			S : numpy array or tensor in given state S 
		Returns: 
			action A
		'''
		A = self._forward_policy(S, training=training)
		return A

	def forward_network(self, inputs, training=False):
		return self._forward_network(inputs, training)

	def fully_generated_action(self, inputs):
		A = self.fully_generated_action(inputs)
		return A

	def scheduling_eps(self):
		self._scheduling_eps()

	def get_variable(self, flatten=True):
		if flatten:
			return flatten_shape(xs=self.network.trainable_variables)
		else:
			return self.network.trainable_variables

	def assign_variable(self, inputs, from_flat=True):
		if from_flat:
			assign_from_flat(self.network, inputs)
		else:
			self.network.set_weights(inputs)

	def get_num_params(self):
		return self.get_variable.shape

	def _forward_policy(self, inputs, training=False):
		raise NotImplementedError
	def _forward_network(self, inputs, training=False):
		raise NotImplementedError
	def fully_generated_action(self, inputs):
		raise NotImplementedError
	def _scheduling_eps(self):
		raise NotImplementedError
	def save_model(self, dir_path='saved_policy/model'):
		self.network.save(dir_path)
	def load_model(self, dir_path):
		self.network = keras.models.load_model(dir_path)
		#TODO : assert the valid loaded model shape

class SimplePolicy(BasePolicy):
	'''A stochastic policy for PG algorithm'''
	def __init__(self, network, obs_spec, act_spec, epsilon=1., clip_act=1., clip_obs=3., distribution=None):
		super().__init__(network, obs_spec, act_spec, epsilon, clip_act, clip_obs, distribution)
		assert not (self.act_spec.action_type == ActionTYPE.MIXED), 'do not currently support mixed action space'
		self.old_logits = []
		
	def _forward_policy(self, inputs, training=False):
			
		reps = self._forward_network(inputs, training=training)
		reps = self._prep_logit(reps)
		self.old_logits.append(reps)
		draws = self.distribution.assign_logits(reps)
		draws = self.distribution.sample()
		assert len(draws.shape) == 1, 'draws dim must be 1'
		
		return draws

	def _forward_network(self, inputs, training=False):
		# if not training:
		# 	assert (inputs.shape == self.obs_spec.OBSERVATION_SPACE)
		if len(inputs.shape) == len(self.obs_spec.OBSERVATION_SPACE): # if equal then add batch dimension to it
			inputs = tf.expand_dims(inputs, 0)

		return self.network(inputs, training=training)

	def _prep_logit(self, inputs):
		return inputs

