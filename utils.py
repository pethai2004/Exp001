from base_wrap import *
from distributions import *
from trajectory import *

import time

def create_distribution(act_spec: ActionSpec):
    dic = len(act_spec.DISCRETE_SPACE) # length of discrete branches
    con = act_spec.CONTINUOUS_SPACE

    if dic >= 1 & con == 0 :
        return DiscreteDistrib(branches=act_spec.DISCRETE_SPACE, dtype=tf.float32)
    if dic == 0 & con >= 1 :
        return ContinuousDistrib(branches=con, dtype=tf.float32)
    if dic >= 1 & con >=1  :
        raise NotImplementedError('no distribution support for mixed control')

def compute_grad_flat(f, x, tape):
    return flatten_shape(tape.gradient(f, x))

def discount_rewards(r, gamma=0.99, value_next=0.0):
    discounted_r = np.zeros_like(r)
    running_add = value_next
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return tf.stop_gradient(discounted_r)

def reverse_var_shape(var_flatten, net0):
    var_flatten = np.array(var_flatten)
    v = []
    prods = 0
    for each_layer in net0.get_weights():
        shape= each_layer.shape
        prods0 = int(prods + np.prod(shape))
        v.append(var_flatten[prods:prods0].reshape(shape))
        prods = prods0
    return v 

def assign_from_flat(network, params):
	vars_ = reverse_var_shape(params, network)
	network.set_weights(vars_)

def flatten_shape(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)

def flatten_shape_batch(xs):
    return  tf.reshape(xs, (xs.shape[0], intprod(xs.shape[1:])))
    
def clip_gradient(grad_and_var, max_norm):
    '''cliping multiple gradient'''   	
    clipped_grad = []
    for grad, var in grad_and_var:
        grad = tf.clip_by_norm(grad, max_norm)
        clipped_grad.append((grad, var))

    return clipped_grad

def flatgrad(grads, var_list, clip_norm=None):
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])
def numel(x):
    return intprod(var_shape(x))
def intprod(x):
    return int(np.prod(x))
def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out)
    return out

def reward_from_trj(trj):
    slices = get_slice_from_step_type(trj.step_type)
    R_hist = []
    for i in slices:
        R_hist.append(np.sum(trj.rewards[i]))
    return tf.convert_to_tensor(R_hist)

class Timer(object):
    
	def __init__(self):
		self._accumulator = 0
		self._last = None
	def __enter__(self):
		self.start()
	def __exit__(self, *args):
		self.stop()
	def start(self):
		self._last = time.time()
	def stop(self):
		self._accumulator += time.time() - self._last
	def value(self):
		return self._accumulator
	def reset(self):
		self._accumulator = 0

def GAE(values, rewards, gamma=0.99, lam=0.95):
    """Computes generalized advantage estimation (GAE) see https://arxiv.org/abs/1506.02438"""
    try:
        rewards = rewards.numpy().flatten()
        values = values.numpy().flatten()
    except:
        rewards = rewards.flatten()
        values = values.flatten()
    D = rewards[:-1] + gamma * values[1:] - values[:-1]
    advs = discount_rewards(D, gamma * lam)
    rs = discount_rewards(rewards, gamma)[:-1]

    return tf.stop_gradient(advs), tf.stop_gradient(rs)

def get_slice_from_step_type(step_type):
	s0 = 0
	s1 = 0
	p = []
	for t, i in enumerate(step_type.numpy()):
		if i == 0:
			s0 = t
		if i == (2 or 3):
			s1 = t + 1
			p.append(slice(s0, s1))
	return p

def discount_rewards(r, gamma=0.99, value_next=0.0):
    discounted_r = np.zeros_like(r)
    running_add = value_next
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return tf.stop_gradient(discounted_r)

def set_last_v_zero(buffer, x):
    x = x.numpy() if isinstance(x, tf.Tensor) else x
    slices = get_slice_from_step_type(buffer.step_type)
    last_slices = [i.stop for i in slices]
    splited = np.array_split(x, last_slices)
    for i in range(len(splited)):
        splited[i] = np.append(splited[i], 0)

    return splited

def compute_adv_from_Buff(B:FullBuffer, v_pred:tf.Tensor, gamma=0.99, lam=0.99):
    '''Compute advantage estimation from FullBuffer with
    Input: 
        B (FullBuffer)
        v_pred (Tensor) : value prediction with shape [batch, 1]
    Returns:
        advantage : a tensor of shape [batch, 1]
        returns : a tensor of shape [batch, 1]
        '''
    R = set_last_v_zero(B, B.rewards)
    v_pred = set_last_v_zero(B, v_pred)
    gae_adv =  []
    r_dis = []
    for r0, v0 in zip(R, v_pred):
        Out = GAE(v0, r0, gamma, lam)
        gae_adv.append(Out[0])
        r_dis.append(Out[1])
    gae_adv = tf.cast(tf.expand_dims(tf.concat(gae_adv, axis=0), -1), dtype=tf.float32)
    r_dis = tf.cast(tf.expand_dims(tf.concat(r_dis, axis=0), -1), dtype=tf.float32)

    return gae_adv, r_dis

def normalize_advantage(adv, axes=(0, 1)):
    m, v = tf.nn.moments(adv, axes=axes, keepdims=True)
    normalized_adv = tf.nn.batch_normalization(adv, m, v, offset=None, scale=None, variance_epsilon=1e-8)
    return normalized_adv

class RunningStat:
    '''Class use for update normalization of the observations'''
    def __init__(self):
        self.K = 0.
        self.n = 0.
        self.Ex = 0.
        self.Ex2 = 0.

    def add(self, x):
        if len(x.shape) > 1:
            x = tf.reshape(x, -1)
        if self.n == 0.:
            self.K = x
        self.n += 1
        self.Ex += x - self.K
        self.Ex2 += (x - self.K) * (x - self.K)

    def remove(self, x):
        if len(x.shape) > 1:
            x = tf.reshape(x, -1)
        self.n -= 1
        self.Ex -= x - self.K
        self.Ex2 -= (x - self.K) * (x - self.K)

    def get_mean(self):
        return tf.reduce_mean(self.K + self.Ex / self.n)

    def get_var(self):
        return tf.reduce_mean((self.Ex2 - (self.Ex * self.Ex) / self.n) / (self.n - 1))

    def get_std(self):
        '''should not be called if only data is added only once'''
        return tf.math.sqrt(self.get_var())

    def get_mean_and_std(self):
        return self.get_mean(), self.get_std()

def mean_and_std(x):
    if isinstance(x, np.ndarray):
        m, std = x.mean(), x.std()
    elif isinstance(x, tf.Tensor): 
        m, std = tf.math.reduce_mean(x), tf.math.reduce_std(x)
    return m, std
    
def normalizing(x):
    m, std = mean_and_std(x)
    return (x - m) / std

def denormalizing(x):
    m, std = mean_and_std(x)
    return x * std + m

def clip_and_norm(x, high=1, low=-1):
    return normalizing(tf.clip_by_value(x, low, high))

def norm_x(x):
    x = flatten_shape(x)
    return tf.norm(x)