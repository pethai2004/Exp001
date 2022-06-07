import tensorflow as tf
import tensorflow_probability as tfp 
import numpy as np 

from typing import NamedTuple

class BaseDistribution:

    def __init__(self, branches=None, dtype=tf.float32):
        self.branches = branches
        self.dtype =dtype
    def assign_logits(self, logit):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def log_prob(self, inputs):
        raise NotImplementedError
    def prob(self, inputs):
        raise NotImplementedError 
    def sample(self, n):
        raise NotImplementedError 
    def variance(self):
        raise NotImplementedError 
    def kl_divergence(self, other):
        raise NotImplementedError
    def mean(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError 
    def stddev(self):
        raise NotImplementedError

class DiscreteDistrib(BaseDistribution):
    '''Supporting both single and multiple discrete action space

    Inputs: 
        branches (list) : number of action branches, eg. [5, 3, 7, 4]
        dtype (type) : type of return calculation
    '''
    def __init__(self, branches, dtype=tf.float32):
        super().__init__(branches=branches, dtype=dtype)
        self._logits = None
        
    def assign_logits(self, x):
        """ assigning logits to distribution
        Input : 
            x : shape of [None, branches, max_actions] or sequential call with shape with shape [branches, max_actions]
        """
        assert x.shape[-2] == len(self.branches), 'x must have same length as branches'
        assert x.shape[-1] == max(self.branches), 'x must match the max branches'
        
        self._logits = tf.cast(x, dtype=self.dtype)

    def sample(self, n=1, sampling_dtype=tf.int32): 
        ''' Sample from logits either batch or sequential
        return : 
            1. if assign_logits sequential, return sample of shape [branches]
            2. if assign_logits of batch, return sample of shape [batch, branches]
        '''
        if len(self._logits.shape) > 2:
            res = tf.reduce_prod(self._logits.shape[:-1])
            
            lg = tf.reshape(self._logits, (res, self._logits.shape[-1]))
            draws = tf.random.categorical(lg, 1)
            draws = tf.transpose(
                tf.reshape(draws, self._logits.shape[:-1])
            )
        else:
            draws = tf.random.categorical(self._logits, n, dtype=tf.int32)
            return tf.cast(tf.reshape(draws, -1), dtype=sampling_dtype)
        
        return tf.cast(tf.transpose(draws), dtype=sampling_dtype)

    def log_prob(self, k, sum_branches=True):
        ''' Calculate log-prob from logits
        input :
            k (array of type int) : indexes of samples
            sum_branches (bool) : specify either to sum the log prob branches together or return as in is
        return : 
            1. if sum_branches is True, then return shape of [batch, 1]
            2. if sum_branches is False, then return shape of [batch, branches]
            3. if logit has to batch dim, then no batch dim return
        '''
        lp = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=k, logits=self._logits)
        if sum_branches:
            return tf.reduce_sum(lp, axis=-1, keepdims=True)

        return tf.cast(lp, dtype=self.dtype)

    def entropy(self, sum_branches=True):
        '''Calculate entropy of given logits
        inputs : sum_branches (bool), specify either to sum the entropy branches together or return as in is
        '''
        logits = self._logits - tf.reduce_max(self._logits, axis=-1, keepdims=True)
        lse_logits = tf.reduce_logsumexp(logits, axis=-1)

        masked_logits = tf.where(
            (tf.math.is_inf(logits) & (logits < 0)),
            tf.cast(1.0, dtype=self.dtype), logits)
        
        ent = lse_logits - tf.reduce_sum(
            tf.math.multiply_no_nan(masked_logits, tf.math.exp(logits)),
            axis=-1) / tf.math.exp(lse_logits)
        
        if sum_branches:
            ent = tf.reduce_sum(ent, axis=-1, keepdims=True) # sum over branches
        
        return ent

    def deterministic(self):
        '''Mode of distribution'''
        ms = tf.argmax(self._logits, axis=-1)
        return ms

    def kl_divergence(self, other, sum_branches=True):
        '''Calculate entropy of given logits
        inputs : 
            other : either instance of DiscreteDistrib or tensor
            sum_branches (bool), specify either to sum the KL branches together or return as in is
        raise:
            ValueError if other is instance of DiscreteDistrib and logit is None
        '''
        if isinstance(other, DiscreteDistrib):
            assert other._logits is not None, "other' is instance of DiscreteDisbrib has None _logits"
            b_logits = other._logits
        else:
            b_logits = tf.convert_to_tensor(other)
            
        Multsum = tf.reduce_sum(
            tf.math.multiply_no_nan(
                tf.math.log_softmax(self._logits) - tf.math.log_softmax(b_logits),
                tf.math.softmax(self._logits)),
            axis=-1)

        if sum_branches:
            Multsum = tf.reduce_sum(Multsum, axis=-1, keepdims=True) # sum over branches

        return Multsum

    def probs_params(self):
        z_probs = tf.math.softmax(self._logits)
        return tf.where((z_probs==0), -np.inf, z_probs) # assign -inf to zeros probs


class ContinuousDistrib(BaseDistribution):
    '''Diagonal Multivariate Gaussian Distribution
    inputs:
        branches (list) : number of action space
    '''
    def __init__(self, branches: int, dtype=tf.float32):
        super().__init__(branches=branches, dtype=dtype)
        self._mean = None
        self._std = None
        
    def assign_logits(self, mean, std=None):
        
        assert (mean.shape[-1] == self.branches), 'last dim of mean must be equal to branches'
        assert len(mean.shape) <= 2, 'dimension of mean must be <= 2'
        
        if std is None:
            self._std = tf.ones_like(mean, dtype=self.dtype)
        else: #TODO : use assert same shape with mean instead
            assert (std.shape[-1] == self.branches), 'last dim of std must be equal to branches'
            assert len(std.shape) <= 2, 'dimension of std must be <= 2'
            self._std = tf.cast(std, dtype=self.dtype)
        
        self._mean = tf.cast(mean, dtype=self.dtype)
        
    def sample(self, n=1):
        
        shape = self._mean.shape
        sampled = tf.random.normal(shape=shape, mean=0., stddev=1., dtype=self.dtype)
        draws = sampled * self._std + self._mean
        return draws

    def log_prob(self, x, sum_branches=True):
        
        std = self._std
        log_unnorm = -0.5 * tf.math.squared_difference(
            x / std, self._mean / std)
        log_norm = tf.constant(
            0.5 * tf.math.log(2. * np.pi), dtype=self.dtype) + tf.math.log(std)
        if sum_branches:
            return tf.reduce_sum(log_unnorm - log_norm, axis=-1, keepdims=True)
        return log_unnorm - log_norm

    def standardize(self, x, std=None):
        return (x - self._mean) / (self._std if std is None else std)
    
    def entropy(self, sum_branches=True):
        
        log_norm = tf.constant(
        0.5 * np.log(2. * np.pi), dtype=self.dtype) + tf.math.log(self._std)
        ent = 0.5 + log_norm
        
        if sum_branches:
            return tf.reduce_sum(ent * tf.ones_like(self._mean), axis=-1, keepdims=True)
        return ent * tf.ones_like(self._mean)
    
    def kl_divergence(self, other=None, m1=None, std1=None, sum_branches=True):
        
        if other is not None:
            assert isinstance(other, ContinuousDistrib), 'other must be instance of ContinuousDistrib'
            m1, std1 = other._mean, other._std
        else:
            m1 = tf.cast(m1, dtype=self.dtype)
            std1 = tf.cast(std1, dtype=self.dtype)
            
        diff_log_scale = tf.math.log(self._std) - tf.math.log(std1)
        kl = (
            0.5 * tf.math.squared_difference(self._mean / std1, m1 / std1) +
            0.5 * tf.math.expm1(2. * diff_log_scale) - diff_log_scale
        )
        if sum_branches:
            return tf.reduce_sum(kl, axis=-1, keepdims=True)
        
        return kl


def Gauss_KL(m0, std0, m1, std1, dtype=tf.float32):
    
    m0 = tf.cast(m0, dtype=dtype)
    std0 = tf.cast(std0, dtype=dtype)
    m1 = tf.cast(m1, dtype=dtype)
    std1 = tf.cast(std1, dtype=dtype)

    diff_log_scale = tf.math.log(std0) - tf.math.log(std1)
    kl = (
        0.5 * tf.math.squared_difference(m0 / std1, m1 / std1) +
        0.5 * tf.math.expm1(2. * diff_log_scale) - diff_log_scale
    )

    return tf.reduce_sum(kl, axis=-1, keepdims=True)

def Cat_KL(a_logits, b_logits):
    '''Input of [None, action_branches, max_action]
    '''
    Multsum = tf.reduce_sum(
        tf.math.multiply_no_nan(
            tf.math.log_softmax(a_logits) - tf.math.log_softmax(b_logits),
            tf.math.softmax(a_logits)),
        axis=-1)
    
    Multsum = tf.reduce_sum(Multsum, axis=-1, keepdims=True) # sum over branches
    
    return Multsum

