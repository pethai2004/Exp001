import copy
from collections import namedtuple

import numpy as np
import tensorflow as tf

from utils import *
EPS = 0.00000001

def conjugate_gradient(Ax, b, max_iters=100, tol=0.001):
    'A is (S++)'
    x = np.zeros_like(b)
    r = copy.deepcopy(b)  
    p = copy.deepcopy(r)
    r_dot_old = np.dot(r, r)

    for _ in range(max_iters):
        z = Ax(p)
        alpha = r_dot_old /( np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        if tol >= np.linalg.norm(r):
            return x
        
        r_dot_new = np.dot(r, r)
        p = r + (r_dot_new / (r_dot_old + EPS)) * p
        r_dot_old = r_dot_new

    return x

Results = namedtuple('Results_TRJ', ['state', 'actions', 'returns', 'advantages', 'values','log_prob'])

class TRPO:

    def __init__(self, env, policy, env_spec, value_network=None, preprocess_fn=None, distribution=None, 
        delta=1.0, max_rollouts=4500, value_lr=0.01, dir_sum=None, use_line_search=True):

        self.env = env 
        self.env_spec = env_spec
        self.preprocess_fn = preprocess_fn
        self.distribution = distribution

        self.old_logits = None
        self.policy = policy
        self.value_network = value_network
        self.value_lr = value_lr
        self.value_opt = tf.keras.optimizers.Adam(value_lr)
        self.parameters = self.policy.get_flat_var()
        self.params_shape = self.policy.get_flat_shape()
        self.delta = delta
        self.gamma = 0.99
        self.lam = 0.92
        self.max_value_epoch = 50
        self.max_rollouts = max_rollouts
        self.dir_sum = dir_sum
        self.use_line_search = use_line_search
        self.state_normalizer = RunningStat()

    def run_trj(self, max_steps):

        TRJ, logits_trj = None
        acts = tf.reshape(TRJ.actions, (-1, 1))
        log_prob0 = tf.reshape(self.policy.distribution.ref_distribution(logits_trj).log_prob(acts), (-1, 1))
        v_predict = tf.stop_gradient(self.value_network(TRJ.observations))
        rewards = TRJ.rewards

        ADV, returns = compute_adv_from_Buff(TRJ, v_predict, gamma=self.gamma, lamda=self.lam)
        ADV = tf.cast(tf.expand_dims(tf.concat(ADV, axis=0), -1), dtype=tf.float32)

        ADV = tf.debugging.check_numerics(ADV, 'ADV')
        returns = tf.cast(tf.expand_dims(tf.concat(returns, axis=0), -1), dtype=tf.float32)

        ADV = normalize_advantage(ADV)
        returns = normalize_advantage(returns)
        obs = TRJ.observations
        
        return Results(obs, acts, returns, ADV, v_predict, log_prob0)
		
    def update_value(self, S, R):

        if not len(R.shape) >=2:
            R = tf.reshape(R, (-1, 1))
        assert len(S.shape) >=2, 's must have batch dim'
        assert len(R.shape) >=2, 'r must have batch dim'

        with tf.GradientTape() as tape:
            l_value = tf.reduce_mean((self.value_network(S) - R) ** 2)
        g_value = tape.gradient(l_value, self.value_network.trainable_variables)
        self.value_opt.apply_gradients(zip(g_value, self.value_network.trainable_variables))
        
        return l_value, g_value

    def policy_loss(self, S, A, ADV, log_prob0):
        
        log_prob1 = self.policy.get_log_prob(S, A)
        surr = tf.math.exp(log_prob1 - log_prob0) * ADV
        return - tf.reduce_mean(surr)

    def forward_trpo(self, S, A, ADV, log_prob0, direct_hessian=True, max_cg=20, tol_cg=0.001):
        
        with tf.GradientTape() as tape:
            p_loss = self.policy_loss(S, A, ADV, log_prob0) # 'g' policy grad of loss with respect to params
        p_loss = tf.debugging.check_numerics(p_loss, 'p_loss')
        gk = tape.gradient(p_loss, self.policy.network.trainable_variables)
        gk = flatten_shape(gk) 
        gk = tf.debugging.check_numerics(gk, 'gradient_policy')

        HVX = lambda x : self.direct_hessian(S, A, ADV, log_prob0, x, damp=0)
        if direct_hessian:
            Ftt = HVX
        else:
            raise NotImplementedError('not support FIM')

        x = conjugate_gradient(Ftt, gk, max_iters=max_cg, tol=tol_cg) # solve Hx = g
        x = tf.debugging.check_numerics(x, 'cg_x')

        direction = tf.cast(tf.math.sqrt(2 * self.delta / np.dot(x, Ftt(x))), dtype=tf.float32) * x
        direction = tf.debugging.check_numerics(direction, 'direction')

        if self.use_line_search:
                
            def line_search(alpha=1., tol=0.9, max_ls=10):

                self.policy.assign_flat_var(self.parameters + alpha * direction)
                eval0, kl0 = self.evals(s=S, a=A, adv=ADV, lp0=log_prob0)
                kl0 = tf.reduce_mean(kl0)

                for ls_T in range(max_ls):
                    alpha *= tol
                    x1 = self.parameters + alpha * direction
                    self.policy.assign_flat_var(x1)
                    eval1, kl1 = self.evals(s=S, a=A, adv=ADV, lp0=log_prob0)
                    kl1 = tf.reduce_mean(kl1)

                    if eval1 >= eval0 and kl0<=kl1:
                        return alpha
                    
                    if ls_T  >=  max_ls - 1:
                        self.policy.assign_flat_var(self.parameters) # no improve, set back to original
                        return 1.
            a = line_search(alpha=1., tol=0.9, max_ls=10)
        
        else:	
            a = 1. # not using line search
        self.parameters = self.parameters + a * direction
        self.policy.assign_flat_var(self.parameters)

    def fim_hessian(self, vec):
        raise NotImplementedError('not support FIM')

    def kl_div(self, s):
            
        logit1 = self.policy.network(s)
        logit1_dist = self.policy.distribution.ref_distribution(logit1)
        logit0_dist = self.policy.distribution.ref_distribution(self.old_logits)
        kl_t = logit1_dist.kl_divergence(logit0_dist)
        kl_t = tf.debugging.check_numerics(kl_t, 'kl_0')
        return kl_t

    def direct_hessian(self, s_batch, a_batch, adv_batch, lp_batch, vec_batch, damp=0):
        '''direct computation of Hessian
        params:
            vec_batch: any vector to be multiplied with hessian
        '''
        params = self.policy.network.trainable_variables

        with tf.GradientTape() as tape_2:
            with tf.GradientTape() as tape_1:
                kl_k = self.kl_div(s_batch)
            g = flatten_shape(tape_1.gradient(kl_k, params)) # g of kl
            g = tf.reduce_sum(g * vec_batch)

        h = tape_2.gradient(g, params)
        h = flatten_shape(h)
        h = tf.debugging.check_numerics(h, 'hessian')

        if damp>0:
            h += damp * vec_batch

        return h

    def evals(self, s, a, adv, lp0):
        '''evaluate line search step, return rewards and kl-divergence'''
        loss_eval = self.policy_loss(s, a, adv, lp0)
        kl_eval = self.kl_div(s=s)
        return loss_eval, kl_eval

    def train(self, epoch=3):

        for Ep in range(epoch):
            Results = self.run_trj(max_steps=self.max_rollouts)
            
            S_b, A_b, R_b, Adv_b, V_b, lp_b = Results

            #p_grad = self.policy_grad(S=s_b, A=a_b, ADV=adv_b, log_prob0=lp_b)
            self.forward_trpo(S_b, A_b, Adv_b, lp_b)

            for _ in range(self.max_value_epoch):
                value_loss = self.update_value(S_b, R_b)