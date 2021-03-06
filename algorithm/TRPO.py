import copy
from collections import namedtuple

import numpy as np
import tensorflow as tf

from utils import *
from base_agent import ActorCritic
from distributions import *
from runner import GymRunner

EPS = 0.00000001

def conjugate_gradient(Ax, b, max_iters=100, tol=0.001, CG_time_step=0):
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
        CG_time_step+=1
        if tol >= np.linalg.norm(r):
            tf.summary.scalar('cg_time_elasp', CG_time_step, CG_time_step)
            return x, CG_time_step
        
        r_dot_new = np.dot(r, r)
        p = r + (r_dot_new / (r_dot_old + EPS)) * p
        r_dot_old = r_dot_new

    return x, CG_time_step

class TrustRegionPO(ActorCritic):

    def __init__(self, env, policy, value_network=None, gamma=0.99, max_trj=3000, summary=True, delta=2., use_line_search=20,
                max_cg=20, tol_cg=0.4, damping=0, assign_first=True, optimizer=None, grad_clip_norm=1., policy_lr=0.009, value_lr=0.01, use_gae=True, lam=0.9):
        '''Trust Region Policy Optimization instance
        Input: 
            delta (float) : KL-constraint for TRPO
            use_line_search (int) : max number of line search, if 0 then no line search is used.
            max_cg (int) : max number of for conjugate gradient iterations
            tol_cg (float) : (0 - 1) the decay step size in line search
            assign_first (bool) : whether the line search should pick the first alpha once it found the good step size
                                , if False then it will keep store step size alpha and continue reaching its number of max line search and then pick from this set.
                                For faster line search use assign_first=True, if want the best use assign_first=False
        '''
        super().__init__(env, policy, value_network, gamma, summary, optimizer, grad_clip_norm, policy_lr, value_lr, use_gae, lam)
        self.parameters = self.policy.get_variable()
            
        self.delta = delta
        self.max_value_epoch = 80
        self.max_policy_epoch = 80
        self.max_trj = max_trj
        self.use_line_search = use_line_search
        self.max_cg = max_cg
        self.tol_cg = tol_cg
        self.damping = damping
        self.assign_first = assign_first
        self.entropy_rate = 0.1
        self.k_train_step_counter =  tf.compat.v1.train.get_or_create_global_step()
        self.step_runner = tf.compat.v1.train.get_or_create_global_step()

    def update_value(self, S, R):
        '''is update independently from policy updation'''
        with tf.GradientTape() as tape:
            l_value = self.value_loss(S_buf=S, R_buf=R, training=True)
        g_value = tape.gradient(l_value, self.value_network.trainable_variables)
        self.value_opt.apply_gradients(zip(g_value, self.value_network.trainable_variables))

        return l_value, g_value

    def policy_loss(self, S, A, ADV, log_prob0):
        'surrogate objective of policy gradient'
        logits1 = self.policy.forward_network(S)

        if len(self.env.action_spec.DISCRETE_SPACE) == 1:
            logits1 = tf.expand_dims(logits1, 1) 
        self.policy.distribution.assign_logits(logits1)
        log_prob1 = self.policy.distribution.log_prob(A) 

        assert log_prob1.shape == ADV.shape
        ratio = tf.math.exp(log_prob1 - log_prob0)
        p_loss = - tf.reduce_mean(ratio * ADV)
        p_loss = tf.debugging.check_numerics(p_loss, 'p_loss')
        
        ent = tf.reduce_mean(self.policy.distribution.entropy()) if self.entropy_rate > 0 else 0.

        if self._policy_reg > 0.:
            p_var_reg = (v for v in self.policy.network.trainable_weights if 'kernel' in v.name)
            p_reg_loss = tf.nn.scale_regularization_loss([tf.nn.l2_loss(v) for v in p_var_reg]) * self._policy_reg
            p_reg_loss = tf.debugging.check_numerics(p_reg_loss, 'p_reg_loss')
        else:
            p_reg_loss = 0.  

        all_p_loss = p_loss + ent * self.entropy_rate + p_reg_loss * self._policy_reg
        
        return all_p_loss, logits1, p_loss, ent, p_reg_loss

    def forward_trpo(self, S, A, ADV, log_prob0, direct_hessian=True):
        var_policy = self.policy.network.trainable_variables
        assert var_policy, 'no variable has been initialized for given policy.network'
        
        with tf.GradientTape() as tape:
            pol_loss, logits1_k, p_l, e_l, r_l = self.policy_loss(S, A, ADV, log_prob0) # 'g' policy grad of loss with respect to params
        gk = tape.gradient(pol_loss, var_policy)
        assert None not in gk, 'cannot compute gradient'
        gk = flatten_shape(gk) 
        gk = tf.debugging.check_numerics(gk, 'gk_forward_trpo')
        
        HVX = lambda x_vec : self.direct_hessian(s_batch=S, vec_batch=x_vec, damp=self.damping)
        # jax : grad(grad(f).dot(v))
        if direct_hessian:
            Ftt = HVX
        else:
            raise NotImplementedError('not support FIM')

        if self.max_cg == 0: # not using cg, fall back to solving generic system of equation
            raise NotImplementedError

        x, step_cg = conjugate_gradient(Ftt, gk, max_iters=self.max_cg, tol=self.tol_cg) # solve Hx = g
        print('step_conjugate : ', step_cg)
        print('x_solve : ', x)
        x = tf.debugging.check_numerics(x, 'cg_x')
        tf.summary.scalar('norm_solved_x', norm_x(x), self.k_train_step_counter)
        denom = np.dot(x, Ftt(x))
        print('donominator : ', denom)
        denom = tf.debugging.check_numerics(denom, 'denom')
        sqrt = tf.math.sqrt(2 * self.delta / denom)
        sqrt = tf.debugging.check_numerics(sqrt, 'sqrt')
        direction = tf.cast(sqrt, dtype=tf.float32) * x
        direction = tf.debugging.check_numerics(direction, 'direction')

        if self.use_line_search > 0:

            def line_search(alpha=1., tol=self.tol_cg, max_ls=self.use_line_search):
                # add first param0, loop though and if 'goodness' is better then  use it in place, after end max_iter, pick the best alpha from it. 
                # if failed then return alpha = 1, DO : use 'satisfied' for counting iters line_search that it find line search.
                self.policy.assign_variable(self.parameters + alpha * direction, from_flat=True)
                eval0, kl0 = self.evals(s=S, a=A, adv=ADV, lp0=log_prob0)
                id_alpha = (alpha, eval0, kl0) # (alpha, loss, kl)
                satisfied = 0

                for ls_T in range(self.use_line_search):
                    alpha *= tol
                    x1 = self.parameters + alpha * direction
                    self.policy.assign_variable(x1, from_flat=True)
                    eval1, kl1 = self.evals(s=S, a=A, adv=ADV, lp0=log_prob0)

                    satisfied += 1
                    if eval0 >= eval1 and kl0 >= kl1:
                        if self.assign_first:
                            return alpha, satisfied
                        else:
                            if eval1 <= id_alpha[1] and kl1 <= id_alpha[2]:
                                id_alpha = (alpha, eval1, kl1)

                            if ls_T >= max_ls -1:
                                # return id_alpha[0] is the best
                                return id_alpha[0], satisfied # satisfied will always equal to self.use_line_search
                    if ls_T  >=  max_ls - 1:
                        return 1., 0

            a, satisfies_LS = line_search(alpha=1., tol=0.9, max_ls=10)
        else:	
            a = 1. # not using line search
        self.parameters = self.parameters + a * direction
        self.policy.assign_variable(self.parameters, from_flat=True)

        norm_grad_p = tf.norm(gk)
        norm_direction = tf.norm(direction)
        norm_params = tf.norm(self.parameters)
        
        tf.summary.scalar('step_cg_', step_cg, self.k_train_step_counter)
        
        return pol_loss, norm_grad_p, norm_direction, norm_params, a, satisfies_LS

    def fim_hessian(self, vec):
        raise NotImplementedError('not support FIM')

    def kl_div(self, params0=None, params1=None, obs=None):
        if obs is not None and params1 is None:
            params1 = self.policy.forward_network(obs, training=True)
            #self.policy.distribution.assign_logits(params1)

        if self.env.action_spec.is_discrete():
            kl_t = Cat_KL(params0, params1)
        elif self.env.action_spec.is_continuous():
            kl_t = Gauss_KL(*params0, *params1,)
        else:
            if not isinstance(self.policy.distribution, [DiscreteDistrib, ContinuousDistrib]):
                raise NotImplementedError('not support calculating kl from given BasePolicy.distribution')
    
        kl_t = tf.debugging.check_numerics(kl_t, 'kl_0')
        
        return kl_t

    def direct_hessian(self, s_batch, vec_batch, damp=0): 
        '''direct computation of Hessian vector product
        params:
            vec_batch: any vector to be multiplied with hessian
        '''
        params = self.policy.network.trainable_variables

        with tf.GradientTape() as tape_2:
            with tf.GradientTape() as tape_1:
                kl_k = self.kl_div(obs=s_batch, params0=self.old_logits) ### gradient g of kl-divergence not policy gradient 
                kl_k = tf.reduce_mean(kl_k)
                
            g = tape_1.gradient(kl_k, params)
            g = flatten_shape(g) 
            g = tf.reduce_sum(g * vec_batch)

        h = tape_2.gradient(g, params)
        h = flatten_shape(h)
        h = tf.debugging.check_numerics(h, 'hessian')

        if damp>0:
            h += damp * vec_batch
        return tf.linalg.diag(h)

    def evals(self, s, a, adv, lp0):
        '''evaluate line search step, return rewards and kl-divergence'''
        loss_eval, logits_new, _, ent, _ = self.policy_loss(s, a, adv, lp0)
        kl_eval = tf.reduce_mean(self.kl_div(params0=self.old_logits, params1=logits_new))

        tf.summary.scalar('eval_loss', loss_eval, self.k_train_step_counter)
        tf.summary.scalar('eval_kl', kl_eval, self.k_train_step_counter)
        tf.summary.scalar('eval_entropy', ent, self.k_train_step_counter)

        return loss_eval, kl_eval

    def _train(self, epochs=30):
    
        for ep in range(epochs):
    
            DATA = self.preprocess_data(max_trj=self.max_trj, validate_trj=True, compute_log_prob=True)
            #step_type, observations, returns, actions, log_prob0, advantage, value_pred
            batch_p_l, batch_n_g, batch_n_d, batch_n_p, batch_a_k, batch_st_ep = [], [], [], [], [], []
            batch_v_l, batch_v_g= [], []

            for po_e in range(self.max_policy_epoch):

                p_l, n_g, n_d, n_p, a_k, st_ep = self.forward_trpo(DATA.observations, DATA.actions, DATA.advantages, DATA.log_prob0)
                
                tf.summary.scalar('policy_loss_k', p_l, self.k_train_step_counter)
                tf.summary.scalar('policy_grad_norm_k', n_g, self.k_train_step_counter)
                tf.summary.scalar('norm_direction_k', n_d, self.k_train_step_counter)
                tf.summary.scalar('norm_params_k', n_d, self.k_train_step_counter)
                tf.summary.scalar('step_size_TRPO_k', a_k, self.k_train_step_counter)
                tf.summary.scalar('step_linesearch_count_k', st_ep, self.k_train_step_counter)
                self.k_train_step_counter.assign_add(1)
            
            for vo_e in range(self.max_value_epoch):
                value_loss, value_grad = self.update_value(DATA.observations, DATA.returns)
                
            self.train_step_counter.assign_add(1)
            self.policy.save_model()

    def _forward_trajectory(self, max_step=100):
        TRJ = GymRunner(self.env, self.policy, max_step, self.step_runner )
        return TRJ