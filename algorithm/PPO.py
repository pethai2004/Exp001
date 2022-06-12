from base_agent import ActorCritic
from runner import GymRunner
from utils import *
from distributions import *

import tensorflow as tf
import warnings

class PPO(ActorCritic):

    def __init__(self, env, policy, value_network=None, gamma=0.90, max_trj=3000, summary=True, clip_ratio= 0.4, kl_beta=0.0, kl_target=0.01,
            kl_tol=2, kl_low=1.5, kl_cutoff=0.1, optimizer=None, grad_clip_norm=1., policy_lr=0.009, value_lr=0.01, 
            use_gae=True, lam=0.9):
        
        super().__init__(env, policy, value_network, gamma, summary, optimizer, grad_clip_norm, policy_lr, value_lr, use_gae, lam)
        self.entropy_rate = 0.05
        self.kl_penalty = 0.0
        self.optimizer = tf.keras.optimizers.Adam(self._policy_lr)
        self.max_trj = max_trj
        self.pol_iters = 80
        self.entropy_rate = 0.1

        self.clip_ratio = clip_ratio or 0.0
        self.kl_beta = kl_beta or 0.0 
        self.kl_target = kl_target or 0.01
        self.kl_tol = kl_tol or 2
        self.kl_low = kl_low or 1.5
        self.cutoff = kl_cutoff or 0.1
        self.cut_const = 0.1
        
        if self.clip_ratio == 0. and self.kl_beta == 0.:
            warnings.warn('clip ratio and kl_penalty is both zero, this is equivalent for using VG class')

    def _forward_trajectory(self, max_step=100):
        TRJ = GymRunner(self.env, self.policy, max_step)
        return TRJ

    def _train(self, epochs=30):

        for ep in range(epochs):
    
            DATA = self.preprocess_data(max_trj=self.max_trj, validate_trj=True, compute_log_prob=True)

            var_to_train = self.value_network.trainable_variables + self.policy.network.trainable_variables
            assert var_to_train, 'no variable is initialized'
            m_V, m_P, m_L, m_ng, m_et, m_kl = [], [], [], [], [], []
            
            for Epps in range(self.pol_iters):
                with tf.GradientTape() as tape:
                    v_loss_k = self.value_loss(DATA.observations, DATA.returns, training=True, summary=self.summary)
                    p_loss_k, log_prob1, ent1, kl1 = self.policy_loss(DATA.observations, DATA.actions, DATA.advantages, DATA.log_prob0, summary=self.summary)
                    LOSS = v_loss_k * self.value_loss_const + p_loss_k
                grads = tape.gradient(LOSS, var_to_train)
                assert grads, 'gradient cannot be computed'
                if self.grad_clip_norm:
                    grads = tf.clip_by_global_norm(grads, self.grad_clip_norm)

                self.optimizer.apply_gradients(zip(grads, var_to_train))

                m_V.append(v_loss_k)
                m_P.append(p_loss_k)
                m_L.append(LOSS)
                m_ng.append(norm_x(grads))
                m_et.append(ent1)
                m_kl.append(kl1)

            if self.summary:

                grad_norm = tf.reduce_mean(m_ng)
                p_params_norm = tf.norm(self.policy.get_variable(flatten=True))
                v_params_norm = tf.norm(flatten_shape(self.value_network.trainable_variables))
                tf.summary.scalar('grad_norm', grad_norm, self.train_step_counter)
                tf.summary.scalar('p_params_norm', p_params_norm, self.train_step_counter)
                tf.summary.scalar('v_params_norm', v_params_norm, self.train_step_counter)

                tf.summary.scalar('Loss', tf.reduce_mean(m_L), self.train_step_counter)
                tf.summary.scalar('v_loss', tf.reduce_mean(m_V), self.train_step_counter)
                tf.summary.scalar('p_Loss', tf.reduce_mean(m_P), self.train_step_counter)

                tf.summary.scalar('advantage', tf.reduce_mean(DATA.advantages), self.train_step_counter)
                log_prob_ratio = tf.reduce_mean(tf.exp(DATA.log_prob0 - log_prob1))
                tf.summary.scalar('log_prob_ratio', log_prob_ratio, self.train_step_counter)
                tf.summary.scalar('entropy', tf.reduce_mean(m_et), self.train_step_counter)
                tf.summary.scalar('kl_divergence', tf.reduce_mean(m_kl), self.train_step_counter)
                with tf.name_scope('step_size/'):
                    p = self.optimizer.learning_rate
                    tf.summary.scalar('stepsize', p, self.train_step_counter)
            
            self.train_step_counter.assign_add(1)

    def get_all_loss(self, obs, act, adv, lp0, ret):
        V_LOSS = self.policy_Loss(obs, act, adv, lp0)
        P_LOSS = self.value_loss(obs, ret, training=True)

    def policy_loss(self, S_batch, A_batch, ADV_b, log_prob0, summary=True):

        logit1 = self.policy.forward_network(S_batch)
         # expand from (batch, action_space) to include branches dim, [batch, branches, action_space]
         # for supporting input in distribution.assign_logit
        if len(self.env.action_spec.DISCRETE_SPACE) == 1:
            logit1 = tf.expand_dims(logit1, 1) 
        self.policy.distribution.assign_logits(logit1)
        log_prob1 = self.policy.distribution.log_prob(A_batch) 

        assert log_prob1.shape == ADV_b.shape
        ratio = tf.math.exp(log_prob1 - log_prob0)
        clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        ratio = tf.debugging.check_numerics(ratio, 'ratio')
        clipped = tf.debugging.check_numerics(clipped, 'clipped_ratio')
        L1 = ratio * ADV_b
        L2 = clipped * ADV_b
        L_min = tf.minimum(L1, L2)
        p_loss = L_min if self.clip_ratio > 0 else L1
        p_loss = - tf.reduce_mean(p_loss)
        p_loss = tf.debugging.check_numerics(p_loss, 'p_loss')

        if self._policy_reg > 0.:
            p_var_reg = (v for v in self.policy.network.trainable_weights if 'kernel' in v.name)
            p_reg_loss = tf.nn.scale_regularization_loss([tf.nn.l2_loss(v) for v in p_var_reg]) * self._policy_reg
            p_reg_loss = tf.debugging.check_numerics(p_reg_loss, 'p_reg_loss')
            tf.summary.scalar('p_reg_loss', p_reg_loss, self.train_step_counter)        
        else:
            p_reg_loss = 0.   
        if self.kl_beta > 0:
            kl_penalty = self.kl_penalty(params0=self.old_logits, params1=logit1, obs=None)
        else:
            kl_penalty = 0.
        if self.entropy_rate >0.:
            ent = self.entropy_loss(A_batch, logit1)
        else:
            ent = 0.

        total_p_loss = p_loss + p_reg_loss * self._policy_reg + kl_penalty * self.kl_beta + ent * self.entropy_rate
        if summary:
            tf.summary.scalar('total_p_loss', total_p_loss, self.train_step_counter)
            tf.summary.scalar('p_loss', p_loss, self.train_step_counter)
            tf.summary.scalar('ratio', ratio, self.train_step_counter)
            tf.summary.scalar('clipped_ratio', clipped, self.train_step_counter)
        
        return total_p_loss, log_prob1, ent, kl_penalty

    def entropy_loss(self, actions, logits=None, summary=False):
        if self.entropy_rate > 0:
            with tf.name_scope('entropy_regularization'):
                if logits is not None:
                    self.policy.distribution.assign_logit(logits)
                entropy = self.policy.distribution.entropy(actions)
                entropy = - tf.reduce_mean(entropy * self.entropy_rate)
                entropy = tf.debugging.check_numerics(entropy, 'entropy_reg')
            if summary:
                tf.summary.histogram('entropy_reg', entropy, self.train_step_counter)		
        else:
            entropy = tf.constant(0.0, dtype=tf.float32)	
        return entropy * self.entropy_rate

    def kl_penalty(self, params0=None, params1=None, obs=None, summary=True):
        if obs is not None and params1 is None:
            params1 = self.policy.forward_network(obs, training=True)
            self.policy.distribution.assign_logits(params1)

        if self.env.action_spec.is_discrete():
            kls = Cat_KL(params0, params1)
        elif self.env.action_spec.is_continuous():
            kls = Gauss_KL(*params0, *params1,)
        else:
            if not isinstance(self.policy.distribution, [DiscreteDistrib, ContinuousDistrib]):
                raise NotImplementedError('not support calculating kl from given BasePolicy.distribution')
        kl_pt = tf.reduce_mean(kls)
        
        if self.cutoff > 0.:
            CoF = self.kl_target * self.cutoff
            KL_CoF = tf.maximum(kl_pt - CoF, 0.)
            cutL = self.cut_off * tf.math.square(KL_CoF)
            kl_pt += cutL
            tf.summary.scalar('kl_cutoff_count', tf.reduce_sum(tf.cast(kls > CoF)), self.train_step_counter)
            tf.summary.scalr('cut_off_loss', cutL, self.train_step_counter)
        if summary:
            tf.summary.scalar('kl_penalty', KL_CoF, self.train_step_counter)
        
        return kl_pt     

    def update_kl_beta(self, mean_kl, summary=True):
        if self.kl_beta is None:
            return tf.no_op()
        c1 = mean_kl < self.kl_target / self.kl_low
        c2 = mean_kl > self.kl_target * self.kl_low
        if c1 == True:
            new_beta = self.kl_beta / self.kl_tol
        elif c2 == True:
            new_beta = self.kl_beta * self.kl_tol
        else:
            new_beta = self.kl_beta
        new_beta = tf.maximum(new_beta, 10e-16)
        self.kl_beta = new_beta.numpy()

        if summary:
            tf.summary.scalar('kl_beta', self.kl_beta, self.train_step_counter)