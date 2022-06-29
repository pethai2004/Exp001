from base_agent import ActorCritic
from runner import GymRunner, GymRunnerStack
from utils import *

import tensorflow as tf

class VanillaPG(ActorCritic):

    def __init__(self, env, policy, value_network=None, gamma=0.98, max_trj=3000, summary=True, optimizer=None, 
            grad_clip_norm=2., policy_lr=0.001, value_lr=0.01, use_gae=True, lam=0.95):
        super().__init__(env, policy, value_network, gamma, summary, optimizer, grad_clip_norm, policy_lr, value_lr, use_gae, lam)
        self.entropy_rate = 0.1
        self.kl_penalty = 0.0
        self.optimizer = tf.keras.optimizers.Adam(self._policy_lr)
        self.max_trj = max_trj
        self.pol_iters = 80
        self.value_loss_const = 1.
        self.step_runner = tf.Variable(0, dtype=tf.int64, name='step_runner')
        self.model_save_dir = 'vg_model'
        
    def _forward_trajectory(self, max_step=100):
        if self.stack < 1:
            TRJ = GymRunner(self.env, self.policy, max_step, self.step_runner)
        else:
            TRJ = GymRunnerStack(self.env, self.policy, max_step, self.step_runner, self.stack)
        return TRJ

    def _train(self, epochs=30):

        for ep in range(epochs):

            DATA = self.preprocess_data(max_trj=self.max_trj, validate_trj=True, compute_log_prob=True)

            var_to_train = self.value_network.trainable_variables + self.policy.network.trainable_variables
            assert var_to_train, 'no variable is initialized'
            
            for Epps in range(self.pol_iters):
                with tf.GradientTape() as tape:
                    v_loss_k = self.value_loss(DATA.observations, DATA.returns, training=True, summary=self.summary)
                    p_loss_k, log_prob1 = self.policy_loss(DATA.observations, DATA.actions, DATA.advantages)
                    LOSS = v_loss_k * self.value_loss_const + p_loss_k
                grads = tape.gradient(LOSS, var_to_train)
                assert grads, 'gradient cannot be computed'
                if self.grad_clip_norm:
                    grads, grad_norm = tf.clip_by_global_norm(grads, self.grad_clip_norm)
                else:
                    grad_norm = norm_x(grads)
                self.optimizer.apply_gradients(zip(grads, var_to_train))
            
            if self.summary: # this is the lasted summary not average

                p_params_norm = tf.norm(self.policy.get_variable(flatten=True))
                v_params_norm = tf.norm(flatten_shape(self.value_network.trainable_variables))
                tf.summary.scalar('grad_norm', grad_norm, self.train_step_counter)
                tf.summary.scalar('p_params_norm', p_params_norm, self.train_step_counter)
                tf.summary.scalar('v_params_norm', v_params_norm, self.train_step_counter)
                tf.summary.scalar('Loss', LOSS, self.train_step_counter)
                tf.summary.scalar('v_loss', v_loss_k, self.train_step_counter)
                tf.summary.scalar('p_Loss', p_loss_k, self.train_step_counter)

                tf.summary.scalar('advantage', tf.reduce_mean(DATA.advantages), self.train_step_counter)
                log_prob_ratio = tf.reduce_mean(tf.exp(DATA.log_prob0 - log_prob1))
                tf.summary.scalar('log_prob_ratio', log_prob_ratio, self.train_step_counter)

            self.train_step_counter.assign_add(1)
            self.policy.save_model(self.model_save_dir)
            print('-----------------------EPOCH--------------------------')


    def policy_loss(self, S_batch, A_batch, ADV_b):

        logit1 = self.policy.forward_network(S_batch)
         # expand from (batch, action_space) to include branches dim, [batch, branches, action_space]
         # for supporting input in distribution.assign_logit
        if len(self.env.action_spec.DISCRETE_SPACE) == 1:
            logit1 = tf.expand_dims(logit1, 1) 
        self.policy.distribution.assign_logits(logit1)
        log_prob1 = self.policy.distribution.log_prob(A_batch) 

        assert log_prob1.shape == ADV_b.shape
        p_loss = - tf.reduce_mean(log_prob1 * ADV_b)
        p_loss = tf.debugging.check_numerics(p_loss, 'p_loss')
        
        if self.entropy_rate > 0.:
            ent = tf.reduce_mean(self.policy.distribution.entropy())
            tf.summary.scalar('entropy', ent, self.train_step_counter)
        else:
            ent = 0.

        if self._policy_reg > 0.:
            p_var_reg = (v for v in self.policy.network.trainable_weights if 'kernel' in v.name)
            p_reg_loss = tf.nn.scale_regularization_loss([tf.nn.l2_loss(v) for v in p_var_reg]) * self._policy_reg
            p_reg_loss = tf.debugging.check_numerics(p_reg_loss, 'p_reg_loss')
            tf.summary.scalar('p_reg_loss', p_reg_loss, self.train_step_counter)
            
        else:
            p_reg_loss = 0.   

        total_p_loss = p_loss + ent * self.entropy_rate + p_reg_loss * self._policy_reg

        if self.kl_penalty > 0.:
            raise NotImplementedError('not current support kl regularization for policy loss')

        tf.summary.scalar('total_p_loss', total_p_loss, self.train_step_counter)

        return total_p_loss, log_prob1
