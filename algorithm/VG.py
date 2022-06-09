from base_agent import ActorCritic
from runner import GymRunner
from utils import *
import tensorflow as tf

class VanillaPG(ActorCritic):

    def __init__(self, env, policy, value_network=None, gamma=0.90, summary=True, optimizer=None, 
    grad_clip_norm=1., policy_lr=0.001, value_lr=0.01, use_gae=True, lam=0.9):
        super().__init__(env, policy, value_network, gamma, summary, optimizer, grad_clip_norm, policy_lr, value_lr, use_gae, lam)
        self.entropy_rate = 0.1
        self.policy_l2_reg = 0.1
        self.kl_penalty = 0.0
        self.optimizer = tf.keras.optimizers.Adam(self._policy_lr)

    def _forward_trajectory(self, max_step=100):
        TRJ = GymRunner(self.env, self.policy, max_step)
        return TRJ

    def _train(self, epochs=30):

        DATA = self.preprocess_data(max_trj=100, validate_trj=True, compute_log_prob=True)
        value_var = self.value_network.trainable_variables
        policy_var = self.policy.network.trainable_variables
        assert (value_var and policy_var), 'variable must not be empty'
        
        if not self.shared_network:
            # TODO: using train on batch
            for ep in range(epochs):

                with tf.GradientTape() as v_tape:
                    v_loss_k = self.value_loss(DATA.observations, DATA.returns, training=True, summary=self.summary)
                #v_grad = self.apply_optimize(v_loss_k, value_var, v_tape, self.value_opt)
                v_grad = v_tape.gradient(v_loss_k, value_var)
                self.value_opt.apply_gradients(zip(v_grad, self.value_network.trainable_variables))

                with tf.GradientTape() as p_tape:
                    p_loss_k = self.policy_loss(DATA.observations, DATA.actions, DATA.advantages)
                #p_grad = self.apply_optimize(p_loss_k, policy_var, p_tape, self.optimizer)
                p_grad = p_tape.gradient(p_loss_k, policy_var)
                self.optimizer.apply_gradients(zip(p_grad, self.policy.network.trainable_variables))

                assert (v_grad and p_grad), 'could not compute gradient'

                if self.summary:
                    tf.summary.scalar('value_grad_norm', norm_x(v_grad), self.train_step_counter)
                    tf.summary.scalar('policy_grad_norm', norm_x(p_grad), self.train_step_counter)
                    tf.summary.scalar('norm_params', tf.norm(self.policy.get_variable(flatten=True)), self.train_step_counter)
                    
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
            total_p_loss = p_loss + self.entropy_rate * ent
            tf.summary.scalar('entropy', ent, self.train_step_counter)
        else:
            ent = 0.

        if self.policy_l2_reg > 0.:
            p_var_reg = (v for v in self.policy.network.trainable_weights if 'kernel' in v.name)
            p_reg_loss = tf.nn.scale_regularization_loss([tf.nn.l2_loss(v) for v in p_var_reg]) * self.policy_l2_reg
            p_reg_loss = tf.debugging.check_numerics(p_reg_loss, 'p_reg_loss')
            tf.summary.scalar('p_reg_loss', p_reg_loss, self.train_step_counter)
            
        else:
            p_reg_loss = 0.   

        total_p_loss = p_loss + ent * self.entropy_rate + p_reg_loss * self.policy_l2_reg

        if self.kl_penalty > 0.:
            raise NotImplementedError('not current support kl regularization for policy loss')

        tf.summary.scalar('total_p_loss', total_p_loss, self.train_step_counter)

        return total_p_loss
