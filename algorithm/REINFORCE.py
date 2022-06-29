from base_agent import BaseAgent
from runner import GymRunner_Episodic, GymRunner_EpisodicStack
from utils import *

import tensorflow as tf
from collections import namedtuple
PREPS_RESULT = namedtuple('PREPS_RESULT', ['observations', 'returns', 'actions', 'log_prob0'])

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior() # in discounted sum of rewards

class ReinforcePG(BaseAgent):
    '''REINFORCE policy gradient
    Input : 
        max_steps (int) : maximum env steps for one epochs of training
        max_spisode_steps (int) : maximum running episode steps
    '''
    def __init__(self, env, policy, gamma=0.98, max_steps=10000, optimizer=None, 
            grad_clip_norm=2., policy_lr=0.001):
        super().__init__(env, policy, gamma, grad_clip_norm=grad_clip_norm)

        self._policy_lr = policy_lr
        self._policy_reg = 0.1
        self.entropy_rate = 0.9
        self.kl_penalty = 0.0
        self.optimizer = tf.keras.optimizers.Adam(self._policy_lr)
        self.max_steps = max_steps
        self.max_episode_steps = 2000
        self.state_normalizer = RunningStat()
        self.old_logits = None
        self.model_save_dir = 'reinforce_model'
        
    def preprocess_episode(self, global_step=None):
        
        if self.stack < 1:
            trj = GymRunner_Episodic(self.env, self.policy, self.max_episode_steps, global_step=global_step)
        else:
            trj = GymRunner_EpisodicStack(self.env, self.policy, self.max_episode_steps, global_step, self.stack)
        
        r = discount_rewards(trj.rewards, gamma=self.gamma)
        r = normalize_advantage(r)
        obs = trj.observations

        if self.env.action_spec.is_discrete() and not self.env.action_spec.is_continuous():
            action = tf.cast(trj.actions, dtype=tf.int32) # pure discrete action spaces
        elif self.env.action_spec.is_discrete() & self.env.action_spec.is_continuous():
            raise NotImplementedError('not support mixed action space yet for VG algorithm')
        else:
            action = trj.actions

        self.old_logits = tf.convert_to_tensor(self.policy.old_logits) # old_logits in agent will not be deleted until next running trj
        self.policy.distribution.assign_logits(self.old_logits)
        log_prob0 = self.policy.distribution.log_prob(action)
        #erase old_logits from policy.policy
        self.policy.old_logits = [] 

        return PREPS_RESULT(obs, r, action, log_prob0)

    def _train(self, epochs=30):

        for ep in range(epochs):
            T =  tf.Variable(0, dtype=tf.int64)
            Tk = tf.Variable(0, dtype=tf.int64, name='general_step_counter')
            while T < self.max_steps:

                DATA = self.preprocess_episode(global_step=Tk)

                var_to_train = self.policy.network.trainable_variables
                assert var_to_train, 'no variable is initialized'
    
                with tf.GradientTape() as tape:
                    p_loss_k, log_prob1 = self.policy_loss(DATA.observations, DATA.actions, DATA.returns)

                grads = tape.gradient(p_loss_k, var_to_train)
                assert grads, 'gradient cannot be computed'
                if self.grad_clip_norm:
                    grads, grad_norm = tf.clip_by_global_norm(grads, self.grad_clip_norm)
                else:
                    grad_norm = norm_x(grads)

                self.optimizer.apply_gradients(zip(grads, var_to_train))
                self.train_step_counter.assign_add(1) # assing every trajactory not epoch
                tf.summary.scalar('Loss', p_loss_k, self.train_step_counter)
                tf.summary.scalar('grad_norm', grad_norm, self.train_step_counter)
                tf.summary.scalar('params_norm', norm_x(var_to_train), self.train_step_counter)

            self.policy.save_model(self.model_save_dir)
            print('-----------------------EPOCH--------------------------')

    def policy_loss(self, S_batch, A_batch, R_batch):

        logit1 = self.policy.forward_network(S_batch)
         # expand from (batch, action_space) to include branches dim, [batch, branches, action_space]
         # for supporting input in distribution.assign_logit
        if len(self.env.action_spec.DISCRETE_SPACE) == 1:
            logit1 = tf.expand_dims(logit1, 1) 
        self.policy.distribution.assign_logits(logit1)
        log_prob1 = self.policy.distribution.log_prob(A_batch) 

        assert log_prob1.shape == R_batch.shape
        p_loss = - tf.reduce_mean(log_prob1 * R_batch)
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