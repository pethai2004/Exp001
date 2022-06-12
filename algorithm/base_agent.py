import tensorflow as tf
from utils import clip_gradient, compute_adv_from_Buff, RunningStat, mean_and_std, normalizing, normalize_advantage
from utils import flatten_shape
from trajectory import FullDatasetPG, FullBuffer

class BaseAgent:

    def __init__(self, env, policy, value_network=None, gamma=0.90, summary=True, optimizer=None,
     grad_clip_norm=1.):
        
        self.env = env
        self.policy = policy
        self.value_network = value_network
        self.gamma = gamma
        self.summary = summary
        self.optimizer = optimizer
        self.grad_clip_norm = grad_clip_norm
        self.train_step_counter = tf.compat.v1.train.get_or_create_global_step()
        self.clip_obs = 10.
        self.normalize_obs = True
        self.log_dir = 'run_train'
        self.grad_clip = None
        
    def train(self, epoch=3):
        self._train(epoch)

    def _train(self):
        raise NotImplementedError

    def apply_optimize(self, loss, var_to_train, grad_tape, optimizer):
    
        tf.debugging.check_numerics(loss, "Loss is inf or nan")
        assert list(var_to_train), "No variables in the network"

        gradients = grad_tape.gradient(loss, var_to_train)
        grads_and_vars = list(zip(gradients, var_to_train))

        if self.grad_clip_norm:
            clip_gradient(grads_and_vars, self.grad_clip_norm)

        optimizer.apply_gradients(grads_and_vars)
        return gradients

class ActorCritic(BaseAgent):

    def __init__(self, env, policy, value_network=None, discount=0.90, summary=True, optimizer='Adam', grad_clip_norm=1., policy_lr=0.001, 
        value_lr=0.01, use_gae=True, lam=0.9):
        super().__init__(env, policy, value_network, discount, summary, optimizer, grad_clip_norm)
        
        self._policy_lr = policy_lr
        self._value_lr = value_lr
        self._policy_reg = 0.
        self._value_reg = 0.
        self.use_gae = use_gae
        self.lam = lam
        self.state_normalizer = RunningStat()
        self.shared_network = None
        self.value_opt = tf.keras.optimizers.SGD(self._value_lr, momentum=0.1, nesterov=False)
        self.old_logits = None

        if self.shared_network is None and self.value_network is None:
            raise ValueError('if not using shared network, the value_network must be provided')

    def _forward_trajectory(self, max_trj):
        '''Gather trajectory information
        Returns:
            Trajectory (FullBuffer) describe result from running env.
        '''
        raise NotImplementedError
    
    def validate_trajectory(self, trj:FullBuffer, num_steps):
        assert isinstance(trj, FullBuffer)
        # use assert in env the obs_spec must be a list of observation_spec
        # if len(self.env.observation_spec) == 1: # multiple obs not support 
        #     assert trj.observations.shape == (num_steps, *self.env.observation_spec.OBSERVATION_SPACE)
        # else:
        #     raise NotImplementedError('not currently support multiple observation space')
        assert trj.rewards.shape == (num_steps, 1)
        assert trj.actions.shape == (num_steps, len(self.env.action_spec))

    def preprocess_data(self, max_trj:int=100, validate_trj=True, compute_log_prob=True):

        TRJ = self._forward_trajectory(max_trj)
        if validate_trj:
            self.validate_trajectory(TRJ, max_trj)
        obs = TRJ.observations

        if self.clip_obs and self.normalize_obs > 0.:
            self.state_normalizer.add(flatten_shape(obs))
            obs_mean, obs_std = self.state_normalizer.get_mean(), self.state_normalizer.get_std()
            if tf.math.is_nan(obs_std):
                obs_std = tf.math.reduce_std(obs)
                
            obs = tf.clip_by_value(obs, - self.clip_obs, self.clip_obs)
            obs = (obs - obs_mean) / obs_std + 0.00000001
            obs = tf.debugging.check_numerics(obs, 'obs')

        if self.env.action_spec.is_discrete() and not self.env.action_spec.is_continuous():
            action = tf.cast(TRJ.actions, dtype=tf.int32) # pure discrete action spaces
        elif self.env.action_spec.is_discrete() & self.env.action_spec.is_continuous():
            raise NotImplementedError('not support mixed action space yet for VG algorithm')
        else:
            action = TRJ.actions

        v_predict = self.value_network(TRJ.observations)

        if compute_log_prob:
            self.old_logits = tf.convert_to_tensor(self.policy.old_logits) # old_logits in agent will not be deleted until next running trj
            self.policy.distribution.assign_logits(self.old_logits)
            log_prob0 = self.policy.distribution.log_prob(action)
            #erase old_logits from policy.policy
            self.policy.old_logits = [] 
        else:
            log_prob0 = tf.zeros_like(v_predict)
        
        if self.use_gae:
            advs, returns = compute_adv_from_Buff(TRJ, v_pred=v_predict, gamma=self.gamma, lam=self.lam)
        else:
            raise NotImplementedError('not currently support empirical computing of advantage function')

        advs = normalize_advantage(advs) #TODO: delete normalize_advantage and use normalizing instead since it is the same
        returns = normalize_advantage(advs)
        
        return FullDatasetPG(TRJ.step_type, obs, returns, action, log_prob0, advs, v_predict)
        
    def value_loss(self, S_buf, R_buf, training=True, summary=True):

        predict_v = self.value_network(S_buf, training=training)
        v_loss = tf.reduce_mean(predict_v - R_buf) ** 2
        
        v_loss = tf.debugging.check_numerics(v_loss, 'v_loss')
        
        if self._value_reg > 0.:
            v_var_reg = (v for v in self.value_network.trainable_weights if 'kernel' in v.name)
            v_reg_loss = tf.nn.scale_regularization_loss([tf.nn.l2_loss(v) for v in v_var_reg]) * self._value_reg
            v_reg_loss = tf.debugging.check_numerics(v_reg_loss, 'value_l2_loss')
            total_v_loss = v_loss + v_reg_loss
            if summary:
                tf.summary.scalar('value_l2_loss', v_reg_loss, self.train_step_counter)
                tf.summary.scalar('total_value_loss', total_v_loss, self.train_step_counter)	
        else:
            v_reg_loss = 0.
            total_v_loss = v_loss

        if summary:
            tf.summary.scalar('value_pred_avg', tf.reduce_mean(predict_v), self.train_step_counter)
            tf.summary.scalar('value_actual_avg', tf.reduce_mean(R_buf), self.train_step_counter)
            tf.summary.scalar('value_loss', v_loss, self.train_step_counter)

        return total_v_loss
    
    