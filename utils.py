from base_wrap import *
from distributions import *
from trajectory import *
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

