import tensorflow as tf 
from tensorflow import keras

def create_simple_conv(input_dim, output_dim, actv='relu', output_actv=None, name='conv'):
    	
    ins = keras.layers.Input(shape=input_dim)
    x = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(3, 3), activation=actv, use_bias=True)(ins)
    x = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation=actv, use_bias=True)(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation=actv, use_bias=True)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation=actv)(x)
    out = keras.layers.Dense(output_dim, activation=output_actv)(x)

    return keras.Model(ins, out)

    return keras.Model(ins, out)
		
def conv_rs(x, n_hidden=3, filters=128, kernel=(3, 3), stride=(2, 2), act='sigmoid', b=True, flatten=False):
    for i in range(n_hidden):
        x = tf.keras.layers.Conv2D(filters, kernel, stride, activation=act, use_bias=b)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    if flatten:
        x = tf.keras.layers.Flatten()(x)
    return x

def dense_rs(x, n_hidden=3, size=128, act='sigmoid', b=True):
    for i in range(n_hidden):
        x = tf.keras.layers.Dense(size, activation=act, use_bias=b)(x)
    return x

def create_dense_inp(ins, outs, hidden=128, n_hidden=3, 
                     act='sigmoid', out_act='sigmoid', b=True, concat=-1):
    ip0 = [tf.keras.layers.Input(shape=(ik, )) for ik in ins]
    ds0 = [tf.keras.layers.Dense(hidden, activation=act, use_bias=b)(xk) for xk in ip0]
    c_ = tf.keras.layers.Concatenate(-1)(ds0)
    ds1 = dense_rs(c_, n_hidden, size=hidden, act=act)
    op0 = [tf.keras.layers.Dense(od, activation=act, use_bias=b)(ds1) for od in outs]
    if concat:
        c_ = tf.keras.layers.Concatenate(concat)(op0)
        return tf.keras.Model(ip0, c_)
    return tf.keras.Model(ip0, op0)