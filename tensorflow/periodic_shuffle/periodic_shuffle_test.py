from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

halide_module = tf.load_op_library('periodic_shuffle_op.so')

def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat_v2([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat_v2([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat_v2([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X

def test(input_shape, r, device = '/cpu:0'):
    tf.reset_default_graph()
    assert len(input_shape) == 4
    # channels must be equal to r^2
    assert input_shape[3] == (r * r)
    scope = 'phase_shift'
    with tf.variable_scope(scope):
        with tf.device(device):
            input_vals = np.random.rand(*input_shape).astype(np.float32)

            X = tf.constant(input_vals,
                            shape = input_shape,
                            dtype = tf.float32)

            ref_ps_op = PS(X, r)
            halide_ps_op = halide_module.periodic_shuffle(X, r)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                ref_out = sess.run(ref_ps_op)
                halide_out = sess.run(halide_ps_op)
            print(halide_out.shape)
            print(ref_out.shape)

test((16, 64, 64, 64), 8)
test((8, 64, 64, 4), 2)
test((1, 54, 54, 16), 4)
