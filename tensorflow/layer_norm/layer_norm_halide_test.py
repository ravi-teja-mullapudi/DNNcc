from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def test(input_shape, device = '/cpu:0'):
    tf.reset_default_graph()
    scope = 'layer_norm'
    with tf.variable_scope(scope):
        with tf.device(device):
            input_vals = np.random.rand(*input_shape).astype(np.float32)
            beta_vals = np.random.rand(*input_shape[-1:]).astype(np.float32)
            gamma_vals = np.random.rand(*input_shape[-1:]).astype(np.float32)

            inputs = tf.constant(input_vals,
                                 shape = input_shape,
                                 dtype = tf.float32)
            beta = tf.get_variable("beta",
                                    initializer = beta_vals,
                                    dtype = tf.float32)
            gamama = tf.get_variable("gamma",
                                      initializer = gamma_vals,
                                      dtype = tf.float32)

            ref_ln_op = tf.contrib.layers.layer_norm(inputs,
                                                     center = True,
                                                     scale = True)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                ref_out = sess.run(ref_ln_op)
            print(np.sum(ref_out))

test((16, 128))
test((16, 10))
test((16, 64, 64, 10))
