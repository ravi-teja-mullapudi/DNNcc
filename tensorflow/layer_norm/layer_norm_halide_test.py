from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

halide_module = tf.load_op_library('lnorm_op.so')

def test(input_shape, device = '/cpu:0'):
    tf.reset_default_graph()
    scope = 'layer_norm'
    with tf.variable_scope(scope):
        with tf.device(device):
            input_vals = np.random.rand(*input_shape).astype(np.float32)
            #input_vals = np.ones(input_shape, dtype = np.float32) * 2

            inputs = tf.constant(input_vals,
                                 shape = input_shape,
                                 dtype = tf.float32)

            beta = tf.get_variable("beta",
                                    input_shape[1:],
                                    initializer =
                                    tf.zeros_initializer(dtype = tf.float32),
                                    dtype = tf.float32)
            gamma = tf.get_variable("gamma",
                                     input_shape[1:],
                                     initializer =
                                     tf.ones_initializer(dtype = tf.float32),
                                     dtype = tf.float32)

            ref_ln_op = tf.contrib.layers.layer_norm(inputs,
                                                     center = True,
                                                     scale = True)

            halide_op = halide_module.lnorm(inputs, beta, gamma)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                ref_out = sess.run(ref_ln_op)
                halide_out = sess.run(halide_op)

            print(halide_out.min(), halide_out.max())
            print(ref_out.min(), ref_out.max())

test((128, 16))
test((16, 128))
test((16, 10))
test((16, 64, 64, 10))
test((128, 57, 14, 128))
