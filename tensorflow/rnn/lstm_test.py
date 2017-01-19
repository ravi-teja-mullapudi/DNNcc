from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def test(batch_size, n_steps,
         n_hidden, in_size,
         device = '/cpu:0'):
    with tf.variable_scope('basic_lstm'):
        with tf.device(device):
            input_shape = [n_steps, batch_size, in_size]
            input_vals = np.random.rand(*input_shape).astype(np.float32)
            input = tf.constant(input_vals,
                                shape = input_shape,
                                dtype = tf.float32)

            lstm = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
            state = init_state = lstm.zero_state(batch_size, tf.float32)

            for t in xrange(0, n_steps):
                if t > 0: tf.get_variable_scope().reuse_variables()
                out, state = lstm(input[t, :, :], state)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                ref_out = sess.run([out, state])

test(16, 4, 1000, 10)
