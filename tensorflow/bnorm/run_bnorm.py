from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time

halide_module = tf.load_op_library('bnorm_op.so')

def run_sub_graph(out_dict, in_dict, num_runs):
    res = {}
    best_time = float('inf')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t in xrange(0, num_runs):
            start_time = time.time() * 1000
            res = sess.run(out_dict, feed_dict = in_dict)
            end_time = time.time() * 1000
            best_time = min(best_time, end_time - start_time)
    return (res, best_time)

def benchmark_batch_norm(batch_size,
                         in_h,
                         in_w,
                         in_ch,
                         epsilon,
                         halide_op = False,
                         num_runs = 5,
                         device = '/gpu:0'):
    scope = 'bn'
    if halide_op:
        scope = 'bn_halide'
    with tf.variable_scope(scope):
        with tf.device(device):
            input = tf.placeholder(tf.float32, shape = [batch_size, in_h, in_w, in_ch])
            input_shape = input.get_shape()
            params_shape = input_shape[-1:]
            beta = tf.get_variable("beta", params_shape,
                                   initializer = tf.zeros_initializer(dtype = tf.float32))
            gamma = tf.get_variable("gamma", params_shape,
                                    initializer = tf.zeros_initializer(dtype = tf.float32))
            mean = tf.get_variable("mean", params_shape,
                                   initializer = tf.zeros_initializer(dtype = tf.float32))
            variance = tf.get_variable("variance", params_shape,
                                        initializer = tf.zeros_initializer(dtype = tf.float32))
            out = tf.nn.batch_normalization(input, mean, variance, beta, gamma, epsilon)
            if halide_op:
                out = halide_module.bnorm(input, gamma, beta, mean, variance)

    # Create a random input
    rand_in = np.random.rand(batch_size, in_h, in_w, in_ch)
    # Run the sub graph multiple times
    in_dict = { input : rand_in }
    out_dict = { "out" : out }
    _, best_time = run_sub_graph(out_dict, in_dict, num_runs)
    return best_time

print("%f ms" % (benchmark_batch_norm(16, 64, 64, 64, 1e-05, True, 5, '/cpu:0')))
print("%f ms" % (benchmark_batch_norm(16, 64, 64, 64, 1e-05, False, 5, '/cpu:0')))
