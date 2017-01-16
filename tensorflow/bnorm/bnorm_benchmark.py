from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
from tensorflow.python.client import timeline

halide_module = tf.load_op_library('bnorm_op.so')

def run_sub_graph(out_dict, num_runs):
    best_time = float('inf')
    with tf.Session(config = tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for t in xrange(0, num_runs):
            start_time = time.time() * 1000
            sess.run(out_dict,
                     options=run_options,
                     run_metadata=run_metadata)
            end_time = time.time() * 1000
            best_time = min(best_time, end_time - start_time)
        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
    return best_time

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
            #input = tf.placeholder(tf.float32, shape = [batch_size, in_h, in_w, in_ch])
            input = tf.random_uniform(shape = [batch_size, in_h, in_w, in_ch],
                                      dtype = tf.float32,
                                      minval = 0.,
                                      maxval = 1.0)
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
            if halide_op:
                out = halide_module.bnorm(input, gamma, beta, mean, variance)
            else:
                out = tf.nn.batch_normalization(input, mean, variance, beta, gamma, epsilon)

    # Run the sub graph multiple times
    out_dict = { "out" : out.op }
    best_time = run_sub_graph(out_dict, num_runs)
    return best_time

print("%f ms" % (benchmark_batch_norm(128, 64, 64, 64, 1e-05, True, 100, '/cpu:0')))
print("%f ms" % (benchmark_batch_norm(128, 64, 64, 64, 1e-05, False, 100, '/cpu:0')))
