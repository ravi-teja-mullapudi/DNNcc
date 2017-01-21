from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                            (1. - targets) * tf.log(1. - preds + eps)))

def test(pred_shape, device = '/gpu:0'):
    tf.reset_default_graph()
    scope = 'bce'
    with tf.variable_scope(scope):
        with tf.device(device):
            pred_vals = np.random.rand(*pred_shape).astype(np.float32)
            target_vals = np.random.rand(*pred_shape).astype(np.float32)

            preds = tf.constant(pred_vals,
                                shape = pred_shape,
                                dtype = tf.float32)

            targets = tf.constant(target_vals,
                                 shape = pred_shape,
                                 dtype = tf.float32)

            ref_bce_op = binary_cross_entropy(preds, targets)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for r in xrange(0 ,5):
                    start_time = time.time() * 1000
                    ref_bce_out = sess.run(ref_bce_op)
                    end_time = time.time() * 1000
                    print(end_time - start_time)

            print(ref_bce_out.shape)

test((64, 128, 128, 3))
