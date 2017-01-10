import tensorflow as tf
zero_out_module = tf.load_op_library('zero_op.so')
with tf.Session(''):
    zero_out_module.zero([1, 2, 3, 4]).eval()
