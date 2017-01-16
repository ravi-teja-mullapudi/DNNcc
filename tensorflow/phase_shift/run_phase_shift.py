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

def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, color=False):
    if color:
        Xc = tf.split(3, 3, X)
        X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
    else:
        X = _phase_shift(X, r)
    return X
