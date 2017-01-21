from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time

def dna_transformation(prev_image,
                       dna_input,
                       relu_shift = 1e-12,
                       dna_kern_size = 5):
  """Apply dynamic neural advection to previous image.

  Args:
    prev_image: previous image to be transformed.
    dna_input: hidden lyaer to be used for computing DNA transformation.
  Returns:
    List of images transformed by the predicted CDNA kernels.
  """
  # Construct translated images.
  prev_image_pad = tf.pad(prev_image, [[0, 0], [2, 2], [2, 2], [0, 0]])
  image_height = int(prev_image.get_shape()[1])
  image_width = int(prev_image.get_shape()[2])

  inputs = []
  for xkern in range(dna_kern_size):
    for ykern in range(dna_kern_size):
      inputs.append(
          tf.expand_dims(
              tf.slice(prev_image_pad, [0, xkern, ykern, 0],
                       [-1, image_height, image_width, -1]), [3]))
  inputs = tf.concat_v2(inputs, 3)

  # Normalize channels to 1.
  kernel = tf.nn.relu(dna_input - relu_shift) + relu_shift
  kernel = tf.expand_dims(
      kernel / tf.reduce_sum(
          kernel, [3], keep_dims=True), [4])
  return tf.reduce_sum(kernel * inputs, [3], keep_dims=False)

def test(image_shape, dna_input_shape, device = '/cpu:0'):
    tf.reset_default_graph()
    scope = 'dna'
    with tf.variable_scope(scope):
        with tf.device(device):
            image_vals = np.random.rand(*image_shape).astype(np.float32)
            dna_vals = np.random.rand(*dna_input_shape).astype(np.float32)

            image = tf.constant(image_vals,
                                shape = image_shape,
                                dtype = tf.float32)

            dna_in = tf.constant(dna_vals,
                                 shape = dna_input_shape,
                                 dtype = tf.float32)

            ref_dna_op = dna_transformation(image, dna_in)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for r in xrange(0 ,5):
                    start_time = time.time() * 1000
                    ref_dna_out = sess.run(ref_dna_op)
                    end_time = time.time() * 1000
                    print(end_time - start_time)

            print(ref_dna_out.shape)

test((64, 64, 64, 3), (64, 64, 64, 25), '/cpu:0')
