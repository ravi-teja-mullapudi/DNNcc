from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time

import tensorflow.contrib.slim as slim

def cdna_transformation(prev_image,
                        cdna_input,
                        num_masks,
                        color_channels,
                        relu_shift = 1e-12,
                        dna_kern_size = 5):
  """Apply convolutional dynamic neural advection to previous image.

  Args:
    prev_image: previous image to be transformed.
    cdna_input: hidden lyaer to be used for computing CDNA kernels.
    num_masks: the number of masks and hence the number of CDNA transformations.
    color_channels: the number of color channels in the images.
  Returns:
    List of images transformed by the predicted CDNA kernels.
  """
  batch_size = int(cdna_input.get_shape()[0])

  # Predict kernels using linear function of last hidden layer.
  cdna_kerns = slim.layers.fully_connected(
      cdna_input,
      dna_kern_size * dna_kern_size * num_masks,
      scope='cdna_params',
      activation_fn=None)

  # Reshape and normalize.
  cdna_kerns = tf.reshape(
      cdna_kerns, [batch_size, dna_kern_size, dna_kern_size, 1, num_masks])
  cdna_kerns = tf.nn.relu(cdna_kerns - relu_shift) + relu_shift
  norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
  cdna_kerns /= norm_factor

  cdna_kerns = tf.tile(cdna_kerns, [1, 1, 1, color_channels, 1])
  cdna_kerns = tf.split(cdna_kerns, batch_size, 0)
  prev_images = tf.split(prev_image, batch_size, 0)

  # Transform image.
  transformed = []
  for kernel, preimg in zip(cdna_kerns, prev_images):
    kernel = tf.squeeze(kernel)
    if len(kernel.get_shape()) == 3:
      kernel = tf.expand_dims(kernel, -1)
    transformed.append(
        tf.nn.depthwise_conv2d(preimg, kernel, [1, 1, 1, 1], 'SAME'))
  transformed = tf.concat_v2(transformed, 0)
  transformed = tf.split(transformed, num_masks, 3)
  return transformed

def test(image_shape, cdna_input_shape, num_masks, color_channels, device = '/cpu:0'):
    tf.reset_default_graph()
    scope = 'cdna'
    with tf.variable_scope(scope):
        with tf.device(device):
            image_vals = np.random.rand(*image_shape).astype(np.float32)
            cdna_vals = np.random.rand(*cdna_input_shape).astype(np.float32)

            image = tf.constant(image_vals,
                                shape = image_shape,
                                dtype = tf.float32)

            cdna_in = tf.constant(cdna_vals,
                                  shape = cdna_input_shape,
                                  dtype = tf.float32)

            ref_cdna_op = cdna_transformation(image,
                                              cdna_in,
                                              num_masks,
                                              color_channels)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for r in xrange(0 ,5):
                    start_time = time.time() * 1000
                    ref_cdna_out = sess.run(ref_cdna_op)
                    end_time = time.time() * 1000
                    print(end_time - start_time)

            for out in ref_cdna_out:
                print(out.shape)

test((32, 64, 64, 3), (32, 8192), 10, 3, '/cpu:0')
