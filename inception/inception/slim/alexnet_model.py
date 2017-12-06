# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.

Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)

@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from inception.slim import ops
from inception.slim import scopes
from inception.slim import losses

#slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def alexnet_v2_parameters(weight_decay=0.0005):
  with scopes.arg_scope([ops.conv2d, ops.fc],
                      activation=tf.nn.relu):
    with scopes.arg_scope([ops.conv2d], padding='SAME'):
      with scopes.arg_scope([ops.max_pool], padding='VALID') as arg_sc:
        return arg_sc


def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=False,
               scope='alexnet_v2'):
  """AlexNet version 2.

  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224. To use in fully
        convolutional mode, set spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  print("alexnet_model.alexnet_v2.inputs shape:", inputs.get_shape())
  with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.max_pool]):
      net = ops.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
      net = ops.max_pool(net, [3, 3], 2, scope='pool1')
      net = ops.conv2d(net, 192, [5, 5], scope='conv2')
      net = ops.max_pool(net, [3, 3], 2, scope='pool2')
      net = ops.conv2d(net, 384, [3, 3], scope='conv3')
      net = ops.conv2d(net, 384, [3, 3], scope='conv4')
      net = ops.conv2d(net, 256, [3, 3], scope='conv5')
      net = ops.max_pool(net, [3, 3], 2, scope='pool5')

      # Use conv2d instead of fully_connected layers.
      with scopes.arg_scope([ops.conv2d]):
        net = ops.conv2d(net, 4096, [5, 5], padding='VALID',
                          scope='fc6')
        net = ops.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = ops.conv2d(net, 4096, [1, 1], scope='fc7')
        net = ops.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = ops.conv2d(net, num_classes, [1, 1], scope='fc8')

      # Convert end_points_collection into a end_point dict.
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
      
      tf.logging.info('The shape is {}!!'.format(tf.shape(net)))
      return net
alexnet_v2.default_image_size = 224
