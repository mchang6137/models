# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build the Inception v3 network on ImageNet data set.

The Inception v3 architecture is described in http://arxiv.org/abs/1512.00567

Summary of available functions:
 inference: Compute inference on the model inputs to make a prediction
 loss: Compute the loss of the prediction with respect to the labels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

import model.resnet_v2 as resnet
# import model.resnet_utils as resnet_utils
import tensorflow.contrib.slim.nets as nets

# resnet = nets.resnet_v2
resnet_utils = nets.resnet_utils

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

# If a model is trained using multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Inception v3 model architecture.

  See here for reference: http://arxiv.org/abs/1512.00567

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  with slim.arg_scope(resnet_utils.resnet_arg_scope(batch_norm_decay=BATCHNORM_MOVING_AVERAGE_DECAY,
                                                    batch_norm_epsilon=0.001)):
    outputs,end_points = resnet.resnet_v2_101(images, num_classes=num_classes)
    outputs = tf.squeeze(outputs) # Uncomment this if you have dimension 1's to remove
  return outputs, end_points

def loss(logits, labels, batch_size=None):
  """Adds all losses for the model.

  Note the final loss is not returned. Instead, the list of losses are collected
  by slim.losses. The losses are accumulated in tower_loss() and summed to
  calculate the total loss.

  Args:
    logits: List of logits from inference(). Each entry is a 2-D float Tensor.
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    batch_size: integer
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Reshape the labels into a dense Tensor of
  # shape [FLAGS.batch_size, num_classes].
  sparse_labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
  concated = tf.concat(axis=1, values=[indices, sparse_labels])
  num_classes = logits.get_shape()[-1].value
  dense_labels = tf.sparse_to_dense(concated,
                                    [batch_size, num_classes],
                                    1.0, 0.0)

  # Cross entropy loss for the main softmax prediction.
  slim.losses.softmax_cross_entropy(logits,
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weights=1.0)

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _activation_summaries(endpoints):
  with tf.name_scope('summaries'):
    for act in endpoints.values():
      _activation_summary(act)
