# Copyright 2016 Google Inc. All Rights Reserved.
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
""" VGG-16 expressed in TensorFlow-Slim. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes


def vgg(inputs,
          dropout_keep_prob=0.8,
          num_classes=1000,
          is_training=True,
          restore_logits=True,
          scope=''):
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.name_scope(scope, 'vgg', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.dropout], is_training=is_training):
      print('>>> inputs shape: {}'.format(inputs.get_shape()))
      print('>>>>> inputs dtype: {}'.format(inputs.dtype))

      end_points['conv0'] = ops.conv2d(inputs, 64, [3, 3], scope='conv0')
      print('>>> conv0 shape: {}'.format(end_points['conv0'].get_shape()))
      print('>>>>> conv0 dtype: {}'.format(end_points['conv0'].dtype))

      end_points['conv1'] = ops.conv2d(end_points['conv0'], 64, [3, 3], scope='conv1')
      print('>>> conv1 shape: {}'.format(end_points['conv1'].get_shape()))
      
      end_points['pool1'] = ops.max_pool(end_points['conv1'], [2, 2], scope='pool1')
      print('>>> pool1 shape: {}'.format(end_points['pool1'].get_shape()))

      end_points['conv2'] = ops.conv2d(end_points['pool1'], 128, [3, 3], scope='conv2')
      print('>>> conv2 shape: {}'.format(end_points['conv2'].get_shape()))

      end_points['conv3'] = ops.conv2d(end_points['conv2'], 128, [3, 3], scope='conv3')
      print('>>> conv3 shape: {}'.format(end_points['conv3'].get_shape()))

      end_points['pool2'] = ops.max_pool(end_points['conv3'], [2, 2], scope='pool2')
      print('>>> pool2 shape: {}'.format(end_points['pool2'].get_shape()))

      end_points['conv4'] = ops.conv2d(end_points['pool2'], 256, [3, 3], scope='conv4')
      print('>>> conv4 shape: {}'.format(end_points['conv4'].get_shape()))

      end_points['conv5'] = ops.conv2d(end_points['conv4'], 256, [3, 3], scope='conv5')
      print('>>> conv5 shape: {}'.format(end_points['conv5'].get_shape()))

      end_points['conv6'] = ops.conv2d(end_points['conv5'], 256, [3, 3], scope='conv6')
      print('>>> conv6 shape: {}'.format(end_points['conv6'].get_shape()))

      end_points['pool3'] = ops.max_pool(end_points['conv6'], [2, 2], scope='pool3')
      print('>>> pool3 shape: {}'.format(end_points['pool3'].get_shape()))

      end_points['conv7'] = ops.conv2d(end_points['pool3'], 512, [3, 3], scope='conv7')
      print('>>> conv7 shape: {}'.format(end_points['conv7'].get_shape()))

      end_points['conv8'] = ops.conv2d(end_points['conv7'], 512, [3, 3], scope='conv8')
      print('>>> conv8 shape: {}'.format(end_points['conv8'].get_shape()))

      end_points['conv9'] = ops.conv2d(end_points['conv8'], 512, [3, 3], scope='conv9')
      print('>>> conv9 shape: {}'.format(end_points['conv9'].get_shape()))

      end_points['pool4'] = ops.max_pool(end_points['conv9'], [2, 2], scope='pool4')
      print('>>> pool4 shape: {}'.format(end_points['pool4'].get_shape()))

      end_points['conv10'] = ops.conv2d(end_points['pool4'], 512, [3, 3], scope='conv10')
      print('>>> conv10 shape: {}'.format(end_points['conv10'].get_shape()))

      end_points['conv11'] = ops.conv2d(end_points['conv10'], 512, [3, 3], scope='conv11')
      print('>>> conv11 shape: {}'.format(end_points['conv11'].get_shape()))

      end_points['conv12'] = ops.conv2d(end_points['conv11'], 512, [3, 3], scope='conv12')
      print('>>> conv12 shape: {}'.format(end_points['conv12'].get_shape()))

      end_points['pool5'] = ops.max_pool(end_points['conv12'], [2, 2], scope='pool5')
      print('>>> pool5 shape: {}'.format(end_points['pool5'].get_shape()))

      end_points['flatten5'] = ops.flatten(end_points['pool5'], scope='flatten5')
      print('>>> flatten5 shape: {}'.format(end_points['flatten5'].get_shape()))

      end_points['fc1'] = ops.fc(end_points['flatten5'], 4096, scope='fc1')
      print('>>> fc1 shape: {}'.format(end_points['fc1'].get_shape()))

      end_points['dropout1'] = ops.dropout(end_points['fc1'], dropout_keep_prob, scope='dropout1')
      print('>>> dropout1 shape: {}'.format(end_points['dropout1'].get_shape()))

      end_points['fc2'] = ops.fc(end_points['dropout1'], 4096, scope='fc2')
      print('>>> fc2 shape: {}'.format(end_points['fc2'].get_shape()))

      end_points['dropout2'] = ops.dropout(end_points['fc2'], dropout_keep_prob, scope='dropout2')
      print('>>> dropout2 shape: {}'.format(end_points['dropout2'].get_shape()))

      end_points['logits'] = ops.fc(end_points['dropout2'], num_classes, activation=None,
                                    stddev=0.001, restore=restore_logits, scope='logits')
      print('>>> logits shape: {}'.format(end_points['logits'].get_shape()))
      print('>>>>> logits dtype: {}'.format(end_points['logits'].dtype))
      return end_points['logits'], end_points
