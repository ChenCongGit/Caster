# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, fully_connected
from tensorflow.contrib.framework import arg_scope

from Caster.utils import shape_utils


class CrnnNet(object):
    """
    使用原始的CRNN网络卷积部分作为CNN
    """
    def __init__(self,
                 conv_hyperparams=None,
                 summarize_activations=False,
                 is_training=True):
        self._conv_hyperparams = conv_hyperparams
        self._summarize_activations = summarize_activations
        self._is_training = is_training

    def preprocess(self, resized_inputs, scope=None):
        with tf.variable_scope(scope, 'CnnNetPreprocess', [resized_inputs]):
            preprocessed_inputs = (2.0 / 255.0) * resized_inputs - 1.0
            if self._summarize_activations:
                tf.summary.image('CnnNetPreprocessed_inputs',preprocessed_inputs,max_outputs=1)
        return preprocessed_inputs
    
    def forward(self, preprocessed_inputs, scope=None):
        with tf.variable_scope(scope, 'CnnNetForward', [preprocessed_inputs]):
            shape_assert = self._shape_check(preprocessed_inputs)
            if shape_assert is None:
                shape_assert = tf.no_op()
            with tf.control_dependencies([shape_assert]), arg_scope(self._conv_hyperparams):
                with arg_scope([conv2d], kernel_size=3, padding='SAME', stride=1),\
                     arg_scope([max_pool2d], kernel_size=2, padding='VALID', stride=2):
                    conv1 = conv2d(preprocessed_inputs, 64, scope='conv1') # [B, 32, W, 64]
                    pool1 = max_pool2d(conv1, scope='pool1') # [B, 16, W/2, 64]
                    conv2 = conv2d(pool1, 128, scope='conv2') # [B, 16, W/2, 128]
                    pool2 = max_pool2d(conv2, scope='pool2') # [B, 8, W/4, 128]
                    conv3 = conv2d(pool2, 256, scope='conv3') # [B, 8, W/4, 256]
                    pool3 = max_pool2d(conv3, scope='pool3') # [B, 4, W/8, 256]
                    conv4 = conv2d(pool3, 512, scope='conv4') # [B, 4, W/8, 512]
                    pool4 = max_pool2d(conv4, stride=[2, 1], scope='pool4') # [B, 2, W/8, 512]
                    conv5 = conv2d(pool4, 512, scope='conv5') # [B, 2, W/8, 512]
                    conv6 = conv2d(conv5, 512, scope='conv6') # [B, 2, W/8, 512]
                    pool6 = max_pool2d(conv6, stride=[2, 1], scope='pool6') # [B, 1, W/16, 512]
                    conv7 = conv2d(pool6, 512, kernel_size=[2,1], padding='VALID', scope='conv7') # [B, 1, W/16, 512]
                    feature_maps_dict = {
                        'conv1': conv1, 'conv2': conv2, 'conv3': conv3, 'conv4': conv4,
                        'conv5': conv5, 'conv6': conv6, 'conv7': conv7}
                if self._summarize_activations:
                    for key, value in feature_maps_dict.items():
                        tf.summary.histogram('Activation/'+key, value)
        return [feature_maps_dict['conv7']]

    def _shape_check(self, preprocessed_inputs):
        preprocessed_inputs.get_shape().assert_has_rank(4)
        shape_assert = tf.Assert(
            tf.greater_equal(tf.shape(preprocessed_inputs)[1], 32),
            ['image height must be at least 32.'])
        return shape_assert


class ResNet(object):
    """
    ResNet网络用作特征提取的卷积部分CNN
    """
    def __init__(self,
                 conv_hyperparams=None,
                 summarize_activations=False,
                 is_training=True):
        resnet_spec = [('Block_1', 3, 32, [2, 2]),
                       ('Block_2', 4, 64, [2, 2]),
                       ('Block_3', 6, 128, [2, 1]),
                       ('Block_4', 6, 256, [2, 1]),
                       ('Block_5', 3, 512, [2, 1])]
        
        self._conv_hyperparams = conv_hyperparams
        self._summarize_activations = summarize_activations
        self._is_training = is_training
        self._resnet_spec = resnet_spec

    def preprocess(self, resized_inputs, scope=None):
        with tf.variable_scope(scope, 'ResNetPreprocess', [resized_inputs]):
            preprocessed_inputs = (2.0 / 255.0) * resized_inputs - 1.0
            if self._summarize_activations:
                tf.summary.image('CnnNetPreprocessed_inputs',preprocessed_inputs,max_outputs=1)
        return preprocessed_inputs

    def forward(self, preprocessed_inputs, scope=None):
        with tf.variable_scope(scope, 'ResnetForward', [preprocessed_inputs]):
            shape_assert = self._shape_check(preprocessed_inputs)
            if shape_assert is None:
                shape_assert = tf.no_op()
            with tf.control_dependencies([shape_assert]), arg_scope(self._conv_hyperparams):
                with arg_scope([conv2d], kernel_size=3, padding='SAME', stride=1),\
                     arg_scope([max_pool2d], kernel_size=2, padding='VALID', stride=2):
                    conv_0 = conv2d(preprocessed_inputs, 32, scope='Conv0') # [B, 32, W, 32]
                    block_outputs_list = [conv_0]
                    for (scope, num_units, num_outputs, first_subsample) in self._resnet_spec:
                        block_outputs = self._residual_block(block_outputs_list[-1], num_units, num_outputs, first_subsample, scope)
                        block_outputs_list.append(block_outputs)

                    block_outputs_dict = {}
                    for index, block_outputs in enumerate(block_outputs_list):
                        block_outputs_dict['Block_{}'.format(index)] = block_outputs

                if self._summarize_activations:
                    for key, value in block_outputs_dict.items():
                        tf.summary.histogram('Activation/'+key, value)
        return [block_outputs_list[-1]]

    def _output_endpoints(self, feature_maps_dict):
        return [feature_maps_dict['Block_5']]

    def _residual_block(self, inputs, num_units, num_outputs, first_subsample, scope):
        with tf.variable_scope(scope, 'ResidualBlock', [inputs]):
            unit_outputs = self._residual_unit(inputs, num_outputs, first_subsample, scope + 'unit_0')
            for i in range(1, num_units):
                unit_outputs = self._residual_unit(unit_outputs, num_outputs, None, scope + 'unit_{}'.format(i))
        return unit_outputs

    def _residual_unit(self, inputs, num_outputs, first_subsample, scope):
        with tf.variable_scope(scope, 'ResidualUnit', [inputs]):
            with arg_scope([conv2d], kernel_size=3, padding='SAME', stride=1),\
                 arg_scope([max_pool2d], kernel_size=2, padding='VALID', stride=2):
                if not first_subsample:
                    conv1 = conv2d(inputs, num_outputs, kernel_size=1, scope='Conv1')
                    shortcut = tf.identity(inputs, name='ShortCut')
                else:
                    conv1 = conv2d(inputs, num_outputs, stride=first_subsample, scope='Conv1')
                    shortcut = conv2d(inputs, num_outputs, stride=first_subsample, scope='ShortCut')
                conv2 = conv2d(conv1, num_outputs, activation_fn=None, scope='Conv2')
                outputs = tf.nn.relu(tf.add(conv2, shortcut))
        return outputs

    def _shape_check(self, preprocessed_inputs):
        preprocessed_inputs.get_shape().assert_has_rank(4)
        shape_assert = tf.Assert(
            tf.greater_equal(tf.shape(preprocessed_inputs)[1], 32),
            ['image height must be at least 32.'])
        return shape_assert
