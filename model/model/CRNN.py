# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, fully_connected
from tensorflow.contrib.framework import arg_scope

from Caster.utils import shape_utils


class ExtractFeature(object):
    """
    CNN+RNN特征提取
    """
    def __init__(self,
                 convnet=None,
                 brnn_object_list=[],
                 summarize_activations=False,
                 is_training=True):
        self._convnet = convnet
        self._brnn_object_list = brnn_object_list
        self._summarize_activations = summarize_activations

    def preprocess(self, resized_inputs, scope=None):
        with tf.variable_scope(scope, 'CrnnPreprocess', [resized_inputs]):
            preprocessed_inputs = (2.0 / 255.0) * resized_inputs - 1.0
            if self._summarize_activations:
                tf.summary.image('CrnnPreprocessed_inputs',preprocessed_inputs,max_outputs=1)
        return preprocessed_inputs    
    
    def extract_feature(self, preprocessed_inputs, scope=None):
        with tf.variable_scope(scope, 'ExtractFeature', [preprocessed_inputs]):
            # CNN层
            feature_maps = self._convnet.forward(preprocessed_inputs, scope)

            if len(self._brnn_object_list) > 0:
                feature_sequences_list = []
                for i, feature_map in enumerate(feature_maps):
                    # 确保送入BRNN的卷积特征图高度为1
                    shape_assert = tf.Assert(tf.equal(tf.shape(feature_map)[1],1),
                        ['Feature map height must be 1 if bidirectional RNN is going to be applied.'])

                    # 卷积特征图转换为RNN特征序列，这里特征序列长度为卷积特征图宽度
                    batch_size, _, _, depth = shape_utils.combined_static_and_dynamic_shape(feature_map) # [B, 1, w, c]
                    with tf.control_dependencies([shape_assert]):
                        feature_sequence = tf.reshape(feature_map, [batch_size, -1, depth]) # [B, w, c] RNN特征序列形状
                        # 多层BRNN层
                        for j in range(len(self._brnn_object_list)):
                            brnn_object = self._brnn_object_list[j]
                            feature_sequence = brnn_object.forward(feature_sequence, scope='BidirectionalRnn_Branch_{}_{}'.format(i, j))
                        feature_sequences_list.append(feature_sequence)

                feature_maps = [tf.expand_dims(fmap, axis=1) for fmap in feature_sequences_list] # [B, 1, w, c]
            return feature_maps

