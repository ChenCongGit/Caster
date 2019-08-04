# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import fully_connected


"""
动态RNN和静态RNN区别

调用static_rnn实际上是生成了rnn按时间序列展开之后的图。打开tensorboard你会看到sequence_length
个rnn_cell stack在一起，只不过这些cell是share weight的。因此，sequence_length就和图的拓扑结构
绑定在了一起，因此也就限制了每个batch的sequence_length必须是一致。

调用dynamic_rnn不会将rnn展开，而是利用tf.while_loop这个api，通过Enter, Switch, Merge, 
LoopCondition, NextIteration等这些control flow的节点，生成一个可以执行循环的图（这个图应
该还是静态图，因为图的拓扑结构在执行时是不会变化的）。在tensorboard上，你只会看到一个rnn_cell, 
外面被一群control flow节点包围着。对于dynamic_rnn来说，sequence_length仅仅代表着循环的次数，
而和图本身的拓扑没有关系，所以每个batch可以有不同sequence_length。
"""

class DynamicBidirectionalRnn(object):
    """
    动态双向RNN，即循环状态下的RNN图，输入三维序列，输出最后时刻的状态序列，这种RNN图的特点是完全共享中间状态
    """
    def __init__(self,
                 fw_cell,
                 bw_cell,
                 rnn_regularizer,
                 num_output_units,
                 fc_hyperparams,
                 summarize_activations):
        self._fw_cell = fw_cell
        self._bw_cell = bw_cell
        self._rnn_regularizer = rnn_regularizer
        self._num_output_units = num_output_units
        self._fc_hyperparams = fc_hyperparams
        self._summarize_activations = summarize_activations

    def forward(self, inputs, scope=None):
        with tf.variable_scope(scope, 'DynamicBrnn', [inputs]) as scope:
            # 双向LSTM
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                self._fw_cell, self._bw_cell,inputs, time_major=False, dtype=tf.float32) # inputs:[B, T, C]
            rnn_outputs = tf.concat([output_fw, output_bw], 2) # [B, T, C]

            # 正则化
            filter_weights = lambda vars : [x for x in vars if x.op.name.endswith('kernel')]
            tf.contrib.layers.apply_regularization(
                self._rnn_regularizer, filter_weights(self._fw_cell.trainable_weights))
            tf.contrib.layers.apply_regularization(
                self._rnn_regularizer, filter_weights(self._bw_cell.trainable_weights))

            # 全连接层
            if self._num_output_units > 0:
                with arg_scope(self._fc_hyperparams):
                    rnn_outputs = fully_connected(rnn_outputs, self._num_output_units, activation_fn=tf.nn.relu)

            if self._summarize_activations:
                max_time = rnn_outputs.get_shape().as_list()[1]
                for t in range(max_time):
                    output = rnn_outputs[:,t,:]
                    tf.summary.histogram('Activation/{}/Step_{}'.format(scope.name, t), output)
            return rnn_outputs



class StaticBidirectionalRnn(object):
    """
    静态双向RNN，即RNN网络的按时间静态展开图，输入的是一个向量的序列列表，输出每个时刻的序列输出状态列表，因此可以获得RNN的中间状态
    网络参数的数量比动态RNN网络大，计算时间更长，因为存储序列中每一个输入时所产生的中间状态信息，所以占用更多的内存
    """
    def __init__(self,
                 fw_cell,
                 bw_cell,
                 rnn_regularizer,
                 num_output_units,
                 fc_hyperparams,
                 summarize_activations):
        self._fw_cell = fw_cell
        self._bw_cell = bw_cell
        self._rnn_regularizer = rnn_regularizer
        self._num_output_units = num_output_units
        self._fc_hyperparams = fc_hyperparams
        self._summarize_activations = summarize_activations

    def forward(self, inputs, scope=None):
        with tf.Variable_scope(scope, 'StaticBrnn', [inputs]) as scope:
            # inputs [B, max_time, depth]
            inputs_list = tf.unstack(inputs, axis=1) # [B, depth]
            outputs_list, _, _ = tf.nn.static_bidirectional_rnn(
                self._fw_cell, self._bw_cell, inputs_list, dtype=tf.float32)
            
            # 正则化
            filter_weights = lambda vars : [x for x in vars if x.op.name.endswith('kernel')]
            tf.contrib.layers.apply_regularization(
                self._rnn_regularizer, filter_weights(self._fw_cell.trainable_weights))
            tf.contrib.layers.apply_regularization(
                self._rnn_regularizer, filter_weights(self._bw_cell.trainable_weights))

            rnn_outputs = tf.stack(outputs_list, axis=1) # [B, max_time, depth]

            # 全连接层
            if self._num_output_units > 0:
                with arg_scope(self._fc_hyperparams):
                    rnn_outputs = fully_connected(rnn_outputs, self._num_output_units, activation_fn=tf.nn.relu)

            if self._summarize_activations:
                for i in range(len(outputs_list)):
                    tf.summary.histogram('Activation/{}/Step_{}'.format(scope.name, i), outputs_list[i])
            return rnn_outputs