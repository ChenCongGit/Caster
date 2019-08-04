# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import fully_connected

from Caster.model.model import BasicModel
from Caster.utils import shape_utils


class CTCModel(BasicModel.Model):
    """
    aster的基于CTC解码器网络模型
    """
    def __init__(self,
                 feature_extractor=None,
                 label_map=None,
                 fc_hyperparams=None,
                 is_training=True,
                 summarize_activation=True):
        super(CTCModel,self).__init__(
            feature_extractor,is_training,summarize_activation
        )
        self._label_map = label_map
        self._fc_hyperparams = fc_hyperparams
        self._groundtruth_dict = {}

    def predict(self, input_tensor, scope=None):
        """
        Args:
            input_tensor: a float tensor with shape [batch_size, image_height, image_width, 3]
        Returns:
            predictions_dict: a diction of predicted tensors
        """
        with tf.variable_scope(scope, 'CTCModel', [input_tensor]):
            # CRNN特征提取
            with tf.variable_scope('Feature_extractor', [input_tensor]):
                preprocessed_inputs = self._feature_extractor.preprocess(input_tensor)
                feature_maps = self._feature_extractor.extract_feature(preprocessed_inputs) # [B, 1, w, C]

                if len(feature_maps) != 1:
                    raise ValueError('CTCModel only accepts single feature sequence')
                feature_sequence = tf.squeeze(feature_maps[0], axis=1) # [B, W, C]

            # 全连接层
            with tf.variable_scope('Predictor', [feature_sequence]),arg_scope(self._fc_hyperparams):
                logits = fully_connected(feature_sequence, self._label_map.num_classes + 1, activation_fn=None) # [B, W, num_classes+1]
        return {'logits': logits}

    def provide_groundtruth(self, groundtruth_text_list, scope=None):
        """
        提供数据集标签，转化为Sparsetensor类型
        """
        with tf.variable_scope(scope, 'CTCProvideGroundtruth', groundtruth_text_list):
            batch_groundtruth_text = tf.stack(groundtruth_text_list, axis=0)
            groundtruth_text_labels_sp, text_lengths = self._label_map.text_to_labels(
                 batch_groundtruth_text,
                 return_dense=False,
                 return_lengths=True
            )
            self._groundtruth_dict['gt_labels_sp'] = groundtruth_text_labels_sp
            self._groundtruth_dict['gt_labels_length'] = text_lengths

    def loss(self, predictions_dict, scope=None):
        """
        标签与预测结果计算CTC损失
        """
        with tf.variable_scope(scope, 'CTCLoss', list(predictions_dict.values())):
            logits = predictions_dict['logits'] # [B, W, num_classes+1]
            batch_size, max_time, _ = shape_utils.combined_static_and_dynamic_shape(logits)
            # 计算CTC损失
            losses = tf.nn.ctc_loss(
                tf.cast(self._groundtruth_dict['gt_labels_sp'], tf.int32),
                predictions_dict['logits'],
                tf.fill([batch_size], max_time),
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=True,
                time_major=False
            ) # [B]

            # 在batch维度上计算平均损失
            loss = tf.reduce_mean(losses, axis=0, keep_dims=False)
        return {'RecognitionLoss': loss}

    def postprocess(self, predictions_dict, scope=None):
        """
        预测结果的CTC解码
        """
        with tf.variable_scope(scope, 'CTCPostprocess', list(predictions_dict.values())):
            logits = predictions_dict['logits'] # [B, W, num_classes+1]
            batch_size, max_time, _ = shape_utils.combined_static_and_dynamic_shape(logits)
            logits_time_major = tf.transpose(logits, [1,0,2]) # [W, B, num_classes+1]
            
            """
            tf.nn.ctc_greedy_decoder
            Args:
                inputs: 3-D float Tensor sized [max_time, batch_size, num_classes]. The logits.
                sequence_length: 1-D int32 vector containing sequence lengths, having size [batch_size].
                merge_repeated: Boolean. Default: True.
            Returns:
                A tuple (decoded, neg_sum_logits) where * decoded: A single-element list. 
                decoded[0] is an SparseTensor containing the decoded outputs s.t.: 
                decoded.indices: Indices matrix (total_decoded_outputs, 2). 
                The rows store: [batch, time]. decoded.values: Values vector, 
                size (total_decoded_outputs). The vector stores the decoded classes. 
                decoded.dense_shape: Shape vector, size (2). The shape values are: [batch_size, 
                max_decoded_length] * neg_sum_logits: A float matrix (batch_size x 1) containing, 
                for the sequence found, the negative of the sum of the greatest logit at 
                each timeframe.
            """
            sparse_labels, _ = tf.nn.ctc_greedy_decoder(
                logits_time_major,
                tf.fill([batch_size], max_time), # sequence_length: 1-D int32 vector containing sequence lengths, having size [batch_size]
                merge_repeated=True
            )

            predictions_text = tf.sparse_tensor_to_dense(sparse_labels[0], default_value=-1)
            text = self._label_map.labels_to_text(predictions_text)
            recognitions_dict = {'text': text}
        return recognitions_dict

    


