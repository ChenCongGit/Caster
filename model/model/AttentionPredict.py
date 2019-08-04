import functools

import tensorflow as tf

from Caster.utils import shape_utils

from Caster.model.model import BasicPredictor
from Caster.model.model import sync_attention_wrapper
from Caster.c_ops import ops


class AttentionPredictor(BasicPredictor.BasicPredictor):
    """
    基于注意力机制的seq2seq模型
    """
    def __init__(self,
                 rnn_cell=None,
                 rnn_regularizer=None,
                 num_attention_units=None,
                 max_num_steps=None,
                 multi_attention=False,
                 beam_width=None,
                 reverse=False,
                 label_map=None,
                 loss=None,
                 sync=False,
                 is_training=True):
        super(AttentionPredictor, self).__init__(is_training)
        self._rnn_cell = rnn_cell
        self._rnn_regularizer = rnn_regularizer
        self._num_attention_units = num_attention_units
        self._max_num_steps = max_num_steps
        self._multi_attention = multi_attention
        self._beam_width = beam_width
        self._reverse = reverse
        self._label_map = label_map
        self._sync = sync
        self._loss = loss

        self._groundtruth_dict = {}

        if not self._is_training and not beam_width > 0:
            raise ValueError('Beam width must be > 0 during inference')

    @property
    def start_label(self):
        return 0
    
    @property
    def end_label(self):
        return 1

    @property
    def num_classes(self):
        return self._label_map.num_classes + 2

    def predict(self, feature_maps, scope=None):
        if not isinstance(feature_maps, (list, tuple)):
            raise ValueError('feature_maps must be list or tuple')

        with tf.variable_scope(scope, 'Predictors', feature_maps):
            batch_size, _, _, _ = shape_utils.combined_static_and_dynamic_shape(feature_maps[0])
            decoder_cell = self._build_decoder_cell(feature_maps)
            decoder = self._build_decoder(decoder_cell, batch_size)
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=self._max_num_steps    
            ) # final_outputs: [batch_size, T, beam_width], 

            # apply regularizer
            filter_weights = lambda vars : [x for x in vars if x.op.name.endswith('kernel')]
            tf.contrib.layers.apply_regularization(
                self._rnn_regularizer,
                filter_weights(decoder_cell.trainable_weights)
            )

            if self._is_training:
                assert isinstance(final_outputs, tf.contrib.seq2seq.BasicDecoderOutput)
                outputs_dict = {
                    'labels': final_outputs.sample_id,
                    'logits': final_outputs.rnn_output
                }
            else:
                assert isinstance(final_outputs, tf.contrib.seq2seq.FinalBeamSearchDecoderOutput)
                prediction_labels = final_outputs.predicted_ids[:,:,0] # [B,T,1] 解码器输出预测结果序列
                prediction_lengths = final_sequence_lengths[:,0] # [B,1] 解码器最终真实序列长度
                prediction_scores = tf.gather_nd(
                    final_outputs.beam_search_decoder_output.scores[:,:,0],
                    tf.stack([tf.range(batch_size),prediction_lengths-1], axis=1)
                )
                outputs_dict = {
                    'labels': prediction_labels,
                    'lengths': prediction_lengths,
                    'scores': prediction_scores
                }
            return outputs_dict

    def loss(self, predictions_dict, scope=None):
        assert 'logits' in predictions_dict
        with tf.variable_scope(scope, 'Loss', list(predictions_dict.values())):
            loss_tensor = self._loss(
                predictions_dict['logits'],
                self._groundtruth_dict['decoder_targets'],
                self._groundtruth_dict['decoder_lengths']
            )
            return loss_tensor

    def postprocess(self, predictions_dict, scope=None):
        assert 'scores' in predictions_dict
        with tf.variable_scope(scope, 'Postprocess', list(predictions_dict.values())):
            text = self._label_map.label_to_text(predictions_dict['labels'])
            if self._reverse:
                text = ops.string_reverse(text)
            scores = predictions_dict['scores']
        return {'text': text, 'scores': scores}

    def provide_groundtruth(self, batch_gt_text, scope=None):
        """
        seq2seq模型训练时，需要将真实标签作为解码器输入，这里是decoder_inputs，而decoder_targets用于
        计算每一时刻损失。当self._sync为true时，解码器输入start时开始进行解码，并且直接输出预测结果，而
        self._sync为false时，解码器在输出一个start标志后才进行解码，直到遇到end标志。
        """
        with tf.name_scope(scope, 'ProvideGroundtruth', [batch_gt_text]):
            batch_size = shape_utils.combined_static_and_dynamic_shape(batch_gt_text)[0]

            if self._reverse:
                batch_gt_text = ops.string_reverse(batch_gt_text)
            batch_gt_labels, batch_gt_lengths = self._label_map.text_to_label(
                batch_gt_text, pad_value=self.end_label, return_lengths=True) # [batch_size, batch_gt_lengths]
            start_labels = tf.fill([batch_size, 1], tf.constant(self.start_label, dtype=tf.int64)) # [batch_size, 1]
            end_labels = tf.fill([batch_size, 1], tf.constant(self.end_label, dtype=tf.int64)) # [batch_size, 1]
            
            if not self._sync:
                decoder_inputs = tf.concat([start_labels, start_labels, batch_gt_labels], axis=1)
                decoder_targets = tf.concat([start_labels, batch_gt_labels, end_labels])
                decoder_lengths = batch_gt_lengths + 2
            else:
                decoder_inputs = tf.concat([start_labels, batch_gt_labels], axis=1)
                decoder_targets = tf.concat([batch_gt_labels, end_labels], axis=1)
                decoder_lengths = batch_gt_lengths + 1

            # set maximum lengths
            decoder_inputs = decoder_inputs[:,:self._max_num_steps]
            decoder_targets = decoder_targets[:,:self._max_num_steps]
            decoder_lengths = tf.minimum(decoder_lengths, self._max_num_steps)

            self._groundtruth_dict['decoder_inputs'] = decoder_inputs
            self._groundtruth_dict['decoder_targets'] = decoder_targets
            self._groundtruth_dict['decoder_lengths'] = decoder_lengths

            return self._groundtruth_dict

    def _build_decoder_cell(self, feature_maps):
        """
        将rnn_cell和attention封装到一起
        """
        attention_mechanism = self._build_attention_mechanism(feature_maps)
        wrapper_class = tf.contrib.seq2seq.AttentionWrapper if not self._sync else sync_attention_wrapper.SyncAttentionWrapper
        attention_rnn_cell = wrapper_class(
            self._rnn_cell,
            attention_mechanism,
            output_attention=False
        )
        return attention_rnn_cell

    def _build_decoder(self, decoder_cell, batch_size):
        outout_layer = tf.layers.Dense(
            self.num_classes,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.variance_scaling_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

        if self._is_training:
            train_helper = tf.contrib.seq2seq.TrainingHelper(
                tf.one_hot(self._groundtruth_dict['decoder_inputs'], depth=self.num_classes),
                sequence_length=self._groundtruth_dict['decoder_lengths'],
                time_major=False
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=train_helper,
                initial_state=decoder_cell.zero_state(batch_size, dtype=tf.float32),
                output_layer=outout_layer
            )
        else:
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=functools.partial(tf.one_hot, depth=self.num_classes),
                start_tokens=tf.fill([batch_size], self.start_label),
                end_token=self.end_label,
                initial_state=decoder_cell.zero_state(batch_size * self._beam_width, dtype=tf.float32),
                beam_width=self._beam_width,
                output_layer=outout_layer,
                length_penalty_weight=0.0
            )
        return decoder

    def _build_attention_mechanism(self, feature_maps):
        """Build (possibly multiple) attention mechanisms."""
        def _build_single_attention_mechanism(memory):
            if not self._is_training:
                memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=self._beam_width)
            return tf.contrib.seq2seq.BahdanauAttention(
                self._num_attention_units,
                memory,
                memory_sequence_length=None
            )
        
        feature_sequences = [tf.squeeze(map, axis=1) for map in feature_maps]
        if self._multi_attention:
            attention_mechanism = []
            for i, feature_sequence in enumerate(feature_sequences):
                memory = feature_sequence
                attention_mechanism.append(_build_single_attention_mechanism(memory))
        else:
            memory = tf.concat(feature_sequences, axis=1)
            attention_mechanism = _build_single_attention_mechanism(memory)
        return attention_mechanism