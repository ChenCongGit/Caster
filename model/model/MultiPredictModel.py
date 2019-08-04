# -*- coding: utf-8 -*-

import tensorflow as tf

from Caster.model.model import BasicModel
from Caster.utils import shape_utils


class MultiPredictorsRecognitionModel(BasicModel.Model):
    """
    Aster网络模型
    变量:
        _spatial_transformer: stn对象
        _predictors_dict: 
    """
    def __init__(self,
                 spatial_transformer=None,
                 feature_extractor=None,
                 predictors_dict=None,
                 regression_loss=None,
                 keypoint_supervision=None,
                 is_training=True,
                 summarize_activation=True):
        super(MultiPredictorsRecognitionModel,self).__init__(
            feature_extractor,is_training,summarize_activation
        )
        self._spatial_transformer = spatial_transformer
        self._predictors_dict = predictors_dict
        self._regression_loss = regression_loss
        self._keypoint_supervision = keypoint_supervision
        self._is_training = is_training

        if len(self._predictors_dict) == 0:
            raise ValueError('predictors_list is empty!')

        self._groundtruth_dict = {}

    def predict(self,input_tensor,scope=None):
        """
        网络的前向传播，获得文本和控制点预测结果，加入
        """
        with tf.variable_scope(scope, 'ModelPredict', [input_tensor]):
            predictions_dict = {}
            # 带stn矫正的图像预处理
            if self._spatial_transformer:
                with tf.variable_scope('Stn',[input_tensor]):
                    stn_inputs = self.preprocess(input_tensor)
                    stn_output_dict = self._spatial_transformer.batch_transform(stn_inputs)
                    preprocessed_inputs = stn_output_dict['rectified_images']
                    control_points = stn_output_dict['control_points']
                    predictions_dict.update({
                        'rectified_images': preprocessed_inputs,
                        'control_points': control_points
                    })
            else:
                preprocessed_inputs = self.preprocess(input_tensor)
            
            # feature_extractor卷积特征提取
            feature_maps = self._feature_extractor.extract_feature(preprocessed_inputs)

            # predictor预测 
            """
            Attention，我们使用多个输出预测器，他们具有不同的名字，保存在self._predictors_dict中
            self._predictors_dict 键为预测器名字，值为预测器AttentionPredictor对象
            """
            for name, predictor in self._predictors_dict.items():
                predictor_outputs = predictor.predict(feature_maps, scope='{}/Predictor'.format(name))
                predictions_dict.update({
                    '{}/{}'.format(name, k): v for k, v in predictor_outputs.items()
                })
            return predictions_dict

    def loss(self, predictions_dict, scope=None):
        """
        分别计算不同预测器识别损失和控制点回归损失
        """
        with tf.variable_scope(scope, 'ModelLoss', [predictions_dict]):
            losses_dict = {}
            # 多预测器Attention识别损失，计算损失需要与各自的预测结果相对应，即相同的name
            for name, predictor in self._predictors_dict.items():
                predictor_loss = predictor.loss(
                    {k.split('/')[1]:v for k, v in predictions_dict.items() if k.startswith('{}/'.format(name))},
                    scope='{}/Loss'.format(name))
                losses_dict[name] = predictor_loss

            # STN矫正网络回归损失
            if self._keypoint_supervision and self._spatial_transformer:
                predict_control_points = predictions_dict['control_points']
                num_control_points = tf.shape(predict_control_points)[1]
                masked_control_points = tf.boolean_mask(predict_control_points, self._groundtruth_dict['control_points_mask'])
                flat_masked_control_points = tf.reshape([
                    -1, 2 * self._spatial_transformer._num_control_points]) # [bacth_size, 2K]
                losses_dict['KeypointSupervisionLoss'] = self._regression_loss(
                    flat_masked_control_points, self._groundtruth_dict['control_points']
                )
            return losses_dict

    def postprocess(self, predictions_dict, scope=None):
        """
        返回文本识别结果
        """
        with tf.variable_scope(scope, 'Postprocess', list(predictions_dict.values())):
            recognition_text_list = []
            recognition_scores_list = []
            for name, predictor in self._predictors_dict.items():
                predictor_outputs = predictor.postprocess(
                    {k.split('/')[1]:v for k, v in predictions_dict.items() if k.startswith('{}/'.format(name))},
                    scope = '{}/PostProcess'.format(name)
                )
                recognition_text_list.append(predictor_outputs['text'])
                recognition_scores_list.append(predictor_outputs['scores'])
            aggregated_recognition_dict = self._aggregate_recognition_results(
                recognition_text_list, recognition_scores_list
            )
            return aggregated_recognition_dict

    def provide_groundtruth(self, groundtruth, scope=None):
        """
        提供文本识别和控制点标签，加入 self._groundtruth_dict
        """
        with tf.variable_scope(scope, 'ModelGT', [groundtruth]):
            # provide groundtruth_text to all predictors
            batch_gt_text = groundtruth
            for name, predictor in self._predictors_dict.items():
                self._groundtruth_dict[name] = predictor.provide_groundtruth(batch_gt_text, scope='{}/ProvideGroundtruth'.format(name))

            # # provide groundtruth keypoints
            # if self._keypoint_supervision and self._spatial_transformer:
            #     batch_gt_keypoints_lengths = tf.stack([
            #         tf.shape(keypoints)[0] for keypoints in groundtruth_lists['groundtruth_keypoints']], axis=0) # [B]
            #     max_gt_keypoints_length = tf.reduce_max(batch_gt_keypoints_lengths) # lengths scalar
                
            #     # 选择等于最多控制点数的gt_keypoints作为最后的groundtruth，少于这个点数的图像不使用
            #     has_gt_keypoints_lengths = tf.equal(max_gt_keypoints_length, batch_gt_keypoints_lengths)

            #     batch_gt_keypoints = tf.stack([
            #         tf.pad(keypoints, [[0, max_keypoints_length - tf.shape(keypoints)[0]]])
            #         for keypoints in groundtruth_lists['groundtruth_keypoints']
            #     ])
            #     batch_gt_keypoints = tf.boolean_mask(batch_gt_keypoints, has_gt_keypoints_lengths)
            #     groundtruth_control_points = ops.divide_curve(
            #         batch_gt_keypoints,
            #         num_key_points=self._spatial_transformer._num_control_points)
                # self._groundtruth_dict['control_points_mask'] = has_gt_keypoints_lengths
                # self._groundtruth_dict['control_points'] = groundtruth_control_points


    def _aggregate_recognition_results(self, text_list, scores_list, scope=None):
        """
        在多个预测器的结果中挑选score最高的作为最终的识别结果
        text_list: # [batch_text1, ...batch_textK] 长度为预测器的类别
        scores_list: # [batch_score1, ...batch_scoreK]
        """
        with tf.variable_scope(scope, 'SelectRecognitionResult', (text_list + scores_list)):
            batch_texts = tf.stack(text_list, axis=1) # [B, K]
            batch_scores = tf.stack(scores_list, axis=1) # [B, K]
            argmax_scores = tf.argmax(batch_scores, axis=1) # [B]
            batch_size = shape_utils.combined_static_and_dynamic_shape(batch_texts)[0]
            indices = tf.stack([tf.range(batch_size, dtype=tf.int64), argmax_scores], axis=1)
            aggregated_text = tf.gather_nd(batch_texts, indices) # [B]
            aggregated_score = tf.gather_nd(batch_scores, indices) # [B]
            recognition_results = {'text': aggregated_text, 'score': aggregated_score}
            return recognition_results
