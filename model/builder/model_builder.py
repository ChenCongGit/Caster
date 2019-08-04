# -*- coding: utf-8 -*-

"""
根据config所提供的参数，初始化用于模型结构相应的类对象
"""
import functools

import tensorflow as tf
from Caster.protos import model_pb2
from Caster.protos import hyperparams_pb2
from Caster.protos import label_map_pb2
from Caster.protos import loss_pb2

from Caster.model.model import STN
from Caster.model.model import CRNN
from Caster.model.model import ConvNet
from Caster.model.model import Brnn
from Caster.model.model import CTCModel
from Caster.model.model import AttentionPredict
from Caster.model.model import MultiPredictModel

from Caster.model.builder import hyperparams_builder
from Caster.model.builder import label_map_builder
from Caster.model.builder import loss_builder


def build(config, is_training):
    if not isinstance(config, model_pb2.Model):
        raise ValueError('config not of type model_pb2.Model')
    model_oneof = config.WhichOneof('model_oneof')
    if model_oneof == 'multi_predictors_recognition_model':
        model_object = _build_multi_predictor_recognition_model(
            config.multi_predictors_recognition_model, is_training
        )
        return model_object
    elif model_oneof == 'ctc_recognition_model':
        model_object = _build_ctc_recognition_model(
            config.ctc_recognition_model, is_training
        )
    else:
        raise ValueError('Unknown model_oneof: {}'.format(model_oneof))


def _build_multi_predictor_recognition_model(config, is_training):
    """
    构建整体网络模型，输入配置信息，返回model对象
    """
    if not isinstance(config, model_pb2.MultiPredictorsRecognitionModel):
        raise ValueError('config not of type model_pb2.MultiPredictorsRecognitionModel')

    spatial_transformer_object = None
    if config.HasField('spatial_transformer'):
        spatial_transformer_object = stn_build(
            config.spatial_transformer,
            is_training
        )
    
    feature_extractor_object = feat_extract_build(
        config.feature_extractor,
        is_training
    )

    predictors_dict = {
        predictor_config.name: Predictor_build(predictor_config, is_training)
        for predictor_config in config.predictor
    }

    regression_loss = (
        None if not config.keypoint_supervision else
        loss_builder.build([config.regression_loss])
    )

    return MultiPredictModel.MultiPredictorsRecognitionModel(
            spatial_transformer=spatial_transformer_object,
            feature_extractor=feature_extractor_object,
            predictors_dict=predictors_dict,
            regression_loss=regression_loss,
            keypoint_supervision=config.keypoint_supervision,
            is_training=is_training,
            summarize_activation=config.summarize_activation)


def _build_ctc_recognition_model(config, is_training):
    """
    创建CTC网络模型
    """
    if not isinstance(config, model_pb2.CtcRecognitionModel):
        raise ValueError('config not of type model_pb2.CtcRecognitionModel')

    feature_extractor_object = feat_extract_build(config.feature_extractor, is_training)

    if config.HasField('fc_hyperparams'):
        fc_hyperparams_object = hyperparams_builder.build(config.fc_hyperparams, is_training)

    if config.HasField('label_map'):
        label_map_object = label_map_builder.build(config.label_map, is_training)

    return CTCModel.CTCModel(
            feature_extractor=feature_extractor_object,
            fc_hyperparams=fc_hyperparams_object,
            label_map=label_map_object,
            is_training=is_training,
            summarize_activation=config.summarize_activation)


def stn_build(config, is_training):
    """
    初始化stn矫正网络，输入配置信息，返回SpatialTransformer对象
    """
    if not isinstance(config, model_pb2.SpatialTransformer):
        raise ValueError('config not of type SpatialTransformer')

    return STN.SpatialTransformer(
            localization_image_size=(config.localization_h, config.localization_w),
            output_image_size=(config.output_h, config.output_w),
            num_control_points=config.num_control_points,
            init_bias_pattern=config.init_bias_pattern,
            activation=config.activation,
            margins=(config.margin_x, config.margin_y),
            summarize_activations=config.summarize_activations)


def feat_extract_build(config, is_training):
    """
    构建CRNN特征提取网络，输入配置信息，返回FeatureExtractor对象
    """
    if not isinstance(config,model_pb2.FeatureExtractor):
        raise ValueError('config not of type FeatureExtractor')

    convnet_object = convnet_build(config.convnet, is_training)
    
    brnn_object_list = []
    for brnn_config in config.bidirectional_rnn:
        brnn_object_list.append(bidirectional_rnn_build(brnn_config, is_training))

    return CRNN.ExtractFeature(
            convnet=convnet_object,
            brnn_object_list=brnn_object_list,
            summarize_activations=config.summarize_activations,
            is_training=is_training)


def convnet_build(config, is_training):
    """
    构建CRNN中的卷积层网络，输入配置信息，返回CrnnNet或ResNet对象
    """
    if not isinstance(config,model_pb2.Convnet):
        raise ValueError('config not of type Convnet')
    
    convnet_oneof = config.WhichOneof('convnet_oneof')
    if convnet_oneof == 'resnet':
        conv_hyperparams = hyperparams_builder.build(config.resnet.conv_hyperparams, is_training)
        convnet_object = ConvNet.ResNet(
                conv_hyperparams,
                is_training,
                config.resnet.summarize_activations)
    elif convnet_oneof == 'crnn_net':
        conv_hyperparams = hyperparams_builder.build(config.crnn_net.conv_hyperparams, is_training)
        convnet_object = ConvNet.CrnnNet(
                conv_hyperparams,
                is_training,
                config.crnn_net.summarize_activations)
    else:
        raise ValueError('Unknown convnet_oneof: {}'.format(convnet_oneof))
    return convnet_object


def bidirectional_rnn_build(config, is_training):
    """
    构建CRNN中的双向RNN网络，输入配置信息，返回Brnn对象
    """
    if not isinstance(config, model_pb2.BidirectionalRnn):
        raise ValueError('config not of type BidirectionalRnn')

    fw_cell_object = rnn_cell_build(config.rnn_cell)
    bw_cell_object = rnn_cell_build(config.rnn_cell)
    rnn_regularizer_object = hyperparams_builder._build_regularizer(config.rnn_regularizer)
    fc_hyperparams_object = None
    if config.num_output_units > 0:
        if config.fc_hyperparams.op != hyperparams_pb2.Hyperparams.FC:
            raise ValueError('op type must be FC')
        fc_hyperparams_object = hyperparams_builder.build(config.fc_hyperparams, is_training)
    
    if config.static:
        Brnn_object = Brnn.StaticBidirectionalRnn(
            fw_cell_object, bw_cell_object,
            rnn_regularizer=rnn_regularizer_object,
            num_output_units=config.num_output_units,
            fc_hyperparams=fc_hyperparams_object,
            summarize_activations=config.summarize_activations
        )
    else:
        Brnn_object = Brnn.DynamicBidirectionalRnn(
            fw_cell_object, bw_cell_object,
            rnn_regularizer=rnn_regularizer_object,
            num_output_units=config.num_output_units,
            fc_hyperparams=fc_hyperparams_object,
            summarize_activations=config.summarize_activations
        )
    return Brnn_object


def rnn_cell_build(config):
    """
    创建LSTM或GRU对象
    """
    if not isinstance(config, model_pb2.RnnCell):
        raise ValueError('config not of type RnnCell')

    rnn_cell_oneof = config.WhichOneof('rnn_cell_oneof')
    if rnn_cell_oneof == 'lstm_cell':
        lstm_cell_config = config.lstm_cell
        weights_initializer_object = hyperparams_builder._build_initializer(lstm_cell_config.initializer)
        lstm_cell_object = tf.contrib.rnn.LSTMCell(
            lstm_cell_config.num_units,
            use_peepholes=lstm_cell_config.use_peepholes,
            forget_bias=lstm_cell_config.forget_bias,
            initializer=weights_initializer_object)
        return lstm_cell_object
    
    elif rnn_cell_oneof == 'gru_cell':
        gru_cell_config = config.gru_cell_config
        weights_initializer_object = hyperparams_builder._build_initializer(gru_cell_config.initializer)
        gru_cell_object = tf.contrib.rnn.GRUCell(
            gru_cell_config.num_units,
            kernel_initializer=weights_initializer_object
        )
        return gru_cell_object

    else:
        raise ValueError('Unknown rnn_cell_oneof: {}'.format(rnn_cell_oneof))


def Predictor_build(config, is_training):
    """
    创建AttentionPredictor类对象
    """
    if not isinstance(config, model_pb2.Predictor):
        raise ValueError('config not of type Predictor')

    predictor_oneof = config.WhichOneof('predictor_oneof')

    if predictor_oneof == 'attention_predictor':
        config = config.attention_predictor
        rnn_cell_object = rnn_cell_build(config.rnn_cell)
        rnn_regularizer_object = hyperparams_builder._build_regularizer(config.rnn_regularizer)
        label_map_object = label_map_builder.build(config.label_map)
        loss_object = loss_builder.build(config.loss)

        return AttentionPredict.AttentionPredictor(
                rnn_cell=rnn_cell_object,
                rnn_regularizer=rnn_regularizer_object,
                num_attention_units=config.num_attention_units,
                max_num_steps=config.max_num_steps,
                multi_attention=config.multi_attention,
                beam_width=config.beam_width,
                reverse=config.reverse,
                label_map=label_map_object,
                loss=loss_object,
                sync=config.sync,
                is_training=is_training)
    else:
        raise ValueError('Unknown predictor_oneof: {}'.format(predictor_oneof))


        