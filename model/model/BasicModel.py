# -*- coding: utf-8 -*-

from abc import abstractmethod

import tensorflow as tf


class Model(object):
    """
    Model基类，提供基本的模型常规处理方法框架，即输入预处理，前向传播预测，计算损失，后处理，提供标签
    变量:
        _feature_extractor: Feature_extractor类对象
    """
    def __init__(self, 
                 feature_extractor=None, 
                 is_training=True, 
                 summarize_activation=True):
        self._feature_extractor = feature_extractor
        self._is_training = is_training
        self._predictors = {}
        self._groundtruth_dict = {}
        self._summarize_activation = summarize_activation

    def preprocess(self,input_tensor):
        """
        输入张量预处理，将图像张量的值压缩到-1到1的区间，-1表示原像素0，1表示原像素255
        """
        with tf.variable_scope('ModelPreprocess', [input_tensor]):
            if input_tensor.dtype is not tf.float32:
                raise ValueError('Preprocess need a tf.float32 tensor')
            preprocessed_inputs = (2.0/255.0)*input_tensor-1.0
            if self._summarize_activation:
                tf.summary.image('preprocessed_inputs',preprocessed_inputs)
        return preprocessed_inputs

    @abstractmethod
    def predict(self,preprocessed_inputs,scope=None):
        """
        前向传播，返回输入张量对应的预测结果，具体网络需要具体的定义
        """
        pass

    @abstractmethod
    def loss(self,predicted_tensor,scope=None):
        """
        计算损失，返回Loss损失张量，具体网络需要具体的定义
        """
        pass

    @abstractmethod
    def postprocess(self,predicted_tensor,scope=None):
        """
        前向传播预测结果后处理，返回处理后的预测结果张量
        """
        pass

    @abstractmethod
    def provide_groundtruth(self,groundtruth_lists,scope=None):
        """
        提供计算损失所需要的标签
        """
        pass