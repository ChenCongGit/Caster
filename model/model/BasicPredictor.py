from abc import abstractmethod
from abc import ABCMeta

import tensorflow as tf


class BasicPredictor(object):
    """
    Predictor基类
    """
    __metaclass__ = ABCMeta

    def __init__(self, is_training=True):
        self._is_training = is_training
        self._grounttruth_dict = {}

    @abstractmethod
    def predict(self, feature_maps, scope):
        pass

    @abstractmethod
    def loss(self, predictions_dict, scope):
        pass
    
    @abstractmethod
    def provide_groundtruth(self, groundtruth_list, scope):
        pass

    @abstractmethod
    def postprocess(self, predictions_dict, scope):
        pass
