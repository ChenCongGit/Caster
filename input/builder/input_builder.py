# -*- coding: utf-8 -*-

import os
import logging

import tensorflow as tf

from Caster.protos import input_reader_pb2

from Caster.input.tfrecord_tools import tfrecord_ic15
from Caster.input.tfrecord_tools import tfrecord_synth90k
from Caster.input.tfrecord_tools import tfrecord_decoder

from Caster.input.preprocess import data_preprocess
from Caster.input.builder import preprocess_builder

slim = tf.contrib.slim

def build(config,path):
    """
    生成或读取tfrecord文件，并解码，组合为数据队列，得到数据集的张量字典
    Caster/datasets --ocr --train
                          --test
                    --ic13 --train
                           --test
                    --ic15 --train
                           --test
                    --....
    以ocr为例，数据集的训练集保存路径为‘Caster/datasets/ocr/train/’
    """
    if not isinstance(config, input_reader_pb2.InputReader):
        raise ValueError('config not of type input_reader_pb2.InputReader')
    
    dataset_name = config.dataset_name
    if config.is_training:
        train_or_test = "train"
    else:
        train_or_test = "test"

    dataset_dir = os.path.join(path, dataset_name + '/' + train_or_test + '/')
    tfrecord_dir = os.path.join(path, dataset_name + '_' + train_or_test + '.tfrecord')
    
    if not os.path.exists(tfrecord_dir):
        logging.info("The tfrecord is not exist or the path is wrong, so new create a tfrecord.")
        tfrecord.create_tfrecord(dataset_dir, tfrecord_dir, train_or_test)
    
    tfrecord_decoder_config = config.tfrecord_decoder
    input_batch_config = config.input_batch
    
    tensor_dict = tfrecord_decoder.decode(tfrecord_dir, tfrecord_decoder_config)
    tensor_dict, data_batch = create_input_batch(tensor_dict, input_batch_config)
    batch_queue = slim.prefetch_queue.prefetch_queue(data_batch, capacity = 256)

    return batch_queue


def create_input_batch(tensor_dict, config):
    """
    输入数据预处理并组合成batch形式
    """
    tensor_dict['image'] = tf.to_float(tensor_dict['image'])
    preprocess_options = [preprocess_builder.build(preprocess_config) for preprocess_config in config.preprocess_option]
    tensor_dict = data_preprocess.preprocess(tensor_dict, preprocess_options)
    data_batch = tf.train.batch(
        tensor_dict,
        batch_size=config.batch_size,
        num_threads=config.num_threads,
        capacity=config.capacity,
        shapes=None,
        allow_smaller_final_batch=False
    )
    return tensor_dict, data_batch
