# -*- coding: utf-8 -*-

import os
import io
import random
import re
import glob
import logging
from tqdm import tqdm

from PIL import Image
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', './Caster/datasets/synth90k/train', 'Root dir of dataset')
flags.DEFINE_string('output_path', './Caster/datasets/synth90k_train.tfrecord', 'Output tfrecord file to')
FLAGS = flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord(dataset_dir, tfrecord_dir, train_or_test):
    writer = tf.python_io.TFRecordWriter(tfrecord_dir)

    groundtruth_file = os.path.join(dataset_dir, 'annotation.txt')
    with open(groundtruth_file, 'r') as fr:
        groundtruth_lines = fr.readlines()
    
    num_images = len(groundtruth_lines)
    indices = list(range(num_images))
    random.shuffle(indices)

    num_images = 0
    for index in tqdm(indices):
        image_partial_path = groundtruth_lines[index].split(' ')[0][2:]
        image_path = os.path.join(dataset_dir, image_partial_path)

        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                image_jpeg = f.read()
            
            # extract groundtruth
            groundtruth_text = image_partial_path.split('_')[1]


            example = tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': _bytes_feature(image_jpeg),
                'image/format': _bytes_feature('jpeg'.encode('utf-8')),
                'image/filename': _bytes_feature(image_partial_path.encode('utf-8')),
                'image/channels': _int64_feature(3),
                'image/colorspace': _bytes_feature('rgb'.encode('utf-8')),
                'image/groundtruth': _bytes_feature(groundtruth_text.encode('utf-8'))
            }))
            writer.write(example.SerializeToString())
            num_images += 1
    writer.close()
    logging.info('{} images tfrecor created'.format(num_images))


if __name__ == '__main__':
    create_tfrecord(FLAGS.data_dir, FLAGS.output_path, "train")
