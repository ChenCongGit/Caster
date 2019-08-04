# -*- coding: utf-8 -*-

import os
import io
import random
import re
import glob
import logging

from PIL import Image
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', './Caster/datasets/new_ocr_test/', 'Root dir of dataset')
flags.DEFINE_string('output_path', './Caster/datasets/ocr/ocr_test.tfrecord', 'Output tfrecord file to')
FLAGS = flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord(dataset_dir, tfrecord_dir, train_or_test):
    writer = tf.python_io.TFRecordWriter(tfrecord_dir)
    groundtruth_text_file_path = os.path.join(dataset_dir, '{}_gt.txt'.format(train_or_test))

    with open(groundtruth_text_file_path, 'r') as f:
        im_gt_list = [line.strip().split(',') for line in f.readlines()]
        for im_gt in im_gt_list:
            img_name = im_gt[0]
            gt = im_gt[1]
            logging.info('ImagName:{}, GroundTruth:{}'.format(img_name, gt))

            img_path = os.path.join(dataset_dir, img_name)
            img = Image.open(img_path)
            img = img.convert('RGB')

            # save img object to bytes fromat
            img_height = img.size[1]
            img_width = img.size[0]
            img_buff = io.BytesIO()
            img.save(img_buff, format='jpeg')
            bytes_image = img_buff.getvalue()
            filename = os.path.basename(img_path)

            # tfrecord object
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': _bytes_feature(bytes_image),
                'image/format': _bytes_feature('jpeg'.encode('utf-8')),
                'image/filename': _bytes_feature(filename.encode('utf-8')),
                'image/height': _int64_feature(img_height),
                'image/width': _int64_feature(img_width),
                'image/channels': _int64_feature(3),
                'image/colorspace': _bytes_feature('rgb'.encode('utf-8')),
                'image/groundtruth': _bytes_feature(gt.encode('utf-8'))
            }))

            writer.write(example.SerializeToString())
    writer.close()
    logging.info('tfrecor created')


if __name__ == '__main__':
    create_tfrecord(FLAGS.data_dir, FLAGS.output_path, "train")