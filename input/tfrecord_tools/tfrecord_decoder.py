# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/AI/chencong/')
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from Caster.input.preprocess import data_preprocess as preprocessor

slim = tf.contrib.slim


def decode(tfrecord_path, config):
    """
    解码tfrecord文件，返回张量字典
    """
    # 将tf.train.Example反序列化成存储之前的格式
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/channels': tf.FixedLenFeature([], tf.int64, default_value=3),
        'image/colorspace': tf.FixedLenFeature([], tf.string, default_value='rgb'),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/groundtruth': tf.FixedLenFeature([], tf.string),
        'image/lexicon': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    # 将反序列化的数据组装成更高级的格式
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(
            image_key='image/encoded',
            format_key='image/format',
            channels=3
        ),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
        'groundtruth_text': slim.tfexample_decoder.Tensor('image/groundtruth'),
        'lexicon': slim.tfexample_decoder.ItemHandlerCallback(['image/lexicon'], _split_lexicon)
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    labels_to_names = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'filename': 'string',
        'groundtruth_text': 'A string.'}
    
    dataset = slim.dataset.Dataset(
        data_sources=tfrecord_path,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=config.num_samples,
        num_classes=config.num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, 
        num_readers=config.num_readers, 
        common_queue_capacity=config.queue_capacity, 
        common_queue_min=config.queue_min)

    tensor_dict = {'image':None, 'filename':None, 'groundtruth_text':None}
    tensor_dict['image'], tensor_dict['filename'], tensor_dict['groundtruth_text'] = provider.get(['image', 'filename', 'groundtruth_text'])

    return tensor_dict
  

def _split_lexicon(keys_to_tensors):
    joined_lexicon = keys_to_tensors['image/lexicon']
    lexicon_sparse = tf.string_split([joined_lexicon], delimiter='\t')
    lexicon = tf.sparse_tensor_to_dense(lexicon_sparse, default_value='')[0]
    return lexicon



# flags = tf.app.flags
# flags.DEFINE_string('tfrecord_path', '/home/AI/chencong/Caster/datasets/ocr_test.tfrecord', 'tfrecord file path')
# flags.DEFINE_integer('num_classes', 856, 'num_class')
# flags.DEFINE_integer('num_samples', 11600, 'num_class')
# flags.DEFINE_string('batch_data_dir_path', '/home/AI/chencong/Caster/datasets/preprocessed_img/','save one batch data to')
# FLAGS = flags.FLAGS

# def create_input_batch(tensor_dict, preprocess_option):
#     """
#     输入数据预处理并组合成batch形式
#     """
#     tensor_dict['image'] = tf.to_float(tensor_dict['image'])
#     # tensor_dict['filename'] = tf.reshape(tensor_dict['filename'], [1])
#     # tensor_dict['groundtruth_text'] = tf.reshape(tensor_dict['groundtruth_text'], [1])
#     tensor_dict = preprocessor.preprocess(tensor_dict, preprocess_option)
#     data_batch = tf.train.batch(
#         tensor_dict,
#         batch_size=32,
#         num_threads=1,
#         capacity=256,
#         shapes=None,
#         allow_smaller_final_batch=False
#     )
#     return tensor_dict, data_batch


# if __name__ == '__main__':
#     tensor_dict = decode(FLAGS.tfrecord_path)
#     preprocess_option = [#(preprocessor.rgb_to_gray, {'three_channel':True}),
#                          (preprocessor.resize_padding_image, {})]
#     tensor_dict, data_batch = create_input_batch(tensor_dict, preprocess_option)
#     batch_queue = slim.prefetch_queue.prefetch_queue(data_batch, capacity = 256)
#     read_data = batch_queue.dequeue()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run(tf.local_variables_initializer())
#         coord=tf.train.Coordinator()
#         threads= tf.train.start_queue_runners(coord=coord)
#         read_data = sess.run(read_data)
#         # print(read_data)
#         for i in range(len(read_data['image'])):
#             image = read_data['image'][i].astype(int)
#             filename = read_data['filename'][i].decode('utf-8')
#             print(image.shape)
#             cv2.imwrite(FLAGS.batch_data_dir_path + filename, image)

#         coord.request_stop()
#         coord.join(threads)