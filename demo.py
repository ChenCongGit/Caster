import os
import re
import logging
import tensorflow as tf
from PIL import Image
from google.protobuf import text_format
import numpy as np

from Caster.protos import config_pb2
from Caster.model.builder import model_builder


# supress TF logging duplicates
logging.getLogger('tensorflow').propagate = False
tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('exp_dir', './Caster/experiments/',
                    'Directory containing config, training log and evaluations')
flags.DEFINE_string('input_image_dir', './Caster/datasets/', 'Demo image')
flags.DEFINE_integer('check_num', 200000, 'choose one model to test')
FLAGS = flags.FLAGS


def _read_config():
    config_path = os.path.join(FLAGS.exp_dir, 'config/trainval.prototxt')
    trainval_config = config_pb2.TrainEvalConfig()

    with tf.gfile.GFile(config_path, 'r') as f:
        text_format.Merge(f.read(), trainval_config)
    model_config = trainval_config.model
    return model_config


def main(_):
  model_config = _read_config()

  model = model_builder.build(model_config, is_training=False)

  input_image_str_tensor = tf.placeholder(
    dtype=tf.string,
    shape=[])
  input_image_tensor = tf.image.decode_jpeg(
    input_image_str_tensor,
    channels=3,
  )
  resized_image_tensor = tf.image.resize_images(
    tf.to_float(input_image_tensor),
    [64, 256])

  predictions_dict = model.predict(tf.expand_dims(resized_image_tensor, 0))
  recognitions = model.postprocess(predictions_dict)
  recognition_text = recognitions['text'][0]
  rectified_images = predictions_dict['rectified_images']

  saver = tf.train.Saver(tf.global_variables())
  check_num = FLAGS.check_num
  checkpoint = os.path.join(FLAGS.exp_dir, 'log/model.ckpt-{}'.format(check_num))

  fetches = {
    'original_image': input_image_tensor,
    'recognition_text': recognition_text,
    'rectified_images': predictions_dict['rectified_images']
  }

  test_dataset_path = os.path.join(FLAGS.input_image_dir, 'ic15/train/')
  f = open(test_dataset_path + 'train_gt.txt')
  img_label_list = [line.split(',') for line in f.readlines()]
  img_name_list = [os.path.join(test_dataset_path, img_label[0]) for img_label in img_label_list]
  label_list = [img_label[1].strip() for img_label in img_label_list]
  
  input_image_str = [open(img_name, 'rb').read() for img_name in img_name_list]

  with tf.Session() as sess:
    sess.run([
      tf.global_variables_initializer(),
      tf.local_variables_initializer(),
      tf.tables_initializer()])
    saver.restore(sess, checkpoint)
    count=0
    count_f=0

    for i in range(len(input_image_str)):
      sess_outputs = sess.run(fetches, feed_dict={input_image_str_tensor: input_image_str[i]})
      # print(sess.run(predictions_dict['Forward/labels'], feed_dict={input_image_str_tensor: input_image_str[i]}))
      # print(sess.run(recognitions, feed_dict={input_image_str_tensor: input_image_str[i]}))
      # print('Recognized text: {}'.format(sess_outputs['recognition_text'].decode('utf-8')))
      # fw=open('aster/data/result_txt','a')
      # fw.write(img_name[i])
      # fw.write('{}'.format(sess_outputs['recognition_text'].decode('utf-8'))+'\n')
      # rectified_image = sess_outputs['rectified_images'][0]
      # rectified_image_pil = Image.fromarray((128 * (rectified_image + 1.0)).astype(np.uint8))
      # input_image_dir = 'aster/result/'
      # rectified_image_save_path = os.path.join(input_image_dir, img_name[i])
      # rectified_image_pil.save(rectified_image_save_path)
      # print('Rectified image saved to {}'.format(rectified_image_save_path))
      print(sess_outputs['recognition_text'])
      predict=sess_outputs['recognition_text'].decode('utf-8')
      label = label_list[i]
      if True:
        count_f+=1
        
        print(label,predict)
        if predict==label:
          count+=1
      print(count, count_f)
    print(count/count_f)


if __name__ == '__main__':
  tf.app.run()
