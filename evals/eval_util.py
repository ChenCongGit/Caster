# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Common functions for repeatedly evaluating a checkpoint.
"""
import copy
import logging
import os
import time

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Caster.utils import recognition_evaluation
from Caster.utils import visualization_utils as vis_utils


def write_metrics(metrics, global_step, summary_dir):
  """Write metrics to a summary directory.

  Args:
    metrics: A dictionary containing metric names and values.
    global_step: Global step at which the metrics are computed.
    summary_dir: Directory to write tensorflow summaries to.
  """
  logging.info('Writing metrics to tf summary.')
  summary_writer = tf.summary.FileWriter(summary_dir)
  for key in sorted(metrics):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=key, simple_value=metrics[key]),
    ])
    summary_writer.add_summary(summary, global_step)
    logging.info('%s: %f', key, metrics[key])
  summary_writer.close()
  logging.info('Metrics written to tf summary.')


def run_checkpoint_once(tensor_dict,
                        update_op,
                        summary_dir,
                        aggregated_result_processor=None,
                        batch_processor=None,
                        checkpoint_dirs=None,
                        variables_to_restore=None,
                        restore_fn=None,
                        num_batches=1,
                        master='',
                        save_graph=False,
                        save_graph_dir='',
                        metric_names_to_values=None,
                        keys_to_exclude_from_results=()):
  """Evaluates both python metrics and tensorflow slim metrics.

  Python metrics are processed in batch by the aggregated_result_processor,
  while tensorflow slim metrics statistics are computed by running
  metric_names_to_updates tensors and aggregated using metric_names_to_values
  tensor.
  """
  if save_graph and not save_graph_dir:
    raise ValueError('`save_graph_dir` must be defined.')
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  sess = tf.Session(master, graph=tf.get_default_graph(), config=config)
  sess.run([
    tf.global_variables_initializer(),
    tf.local_variables_initializer(),
    tf.tables_initializer()])
  if restore_fn:
    restore_fn(sess)
  else:
    if not checkpoint_dirs:
      raise ValueError('`checkpoint_dirs` must have at least one entry.')
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dirs[0])
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint_file)

  if save_graph:
    tf.train.write_graph(sess.graph_def, save_graph_dir, 'eval.pbtxt')

  valid_keys = list(set(tensor_dict.keys()) - set(keys_to_exclude_from_results))
  result_lists = {key: [] for key in valid_keys}
  counters = {'skipped': 0, 'success': 0}
  other_metrics = None
  with tf.contrib.slim.queues.QueueRunners(sess):
    try:
      for batch in range(int(num_batches)):
        if (batch + 1) % 100 == 0:
          logging.info('Running eval ops batch %d/%d', batch + 1, num_batches)
        if not batch_processor:
          try:
            (result_dict, _) = sess.run([tensor_dict, update_op])
            counters['success'] += 1
          except tf.errors.InvalidArgumentError:
            logging.info('Skipping image')
            counters['skipped'] += 1
            result_dict = {}
        else:
          result_dict = batch_processor(
              tensor_dict, sess, batch, counters, update_op)
        for key in result_dict:
          if key in valid_keys:
            result_lists[key].append(result_dict[key])
      if metric_names_to_values is not None:
        other_metrics = sess.run(metric_names_to_values)
      logging.info('Running eval batches done.')
    except tf.errors.OutOfRangeError:
      logging.info('Done evaluating -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      metrics = aggregated_result_processor(result_lists)
      if other_metrics is not None:
        metrics.update(other_metrics)
      global_step = tf.train.global_step(sess, tf.train.get_global_step())
      write_metrics(metrics, global_step, summary_dir)
      logging.info('# success: %d', counters['success'])
      logging.info('# skipped: %d', counters['skipped'])
  sess.close()

# TODO: Add tests.
def repeated_checkpoint_run(tensor_dict,
                            update_op,
                            summary_dir,
                            aggregated_result_processor=None,
                            batch_processor=None,
                            checkpoint_dirs=None,
                            variables_to_restore=None,
                            restore_fn=None,
                            num_batches=1,
                            eval_interval_secs=120,
                            max_number_of_evaluations=None,
                            master='',
                            save_graph=False,
                            save_graph_dir='',
                            metric_names_to_values=None,
                            keys_to_exclude_from_results=()):
  """Periodically evaluates desired tensors using checkpoint_dirs or restore_fn.

  This function repeatedly loads a checkpoint and evaluates a desired
  set of tensors (provided by tensor_dict) and hands the resulting numpy
  arrays to a function result_processor which can be used to further
  process/save/visualize the results.
  """
  if max_number_of_evaluations and max_number_of_evaluations <= 0:
    raise ValueError(
        '`number_of_steps` must be either None or a positive number.')

  if not checkpoint_dirs:
    raise ValueError('`checkpoint_dirs` must have at least one entry.')

  last_evaluated_model_path = None
  number_of_evaluations = 0
  while True:
    start = time.time()
    logging.info('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                           time.gmtime()))
    model_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
    if not model_path:
      logging.info('No model found in %s. Will try again in %d seconds',
                   checkpoint_dirs[0], eval_interval_secs)
    elif model_path == last_evaluated_model_path:
      logging.info('Found already evaluated checkpoint. Will try again in %d '
                   'seconds', eval_interval_secs)
    else:
      last_evaluated_model_path = model_path
      run_checkpoint_once(tensor_dict, update_op, summary_dir,
                          aggregated_result_processor,
                          batch_processor, checkpoint_dirs,
                          variables_to_restore, restore_fn, num_batches, master,
                          save_graph, save_graph_dir, metric_names_to_values,
                          keys_to_exclude_from_results)
    number_of_evaluations += 1

    if (max_number_of_evaluations and
        number_of_evaluations >= max_number_of_evaluations):
      logging.info('Finished evaluation!')
      break
    time_to_next_eval = start + eval_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)


def evaluate_recognition_results(result_lists):
  expected_keys = [
      'groundtruth_text',
      'recognition_text',
      'filename']
  if not set(expected_keys).issubset(set(result_lists.keys())):
    raise ValueError('result_lists does not have expected key set.')
  num_results = len(result_lists[expected_keys[0]])
  for key in expected_keys:
    if len(result_lists[key]) != num_results:
      raise ValueError('Inconsistent list sizes in result_lists')

  evaluator = recognition_evaluation.RecognitionEvaluation()
  print('11111111111', result_lists['recognition_text'][0])
  print('22222222222', result_lists['groundtruth_text'][0])
  print('33333333333', result_lists['filename'][0])
  for idx, image_id in enumerate(result_lists['filename']):
    evaluator.add_single_image_recognition_info(
        image_id,
        result_lists['recognition_text'][idx],
        result_lists['groundtruth_text'][idx])
  return evaluator.evaluate_all()


def visualize_recognition_results(result_dict, tag, global_step,
                                  summary_dir=None,
                                  export_dir=None,
                                  summary_writer=None,
                                  only_visualize_incorrect=False):
  import string
  from scipy.misc import imresize

  # vis_utils.draw_keypoints_on_image_array(image, control_points[:,::-1], radius=1)
  # summary = tf.Summary(value=[
  #     tf.Summary.Value(tag=tag, image=tf.Summary.Image(
  #         encoded_image_string=vis_utils.encode_image_array_as_png_bytes(image)))])
  # summary_writer.add_summary(summary, global_step)
  # logging.info('Detection visualizations written to summary with tag %s.', tag)

  # export visualization
  fig = plt.figure(frameon=False)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)

  image = result_dict['original_image']
  image_h, image_w, _ = image.shape
  image = imresize(image, (128. / image_w))

  ax.imshow(image)
  image_h, image_w, _ = image.shape
  ax.imshow(image.astype(np.uint8))

  if 'control_points' in result_dict:
    control_points = result_dict['control_points'][0]
    ax.scatter(control_points[:,0] * image_w, control_points[:,1] * image_h, marker='+', c='#42f4aa', s=100)

  def _normalize_text(text):
    text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
    return text.lower()

  gt_text = _normalize_text(result_dict['groundtruth_text'].decode('utf-8'))
  rec_text = _normalize_text(result_dict['recognition_text'].decode('utf-8'))

  if only_visualize_incorrect and gt_text == rec_text:
    return

  # plt.title(result_dict['groundtruth_text'].decode('utf-8'))

  # plt.subplot(2,1,2)
  
  # plt.imshow(rectified_image)
  # plt.title(result_dict['recognition_text'].decode('utf-8'))

  if not os.path.exists(export_dir):
    os.makedirs(export_dir)
  save_path = os.path.join(export_dir, tag + '_original_{}_{}'.format(gt_text, rec_text) + '.pdf')
  plt.savefig(save_path, bbox_inches='tight')
  logging.info('Detailed visualization exported to {}'.format(save_path))

  if 'rectified_images' in result_dict:
    rectified_image = result_dict['rectified_images'][0]
    rectified_image = (127.5 * (rectified_image + 1.0)).astype(np.uint8)
    ax.clear()
    ax.set_axis_off()
    ax.imshow(rectified_image)
    save_path = os.path.join(export_dir, tag + '_rectified' + '.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    logging.info('Detailed visualization exported to {}'.format(save_path))

  plt.close()
