import os
import logging
import tensorflow as tf
import numpy as np
import editdistance
import functools

from PIL import Image

from Caster.input.builder import input_builder
from Caster.model.builder import model_builder
from Caster.evals import eval_util
from Caster.utils import shape_utils


EVAL_METRICS_FN_DICT = {
  'recognition_metrics': eval_util.evaluate_recognition_results,
}


def _extract_prediction_tensors(model_func, input_data_func, evaluate_with_lexicon=False):
    """
    输入测试数据前向传播，得到预测结果
    """
    # 读取测试集数据
    batch_queue = input_data_func()
    read_data = batch_queue.dequeue()
    image = read_data['image'][0]
    groundtruth_text = read_data['groundtruth_text'][0]
    filename = read_data['filename'][0]

    # 前向传播
    model = model_func()
    predictions_dict = model.predict(tf.expand_dims(image, axis=0))
    recognitions_dict = model.postprocess(predictions_dict) # [1, ]

    # 有字典纠正
    def _lexicon_search(lexicon, word):
        edit_distances = []
        for lex_word in lexicon:
            edit_distances.append(editdistance.eval(lex_word.lower(), word.lower()))
            edit_distances = np.asarray(edit_distances, dtype=np.int)
            argmin = np.argmin(edit_distances)
        return lexicon[argmin]

    # 是否通过字典进行纠正
    if evaluate_with_lexicon:
        lexicon = input_data_func['lexicon']
        recognition_text = tf.py_func(_lexicon_search, 
                [lexicon, recognitions_dict['text'][0]], tf.string, stateful=False)
    else:
        recognition_text = recognitions_dict['text'][0]

    # 保存预测结果
    tensor_dict = {'image': image,
                   'filename': filename,
                   'groundtruth_text': groundtruth_text,
                   'recognition_text': recognition_text}
    if 'control_points' in predictions_dict:
        tensor_dict.update({
            'control_points': predictions_dict['control_points'],
            'rectified_images': predictions_dict['rectified_images']
        })

    return tensor_dict


def evaluate(path_dict, config_dict, repeat_evaluation=True):
    """
    模型的测试函数
    """
    # 配置加载数据集输入
    input_data_func = functools.partial(input_builder.build, config=config_dict['input_config'], path=path_dict['dataset_dir'])

    # 配置加载网络模型图
    model_func = functools.partial(model_builder.build, config=config_dict['model_config'], is_training=False)

    # 获得预测结果张量
    tensor_dict = _extract_prediction_tensors(model_func, input_data_func, config_dict['eval_config'].eval_with_lexicon)

    summary_writer = tf.summary.FileWriter(path_dict['eval_dir'])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session('', graph=tf.get_default_graph(), config=config)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

    variables_to_restore = tf.global_variables()
    global_step = tf.train.get_or_create_global_step()
    variables_to_restore.append(global_step)
    saver = tf.train.Saver(variables_to_restore)
    
    latest_checkpoint = tf.train.latest_checkpoint(path_dict['log_dir'])
    saver.restore(sess, latest_checkpoint)

    result_lists = {key: [] for key in list(set(tensor_dict.keys()))}

    with tf.contrib.slim.queues.QueueRunners(sess):
        

