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
    
    def _process_batch(tensor_dict, sess, batch_index, counters, update_op):
        eval_config = config_dict['eval_config']
        if batch_index >= eval_config.num_visualizations:
            if '111_image' in tensor_dict:
                tensor_dict = {k: v for (k, v) in tensor_dict.items()
                            if k != 'image'}
        try:
            (result_dict, _) = sess.run([tensor_dict, update_op])
            print('result_dict_filename', result_dict['filename'])
            print('result_dict_groundtruth_text', result_dict['groundtruth_text'])
            print('result_dict_recognition_text', result_dict['recognition_text'])

            img = Image.fromarray(np.uint8(result_dict['image']))
            img.save('/home/AI/chencong/Caster/experiments/eval/{}.jpg'.format(str(result_dict['filename']).split('.')[0]))

            counters['success'] += 1
        except tf.errors.InvalidArgumentError:
            logging.info('Skipping image')
            counters['skipped'] += 1
            return {}
        global_step = tf.train.global_step(sess, tf.train.get_global_step())
        if batch_index < eval_config.num_visualizations:
            eval_util.visualize_recognition_results(
                result_dict,
                'Recognition_{}'.format(batch_index),
                global_step,
                summary_dir=path_dict['eval_dir'],
                export_dir=os.path.join(path_dict['eval_dir'], 'vis'),
                summary_writer=summary_writer,
                only_visualize_incorrect=eval_config.only_visualize_incorrect)

        return result_dict
    
    def _process_aggregated_results(result_lists):
        eval_metric_fn_key = eval_config.metrics_set
        if eval_metric_fn_key not in EVAL_METRICS_FN_DICT:
            raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
        return EVAL_METRICS_FN_DICT[eval_metric_fn_key](result_lists)
    
    
    # 加载ckpt模型变量
    """
    Global variables are variables that are shared across machines in a distributed environment. 
    The Variable() constructor or get_variable() automatically adds new variables to the graph 
    collection GraphKeys.GLOBAL_VARIABLES. This convenience function returns the contents of 
    that collection.
    """
    variables_to_restore = tf.global_variables()
    global_step = tf.train.get_or_create_global_step()
    variables_to_restore.append(global_step)
    saver = tf.train.Saver(variables_to_restore)
    def _restore_latest_checkpoint(sess):
        latest_checkpoint = tf.train.latest_checkpoint(path_dict['log_dir'])
        saver.restore(sess, latest_checkpoint)

    eval_config = config_dict['eval_config']
    eval_util.repeated_checkpoint_run(
        tensor_dict=tensor_dict,
        update_op=tf.no_op(),
        summary_dir=path_dict['eval_dir'],
        aggregated_result_processor=_process_aggregated_results,
        batch_processor=_process_batch,
        checkpoint_dirs=[path_dict['log_dir']],
        variables_to_restore=None,
        restore_fn=_restore_latest_checkpoint,
        num_batches=eval_config.num_examples,
        eval_interval_secs=eval_config.eval_interval_secs,
        max_number_of_evaluations=(
            1 if eval_config.ignore_groundtruth else
            eval_config.max_evals if eval_config.max_evals else
            None if repeat_evaluation else 1),
        master=eval_config.eval_master,
        save_graph=eval_config.save_graph,
        save_graph_dir=(path_dict['eval_dir'] if eval_config.save_graph else ''))
    
    summary_writer.close()


    


