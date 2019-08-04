import os
import functools
import logging
import tensorflow as tf

from google.protobuf import text_format

from Caster.protos import config_pb2
from Caster.evals import evalter



logging.getLogger('tensorflow').propagate = False # avoid logging duplicates
tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)

flags = tf.app.flags
flags.DEFINE_boolean('repeat', True, 'If true, evaluate repeatedly.')
flags.DEFINE_string('exp_dir','./Caster/experiments/','Directory containing config, training log and evaluations')
flags.DEFINE_string('dataset_dir','./Caster/datasets/', '...')
FLAGS = flags.FLAGS


def _read_config():
    config_path = os.path.join(FLAGS.exp_dir, 'config/trainval.prototxt')
    trainval_config = config_pb2.TrainEvalConfig()

    with tf.gfile.GFile(config_path, 'r') as f:
        text_format.Merge(f.read(), trainval_config)

    input_config = trainval_config.eval_input_reader
    model_config = trainval_config.model
    eval_config = trainval_config.eval_config

    return input_config, model_config, eval_config


def main(_):

    # 配置测试路径参数
    if not FLAGS.exp_dir:
        raise ValueError('give a path to save trained model and get config files')
    if not FLAGS.dataset_dir:
        raise ValueError('give a path to get datasets')

    model_log_dir = os.path.join(FLAGS.exp_dir,'log')
    config_dir = os.path.join(FLAGS.exp_dir,'config')
    eval_dir = os.path.join(FLAGS.exp_dir, 'eval')
    dataset_dir = FLAGS.dataset_dir
    
    Path_dict = {'log_dir':model_log_dir,
                 'config_dir':config_dir,
                 'eval_dir':eval_dir,
                 'dataset_dir':dataset_dir}

    # 加载数据输入参数，模型参数以及测试参数
    input_config, model_config, eval_config = _read_config()
    Config_dict = {'input_config':input_config,
                   'model_config':model_config,
                   'eval_config':eval_config}

    evalter.evaluate(Path_dict, Config_dict, FLAGS.repeat)



if __name__ == '__main__':
    tf.app.run()
