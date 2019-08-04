# -*- coding: utf-8 -*-

import json
import os
import logging

import tensorflow as tf
from google.protobuf import text_format

from Caster.trainor import trainer
from Caster.model.builder import model_builder
from Caster.protos import config_pb2
from Caster.utils import log

logging.getLogger('tensorflow').propagate = False # avoid logging duplicates


flags = tf.app.flags
flags.DEFINE_integer('num_clones',1,'Num of clones to deploy per worker')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_string('dataset_dir','./Caster/datasets/', '...')
flags.DEFINE_string('exp_dir','./Caster/experiments/','Directory containing config, training log and evaluations')
FLAGS = flags.FLAGS


def _read_config():
    config_path = os.path.join(FLAGS.exp_dir, 'config/trainval.prototxt')
    trainval_config = config_pb2.TrainEvalConfig()

    with tf.gfile.GFile(config_path, 'r') as f:
        text_format.Merge(f.read(), trainval_config)

    input_config = trainval_config.train_input_reader
    model_config = trainval_config.model
    train_config = trainval_config.train_config

    return input_config, model_config, train_config


def main(_):
    
    # 配置训练路径参数
    if not FLAGS.exp_dir:
        raise ValueError('give a path to save trained model and get config files')
    if not FLAGS.dataset_dir:
        raise ValueError('give a path to get datasets')

    model_log_dir = os.path.join(FLAGS.exp_dir,'log')
    tf.logging.set_verbosity(tf.logging.INFO)
    logging.basicConfig(filename=os.path.join(model_log_dir,'log.txt'), level=logging.INFO) # 配置日志消息和格式

    config_dir = os.path.join(FLAGS.exp_dir,'config')
    pretained_model_dir = os.path.join(FLAGS.exp_dir,'checkpoint_dir')
    dataset_dir = FLAGS.dataset_dir

    Path_dict = {'log_dir':model_log_dir,
                 'config_dir':config_dir,
                 'dataset_dir':dataset_dir,
                 'pretained_checkpoint_dir':pretained_model_dir}

    # 加载数据输入参数，模型参数以及训练参数
    input_config, model_config, train_config = _read_config()
    Config_dict = {'input_config':input_config,
                   'model_config':model_config,
                   'train_config':train_config}
    
    # 配置分布式训练
    """
    获得多个设备端口的端口号，当在不同的终端窗口中使用不同的显卡进行分布式时，终端中环境变量将会通过TF_CONFIG
    记录分布式集群信息，使用tf.train.ClusterSpec创建由这些设备组成的分布式训练集群，常见的TF_CONFIG值如下：
    {
        "cluster":{
            "master":[ 
                "distributed-mnist-master-0:2222"
            ],
            "ps":[  
                "distributed-mnist-ps-0:2222"
            ],
            "worker":[  
                "distributed-mnist-worker-0:2222",
                "distributed-mnist-worker-1:2222"
            ]
        },
        "task":{  
            "type":"worker",
            "index":0
        },
        "environment":"cloud"
    }
    """
    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster',None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task',None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # 初始化训练分布式参数字典，初始值为单工作站下的变量值
    Distributed_dict = {'master':'',
                        'task':0,
                        'num_clones':FLAGS.num_clones,
                        'worker_replicas':1,
                        'clone_on_cpu':FLAGS.clone_on_cpu,
                        'ps_tasks':0,
                        'worker_job_name':'lonely_worker',
                        'is_chief':True}
    
    worker_replicas = 1
    ps_tasks = 0
    # 工作站包括'master'和'worker',即主工作站和其他工作站
    if cluster_data and 'worker' in cluster_data:
        worker_replicas = len(cluster_data['worker']) + 1
    # 设备集群中'ps'参数服务器的数量
    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])
    
    if  worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('Distributed training must have at least one ps')

    if worker_replicas >= 1 and ps_tasks >= 1:
        """
        配置分布式训练
        """
        server = tf.train.Server(tf.train.ClusterSpec(cluster),
                                 protocol='grpc',
                                 job_name=task_data['type'],
                                 task_index=task_data['index'])
        
        # 将运行参数服务器终端挂起
        if task_data['type'] == 'ps':
            server.join()
            return
        
        Distributed_dict['master'] = server.target
        Distributed_dict['task'] = task_data['index']
        Distributed_dict['worker_replicas'] = worker_replicas
        Distributed_dict['ps_tasks'] = ps_tasks
        Distributed_dict['worker_job_name'] = '%s/task:%d' %(task_data['type'],task_data['index'])
        Distributed_dict['is_chief'] = (task_data['type']=='master')
    
    trainer.train(Path_dict, Config_dict, Distributed_dict)



if __name__ == '__main__':
    tf.app.run()