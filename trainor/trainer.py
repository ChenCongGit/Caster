# -*- coding: utf-8 -*-

import os
import logging
import functools

import tensorflow as tf

from Caster.input.builder import input_builder
from Caster.input.builder import preprocess_builder
from Caster.model.builder import model_builder
from Caster.trainor import trainer_builder
from Caster.utils import model_deploy
from Caster.utils import profile_session_run_hooks


def data_dequeue(input_queue_list):
    """
    从不同的数据集中解码取数据队列，并融合为一个数据列表，这样能够将不同的数据集组合到同一个BATCH中
    """
    images_list = []
    filename_list = []
    groundtruth_text_list = []

    for input_queue in input_queue_list:
        read_data = input_queue.dequeue()
        images_list.append(read_data['image'])
        filename_list.append(read_data['filename'])
        groundtruth_text_list.append(read_data['groundtruth_text'])
    return images_list, groundtruth_text_list


def forward(input_queue_list, model_func, global_summaries):
    """
    创建模型前向传播函数，在model_deploy.py文件中运行
    """
    # 构建网络模型图，初始化各个模型类
    model = model_func()

    if not isinstance(input_queue_list, (list, tuple)):
        input_queue_list = [input_queue_list]

    # 读取数据队列，返回不同数据集batch组成的列表
    image_list, groundtruth_text_list = data_dequeue(input_queue_list)

    # 组合不同数据集为一个大batch
    images = tf.concat(image_list, axis=0)
    groundtruth_texts = tf.concat(groundtruth_text_list, axis=0)
    global_summaries.add(tf.summary.image('input_batch_image', images))

    # 提供标签，进行预测以及计算损失
    model.provide_groundtruth(groundtruth_texts, scope='ModelGT')
    predictions_dict = model.predict(images, scope='ModelPredict')
    losses_dict = model.loss(predictions_dict, scope='ModelLoss')

    for loss_name, loss_tensor in losses_dict.items():
        tf.losses.add_loss(loss_tensor)


def train(path_dict, config_dict, distributed_dict):
    """
    模型的训练函数
    """
    # 配置加载数据集输入，可以加载多个训练数据集，所以这里保存为加载函数列表
    input_data_func_list = [functools.partial(input_builder.build,config=config,path=path_dict['dataset_dir'])
                            for config in config_dict['input_config']]
    
    # 配置加载网络模型
    model_func = functools.partial(model_builder.build,config=config_dict['model_config'],is_training=True)
    
    # 构建计算图
    with tf.Graph().as_default():
        
        # Deploy Slim models across multiple clones and replicas
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=distributed_dict['num_clones'],
            clone_on_cpu=distributed_dict['clone_on_cpu'],
            replica_id=distributed_dict['task'],
            num_replicas=distributed_dict['worker_replicas'],
            num_ps_tasks=distributed_dict['ps_tasks'],
            worker_job_name=distributed_dict['worker_job_name'],
        )

        # 在所有的“参数服务器ps”中都存储定义的variables变量
        with tf.device(deploy_config.variables_device()):
            global_step = tf.train.create_global_step()

        # 在所有的“工作站worker”中传入输入队列
        with tf.device(deploy_config.inputs_device()),tf.name_scope('Input'):
            # 读取数据，这里用列表表示不同数据集的数据队列            
            input_queue_list = []
            for input_data_func in input_data_func_list:
                batch_queue = input_data_func()
                input_queue_list.append(batch_queue)

        # 定义tensorboard的summaries，所有默认图中创建的summaries都保存在集合tf.GraphKeys.SUMMARIES中
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        # global_summaries用于记录所有clones的全局的信息
        global_summaries = set([])

        # 定义网络模型model_fn，并将模型变量复制到所有设备
        """
        create_clones函数返回由下面形式的Clone组成的列表，保存克隆的设备，变量空间和模型输出信息。
        Clone = collections.namedtuple('Clone',
                               ['outputs',  # Whatever model_fn() returned.
                                'scope',  # The scope used to create it.
                                'device',  # The device used to create.
                               ])
        """
        model_forward = functools.partial(forward, model_func=model_func, global_summaries=global_summaries)
        clones = model_deploy.create_clones(deploy_config, model_forward, [input_queue_list])
        
        # 获得clones中每一个clone的变量空间中的UPDATE_OPS的值，这个值包括在model_fn中的可更新的变量
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, clones[0].scope)

        # 在所有“工作站worker”中定义同步梯度更新优化器
        with tf.device(deploy_config.optimizer_device()),tf.name_scope('Optimizer'):
            optimizer_option = config_dict['train_config'].optimizer
            training_optimizer = trainer_builder.optimizer_build(optimizer_option,global_summaries)

        # 分布式训练同步梯度优化，多个工作站协同创建分布式优化器
        """
        tf.train.SyncReplicasOptimizer设置同步梯度更新分布式优化，
        由于不同的worker计算梯度的速度有较大的差异，同步梯度更新将会受限于运算速
        度最慢的那个设备，所以在这里使用两个参数，通过backup worker方法计算
        ####参数total_num_replicas表示将原始的给所有的workers分发的batch的个数，
        ####参数replicas_to_aggregate表示由上面的这么多个batch计算出的梯度，其中
        选择部分batch用来更新权值，batch个数
        """
        sync_config = config_dict['train_config'].sync_replicas
        if sync_config:
            training_optimizer = tf.train.SyncReplicasOptimizer(
                opt=training_optimizer,
                replicas_to_aggregate=config_dict['train_config'].replicas_to_aggregate,
                total_num_replicas=config_dict['train_config'].total_num_replicas
            )

        # 加载预训练模型checkpoint
        """
        |--checkpoint_dir
        |    |--checkpoint
        |    |--MyModel.meta
        |    |--MyModel.data-00000-of-00001
        |    |--MyModel.index
        """
        init_fn = None
        # if config_dict['train_config'].fine_tune_pretrained:
        #     pretrained_checkpoint_dir = path_dict['pretained_checkpoint_dir']
        #     init_saver = tf.train.import_meta_graph(pretrained_checkpoint_dir + '/model.meta')
        #     def initialize_fn(sess):
        #         init_saver.restore(sess, pretrained_checkpoint_dir)
        #     init_fn = initialize_fn

        with tf.device(deploy_config.optimizer_device()),tf.variable_scope('OptimizerClones'):
            # 对上面的所有clone计算总损失，但要注意分布式同步优化器的两个重要参数
            total_loss, grads_and_vars = model_deploy.optimize_clones(
                clones=clones,
                optimizer=training_optimizer,
                regularization_losses=None
            )
            total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

            # 梯度更新updates
            grad_updates = training_optimizer.apply_gradients(
                grads_and_vars,
                global_step=global_step
            )
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)

            # tf.control_dependencies用于控制计算过程，即先进行更新操作，再计算total_loss
            # tf.identity表示创建一个输入张量的副本，返回一个新的张量，通常与control_dependencies一起使用
            # 在这里用作total_loss执行过的标志——即创建了一个新的临时节点
            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(total_loss, name='train_op')

        # 创建summary操作
        for (grad,var) in grads_and_vars:
            var_name = var.op.name
            grad_name = 'grad/' + var_name
            global_summaries.add(tf.summary.histogram(grad_name,grad))
            global_summaries.add(tf.summary.histogram(var_name,var))

        total_loss = tf.losses.get_total_loss()
        global_summaries.add(tf.summary.scalar('Total_loss',total_loss))
        for loss_tensor in tf.losses.get_losses():
            global_summaries.add(tf.summary.scalar(loss_tensor.op.name,loss_tensor))

        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, clones[0].scope))
        summaries |= global_summaries

        summary_op = tf.summary.merge(list(summaries),name='summary_op')

        # 配置session参数
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )

        # 保存训练模型
        keep_checkpoint_every_n_hours = config_dict['train_config'].keep_checkpoint_every_n_hours
        saver = tf.train.Saver(
            max_to_keep=10,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours
        )

        # When you build a model for training you usually need ops to initialize variables,
        # a Saver to checkpoint them, an op to collect summaries for the visualizer, and so on.
        scaffold = tf.train.Scaffold(
            init_fn=None,
            saver=saver,
            summary_op=summary_op,
        )

        # 定义ps和worker程序终止条件，
        stop_hook = tf.train.StopAtStepHook(
            num_steps=(config_dict['train_config'].num_steps if config_dict['train_config'].num_steps else None)
        )
        profile_hook = profile_session_run_hooks.ProfileAtStepHook(
            at_step=200,
            checkpoint_dir=path_dict['log_dir']
        )

        tensors_to_log = {"global_step": global_step, "total_loss": total_loss}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)

        # train
        tf.contrib.training.train(
            train_op=train_tensor,
            logdir=path_dict['log_dir'],
            master=distributed_dict['master'],
            is_chief=distributed_dict['is_chief'],
            scaffold=scaffold,
            hooks=[stop_hook, profile_hook, logging_hook],
            chief_only_hooks=None,
            save_checkpoint_secs=config_dict['train_config'].save_checkpoint_secs,
            save_summaries_steps=config_dict['train_config'].save_summaries_steps,
            config=session_config
        )