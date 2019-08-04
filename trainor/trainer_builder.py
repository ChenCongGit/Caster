# -*- coding: utf-8 -*-

import tensorflow as tf

from Caster.protos import optimizer_pb2
from Caster.utils import learning_schedules

def optimizer_build(optimizer_option, global_summaries):
    """
    根据optimizer_option选择优化方法，返回优化器对象
    """
    if not isinstance(optimizer_option, optimizer_pb2.Optimizer):
        raise ValueError('config not of type Optimizer')

    optimizer_oneof = optimizer_option.WhichOneof('optimizer')

    if optimizer_oneof == 'gradient_descent_optimizer':
        optimizer_config = optimizer_option.gradient_descent_optimizer
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate_build(optimizer_config.learning_rate, global_summaries)
        )
    
    if optimizer_oneof == 'rms_prop_optimizer':
        optimizer_config = optimizer_option.rms_prop_optimizer
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate_build(optimizer_config.learning_rate,global_summaries),
            decay=optimizer_config.decay,
            momentum=optimizer_config.momentum,
            epsilon=optimizer_config.epsilon
        )

    if optimizer_oneof == 'momentum_optimizer':
        optimizer_config = optimizer_option.momentum_optimizer
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate_build(optimizer_config.learning_rate,global_summaries),
            momentum=optimizer_config.momentum
        )

    if optimizer_oneof == 'adam_optimizer':
        optimizer_config = optimizer_option.adam_optimizer
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate_build(optimizer_config.learning_rate,global_summaries)
        )

    if optimizer_oneof == 'nadam_optimizer':
        optimizer_config = optimizer_option.nadam_optimizer
        optimizer = tf.contrib.opt.NadamOptimizer(
            learning_rate=learning_rate_build(optimizer_config.learning_rate,global_summaries)
        )

    if optimizer_oneof == 'adagrad_optimizer':
        optimizer_config = optimizer_option.adagrad_optimizer
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=learning_rate_build(optimizer_config.learning_rate,global_summaries)
        )
    
    if optimizer_oneof == 'adadelta_optimizer':
        optimizer_config = optimizer_option.adadelta_optimizer
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=learning_rate_build(optimizer_config.learning_rate,global_summaries),
            rho=optimizer_config.rho
        )

    if optimizer_option.use_moving_average:
        optimizer = tf.contrib.opt.MovingAverageOptimizer(
            optimizer, average_decay=optimizer_option.moving_average_decay)

    return optimizer


def learning_rate_build(learning_rate_option,global_summaries):
    """
    根据配置信息设置优化器学习率
    """
    if not isinstance(learning_rate_option, optimizer_pb2.LearningRate):
        raise ValueError('config not of type LearningRate')

    learning_rate_oneof = learning_rate_option.WhichOneof('learning_rate')
    if learning_rate_oneof == 'constant_learning_rate':
        learning_rate = learning_rate_option.constant_learning_rate.learning_rate

    if learning_rate_oneof == 'exponential_decay_learning_rate':
        config = learning_rate_option.exponential_decay_learning_rate
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate=config.initial_learning_rate,
            global_step=tf.train.get_or_create_global_step(),
            decay_steps=config.decay_steps,
            decay_rate=config.decay_rate,
            staircase=config.staircase
        )

    if learning_rate_oneof == 'manual_step_learning_rate':
        config = learning_rate_option.manual_step_learning_rate
        if not config.schedule:
            raise ValueError('Empty learning rate schedule.')
        learning_rate_step_boundaries = [x.step for x in config.schedule]
        learning_rate_sequence = [config.initial_learning_rate]
        learning_rate_sequence += [x.learning_rate for x in config.schedule]
        learning_rate = learning_schedules.manual_stepping(
            tf.train.get_or_create_global_step(), learning_rate_step_boundaries,
            learning_rate_sequence)

    # summary记录学习率标量变化曲线
    global_summaries.add(tf.summary.scalar('LearningRate',learning_rate))

    return learning_rate