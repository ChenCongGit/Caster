import tensorflow as tf

from Caster.utils import shape_utils


class SequenceCrossEntropyLoss(object):
    """
    识别文本序列与标签文本序列的交叉熵损失
    """
    def __init__(self,
                 sequence_normalize=None,
                 sample_normalize=None,
                 weight=None):

        self._sequence_normalize = sequence_normalize
        self._sample_normalize = sample_normalize
        self._weight = weight

    
    def __call__(self, logits, labels, lengths, scope=None):
        """
        Args:
            logits: float32 tensor with shape [batch_size, max_time, num_classes]
            labels: int32 tensor with shape [batch_size, max_time]
            lengths: int32 tensor with shape [batch_size]
        
        tf.nn.sparse_softmax_cross_entropy_with_logits:
        A common use case is to have logits and labels of shape [batch_size, num_classes], 
        but higher dimensions are supported, in which case the dim-th dimension is assumed 
        to be of size num_classes. logits and labels must have the same dtype (either float16, 
        float32, or float64).
        """
        with tf.name_scope(scope, 'SequenceCrossEntropyLoss', [logits, labels, lengths]):
            # 原始交叉熵损失
            raw_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
            batch_size, max_time = shape_utils.combined_static_and_dynamic_shape(labels)
            
            # 计算指定序列长度以内的损失
            mask = tf.less(tf.tile([tf.range(max_time)],[batch_size,1]), tf.expand_dims(lengths,1), name='mask')
            masked_losses = tf.multiply(raw_losses, tf.cast(mask, tf.float32), name='masked_losses') # => [batch_size, max_time]
            row_losses = tf.reduce_sum(masked_losses, 1, name='row_losses') # 序列不同时刻损失值和 [batch_size]
            
            # 损失序列长度归一化
            if self._sequence_normalize:
                loss = tf.truediv(row_losses, tf.cast(tf.maximum(lengths),1),tf.float32, name='seq_normed_losses')
            
            loss = tf.reduce_sum(row_losses)
            
            # 损失batch归一化
            if self._sample_normalize:
                loss = tf.truediv(loss, tf.cast(tf.maximum(batch_size, 1),tf.float32))

            # 交叉熵损失权值
            if self._weight:
                loss = loss * self._weight
            return loss
            


class STNRegressionLoss(object):
    """
    STN矫正定位网络回归损失（平方损失）
    """
    def __init__(self, weight):
        self._weight = weight


    def __call__(self, prediction, target, scope=None):
        """
        Args:
            prediction: float32 tensor with shape [batch_size, 2 * num_control_point]
            target: int32 tensor with shape [batch_size, 2 * num_control_point]
        """
        with tf.name_scope(scope, 'STNRegressionLoss', [prediction, target]):
            diff = prediction - target
            losses = tf.reduce_sum(tf.square(diff), axis=1) # 2K维度计算损失和
            loss = tf.reduce_mean(losses, axis=0) # batch维度计算平均损失

            # 关键点回归损失权值
            if self._weight:
                loss = loss * self._weight
            return loss
