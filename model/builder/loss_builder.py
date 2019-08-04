import tensorflow as tf

from Caster.model.model import Loss

from Caster.protos import loss_pb2


def build(config):
    if not isinstance(config, loss_pb2.Loss):
        raise ValueError('config not of type loss_pb2.Loss')

    loss_oneof = config.WhichOneof('loss_oneof')
    if loss_oneof == 'sequence_cross_entropy_loss':
        sequence_cross_entropy_loss_config = config.sequence_cross_entropy_loss
        return Loss.SequenceCrossEntropyLoss(
            sequence_normalize=sequence_cross_entropy_loss_config.sequence_normalize,
            sample_normalize=sequence_cross_entropy_loss_config.sample_normalize,
            weight=sequence_cross_entropy_loss_config.weight
        )
    elif loss_oneof == 'tfseq2seq_loss':
        raise NotImplementedError
    elif loss_oneof == 'l2_regression_loss':
        loss_config = config.l2_regression_loss
        return Loss.STNRegressionLoss(weight=loss_config.weight)
    else:
        raise ValueError('Unknown loss_oneof: {}'.format(loss_oneof))
