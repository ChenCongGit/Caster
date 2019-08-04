# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Caster/protos/optimizer.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Caster/protos/optimizer.proto',
  package='Caster.protos',
  serialized_pb=_b('\n\x1d\x43\x61ster/protos/optimizer.proto\x12\rCaster.protos\"\xe0\x03\n\tOptimizer\x12M\n\x1agradient_descent_optimizer\x18\x01 \x01(\x0b\x32\'.Caster.protos.GradientDescentOptimizerH\x00\x12=\n\x12rms_prop_optimizer\x18\x02 \x01(\x0b\x32\x1f.Caster.protos.RMSPropOptimizerH\x00\x12>\n\x12momentum_optimizer\x18\x03 \x01(\x0b\x32 .Caster.protos.MomentumOptimizerH\x00\x12\x36\n\x0e\x61\x64\x61m_optimizer\x18\x04 \x01(\x0b\x32\x1c.Caster.protos.AdamOptimizerH\x00\x12\x38\n\x0fnadam_optimizer\x18\x05 \x01(\x0b\x32\x1d.Caster.protos.NadamOptimizerH\x00\x12>\n\x12\x61\x64\x61\x64\x65lta_optimizer\x18\x06 \x01(\x0b\x32 .Caster.protos.AdadeltaOptimizerH\x00\x12 \n\x12use_moving_average\x18\x07 \x01(\x08:\x04true\x12$\n\x14moving_average_decay\x18\x08 \x01(\x02:\x06\x30.9999B\x0b\n\toptimizer\"N\n\x18GradientDescentOptimizer\x12\x32\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x1b.Caster.protos.LearningRate\"\x95\x01\n\x10RMSPropOptimizer\x12\x32\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x1b.Caster.protos.LearningRate\x12%\n\x18momentum_optimizer_value\x18\x02 \x01(\x02:\x03\x30.9\x12\x12\n\x05\x64\x65\x63\x61y\x18\x03 \x01(\x02:\x03\x30.9\x12\x12\n\x07\x65psilon\x18\x04 \x01(\x02:\x01\x31\"n\n\x11MomentumOptimizer\x12\x32\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x1b.Caster.protos.LearningRate\x12%\n\x18momentum_optimizer_value\x18\x02 \x01(\x02:\x03\x30.9\"C\n\rAdamOptimizer\x12\x32\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x1b.Caster.protos.LearningRate\"D\n\x0eNadamOptimizer\x12\x32\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x1b.Caster.protos.LearningRate\"Z\n\x11\x41\x64\x61\x64\x65ltaOptimizer\x12\x32\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x1b.Caster.protos.LearningRate\x12\x11\n\x03rho\x18\x02 \x01(\x02:\x04\x30.95\"\x8a\x02\n\x0cLearningRate\x12\x45\n\x16\x63onstant_learning_rate\x18\x01 \x01(\x0b\x32#.Caster.protos.ConstantLearningRateH\x00\x12V\n\x1f\x65xponential_decay_learning_rate\x18\x02 \x01(\x0b\x32+.Caster.protos.ExponentialDecayLearningRateH\x00\x12J\n\x19manual_step_learning_rate\x18\x03 \x01(\x0b\x32%.Caster.protos.ManualStepLearningRateH\x00\x42\x0f\n\rlearning_rate\"4\n\x14\x43onstantLearningRate\x12\x1c\n\rlearning_rate\x18\x01 \x01(\x02:\x05\x30.002\"\x97\x01\n\x1c\x45xponentialDecayLearningRate\x12$\n\x15initial_learning_rate\x18\x01 \x01(\x02:\x05\x30.002\x12\x1c\n\x0b\x64\x65\x63\x61y_steps\x18\x02 \x01(\r:\x07\x34\x30\x30\x30\x30\x30\x30\x12\x1a\n\x0c\x64\x65\x63\x61y_factor\x18\x03 \x01(\x02:\x04\x30.95\x12\x17\n\tstaircase\x18\x04 \x01(\x08:\x04true\"\xd0\x01\n\x16ManualStepLearningRate\x12$\n\x15initial_learning_rate\x18\x01 \x01(\x02:\x05\x30.002\x12L\n\x08schedule\x18\x02 \x03(\x0b\x32:.Caster.protos.ManualStepLearningRate.LearningRateSchedule\x1a\x42\n\x14LearningRateSchedule\x12\x0c\n\x04step\x18\x01 \x01(\r\x12\x1c\n\rlearning_rate\x18\x02 \x01(\x02:\x05\x30.002')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_OPTIMIZER = _descriptor.Descriptor(
  name='Optimizer',
  full_name='Caster.protos.Optimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='gradient_descent_optimizer', full_name='Caster.protos.Optimizer.gradient_descent_optimizer', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rms_prop_optimizer', full_name='Caster.protos.Optimizer.rms_prop_optimizer', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='momentum_optimizer', full_name='Caster.protos.Optimizer.momentum_optimizer', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='adam_optimizer', full_name='Caster.protos.Optimizer.adam_optimizer', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='nadam_optimizer', full_name='Caster.protos.Optimizer.nadam_optimizer', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='adadelta_optimizer', full_name='Caster.protos.Optimizer.adadelta_optimizer', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='use_moving_average', full_name='Caster.protos.Optimizer.use_moving_average', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='moving_average_decay', full_name='Caster.protos.Optimizer.moving_average_decay', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.9999,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='optimizer', full_name='Caster.protos.Optimizer.optimizer',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=49,
  serialized_end=529,
)


_GRADIENTDESCENTOPTIMIZER = _descriptor.Descriptor(
  name='GradientDescentOptimizer',
  full_name='Caster.protos.GradientDescentOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='Caster.protos.GradientDescentOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=531,
  serialized_end=609,
)


_RMSPROPOPTIMIZER = _descriptor.Descriptor(
  name='RMSPropOptimizer',
  full_name='Caster.protos.RMSPropOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='Caster.protos.RMSPropOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='momentum_optimizer_value', full_name='Caster.protos.RMSPropOptimizer.momentum_optimizer_value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.9,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='decay', full_name='Caster.protos.RMSPropOptimizer.decay', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.9,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='epsilon', full_name='Caster.protos.RMSPropOptimizer.epsilon', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=612,
  serialized_end=761,
)


_MOMENTUMOPTIMIZER = _descriptor.Descriptor(
  name='MomentumOptimizer',
  full_name='Caster.protos.MomentumOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='Caster.protos.MomentumOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='momentum_optimizer_value', full_name='Caster.protos.MomentumOptimizer.momentum_optimizer_value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.9,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=763,
  serialized_end=873,
)


_ADAMOPTIMIZER = _descriptor.Descriptor(
  name='AdamOptimizer',
  full_name='Caster.protos.AdamOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='Caster.protos.AdamOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=875,
  serialized_end=942,
)


_NADAMOPTIMIZER = _descriptor.Descriptor(
  name='NadamOptimizer',
  full_name='Caster.protos.NadamOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='Caster.protos.NadamOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=944,
  serialized_end=1012,
)


_ADADELTAOPTIMIZER = _descriptor.Descriptor(
  name='AdadeltaOptimizer',
  full_name='Caster.protos.AdadeltaOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='Caster.protos.AdadeltaOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rho', full_name='Caster.protos.AdadeltaOptimizer.rho', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.95,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1014,
  serialized_end=1104,
)


_LEARNINGRATE = _descriptor.Descriptor(
  name='LearningRate',
  full_name='Caster.protos.LearningRate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='constant_learning_rate', full_name='Caster.protos.LearningRate.constant_learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='exponential_decay_learning_rate', full_name='Caster.protos.LearningRate.exponential_decay_learning_rate', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='manual_step_learning_rate', full_name='Caster.protos.LearningRate.manual_step_learning_rate', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='learning_rate', full_name='Caster.protos.LearningRate.learning_rate',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=1107,
  serialized_end=1373,
)


_CONSTANTLEARNINGRATE = _descriptor.Descriptor(
  name='ConstantLearningRate',
  full_name='Caster.protos.ConstantLearningRate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='Caster.protos.ConstantLearningRate.learning_rate', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.002,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1375,
  serialized_end=1427,
)


_EXPONENTIALDECAYLEARNINGRATE = _descriptor.Descriptor(
  name='ExponentialDecayLearningRate',
  full_name='Caster.protos.ExponentialDecayLearningRate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='initial_learning_rate', full_name='Caster.protos.ExponentialDecayLearningRate.initial_learning_rate', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.002,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='decay_steps', full_name='Caster.protos.ExponentialDecayLearningRate.decay_steps', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=4000000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='decay_factor', full_name='Caster.protos.ExponentialDecayLearningRate.decay_factor', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.95,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='staircase', full_name='Caster.protos.ExponentialDecayLearningRate.staircase', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1430,
  serialized_end=1581,
)


_MANUALSTEPLEARNINGRATE_LEARNINGRATESCHEDULE = _descriptor.Descriptor(
  name='LearningRateSchedule',
  full_name='Caster.protos.ManualStepLearningRate.LearningRateSchedule',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='step', full_name='Caster.protos.ManualStepLearningRate.LearningRateSchedule.step', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='Caster.protos.ManualStepLearningRate.LearningRateSchedule.learning_rate', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.002,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1726,
  serialized_end=1792,
)

_MANUALSTEPLEARNINGRATE = _descriptor.Descriptor(
  name='ManualStepLearningRate',
  full_name='Caster.protos.ManualStepLearningRate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='initial_learning_rate', full_name='Caster.protos.ManualStepLearningRate.initial_learning_rate', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.002,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='schedule', full_name='Caster.protos.ManualStepLearningRate.schedule', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_MANUALSTEPLEARNINGRATE_LEARNINGRATESCHEDULE, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1584,
  serialized_end=1792,
)

_OPTIMIZER.fields_by_name['gradient_descent_optimizer'].message_type = _GRADIENTDESCENTOPTIMIZER
_OPTIMIZER.fields_by_name['rms_prop_optimizer'].message_type = _RMSPROPOPTIMIZER
_OPTIMIZER.fields_by_name['momentum_optimizer'].message_type = _MOMENTUMOPTIMIZER
_OPTIMIZER.fields_by_name['adam_optimizer'].message_type = _ADAMOPTIMIZER
_OPTIMIZER.fields_by_name['nadam_optimizer'].message_type = _NADAMOPTIMIZER
_OPTIMIZER.fields_by_name['adadelta_optimizer'].message_type = _ADADELTAOPTIMIZER
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['gradient_descent_optimizer'])
_OPTIMIZER.fields_by_name['gradient_descent_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['rms_prop_optimizer'])
_OPTIMIZER.fields_by_name['rms_prop_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['momentum_optimizer'])
_OPTIMIZER.fields_by_name['momentum_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['adam_optimizer'])
_OPTIMIZER.fields_by_name['adam_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['nadam_optimizer'])
_OPTIMIZER.fields_by_name['nadam_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['adadelta_optimizer'])
_OPTIMIZER.fields_by_name['adadelta_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_GRADIENTDESCENTOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_RMSPROPOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_MOMENTUMOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_ADAMOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_NADAMOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_ADADELTAOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_LEARNINGRATE.fields_by_name['constant_learning_rate'].message_type = _CONSTANTLEARNINGRATE
_LEARNINGRATE.fields_by_name['exponential_decay_learning_rate'].message_type = _EXPONENTIALDECAYLEARNINGRATE
_LEARNINGRATE.fields_by_name['manual_step_learning_rate'].message_type = _MANUALSTEPLEARNINGRATE
_LEARNINGRATE.oneofs_by_name['learning_rate'].fields.append(
  _LEARNINGRATE.fields_by_name['constant_learning_rate'])
_LEARNINGRATE.fields_by_name['constant_learning_rate'].containing_oneof = _LEARNINGRATE.oneofs_by_name['learning_rate']
_LEARNINGRATE.oneofs_by_name['learning_rate'].fields.append(
  _LEARNINGRATE.fields_by_name['exponential_decay_learning_rate'])
_LEARNINGRATE.fields_by_name['exponential_decay_learning_rate'].containing_oneof = _LEARNINGRATE.oneofs_by_name['learning_rate']
_LEARNINGRATE.oneofs_by_name['learning_rate'].fields.append(
  _LEARNINGRATE.fields_by_name['manual_step_learning_rate'])
_LEARNINGRATE.fields_by_name['manual_step_learning_rate'].containing_oneof = _LEARNINGRATE.oneofs_by_name['learning_rate']
_MANUALSTEPLEARNINGRATE_LEARNINGRATESCHEDULE.containing_type = _MANUALSTEPLEARNINGRATE
_MANUALSTEPLEARNINGRATE.fields_by_name['schedule'].message_type = _MANUALSTEPLEARNINGRATE_LEARNINGRATESCHEDULE
DESCRIPTOR.message_types_by_name['Optimizer'] = _OPTIMIZER
DESCRIPTOR.message_types_by_name['GradientDescentOptimizer'] = _GRADIENTDESCENTOPTIMIZER
DESCRIPTOR.message_types_by_name['RMSPropOptimizer'] = _RMSPROPOPTIMIZER
DESCRIPTOR.message_types_by_name['MomentumOptimizer'] = _MOMENTUMOPTIMIZER
DESCRIPTOR.message_types_by_name['AdamOptimizer'] = _ADAMOPTIMIZER
DESCRIPTOR.message_types_by_name['NadamOptimizer'] = _NADAMOPTIMIZER
DESCRIPTOR.message_types_by_name['AdadeltaOptimizer'] = _ADADELTAOPTIMIZER
DESCRIPTOR.message_types_by_name['LearningRate'] = _LEARNINGRATE
DESCRIPTOR.message_types_by_name['ConstantLearningRate'] = _CONSTANTLEARNINGRATE
DESCRIPTOR.message_types_by_name['ExponentialDecayLearningRate'] = _EXPONENTIALDECAYLEARNINGRATE
DESCRIPTOR.message_types_by_name['ManualStepLearningRate'] = _MANUALSTEPLEARNINGRATE

Optimizer = _reflection.GeneratedProtocolMessageType('Optimizer', (_message.Message,), dict(
  DESCRIPTOR = _OPTIMIZER,
  __module__ = 'Caster.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.Optimizer)
  ))
_sym_db.RegisterMessage(Optimizer)

GradientDescentOptimizer = _reflection.GeneratedProtocolMessageType('GradientDescentOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _GRADIENTDESCENTOPTIMIZER,
  __module__ = 'Caster.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.GradientDescentOptimizer)
  ))
_sym_db.RegisterMessage(GradientDescentOptimizer)

RMSPropOptimizer = _reflection.GeneratedProtocolMessageType('RMSPropOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _RMSPROPOPTIMIZER,
  __module__ = 'Caster.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.RMSPropOptimizer)
  ))
_sym_db.RegisterMessage(RMSPropOptimizer)

MomentumOptimizer = _reflection.GeneratedProtocolMessageType('MomentumOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _MOMENTUMOPTIMIZER,
  __module__ = 'Caster.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.MomentumOptimizer)
  ))
_sym_db.RegisterMessage(MomentumOptimizer)

AdamOptimizer = _reflection.GeneratedProtocolMessageType('AdamOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _ADAMOPTIMIZER,
  __module__ = 'Caster.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.AdamOptimizer)
  ))
_sym_db.RegisterMessage(AdamOptimizer)

NadamOptimizer = _reflection.GeneratedProtocolMessageType('NadamOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _NADAMOPTIMIZER,
  __module__ = 'Caster.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.NadamOptimizer)
  ))
_sym_db.RegisterMessage(NadamOptimizer)

AdadeltaOptimizer = _reflection.GeneratedProtocolMessageType('AdadeltaOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _ADADELTAOPTIMIZER,
  __module__ = 'Caster.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.AdadeltaOptimizer)
  ))
_sym_db.RegisterMessage(AdadeltaOptimizer)

LearningRate = _reflection.GeneratedProtocolMessageType('LearningRate', (_message.Message,), dict(
  DESCRIPTOR = _LEARNINGRATE,
  __module__ = 'Caster.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.LearningRate)
  ))
_sym_db.RegisterMessage(LearningRate)

ConstantLearningRate = _reflection.GeneratedProtocolMessageType('ConstantLearningRate', (_message.Message,), dict(
  DESCRIPTOR = _CONSTANTLEARNINGRATE,
  __module__ = 'Caster.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.ConstantLearningRate)
  ))
_sym_db.RegisterMessage(ConstantLearningRate)

ExponentialDecayLearningRate = _reflection.GeneratedProtocolMessageType('ExponentialDecayLearningRate', (_message.Message,), dict(
  DESCRIPTOR = _EXPONENTIALDECAYLEARNINGRATE,
  __module__ = 'Caster.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.ExponentialDecayLearningRate)
  ))
_sym_db.RegisterMessage(ExponentialDecayLearningRate)

ManualStepLearningRate = _reflection.GeneratedProtocolMessageType('ManualStepLearningRate', (_message.Message,), dict(

  LearningRateSchedule = _reflection.GeneratedProtocolMessageType('LearningRateSchedule', (_message.Message,), dict(
    DESCRIPTOR = _MANUALSTEPLEARNINGRATE_LEARNINGRATESCHEDULE,
    __module__ = 'Caster.protos.optimizer_pb2'
    # @@protoc_insertion_point(class_scope:Caster.protos.ManualStepLearningRate.LearningRateSchedule)
    ))
  ,
  DESCRIPTOR = _MANUALSTEPLEARNINGRATE,
  __module__ = 'Caster.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.ManualStepLearningRate)
  ))
_sym_db.RegisterMessage(ManualStepLearningRate)
_sym_db.RegisterMessage(ManualStepLearningRate.LearningRateSchedule)


# @@protoc_insertion_point(module_scope)
