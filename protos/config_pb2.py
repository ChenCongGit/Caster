# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Caster/protos/config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import Caster.protos.input_reader_pb2
import Caster.protos.model_pb2
import Caster.protos.train_pb2
import Caster.protos.eval_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='Caster/protos/config.proto',
  package='Caster.protos',
  serialized_pb=_b('\n\x1a\x43\x61ster/protos/config.proto\x12\rCaster.protos\x1a Caster/protos/input_reader.proto\x1a\x19\x43\x61ster/protos/model.proto\x1a\x19\x43\x61ster/protos/train.proto\x1a\x18\x43\x61ster/protos/eval.proto\"\x87\x02\n\x0fTrainEvalConfig\x12#\n\x05model\x18\x01 \x01(\x0b\x32\x14.Caster.protos.Model\x12\x30\n\x0ctrain_config\x18\x02 \x01(\x0b\x32\x1a.Caster.protos.TrainConfig\x12\x36\n\x12train_input_reader\x18\x03 \x03(\x0b\x32\x1a.Caster.protos.InputReader\x12.\n\x0b\x65val_config\x18\x04 \x01(\x0b\x32\x19.Caster.protos.EvalConfig\x12\x35\n\x11\x65val_input_reader\x18\x05 \x01(\x0b\x32\x1a.Caster.protos.InputReader')
  ,
  dependencies=[Caster.protos.input_reader_pb2.DESCRIPTOR,Caster.protos.model_pb2.DESCRIPTOR,Caster.protos.train_pb2.DESCRIPTOR,Caster.protos.eval_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_TRAINEVALCONFIG = _descriptor.Descriptor(
  name='TrainEvalConfig',
  full_name='Caster.protos.TrainEvalConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='Caster.protos.TrainEvalConfig.model', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='train_config', full_name='Caster.protos.TrainEvalConfig.train_config', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='train_input_reader', full_name='Caster.protos.TrainEvalConfig.train_input_reader', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eval_config', full_name='Caster.protos.TrainEvalConfig.eval_config', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eval_input_reader', full_name='Caster.protos.TrainEvalConfig.eval_input_reader', index=4,
      number=5, type=11, cpp_type=10, label=1,
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
  serialized_start=160,
  serialized_end=423,
)

_TRAINEVALCONFIG.fields_by_name['model'].message_type = Caster.protos.model_pb2._MODEL
_TRAINEVALCONFIG.fields_by_name['train_config'].message_type = Caster.protos.train_pb2._TRAINCONFIG
_TRAINEVALCONFIG.fields_by_name['train_input_reader'].message_type = Caster.protos.input_reader_pb2._INPUTREADER
_TRAINEVALCONFIG.fields_by_name['eval_config'].message_type = Caster.protos.eval_pb2._EVALCONFIG
_TRAINEVALCONFIG.fields_by_name['eval_input_reader'].message_type = Caster.protos.input_reader_pb2._INPUTREADER
DESCRIPTOR.message_types_by_name['TrainEvalConfig'] = _TRAINEVALCONFIG

TrainEvalConfig = _reflection.GeneratedProtocolMessageType('TrainEvalConfig', (_message.Message,), dict(
  DESCRIPTOR = _TRAINEVALCONFIG,
  __module__ = 'Caster.protos.config_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.TrainEvalConfig)
  ))
_sym_db.RegisterMessage(TrainEvalConfig)


# @@protoc_insertion_point(module_scope)