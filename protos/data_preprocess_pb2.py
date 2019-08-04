# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Caster/protos/data_preprocess.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import Caster.protos.label_map_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='Caster/protos/data_preprocess.proto',
  package='Caster.protos',
  serialized_pb=_b('\n#Caster/protos/data_preprocess.proto\x12\rCaster.protos\x1a\x1d\x43\x61ster/protos/label_map.proto\"\xf5\x07\n\x11PreprocessingStep\x12\x32\n\x0cresize_image\x18\x01 \x01(\x0b\x32\x1a.Caster.protos.ResizeImageH\x00\x12\x41\n\x14resize_padding_image\x18\x02 \x01(\x0b\x32!.Caster.protos.ResizePaddingImageH\x00\x12L\n\x1aresize_image_random_method\x18\x03 \x01(\x0b\x32&.Caster.protos.ResizeImageRandomMethodH\x00\x12\x38\n\x0fnormalize_image\x18\x04 \x01(\x0b\x32\x1d.Caster.protos.NormalizeImageH\x00\x12H\n\x18random_pixel_value_scale\x18\x05 \x01(\x0b\x32$.Caster.protos.RandomPixelValueScaleH\x00\x12/\n\x0brgb_to_gray\x18\x06 \x01(\x0b\x32\x18.Caster.protos.RgbToGrayH\x00\x12<\n\x12random_rgb_to_gray\x18\x07 \x01(\x0b\x32\x1e.Caster.protos.RandomRgbToGrayH\x00\x12H\n\x18random_adjust_brightness\x18\x08 \x01(\x0b\x32$.Caster.protos.RandomAdjusBrightnessH\x00\x12\x45\n\x16random_adjust_contrast\x18\t \x01(\x0b\x32#.Caster.protos.RandomAdjustContrastH\x00\x12;\n\x11random_adjust_hue\x18\n \x01(\x0b\x32\x1e.Caster.protos.RandomAdjustHueH\x00\x12I\n\x18random_adjust_saturation\x18\x0b \x01(\x0b\x32%.Caster.protos.RandomAdjustSaturationH\x00\x12\x41\n\x14random_distort_color\x18\x0c \x01(\x0b\x32!.Caster.protos.RandomDistortColorH\x00\x12\x35\n\x0eimage_to_float\x18\r \x01(\x0b\x32\x1b.Caster.protos.ImageToFloatH\x00\x12\x43\n\x15subtract_channel_mean\x18\x0e \x01(\x0b\x32\".Caster.protos.SubtractChannelMeanH\x00\x12:\n\x10string_filtering\x18\x0f \x01(\x0b\x32\x1e.Caster.protos.StringFilteringH\x00\x42\x14\n\x12preprocessing_step\"\xcd\x01\n\x0bResizeImage\x12\x1a\n\rtarget_height\x18\x01 \x01(\x05:\x03\x35\x31\x32\x12\x19\n\x0ctarget_width\x18\x02 \x01(\x05:\x03\x35\x31\x32\x12\x42\n\rresize_method\x18\x03 \x01(\x0e\x32!.Caster.protos.ResizeImage.Method:\x08\x42ILINEAR\"C\n\x06Method\x12\x08\n\x04\x41REA\x10\x01\x12\x0b\n\x07\x42ICUBIC\x10\x02\x12\x0c\n\x08\x42ILINEAR\x10\x03\x12\x14\n\x10NEAREST_NEIGHBOR\x10\x04\"\xd5\x01\n\x12ResizePaddingImage\x12\x17\n\nmax_height\x18\x01 \x01(\x05:\x03\x35\x31\x32\x12\x16\n\tmax_width\x18\x02 \x01(\x05:\x03\x35\x31\x32\x12I\n\rresize_method\x18\x03 \x01(\x0e\x32(.Caster.protos.ResizePaddingImage.Method:\x08\x42ILINEAR\"C\n\x06Method\x12\x08\n\x04\x41REA\x10\x01\x12\x0b\n\x07\x42ICUBIC\x10\x02\x12\x0c\n\x08\x42ILINEAR\x10\x03\x12\x14\n\x10NEAREST_NEIGHBOR\x10\x04\"P\n\x17ResizeImageRandomMethod\x12\x1a\n\rtarget_height\x18\x01 \x01(\x05:\x03\x35\x31\x32\x12\x19\n\x0ctarget_width\x18\x02 \x01(\x05:\x03\x35\x31\x32\"v\n\x0eNormalizeImage\x12\x17\n\x0foriginal_minval\x18\x01 \x01(\x02\x12\x17\n\x0foriginal_maxval\x18\x02 \x01(\x02\x12\x18\n\rtarget_minval\x18\x03 \x01(\x02:\x01\x30\x12\x18\n\rtarget_maxval\x18\x04 \x01(\x02:\x01\x31\"U\n\x15RandomPixelValueScale\x12\x13\n\x06minval\x18\x01 \x01(\x02:\x03\x30.9\x12\x13\n\x06maxval\x18\x02 \x01(\x02:\x03\x31.1\x12\x12\n\x04seed\x18\x03 \x01(\x05:\x04\x31\x32\x33\x34\"*\n\tRgbToGray\x12\x1d\n\x0ethree_channels\x18\x01 \x01(\x08:\x05\x66\x61lse\"+\n\x0fRandomRgbToGray\x12\x18\n\x0bprobability\x18\x01 \x01(\x02:\x03\x30.1\"/\n\x15RandomAdjusBrightness\x12\x16\n\tmax_delta\x18\x01 \x01(\x02:\x03\x30.2\"G\n\x14RandomAdjustContrast\x12\x16\n\tmin_delta\x18\x01 \x01(\x02:\x03\x30.8\x12\x17\n\tmax_delta\x18\x02 \x01(\x02:\x04\x31.25\"*\n\x0fRandomAdjustHue\x12\x17\n\tmax_delta\x18\x01 \x01(\x02:\x04\x30.02\"I\n\x16RandomAdjustSaturation\x12\x16\n\tmin_delta\x18\x01 \x01(\x02:\x03\x30.8\x12\x17\n\tmax_delta\x18\x02 \x01(\x02:\x04\x31.25\",\n\x12RandomDistortColor\x12\x16\n\x0e\x63olor_ordering\x18\x01 \x01(\x05\"\x0e\n\x0cImageToFloat\"$\n\x13SubtractChannelMean\x12\r\n\x05means\x18\x01 \x03(\x02\"b\n\x0fStringFiltering\x12\x19\n\nlower_case\x18\x01 \x01(\x08:\x05\x66\x61lse\x12\x34\n\x0finclude_charset\x18\x02 \x01(\x0b\x32\x1b.Caster.protos.CharacterSet')
  ,
  dependencies=[Caster.protos.label_map_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_RESIZEIMAGE_METHOD = _descriptor.EnumDescriptor(
  name='Method',
  full_name='Caster.protos.ResizeImage.Method',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AREA', index=0, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BICUBIC', index=1, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BILINEAR', index=2, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NEAREST_NEIGHBOR', index=3, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1240,
  serialized_end=1307,
)
_sym_db.RegisterEnumDescriptor(_RESIZEIMAGE_METHOD)

_RESIZEPADDINGIMAGE_METHOD = _descriptor.EnumDescriptor(
  name='Method',
  full_name='Caster.protos.ResizePaddingImage.Method',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AREA', index=0, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BICUBIC', index=1, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BILINEAR', index=2, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NEAREST_NEIGHBOR', index=3, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1240,
  serialized_end=1307,
)
_sym_db.RegisterEnumDescriptor(_RESIZEPADDINGIMAGE_METHOD)


_PREPROCESSINGSTEP = _descriptor.Descriptor(
  name='PreprocessingStep',
  full_name='Caster.protos.PreprocessingStep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='resize_image', full_name='Caster.protos.PreprocessingStep.resize_image', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='resize_padding_image', full_name='Caster.protos.PreprocessingStep.resize_padding_image', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='resize_image_random_method', full_name='Caster.protos.PreprocessingStep.resize_image_random_method', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='normalize_image', full_name='Caster.protos.PreprocessingStep.normalize_image', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='random_pixel_value_scale', full_name='Caster.protos.PreprocessingStep.random_pixel_value_scale', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rgb_to_gray', full_name='Caster.protos.PreprocessingStep.rgb_to_gray', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='random_rgb_to_gray', full_name='Caster.protos.PreprocessingStep.random_rgb_to_gray', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='random_adjust_brightness', full_name='Caster.protos.PreprocessingStep.random_adjust_brightness', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='random_adjust_contrast', full_name='Caster.protos.PreprocessingStep.random_adjust_contrast', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='random_adjust_hue', full_name='Caster.protos.PreprocessingStep.random_adjust_hue', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='random_adjust_saturation', full_name='Caster.protos.PreprocessingStep.random_adjust_saturation', index=10,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='random_distort_color', full_name='Caster.protos.PreprocessingStep.random_distort_color', index=11,
      number=12, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_to_float', full_name='Caster.protos.PreprocessingStep.image_to_float', index=12,
      number=13, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='subtract_channel_mean', full_name='Caster.protos.PreprocessingStep.subtract_channel_mean', index=13,
      number=14, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='string_filtering', full_name='Caster.protos.PreprocessingStep.string_filtering', index=14,
      number=15, type=11, cpp_type=10, label=1,
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
      name='preprocessing_step', full_name='Caster.protos.PreprocessingStep.preprocessing_step',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=86,
  serialized_end=1099,
)


_RESIZEIMAGE = _descriptor.Descriptor(
  name='ResizeImage',
  full_name='Caster.protos.ResizeImage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='target_height', full_name='Caster.protos.ResizeImage.target_height', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=512,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='target_width', full_name='Caster.protos.ResizeImage.target_width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=512,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='resize_method', full_name='Caster.protos.ResizeImage.resize_method', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _RESIZEIMAGE_METHOD,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1102,
  serialized_end=1307,
)


_RESIZEPADDINGIMAGE = _descriptor.Descriptor(
  name='ResizePaddingImage',
  full_name='Caster.protos.ResizePaddingImage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='max_height', full_name='Caster.protos.ResizePaddingImage.max_height', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=512,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='max_width', full_name='Caster.protos.ResizePaddingImage.max_width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=512,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='resize_method', full_name='Caster.protos.ResizePaddingImage.resize_method', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _RESIZEPADDINGIMAGE_METHOD,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1310,
  serialized_end=1523,
)


_RESIZEIMAGERANDOMMETHOD = _descriptor.Descriptor(
  name='ResizeImageRandomMethod',
  full_name='Caster.protos.ResizeImageRandomMethod',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='target_height', full_name='Caster.protos.ResizeImageRandomMethod.target_height', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=512,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='target_width', full_name='Caster.protos.ResizeImageRandomMethod.target_width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=512,
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
  serialized_start=1525,
  serialized_end=1605,
)


_NORMALIZEIMAGE = _descriptor.Descriptor(
  name='NormalizeImage',
  full_name='Caster.protos.NormalizeImage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='original_minval', full_name='Caster.protos.NormalizeImage.original_minval', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='original_maxval', full_name='Caster.protos.NormalizeImage.original_maxval', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='target_minval', full_name='Caster.protos.NormalizeImage.target_minval', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='target_maxval', full_name='Caster.protos.NormalizeImage.target_maxval', index=3,
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
  serialized_start=1607,
  serialized_end=1725,
)


_RANDOMPIXELVALUESCALE = _descriptor.Descriptor(
  name='RandomPixelValueScale',
  full_name='Caster.protos.RandomPixelValueScale',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='minval', full_name='Caster.protos.RandomPixelValueScale.minval', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.9,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='maxval', full_name='Caster.protos.RandomPixelValueScale.maxval', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1.1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='seed', full_name='Caster.protos.RandomPixelValueScale.seed', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1234,
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
  serialized_start=1727,
  serialized_end=1812,
)


_RGBTOGRAY = _descriptor.Descriptor(
  name='RgbToGray',
  full_name='Caster.protos.RgbToGray',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='three_channels', full_name='Caster.protos.RgbToGray.three_channels', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
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
  serialized_start=1814,
  serialized_end=1856,
)


_RANDOMRGBTOGRAY = _descriptor.Descriptor(
  name='RandomRgbToGray',
  full_name='Caster.protos.RandomRgbToGray',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='probability', full_name='Caster.protos.RandomRgbToGray.probability', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.1,
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
  serialized_start=1858,
  serialized_end=1901,
)


_RANDOMADJUSBRIGHTNESS = _descriptor.Descriptor(
  name='RandomAdjusBrightness',
  full_name='Caster.protos.RandomAdjusBrightness',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='max_delta', full_name='Caster.protos.RandomAdjusBrightness.max_delta', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.2,
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
  serialized_start=1903,
  serialized_end=1950,
)


_RANDOMADJUSTCONTRAST = _descriptor.Descriptor(
  name='RandomAdjustContrast',
  full_name='Caster.protos.RandomAdjustContrast',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_delta', full_name='Caster.protos.RandomAdjustContrast.min_delta', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='max_delta', full_name='Caster.protos.RandomAdjustContrast.max_delta', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1.25,
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
  serialized_start=1952,
  serialized_end=2023,
)


_RANDOMADJUSTHUE = _descriptor.Descriptor(
  name='RandomAdjustHue',
  full_name='Caster.protos.RandomAdjustHue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='max_delta', full_name='Caster.protos.RandomAdjustHue.max_delta', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.02,
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
  serialized_start=2025,
  serialized_end=2067,
)


_RANDOMADJUSTSATURATION = _descriptor.Descriptor(
  name='RandomAdjustSaturation',
  full_name='Caster.protos.RandomAdjustSaturation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_delta', full_name='Caster.protos.RandomAdjustSaturation.min_delta', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='max_delta', full_name='Caster.protos.RandomAdjustSaturation.max_delta', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=1.25,
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
  serialized_start=2069,
  serialized_end=2142,
)


_RANDOMDISTORTCOLOR = _descriptor.Descriptor(
  name='RandomDistortColor',
  full_name='Caster.protos.RandomDistortColor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='color_ordering', full_name='Caster.protos.RandomDistortColor.color_ordering', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=2144,
  serialized_end=2188,
)


_IMAGETOFLOAT = _descriptor.Descriptor(
  name='ImageToFloat',
  full_name='Caster.protos.ImageToFloat',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
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
  serialized_start=2190,
  serialized_end=2204,
)


_SUBTRACTCHANNELMEAN = _descriptor.Descriptor(
  name='SubtractChannelMean',
  full_name='Caster.protos.SubtractChannelMean',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='means', full_name='Caster.protos.SubtractChannelMean.means', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=2206,
  serialized_end=2242,
)


_STRINGFILTERING = _descriptor.Descriptor(
  name='StringFiltering',
  full_name='Caster.protos.StringFiltering',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lower_case', full_name='Caster.protos.StringFiltering.lower_case', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='include_charset', full_name='Caster.protos.StringFiltering.include_charset', index=1,
      number=2, type=11, cpp_type=10, label=1,
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
  serialized_start=2244,
  serialized_end=2342,
)

_PREPROCESSINGSTEP.fields_by_name['resize_image'].message_type = _RESIZEIMAGE
_PREPROCESSINGSTEP.fields_by_name['resize_padding_image'].message_type = _RESIZEPADDINGIMAGE
_PREPROCESSINGSTEP.fields_by_name['resize_image_random_method'].message_type = _RESIZEIMAGERANDOMMETHOD
_PREPROCESSINGSTEP.fields_by_name['normalize_image'].message_type = _NORMALIZEIMAGE
_PREPROCESSINGSTEP.fields_by_name['random_pixel_value_scale'].message_type = _RANDOMPIXELVALUESCALE
_PREPROCESSINGSTEP.fields_by_name['rgb_to_gray'].message_type = _RGBTOGRAY
_PREPROCESSINGSTEP.fields_by_name['random_rgb_to_gray'].message_type = _RANDOMRGBTOGRAY
_PREPROCESSINGSTEP.fields_by_name['random_adjust_brightness'].message_type = _RANDOMADJUSBRIGHTNESS
_PREPROCESSINGSTEP.fields_by_name['random_adjust_contrast'].message_type = _RANDOMADJUSTCONTRAST
_PREPROCESSINGSTEP.fields_by_name['random_adjust_hue'].message_type = _RANDOMADJUSTHUE
_PREPROCESSINGSTEP.fields_by_name['random_adjust_saturation'].message_type = _RANDOMADJUSTSATURATION
_PREPROCESSINGSTEP.fields_by_name['random_distort_color'].message_type = _RANDOMDISTORTCOLOR
_PREPROCESSINGSTEP.fields_by_name['image_to_float'].message_type = _IMAGETOFLOAT
_PREPROCESSINGSTEP.fields_by_name['subtract_channel_mean'].message_type = _SUBTRACTCHANNELMEAN
_PREPROCESSINGSTEP.fields_by_name['string_filtering'].message_type = _STRINGFILTERING
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['resize_image'])
_PREPROCESSINGSTEP.fields_by_name['resize_image'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['resize_padding_image'])
_PREPROCESSINGSTEP.fields_by_name['resize_padding_image'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['resize_image_random_method'])
_PREPROCESSINGSTEP.fields_by_name['resize_image_random_method'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['normalize_image'])
_PREPROCESSINGSTEP.fields_by_name['normalize_image'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['random_pixel_value_scale'])
_PREPROCESSINGSTEP.fields_by_name['random_pixel_value_scale'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['rgb_to_gray'])
_PREPROCESSINGSTEP.fields_by_name['rgb_to_gray'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['random_rgb_to_gray'])
_PREPROCESSINGSTEP.fields_by_name['random_rgb_to_gray'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['random_adjust_brightness'])
_PREPROCESSINGSTEP.fields_by_name['random_adjust_brightness'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['random_adjust_contrast'])
_PREPROCESSINGSTEP.fields_by_name['random_adjust_contrast'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['random_adjust_hue'])
_PREPROCESSINGSTEP.fields_by_name['random_adjust_hue'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['random_adjust_saturation'])
_PREPROCESSINGSTEP.fields_by_name['random_adjust_saturation'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['random_distort_color'])
_PREPROCESSINGSTEP.fields_by_name['random_distort_color'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['image_to_float'])
_PREPROCESSINGSTEP.fields_by_name['image_to_float'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['subtract_channel_mean'])
_PREPROCESSINGSTEP.fields_by_name['subtract_channel_mean'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['string_filtering'])
_PREPROCESSINGSTEP.fields_by_name['string_filtering'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_RESIZEIMAGE.fields_by_name['resize_method'].enum_type = _RESIZEIMAGE_METHOD
_RESIZEIMAGE_METHOD.containing_type = _RESIZEIMAGE
_RESIZEPADDINGIMAGE.fields_by_name['resize_method'].enum_type = _RESIZEPADDINGIMAGE_METHOD
_RESIZEPADDINGIMAGE_METHOD.containing_type = _RESIZEPADDINGIMAGE
_STRINGFILTERING.fields_by_name['include_charset'].message_type = Caster.protos.label_map_pb2._CHARACTERSET
DESCRIPTOR.message_types_by_name['PreprocessingStep'] = _PREPROCESSINGSTEP
DESCRIPTOR.message_types_by_name['ResizeImage'] = _RESIZEIMAGE
DESCRIPTOR.message_types_by_name['ResizePaddingImage'] = _RESIZEPADDINGIMAGE
DESCRIPTOR.message_types_by_name['ResizeImageRandomMethod'] = _RESIZEIMAGERANDOMMETHOD
DESCRIPTOR.message_types_by_name['NormalizeImage'] = _NORMALIZEIMAGE
DESCRIPTOR.message_types_by_name['RandomPixelValueScale'] = _RANDOMPIXELVALUESCALE
DESCRIPTOR.message_types_by_name['RgbToGray'] = _RGBTOGRAY
DESCRIPTOR.message_types_by_name['RandomRgbToGray'] = _RANDOMRGBTOGRAY
DESCRIPTOR.message_types_by_name['RandomAdjusBrightness'] = _RANDOMADJUSBRIGHTNESS
DESCRIPTOR.message_types_by_name['RandomAdjustContrast'] = _RANDOMADJUSTCONTRAST
DESCRIPTOR.message_types_by_name['RandomAdjustHue'] = _RANDOMADJUSTHUE
DESCRIPTOR.message_types_by_name['RandomAdjustSaturation'] = _RANDOMADJUSTSATURATION
DESCRIPTOR.message_types_by_name['RandomDistortColor'] = _RANDOMDISTORTCOLOR
DESCRIPTOR.message_types_by_name['ImageToFloat'] = _IMAGETOFLOAT
DESCRIPTOR.message_types_by_name['SubtractChannelMean'] = _SUBTRACTCHANNELMEAN
DESCRIPTOR.message_types_by_name['StringFiltering'] = _STRINGFILTERING

PreprocessingStep = _reflection.GeneratedProtocolMessageType('PreprocessingStep', (_message.Message,), dict(
  DESCRIPTOR = _PREPROCESSINGSTEP,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.PreprocessingStep)
  ))
_sym_db.RegisterMessage(PreprocessingStep)

ResizeImage = _reflection.GeneratedProtocolMessageType('ResizeImage', (_message.Message,), dict(
  DESCRIPTOR = _RESIZEIMAGE,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.ResizeImage)
  ))
_sym_db.RegisterMessage(ResizeImage)

ResizePaddingImage = _reflection.GeneratedProtocolMessageType('ResizePaddingImage', (_message.Message,), dict(
  DESCRIPTOR = _RESIZEPADDINGIMAGE,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.ResizePaddingImage)
  ))
_sym_db.RegisterMessage(ResizePaddingImage)

ResizeImageRandomMethod = _reflection.GeneratedProtocolMessageType('ResizeImageRandomMethod', (_message.Message,), dict(
  DESCRIPTOR = _RESIZEIMAGERANDOMMETHOD,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.ResizeImageRandomMethod)
  ))
_sym_db.RegisterMessage(ResizeImageRandomMethod)

NormalizeImage = _reflection.GeneratedProtocolMessageType('NormalizeImage', (_message.Message,), dict(
  DESCRIPTOR = _NORMALIZEIMAGE,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.NormalizeImage)
  ))
_sym_db.RegisterMessage(NormalizeImage)

RandomPixelValueScale = _reflection.GeneratedProtocolMessageType('RandomPixelValueScale', (_message.Message,), dict(
  DESCRIPTOR = _RANDOMPIXELVALUESCALE,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.RandomPixelValueScale)
  ))
_sym_db.RegisterMessage(RandomPixelValueScale)

RgbToGray = _reflection.GeneratedProtocolMessageType('RgbToGray', (_message.Message,), dict(
  DESCRIPTOR = _RGBTOGRAY,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.RgbToGray)
  ))
_sym_db.RegisterMessage(RgbToGray)

RandomRgbToGray = _reflection.GeneratedProtocolMessageType('RandomRgbToGray', (_message.Message,), dict(
  DESCRIPTOR = _RANDOMRGBTOGRAY,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.RandomRgbToGray)
  ))
_sym_db.RegisterMessage(RandomRgbToGray)

RandomAdjusBrightness = _reflection.GeneratedProtocolMessageType('RandomAdjusBrightness', (_message.Message,), dict(
  DESCRIPTOR = _RANDOMADJUSBRIGHTNESS,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.RandomAdjusBrightness)
  ))
_sym_db.RegisterMessage(RandomAdjusBrightness)

RandomAdjustContrast = _reflection.GeneratedProtocolMessageType('RandomAdjustContrast', (_message.Message,), dict(
  DESCRIPTOR = _RANDOMADJUSTCONTRAST,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.RandomAdjustContrast)
  ))
_sym_db.RegisterMessage(RandomAdjustContrast)

RandomAdjustHue = _reflection.GeneratedProtocolMessageType('RandomAdjustHue', (_message.Message,), dict(
  DESCRIPTOR = _RANDOMADJUSTHUE,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.RandomAdjustHue)
  ))
_sym_db.RegisterMessage(RandomAdjustHue)

RandomAdjustSaturation = _reflection.GeneratedProtocolMessageType('RandomAdjustSaturation', (_message.Message,), dict(
  DESCRIPTOR = _RANDOMADJUSTSATURATION,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.RandomAdjustSaturation)
  ))
_sym_db.RegisterMessage(RandomAdjustSaturation)

RandomDistortColor = _reflection.GeneratedProtocolMessageType('RandomDistortColor', (_message.Message,), dict(
  DESCRIPTOR = _RANDOMDISTORTCOLOR,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.RandomDistortColor)
  ))
_sym_db.RegisterMessage(RandomDistortColor)

ImageToFloat = _reflection.GeneratedProtocolMessageType('ImageToFloat', (_message.Message,), dict(
  DESCRIPTOR = _IMAGETOFLOAT,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.ImageToFloat)
  ))
_sym_db.RegisterMessage(ImageToFloat)

SubtractChannelMean = _reflection.GeneratedProtocolMessageType('SubtractChannelMean', (_message.Message,), dict(
  DESCRIPTOR = _SUBTRACTCHANNELMEAN,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.SubtractChannelMean)
  ))
_sym_db.RegisterMessage(SubtractChannelMean)

StringFiltering = _reflection.GeneratedProtocolMessageType('StringFiltering', (_message.Message,), dict(
  DESCRIPTOR = _STRINGFILTERING,
  __module__ = 'Caster.protos.data_preprocess_pb2'
  # @@protoc_insertion_point(class_scope:Caster.protos.StringFiltering)
  ))
_sym_db.RegisterMessage(StringFiltering)


# @@protoc_insertion_point(module_scope)
