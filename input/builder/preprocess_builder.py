# -*- coding: utf-8 -*-

from Caster.protos import data_preprocess_pb2

from Caster.input.preprocess import data_preprocess as preprocessor
from Caster.model.builder import label_map_builder


def build(config):
    if not isinstance(config, data_preprocess_pb2.PreprocessingStep):
        raise ValueError('config not of type data_preprocess_pb2.PreprocessingStep')
    
    step_type = config.WhichOneof('preprocessing_step')

    if step_type == 'resize_image':
        preprocess_config = config.resize_image
        return (preprocessor.resize_image,
                {'target_size': [preprocess_config.target_height, preprocess_config.target_width],
                 'method': preprocess_config.resize_method})

    if step_type == 'resize_padding_image':
        preprocess_config = config.resize_padding_image
        return (preprocessor.resize_padding_image,
                {'max_height': preprocess_config.max_height,
                 'max_width': preprocess_config.max_width,
                 'method': preprocess_config.resize_method})

    if step_type == 'resize_image_random_method':
        preprocess_config = config.resize_image_random_method
        return (preprocessor.resize_image_random_method,
                {'target_size': [preprocess_config.target_height, preprocess_config.target_width]})

    if step_type == 'normalize_image':
        preprocess_config = config.normalize_image
        return (preprocessor.normalize_image,
                {'original_minval': preprocess_config.original_minval,
                 'original_maxval': preprocess_config.original_maxval,
                 'target_minval': preprocess_config.target_minval,
                 'target_maxval': preprocess_config.target_maxval})

    if step_type == 'random_pixel_value_scale':
        preprocess_config = config.random_pixel_value_scale
        return (preprocessor.random_pixel_value_scale,
                {'minval': preprocess_config.minval,
                 'maxval': preprocess_config.maxval,
                 'seed': preprocess_config.seed})

    if step_type == 'rgb_to_gray':
        preprocess_config = config.rgb_to_gray
        return (preprocessor.rgb_to_gray,
                {'three_channel': preprocess_config.three_channels})

    if step_type == 'random_rgb_to_gray':
        preprocess_config = config.random_rgb_to_gray
        return (preprocessor.random_rgb_to_gray,
                {'probability': preprocess_config.probability})

    if step_type == 'random_adjust_brightness':
        preprocess_config = config.random_adjust_brightness
        return (preprocessor.random_adjust_brightness,
                {'max_delta': preprocess_config.max_delta})

    if step_type == 'random_adjust_contrast':
        preprocess_config = config.random_adjust_contrast
        return (preprocessor.random_adjust_contrast,
                {'min_delta': preprocess_config.min_delta,
                 'max_delta': preprocess_config.max_delta})

    if step_type == 'random_adjust_hue':
        preprocess_config = config.random_adjust_hue
        return (preprocessor.random_adjust_hue,
                {'max_delta': preprocess_config.max_delta})

    if step_type == 'random_adjust_saturation':
        preprocess_config = config.random_adjust_saturation
        return (preprocessor.random_adjust_saturation,
                {'min_delta': preprocess_config.min_delta,
                 'max_delta': preprocess_config.max_delta})

    if step_type == 'random_distort_color':
        preprocess_config = config.random_distort_color
        return (preprocessor.random_distort_color,
                {'color_ordering': preprocess_config.color_ordering})

    if step_type == 'image_to_float':
        preprocess_config = config.image_to_float
        return (preprocessor.image_to_float, {})

    if step_type == 'subtract_channel_mean':
        preprocess_config = config.subtract_channel_mean
        return (preprocessor.subtract_channel_mean, 
                {'means': preprocess_config.means})

    if step_type == 'string_filtering':
        preprocess_config = config.string_filtering
        include_charset_list = label_map_builder._build_character_set(preprocess_config.include_charset)
        include_charset = ''.join(include_charset_list)
        return (preprocessor.string_filtering, 
                {'lower_case': preprocess_config.lower_case,
                 'include_charset': include_charset})
