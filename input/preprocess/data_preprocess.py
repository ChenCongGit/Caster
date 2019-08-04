import sys
import string
import numpy as np

import tensorflow as tf

from Caster.c_ops import ops


def get_func_arg_map():
    """
    预处理操作名和操作参数列表的映射，值操作参数列表不为None的表示进行操作所必需的参数名
    """
    prep_func_arg_map = {
        resize_image: 'image',
        resize_padding_image: 'image',
        resize_image_random_method: 'image',
        normalize_image: 'image',
        random_pixel_value_scale: 'image',
        rgb_to_gray: 'image',
        random_rgb_to_gray: 'image',
        random_adjust_brightness: 'image',
        random_adjust_contrast: 'image',
        random_adjust_hue: 'image',
        random_adjust_saturation: 'image',
        random_distort_color: 'image',
        image_to_float: 'image',
        subtract_channel_mean: 'image',
        string_filtering: 'groundtruth_text'
    }
    return prep_func_arg_map


def preprocess(tensor_dict, preprocess_options, func_arg_map=None):
    """
    对于图像和标签文本预处理

    Args:
        tensor_dict: 一个包括图像和标签张量的字典
                     图像需要是三维张量，以便于图像预处理（height, width, channel）
        preprocess_options: 一个预处理选项列表，列表的每一项是一个元组，元组第一项为预处理操作名称，元组第二项为操作所需参数
        func_arg_map: 一个操作方法的字典，键为各种预处理函数名，值为所需参数元组
    Returns:
        tensor_dict: 一个经过预处理的包括图像和标签张量的字典
    Raise:
        ValueError: (a) If image or groundtruth_text not in 
                        tensor_dict
                    (b) If the functions passed to Preprocess
                        are not in func_arg_map.
                    (c) If image in tensor_dict is not rank 3
    """
    if 'image' not in tensor_dict.keys() or 'groundtruth_text' not in tensor_dict.keys():
        raise ValueError('tensor_dict must include image and groundtruth_text')
    if func_arg_map is None:
        func_arg_map = get_func_arg_map()
    if len(tensor_dict['image'].shape) != 3:
        raise ValueError('image shape must be rank 3')
    
    # input preprocess based on preprocess_option
    for option in preprocess_options:
        func, params = option
        if func not in func_arg_map:
            raise ValueError('The function %s does not exist in func_arg_map' %(func.__name__))
        
        arg_name = func_arg_map[func] 
        args = tensor_dict[arg_name]
        results = func(args, **params)
        # if not isinstance(results, (list, tuple)):
        #     results = (results,)
        if arg_name is not None:
            tensor_dict[arg_name] = results
        
    return tensor_dict
        

def resize_image(image,
                 target_size,
                 method = tf.image.ResizeMethod.BILINEAR,
                 align_corners = False):
    """
    对于所以有图像使用固定方法缩放图像
    """
    with tf.name_scope('ResizeImage', values=[image]):
        resized_image = tf.image.resize_images(image, target_size, method, align_corners)
    return resized_image

def resize_padding_image(image,
                        max_height=128,
                        max_width=1024,
                        method = tf.image.ResizeMethod.BILINEAR,
                        align_corners = False):
    """
    将图像高度缩放到32,并且宽高比保持不变，将宽度padding到所有图像的最大宽度
    """
    with tf.name_scope('ResizePaddingImage', values=[image]):
        resized_image = tf.image.resize_image_with_crop_or_pad(image, max_height,max_width)
    return resized_image

def resize_image_random_method(image, target_size):
    """
    对于不同的图像使用随机方法缩放图像
    """
    with tf.name_scope('ResizeRandomMethod', values=[image]):
        method = np.random.randint(4, size=1)
        resized_image = tf.image.resize_images(image, target_size, method)
    return resized_image

def normalize_image(image, original_minval, original_maxval, target_minval,
                    target_maxval):
    """Normalizes pixel values in the image.

    Moves the pixel values from the current [original_minval, original_maxval]
    range to a the [target_minval, target_maxval] range.

    Args:
        image: rank 3 float32 tensor containing 1
            image -> [height, width, channels].
        original_minval: current image minimum value.
        original_maxval: current image maximum value.
        target_minval: target image minimum value.
        target_maxval: target image maximum value.

    Returns:
        image: image which is the same shape as input image.
    """
    with tf.name_scope('NormalizeImage', values=[image]):
        original_minval = float(original_minval)
        original_maxval = float(original_maxval)
        target_minval = float(target_minval)
        target_maxval = float(target_maxval)
        image = tf.to_float(image)
        image = tf.subtract(image, original_minval)
        image = tf.multiply(image, (target_maxval - target_minval) /
                            (original_maxval - original_minval))
        image = tf.add(image, target_minval)
        return image

def random_pixel_value_scale(image, minval=0.9, maxval=1.1, seed=None):
    """
    随机改变图像的每一个像素值（只进行轻微改变）
    输入的图像张量是tf.float32类型的，且每一个像素值都被归一化到0-1的范围
    """
    with tf.name_scope('RandomPixelValueScale', values=[image]):
        color_coef = tf.random_uniform(tf.shape(image),
                                    minval=minval,
                                    maxval=maxval,
                                    dtype=tf.float32,
                                    seed=seed)
        image = tf.multiply(image,color_coef)
        image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def rgb_to_gray(image, three_channels=False):
    """
    将三通道的RGB图像转化为单通道的灰度图像
    """
    gray_image = tf.image.rgb_to_grayscale(image)
    if three_channels:
        gray_image = tf.tile(gray_image, [1,1,3])
    return gray_image
        
def random_rgb_to_gray(image, probability=0.1, seed=None):
    """
    随机概率将RGB图像转化为单通道的灰度图像
    """
    with tf.name_scope('RandomRGBtoGray', values=[image]):
        # 生成一个0-1的随机值标量，通过与probability比较，进行随机操作
        random_probab = tf.random_uniform([],minval=0.0,maxval=1.0,dtype=tf.float32,seed=seed)
        image = tf.cond(tf.greater(random_probab,probability),
                lambda: image, lambda: rgb_to_gray(image, True))
    return image

def random_adjust_brightness(image, max_delta=0.2):
    """
    随机调整亮度，max_delta为亮度调整的程度
    """
    with tf.name_scope('RandomAdjustBrightness', values=[image]):
        image = tf.image.random_brightness(image, max_delta)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image

def random_adjust_contrast(image, min_delta=0.8, max_delta=1.25):
    """
    随即调整对比度，在min_delta和max_delta之间取一个值，使图像对比度与之相乘
    """
    with tf.name_scope('RandomAdjustcontrast', values=[image]):
        image = tf.image.random_contrast(image, min_delta, max_delta)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image

def random_adjust_hue(image, max_delta=0.02):
    """
    随即调整色度
    """
    with tf.name_scope('RandomAdjusthue', values=[image]):
        image = tf.image.random_hue(image, max_delta)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image

def random_adjust_saturation(image, min_delta=0.8, max_delta=1.25):
    """
    随机调整饱和度
    """
    with tf.name_scope('RandomAdjustsaturation', values=[image]):
        image = tf.image.random_saturation(image, min_delta, max_delta)
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image

def random_distort_color(image, color_ordering=0):
    """
    组合随机亮度，对比度，饱和度，色度的调整，color_ordering为不同的组合顺序
    color_ordering的值只能为{0,1}
    """
    with tf.name_scope('RandomDistortColor', value=[image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        else:
            raise ValueError('color_ordering must be in {0, 1}')
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image

def image_to_float(image):
    """
    tf.uint8类型图像转化为tf.float32类型
    """
    with tf.name_scope('ImageToFloat', values=[image]):
        image = tf.to_float(image)
    return image

def subtract_channel_mean(image, means=None):
    """
    对每一幅图像的每一个通道去均值，means是一个三元组，表示整个数据集的三个通道平均值
    """
    with tf.name_scope('SubstractChannelMean', values=[image]):
        if len(image.shape)!=3:
            raise ValueError('Input must be of size [height, width, channels]')
        if len(means)!=image.shape[-1]:
            raise ValueError('len(means) must match the number of channels')
    return image - [[means]]

def string_filtering(text, lower_case=False, include_charset=""):
    return ops.string_filtering([text], lower_case=lower_case, include_charset=include_charset)[0]
