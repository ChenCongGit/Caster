# -*- coding: utf-8 -*-

import sys
sys.path.append('../utils')
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import conv2d, max_pool2d, fully_connected
from tensorflow.contrib.framework import arg_scope

from Caster.utils import shape_utils

eps = 1e-6


class SpatialTransformer(object):
    """
    基于TPS薄板样条插值的空间变换网络STN
    STN基本原理: STN包括三个部分，定位网络，网格生成器和采样器，应用于TPS薄板样条插值，
    定位网络:首先定位网络回归得到原图中的K个控制点的坐标，这里定位网络使用了卷积池化全链接，并输出一个2K个
    节点的向量，定位网络首先将原始图像和矫正后图像坐标进行归一化，图像左上角为(0,0),右下角为(1,1)。
    
    网格生成器:根据定位网络找到的2K个控制点坐标计算得到图像中矫正后图像所有的点在原图中的映射位置，
    网格生成器提供了两个输入:被矫正图像控制点坐标C‘和矫正后图像像素点网格P，一个输出:被矫正图像像素点网格P’

    采样器: 在原图和矫正后图像之间对应坐标网格使用线性插值映射像素值，以保证梯度反向传播
    """
    def __init__(self,
                 localization_image_size=None,
                 output_image_size=None,
                 num_control_points=None,
                 init_bias_pattern=None,
                 activation=None,
                 margins=None,
                 summarize_activations=False):
        self._localization_image_size = localization_image_size
        self._output_image_size = output_image_size
        self._num_control_points = num_control_points
        self._init_bias_pattern = init_bias_pattern
        self._activation = activation
        self._margins = margins
        self._summarize_activations = summarize_activations

        self._output_control_points = self._build_output_control_points()
        self._output_grid = self._build_output_grid()
        self._delta_C_inv = self._build_delta_C_inv() # [k+3,k+3]
        self._init_bias = self._build_init_bias()

    def batch_transform(self,preprocessed_inputs):
        with tf.variable_scope('LocalizationNet',[preprocessed_inputs]):
            resized_image = tf.image.resize_images(preprocessed_inputs,self._localization_image_size)
            control_points_tensor = self._localize(resized_image)

        with tf.variable_scope('GridGenerator',[control_points_tensor]):
            sampling_grid = self._batch_generate_grid(control_points_tensor)

        with tf.variable_scope('Sampler',[sampling_grid, preprocessed_inputs]):
            rectified_images = self._batch_sample(preprocessed_inputs, sampling_grid)

        return {'rectified_images':rectified_images,'control_points':control_points_tensor}


    def _localize(self,resized_image):
        """
        常规的卷积池化全连接网络，获得k个控制点的坐标
        """
        batch_size = shape_utils.combined_static_and_dynamic_shape(resized_image)[0]
        k = self._num_control_points
        with arg_scope([conv2d],kernel_size=3,activation_fn=tf.nn.relu),\
             arg_scope([max_pool2d], kernel_size=2, stride=2),\
             arg_scope([fully_connected], activation_fn=tf.nn.relu):
            conv1 = conv2d(resized_image, 32, scope='conv1') # [B,h,w,32]
            pool1 = max_pool2d(conv1, scope='pool1')  # [B,h/2,w/2,32]
            conv2 = conv2d(pool1, 64, scope='conv2')  # [B,h/2,w/2,64]
            pool2 = max_pool2d(conv2, scope='pool2')  # [B,h/4,w/4,64]
            conv3 = conv2d(pool2, 128, scope='conv3') # [B,h/4,w/4,128]
            pool3 = max_pool2d(conv3, scope='pool3')  # [B,h/8,w/8,128]
            conv4 = conv2d(pool3, 256, scope='conv4') # [B,h/8,w/8,256]
            pool4 = max_pool2d(conv4, scope='pool4')  # [B,h/16,w/16,256]
            conv5 = conv2d(pool4, 256, scope='conv5') # [B,h/16,w/16,256]
            pool5 = max_pool2d(conv5, scope='pool5')  # [B,h/32,w/32,256]
            conv6 = conv2d(pool5, 256, scope='conv6') # [B,h/32,w/32,256]
            flatten = tf.reshape(conv6,[batch_size,-1])
            fc1 = fully_connected(flatten,512)
            fc2 = fully_connected(0.1*fc1,2*k,weights_initializer=tf.zeros_initializer(),biases_initializer=tf.constant_initializer(self._init_bias))

            if self._activation == 'sigmoid':
                output = tf.sigmoid(fc2) # [B, 2*k]
            elif self._activation == 'none':
                output = fc2
            else:
                raise ValueError('Unknown activation: {}'.format(self._activation))
            
            if self._summarize_activations:
                tf.summary.histogram('fc1', fc1)
                tf.summary.histogram('fc2', fc2)
                tf.summary.histogram('output', output)

        control_points = tf.reshape(output, [batch_size, k, 2]) # [B, k, 2]
        return control_points


    def _batch_generate_grid(self,control_points):
        """
        网格生成器
        TPS薄板样条插值原理:
        假设原始图像I中的点的坐标为x=[x_1, x_2]，变换后图像I_r中的点的坐标为y=[y_1, y_2]，则TPS插值变换公式为: 
        y=[y_1, y_2]=[c_x_1+a_x_1_T*x_1+w_x_1_T*s(x_1),c_x_2+a_x_2_T*x_2+w_x_2_T*s(x_2)]
         =[[c_x_1, a_x_1_1, a_x_1_2, w_x_1_T], [c_x_2, a_x_2_1, a_x_2_2, w_x_2_T]*[1, x_1, x_2, s(x)]_T
         =[[a_0, a_1, a_2, u],[b_0, b_1, b_2, v]]*[1, x_1, x_2, s(x)]_T
         =T*[1, x_1, x_2, s(x)]_T
        其中u,v为s(x)的系数向量，u,v的大小为1×K

        假设原始图像I中所有控制点坐标为C_p=[C_p__1, C_p__2, C_p__3,..., C_p__K]=[[C_p_1_1,C_p_2_1],...[C_p_1_K,C_p_2_K]]
        假设原始图像I中任意一点P=[P_1, P_2]，则
        s(P)=[phi(norm(P,C__1)),phi(norm(P,C__2)),...,phi(norm(P,C__K))]
        
        变换矩阵T共有2×(3+K)个参数，对应于2×(3+K)个方程，其中2×K个控制点方程，6个其他约束条件
        约束条件: C_p=c+a_T*C_p+w_T*s(C) # 2×K个控制点方程
                 w_T*1=0 # 2个方程(y包括y_1,y_2,1为向量)
                 w_T*C=0 # 4个方程(x包括x_1,x_2,y包括y_1,y_2)
        得到方程组:
        #######[[S, 1, C], [1_T, 0, 0], [C_T, 0, 0]]*[w, c, a]_T = [C_p, 0, 0]_T#######
        其中C_p=[[C_p_1_1,C_p_2_1],[C_p_1_2,C_p_2_2],...,[C_p_1_K,C_p_2_K]]
           C=[[C_1_1,C_2_1],[C_1_2,C_2_2],...,[C_1_K,C_2_K]]
           S=[[phi(norm(C__1,C__1),phi(norm(C__2,C__1),...,phi(norm(C__K,C__1)],
              [phi(norm(C__1,C__2),phi(norm(C__2,C__2),...,phi(norm(C__K,C__2)],
              [...],
              [phi(norm(C__1,C__K),phi(norm(C__2,C__K),...,phi(norm(C__K,C__K)]]
           1=[1,1,1,...,1]_T

        令 delta_C = [[S, 1, C], [1_T, 0, 0], [C_T, 0, 0]]
        则变换矩阵 T = [w, c, a]_T = delta_C_-1 × [C_p, 0, 0]_T

        网格生成器的作用就是根据矫正后图像控制点到原始图像控制点的变换矩阵T对矫正后图像的坐标网格进行变换，映射到原始图像中
        假设矫正后图像的坐标网格为G，控制点坐标为C，则S为:
        S(G)=[S(G_1),S(G_2),...,S(G_n)]
            =[[phi(norm(G_1,C__1)),phi(norm(G_1,C__2)),...,phi(norm(G_1,C__K))],
              [phi(norm(G_2,C__1)),phi(norm(G_2,C__2)),...,phi(norm(G_2,C__K))],
              [...],
              [phi(norm(G_n,C__1)),phi(norm(G_n,C__2)),...,phi(norm(G_n,C__K))]]
        G_p=[G_p_1,G_p_2,...,G_p_n]
           =[[S(G_1), 1, G_1]*[w, c, a]_T],
             [S(G_2), 1, G_2]*[w, c, a]_T],
             [...],
             [S(G_n), 1, G_n]*[w, c, a]_T]]
           =[[S(G_1), 1, G_1],[S(G_2), 1, G_2],[...],[S(G_n), 1, G_n]]*[w, c, a]_T
           =[S(G), 1, G]*[w, c, a]_T
        其中S(G)如上，大小为[n,k]，1大小为[n,1]，G大小为[n,2]

        Args
            input_ctrl_pts: float32 tensor of shape [batch_size, num_ctrl_pts, 2]
        Returns
            sampling_grid: float32 tensor of shape [num_sampling_pts, 2]
        """
        C = tf.constant(self._output_control_points,tf.float32) # [k,2]
        G = tf.constant(self._output_grid.reshape([-1,2]),dtype=tf.float32) # [n,2]
        n = G.shape[0] # n个采样点
        k = self._num_control_points

        batch_C_p = control_points # [batch_size, k, 2]
        batch_size = batch_C_p.shape[0]

        # 获得变换矩阵T
        delta_C_inv = tf.constant(self._delta_C_inv,dtype=tf.float32) # [k+3,k+3]
        batch_delta_C_inv = tf.tile(tf.expand_dims(delta_C_inv,0),[batch_size,1,1]) # [batch_size,k+3,k+3]
        batch_C_p_zero = tf.concat([batch_C_p,tf.zeros([batch_size,3,2])],axis=1) # [batch_size,k+3,2]
        batch_T = tf.matmul(batch_delta_C_inv,batch_C_p_zero) # [batch_size,k+3,2]


        # 基于变换矩阵T计算n个采样网格点在原图中的坐标
        G_tile = tf.tile(tf.expand_dims(G,1),[1,k,1]) # [n,k,2]
        C_tile = tf.tile(tf.expand_dims(C,0),[n,1,1]) # [n,k,2]
        S_G = tf.norm(G_tile-C_tile,axis=2,ord=2,keep_dims=False) # [n,k]
        S_G = tf.multiply(tf.square(S_G),tf.log(S_G+eps))
        G_p = tf.concat([S_G,tf.ones([n,1]),G],axis=1) # [n,k+3]
        batch_G_p = tf.tile(tf.expand_dims(G_p,0),[batch_size,1,1]) # [batch_size,n,k+3]
        batch_G_p = tf.matmul(batch_G_p,batch_T) # [batch_size,n,2]
        return batch_G_p


    def _batch_sample(self,images,batch_sample_grid):
        """
        像素采样器，前一步生成的原图中的采样网格的像素值进行线性插值，映射到校正后图像中
        我们已知原始采样网格点batch_G=[batch_Gx,batch_Gy]，首先取离该点最近的四个整数点，分别为
        我们已知原始采样网格点batch_G_00=[batch_Gx0,batch_Gy0],...,batch_G_11=[batch_Gx1,batch_Gy1]
        我们根据这四个整数点的像素值batch_I00，batch_I01，batch_I10，batch_I11使用插值法计算原始采样点的像素值
        batch_I=batch_I00*(1-abs(batch_Gx0-batch_Gx))*(1-abs(batch_Gy0-batch_Gy))
               +batch_I01*(1-abs(batch_Gx0-batch_Gx))*(1-abs(batch_Gy1-batch_Gy))
               +batch_I10*(1-abs(batch_Gx1-batch_Gx))*(1-abs(batch_Gy0-batch_Gy))
               +batch_I11*(1-abs(batch_Gx1-batch_Gx))*(1-abs(batch_Gy1-batch_Gy))
        
        """
        if images.dtype != tf.float32:
            raise ValueError('image must be of type tf.float32')
        batch_G = batch_sample_grid
        batch_size,image_h,image_w,image_c = shape_utils.combined_static_and_dynamic_shape(images)
        n = shape_utils.combined_static_and_dynamic_shape(batch_sample_grid)[1]
        
        # 网格是在0-1的范围内计算的，这里需要返回原图大小的网格，注意去掉超过边界的网格点
        batch_Gx = image_w * batch_G[:,:,0] # [batch_size, n, 1]
        batch_Gy = image_h * batch_G[:,:,1]
        batch_Gx = tf.clip_by_value(batch_Gx, 0.0, image_w-2)
        batch_Gy = tf.clip_by_value(batch_Gy, 0.0, image_h-2)

        # 根据4个整数坐标点位置像素线性插值
        batch_Gx0 = tf.cast(tf.floor(batch_Gx),tf.int32) # [batch_size,n,1]
        batch_Gy0 = tf.cast(tf.floor(batch_Gy),tf.int32)
        batch_Gx1 = batch_Gx0 + 1
        batch_Gy1 = batch_Gy0 + 1

        def _get_pixels(images, batch_x, batch_y, batch_indices):
            indices = tf.stack([batch_indices, batch_y, batch_x], axis=2)
            pixels = tf.gather_nd(images, indices)
            return pixels

        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1),[1,n])
        # 周围四个点的像素值
        batch_I00 = _get_pixels(images, batch_Gx0, batch_Gy0, batch_indices)
        batch_I01 = _get_pixels(images, batch_Gx0, batch_Gy1, batch_indices)
        batch_I10 = _get_pixels(images, batch_Gx1, batch_Gy0, batch_indices)
        batch_I11 = _get_pixels(images, batch_Gx1, batch_Gy1, batch_indices) # => [B, n, d]

        batch_Gx0 = tf.to_float(batch_Gx0)
        batch_Gx1 = tf.to_float(batch_Gx1)
        batch_Gy0 = tf.to_float(batch_Gy0)
        batch_Gy1 = tf.to_float(batch_Gy1)

        batch_w00 = (1-tf.abs(batch_Gx0-batch_Gx))*(1-tf.abs(batch_Gy0-batch_Gy))
        batch_w01 = (1-tf.abs(batch_Gx0-batch_Gx))*(1-tf.abs(batch_Gy1-batch_Gy))
        batch_w10 = (1-tf.abs(batch_Gx1-batch_Gx))*(1-tf.abs(batch_Gy0-batch_Gy))
        batch_w11 = (1-tf.abs(batch_Gx1-batch_Gx))*(1-tf.abs(batch_Gy1-batch_Gy)) # [batch_size, n]

        batch_I = tf.add_n([
            tf.tile(tf.expand_dims(batch_w00, 2), [1, 1, image_c]) * batch_I00,
            tf.tile(tf.expand_dims(batch_w01, 2), [1, 1, image_c]) * batch_I01,
            tf.tile(tf.expand_dims(batch_w10, 2), [1, 1, image_c]) * batch_I10,
            tf.tile(tf.expand_dims(batch_w11, 2), [1, 1, image_c]) * batch_I11,
        ]) # [batch_size, n, d]

        output_h, output_w = self._output_image_size
        output_maps = tf.reshape(batch_I, [batch_size,output_h, output_w, -1])
        output_maps = tf.cast(output_maps,dtype=images.dtype)

        if self._summarize_activations:
            tf.summary.image('InputImage1',images, max_outputs=2)
            tf.summary.image('RectifiedImage1',output_maps, max_outputs=2)
        
        return output_maps


    def _build_output_control_points(self):
        """
        构建输出控制点
        ctrl_pts_x和ctrl_pts_y_top分别为横纵坐标向量
        当_num_control_points=20，_margins=(0,0)
        ctrl_pts_x: array([0.0, 0.1,0.2,...,1.0])
        ctrl_pts_y_top: array([0.0,0.0,0.0,...,0.0])
        ctrl_pts_y_bottom: array([1.0,1.0,1.0,...,1.0])
        ctrl_pts_top: array([[0.0,0.0],[0.1,0.0],...[1.0,0.0]])
        ctrl_pts_bottom: array([[0.0,1.0],[0.1,1.0],...[1.0,1.0]])
        output_ctrl_pts: array([[0.0,0.0],[0.1,0.0],...[1.0,0.0],[0.0,1.0],[0.1,1.0],...[1.0,1.0]])
        """
        margin_x, margin_y = self._margins
        if margin_x>0.5 or margin_y>0.5:
            raise ValueError('The margin must be smaller than 0.5.')
        
        num_control_points_per_side = self._num_control_points // 2
        ctrl_pts_x = np.linspace(margin_x, 1.0-margin_x, num_control_points_per_side)
        ctrl_pts_y_top = np.ones(num_control_points_per_side) * margin_y
        ctrl_pts_y_bottom = np.ones(num_control_points_per_side) * (1.0-margin_y)
        ctrl_pts_top = np.stack([ctrl_pts_x,ctrl_pts_y_top],axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x,ctrl_pts_y_bottom],axis=1)
        output_ctrl_pts = np.concatenate([ctrl_pts_top,ctrl_pts_bottom],axis=0)
        return output_ctrl_pts

    
    def _build_output_grid(self):
        """
        使用np中的meshgrid对整幅图像生成标准的矩形网格坐标矩阵
        Xm,Ym分别为图像网格对应的横坐标和总坐标，通过np.stack将横总坐标组合在一起
        output_grid的格式为: 
        array([[[x00,y00],[x01,y01],...,[x0w,y0w]],
               [[x10,y10],[x11,y11],...,[x1w,y1w]],
                ...  ...   ...  ... ...  ... ...  ,
               [[xh0,yh0],[xh1,yh1],...,[xhw,yhw]]])
        [h,w,2]
        """
        outout_h, outout_w = self._output_image_size
        x = (np.arange(outout_w) + 0.5) / outout_w
        y = (np.arange(outout_h) + 0.5) / outout_h
        Xm,Ym = np.meshgrid(x,y)
        output_grid = np.stack((Xm, Ym), axis=2)
        return output_grid


    def _build_delta_C_inv(self):
        """
        根据薄板样条插值的规则，求解变换矩阵方程系数矩阵
        delta_C = [[S, 1, C], [1_T, 0, 0], [C_T, 0, 0]]
        """
        C = self._output_control_points
        k = self._num_control_points
        S = np.zeros((k,k),dtype=float)
        for i in range(k):
            for j in range(k):
                S[i,j] = np.linalg.norm(C[i]-C[j])
        np.fill_diagonal(S, 1)
        S = (S ** 2) * np.log(S)
        delta_C = np.concatenate([
            np.concatenate([S, np.ones((k,1)), C], axis=1),
            np.concatenate([np.ones((1,k)), np.zeros((1,3))], axis=1),
            np.concatenate([np.transpose(C), np.zeros((2,3))], axis=1)],axis=0)
        delta_C_inv = np.linalg.inv(delta_C)
        return delta_C_inv


    def _build_init_bias(self):
        margin_x, margin_y = self._margins
        num_ctrl_pts_per_side = self._num_control_points // 2
        upper_x = np.linspace(margin_x, 1.0-margin_x, num=num_ctrl_pts_per_side)
        lower_x = np.linspace(margin_x, 1.0-margin_x, num=num_ctrl_pts_per_side)

        if self._init_bias_pattern == 'slope':
            upper_y = np.linspace(margin_y, 0.3, num=num_ctrl_pts_per_side)
            lower_y = np.linspace(0.7, 1.0-margin_y, num=num_ctrl_pts_per_side)
        elif self._init_bias_pattern == 'identity':
            upper_y = np.linspace(margin_y, margin_y, num=num_ctrl_pts_per_side)
            lower_y = np.linspace(1.0-margin_y, 1.0-margin_y, num=num_ctrl_pts_per_side)
        elif self._init_bias_pattern == 'sine':
            upper_y = 0.25 + 0.2 * np.sin(2 * np.pi * upper_x)
            lower_y = 0.75 + 0.2 * np.sin(2 * np.pi * lower_x)
        else:
            raise ValueError('Unknown initialization pattern: {}'.format(self._init_bias_pattern))

        init_ctrl_pts = np.concatenate([
            np.stack([upper_x, upper_y], axis=1),
            np.stack([lower_x, lower_y], axis=1),
        ], axis=0)

        if self._activation == 'sigmoid':
            init_biases = -np.log(1. / init_ctrl_pts - 1.)
        elif self._activation == 'none':
            init_biases = init_ctrl_pts
        else:
            raise ValueError('Unknown activation type: {}'.format(self._activation))
        
        return init_biases
