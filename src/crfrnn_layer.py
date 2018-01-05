#coding=utf-8
"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
import high_dim_filter_loader
import keras.backend as K
custom_module = high_dim_filter_loader.custom_module


class CrfRnnLayer(Layer):
    """ Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        super(CrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel, 
        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes, self.num_classes),
                                                   initializer='uniform',
                                                   trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes, self.num_classes),
                                                     initializer='uniform',
                                                     trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer='uniform',
                                                    trainable=True)

        super(CrfRnnLayer, self).build(input_shape)

    def call(self, inputs):

        # inputs[0]: [1, 56, 56, 2]
        # inputs[1]: [None, 56, 56, 1] [0,255] 二值化
        
        ret_tensor = []
        input_unary, input_rgb = tf.split(inputs,2,0) # [1, height, width]

            
        cur_unary = input_unary[0] # after sigmoid:[h, w]
        cur_unary = tf.expand_dims(cur_unary, -1)# [h, w ,1]
        cur_unary = tf.concat([1.-cur_unary, cur_unary], axis=-1)#[h, w, 2]
        unaries = tf.transpose(cur_unary, perm=(2, 0, 1)) #[2, h, w]

        cur_rgb = input_rgb[0] # [h, w], bool
        cur_rgb = tf.expand_dims(cur_rgb, 0) # [1, h, w], bool
        cur_rgb = tf.cast(cur_rgb, tf.float32)*tf.constant(255., dtype=tf.float32) # [1, h, w], int32
        rgb = tf.tile(cur_rgb, [3, 1, 1]) # [3, h, w], int32


        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        all_ones = np.ones((c, h, w), dtype=np.float32)

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)
        q_values = unaries

        for i in range(self.num_iterations):
            if i==0:# i=0时已经用sigmoid归一化过了
                softmax_out = q_values
            else: 
                softmax_out = tf.nn.softmax(q_values, dim=0)#(2, h, w)

            # Spatial filtering
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals # (2, h, w)

            # Bilateral filtering
            bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                          theta_alpha=self.theta_alpha,
                                                          theta_beta=self.theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals # (2, h, w)

            # Weighting filter outputs: 2 x 2 * 2 x h x w = (2, h*w)
            message_passing = (tf.matmul(self.spatial_ker_weights,
                                         tf.reshape(spatial_out, (c, -1))) +
                               tf.matmul(self.bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (c, -1)))) 

            # Compatibility transform: 2x2 * 2x(h*w) = (2, h*w)
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w))
            q_values = unaries - pairwise

        cur_q = tf.transpose(tf.reshape(q_values, (c, h, w)), perm=(1, 2, 0)) #[h, w, 2]
        # change to input shape: [N, height, width] -> 返回为1的概率
        # channel 0: background prob ; channel 1: object prob
        # softmax之后，只用返回channel1的概率
        obj_prob = tf.nn.softmax(cur_q, dim=2)[:,:,1]#(h, w)

        # 输出: ret_tensor [1, h, w]
        return tf.reshape(obj_prob, (1, 56, 56))

    def compute_output_shape(self, input_shape):
        return input_shape
