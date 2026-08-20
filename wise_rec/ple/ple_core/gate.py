#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.

from typing import List, Union
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers

from modelflow.common.config import convert_config_to_dict, Config
from modelflow.integrations.tensorflow.modules.module import Module


class Gate(layers.Layer):
    """Gate基类

    :param activation: 权重映射层激活函数
    :param gate_type: gate粒度，elements表示维度层面,vector表示Tensor层面
    :param extra_proj_config: 权重映射层额外处理，对key先extra_proj在dense
    :param kernel_initializer: 权重映射层权重初始化类型
    :param bias_initializer: 权重映射层Bias初始化类型
    :param name: 输出层名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 activation: str = 'sigmoid',
                 gate_type: str = 'elements',
                 use_bias: bool = True,
                 extra_proj_config: Union[str, Config, None] = None,
                 kernel_initializer: Union[str, Config, None] = 'glorot_uniform',
                 bias_initializer: Union[str, Config, None] = 'zeros',
                 name="gate",
                 **kwargs):
        super(Gate, self).__init__(name=name, **kwargs)
        self.activation = activation
        self.gate_type = gate_type
        self.use_bias = use_bias
        self.extra_proj_config = extra_proj_config
        self.kernel_initializer = convert_config_to_dict(kernel_initializer)
        self.bias_initializer = convert_config_to_dict(bias_initializer)
        self.weight_projection_layer: Optional[layers.Layer] = None
        self.extra_projection_layer: Optional[layers.Layer] = None

    def build(self, input_shape):

        if self.extra_proj_config:
            self.extra_projection_layer = Module.from_config(self.extra_proj_config)
        if self.gate_type == 'elements':
            weight_dim = input_shape[-1][-1]
        elif self.gate_type == 'vector':
            weight_dim = len(input_shape) - 1
        elif self.gate_type == 'vector_matrix':
            weight_dim = input_shape[-1][-2]
        else:
            raise ValueError('parameter gate_type must be elements, vector')

        self.weight_projection_layer = layers.Dense(weight_dim,
                                                    activation=self.activation,
                                                    use_bias=self.use_bias,
                                                    kernel_initializer=self.kernel_initializer,
                                                    bias_initializer=self.bias_initializer)

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        pass

    def weight_projection(self, key: tf.Tensor):
        key = self.extra_projection_layer(key) if self.extra_proj_config else key
        return self.weight_projection_layer(key)


@Module.register("single_gate")
class SingleGate(Gate):
    """SingleGate 参考y = a*x1

    :param activation: 权重映射层激活函数
    :param gate_type: gate粒度，elements表示维度层面,vector表示Tensor层面
    :param extra_proj_config: 权重映射层额外处理，对key先extra_proj在dense
    :param kernel_initializer: 权重映射层权重初始化类型
    :param bias_initializer: 权重映射层Bias初始化类型
    :param name: 输出层名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 activation: str = 'sigmoid',
                 gate_type: str = 'elements',
                 use_bias: bool = True,
                 extra_proj_config: Union[str, Config, None] = None,
                 scale: float = 1.0,
                 kernel_initializer: Union[str, Config, None] = 'glorot_uniform',
                 bias_initializer: Union[str, Config, None] = 'zeros',
                 name="binary_gate",
                 **kwargs):
        super(SingleGate, self).__init__(activation=activation,
                                         gate_type=gate_type,
                                         use_bias=use_bias,
                                         extra_proj_config=extra_proj_config,
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer,
                                         name=name,
                                         **kwargs)
        self.scale = scale

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]], **kwargs) -> tf.Tensor:
        """调用SingleGate层

        :param inputs: 2个Tensor
           * inputs[0]: key,权重计算的输入，维度是[batch_size, key_embedding_size]
           * inputs[1]: value，维度[batch_size, value_embedding_size]
        :param kwargs: 额外参数
        :return: 输出维度为[batch_size, ..., key_embedding_size]
        """
        key = inputs[0]
        value = inputs[1] if len(inputs) > 1 else inputs[0]
        w = self.weight_projection(key) * self.scale
        output = tf.multiply(w, value)
        return output


@Module.register("binary_gate")
class BinaryGate(Gate):
    """BinaryGate 参考y = ax1+(1-a)x2

    :param activation: 权重映射层激活函数
    :param gate_type: gate粒度，elements表示维度层面,vector表示Tensor层面
    :param extra_proj_config: 权重映射层额外处理，对key先extra_proj在dense
    :param kernel_initializer: 权重映射层权重初始化类型
    :param bias_initializer: 权重映射层Bias初始化类型
    :param name: 输出层名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 activation: str = 'sigmoid',
                 gate_type: str = 'elements',
                 use_bias: bool = True,
                 extra_proj_config: Union[str, Config, None] = None,
                 kernel_initializer: Union[str, Config, None] = 'glorot_uniform',
                 bias_initializer: Union[str, Config, None] = 'zeros',
                 name="binary_gate",
                 **kwargs):
        super(BinaryGate, self).__init__(activation=activation,
                                         gate_type=gate_type,
                                         use_bias=use_bias,
                                         extra_proj_config=extra_proj_config,
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer,
                                         name=name,
                                         **kwargs)

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]], **kwargs) -> tf.Tensor:
        """调用BinaryGate层

        :param inputs: 三个Tensor
           * inputs[0]: key,权重计算的输入，维度是[batch_size, key_embedding_size]
           * inputs[1]: value1，维度[batch_size, value_embedding_size]
           * inputs[2]: value2，维度[batch_size, value_embedding_size]
        :param kwargs: 额外参数
        :return: 输出维度为[batch_size, ..., key_embedding_size]
        """
        key, v1, v2 = inputs
        w = self.weight_projection(key)
        output = tf.multiply(w, v1) + tf.multiply(1 - w, v2)
        return output


@Module.register("mlp_gate")
class MLPGate(Gate):
    """MLPGate 参考y = a1x1+a2x2+...+anxn

    :param activation: 权重映射层激活函数
    :param gate_type: gate粒度，elements表示维度层面,vector表示Tensor层面
    :param extra_proj_config: 权重映射层额外处理，对key先extra_proj在dense
    :param kernel_initializer: 权重映射层权重初始化类型
    :param bias_initializer: 权重映射层Bias初始化类型
    :param name: 输出层名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 activation: str = 'softmax',
                 gate_type: str = 'vector',
                 use_bias: bool = True,
                 extra_proj_config: Union[str, Config, None] = None,
                 kernel_initializer: Union[str, Config, None] = 'glorot_uniform',
                 bias_initializer: Union[str, Config, None] = 'zeros',
                 name="mlp_gate",
                 **kwargs):
        super(MLPGate, self).__init__(activation=activation,
                                      gate_type=gate_type,
                                      use_bias=use_bias,
                                      extra_proj_config=extra_proj_config,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer,
                                      name=name,
                                      **kwargs)

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]], **kwargs) -> tf.Tensor:
        """调用MLPGate层

        :param inputs: 三个Tensor
           * inputs[0]: key,权重计算的输入，维度是[batch_size, key_embedding_size]
           * inputs[1]: value1，维度[batch_size, value_embedding_size] or [batch_size, value_num ,value_embedding_size]
           ......
           * inputs[n]: valuen，维度[batch_size, value_embedding_size]
        :param kwargs: 额外参数
        :return: 输出维度为[batch_size, value_embedding_size]
        """
        key = inputs[0]
        if len(inputs) > 2:
            expand_inputs = []
            for i in range(1, len(inputs)):
                expand_inputs.append(tf.expand_dims(inputs[i], axis=1))
            value = tf.concat(expand_inputs, axis=1)
        else:
            value = inputs[1]
        w = tf.expand_dims(self.weight_projection(key), axis=-1)
        output = tf.reduce_sum(value * w, axis=1)
        return output


@Module.register("mask_gate")
class MaskGate(Gate):
    """MaskGate 基于MLPGate，增加了屏蔽某个专家的操作
    公式参考y = a(1)x(1)+...a(i-1)x(i-1)+0*x(i)+a(i+1)x(i+1)...+a(n)x(n)

    :param activation: 权重映射层激活函数
    :param gate_type: gate粒度，elements表示维度层面,vector表示Tensor层面
    :param extra_proj_config: 权重映射层额外处理，对key先extra_proj在dense
    :param kernel_initializer: 权重映射层权重初始化类型
    :param bias_initializer: 权重映射层Bias初始化类型
    :param name: 输出层名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 activation: str = 'linear',
                 gate_type: str = 'vector_matrix',
                 use_bias: bool = True,
                 extra_proj_config: Union[str, Config, None] = None,
                 kernel_initializer: Union[str, Config, None] = 'glorot_uniform',
                 bias_initializer: Union[str, Config, None] = 'zeros',
                 name="mask_gate",
                 **kwargs):
        super().__init__(activation=activation,
                         gate_type=gate_type,
                         use_bias=use_bias,
                         extra_proj_config=extra_proj_config,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         name=name,
                         **kwargs)

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]], **kwargs) -> tf.Tensor:
        """

        :param inputs:
           * inputs[0]: key,权重计算的输入，维度是[batch_size, key_embedding_size]
           * inputs[1]: mask_index,[batch_size]
           * inputs[2]: value，[batch_size, value_num , value_embedding_size]
        :param kwargs: 额外参数
        :return: 输出维度为[batch_size, value_embedding_size]
        """
        key = inputs[0]
        mask_index = inputs[1]
        value = inputs[2]

        batch_size = tf.shape(mask_index)[0]
        # 获取网络后的输出
        nn_out = self.weight_projection(key)
        # 构建索引
        indices = tf.stack([tf.range(batch_size), mask_index], axis=1)
        # 更新私有专家的权重为负无穷，softmax之后该专家权重为0
        updates = tf.fill((batch_size,), -tf.float32.max)
        output_tensor = tf.tensor_scatter_nd_update(nn_out, indices, updates)
        w = tf.expand_dims(tf.nn.softmax(output_tensor), axis=-1)

        output = tf.reduce_sum(value * w, axis=1)
        return output
