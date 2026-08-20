#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
from typing import Any
from typing import List
from typing import Union

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer

from modelflow.common import logger
from modelflow.common import Config
from modelflow.common.config import convert_config_to_dict
from modelflow.common.exception import ConfigError
from .base import NormDropoutLayer
from .module import Module


@Module.register("dnn")
class DNN(Layer):
    """ DNN layer

    隐藏层支持配置dense，激活函数，归一化，dropout以及激活函数，归一化，dropout的顺序；输出层仅支持配置dense和激活函数

    :register name: dnn

    >>> from tensorflow.python.keras.utils import tf_utils
    >>> dnn = DNN(hidden_dims=[16,8,4],
    ...           hidden_activation=["tanh", "relu"],
    ...           output_activation=Config({"type": "leaky_relu", "alpha": 0.1}),
    ...           norm="batch_norm",
    ...           orders="and")
    >>> input_tensor = tf.random.normal((3, 32))
    >>> output_tensor = dnn(input_tensor)
    >>> tf_utils.get_shapes(output_tensor)
    TensorShape([3, 4])
    >>> dnn.return_hidden_output = True
    >>> output_tensor = dnn(input_tensor)
    >>> tf_utils.get_shapes(output_tensor)
    [TensorShape([3, 16]), TensorShape([3, 8]), TensorShape([3, 4])]

    :param hidden_dims: 隐藏层和输出层单元个数，列表的长度表示隐藏层的数量；**列表中的最后一个元素表示输出层的单元个数**
    :param hidden_activation: 隐藏层（不包括输出层）的激活函数；支持list输入，list长度和隐藏层的长度 `len(hidden_dim) - 1` 相同。
       支持的激活函数包含标签的 `activation` 的Module
    :param output_activation: 输出层的激活函数，支持的激活函数包含标签的 `activation` 的Module
    :param dropout_rate: 隐藏层dropout比率；支持list输入，list长度和隐藏层的长度 `len(hidden_dim) - 1` 相同
    :param use_bias: 是否使用bias偏移
    :param norm: 隐藏层归一化层配置，配置后，该层会放置在每一个隐藏层之后。样例： ``{"type": "batch_norm", "axis": -1}``。
       支持list输入，list长度和隐藏层的长度 `len(hidden_dim) - 1` 相同
    :param orders: 隐藏层activation, dropout, norm的执行顺序，分别使用 `a`, `d`, `n` 来表示。
       例如 `nad` 表示执行顺序是norm, activation, dropout
    :param kernel_initializer: kernel weight的初始化方法
    :param bias_initializer: bias的初始化方法
    :param kernel_regularizer: kernel weight的正则化方法
    :param bias_regularizer: bias的正则化方法
    :param activity_regularizer: 激活层的正则化方法
    :param kernel_constraint: kernel weight的约束函数
    :param bias_constraint: bias的约束函数
    :param return_hidden_output: 是否返回隐藏层的结果。可以接受bool和str值，str有三个选项'raw_out', 'shallow_out', 'deep_out'，
        分别对应最后一层输出，不包含norm的输出，包含norm的输出。bool的false和true对应'raw_out'与 'shallow_out'
    :param name: layer名
    :param kwargs:  :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等

    .. versionchanged:: 23.4.1
       将参数 `activation` 改为 `hidden_activation`, 表示隐藏层（不包括最后一层）的激活函数；新增参数 `output_activation`,
       表示最后一层的激活函数；取消参数 `pure_output`, 改为使用 `output_activation` 替代

    .. versionchanged:: 23.9.1
       参数 `hidden_activation`, `dropout_rate`, `norm` 支持输入list类型的参数；新增参数 `orders`，用来设定激活函数，归一化和dropout
       之间的顺序；新增参数 `return_hidden_output`，支持隐藏层结果的返回

    """

    def __init__(self,
                 hidden_dims: List[int],
                 hidden_activation: Union[str, Config, None, List[Union[str, Config, None]]] = None,
                 output_activation: Union[str, Config, None] = None,
                 dropout_rate: Union[float, List[float]] = 0.0,
                 use_bias: bool = True,
                 norm: Union[str, Config, None, List[Union[str, Config, None]]] = None,
                 output_norm: str = None,
                 orders: Union[str, List[str]] = "adn",
                 kernel_initializer: Union[str, Config, None] = 'glorot_uniform',
                 bias_initializer: Union[str, Config, None] = 'zeros',
                 kernel_regularizer: Union[str, Config, None] = None,
                 bias_regularizer: Union[str, Config, None] = None,
                 activity_regularizer: Union[str, Config, None] = None,
                 kernel_constraint: Union[str, Config, None] = None,
                 bias_constraint: Union[str, Config, None] = None,
                 return_hidden_output: Union[bool, str] = False,
                 name="dnn",
                 **kwargs):
        super(DNN, self).__init__(name=name, **kwargs)

        if not hidden_dims:
            raise ConfigError(f"Units is empty in DNN with name {name}. ")

        self.hidden_dims = hidden_dims
        self.hidden_activation = self._validate_hidden_config(hidden_activation)
        self.output_activation = output_activation
        self.dropout_rate = self._validate_hidden_config(dropout_rate)
        self.use_bias = use_bias
        self.norm = self._validate_hidden_config(norm)
        self.orders = self._validate_hidden_config(orders)
        self.kernel_initializer = convert_config_to_dict(kernel_initializer)
        self.bias_initializer = convert_config_to_dict(bias_initializer)
        self.kernel_regularizer = convert_config_to_dict(kernel_regularizer)
        self.bias_regularizer = convert_config_to_dict(bias_regularizer)
        self.activity_regularizer = convert_config_to_dict(activity_regularizer)
        self.kernel_constraint = convert_config_to_dict(kernel_constraint)
        self.bias_constraint = convert_config_to_dict(bias_constraint)
        self.return_hidden_output = self.convert_return_hidden_output(return_hidden_output)
        self.output_norm = output_norm
        self.layers: List[List[Layer]] = []

        self.supports_masking = True

    def convert_return_hidden_output(self, return_hidden_output) -> str:
        '''
        返回str，可返回'raw_out','shallow_out','deep_out'
        如果是True,则对应'shallow_out'
        如果是False,则对应'raw_out'

        :param return_hidden_output:
        :return:
        '''

        if isinstance(return_hidden_output, bool):
            if return_hidden_output:
                return 'shallow_out'
            else:
                return 'raw_out'
        if return_hidden_output in ['raw_out', 'shallow_out', 'deep_out']:
            return return_hidden_output
        else:
            logger.warning("Cannot match `return_hidden_output` in ['raw_out','shallow_out','deep_out']."
                           " Default return 'raw_out'.")
            return 'raw_out'

    def _validate_hidden_config(self, hidden_config: Any) -> list:
        """校验隐藏层的参数，并将参数的类型转换为list

        :param hidden_config: 隐藏层的参数
        :return: list类型的隐藏层参数
        """
        hidden_count = len(self.hidden_dims) - 1
        if isinstance(hidden_config, list):
            if len(hidden_config) != hidden_count:
                raise ConfigError(f"The length of hidden config in DNN Module `{self.name}` is not {hidden_count}. "
                                  f"Please check it.")
            return hidden_config

        # 处理配置不是list格式的情况
        return [hidden_config] * hidden_count

    def build(self, input_shape: tf.TensorShape):
        """ 构建DNN隐藏层，dropout层和norm层

        :param input_shape: 输入tensor的维度
        """

        # 构建隐藏层
        for index, units in enumerate(self.hidden_dims[:-1]):
            dense_layer = Dense(name=f"dense_{index}",
                                units=units,
                                activation=None,
                                use_bias=self.use_bias,
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                activity_regularizer=self.activity_regularizer,
                                kernel_constraint=self.kernel_constraint,
                                bias_constraint=self.bias_constraint)
            norm_dropout_layer = NormDropoutLayer(norm=self.norm[index],
                                                  dropout_rate=self.dropout_rate[index],
                                                  activation=self.hidden_activation[index],
                                                  orders=self.orders[index],
                                                  name=f"norm_dropout_{index}")
            self.layers.append([dense_layer, norm_dropout_layer])

        # 构建输出层
        last_activation = Module.from_config(self.output_activation, tags=["activation"])
        last_dense_layer = Dense(name=f"dense_{len(self.hidden_dims) - 1}",
                                 units=self.hidden_dims[-1],
                                 activation=last_activation,
                                 use_bias=self.use_bias,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.bias_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer,
                                 activity_regularizer=self.activity_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 bias_constraint=self.bias_constraint)
        last_norm_layer = Module.from_config(self.output_norm, tags=["norm"])

        if last_norm_layer:
            self.layers.append([last_dense_layer, last_norm_layer])
        else:
            self.layers.append([last_dense_layer])
        super(DNN, self).build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> Union[tf.Tensor, List[tf.Tensor]]:
        """ 执行DNN层

        :param inputs: 输入tensor
        :param kwargs: 额外参数
        :return: DNN最后一层的结果; 如果 `return_hidden_output` 为 `True`, 返回所有隐藏层的结果，list类型
        """
        shallow_output = []
        deep_output = []

        last_output = inputs
        for single_layer in self.layers:
            for layer in single_layer:
                last_output = layer(last_output)
                deep_output.append(last_output)
            shallow_output.append(last_output)

        if self.return_hidden_output == 'raw_out':
            return last_output
        elif self.return_hidden_output == 'shallow_out':
            return shallow_output
        else:
            return deep_output
