from typing import Any
from typing import List
from typing import Union

from modelflow.integrations.tensorflow.modules.module import Module

import tensorflow as tf
from tensorflow.keras.layers import Layer

from modelflow.common import Config
from modelflow.common.exception import ConfigError

from modelflow.common.config import convert_config_to_dict
from modelflow.integrations.tensorflow.modules import NormDropoutLayer


@Module.register("dnn_split_v2")
class SerialsplitDNNV2(Layer):
    def __init__(self, hidden_dims, group_num=2, hidden_activation='relu', l2_reg=0, dropout_rate=0.03,
                 use_bn=True, seed=1024,
                 output_activation='relu',
                 use_bias: bool = True,
                 norm: Union[str, Config, None, List[Union[str, Config, None]]] = "batch_norm",
                 orders: Union[str, List[str]] = "adn",
                 kernel_initializer: Union[str, Config, None] = 'glorot_uniform',
                 bias_initializer: Union[str, Config, None] = 'zeros',
                 kernel_regularizer: Union[str, Config, None] = None,
                 bias_regularizer: Union[str, Config, None] = None,
                 activity_regularizer: Union[str, Config, None] = None,
                 kernel_constraint: Union[str, Config, None] = None,
                 bias_constraint: Union[str, Config, None] = None,
                 **kwargs):
        """
        SerialsplitDNNV2 层，用于减少MLP层的计算量
        输入shape：(batch_size, input_dim)
        输出shape：(batch_size, hidden_size[-1])
        :param group_num: 分组MLP的数量
        :param hidden_activation: 激活函数类型
        :param l2_reg: 0~1之间的float参数，定义l2政策话系数，用于kernel weights matrix
        :param dropout_rate: [0, 1)之间的float参数，定义dropout率
        :param use_bn: 在激活函数前是否使用BatchNormalization，bool
        :param seed: 随机种子
        :param kwargs: 其他参数
        """
        self.layer_name = "dnn_split"
        self.hidden_units = hidden_dims
        self.group_num = group_num
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.use_bias = use_bias
        self.output_activation = output_activation

        self.kernels = None
        self.group_interact = None
        self.bias = None
        self.batch_norm = None
        self.activation_layers = None
        self.dropout_layers = None
        self.input_size = None
        self.input_split_size = None

        self.dense_nets = []
        self.group_nets = []
        self.norm_dropout_nets = []

        self.norm = self._validate_hidden_config(norm)
        self.orders = self._validate_hidden_config(orders)
        self.dropout_rate = self._validate_hidden_config(dropout_rate)
        self.hidden_activation = self._validate_hidden_config(hidden_activation)

        self.kernel_initializer = convert_config_to_dict(kernel_initializer)
        self.bias_initializer = convert_config_to_dict(bias_initializer)
        self.kernel_regularizer = convert_config_to_dict(kernel_regularizer)
        self.bias_regularizer = convert_config_to_dict(bias_regularizer)
        self.activity_regularizer = convert_config_to_dict(activity_regularizer)
        self.kernel_constraint = convert_config_to_dict(kernel_constraint)
        self.bias_constraint = convert_config_to_dict(bias_constraint)

        super(SerialsplitDNNV2, self).__init__(**kwargs)

    def _validate_hidden_config(self, hidden_config: Any) -> list:
        """校验隐藏层的参数，并将参数的类型转换为list

        :param hidden_config: 隐藏层的参数
        :return: list类型的隐藏层参数
        """
        hidden_count = len(self.hidden_units) - 1
        if isinstance(hidden_config, list):
            if len(hidden_config) != hidden_count:
                raise ConfigError(f"The length of hidden config in DNN Module `{self.name}` is not {hidden_count}. "
                                  f"Please check it.")
            return hidden_config

        # 处理配置不是list格式的情况
        return [hidden_config] * hidden_count

    def build(self, input_shape):
        hidden_divide_flag = True
        input_size = input_shape[-1]
        self.input_size = input_size

        for hidden_unit in self.hidden_units:
            if hidden_unit % self.group_num != 0:
                hidden_divide_flag = False

        if len(input_shape) != 2:
            raise ValueError("A `SerialsplitDNNV2` should be called on a list of at 2 inputs")

        if not isinstance(self.group_num, int) or self.group_num <= 0 or input_size % self.group_num != 0:
            raise ValueError("The group number must be divisible by input size")

        if not hidden_divide_flag:
            raise ValueError("The group number must be divisible by hidden units")

        hidden_units = [input_size] + list(self.hidden_units)
        self.input_split_size = [
            tf.cast(hidden_units[i] / self.group_num, tf.int32) for i, _ in enumerate(hidden_units)]

        self.kernels = [
            self.add_weight(
                name='kernels_' + str(i) + self.layer_name,
                initializer=self.kernel_initializer,
                shape=(self.group_num, tf.cast(hidden_units[i] / self.group_num, tf.int32), hidden_units[i + 1]),
                regularizer=self.kernel_regularizer,
                trainable=True)
            for i, _ in enumerate(hidden_units[:-1])]

        self.bias = [
            self.add_weight(
                name='bias' + str(i),
                initializer=self.bias_initializer,
                shape=(self.group_num, self.hidden_units[i]),
                trainable=True)
            for i, _ in enumerate(self.hidden_units)]

        self.group_interact = [
            self.add_weight(
                name='group' + str(i),
                initializer=self.kernel_initializer,
                shape=(self.group_num, self.group_num,),
                trainable=True)
            for i, _ in enumerate(self.hidden_units)]

        for i, _ in enumerate(self.hidden_units[:-1]):
            norm_dropout_layer = NormDropoutLayer(norm=self.norm[i],
                                                  dropout_rate=self.dropout_rate[i],
                                                  activation=self.hidden_activation[i],
                                                  orders=self.orders[i],
                                                  name=f"norm_dropout_{i}")

            self.norm_dropout_nets.append(norm_dropout_layer)

        self.norm_dropout_nets.append(Module.from_config(self.output_activation, tags=["activation"]))

        super(SerialsplitDNNV2, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):
        deep_input = inputs
        # 将传入的输入embedding dim切分为group_num份
        for i, _ in enumerate(self.hidden_units):
            deep_input = tf.reshape(deep_input, [-1, self.group_num, self.input_split_size[i]])
            deep_input = tf.einsum("ijk,jkl->ijl", deep_input, self.kernels[i])
            # group interaction
            deep_input = tf.einsum("ijk,jl->ilk", deep_input, self.group_interact[i])
            deep_input = deep_input + self.bias[i]
            deep_input = self.norm_dropout_nets[i](deep_input)
            deep_input = tf.reduce_sum(deep_input, axis=1)

        deep_out = deep_input
        return deep_out
