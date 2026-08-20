#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import inspect
from typing import List, Dict
from typing import Optional
from typing import Union
from typing import get_type_hints
from typing import _GenericAlias

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

from modelflow.common import constants
from modelflow.common import Config
from modelflow.common import logger
from modelflow.common.config import convert_config_to_dict
from modelflow.common.exception import ConfigError
from modelflow.integrations.tensorflow.modules import Module
from modelflow.integrations.tensorflow.modules import NormDropoutLayer


@Module.register("mmoe_framework", tags=["multi_task"])
class MmoeFramework(NormDropoutLayer):
    """MMOE框架层，代码参考自 `MMOE论文 <https://dl.acm.org/doi/10.1145/3219819.3220007>`_ 和
    `github实现 <https://github.com/drawbridge/keras-mmoe/blob/master/mmoe.py>`_

    该类只包含MMOE的框架，不包含专家层的初始化。

    :register name: mmoe_framework

    :param num_tasks: 任务数，也指代gate的数量
    :param stack_outputs:
       是否对输出列表进行stack，如果为 ``True``，输出的维度是：[batch_size, num_tasks, output_dim]；
       如果为 ``False``，输出为一个列表，列表的长度为num_tasks，里面每个元素的维度是 [batch_size, output_dim]
    :param norm: 归一化配置
    :param dropout_rate: dropout比例
    :param activation: 激活函数配置，支持的激活函数包含标签的 `activation` 的Module
    :param orders: activation, dropout, norm的执行顺序，分别使用 `a`, `d`, `n` 来表示。
       例如 `nad` 表示执行顺序是norm, activation, dropout
    :param use_gate_bias: 门控层是否需要添加bias
    :param gate_activation: 门控层激活函数
    :param gate_kernel_initializer: 门控层kernel初始化方法
    :param gate_kernel_regularizer: 门控层kernel正则化方法
    :param gate_kernel_constraint: 门控层kernel约束方法
    :param gate_bias_initializer: 门控层bias初始化方法
    :param gate_bias_regularizer: 门控层bias正则化方法
    :param gate_bias_constraint: 门控层bias约束方法
    :param gate_activity_regularizer: 门控输出层（激活函数）正则化方法
    :param need_gate_output: 存储gate网络的softmax输出
    :param name: layer名字
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等

    .. versionchanged:: 23.9.1
       移除参数 `norm_first`, 新增激活函数配置参数 `activation` 和执行顺序参数 `orders`

    """

    def __init__(self,
                 num_tasks: int,
                 stack_outputs: bool = False,
                 norm: Optional[Union[str, Config]] = None,
                 dropout_rate: float = 0.0,
                 activation: Union[str, Config, None] = None,
                 orders: str = "adn",
                 use_gate_bias: bool = True,
                 gate_activation: Union[str, Config, None] = 'softmax',
                 gate_kernel_initializer: Union[str, Config, None] = 'VarianceScaling',
                 gate_kernel_regularizer: Union[str, Config, None] = None,
                 gate_kernel_constraint: Union[str, Config, None] = None,
                 gate_bias_initializer: Union[str, Config, None] = 'zeros',
                 gate_bias_regularizer: Union[str, Config, None] = None,
                 gate_bias_constraint: Union[str, Config, None] = None,
                 gate_activity_regularizer: Union[str, Config, None] = None,
                 need_gate_output: bool = False,
                 name="mmoe_framework",
                 **kwargs):
        super(MmoeFramework, self).__init__(norm=norm,
                                            dropout_rate=dropout_rate,
                                            activation=activation,
                                            orders=orders,
                                            name=name,
                                            **kwargs)
        self.num_tasks: int = num_tasks
        self.stack_outputs = stack_outputs

        # gate 激活函数配置
        self.gate_activation = convert_config_to_dict(gate_activation)

        # gate kernel配置
        self.gate_kernel_initializer = convert_config_to_dict(gate_kernel_initializer)
        self.gate_kernel_regularizer = convert_config_to_dict(gate_kernel_regularizer)
        self.gate_kernel_constraint = convert_config_to_dict(gate_kernel_constraint)

        # gate bias配置
        self.use_gate_bias = convert_config_to_dict(use_gate_bias)
        self.gate_bias_initializer = convert_config_to_dict(gate_bias_initializer)
        self.gate_bias_regularizer = convert_config_to_dict(gate_bias_regularizer)
        self.gate_bias_constraint = convert_config_to_dict(gate_bias_constraint)

        self.gate_activity_regularizer = convert_config_to_dict(gate_activity_regularizer)

        self.gate_layers: List[Layer] = []
        self.mmoe_gate_output = {}

        self.need_gate_output = need_gate_output

    def build(self, input_shape: List[tf.TensorShape]):
        """构建内部layer和变量

        :param input_shape: 和 ``call`` 函数中的inputs结构相同，标记 ``call`` 函数中每个tensor的维度
        """
        expert_input_shape, expert_output_shape = input_shape

        # 获取expert数量
        num_experts = expert_output_shape[1]

        # 构建门控层
        self._build_gate_layers(num_experts)

        super(MmoeFramework, self).build(input_shape)

    def call(self, inputs: List[tf.Tensor], **kwargs) -> Union[tf.Tensor, List[tf.Tensor]]:
        """ 执行MMOE框架层

        :param inputs: 一个包含两个Tensor的list或tuple，第一个表示门控层的输入（gate_inputs），用于构建门控层；
           第二个是专家层的输出（expert_outputs），用于和门控层的输出进行加权求和
        :param kwargs: 额外参数
        :return: MMOE框架层输出

        :input shape:
           * expert_outputs: [batch_size, num_expert, output_dim]

        :output shape:
           如果 ``stack_output = True``, [batch_size, num_tasks, output_dim]；
           否则为列表格式，列表的长度为num_tasks，里面每个元素的维度是 [batch_size, output_dim]

        """
        # expert_outputs shape: [batch_size, num_expert, output_dim]
        # gate_inputs shape: [batch_size, gate_input_dim]

        gate_inputs, expert_outputs = inputs

        mmoe_outputs: List[tf.Tensor] = []
        for idx, gate_layer in enumerate(self.gate_layers):

            gate_outputs = gate_layer(gate_inputs)

            if self.need_gate_output:
                tf.py_function(self.get_gate_output, [idx, gate_outputs], Tout=[tf.int64])

            # output shape: [batch_size, num_experts, output_dim]
            output = tf.expand_dims(gate_outputs, axis=-1) * expert_outputs

            # output shape: [batch_size, output_dim]
            output = tf.reduce_sum(output, axis=-2)
            output = self.norm_dropout(output)

            mmoe_outputs.append(output)

        if self.stack_outputs:
            # output shape: [batch_size, num_tasks, output_dim]
            return tf.stack(mmoe_outputs, axis=1)
        else:
            return mmoe_outputs

    def get_gate_output(self, idx, gate_output):
        self.mmoe_gate_output[idx.numpy()] = gate_output
        return 0

    def _build_gate_layers(self, num_experts: int):
        for idx in range(self.num_tasks):
            gate_layer = layers.Dense(name=f"gate_layer_{idx}",
                                      units=num_experts,
                                      activation=self.gate_activation,
                                      use_bias=self.use_gate_bias,
                                      kernel_initializer=self.gate_kernel_initializer,
                                      bias_initializer=self.gate_bias_initializer,
                                      kernel_regularizer=self.gate_kernel_regularizer,
                                      bias_regularizer=self.gate_bias_regularizer,
                                      kernel_constraint=self.gate_kernel_constraint,
                                      bias_constraint=self.gate_bias_constraint,
                                      activity_regularizer=self.gate_activity_regularizer)
            self.gate_layers.append(gate_layer)


@Module.register("mmoe", tags=["multi_task"])
class Mmoe(MmoeFramework):
    """MMOE层，代码参考自 `MMOE论文 <https://dl.acm.org/doi/10.1145/3219819.3220007>`_ 和
    `github实现 <https://github.com/drawbridge/keras-mmoe/blob/master/mmoe.py>`_

    相比于 :class:`MmoeFramework`，该层包含了专家层的初始化。但相对于MMOE来说，经过门控层之后的multi-tower没有实现。

    :register name: mmoe

    :param expert_config: 专家层配置，如果为list格式，则list的长度为专家层的数量，如果为dict/Config格式，则代表所有专家层的结构是一致的。
    :param num_tasks: 任务数，也指代gate的数量
    :param num_experts: 专家层数量，如果``expert_config``的类型是list，该参数无效。
    :param stack_outputs:
       是否对输出列表进行stack，如果为 ``True`` ，输出的维度是：[batch_size, num_tasks, output_dim]；
       如果为 ``False`` , 输出为一个列表，列表的长度为num_tasks，里面每个元素的维度是 [batch_size, output_dim]
    :param norm: 归一化配置
    :param dropout_rate: dropout比例
    :param activation: 激活函数配置，支持的激活函数包含标签的 `activation` 的Module
    :param orders: activation, dropout, norm的执行顺序，分别使用 `a`, `d`, `n` 来表示。
       例如 `nad` 表示执行顺序是norm, activation, dropout
    :param use_gate_bias: 门控层是否需要添加bias
    :param gate_activation: 门控层激活函数
    :param gate_kernel_initializer: 门控层kernel初始化方法
    :param gate_kernel_regularizer: 门控层kernel正则化方法
    :param gate_kernel_constraint: 门控层kernel约束方法
    :param gate_bias_initializer: 门控层bias初始化方法
    :param gate_bias_regularizer: 门控层bias正则化方法
    :param gate_bias_constraint: 门控层bias约束方法
    :param gate_activity_regularizer: 门控输出层（激活函数）正则化方法
    :param need_gate_output: 存储gate网络的softmax输出
    :param name: layer名字
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等

    .. versionchanged:: 23.9.1
       移除参数 `norm_first`, 新增激活函数配置参数 `activation` 和执行顺序参数 `orders`

    每个专家不同输入配置，inputs: {
            inputs:
            [[
                ['ref::postprocess.embedding.u_activity_new'],
                ['ref::postprocess.embedding.i_31001_0', 'ref::postprocess.embedding.u_activity_new'],
                ["ref::concat_all.embedding"],
                ["ref::concat_all.embedding"],
            ], #expert_inputs
            ['ref::postprocess.embedding.i_31001_0', 'ref::postprocess.embedding.i_31001_0'] #gate_inputs
            ]
    },

    """

    def __init__(self,
                 expert_config: Union[List[Config], Config],
                 num_tasks: int,
                 num_experts: Optional[int] = None,
                 stack_outputs: bool = False,
                 norm: Union[str, Config, None] = None,
                 dropout_rate: float = 0.0,
                 activation: Union[str, Config, None] = None,
                 orders: str = "adn",
                 use_gate_bias: bool = True,
                 gate_activation: Union[str, Config, None] = 'softmax',
                 gate_kernel_initializer: Union[str, Config, None] = 'VarianceScaling',
                 gate_kernel_regularizer: Union[str, Config, None] = None,
                 gate_kernel_constraint: Union[str, Config, None] = None,
                 gate_bias_initializer: Union[str, Config, None] = 'zeros',
                 gate_bias_regularizer: Union[str, Config, None] = None,
                 gate_bias_constraint: Union[str, Config, None] = None,
                 gate_activity_regularizer: Union[str, Config, None] = None,
                 need_gate_output: bool = False,
                 name="mmoe_framework",
                 **kwargs):
        super().__init__(num_tasks=num_tasks,
                         stack_outputs=stack_outputs,
                         norm=norm,
                         dropout_rate=dropout_rate,
                         activation=activation,
                         orders=orders,
                         use_gate_bias=use_gate_bias,
                         gate_activation=gate_activation,
                         gate_kernel_initializer=gate_kernel_initializer,
                         gate_kernel_regularizer=gate_kernel_regularizer,
                         gate_kernel_constraint=gate_kernel_constraint,
                         gate_bias_initializer=gate_bias_initializer,
                         gate_bias_regularizer=gate_bias_regularizer,
                         gate_bias_constraint=gate_bias_constraint,
                         gate_activity_regularizer=gate_activity_regularizer,
                         need_gate_output=need_gate_output,
                         name=name,
                         **kwargs)
        self.expert_config = expert_config
        self.num_experts = num_experts
        self.expert_layers: List[Layer] = []
        if isinstance(self.expert_config, list) and self.num_experts is None:
            raise ConfigError(f"Parameter num_experts must be configured when expert_config is a list.")

    def build(self, input_shape: tf.TensorShape):
        """构建内部layer和变量

        :param input_shape: 和 ``call`` 函数中的inputs结构相同，标记 ``call`` 函数中每个tensor的维度
        """

        # 构建专家层
        if isinstance(self.expert_config, list):
            expert_config_count = len(self.expert_config)
            if self.num_experts != expert_config_count:
                logger.info("Parameter num_expert is replaced with %s due to expert config. Previous value was %s.",
                            expert_config_count, self.num_experts)
                self.num_experts = expert_config_count
            self.expert_layers = [Module.from_config(config, name=f"expert_{idx}")
                                  for idx, config in enumerate(self.expert_config)]
        else:
            self.expert_layers = [Module.from_config(self.expert_config, name=f"expert_{idx}")
                                  for idx in range(self.num_experts)]

        # 构建门控层
        self._build_gate_layers(self.num_experts)

        super(MmoeFramework, self).build(input_shape)

    def call(self, inputs: Union[List[Union[List[List[tf.Tensor]], List[tf.Tensor]]], tf.Tensor], **kwargs) \
            -> Union[tf.Tensor, List[tf.Tensor]]:
        """ 执行MMOE

        :param inputs: 专家层的输入
        :param kwargs: 额外参数
        :return: MMOE输出

        :input shape:
           [batch_size, input_dim] or
           [expert_inputs, gate_inputs]
           - expert_inputs: 列表，长度=num_experts，每个元素为该专家的输入特征列表
            其中每个特征张量形状：[batch_size, feat_dim]
            拼接后每个专家的输入形状：[batch_size, sum_feat_dims]（sum_feat_dims为该专家所有特征维度之和）

           - gate_inputs: 列表，长度=num_tasks，每个元素为对应任务的gate输入张量
           每个gate输入张量形状：[batch_size, gate_input_dim_j]（gate_input_dim_j为第j个任务的gate输入维度）

        :output shape:
           如果 ``stack_output = True``, [batch_size, num_tasks, output_dim]；
           否则为列表格式，列表的长度为num_tasks，里面每个元素的维度是 [batch_size, output_dim]

        """
        # inputs shape: [batch_size, input_dim] or [list]
        # expert_outputs shape: [batch_size, num_expert, output_dim]
        if isinstance(inputs, list):
            expert_inputs, gate_inputs = inputs

            # 每个专家拼接自己的多特征并计算输出
            expert_output_list: List[tf.Tensor] = []

            # gate特征list concat
            gate_inputs = tf.concat(gate_inputs, axis=-1)
            for i, expert_feats in enumerate(expert_inputs):
                # 拼接当前专家的所有特征（最后一维拼接）
                expert_feat_concat = tf.concat(expert_feats, axis=-1)  # [batch_size, sum_feat_dims]

                # 专家层计算
                expert_out = self.expert_layers[i](expert_feat_concat)
                expert_output_list.append(expert_out)

        else:
            expert_output_list: List[tf.Tensor] = [layer(inputs) for layer in self.expert_layers]
            gate_inputs = inputs

        # 堆叠专家输出：[batch_size, num_experts, expert_output_dim]
        expert_outputs: tf.Tensor = tf.stack(expert_output_list, axis=1)

        outputs = super(Mmoe, self).call([gate_inputs, expert_outputs])

        return outputs


@Module.register("mmoe_v2", tags=["multi_task"])
class MmoeV2(Layer):
    """MMOE层

    相比于 :class:`Mmoe`，支持gate层配置，支持tower层配置。

    :register name: mmoe_v2

    :param expert_config: expert层配置，如果为list格式，则list的长度为expert层的数量，如果为dict/Config格式，则代表所有expert层的结构是一致的。
    :param gate_config: gate层配置，如果为list格式，则list的长度为gate层的数量，如果为dict/Config格式，则代表所有gate层的结构是一致的。
    :param tower_config: tower层配置，如果为list格式，则list的长度为tower层的数量，如果为dict/Config格式，则代表所有tower层的结构是一致的。
    :param num_tasks: 任务数，也指代gate的数量，也指代tower的数量。
    :param num_experts: 专家层数量，如果``expert_config``的类型是list，该参数无效。
    :param stack_outputs:
       是否对输出列表进行stack，如果为 ``True`` ，输出的维度是：[batch_size, num_tasks, output_dim]；
       如果为 ``False`` , 输出为一个列表，列表的长度为num_tasks，里面每个元素的维度是 [batch_size, output_dim]
    :param name: layer名字
    :param use_general_expert_output :表示是否使用通用专家网络输出信息
    :param use_general_extra_output :表示行业模块是否使用通用专家网络输出信息
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等



    """

    def __init__(self,
                 expert_config: Union[List[Config], Config],
                 gate_config: Union[List[Config], Config],
                 tower_config: Union[List[Config], Config],
                 num_tasks: int,
                 tower_extra_input,
                 poso_dim=0,
                 use_general_expert_output: bool = False,
                 use_general_extra_output: bool = False,
                 use_input_enhance: bool = False,
                 input_enhance_position: Optional[str] = None,
                 output_config: Union[List[Config], Config] = None,
                 poso_config: Optional[Config] = None,
                 num_experts: Optional[int] = None,
                 stack_outputs: bool = False,
                 use_tower_output: bool = False,
                 use_poso: bool = False,
                 name="mmoe_v2",
                 save_aux=False,
                 poso_with_ppnet=False,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.expert_config = expert_config
        self.gate_config = gate_config
        self.tower_config = tower_config
        self.output_config = output_config
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.stack_outputs = stack_outputs
        self.tower_extra_input = tower_extra_input
        self.use_input_enhance = use_input_enhance
        self.input_enhance_position = input_enhance_position
        self.use_tower_output = use_tower_output
        self.expert_layers: List[Layer] = []
        self.gate_layers: List[Layer] = []
        self.tower_layers: List[Layer] = []
        self.use_poso: bool = use_poso
        self.poso_config = poso_config
        self.output_layers: List[Layer] = []
        self.save_aux: bool = save_aux
        self.poso_dim = poso_dim
        self.poso_with_ppnet = poso_with_ppnet
        self.use_general_expert_output = use_general_expert_output
        self.use_general_extra_output = use_general_extra_output
        expert_config_count = len(self.expert_config)
        if isinstance(self.expert_config, list) and \
                (self.num_experts is None
                 or self.num_experts != expert_config_count):
            logger.info("Parameter num_expert is replaced with %s due to "
                        "expert config. Previous value was %s.",
                        expert_config_count, self.num_experts)
            self.num_experts = expert_config_count

        if isinstance(self.gate_config, list) and \
                (self.num_tasks != len(self.gate_config)):
            raise ConfigError(f"Parameter num_tasks must be equal with "
                              f"gate_config length.")
        if isinstance(self.tower_config, list) and \
                (self.num_tasks != len(self.tower_config)):
            raise ConfigError(f"Parameter num_tasks must be equal with "
                              f"tower_config length.")
        if self.use_poso and self.use_general_expert_output:
            logger.info("The general expert and poso information to industry expert is activated")


    def build(self, input_shape: tf.TensorShape):
        """构建内部layer和变量

        :param input_shape: 和 ``call`` 函数中的inputs结构相同，标记 ``call`` 函数中每个tensor的维度
        """

        if self.use_general_expert_output:
            logger.info("The general expert information to industry expert is activated")
            self.gate_config['hidden_dims'][1] = len(input_shape['general_expert_input']) + self.num_experts
        # 构建expert层
        self.expert_layers = self._build_layers_from_config(
            self.num_experts, self.expert_config, "expert"
        )

        # 构建gate层
        self.gate_layers = self._build_layers_from_config(
            self.num_tasks, self.gate_config, "gate"
        )

        # 构建tower层
        self.tower_layers = self._build_layers_from_config(
            self.num_tasks, self.tower_config, "tower"
        )
        if self.use_tower_output:
            # 构建output层
            self.output_layers = self._build_layers_from_config(
                self.num_tasks, self.output_config, "dense"
            )
        if self.use_poso:
            self.poso_layers = self._build_layers_from_config(
                self.num_tasks, self.poso_config, "poso"
            )
        super(MmoeV2, self).build(input_shape)

    @staticmethod
    def _build_layers_from_config(num_layers, layers_config, layer_type):
        if isinstance(layers_config, list):
            return_layers = [
                Module.from_config(config, name=f"{layer_type}_{idx}")
                for idx, config in enumerate(layers_config)
            ]
        else:
            return_layers = [
                Module.from_config(layers_config, name=f"{layer_type}_{idx}")
                for idx in range(num_layers)
            ]
        return return_layers

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs) -> \
            Union[tf.Tensor, List[tf.Tensor]]:
        """ 执行MMOE

        :param inputs: MMOE输入
        :param kwargs: 额外参数
        :return: MMOE输出

        :input shape:
           [batch_size, input_dim]

        :output shape:
           如果 ``stack_output = True``, [batch_size, num_tasks]；
           否则为列表格式，列表的长度为num_tasks，里面每个元素的维度是 [batch_size]

        """
        # inputs shape: [batch_size, input_dim]
        # expert_outputs shape: [batch_size, num_expert, output_dim]
        nn_inputs = inputs.get("bottom_input")
        extra_input = None
        enhance_input = None

        if self.tower_extra_input:
            extra_input = inputs.get("tower_extra_input")

        if self.use_input_enhance:
            enhance_input = inputs.get("enhance_input")

        if enhance_input is not None and self.input_enhance_position == "before_expert":
            nn_inputs = tf.concat([nn_inputs, enhance_input], axis=-1)

        expert_output_list: List[tf.Tensor] = [
            layer(nn_inputs) for layer in self.expert_layers
        ]
        if self.use_general_expert_output:
            general_expert_input = inputs.get("general_expert_input")
            extra_input_for_industry = [tf.stop_gradient(x) for x in general_expert_input]
            expert_output_list.extend(extra_input_for_industry)
        expert_outputs: tf.Tensor = tf.stack(expert_output_list, axis=-2)

        if self.use_poso:
            if self.use_general_expert_output:
                poso_gate_output_list = [tf.reshape(layer(extra_input),
                                                    [-1, self.num_experts + len(extra_input_for_industry),
                                                     self.poso_dim])
                                         for layer in self.poso_layers]
                poso_output_list: List[tf.Tensor] = [
                    poso_gate_output_list[index] * expert_outputs for index in range(len(poso_gate_output_list))
                ]
                poso_output = tf.stack(poso_output_list, axis=1)

            else:

                poso_gate_output_list = [tf.reshape(layer(extra_input), [-1, self.num_experts, self.poso_dim])
                                         for layer in self.poso_layers]
                poso_output_list: List[tf.Tensor] = [
                    poso_gate_output_list[index] * expert_outputs for index in range(len(poso_gate_output_list))
                ]
                poso_output = tf.stack(poso_output_list, axis=1)

        # gate_outputs shape: [batch_size, num_task, num_expert]
        gate_output_list: List[tf.Tensor] = [
            layer(nn_inputs) for layer in self.gate_layers
        ]
        gate_outputs: tf.Tensor = tf.stack(gate_output_list, axis=1)
        # bottom_outputs shape: [num_task, batch_size, output_dim]
        if self.use_poso:
            bottom_outputs: List[tf.Tensor] = tf.einsum('bne, bned -> bnd', gate_outputs, poso_output)
        else:
            bottom_outputs: List[tf.Tensor] = tf.matmul(
                gate_outputs, expert_outputs
            )
        bottom_outputs = tf.transpose(bottom_outputs, [1, 0, 2])

        if enhance_input is not None and self.input_enhance_position == "after_expert":
            enhance_input = tf.repeat(tf.expand_dims(enhance_input, axis=0), repeats=tf.shape(bottom_outputs)[0],
                                      axis=0)
            bottom_outputs = tf.concat([bottom_outputs, enhance_input], axis=-1)

        if self.poso_with_ppnet:
            tower_outputs: List[tf.Tensor] = [layer([bottom_outputs[index], extra_input])
                                              for index, layer in enumerate(self.tower_layers)]
        elif self.use_poso:
            tower_outputs = [layer(bottom_outputs[index]) for index, layer in enumerate(self.tower_layers)]
        elif self.tower_extra_input:
            # tower_outputs element shape: [batch_size]
            tower_outputs: List[tf.Tensor] = [layer([bottom_outputs[index], extra_input])
                                              for index, layer in enumerate(self.tower_layers)]
        else:
            tower_outputs = [layer(bottom_outputs[index]) for index, layer in enumerate(self.tower_layers)]

        if self.use_tower_output:
            outputs = [layer(tower_outputs[index]) for index, layer in enumerate(self.output_layers)]
        else:
            outputs: List[tf.Tensor] = tower_outputs

        if enhance_input is not None and self.input_enhance_position == "after_output":
            outputs = [tf.concat([output, enhance_input], axis=-1) for output in outputs]
        if self.use_general_extra_output:
            outputs.append(expert_output_list)
            return outputs
        if self.stack_outputs:
            if self.save_aux:
                return [tf.stack(outputs, axis=1), tf.unstack(bottom_outputs, axis=0)]
            # output shape: [batch_size, num_tasks]
            return tf.stack(outputs, axis=1)
        else:
            if self.save_aux:
                return [outputs, tf.unstack(bottom_outputs, axis=0)]

            return outputs


@Module.register("mmoe_v3", tags=["multi_task"])
class MmoeV3(Mmoe):
    """回传专家hidden output的mmoe
    输入和父类Mmoe一致
    新增return_expert_hidden_output，设置为'deep_out',字段解释同DNN
    要求专家网络支持回传hidden output，且结构为List[tf.Tensor]，比如DNN
    :output shape:
        [各个专家网络的隐藏层输出，mmoe输出]，类型为List[List[tf.Tensor]]
        [expert0_hidden_output,,,expertN_hidden_output,mmoe_output]
    """

    def __init__(self,
                 expert_config: Union[List[Config], Config],
                 num_tasks: int,
                 num_experts: Optional[int] = None,
                 norm: Union[str, Config, None] = None,
                 activation: Union[str, Config, None] = None,
                 dropout_rate: float = 0.0,
                 orders: str = "adn",
                 stack_outputs: bool = False,
                 gate_activation: Union[str, Config, None] = 'softmax',
                 gate_kernel_constraint: Union[str, Config, None] = None,
                 gate_kernel_initializer: Union[str, Config, None] = 'VarianceScaling',
                 gate_kernel_regularizer: Union[str, Config, None] = None,
                 use_gate_bias: bool = True,
                 gate_bias_constraint: Union[str, Config, None] = None,
                 gate_bias_regularizer: Union[str, Config, None] = None,
                 gate_bias_initializer: Union[str, Config, None] = 'zeros',
                 gate_activity_regularizer: Union[str, Config, None] = None,
                 name="mmoe_v3",
                 **kwargs):
        super().__init__(
            norm=norm,
            dropout_rate=dropout_rate,
            expert_config=expert_config,
            num_tasks=num_tasks,
            num_experts=num_experts,
            activation=activation,
            stack_outputs=stack_outputs,
            orders=orders,
            use_gate_bias=use_gate_bias,
            gate_activation=gate_activation,
            gate_activity_regularizer=gate_activity_regularizer,
            gate_kernel_initializer=gate_kernel_initializer,
            gate_bias_initializer=gate_bias_initializer,
            gate_kernel_regularizer=gate_kernel_regularizer,
            gate_bias_regularizer=gate_bias_regularizer,
            gate_kernel_constraint=gate_kernel_constraint,
            gate_bias_constraint=gate_bias_constraint,
            name=name,
            **kwargs)

        # 控制mmoe中专家DNN网络返回hidden output
        self.return_expert_hidden_output = 'deep_out'
        if isinstance(self.expert_config, list):
            for exp_cfg in self.expert_config:
                exp_cfg["return_hidden_output"] = self.return_expert_hidden_output
        else:
            self.expert_config["return_hidden_output"] = self.return_expert_hidden_output

    def build(self, input_shape: tf.TensorShape):
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> List[Union[tf.Tensor, List[tf.Tensor]]]:
        """ 执行MMOE

        :param inputs: 专家层的输入
        :param kwargs: 额外参数
        :return: 专家网络隐藏层输出、MMOE输出

        :input shape:
           [batch_size, input_dim]

        :output shape:
           [expert0_hidden_output,,,expertN_hidden_output,mmoe_output]

        """
        # inputs shape: [batch_size, input_dim]
        # expert_outputs shape: [batch_size, num_expert, output_dim]
        expert_output_list: List[List[tf.Tensor]] = [layer(inputs) for layer in self.expert_layers]
        expert_outputs: tf.Tensor = tf.stack([expert_output[-1] for expert_output in expert_output_list], axis=1)

        outputs = super(Mmoe, self).call([inputs, expert_outputs])
        expert_output_list.append(outputs)
        return expert_output_list


@Module.register("mmoelayer", tags=["multi_task"])
class MmoeLayer(Layer):
    """MMOELayer层

    实现MMOE的专家层、gate层结构。

    :register name: mmoelayer

    :param expert_config: expert层配置，如果为list格式，则list的长度为tower层的数量，如果为dict/Config格式，则代表所有tower层的结构是一致的。
    :param tower_config: tower层配置，如果为list格式，则list的长度为tower层的数量，如果为dict/Config格式，则代表所有tower层的结构是一致的。
    :param num_tasks: 任务数，也指代gate的数量，也指代tower的数量。
    :param num_experts: 专家层数量，如果``expert_config``的类型是list，该参数无效。
    :param name: layer名字
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 expert_config: Union[List[Config], Config],
                 gate_config: Union[List[Config], Config],
                 num_tasks: int,
                 num_experts: Optional[int] = None,
                 name="mmoelayer",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.expert_config = expert_config
        self.gate_config = gate_config
        self.num_tasks = num_tasks
        self.num_experts = num_experts

        self.expert_layers: List[Layer] = []
        self.gate_layers: List[Layer] = []

    def build(self, input_shape: tf.TensorShape):
        """构建内部layer和变量

        :param input_shape: 和 ``call`` 函数中的inputs结构相同，标记 ``call`` 函数中每个tensor的维度
        """
        # 构建expert层
        self.expert_layers = MmoeV2._build_layers_from_config(
            self.num_experts, self.expert_config, "expert"
        )
        # 构建gate层
        self.gate_layers = MmoeV2._build_layers_from_config(
            self.num_tasks, self.gate_config, "gate"
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> \
            Union[tf.Tensor, List[tf.Tensor]]:
        """ 执行MMOE layer

        :param inputs: MMOE输入
        :param kwargs: 额外参数
        :return: MMOE输出

        :input shape:
           [batch_size, input_dim]

        :output shape:
           否则为列表格式，列表的长度为num_tasks，里面每个元素的维度是 [batch_size, output_dim]

        """
        # inputs shape: [batch_size, input_dim]
        # expert_outputs shape: [batch_size, num_expert, output_dim]
        expert_output_list: List[tf.Tensor] = [
            layer(inputs) for layer in self.expert_layers
        ]
        expert_outputs: tf.Tensor = tf.stack(expert_output_list, axis=-2)

        # gate_outputs shape: [batch_size, num_task, num_expert]
        gate_output_list: List[tf.Tensor] = [
            layer(inputs) for layer in self.gate_layers
        ]
        gate_outputs: tf.Tensor = tf.stack(gate_output_list, axis=1)

        # bottom_outputs shape: [batch_size, num_task, output_dim]
        bottom_outputs: List[tf.Tensor] = tf.matmul(
            gate_outputs, expert_outputs
        )
        bottom_outputs = tf.transpose(bottom_outputs, [1, 0, 2])

        return bottom_outputs


@Module.register("multitask_model", tags=["multi_task"])
class MultitaskModel(Layer):
    """多任务框架层

    支持独立配置不同的专家层，tower层。

    :register name: multitask_model

    :param multi_expert_config: 多专家层配置，内部包含expert层、gate层配置。
    :param tower_config: tower层配置，如果为list格式，则list的长度为tower层的数量，如果为dict/Config格式，则代表所有tower层的结构是一致的。
    :param num_tasks: 任务数，也指代gate的数量，也指代tower的数量。
    :param num_experts: 专家层数量，如果``expert_config``的类型是list，该参数无效。
    :param stack_outputs:
        是否对输出列表进行stack，如果为 ``True`` ，输出的维度是：[batch_size, num_tasks, output_dim]；
        如果为 ``False`` , 输出为一个列表，列表的长度为num_tasks，里面每个元素的维度是 [batch_size, output_dim]
    :param name: layer名字
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 multi_expert_config: Union[List[Config], Config],
                 tower_config: Union[List[Config], Config],
                 num_tasks: int,
                 output_config: Union[List[Config], Config] = None,
                 stack_outputs: bool = False,
                 name="multitask",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.multi_expert_config = multi_expert_config
        self.tower_config = tower_config
        self.num_tasks = num_tasks
        self.stack_outputs = stack_outputs
        self.output_config = output_config
        self.expert_layers: List[Layer] = []
        self.tower_layers: List[Layer] = []

    def build(self, input_shape: tf.TensorShape):
        """构建内部layer和变量

        :param input_shape: 和 ``call`` 函数中的inputs结构相同，标记 ``call`` 函数中每个tensor的维度
        """
        # 构建expert层
        self.expert_layers = Module.from_config({**self.multi_expert_config, **{"num_tasks": self.num_tasks}})

        # 构建tower层
        self.tower_layers = MmoeV2._build_layers_from_config(
            self.num_tasks, self.tower_config, "tower")

        # 构建输出层
        if self.output_config:
            self.output_layers = MmoeV2._build_layers_from_config(
                self.num_tasks, self.output_config, "output"
            )

    @staticmethod
    def is_input_list_of_tensors(model):
        """
        判断模型的 call 方法的 inputs 参数是否为 List[tf.Tensor]。
        """
        # 获取 inputs 参数的类型注解
        signature = inspect.signature(model.call)
        inputs_param = signature.parameters[constants.INPUTS]
        input_type = inputs_param.annotation

        # 判断是否为 List[tf.Tensor]
        if isinstance(input_type, _GenericAlias):
            base_type = input_type.__origin__
            type_args = input_type.__args__
            if base_type in (list, List) and type_args[0] == tf.Tensor:
                return True
        return False

    def call(self, inputs, **kwargs) -> \
            Union[tf.Tensor, List[tf.Tensor]]:
        """
        :param inputs: 输入
        :param kwargs: 额外参数
        :return: MMOE输出

        :input shape:
           [[batch_size, input_dim][……]]

        :output shape:
           如果 ``stack_output = True``, [batch_size, num_tasks]；
           否则为列表格式，列表的长度为num_tasks，里面每个元素的维度是 [batch_size]

        """
        # inputs shape: [batch_size, input_dim]
        # bottom_outputs shape: [num_task, batch_size, output_dim]
        bottom_outputs = self.expert_layers(inputs)

        # tower_outputs element shape: [batch_size]
        tower_outputs: List[tf.Tensor] = \
            [layer([bottom_outputs[index], kwargs[constants.TOWER_EXTRA_INPUT]]
                   if constants.TOWER_EXTRA_INPUT in kwargs and MultitaskModel.is_input_list_of_tensors(layer)
                   else bottom_outputs[index])
             for index, layer in enumerate(self.tower_layers)]

        outputs: List[tf.Tensor] = tower_outputs
        if self.output_config:
            outputs = [
                layer(tower_outputs[index])
                for index, layer in enumerate(self.output_layers)
            ]

        if self.stack_outputs:
            # output shape: [batch_size, num_tasks]
            return tf.stack(outputs, axis=1)
        else:
            return outputs
