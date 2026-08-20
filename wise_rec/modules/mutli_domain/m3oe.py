#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from modelflow.integrations.tensorflow.custom_tf_ops.einsum_for_npu import custom_einsum_bij_bjik_bk
from modelflow.integrations.tensorflow.modules.module import Module
from modelflow.integrations.tensorflow.registrable import Config


def merge_masked_outputs(inputs_list: List[tf.Tensor],
                         indices_list: List[tf.Tensor]) -> tf.Tensor:
    """根据每行数据对应的原始输入的下标索引合并，保持与输入数据顺序对应

    :param inputs_list: Tensor列表
    :param indices_list: 每个Tensor列表对应的原始输入的下标
    :return: Tensor
    """
    merged_outputs = tf.concat(inputs_list, axis=0)
    merged_indices = tf.concat(indices_list, axis=0)
    positions = tf.argsort(merged_indices)
    outputs = tf.gather(merged_outputs, positions, axis=0)
    return outputs


def disentangle_fusion_expert(inputs: tf.Tensor,
                              domain_expert_list,
                              current_domain_index: int,
                              beta) -> tf.Tensor:
    """计算同一输入在不同域专家下输出的加权和

    param: inputs:输入
    param: domain_expert_list: 每个域的专家
    param: current_domain_index：当前域专家的下标
    param: beta: 当前域专家的融合权重
    """
    current_domain_expert_output = domain_expert_list[current_domain_index](inputs)
    expert_output = beta * current_domain_expert_output
    domain_num = len(domain_expert_list)
    for other_domain_index in range(domain_num):
        if current_domain_index != other_domain_index:
            other_domain_output = domain_expert_list[other_domain_index](inputs)
            expert_output += (1 - beta) / (domain_num - 1) * other_domain_output
    return expert_output


@Module.register("shared_expert_fusion")
class SharedExpertFusionLayer(layers.Layer):
    """Shared Expert Module

    实现M3oE模型中的Shared Expert Module
    * References：`M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework.
    <https://arxiv.org/pdf/2404.18465>`_

    :register name: shared_expert_fusion

    :config example:
    .. code-block:: jsonnet
      {
        name: "shared_expert",
        type: "shared_expert_fusion",
        parameters: {
          expert_config: {
            type: "dnn",
            orders: "nad",
            hidden_dims: [256],
            hidden_activation: "relu",
            output_activation: "relu",
            norm: { type: "layer_norm" },
            kernel_initializer: "he_uniform",
          },
          domain_index_list: ['1', '2', '3']
          expert_num: 5,
          task_num: 4,
        },
        inputs: {
          inputs: ["ref::concat_output.output", "ref::inputs.scenario"],
        },
        outputs: ["click_output", "play_output", "pay_output", "vip_output"],
      },
    :param expert_config: 专家层配置
    :param domain_index_list: 域标识的编码列表
    :param expert_num: 专家数量
    :param task_num: 任务数量
    :param name: 层名称
    :param use_custom_einsum: 模型内部使用的tf.einsum是否使用替代方案

    :param kwargs: 额外参数
    """

    def __init__(self,
                 expert_config: Config,
                 domain_index_list: List[str],
                 expert_num: int = 1,
                 task_num: int = 2,
                 name="shared_expert_fusion",
                 use_custom_einsum=False,
                 **kwargs):
        self.expert_config = expert_config
        self.domain_index_list = [np.int64(num) for num in domain_index_list]
        self.expert_num = expert_num
        self.domain_num = len(self.domain_index_list)
        self.task_num = task_num
        self.expert_list = []
        self.gate_list = []
        self.gate_num = self.domain_num * task_num
        self.use_custom_einsum = use_custom_einsum
        super(SharedExpertFusionLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.expert_list = [Module.from_config(self.expert_config) for _ in range(self.expert_num)]

        for gate_index in range(self.gate_num):
            self.gate_list.append(layers.Dense(units=self.expert_num,
                                               name="shared_expert_gate_{}".format(gate_index),
                                               activation="softmax"))

    def call(self, inputs: List[tf.Tensor], **kwargs) -> List[tf.Tensor]:
        """ 执行shared_expert_fusion

        :param inputs: 输入tensor列表，维度是[(batch_size, feature_size), (batch_size,1)]，分别是输入特征和域标识
        :param kwargs: 额外参数
        :return: 结果，维度是[(batch_size, output_dim), (batch_size, output_dim)...]，每个任务的输出表征
        """
        features, domain_indicator = inputs
        indices = tf.range(tf.shape(features)[0], dtype=tf.int32)
        outputs_list = []
        indices_list = []
        for domain_rank, domain_index in enumerate(self.domain_index_list):
            scenario_mask = tf.squeeze(tf.equal(domain_indicator, domain_index), axis=1)
            masked_inputs = tf.boolean_mask(features, scenario_mask, axis=0)
            masked_indices = tf.boolean_mask(indices, scenario_mask)

            # 计算并合并多个专家的输出
            expert_outputs = tf.stack([self.expert_list[idx](masked_inputs) for idx in range(self.expert_num)], axis=1)

            # 维度： [batch_size, expert_num, 1, output_dim]
            expand_expert_outputs = tf.expand_dims(expert_outputs, axis=2)

            # 根据每个单独的gate，计算输出
            task_outputs = []
            for task_index in range(self.task_num):
                gate_idx = domain_rank * self.task_num + task_index
                gate_output = self.gate_list[gate_idx](masked_inputs)
                expand_gate_output = tf.expand_dims(gate_output, axis=1)
                if self.use_custom_einsum:
                    task_output = custom_einsum_bij_bjik_bk(expand_gate_output, expand_expert_outputs)
                else:
                    task_output = tf.einsum("bij,bjik->bk", expand_gate_output, expand_expert_outputs)
                task_outputs.append(task_output)

            # 维度：[batch_size, task_num, output_dim]
            domain_output = tf.stack(task_outputs, axis=1)
            outputs_list.append(domain_output)
            indices_list.append(masked_indices)

        # 合并多个域的输出，并按照原始的输入位置对齐
        outputs = merge_masked_outputs(outputs_list, indices_list)
        split_outputs = tf.split(value=outputs, num_or_size_splits=self.task_num, axis=1)
        squeeze_split_output = [tf.squeeze(out, axis=1) for out in split_outputs]
        return squeeze_split_output

    def get_config(self):
        config = super().get_config()
        config["expert_config"] = self.expert_config
        config["domain_index_list"] = self.domain_index_list
        config["expert_num"] = self.expert_num
        config["task_num"] = self.task_num
        return config


@Module.register("domain_expert_fusion")
class DomainExpertFusionLayer(layers.Layer):
    """Domain Expert Module

    实现M3oE模型中的Domain Expert Module
    * References：`M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework.
    <https://arxiv.org/pdf/2404.18465>`_

    :register name: domain_expert_fusion

    :config example:
    .. code-block:: jsonnet
      {
        name: "domain_expert",
        type: "domain_expert_fusion",
        parameters: {
          expert_config: {
            type: "dnn",
            orders: "nad",
            hidden_dims: [512, 256],
            hidden_activation: "relu",
            output_activation: "relu",
            norm: { type: "layer_norm" },
            kernel_initializer: "he_uniform",
          },
          domain_index_list: ['1', '2', '3'],
        },
        inputs: {
          inputs: ["ref::concat_output.output", "ref::inputs.scenario"],
        },
        outputs: "output",
      },
    :param expert_config: 专家配置
    :param domain_index_list: 域标识特征编码列表
    :param name: 层名称
    :param kwargs: 额外参数
    """

    def __init__(self,
                 expert_config: Config,
                 domain_index_list: List[str],
                 name="domain_expert_fusion",
                 **kwargs):
        self.expert_config = expert_config
        self.domain_index_list = [np.int64(num) for num in domain_index_list]
        self.domain_num = len(self.domain_index_list)
        self.expert_list = []
        self.beta = None
        super(DomainExpertFusionLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.expert_list = [Module.from_config(self.expert_config) for _ in range(self.domain_num)]
        self.beta = self.add_weight(name=f"beta",
                                    initializer=tf.zeros_initializer,
                                    shape=(1,),
                                    dtype="float32",
                                    trainable=True)

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """ 执行domain_expert_fusion

        :param inputs: 输入tensor列表，维度是[(batch_size, feature_size), (batch_size,1)]，分别是输入特征和域标识
        :param kwargs: 额外参数
        :return: 结果，维度是(batch_size, output_dim)
        """
        features, domain_indicator = inputs

        # 规避NPU训练空tensor引发梯度nan值异常，为每个域新增一个全0样本（仅参与计算过程，输出时会被过滤）
        raw_indices = tf.range(tf.shape(features)[0], dtype=tf.int32)   # 原始样本的下标位置
        indices = tf.range(tf.shape(features)[0] + self.domain_num, dtype=tf.int32)  # 新增样本后的下标位置
        zero_vectors = tf.zeros(shape=(self.domain_num, tf.shape(features)[1]))  # 新增的全0样本
        features = tf.concat([features, zero_vectors], axis=0)  # 样本特征拼接
        domain_indicator = tf.concat([domain_indicator, tf.reshape(self.domain_index_list, (-1, 1))], axis=0) # 域id拼接

        outputs_list = []
        indices_list = []
        for domain_rank, current_domain_index in enumerate(self.domain_index_list):
            # 根据场景id筛选输入样本，和对应下标位置
            scenario_mask = tf.squeeze(tf.equal(domain_indicator, current_domain_index), axis=1)
            masked_inputs = tf.boolean_mask(features, scenario_mask, axis=0)
            masked_indices = tf.boolean_mask(indices, scenario_mask)

            beta = tf.sigmoid(self.beta)
            expert_output = disentangle_fusion_expert(masked_inputs, self.expert_list, domain_rank, beta)

            outputs_list.append(expert_output)
            indices_list.append(masked_indices)

        # 合并多个域的输出，并按照原始的输入位置对齐
        outputs = merge_masked_outputs(outputs_list, indices_list)

        # 规避NPU训练空tensor引发梯度nan值异常，过滤新增全0样本的计算结果，仅保留原始输入样本的计算结果
        outputs = tf.gather(outputs, raw_indices, axis=0)

        return outputs

    def get_config(self):
        config = super().get_config()
        config["expert_config"] = self.expert_config
        config["domain_index_list"] = self.domain_index_list
        return config


@Module.register("task_expert_fusion")
class TaskExpertFusionLayer(layers.Layer):
    """Task Expert Module

    实现M3oE模型中的Task Expert Module
    * References：`M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework.
    <https://arxiv.org/pdf/2404.18465>`_

    :register name: task_expert_fusion

    :config example:
    .. code-block:: jsonnet
      {
        name: "task_expert",
        type: "task_expert_fusion",
        parameters: {
          expert_config: {
            type: "dnn",
            orders: "nad",
            hidden_dims: [512, 256],
            hidden_activation: "relu",
            output_activation: "relu",
            norm: { type: "layer_norm" },
            kernel_initializer: "he_uniform",
          },
          task_num: 4,
        },
        inputs: {
          inputs: "ref::concat_output.output",
        },
        outputs: ["click_output", "play_output", "pay_output", "vip_output"],
      },
    :param expert_config: 专家配置
    :param task_num: 任务数量
    :param name: 层名称
    :param kwargs: 额外参数
    """

    def __init__(self,
                 expert_config: Config,
                 task_num: int,
                 name="task_expert_fusion",
                 **kwargs):
        self.expert_config = expert_config
        self.task_num = task_num
        self.expert_list = []
        self.beta = None
        super(TaskExpertFusionLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.expert_list = [Module.from_config(self.expert_config) for _ in range(self.task_num)]
        self.beta = self.add_weight(name=f"beta",
                                    initializer=tf.zeros_initializer,
                                    shape=(1,),
                                    dtype="float32",
                                    trainable=True)

    def call(self, inputs: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        """ 执行task_expert_fusion

        :param inputs: 输入tensor，维度是(batch_size, input_dim)
        :param kwargs: 额外参数
        :return: 结果，维度是[(batch_size, output_dim)],长度为任务数量
        """
        outputs = []
        for cur_task_idx in range(self.task_num):
            beta = tf.sigmoid(self.beta)
            fusion_task_output = disentangle_fusion_expert(inputs, self.expert_list, cur_task_idx, beta)
            outputs.append(fusion_task_output)
        return outputs

    def get_config(self):
        config = super().get_config()
        config["expert_config"] = self.expert_config
        config["task_num"] = self.task_num
        return config


@Module.register("multi_view_balancing")
class MultiViewBalancingLayer(layers.Layer):
    """Multi-View Representation Balancing Module

    实现M3oE模型中的Multi-View Representation Balancing Module
    * References：`M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework.
    <https://arxiv.org/pdf/2404.18465>`_

    :register name: multi_view_balancing

    :config example:
    .. code-block:: jsonnet
      {
        name: "click_fusion",
        type: "multi_view_balancing",
        inputs: {
          inputs: [
            "ref::shared_expert.click_output",
            "ref::domain_expert.output",
            "ref::task_expert.click_output",
          ],
        },
        outputs: "output",
      },
    :param name: 层名称
    :param kwargs: 额外参数
    """

    def __init__(self,
                 name="multi_view_balancing", **kwargs):
        self.domain_alpha = None
        self.task_alpha = None
        super(MultiViewBalancingLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.domain_alpha = self.add_weight(name=f"domain_alpha",
                                            initializer=tf.zeros_initializer,
                                            shape=(1,),
                                            dtype="float32",
                                            trainable=True)
        self.task_alpha = self.add_weight(name=f"task_alpha",
                                          initializer=tf.zeros_initializer,
                                          shape=(1,),
                                          dtype="float32",
                                          trainable=True)

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """ 执行multi_view_balancing

        :param inputs: 输入tensor列表，维度是[(batch_size, input_dim), (batch_size, input_dim), (batch_size, input_dim)]，
        分别表示：共享模块、域融合模块、任务融合模块的输出
        :param kwargs: 额外参数
        :return: 结果，维度是(batch_size, input_dim)
        """
        shared_inputs, domain_inputs, task_inputs = inputs
        domain_alpha = tf.sigmoid(self.domain_alpha)
        task_alpha = tf.sigmoid(self.task_alpha)
        merged_outputs = shared_inputs + domain_alpha * domain_inputs + task_alpha * task_inputs
        return merged_outputs


@Module.register("domain_route_tower")
class DomainRouteTowerLayer(layers.Layer):
    """Domain route tower layer

    实现根据样本的域id对训练样本进行路由，保证来自不同域下的样本只训练对应域的塔

    :register name: domain_route_tower

    :config example:
    .. code-block:: jsonnet
      {
        name: "domain_vip_pay",
        type: "domain_route_tower",
        parameters: {
          domain_index_list: ['1', '2', '3', '4', '5'],
          tower_config: {
            type: "dnn",
            hidden_dims: [256],
            kernel_initializer: "truncated_normal",
            output_activation: "relu",
          },
        },
        inputs: {
          inputs: ["ref::mmoe_out.vip_output", "ref::inputs.scenario"],
        },
        outputs: "output",
      }

    :param tower_config: 每个塔的配置
    :param domain_index_list: 域标识特征的编码列表
    :param name: 层名称
    :param kwargs: 额外参数
    """

    def __init__(self,
                 tower_config: Config,
                 domain_index_list: List[str],
                 name: str = 'domain_route_tower', **kwargs):
        self.tower_config = tower_config
        self.domain_index_list = [np.int64(num) for num in domain_index_list]
        self.domain_num = len(domain_index_list)
        self.tower_list = []
        super(DomainRouteTowerLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        for _ in range(self.domain_num):
            tower = Module.from_config(self.tower_config)
            self.tower_list.append(tower)

    def call(self, inputs, **kwargs):
        """ domain_route_tower

        :param inputs: 输入tensor列表，维度是[(batch_size, feature_size), (batch_size,1)]，分别为全局特征和域id特征
        :param kwargs: 额外参数
        :return: 结果，维度是(batch_size, self.output_dim)
        """
        feature, domain_indicator = inputs

        # 规避NPU训练空tensor引发梯度nan值异常，为每个域新增一个全0样本（仅参与计算过程，输出时会被过滤）
        raw_indices = tf.range(tf.shape(feature)[0], dtype=tf.int32)   # 原始样本的下标位置
        indices = tf.range(tf.shape(feature)[0] + self.domain_num, dtype=tf.int32)  # 新增样本后的下标位置
        zero_vectors = tf.zeros(shape=(self.domain_num, tf.shape(feature)[1]))  # 新增的全0样本
        feature = tf.concat([feature, zero_vectors], axis=0)  # 样本特征拼接
        domain_indicator = tf.concat([domain_indicator, tf.reshape(self.domain_index_list, (-1, 1))], axis=0) # 域id拼接

        outputs_list = []
        indices_list = []
        for num, index in enumerate(self.domain_index_list):
            # 根据域id筛选输入样本，和对应下标位置
            domain_mask = tf.squeeze(tf.equal(domain_indicator, index), axis=1)
            masked_inputs = tf.boolean_mask(feature, domain_mask, axis=0)
            masked_indices = tf.boolean_mask(indices, domain_mask)

            # 计算该域的输出
            tower_output = self.tower_list[num](masked_inputs)
            outputs_list.append(tower_output)
            indices_list.append(masked_indices)

        # 合并多个域的输出，并按照原始的输入位置对齐
        outputs = merge_masked_outputs(outputs_list, indices_list)

        # 规避NPU训练空tensor引发梯度nan值异常，过滤新增全0样本的计算结果，仅保留原始输入样本的计算结果
        outputs = tf.gather(outputs, raw_indices, axis=0)

        return outputs

    def get_config(self):
        config = super().get_config()
        config["tower_config"] = self.tower_config
        config["domain_index_list"] = self.domain_index_list
        return config


@Module.register("domain_balancing")
class DomainBalancingLayer(layers.Layer):
    """Domain Representation Balancing Module

    单任务场景，只包括域共享和域特定，不包含任务
    * References：`M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework.
    <https://arxiv.org/pdf/2404.18465>`_

    :register name: domain_balancing

    :config example:
    .. code-block:: jsonnet
      {
        name: "domain_fusion",
        type: "domain_balancing",
        inputs: {
          inputs: [
            "ref::shared_expert.click_output",
            "ref::domain_expert.output"
          ],
        },
        outputs: "output",
      },
    :param name: 层名称
    :param kwargs: 额外参数
    """

    def __init__(self,
                 name="multi_view_balancing", **kwargs):
        self.domain_alpha = None
        super(DomainBalancingLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.domain_alpha = self.add_weight(name=f"domain_alpha",
                                            initializer=tf.zeros_initializer,
                                            shape=(1,),
                                            dtype="float32",
                                            trainable=True)

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """ 执行multi_view_balancing

        :param inputs: 输入tensor列表，维度是[(batch_size, input_dim), (batch_size, input_dim)]，
        分别表示：共享模块、域融合模块的输出
        :param kwargs: 额外参数
        :return: 结果，维度是(batch_size, input_dim)
        """
        shared_inputs, domain_inputs = inputs
        domain_alpha = tf.sigmoid(self.domain_alpha)
        merged_outputs = shared_inputs + domain_alpha * domain_inputs
        return merged_outputs
