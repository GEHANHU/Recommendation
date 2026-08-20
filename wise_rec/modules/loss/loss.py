#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import json
from typing import List, Dict
from typing import Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from modelflow.common import Config
from modelflow.integrations.tensorflow.modules import Module


@Module.register("loss_mask", tags=["loss"])
class LossMask(Layer):
    """ Loss Mask layer


    :register name: loss_mask

    :param mask_config: mask配置。
    :param feature_map_path: 特征map文件路径。
    :param num_tasks: 任务数量。
    :param name: layer名字
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等


    """

    def __init__(self,
                 mask_config: Config,
                 feature_map_path: str,
                 num_tasks: int,
                 name="loss_mask",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_tasks = num_tasks
        self.mask_dict = mask_config["mask_dict"]
        self.masked_leak_values = mask_config["masked_leak_values"]
        self.feature_map_path = feature_map_path
        self.mask_groups = None

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        self.mask_groups = self.gen_mask_groups(
            self.feature_map_path, self.mask_dict
        )
        super(LossMask, self).build(input_shape)

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs) -> \
            Union[tf.Tensor, List[tf.Tensor]]:
        if self.check_param(inputs):
            loss_mask = self.gen_loss_mask(
                self.num_tasks, inputs, self.mask_groups,
                self.masked_leak_values
            )
        else:
            loss_mask = [tf.ones([])]
        return loss_mask

    def check_param(self, inputs):
        check_result = True
        for task_feature_name in self.mask_groups.keys():
            if task_feature_name not in inputs:
                check_result = False
                break
        return check_result

    @staticmethod
    def gen_mask_groups(feature_map_path, mask_dict):
        """通过feature map将mask_dict中的原始值映射成map后的值"""

        return_groups = {}
        for feature_name, feature_values in mask_dict.items():
            feature_map = LossMask.get_feature_map_dict(
                feature_map_path, feature_name
            )

            mapping_values = [
                LossMask.get_mapping_list(feature_value, feature_map)
                for feature_value in feature_values
            ]
            return_groups[feature_name] = mapping_values
        return return_groups

    @staticmethod
    def gen_loss_mask(task_num, features, mask_groups, masked_leak_values):
        """获取loss mask"""

        loss_mask = []
        for index in range(task_num):
            task_mask = []
            for task_feature_name, spec_tower in mask_groups.items():
                # 输入的特征 Tensor
                task_raw_features = features[task_feature_name]
                # 对应塔 需要的 feat_index
                used_mask_group = spec_tower[index]
                if len(used_mask_group) == 0:
                    group_mask = tf.ones_like(
                        task_raw_features, dtype=tf.float32
                    )
                    task_mask.append(group_mask)
                    continue

                group_constant = tf.constant(used_mask_group, dtype=tf.int64)
                group_mask = tf.cast(
                    tf.equal(task_raw_features, group_constant),
                    dtype=tf.float32
                )
                if len(used_mask_group) > 1:
                    group_mask = tf.expand_dims(
                        tf.reduce_sum(group_mask, axis=-1), axis=-1
                    )
                task_mask.append(group_mask)

            if len(task_mask) > 1:
                task_mask = tf.concat(task_mask, axis=1)
                task_mask = tf.reduce_max(task_mask, axis=1, keepdims=True)
            else:
                task_mask = task_mask[0]
            task_mask = (1 - task_mask) * masked_leak_values[index] + task_mask
            loss_mask.append(task_mask)
        return loss_mask

    @staticmethod
    def get_feature_map_dict(feature_map_path: str,
                             feature_name: str) -> dict:
        """获取特征feature_name的feature map"""

        with open(feature_map_path, 'r') as f:
            feature_map_data = json.load(f)

        feature_index_dict = {}
        for feat_val, _id in feature_map_data.items():
            arr = feat_val.split(",")
            if len(arr) == 2 and arr[0] == feature_name:
                (_, val) = arr
                feature_index_dict[val] = int(_id)
        return feature_index_dict

    @staticmethod
    def get_mapping_list(ori_list: list, mapping_dict: dict) -> dict:
        """获取基于feature map映射后的列表"""

        mapping_list = []
        for ori_value in ori_list:
            mapping_index = mapping_dict.get(ori_value, -1)
            if mapping_index == -1:
                raise KeyError('feat map cannot be found')

            mapping_list.append(mapping_index)
        return mapping_list





#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.


import tensorflow as tf
from tensorflow.keras.layers import Layer
from modelflow.integrations.tensorflow.modules.module import Module


@Module.register("resn_loss", tags=["loss"])
class ReSNLossLayer(Layer):
    """ReSN Layer(流行度偏差)实现
      :register name: resn_loss
      :param lam: 计算中间loss的系数
      :param reglam: L2正则的系数
      :param loss_weight: 对任务的loss乘上一个权重weight
      :param method: user_emb和item_emb的计算方式，multiply/matmul
      :param kwargs: 其他参数
      :config example:
      {
        name: 'resn_layer',
        type: 'resn_loss',
        parameters: {
            lam: 1e-9,
            reglam: 1e-9,
            loss_weight: 0.05
        },
        inputs: {
            inputs: ['ref::eta.embedding_others.u_activity_new',
            'ref::eta.embedding_others.u_click_sids',
            'ref::eta.embedding_others.u_sf']
        },
        outputs: 'resn_loss'
    }
    """

    def __init__(self, lam=1e-5, reglam=1e-6, loss_weight=1.0, method='multiply', **kwargs):
        self.tile_num = None
        self.emb_size = None
        self.lam = float(lam)
        self.reglam = float(reglam)
        self.loss_weight = float(loss_weight)
        self.method = method
        super(ReSNLossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        user_shape, pos_item_shape, neg_item_shape = input_shape
        self.tile_num = pos_item_shape[1]
        self.emb_size = pos_item_shape[-1]
        super(ReSNLossLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        user_emb, pos_item_emb, neg_item_emb = inputs
        user_tiled = tf.tile(user_emb, multiples=[1, self.tile_num, 1])
        user_reshape = tf.reshape(user_tiled, [-1, self.emb_size])
        pos_item_reshape = tf.reshape(pos_item_emb, [-1, self.emb_size])
        neg_item_reshape = tf.reshape(neg_item_emb, [-1, self.emb_size])
        all_item = tf.concat([pos_item_emb, neg_item_emb], axis=1)
        cur_user = tf.tile(user_emb, multiples=[1, self.tile_num * 2, 1])
        if self.method == "matmul":
            cur_user_reshape = tf.reshape(cur_user, [-1, self.emb_size])
            all_item_reshape = tf.reshape(all_item, [-1, self.emb_size])
            matrix_ = tf.sigmoid(tf.matmul(cur_user_reshape, tf.transpose(all_item_reshape)))
            user_vector_emb = tf.expand_dims(tf.reduce_sum(matrix_, axis=0), 1)
            q = tf.stop_gradient(user_vector_emb / tf.sqrt(tf.reduce_sum(tf.square(user_vector_emb))))
            regterm = tf.matmul(matrix_, q)
        else:
            cur_user_reshape = tf.reshape(cur_user, [-1, self.emb_size * self.tile_num * 2])
            all_item_reshape = tf.reshape(all_item, [-1, self.emb_size * self.tile_num * 2])
            matrix_ = tf.sigmoid(tf.multiply(cur_user_reshape, all_item_reshape))
            user_vector_emb = tf.expand_dims(tf.reduce_sum(matrix_, axis=1), 1)
            q = tf.stop_gradient(user_vector_emb / tf.sqrt(tf.reduce_sum(tf.square(user_vector_emb))))
            regterm = tf.multiply(matrix_, q)
        pos_preds = tf.expand_dims(tf.reduce_sum(tf.multiply(user_reshape, pos_item_reshape), 1), 1)
        neg_preds = tf.expand_dims(tf.reduce_sum(tf.multiply(user_reshape, neg_item_reshape), 1), 1)
        pos_preds_sig = tf.sigmoid(pos_preds)
        neg_preds_sig = tf.sigmoid(neg_preds)
        regloss = tf.reduce_sum(tf.square(regterm))
        weighted_mse = tf.reduce_sum(tf.square(1 - pos_preds_sig)) + tf.reduce_sum(tf.square(neg_preds_sig))
        reg_term_embeds = tf.nn.l2_loss(user_emb) + tf.nn.l2_loss(pos_item_emb) + tf.nn.l2_loss(neg_item_emb)
        resn_loss = (weighted_mse + self.lam * reg_term_embeds + self.reglam * regloss) * self.loss_weight
        return resn_loss

#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
from typing import List
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from modelflow.common import Config
from modelflow.common.config import convert_config_to_dict
from modelflow.common.exception import ConfigError
from modelflow.data.feature_manager import FeatureManager
from modelflow.integrations.tensorflow.modules.base import FIFOLayer
from modelflow.integrations.tensorflow.modules.module import Module


class SampledSoftmaxLayer(layers.Layer):
    """Sampled Softmax Layer实现

    :register name: sampled_softmax

    :param num_sampled: 随机负样本的个数
    :param num_true: 每个样本的target类别数，默认为1
    :param temperature: 温度系数
    :param name: layer名
    :param kwargs:
    """

    def __init__(self,
                 num_sampled: int,
                 num_true: int = 1,
                 temperature: float = 1.0,
                 name: str = "sampled_softmax",
                 **kwargs):
        super(SampledSoftmaxLayer, self).__init__(name=name, **kwargs)
        self.num_sampled = num_sampled
        self.num_true = num_true
        self.temperature = temperature

        # Init values
        self.vocabulary_size: int = 0
        self.weight = None
        self.bias = None

    def build(self, input_shape):
        self.vocabulary_size = input_shape[0][0]
        self.bias = self.add_weight(shape=[self.vocabulary_size],
                                    initializer="zeros",
                                    dtype=tf.float32,
                                    trainable=False,
                                    name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs: List[tf.Tensor], training=None, **kwargs) -> tf.Tensor:
        """ 调用SampledSoftmaxLayer

        :param inputs: 三个Tensor，维度分别为[vocabulary_size,output_dim]、[batch_size,units]和[batch_size,self.num_true]
        :param training: 训练阶段
        :param kwargs: 额外参数
        :return: 每个样本的loss值，维度为[batch_size,1]
        """
        item_embeddings, model_outputs, labels = inputs
        if labels.dtype != tf.int64:
            labels = tf.cast(labels, tf.int64)
        model_outputs = tf.divide(model_outputs, self.temperature)

        loss = tf.nn.sampled_softmax_loss(weights=item_embeddings,
                                          biases=self.bias,
                                          labels=labels,
                                          inputs=model_outputs,
                                          num_sampled=self.num_sampled,
                                          num_true=self.num_true,
                                          num_classes=self.vocabulary_size)
        return tf.expand_dims(loss, axis=1)


@Module.register("sampled_softmax_with_weight", tags=["loss"])
class SampledSoftmaxWithWeightLayer(SampledSoftmaxLayer):
    """Sampled Softmax Layer
    带权重的Sampled Softmax Layer，权重即为item embedding

    :register name: sampled_softmax_with_weight

    :param units: embedding维度
    :param item_feature_name: 特征名，用于从feature_manager中获取该特征的vocab_size
    :param feature_manager: 特征管理器，用于获取特征维度信息，**在配置项中无需配置**
    :param num_sampled: 随机负样本的个数
    :param num_true: 每个样本的target类别数，默认为1
    :param weight_initializer: 权重初始化器类型
    :param temperature: 温度系数，默认为1
    :param l2_norm: 是否l2正则化
    :param sampler: 采样类型。支持配置 `uniform`, `adaptive`。 如果为 `None`, 表示使用log_uniform_candidate_sampler
    :param distortion: 用于扭曲unigram概率分布.在添加到内部unigram分布之前，首先将每个权重提升到失真的幂。
    distortion = 1.0按照item频率采样，当distortion = 0.0按照均匀分布采样。
    :param name: 层名
    :param kwargs: 额外参数
    """

    def __init__(self,
                 units: int,
                 item_feature_name: str,
                 feature_manager: FeatureManager,
                 num_sampled: int,
                 num_true: int = 1,
                 weight_initializer: Union[str, Config, None] = "glorot_uniform",
                 weight_regularizer: Union[str, Config, None] = None,
                 weight_constraint: Union[str, Config, None] = None,
                 temperature: float = 1.0,
                 l2_norm: bool = False,
                 sampler: Union[str, None] = None,
                 distortion: float = 1.0,
                 name: str = "sampled_softmax_with_weight",
                 **kwargs):
        super(SampledSoftmaxWithWeightLayer, self).__init__(num_sampled=num_sampled,
                                                            num_true=num_true,
                                                            temperature=temperature,
                                                            name=name,
                                                            **kwargs)
        self.units = units
        self.item_feature_name = item_feature_name
        self.weight_initializer = convert_config_to_dict(weight_initializer)
        self.weight_regularizer = convert_config_to_dict(weight_regularizer)
        self.weight_constraint = convert_config_to_dict(weight_constraint)
        self.feature_manager = feature_manager
        self.l2_norm = l2_norm
        self.sampler = sampler

        # Init values
        self.vocabulary_size: int = self.feature_manager.feature_dict[self.item_feature_name].count
        self.weight = None
        self.bias = None
        self.distortion = distortion
        if self.sampler == "frequency":
            self.item_frequency_list = self.feature_manager.get_feature_frequency_list(self.item_feature_name)

    def build(self, input_shape):
        """构建SampledSoftmaxWithWeightLayer

        :param input_shape: 输入的shape
        """
        self.weight = self.add_weight(shape=[self.vocabulary_size, self.units],
                                      initializer=self.weight_initializer,
                                      regularizer=self.weight_regularizer,
                                      constraint=self.weight_constraint,
                                      dtype=tf.float32,
                                      name="weight")
        self.bias = self.add_weight(shape=[self.vocabulary_size],
                                    initializer="zeros",
                                    dtype=tf.float32,
                                    trainable=False,
                                    name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """调用SampledSoftmaxWithWeightLayer层

        :param inputs: 两个tensor组成的list，维度分别为[batch_size,self.unit]和[batch_size,self.num_true]
        :param kwargs: 额外参数
        :return: 每个样本的loss值，维度为[batch_size,1]
        """
        model_outputs, labels = inputs
        if labels.dtype != tf.int64:
            labels = tf.cast(labels, tf.int64)
        model_outputs = tf.divide(model_outputs, self.temperature)

        norm_weight = self.weight
        if self.l2_norm:
            norm_weight = tf.nn.l2_normalize(norm_weight, axis=-1)

        if self.sampler == "uniform":
            sampled_values = tf.random.uniform_candidate_sampler(true_classes=labels,
                                                                 num_true=self.num_true,
                                                                 num_sampled=self.num_sampled,
                                                                 unique=True,
                                                                 range_max=self.vocabulary_size)
        elif self.sampler == "adaptive":
            sampled_values = tf.nn.learned_unigram_candidate_sampler(true_classes=labels,
                                                                     num_true=self.num_true,
                                                                     num_sampled=self.num_sampled,
                                                                     unique=True,
                                                                     range_max=self.vocabulary_size)
        elif self.sampler == "frequency":
            if not self.item_frequency_list:
                raise ConfigError(f"Feature frequency of {self.item_feature_name} is empty. Please check it. ")
            sampled_values = tf.nn.fixed_unigram_candidate_sampler(true_classes=labels,
                                                                   num_true=self.num_true,
                                                                   num_sampled=self.num_sampled,
                                                                   unique=True,
                                                                   range_max=self.vocabulary_size,
                                                                   distortion=self.distortion,
                                                                   unigrams=self.item_frequency_list)
        else:
            sampled_values = None

        loss = tf.nn.sampled_softmax_loss(weights=norm_weight,
                                          biases=self.bias,
                                          labels=labels,
                                          inputs=model_outputs,
                                          num_sampled=self.num_sampled,
                                          num_classes=self.vocabulary_size,
                                          num_true=self.num_true,
                                          sampled_values=sampled_values)
        return tf.expand_dims(loss, axis=1)


@Module.register("in_batch_softmax", tags=["loss"])
class InBatchSoftmaxLayer(layers.Layer):
    """InBatch Softmax Layer

    :register name: in_batch_softmax

    :param item_feature_name: 特征名，用于从feature_manager中获取该特征的vocab_size
    :param feature_manager: 特征管理器，用于获取特征维度信息，**在配置项中无需配置**
    :param temperature: 温度系数，默认为1
    :param alpha: 使用采样频率修正的loss项的系数
    :param identify_same_user: batch内其他样本作为负样本时，是否剔除来自同一用户的正样本
    :param same_user_threshold: 判断是否是同一个用户embedding的阈值，计算方式为两个向量的曼哈顿距离
    :param name: 层名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 item_feature_name: str,
                 feature_manager: FeatureManager,
                 temperature: float = 1.0,
                 alpha: float = 1.0,
                 identify_same_user: bool = False,
                 same_user_threshold: float = 0.0,
                 name="in_batch_softmax",
                 **kwargs):
        super(InBatchSoftmaxLayer, self).__init__(name=name, **kwargs)
        self.item_feature_name = item_feature_name
        self.feature_manager = feature_manager
        self.temperature = temperature
        self.alpha = alpha
        self.identify_same_user = identify_same_user
        self.same_user_threshold = same_user_threshold

        # 获取item特征频次数据
        item_frequency_list = self.feature_manager.get_feature_frequency_list(self.item_feature_name)
        if not item_frequency_list:
            raise ConfigError(f"Feature frequency of item feature is empty. Please check it. ")

        self.item_frequency_percent: List[float] = tf.constant(item_frequency_list / np.sum(item_frequency_list),
                                                               'float32')

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """调用InBatchSoftmaxLayer层

        :param inputs: 三个Tensor组成的list，其中依次包含：

           * user_embedding: 用户embedding, 维度是[batch_size, embedding_size]
           * item_embedding: 物品embedding, 维度是[batch_size, embedding_size]
           * item_idx: 物品索引, 维度是[batch_size, 1]
        :param kwargs: 额外参数
        :return: 每个样本的loss值，维度为[batch_size, 1]
        """
        user_embedding, item_embedding, item_idx = inputs
        if item_idx.dtype != tf.int64:
            item_idx = tf.cast(item_idx, tf.int64)

        # 除以温度系数
        user_embedding = tf.divide(user_embedding, self.temperature)

        # 计算分数
        # logits shape: [batch_size, batch_size]
        logits = tf.matmul(user_embedding, item_embedding, transpose_b=True)

        # 获取item index 对应的频次
        # input_item_frequency： [batch_size, 1]
        input_item_frequency = tf.gather(self.item_frequency_percent, item_idx)

        # logits修正。防止log结果出现-inf, 所以和1e-9取最大值
        # item_frequency： [1, batch_size]
        item_frequency = tf.transpose(tf.math.log(tf.maximum(input_item_frequency, 1e-9)))
        correction_logits = logits - self.alpha * item_frequency

        # 构建标签对角矩阵
        # labels shape: [batch_size, batch_size]
        if self.identify_same_user:
            # 判断当前样本的用户embedding与batch内的其他用户embedding是否相等(曼哈顿距离小于阈值即相等)，
            manhattan_dist_matrix = tf.reduce_sum(
                tf.abs(tf.expand_dims(user_embedding, 1) - tf.expand_dims(user_embedding, 0)), -1)

            # 若相等则不作为当前用户的负样本
            equals_matrix = manhattan_dist_matrix <= self.same_user_threshold
            labels = tf.cast(equals_matrix, logits.dtype)
        else:
            # 直接将batch当前样本之外的其他所有样本作为负样本
            labels = tf.linalg.diag(tf.ones_like(logits[0]))

        # 计算loss
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=correction_logits)

        return tf.expand_dims(loss, axis=1)


@Module.register("cross_batch_softmax", tags=["loss"])
class CrossBatchSoftmaxLayer(layers.Layer):
    """CrossBatch Softmax Layer

    开始训练时，使用InBatchSoftmax进行训练。在训练执行数量的batch数据后，切换至CrossBatchSoftmax，即模型会缓存最近的若干条训练样本，将这些样本
    作为负例添加至当前的训练样本中计算softmax。
    参考论文：`Cross-Batch Negative Sampling for Training Two-Tower Recommenders <https://arxiv.org/pdf/2110.15154.pdf>`_

    :register name: cross_batch_softmax

    :config example:
    .. code-block:: jsonnet

       {
         name: "sampled_softmax",
         type: "cross_batch_softmax",
         parameters: {
           item_feature_name: "target_item_id",
             temperature: 0.05,
             alpha: 0.5,
             bank_queue_size: 50000,
             warm_up_batch: 1000,
           },
           inputs: {
             inputs: ["ref::user_embedding.output", "ref::item_embedding.output", "ref::input_layers.target_item_id"],
           },
           outputs: "loss",
       }

    :param item_feature_name: 特征名，用于从feature_manager中获取该特征的vocab_size
    :param feature_manager: 特征管理器，用于获取特征维度信息，**在配置项中无需配置**
    :param temperature: 温度系数
    :param alpha: 使用采样频率修正的loss项的系数
    :param bank_queue_size: 缓存的样本数量，这些样本将会作为负例参与计算softmax
    :param warm_up_batch: 预热的batch数量，预热前使用InBatchSoftmax计算，预热后使用CrossBatchSoftmax计算
    :param name: 层名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 item_feature_name: str,
                 feature_manager: FeatureManager,
                 temperature: float = 1.0,
                 alpha: float = 1.0,
                 bank_queue_size: int = 1000,
                 warm_up_batch: int = 1000,
                 name="cross_batch_softmax",
                 **kwargs):
        super(CrossBatchSoftmaxLayer, self).__init__(name=name, **kwargs)
        self.item_feature_name = item_feature_name
        self.feature_manager = feature_manager
        self.temperature = temperature
        self.alpha = alpha
        self.bank_queue_size = bank_queue_size
        self.warm_up_batch = warm_up_batch

        # 获取item特征频次数据
        item_frequency_list = self.feature_manager.get_feature_frequency_list(self.item_feature_name)
        if not item_frequency_list:
            raise ConfigError(f"Feature frequency of item feature is empty. Please check it. ")

        self.item_frequency_percent: List[float] = tf.constant(item_frequency_list / np.sum(item_frequency_list),
                                                               'float32')

        self.negative_bank = FIFOLayer(queue_size=self.bank_queue_size)

        # batch计数器
        self.batch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64, name="batch_counter")

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """调用CrossBatchSoftmaxLayer层

        :param inputs: 三个Tensor组成的list，其中依次包含：

           * user_embedding: 用户embedding, 维度是[batch_size, embedding_size]
           * item_embedding: 物品embedding, 维度是[batch_size, embedding_size]
           * item_idx: 物品索引, 维度是[batch_size, 1]
        :param kwargs: 额外参数
        :return: 每个样本的loss值，维度为[batch_size, 1]
        """
        user_embedding, item_embedding, item_idx = inputs
        batch_size, item_embedding_size = item_embedding.shape
        if item_idx.dtype != tf.int64:
            item_idx = tf.cast(item_idx, tf.int64)

        # 获取item index 对应的频次
        # input_item_frequency： [batch_size, 1]
        input_item_frequency = tf.gather(self.item_frequency_percent, item_idx)

        # logits修正。防止log结果出现-inf, 所以和1e-9取最大值
        # item_frequency： [batch_size, 1]
        item_frequency = tf.math.log(tf.maximum(input_item_frequency, 1e-9))

        item_embedding_with_frequency = tf.concat([item_embedding, item_frequency], axis=-1)

        # 更新步数
        self.batch_counter.assign_add(1)

        bank_inputs = tf.where(tf.greater(self.batch_counter, self.warm_up_batch),
                               item_embedding_with_frequency,
                               tf.fill(tf.shape(item_embedding_with_frequency), np.nan))

        bank_outputs = self.negative_bank(bank_inputs)

        bank_embedding, bank_frequency = tf.split(bank_outputs, num_or_size_splits=[item_embedding_size, 1], axis=-1)

        # item_embedding shape: [batch_size + bank_size, embedding_size]
        item_embedding = tf.concat([item_embedding, bank_embedding], axis=0)

        # item_frequency shape: [batch_size + bank_size, 1]
        item_frequency = tf.concat([item_frequency, bank_frequency], axis=0)

        # user embedding除以温度系数
        user_embedding = tf.divide(user_embedding, self.temperature)

        # 计算分数
        # logits shape: [batch_size, batch_size + bank_size]
        logits = tf.matmul(user_embedding, item_embedding, transpose_b=True)
        logits_shape = tf.shape(logits)

        correction_logits = logits - self.alpha * tf.transpose(item_frequency)

        # 构建标签对角矩阵
        # labels shape: [batch_size, batch_size + bank_size]
        labels = tf.linalg.diag(tf.ones((logits_shape[0],)), num_cols=logits_shape[1])

        # 计算loss
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=correction_logits)

        return tf.expand_dims(loss, axis=1)

