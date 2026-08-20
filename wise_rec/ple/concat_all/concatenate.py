#KL自带的模块
Module.register("concatenate", is_custom=False)(kl.Concatenate)

#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
from typing import List
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from modelflow.integrations.tensorflow.modules.module import Module


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


@Module.register("concatenate_by_domain")
class ConcatenateByDomain(layers.Layer):
    def __init__(self,
                 domain_index_list: List[int],
                 hidden_dim: int,
                 name='concatenate_by_domain',
                 **kwargs):
        self.domain_index_list = [np.int64(x) for x in domain_index_list]
        self.domain_num = len(self.domain_index_list)
        self.hidden_dim = hidden_dim
        self.project_dense_list = []
        super(ConcatenateByDomain, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        for _ in range(self.domain_num):
            self.project_dense_list.append(layers.Dense(self.hidden_dim, activation=None))
        super(ConcatenateByDomain, self).build(input_shape)

    def call(self, inputs):
        feature_, domain_indicator = inputs
        feature = dict()
        for key, value in feature_.items():
            feature[np.int64(key)] = tf.concat(value, axis=-1)
        
        indices = tf.expand_dims(tf.range(tf.shape(domain_indicator)[0], dtype=tf.int32), axis=-1)  # (batch_size,1)
        outputs_list = []
        indices_list = []
        for num, index in enumerate(self.domain_index_list):
            domain_mask = tf.squeeze(tf.equal(domain_indicator, index), axis=1)  # (batch_size,)
            masked_inputs = tf.squeeze(tf.boolean_mask(feature[index], domain_mask, axis=0), axis=1)
            masked_indices = tf.squeeze(tf.boolean_mask(indices, domain_mask), axis=1)  # (batch_size,1)

            projected_inputs = self.project_dense_list[num](masked_inputs)
            outputs_list.append(projected_inputs)
            indices_list.append(masked_indices)
        
        outputs = merge_masked_outputs(outputs_list, indices_list)
        return outputs  # (batch_size,...,dim)
