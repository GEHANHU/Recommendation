#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import json
from typing import Dict
from typing import List
from typing import Union


import tensorflow as tf
from tensorflow.keras.layers import Layer

from modelflow.common import Config
from modelflow.common.feature_transfer import feature_transform
from modelflow.integrations.tensorflow.modules.module import Module


@Module.register("scene_routing")
class SceneRouting(Layer):
    """场景路由

    根据场景的id标识，对输入列表进行路由并输出

    :register name: scene_routing

    :param name: 场景路由名字
    :param domain_num: 场景数量
    :param domain_start_index: 场景起始索引，例如使用01作为空置位置，有3个场景的情况下，场景的索引从2到4。
                               那么，参数应当设置为domain_start_index=2，domain_num=3
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 domain_num: int = 0,
                 domain_start_index: int = 0,
                 name="scene_routing",
                 **kwargs):
        self.domain_num = domain_num
        self.embedding_size = 0
        self.domain_start_index = domain_start_index
        super(SceneRouting, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        if isinstance(input_shape[0], List):
            self.domain_num = len(input_shape[0]) if not self.domain_num else self.domain_num
            self.embedding_size = input_shape[0][0][-1]
        else:
            self.domain_num = input_shape[0][-2] if not self.domain_num else self.domain_num
            self.embedding_size = input_shape[0][-1]

    def call(self, inputs: List[Union[List[tf.Tensor], tf.Tensor]], **kwargs) -> tf.Tensor:
        """场景路由

        :param inputs:
           * inputs[0]: embedding列表，维度是List[[batch_size, embedding_size]] or[batch_size, length,embedding_size]
           * inputs[1]: 场景标识，维度[batch_size]
        :param kwargs: 额外参数
        :return: 选择后的embedding [batch_size, embedding_size]
        """
        select_list = inputs[0]
        domain_indicator: tf.Tensor = inputs[1]
        domain_indicator = domain_indicator - self.domain_start_index
        domain_mask = tf.one_hot(tf.squeeze(domain_indicator, axis=1), depth=self.domain_num)
        domain_mask = tf.expand_dims(domain_mask, axis=-1)
        if isinstance(select_list, List):
            output = tf.reduce_sum(tf.stack(select_list, axis=1) * domain_mask, axis=1)
        else:
            output = tf.reduce_sum(select_list * domain_mask, axis=1)

        return output


@Module.register("scene_routing_with_index")
class SceneRoutingWithIndex(Layer):
    """带输出顺序的场景路由

    根据场景的id标识，对输入列表进行路由并输出

    :register name: scene_routing

    :param name: 场景路由名字
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 domain_feature: str,
                 domain_index_list: list,
                 feature_map_path: str,
                 name="scene_routing_with_index",
                 **kwargs):
        self.domain_index_list = self.get_domain_index(domain_feature, domain_index_list, feature_map_path)
        self.domain_num = len(self.domain_index_list)
        self.indicator_order = None
        super(SceneRoutingWithIndex, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.indicator_order = tf.constant(self.domain_index_list, dtype=tf.int64)

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:

        outputs = inputs[0]
        indicator = inputs[1]
        indicator = tf.tile(indicator, [1, self.domain_num])
        mask_tensor = tf.equal(indicator, self.indicator_order)
        masked_output = tf.boolean_mask(outputs, mask_tensor, axis=0)
        return masked_output

    @staticmethod
    def get_domain_index(domain_feature, domain_index_list, feature_map_path):
        res_domain_index = []
        with open(feature_map_path, "r") as f:
            feature_map = json.load(f)["sparse"][domain_feature]
            dft = -1
            for item in domain_index_list:
                domain_index = feature_map.get(item, dft)
                res_domain_index.append(domain_index)
                dft = dft if domain_index != dft else dft-1

        return res_domain_index


@Module.register("mask_routing_with_index")
class MaskRoutingWithIndex(Layer):

    """ 针对输出进行mask路由

    :register name: mask_routing_with_index

    :config example:
    .. code-block:: jsonnet

       {
        name:"mask_routing_with_index",
        type:'mask_routing_with_index',
        parameters: {
            mask_indicator: {
                "ocpx_taget": [ [ ], [ ], [ ], [ '14' ] ],
                "is_activate_no_fastapp": [ [ '1' ], [ ], [ ], [ ] ],
                "is_reengage_no_fastapp_latest": [ [ ], [ '1' ],[ ], [ ] ],
                "is_register_no_fastapp": [ [ ], [ ], [ '1' ], [ ] ],
                "is_fast_app": [ [ ], [ ], [ ], [ '1' ] ]
                },
            transform_tool: '../model_museum/ads/feature_map.json', // {type: 'mmhash2'}
        },
        inputs: {
            features: ["ref::inputs.`name == 'ocpx_taget'`",
                       "ref::inputs.`name == 'is_activate_no_fastapp'`",
                       "ref::inputs.`name == 'is_reengage_no_fastapp_latest'`",
                       "ref::inputs.`name == 'is_reengage_no_fastapp_latest'`",
                       "ref::inputs.`name == 'is_fast_app'`"],
            logits: ['ref::mmoe2.output[0]', 'ref::mmoe2.output[1]', 'ref::mmoe2.output[2]', 'ref::mmoe2.output[3]'],
        },
        outputs: 'output',
      },

    :param mask_indicator: mask特征名和原始值，values为对应特征原始值
    :param transform_util: 转换工具，如果是hash数据，则传入对应的hash函数，如果是feature_map数据，则传入feature_map的路径
    :param name: 层名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 mask_indicator: Dict[str, List[str]],
                 transform_tool: Union[Config, str],
                 task_num: int = 0,
                 name="mask_routing_with_index",
                 **kwargs):
        self.mask_list = self.get_domain_index(mask_indicator, transform_tool)
        self.task_num = task_num
        super(MaskRoutingWithIndex, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, features: List[Dict[str, tf.Tensor]], logits: List[tf.Tensor], **kwargs) -> tf.Tensor:
        res_tensor = []
        for idx in range(self.task_num):
            task_mask = []
            for fea_idx, mask_vec in enumerate(self.mask_list):
                group_mask = tf.cast(tf.equal(list(features[fea_idx].values())[0], mask_vec[idx]), dtype=tf.float32)
                group_mask = tf.expand_dims(tf.reduce_sum(group_mask, axis=-1), axis=-1)
                task_mask.append(group_mask)
            if len(task_mask) > 1:
                task_mask = tf.concat(task_mask, axis=1)
                task_mask = tf.reduce_max(task_mask, axis=1, keepdims=True)
            else:
                task_mask = task_mask[0]

            res_tensor.append(logits[idx] * task_mask)
        masked_task_out_tensor = tf.concat(res_tensor, axis=-1)
        masked_out = tf.reshape(tf.reduce_sum(masked_task_out_tensor, axis=-1), [-1, 1])

        return masked_out

    @staticmethod
    def get_domain_index(mask_indicator, transform_tool):
        transform_fun = feature_transform(transform_tool)
        res_mask = []
        for feature, value in mask_indicator.items():
            tmp_mask_vec = []
            for val in value:
                # 如果val为空，则赋值-1，小模型index不可能为-1，大模型为-1的概率也极低，可以忽略不计
                fea_mask = [transform_fun(feature, f) for f in val] if val else [-1]
                tmp_mask_vec.append(fea_mask)
            res_mask.append(tmp_mask_vec)
        return res_mask


@Module.register("input_routing_mask")
class InputRoutingMask(Layer):
    """ 针对输入进行样本路由

    :register name: input_routing_mask

    :config example:
    .. code-block:: jsonnet

       {
        name:"input_routing_mask",
        type:'input_routing_mask',
        parameters: {
            domain_feature_list: [{
                "feature_key":["some_dense_feature"],
                "feature_value":[[0]],
                "del_branch_value":False,
                "del_masked_sample":False
                }, {...}]
            use_remaining_sample: True,
        },
        inputs: {
            inputs: "ref::inputs",
        },
        outputs: ["branch_masks, "sample_masks"],
      },

    :param name: 输出层名称
    :param domain_feature_dict: 分流逻辑配置
    :param use_remaining_sample: 是否使用剩余样本
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 name: str = "input_routing_mask",
                 use_remaining_sample: bool = False,
                 domain_feature_list: list = None,
                 **kwargs):
        super(InputRoutingMask, self).__init__(name=name, **kwargs)
        self.use_remaining_sample = use_remaining_sample
        self.domain_feature_list = domain_feature_list
        if len(self.domain_feature_list) < 1:
            raise ValueError("domain_feature_list is empty ...")

        self.flattened_feature_list = sum([[value.get("feature_key", 'error_key')]
                                           if isinstance(value.get("feature_key", 'error_key'), str)
                                           else value.get("feature_key", 'error_key')
                                           for value in self.domain_feature_list], [])
        self.flattened_feature_list = list(set(self.flattened_feature_list))

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs):
        domain_feature_list_raw = {k: v for k, v in inputs.items() if k in self.flattened_feature_list}
        branch_masks = []
        temp_masks = []
        sample_masks = tf.ones(shape=[tf.shape(inputs[next(iter(inputs))])[0]])

        for _, branch_i in enumerate(self.domain_feature_list):
            branch_i_key_list = branch_i.get('feature_key', None)
            branch_i_value_list = branch_i.get("feature_value", None)

            if isinstance(branch_i_key_list, str):
                branch_i_key_list = [branch_i_key_list]
            if not isinstance(branch_i_value_list[0], list):
                branch_i_value_list = [branch_i_value_list]
            if not (len(branch_i_key_list) == len(branch_i_value_list)):
                raise ValueError("The length of key_list is not equal with focus_value_list...")

            branch_mask_i_list = []
            for branch_i_key, branch_i_value in (zip(branch_i_key_list, branch_i_value_list)):
                branch_feature_tensor_i = domain_feature_list_raw.get(branch_i_key, None)
                branch_mask_i_list = (branch_mask_i_list +
                                      [tf.equal(branch_feature_tensor_i, value)
                                       for value in branch_i_value])

            branch_mask_i = tf.concat(branch_mask_i_list, axis=-1)
            branch_mask_i = tf.greater(tf.reduce_sum(tf.cast(branch_mask_i, tf.int64), axis=-1), 0)
            if branch_i.get("del_branch_value", False):
                branch_mask_i = ~branch_mask_i
            branch_masks.append(branch_mask_i)
            if branch_i.get("del_masked_sample", False):
                temp_masks.append(branch_mask_i)

        if temp_masks:
            sample_masks = tf.greater(tf.reduce_sum(tf.cast(
                tf.concat([tf.expand_dims(sample_mask_i, axis=-1)
                           for sample_mask_i in temp_masks], axis=-1), tf.int64), axis=-1), 0)
            sample_masks = ~sample_masks

        if self.use_remaining_sample:
            branch_masks.append(sample_masks)

        return branch_masks, sample_masks
