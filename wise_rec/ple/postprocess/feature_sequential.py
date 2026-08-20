#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import inspect
from typing import Dict
from typing import List
from typing import Union

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Lambda

from modelflow.common import Config
from modelflow.integrations.tensorflow.data import TFFeatureManager
from modelflow.integrations.tensorflow.modules.module import Module


@Module.register("feature_sequential")
class FeatureSequential(layers.Layer):
    """ 对输入层中的每个特征构建独立的序贯模型（:class:`tf.keras.Sequential`）并执行。如果特征找不到对应的配置，则直接放通。

    :register name: feature_sequential

    :param name: 层名
    :param methods: 字典格式，用来配置不同特征的序贯模型，如果有的特征没有对应的配置，则直接放通。其中key为特征名或特征查询语句，
       value是列表类型，表示序贯模型的配置。特征查询语句参考：:class:`~modelflow.data.feature_manager.FeatureManager.query_feature`，
       例如::

        {
          "features#multiple_discrete": [{ "type": "average", "axis": 1 }, {"type": "dense", "units": 32}]
          "features#single_discrete": [{ "type": "max", "axis": 1 }, {"type": "dense", "units": 32}]
        }
    :param feature_manager: 特征管理器，**在配置项中无需配置**

    .. versionchanged:: 23.12.1
       `methods` 支持配置函数
    """

    def __init__(self,
                 name,
                 methods: Config,
                 feature_manager: TFFeatureManager,
                 drop_other_features: bool = False,
                 **kwargs):
        super(FeatureSequential, self).__init__(name=name, **kwargs)
        self.methods = methods
        self.feature_manager = feature_manager
        self.layer_dict: Dict[str, layers.Layer] = {}
        self.drop_other_features = drop_other_features

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        input_feature_names = list(input_shape.keys())
        self.layer_dict: Dict[str, layers.Layer] = self._init_layers(input_feature_names)
        super().build(input_shape)

    def _init_layers(self, input_feature_names: List[str]):
        layer_config: Dict[str, Config] = {}
        for query, config in self.methods.items():
            feature_names = self.feature_manager.query_feature(query)
            valid_feature_names = [feature for feature in feature_names if feature in input_feature_names]
            for feature_name in valid_feature_names:
                layer_config[feature_name] = config

        sequential_dict = {}
        for feature_name, config_list in layer_config.items():
            layer_list = []
            for config in config_list:
                module = Module.from_config(config)
                if inspect.isfunction(module):
                    func_config = config.duplicate()
                    func_config.pop("type")
                    func = module
                    module = Lambda(lambda x: func(x, **func_config))
                layer_list.append(module)
            sequential_dict[feature_name] = tf.keras.Sequential(layer_list)
        return sequential_dict

    def call(self, inputs, **kwargs):
        """ 执行特征序贯模型层

        :param inputs: 字典类型，其中key是特征名，value是特征的值
        :param kwargs: 额外参数
        :return: 字典类型，其中key是特征名，value是特征执行序贯模型的结果
        """
        if not isinstance(inputs, dict):
            raise ValueError(f"Call value must be a dictionary. ")

        outputs = {}
        for feature_name, input_tensor in inputs.items():
            if feature_name in self.layer_dict:
                outputs[feature_name] = self.layer_dict.get(feature_name)(input_tensor)
            elif not self.drop_other_features and len(input_tensor.shape) == 2:
                outputs[feature_name] = input_tensor
        return outputs

    def compute_mask(self,
                     inputs: Dict[str, tf.Tensor],
                     mask: Union[Dict[str, tf.Tensor], None] = None) -> Dict[str, tf.Tensor]:
        input_mask_dict = mask or {}
        output_mask_dict = {}
        for feature_name, input_tensor in inputs.items():
            feature_layer = self.layer_dict.get(feature_name)
            input_mask = input_mask_dict.get(feature_name)
            if feature_layer is not None:
                output_mask = feature_layer.compute_mask(input_tensor, input_mask)
            else:
                output_mask = None
            output_mask_dict[feature_name] = output_mask
        return output_mask_dict
