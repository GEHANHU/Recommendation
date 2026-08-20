#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import copy
from typing import Dict
from typing import List
from typing import Union

import tensorflow as tf
from tensorflow.keras import layers

from modelflow.common import Config
from modelflow.common import constants
from modelflow.common import logger
from modelflow.data import Feature
from modelflow.data import FeatureManager
from modelflow.integrations.tensorflow.modules.module import Module


@Module.register("feature_embedding")
class FeatureEmbedding(layers.Layer):
    """ 对输入层中的每个特征构建embedding层。

    在配置embedding层的过程中，有如下配置已经自动生成，并作为embedding层的输入参数，因此这些配置无需在 `methods` 中进行配置。**下面的配置中除了
    output_dim之外，其他配置均不可覆盖**：

    * input_dim: 特征值的数量，通过计算特征的 **最大索引值** 来获得
    * output_dim: 优先使用 `methods` 参数中配置的值。如果没有配置，则使用feature config中的 `embedding_size`。如果 `embedding_size`
      为空，则使用 :func:`~modelflow.data.feature.compute_embedding_size` 方法计算embedding_size
    * input_length: 特征的长度，从feature config中的 `length` 获得
    * feature_name: 特征的名字，从feature config中的 `name` 获得
    * feature_map: 特征map，从feature map中获得
    * name：格式为 `embedding_{feature_name}`，如果embedding需要引用或导出，需要使用此名字。

    .. note::
       `methods` 参数中，仅支持配置tag包含 `embedding` 的module作为embedding层。

    .. admonition:: embedding共享

       如果有特征需要共享embedding，那么必须在feature config文件，需要使用共享embedding特征的配置中，添加配置 `embedding_name`，值为
       所要共享的特征名。例如：特征 `b` 要使用特征 `a` 的embedding，那么就需要在特征 `b` 的配置中添加配置项： `"embedding_name": "a"`


    :register name: feature_embedding

    :param name: 层名
    :param methods: 字典格式，用来配置不同特征的embedding初始化方法，如果有的特征没有对应的配置，则直接放通。其中key为特征名或特征查询语句，
       value是字典类型，表示embedding配置。特征查询语句参考：:class:`~modelflow.data.feature_manager.FeatureManager.query_feature`，
       例如::

        {
          "features#discrete": {
            "type": "embedding",
            "mask_zero": true,
            "output_dim": 16,
          }
        }
    :param feature_manager: 特征管理器，**在配置项中无需配置**
    """

    def __init__(self, name, methods: Config, feature_manager: FeatureManager, **kwargs):
        super(FeatureEmbedding, self).__init__(name=name, **kwargs)
        self.methods = methods
        self.feature_manager = feature_manager
        self.embedding_layer_dict: Dict[str, layers.Layer] = {}

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        input_feature_names = list(input_shape.keys())
        embedding_layer_config: Dict[str, Config] = self._get_embedding_config(input_feature_names)
        self.embedding_layer_dict: Dict[str, layers.Layer] = self._init_embedding_layers(embedding_layer_config)
        super().build(input_shape)

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs) -> Dict[str, tf.Tensor]:
        """ 执行特征embedding层

        :param inputs: 字典类型，其中key是特征名，value是特征的值
        :param kwargs: 额外参数
        :return: 字典类型，其中key是特征名，value是特征的embedding值
        """
        if not isinstance(inputs, dict):
            raise ValueError(f"Call value must be a dictionary. ")

        outputs = {}
        for feature_name, input_tensor in inputs.items():
            if feature_name in self.embedding_layer_dict:
                outputs[feature_name] = self.embedding_layer_dict.get(feature_name)(input_tensor)
            else:
                outputs[feature_name] = input_tensor
        return outputs

    def compute_mask(self,
                     inputs: Dict[str, tf.Tensor],
                     mask: Union[Dict[str, tf.Tensor], None] = None) -> Dict[str, tf.Tensor]:
        input_mask_dict = mask or {}
        output_mask_dict = {}
        for feature_name, input_tensor in inputs.items():
            embedding_layer = self.embedding_layer_dict.get(feature_name)
            input_mask = input_mask_dict.get(feature_name)
            if embedding_layer is not None:
                output_mask = embedding_layer.compute_mask(input_tensor, input_mask)
            else:
                output_mask = None
            output_mask_dict[feature_name] = output_mask
        return output_mask_dict

    def _init_embedding_layers(self, embedding_layer_config: Dict[str, Config]) -> Dict[str, layers.Layer]:
        embedding_layer_dict: Dict[str, layers.Layer] = {}
        share_embedding_name_map: Dict[str, str] = {}
        if self.feature_manager.origin_feature_map is not None:
            sparse_feature_map: Dict[str, Dict[str, int]] = self.feature_manager.origin_feature_map.get("sparse", {})
        else:
            sparse_feature_map: Dict[str, Dict[str, int]] = {}

        for feature_name, config in embedding_layer_config.items():
            feature = self.feature_manager.feature_dict[feature_name]
            share_embedding_name = feature.share_embedding_name or feature_name
            if share_embedding_name in share_embedding_name_map:
                # 根据feature config中的配置项 ``embedding_name`` 共享embedding
                share_feature_name = share_embedding_name_map.get(share_embedding_name)
                embedding_layer_dict[feature_name] = embedding_layer_dict.get(share_feature_name)
                logger.info("Feature `%s` shares its embedding with feature `%s`.", share_feature_name, feature_name)
                continue
            embedding_init_config = self._get_embedding_init_config(config, feature, sparse_feature_map)
            embedding_layer = Module.from_config(embedding_init_config,
                                                 name=f"embedding_{feature_name}",
                                                 tags=["embedding"])
            embedding_layer_dict[feature_name] = embedding_layer
            share_embedding_name_map[share_embedding_name] = feature_name
            logger.info("type: %s, name: %s, size: %s, count: %s",
                        embedding_init_config[constants.TYPE],
                        embedding_layer.name,
                        getattr(embedding_layer, 'output_dim', 'none'),
                        getattr(embedding_layer, 'input_dim', 'none'))

        return embedding_layer_dict

    def _get_embedding_config(self, input_feature_names: List[str]) -> Dict[str, Config]:
        embedding_config: Dict[str, Config] = {}
        feature_dict = self.feature_manager.feature_dict

        for query, config in self.methods.items():
            feature_names = self.feature_manager.query_feature(query)
            valid_feature_names = [feature for feature in feature_names if feature in input_feature_names]
            for feature_name in valid_feature_names:
                embedding_config[feature_name] = config

        # 根据共享embedding名进行排序，为了之后能够找到特征对应的embedding
        sorted_config = sorted(embedding_config.items(),
                               key=lambda kv: feature_dict[kv[0]].share_embedding_name or "")

        embedding_config = dict(sorted_config)

        # 初始化embedding type, 保证所有特征都有embedding初始化
        for feature_name in feature_dict.keys():
            if feature_name not in embedding_config:
                logger.warning("Cannot find init embedding config for feature %s.", feature_name)

        return embedding_config

    def _get_embedding_init_config(self,
                                   config: Config,
                                   feature: Feature,
                                   sparse_feature_map: Dict[str, Dict[str, int]]) -> Config:
        embedding_config = copy.deepcopy(config)
        embedding_config["input_dim"] = feature.count
        embedding_config["output_dim"] = embedding_config.get("output_dim") or feature.embedding_size
        embedding_config["input_length"] = feature.length
        embedding_config["feature_name"] = feature.name
        feature_map_key = feature.share_embedding_name or feature.name
        embedding_config["feature_map"] = sparse_feature_map.get(feature_map_key, None)
        return embedding_config
