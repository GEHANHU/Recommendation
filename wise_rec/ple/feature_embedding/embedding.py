#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
import os
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Union

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers
from tensorflow.python.keras.utils import tf_utils

from modelflow.common import Config
from modelflow.common import logger
from modelflow.common.config import convert_config_to_dict
from modelflow.common.exception import ConfigError
from modelflow.data.embedding_file import EmbeddingFile
from modelflow.integrations.tensorflow.modules.module import Module


@Module.register("embedding", tags=["embedding"])
class Embedding(layers.Embedding):
    """在当前设备中初始化embedding weight

    在 Tensorflow 2.3.4版本的 :class:`tf.keras.layers.Embedding` 类中，当开启eager模型且使用GPU训练时，
    embedding weight会强制在CPU中进行初始化，导致训练速度变慢5倍以上。因此，本类继承了 :class:`tf.keras.layers.Embedding` ，
    强制embedding weight在当前设备中加载。

    如果出现问题，可以使用配置 ``{"type": "origin_embedding"}``

    :register name: embedding

    :param input_dim: 特征值数量
    :param output_dim: embedding长度
    :param embeddings_initializer: embedding weight初始化方法
    :param embeddings_regularizer: embedding weight正则化方法
    :param activity_regularizer: 激活层正则化方法
    :param embeddings_constraint: embedding weight约束函数
    :param mask_zero: 是否需要掩盖index=0的情况
    :param input_length: 输入长度
    :param kwargs:  :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等

    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 embeddings_initializer: Union[str, Config, None] = 'uniform',
                 embeddings_regularizer: Union[str, Config, None] = None,
                 activity_regularizer: Union[str, Config, None] = None,
                 embeddings_constraint: Union[str, Config, None] = None,
                 mask_zero: bool = False,
                 input_length: Union[int, None] = None,
                 **kwargs):
        fixed_embeddings_initializer = convert_config_to_dict(embeddings_initializer)
        fixed_embeddings_regularizer = convert_config_to_dict(embeddings_regularizer)
        fixed_activity_regularizer = convert_config_to_dict(activity_regularizer)
        fixed_embeddings_constraint = convert_config_to_dict(embeddings_constraint)
        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         embeddings_initializer=fixed_embeddings_initializer,
                         embeddings_regularizer=fixed_embeddings_regularizer,
                         activity_regularizer=fixed_activity_regularizer,
                         embeddings_constraint=fixed_embeddings_constraint,
                         mask_zero=mask_zero,
                         input_length=input_length,
                         **kwargs)
        self.embeddings: Union[tf.Variable, None] = None
        self.built: bool = False

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint)
        self.built = True


@Module.register("pretrained_embedding", tags=["embedding"])
class PretrainedEmbedding(Embedding):
    """ 加载预训练的embedding，如果存在有些id找不到对应的预训练embedding，则使用 `missing_embeddings_initializer` 进行初始化

    .. note::
       如果该类是在 :class:`~modelflow.integrations.tensorflow.modules.features.feature_embedding.FeatureEmbedding` 中配置，
       则不需要配置 `input_dim`, `feature_name` 和 `feature_map`。

    :register name: pretrained_embedding

    :param input_dim: 特征值数量
    :param file_path: 加载的embedding文件路径字符串模板，如果文件路径中包含特征名，可以使用 `{feature_name}` 来替代。
    :param file_config: 文件加载所需的参数，参考 :mod:`modelflow.data.embedding_file`。
       如果输入的类型是字符串，那么该字符串表示embedding file的类型，且初始化参数为空。
    :param feature_name: 特征名
    :param feature_map: 特征值到特征索引的映射
    :param missing_embeddings_initializer: 缺失值的初始化方法。如果embedding文件中没有原始的item，只有embedding矩阵，那么此配置无效。
    :param embeddings_regularizer: embedding weight正则化方法
    :param activity_regularizer: 激活层正则化方法
    :param embeddings_constraint: embedding weight约束函数
    :param mask_zero: 是否需要掩盖index=0的情况
    :param input_length: 输入长度
    :param kwargs:  :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 input_dim: int,
                 file_path: Union[str, Path, None] = None,
                 file_config: Union[str, Config, None] = None,
                 feature_name: Union[str, None] = None,
                 feature_map: Optional[Dict[str, int]] = None,
                 missing_embeddings_initializer: Union[str, Config, None] = 'uniform',
                 embeddings_regularizer: Union[str, Config, None] = None,
                 activity_regularizer: Union[str, Config, None] = None,
                 embeddings_constraint: Union[str, Config, None] = None,
                 mask_zero: bool = False,
                 input_length: Optional[int] = None,
                 **kwargs):
        self.file_path = file_path and file_path.format(feature_name=feature_name)
        self.file_config = file_config
        self.feature_name = feature_name
        self.feature_map = feature_map
        self.missing_embeddings_initializer = initializers.get(convert_config_to_dict(missing_embeddings_initializer))

        # output_dim暂时设置为1，因为此项必填；最终的output_dim由文件中的embedding长度决定
        super(PretrainedEmbedding, self).__init__(input_dim=input_dim,
                                                  output_dim=1,
                                                  embeddings_initializer=missing_embeddings_initializer,
                                                  embeddings_regularizer=embeddings_regularizer,
                                                  activity_regularizer=activity_regularizer,
                                                  embeddings_constraint=embeddings_constraint,
                                                  mask_zero=mask_zero,
                                                  input_length=input_length,
                                                  **kwargs)
        # 由于还没有从文件中读取output_dim, 所以将其置为0
        self.output_dim = 0
        self.embeddings: Union[tf.Variable, None] = None
        self.built: bool = False

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        pretrained_embedding, item_list = None, None
        if self.file_config and self.file_path and os.path.exists(self.file_path):
            embedding_file: EmbeddingFile = EmbeddingFile.from_config(self.file_config)
            pretrained_embedding, item_list = embedding_file.load(self.file_path)
            self.output_dim = pretrained_embedding.shape[1]
            logger.info("The size of pretrained embedding is set to %s.", self.output_dim)

        if self.output_dim <= 0:
            raise ConfigError("Failed to load embedding file. Maybe embedding file config or embedding "
                              "file path is empty. Or perhaps embedding file path is not existent. ")

        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint)

        if pretrained_embedding is not None:
            if item_list is None:
                pretrained_embedding_count = pretrained_embedding.shape[0]
                if pretrained_embedding_count != self.input_dim:
                    raise ValueError(f"When origin item list is empty in pretrained embedding file, the number of "
                                     f"pretrained embeddings must be the same as the number of input features. "
                                     f"But the obtained counts are different: "
                                     f"{pretrained_embedding_count} vs {self.input_dim}.")
                K.set_value(self.embeddings, pretrained_embedding)
                logger.info("Pretrained embedding for %s is loaded, but item list is empty.", self.feature_name)
                self.built = True
                return

            valid_count = 0
            feature_map = self.feature_map
            if feature_map:
                valid_feature_index_list = []
                valid_feature_embedding_list = []
                for feature_origin_value, feature_embedding in zip(item_list, pretrained_embedding):
                    if feature_origin_value not in feature_map:
                        continue
                    feature_index = feature_map[feature_origin_value]
                    valid_feature_index_list.append([feature_index])
                    valid_feature_embedding_list.append(feature_embedding)
                valid_count = len(valid_feature_index_list)
                if valid_count:
                    # 执行替换逻辑
                    replace_embedding_tensor = tf.cast(tf.stack(valid_feature_embedding_list, axis=0), dtype=K.floatx())
                    self.embeddings.scatter_nd_update(valid_feature_index_list, replace_embedding_tensor)

            logger.info("Pretrained embedding for %s is loaded, valid count: %d/%d.",
                        self.feature_name, valid_count, self.input_dim)

        self.built = True

    @classmethod
    def from_config(cls, config):
        output_dim = config.pop("output_dim", 0)
        if "embeddings_initializer" in config and "missing_embeddings_initializer" not in config:
            # 此处是为了应对加载模型的过程中，从配置中加载初始化pretrained_embedding的情况。
            # 因为保存的config中没有missing_embeddings_initializer，需要从embeddings_initializer中获取
            config["missing_embeddings_initializer"] = config.pop("embeddings_initializer")
        pretrained_embedding = cls(**config)

        # 加载模型的过程中，output_dim不能为0，因此需要修改output_dim
        pretrained_embedding.output_dim = output_dim

        return pretrained_embedding


@Module.register("norm_embedding", tags=["embedding"])
class NormEmbedding(Embedding):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 norm: Optional[Config] = None,
                 embeddings_initializer: Union[str, Config, None] = 'uniform',
                 embeddings_regularizer: Union[str, Config, None] = None,
                 activity_regularizer: Union[str, Config, None] = None,
                 embeddings_constraint: Union[str, Config, None] = None,
                 mask_zero: bool = False,
                 input_length: Optional[int] = None,
                 **kwargs):
        """ 带归一化的embedding

        :register name: norm_embedding

        :param input_dim: 特征值数量
        :param output_dim: embedding长度
        :param norm: 归一化配置，样例： ``{"type": "batch_norm", "axis": -1}``
        :param embeddings_initializer: embedding weight初始化方法
        :param embeddings_regularizer: embedding weight正则化方法
        :param activity_regularizer: 激活层正则化方法
        :param embeddings_constraint: embedding weight约束函数
        :param mask_zero: 是否需要掩盖index=0的情况
        :param input_length: 输入长度
        :param kwargs:  :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
        """
        super().__init__(input_dim,
                         output_dim,
                         embeddings_initializer,
                         embeddings_regularizer,
                         activity_regularizer,
                         embeddings_constraint,
                         mask_zero,
                         input_length,
                         **kwargs)
        self.norm = norm
        self.norm_layer = Module.from_config(norm, tags=["norm"])

    def call(self, inputs):
        """ 执行带归一化的embedding

        :param inputs: 输入的特征索引
        :return: 特征索引对应的embedding
        """
        output_embedding = super(NormEmbedding, self).call(inputs)
        if self.norm_layer:
            output_embedding = self.norm_layer(output_embedding)
        return output_embedding
