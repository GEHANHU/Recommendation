#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import Union, List, Dict

import tensorflow as tf
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Embedding
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers

from modelflow.common import Config
from modelflow.common.config import convert_config_to_dict
from modelflow.integrations.tensorflow.modules.module import Module


@Module.register("auto_dis")
class AutoDis(layers.Layer):
    """ 对连续值的特征，进行离散化

    输入默认为连续值特征提前concat后的值，输入的维度为(None, dense_feature_num)
    若data_type为"Map"为连续值特征字典，其中key是特征名，value是特征值

    :register name: auto_dis

    :config example:
    .. code-block:: jsonnet

       {
         name: "autodis",
         type: "auto_dis",
         parameters: {
             temp: 0.02,
             bins: 50,
             embedding_size: 16
           },
           inputs: {
             inputs: "ref::float_norm.embedding",
           },
           outputs: "embedding",
       }

    :param temp: 温度系数，用于控制离散化后的分布，其接近0时，得到的分桶概率分布接近于one-hot，当其接近于无穷时，得到的分桶概率分布近视于均匀分布。
    :param bins: 分桶数，将每个域的特征值离散到bins个桶中，每个桶对应一个共享的embedding（meta-embedding）
    :param embedding_size: 控制离散化后的feature_dim，即每个特征离散化后的embedding维度。
    :param weight_initializer: weight初始化方法
    :param weight_regularizer: weight正则化方法
    :param weight_constraint: weight约束函数
    :param data_type: 数据类型，适配模型输入输出是Tensor类型还是以Map类型 ["Tensor", "Map"]
    :param dense_feature_num: 由模型计算而来，或由人为指定
    """

    def __init__(self,
                 temp: float = 0.02,
                 bins: int = 30,
                 embedding_size: int = 16,
                 weight_initializer: Union[str, Config, None] = 'random_uniform',
                 weight_regularizer: Union[str, Config, None] = None,
                 weight_constraint: Union[str, Config, None] = None,
                 data_type: str = "Tensor",
                 name: str = "auto_dis",
                 **kwargs):
        self.temp = temp
        self.bins = bins
        self.embedding_size = embedding_size
        self.data_type = data_type
        fixed_weight_initializer = convert_config_to_dict(weight_initializer)
        fixed_weight_regularizer = convert_config_to_dict(weight_regularizer)
        fixed_weight_constraint = convert_config_to_dict(weight_constraint)
        self.weight_initializer = initializers.get(fixed_weight_initializer)
        self.weight_regularizer = regularizers.get(fixed_weight_regularizer)
        self.weight_constraint = constraints.get(fixed_weight_constraint)
        self.h_logits1, self.h_logits2, self.h_embedding = None, None, None
        self.dense_feature_num = None
        super(AutoDis, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        if self.data_type == "Map":
            input_feature_names = list(input_shape.keys())
            self.dense_feature_num = len(input_feature_names)
        else:
            self.dense_feature_num = input_shape[-1]

        self.h_logits1 = self.add_weight(name="h_logits1",
                                         shape=(self.dense_feature_num, 1, self.bins),
                                         dtype=K.floatx(),
                                         initializer=self.weight_initializer,
                                         regularizer=self.weight_regularizer,
                                         constraint=self.weight_constraint)
        self.h_logits2 = self.add_weight(name="h_logits2",
                                         shape=(self.dense_feature_num, self.bins, self.bins),
                                         dtype=K.floatx(),
                                         initializer=self.weight_initializer,
                                         regularizer=self.weight_regularizer,
                                         constraint=self.weight_constraint)
        # meta-embeddings组构建：每个特征域对应一个meta-embedding组，每个meta-embedding组含bins个meta-embedding
        self.h_embedding = self.add_weight(name="h_embedding",
                                           shape=(self.dense_feature_num, self.bins, self.embedding_size),
                                           dtype=K.floatx(),
                                           initializer=self.weight_initializer,
                                           regularizer=self.weight_regularizer,
                                           constraint=self.weight_constraint)
        super(AutoDis, self).build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs):
        """ 将连续值类型特征映射成embedding，并将embedding后的结果flatten

        :param inputs: shape=(None, dense_feature_num) or 字典类型，其中key是特征名，value是特征的值
        :param kwargs: 额外参数
        :return: 连续值类特征embedding后的结果，默认embedding的结果会展开变回二维，shape=(None, dense_feature_num * embedding_size)
                若data_type为Map, 输出为{ feature_name: feature_embedding}
        """

        #
        index_name = []
        inputs_ls = []
        if self.data_type == "Map":
            for feature_name, input_tensor in inputs.items():
                inputs_ls.append(input_tensor)
                index_name.append(feature_name)
            inputs = tf.concat(inputs_ls, axis=-1)

        # cont_wt_hldr shape: (dense_feature_num, None, 1)
        cont_wt_hldr = tf.transpose(tf.expand_dims(inputs, 2), [1, 0, 2])
        # 将每个特征域中的每个特征值离散到bins个桶中，h_logits_mlp shape:(dense_feature_num, None, bins)
        h_logits_mlp = tf.nn.leaky_relu(tf.matmul(cont_wt_hldr, self.h_logits1))
        # h_logits_trans shape: (dense_feature_num, None, bins)
        h_logits_trans = tf.matmul(h_logits_mlp, self.h_logits2) + h_logits_mlp

        # 归一化, y shape: (dense_feature_num, None, bins)
        y = tf.nn.softmax(h_logits_trans / self.temp)
        # cont_vx_embed shape: (dense_feature_num, None, embedding_size)
        cont_vx_embed = tf.matmul(y, self.h_embedding)
        # cont_vx_embed shape: (None, dense_feature_num, embedding_size)
        cont_vx_embed = tf.transpose(cont_vx_embed, [1, 0, 2])
        # 将embedding后的结果展开，返回结果的维度=(None, dense_feature_num * embedding_size)
        outputs = layers.Flatten(name='dense_embedding_flatten')(cont_vx_embed)

        if self.data_type == "Map":
            outputs = {}
            for i in range(cont_vx_embed.shape[1]):
                outputs[index_name[i]] = cont_vx_embed[:, i, :]

        return outputs

    def get_config(self):
        config = {
            'temp': self.temp,
            'bins': self.bins,
            'embedding_size': self.embedding_size,
            'data_type': self.data_type
        }
        base_config = super(AutoDis, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@Module.register("log_dense")
class LogDis(layers.Layer):
    def __init__(self,
                 embedding_size: int,
                 hash_num: int,
                 log_base: float,
                 hash_num_dict: Union[Dict[str, int], None] = None,
                 log_base_dict: Union[Dict[str, float], None] = None,
                 emb_size_dict: Union[Dict[str, int], None] = None,
                 rate_feature_list: Union[List[str], None] = None,
                 feature_list: Union[List[str], None] = None,
                 embedding_initializer_config: Union[str, Config, None] = None,
                 embedding_regularizer_config: Union[str, Config, None] = None,
                 embedding_constraint_config: Union[str, Config, None] = None,
                 name: str = "log_dense",
                 **kwargs):
        '''

        Log 函数自动离散化dense特征

        :register name: log_dense

        :config example:
        .. code-block:: jsonnet
           {
                name: 'log_dense',
                type: 'log_dense',
                parameters:{
                    embedding_size: 30,
                    hash_num: 10,
                    log_base: 2.7,
                    embedding_initializer_config:
                        {
                            class_name: 'TruncatedNormal',
                            config: {
                                stddev:0.05
                            }
                        },
                    embedding_regularizer_config:
                        {
                            class_name: 'l1_l2',
                            config: {
                                l1:0,
                                l2:0
                            }
                        }


                },
                inputs: {
                  inputs: 'ref::inputs.`features#continuous`',
                },
                outputs: 'embedding',
            }

        :param embedding_size: embedding 大小
        :param hash_num: hash_num分桶大小
        :param log_base: log_base,取log的底数，决定dense特征的分桶粒度
        :param hash_num_dict: hash_num_dict分桶大小，每个特征单独配置
        :param log_base_dict: log_base_dict,取log的底数，决定dense特征的分桶粒度，每个特征单独配置
        :param emb_size_dict: emb_size_dict,emb_size 字典
        :param rate_feature_list: rate_feature_list dense特征中ctr，cvr等取值范围在[0,1)的特征列表
        :param feature_list: 需要log处理的dense特征列表,如果为None，则默认为全处理
        :param embedding_initializer_config: 初始化配置
        :param embedding_regularizer_config: 正则化配置
        :param embedding_constraint_config:  约束配置,
        :param kwargs:
        '''

        self.embedding_size = embedding_size
        self.hash_num = hash_num
        self.log_base = log_base
        self.hash_num_dict = hash_num_dict if hash_num_dict else {}
        self.log_base_dict = log_base_dict if log_base_dict else {}
        self.emb_size_dict = emb_size_dict if emb_size_dict else {}
        self.rate_feature_list = rate_feature_list if rate_feature_list else []
        self.embedding_dict = dict()
        self.feature_list = feature_list
        self.embedding_initializer_config = convert_config_to_dict(
            embedding_initializer_config if embedding_initializer_config else {})
        self.embedding_regularizer_config = convert_config_to_dict(
            embedding_regularizer_config if embedding_regularizer_config else {})
        self.embedding_constraint_config = convert_config_to_dict(
            embedding_constraint_config if embedding_constraint_config else {})
        self.data_type = "Map"

        super(LogDis, self).__init__(name=name, **kwargs)

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        if not isinstance(input_shape, dict):
            raise TypeError(f"The parameter 'input_shape' of method ‘build’ in class ‘LogDis’ only supports "
                            f"variables of type 'dict', but the current type is '{type(input_shape)}'.")
        weight_initializer = initializers.get(
            self.embedding_initializer_config) if self.embedding_initializer_config else None
        weight_regularizer = regularizers.get(
            self.embedding_regularizer_config) if self.embedding_regularizer_config else None
        weight_constraint = constraints.get(
            self.embedding_constraint_config) if self.embedding_constraint_config else None
        preprocessed_features = list(input_shape.keys()) if self.feature_list is None else self.feature_list
        for feat in preprocessed_features:
            if feat in self.hash_num_dict.keys():
                hash_num = self.hash_num_dict[feat]
            else:
                hash_num = self.hash_num
            if feat in self.emb_size_dict.keys():
                embedding_size = self.emb_size_dict[feat]
            else:
                embedding_size = self.embedding_size
            self.embedding_dict[feat] = Embedding(
                input_dim=hash_num + 1,
                output_dim=embedding_size,
                embeddings_initializer=weight_initializer,
                embeddings_regularizer=weight_regularizer,
                embeddings_constraint=weight_constraint,
                mask_zero=False, name=feat)

        super(LogDis, self).build(input_shape)

    def call(self, inputs: Union[tf.Tensor, Dict[str, tf.Tensor]], **kwargs):
        if not isinstance(inputs, dict):
            raise TypeError(f"The parameter 'inputs' of method ‘call’ in class ‘LogDis’ only supports "
                            f"variables of type 'dict', but the current type is '{type(inputs)}'.")
        embeddings_dict = dict()
        embeddings_dict_para = dict()
        preprocessed_features = list(inputs.keys()) if self.feature_list is None else self.feature_list
        for _, feat in enumerate(preprocessed_features):
            if feat in self.hash_num_dict.keys():
                hash_num = self.hash_num_dict[feat]
            else:
                hash_num = self.hash_num
            if feat in self.log_base_dict.keys():
                log_base = self.log_base_dict[feat]
            else:
                log_base = self.log_base

            tf_clip_value = tf.clip_by_value(inputs[feat], 1e-20, tf.reduce_max(inputs[feat]))
            raw_log_index = tf.math.floor(
                tf.math.divide_no_nan(tf.math.log(tf_clip_value), tf.math.log(float(log_base))))
            if feat in self.rate_feature_list:
                tf_log_index = tf.math.mod(tf.clip_by_value(raw_log_index, -hash_num + 1, 0), hash_num) + 1
            else:
                tf_log_index = tf.math.mod(tf.clip_by_value(raw_log_index, -1, hash_num - 1), hash_num) + 1
            embeddings_dict[feat] = self.embedding_dict.get(feat)(tf_log_index)
            embeddings_dict_para[feat] = self.embedding_dict.get(feat).embeddings
        return embeddings_dict, embeddings_dict_para

    def get_config(self):
        config = {
            'embedding_size': self.embedding_size,
            'hash_num': self.hash_num,
            'log_base': self.log_base,
            'hash_num_dict': self.hash_num_dict,
            'log_base_dict': self.log_base_dict,
            'emb_size_dict': self.emb_size_dict,
            'rate_feature_list': self.rate_feature_list,
            'feature_list': self.feature_list,
            'embedding_initializer_config': self.embedding_initializer_config,
            'embedding_regularizer_config': self.embedding_regularizer_config,
            'embedding_constraint_config': self.embedding_constraint_config
        }
        base_config = super(LogDis, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return (None, self.embedding_size * len(self.feature_list))


@Module.register("bucket")
class Bucket(layers.Layer):
    def __init__(self,
                 embedding_size: int,
                 bin_boundaries_dict: Dict[str, List] = None,
                 num_bins_dict: Dict[str, int] = None,
                 embedding_initializer_config: Union[str, Config, None] = None,
                 embedding_regularizer_config: Union[str, Config, None] = None,
                 embedding_constraint_config: Union[str, Config, None] = None,
                 name: str = "bucket",
                 **kwargs):
        """dense特征分桶

        :register name: bucket

        :config example:
        .. code-block:: jsonnet
            {
                name: 'bucket',
                type: 'bucket',
                parameters: {
                    embedding_size: 30,
                    bin_boundaries_dict: {
                        feature1: [100, 10000, 100000, 1000000, 10000000],
                        feature2: [100, 10000, 100000, 1000000, 10000000],
                    },
                    num_bins_dict:{
                        feature3: 5
                    },
                },
                inputs: {
                    inputs: "ref::inputs.`features#continuous`",
                },
                outputs: 'output_dict',
            },

        :param embedding_size: embedding 大小
        :param bin_boundaries_dict: 手动设置边界值进行分桶
        :param num_bins_dict: 设置分桶数量，根据数据集进行自适应分位数分桶
        :param embedding_initializer_config: 初始化配置
        :param embedding_regularizer_config: 正则化配置
        :param embedding_constraint_config:  约束配置,
        :param kwargs:
        """
        self.embedding_size = embedding_size
        self.bin_boundaries_dict = bin_boundaries_dict if bin_boundaries_dict is not None else {}
        self.num_bins_dict = num_bins_dict if num_bins_dict is not None else {}
        self.embedding_dict = dict()
        self.discretizer = dict()
        self.auto_discretizer = dict()
        self.embedding_initializer_config = convert_config_to_dict(
            embedding_initializer_config if embedding_initializer_config else {})
        self.embedding_regularizer_config = convert_config_to_dict(
            embedding_regularizer_config if embedding_regularizer_config else {})
        self.embedding_constraint_config = convert_config_to_dict(
            embedding_constraint_config if embedding_constraint_config else {})

        super(Bucket, self).__init__(name=name, **kwargs)

    def build(self, input_shape: Dict[str, tf.TensorShape]):
        if not isinstance(input_shape, dict):
            raise TypeError(f"The parameter 'input_shape' of method ‘build’ in class ‘Bucket’ only supports "
                            f"variables of type 'dict', but the current type is '{type(input_shape)}'.")
        weight_initializer = initializers.get(
            self.embedding_initializer_config) if self.embedding_initializer_config else None
        weight_regularizer = regularizers.get(
            self.embedding_regularizer_config) if self.embedding_regularizer_config else None
        weight_constraint = constraints.get(
            self.embedding_constraint_config) if self.embedding_constraint_config else None

        for feature, bin_boundaries in self.bin_boundaries_dict.items():
            bin_num = len(bin_boundaries) + 1
            # 创建边界值分桶层, 根据手动设置边界值分桶
            self.discretizer[feature] = tf.keras.layers.Discretization(bin_boundaries=bin_boundaries, name=feature)
            self.embedding_dict[feature] = Embedding(
                input_dim=bin_num,
                output_dim=self.embedding_size,
                embeddings_initializer=weight_initializer,
                embeddings_regularizer=weight_regularizer,
                embeddings_constraint=weight_constraint,
                mask_zero=False, name=feature)

        for feature, num_bins in self.num_bins_dict.items():
            # 创建自动分桶层，tf2.8.0默认采用分位数自动分桶策略, tf2.6.5不支持自动分桶策略
            self.auto_discretizer[feature] = tf.keras.layers.Discretization(num_bins=num_bins, name=feature)
            self.embedding_dict[feature] = Embedding(
                input_dim=num_bins,
                output_dim=self.embedding_size,
                embeddings_initializer=weight_initializer,
                embeddings_regularizer=weight_regularizer,
                embeddings_constraint=weight_constraint,
                mask_zero=False, name=feature)

        super(Bucket, self).build(input_shape)

    def call(self, inputs: Union[tf.Tensor, Dict[str, tf.Tensor]], **kwargs):
        if not isinstance(inputs, dict):
            raise TypeError(f"The parameter 'inputs' of method ‘call’ in class ‘Bucket’ only supports "
                            f"variables of type 'dict', but the current type is '{type(inputs)}'.")
        embeddings_dict = dict()

        for feature in self.bin_boundaries_dict.keys():
            bin_idx = tf.squeeze(self.discretizer.get(feature)(inputs[feature]), axis=-1)
            embeddings_dict[feature] = self.embedding_dict.get(feature)(bin_idx)

        for feature in self.num_bins_dict.keys():
            bin_idx = tf.squeeze(self.auto_discretizer.get(feature)(inputs[feature]), axis=-1)
            embeddings_dict[feature] = self.embedding_dict.get(feature)(bin_idx)

        return embeddings_dict
