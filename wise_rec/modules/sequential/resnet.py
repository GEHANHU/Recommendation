#  !/usr/bin python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from typing import Union, List, Dict

import tensorflow as tf
from tensorflow.keras import layers

from modelflow.common import Config
from modelflow.common.exception import ConfigError
from modelflow.integrations.tensorflow.modules.module import Module
from modelflow.integrations.tensorflow.modules.dnn import DNN


@Module.register('res_net')
class ResNet(layers.Layer):
    """ ResNetLayer

         原始论文：`ResNet: Deep Residual Learning for Image Recognition
        <https://arxiv.org/pdf/2302.01115>`_

        :register name: res_net
        :param out_channels: 输出的维度
        :param mid_channels: 中间层的维度
        :param stride: 卷积步长
        :param kernel_size: 卷积核大小
        :param act_func: 每层输出的激活函数
        :param bn: 是否使用batch_norm
        :param block_num:  res_block数量
        :param kwargs: 其他参数
    """

    def __init__(self, out_channels: int, mid_channels: int = None, stride: int = 1, kernel_size: int = 3,
                 act_func='relu', bn=True, block_num: int = 3, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.encoder_blocks = tf.keras.Sequential([ResConvBlock(out_channels, mid_channels, stride=stride,
                                                                kernel_size=kernel_size, act_func=act_func, bn=bn,
                                                                **kwargs) for _ in range(block_num)])

    def build(self, input_shape):
        """
        构建 ResNet层
        :param input_shape: 输入序列数据shape为(batch_size, seq_length, feature_dim)
        """
        if len(input_shape) != 3:
            raise ConfigError("ResNet requires input shape [batch_size, seq_length, feature_dim].")
        super(ResNet, self).build(input_shape)

    def call(self, inputs: tf.Tensor):
        """
        :param inputs: 序列emb
        :return: 最终输出和输入shape相同
        """
        encoder_output = self.encoder_blocks(inputs)
        return encoder_output


class ResConvBlock(layers.Layer):
    """ 单个ResConvBlock层

         原始论文：`ResNet: Deep Residual Learning for Image Recognition
        <https://arxiv.org/pdf/2302.01115>`_

        :param out_channels: 输出的维度
        :param mid_channels: 中间层的维度
        :param stride: 卷积步长
        :param kernel_size: 卷积核大小
        :param act_func: 每层输出的激活函数
        :param bn: 是否使用batch_norm
        :param kwargs: 其他参数
    """

    def __init__(self, out_channels: int, mid_channels: int = None, stride: int = 1,
                 kernel_size: int = 3, act_func='relu', bn=True, **kwargs):
        super(ResConvBlock, self).__init__()
        print("build ResConvBlock")
        self.out_channels = out_channels
        if mid_channels is None:
            # 第一层的输出维度,不定义默认和输出层相同
            self.mid_channels = out_channels
        else:
            self.mid_channels = mid_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = act_func
        self.batch_norm = bn
        if self.batch_norm:
            self.norm_layer_1 = layers.BatchNormalization()
            self.norm_layer_2 = layers.BatchNormalization()
        self.activation_layer = Module.from_config(self.activation, tags=["activation"])

        self.conv1d_1 = layers.Conv1D(filters=self.mid_channels, kernel_size=self.kernel_size, strides=self.stride,
                                      padding="same")
        self.conv1d_2 = layers.Conv1D(filters=self.out_channels, kernel_size=self.kernel_size, strides=self.stride,
                                      padding="same")
        # stride>1时,对conv输出扩维恢复至原shape,len必须整除self.stride
        if self.stride != 1:
            self.up_sampling = layers.UpSampling1D(size=self.stride)
        self.add = layers.Add()

    def build(self, input_shape):
        """
        :param input_shape: 输入序列数据shape为(batch_size, seq_length, feature_dim)
        """
        if len(input_shape) != 3:
            raise ConfigError("ResConvBlock requires input shape [batch_size, seq_length, feature_dim].")
        super(ResConvBlock, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config["stride"] = self.stride
        config["kernel_size"] = self.kernel_size
        config["mid_channels"] = self.mid_channels
        return config

    def call(self, inputs: tf.Tensor):
        """
        :param inputs: 序列emb
        :return: 最终输出和输入shape相同
        """
        x = inputs
        x = self.conv1d_1(x)
        if self.batch_norm:
            x = self.norm_layer_1(x)
        x = self.activation_layer(x)
        x = self.conv1d_2(x)
        if self.batch_norm:
            x = self.norm_layer_2(x)
        if self.stride != 1:
            x = self.up_sampling(x)
        x = self.add([x, inputs])
        x = self.activation_layer(x)
        return x


class CaPE(layers.Layer):
    """ CapeLayer中的cape模块

         原始论文：`CAPE: A Contextual-Aware Position Encoding for Sequential Recommendation
        <https://arxiv.org/abs/2502.09027`_

        :param cape_c: cape参数
        :param cape_len: 上下文长度
        :param att_type: 内部attention方式
        :param attention_hidden_units: 采用din_attention时的mlp结构
        :param attention_activation: 采用din_attention时的激活函数
        :param attention_dropout: 采用din_attention时的dropout
        :param emb_dim: emb_size
        :param is_inverse: 权重是否反向
        :param batch_norm: 是否使用batch_norm
        :param kwargs: 其他参数
    """
    def __init__(self, cape_c, cape_len=None, att_type='dot', attention_hidden_units=None,
                 attention_activation="dice", attention_dropout=0.1, emb_dim=None, is_inverse=False,
                 batch_norm=False, **kwargs):
        super(CaPE, self).__init__()
        self.T = cape_len if cape_len is not None else cape_c
        self.att_type = att_type
        self.pre_proj = None
        self.is_inverse = is_inverse
        if emb_dim and (cape_c != emb_dim):
            self.pre_proj = layers.Dense(cape_c)
        if attention_hidden_units is None:
            attention_hidden_units = [16, 4]
        if self.att_type == "bilinear":
            self.W_kernel = tf.Variable(tf.eye(cape_c), trainable=True)
        elif self.att_type == "din":
            mlp_layer_dims = [cape_c * 4] + attention_hidden_units + [1]
            bn_cfg = Config({"type": "batch_norm"}) if batch_norm else None
            self.attn_mlp = DNN(hidden_dims=mlp_layer_dims,
                                hidden_activation=Module.from_config(attention_activation, tags=["activation"]),
                                output_activation=Module.from_config(attention_activation, tags=["activation"]),
                                dropout_rate=attention_dropout,
                                norm=bn_cfg)

        self.pos_emb = self.add_weight(name='pe_wt', shape=(1, cape_c, self.T),
                                       initializer=tf.keras.initializers.Zeros(), trainable=True)

    def call(self, inputs):
        target, attention_weights = inputs
        if self.is_inverse:
            G = 1 - tf.sigmoid(attention_weights)
        else:
            G = tf.sigmoid(attention_weights)
        P = tf.cumsum(tf.reverse(G, axis=[-1]), axis=-1)
        P = tf.reverse(P, axis=[-1])
        P = tf.clip_by_value(P, clip_value_min=0, clip_value_max=self.T - 1)
        P_ceil = tf.cast(tf.math.ceil(P), tf.int32)
        P_floor = tf.cast(tf.math.floor(P), tf.int32)
        if self.pre_proj is not None:
            target = self.pre_proj(target)
            target = target * tf.sigmoid(target)
        if self.att_type == 'dot':
            E = tf.matmul(target, self.pos_emb)
        elif self.att_type == 'bilinear':
            E = target @ (self.W_kernel @ self.pos_emb)
        else:
            tile_ratio = self.T // tf.shape(target)[1]
            target_emb = tf.tile(target, [1, tile_ratio, 1])
            pos_emb = tf.tile(tf.transpose(self.pos_emb, perm=[0, 2, 1]), [tf.shape(target_emb)[0], 1, 1])
            din_concat = tf.concat([target_emb, pos_emb, target_emb - pos_emb, target_emb * pos_emb], axis=-1)
            E = self.attn_mlp(tf.reshape(din_concat, [-1, 4 * tf.shape(target_emb)[-1]]))
            E = tf.reshape(E, [tf.shape(target_emb)[0], 1, -1])
        E_ceil = tf.gather(E, P_ceil, batch_dims=2, axis=-1)
        E_floor = tf.gather(E, P_floor, batch_dims=2, axis=-1)
        P_P_floor = P - tf.cast(P_floor, tf.float32)
        E = P_P_floor * E_ceil + (1 - P_P_floor) * E_floor
        return E


@Module.register(name='cape')
class CapeLayer(layers.Layer):
    """ CapeLayer

         原始论文：`CAPE: A Contextual-Aware Position Encoding for Sequential Recommendation
        <https://arxiv.org/abs/2502.09027`_

        :register name: cape
        :param dropout_rate: dropout层rate
        :param batch_norm: 是否使用batch_norm
        :param mlp: attention结构中dnn的配置
        :param use_cape: 是否使用CAPE权重
        :param cape_len: cape上下文长度
        :param cape_att_type: cape内部attention方式
        :param cape_act_func: cape内部激活函数
        :param cape_c: cape参数
        :param cape_inverse: cape是否反向
        :param use_softmax: 输出权重是否做softmax
        :param is_reduce: 是否对输出的axis=1维度做reduce
        :param kwargs: 其他参数
    """

    def __init__(self, dropout_rate=0.1, batch_norm=False, mlp=None, use_cape=True, cape_len=5, cape_att_type="dot",
                 cape_act_func='dice', cape_c=None, cape_inverse=False, use_softmax=False, is_reduce=False, **kwargs):
        # is_reduce是否对(b,1,e) reduce到b,1 兼容不同结构
        super(CapeLayer, self).__init__()

        self.mlp_dims = mlp
        if self.mlp_dims is None:
            self.mlp_dims = [32]
        self.dropout_rate = dropout_rate
        self.bn_cfg = Config({"type": "batch_norm"}) if batch_norm else None
        self.cape_len = cape_len
        self.cape_att_type = cape_att_type
        self.cape_act_func = cape_act_func
        self.cape_c = cape_c
        self.cape_inverse = cape_inverse
        self.use_cape = use_cape
        self.use_softmax = use_softmax
        self.is_reduce = is_reduce
        self.mlp = None
        self.position = None
        self.seq_len = None
        self.item_pre_proj = None

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ConfigError(f"Unexpected inputs dimensions {input_shape} in {self.__class__.__name__} layer.")
        item_shape = input_shape[0]
        item_seq_shape = input_shape[1]
        self.seq_len = item_seq_shape[1]
        emb_dim = item_seq_shape[-1]
        if item_shape[-1] != item_seq_shape[-1]:
            self.item_pre_proj = layers.Dense(emb_dim)
        if self.cape_c is None:
            self.cape_c = emb_dim
        mlp_layer_dims = [4 * emb_dim] + self.mlp_dims + [1]
        self.mlp = DNN(hidden_dims=mlp_layer_dims,
                       hidden_activation=self.cape_act_func,
                       output_activation=self.cape_act_func,
                       dropout_rate=self.dropout_rate,
                       norm=self.bn_cfg)
        if self.use_cape:
            self.position = CaPE(cape_c=self.cape_c, cape_len=self.cape_len, attention_dropout=self.dropout_rate,
                                 att_type=self.cape_att_type, emb_dim=emb_dim,
                                 attention_activation=self.cape_act_func, attention_hidden_units=mlp_layer_dims,
                                 is_inverse=self.cape_inverse)
        super(CapeLayer, self).build(input_shape)

    def call(self, inputs):
        item_eb, item_seq_eb = inputs
        if self.item_pre_proj is not None:
            item_eb = self.item_pre_proj(item_eb)
        item_eb_tile = tf.tile(item_eb, [1, self.seq_len, 1])
        att_layer_1 = tf.concat([item_eb_tile, item_seq_eb, item_eb_tile - item_seq_eb,
                                 item_eb_tile * item_seq_eb], axis=-1)
        att_layer_3 = self.mlp(att_layer_1)
        scores = tf.reshape(att_layer_3, [-1, self.seq_len])

        if self.use_cape:
            pe = self.position((item_eb_tile, tf.tile(tf.expand_dims(scores, axis=1), [1, self.seq_len, 1])))
            scores += tf.reduce_mean(pe, axis=-1)
        scores = tf.reshape(scores, [-1, 1, self.seq_len])
        if self.use_softmax:
            scores = tf.nn.softmax(scores)
        output = tf.matmul(scores, item_seq_eb)
        if self.is_reduce:
            output = tf.reduce_sum(output, axis=1)

        return output
