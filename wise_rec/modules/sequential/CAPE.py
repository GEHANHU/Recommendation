#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from modelflow.integrations.tensorflow.modules import DNN
from modelflow.integrations.tensorflow.modules.module import Module
from modelflow.common import Config


class CAPE(Layer):
    """ CapeDinLayer中的cape模块
        :param scope_name: 位置参数的命名空间
        :param cape_unit: cape参数,对输入层做scale
        :param cape_len: 上下文长度
        :param emb_size: emb_size
        :param inverse: 权重是否反向
        :param kwargs: 其他参数
    """

    def __init__(self, scope_name, cape_unit, cape_len=7, emb_size=64, gather_method="gather", inverse=False, **kwargs):
        self.scope_name = scope_name
        self.cape_unit = int(cape_unit)
        self.cape_len = int(cape_len)
        self.emb_size = emb_size
        self.inverse = inverse
        self.pre_proj = None
        self.pos_emb = None
        self.gather_method = gather_method
        super(CAPE, self).__init__(**kwargs)

    def build(self, input_shape):
        self.pos_emb = self.add_weight(name=f"{self.scope_name}_pos_emb",
                                       dtype=tf.float32,
                                       shape=(1, self.cape_unit, self.cape_len),
                                       initializer=tf.keras.initializers.Zeros(),
                                       trainable=True)
        if self.emb_size and (self.cape_unit != self.emb_size):
            self.pre_proj = Dense(self.cape_unit)
        super(CAPE, self).build(input_shape)

    def call(self, inputs, **kwargs):
        target, attention_weights = inputs
        if self.inverse:
            att_normal = 1 - tf.sigmoid(attention_weights)
        else:
            att_normal = tf.sigmoid(attention_weights)
        p_var = tf.cumsum(tf.reverse(att_normal, axis=[-1]), axis=-1)
        p_var = tf.reverse(p_var, axis=[-1])
        p_var = tf.clip_by_value(p_var, clip_value_min=0, clip_value_max=self.cape_len - 1)
        p_ceil = tf.cast(tf.math.ceil(p_var), tf.int32)
        p_floor = tf.cast(tf.math.floor(p_var), tf.int32)
        if self.pre_proj is not None:
            target = self.pre_proj(target)
            target = tf.nn.swish(target)
        e_var = tf.matmul(target, self.pos_emb)
        if self.gather_method == "gather_nd":
            p_last_dim = p_ceil.get_shape().as_list()[-1]
            mid_dim = e_var.get_shape().as_list()[1]
            batch_size = tf.shape(p_ceil)[0]
            batch_indices = tf.range(batch_size)[:, None, None]
            row_indices = tf.range(mid_dim)[None, :, None]
            batch_indices = tf.tile(batch_indices, [1, mid_dim, p_last_dim])
            row_indices = tf.tile(row_indices, [batch_size, 1, p_last_dim])
            pceil_indices = tf.stack([batch_indices, row_indices, p_ceil], axis=-1)
            pceil_indices = tf.cast(pceil_indices, tf.int32)
            pfoor_indices = tf.stack([batch_indices, row_indices, p_floor], axis=-1)
            pfoor_indices = tf.cast(pfoor_indices, tf.int32)
            e_ceil = tf.gather_nd(e_var, pceil_indices)
            e_floor = tf.gather_nd(e_var, pfoor_indices)
        else:
            e_ceil = tf.gather(e_var, p_ceil, batch_dims=2, axis=2)
            e_floor = tf.gather(e_var, p_floor, batch_dims=2, axis=2)
        p_p_floor = p_var - tf.cast(p_floor, tf.float32)
        res = p_p_floor * e_ceil + (1 - p_p_floor) * e_floor
        return res


@Module.register("cape")
class CapeLayer(Layer):
    """ :register name: cape
        :config example:
        {
        name: "cape",
        type: "cape",
        parameters: {
            scope_name: 'cape_layer',
            cape_unit: 32,
            cape_len: 7,
            emb_size: 64,
            dropout_rate: 0.1,
            use_batch_norm: true,
            layer_units: [1024,256,128],
            use_cape: true,
            hidden_activation: 'dice',
            output_activation: 'relu',
            use_softmax: true,
            inverse: false,
            is_reduce: false,
        }
        :param scope_name: cape中位置参数的命名空间
        :param cape_unit: cape参数,对输入层做scale
        :param cape_len: 上下文长度
        :param emb_size: emb_size
        :param dropout_rate: 丢弃神经元的比率
        :param use_batch_norm: 是否使用bn
        :param layer_units: cape内部dnn的隐藏层
        :param use_cape: 是否使用cape
        :param hidden_activation: 隐藏层的激活函数
        :param output_activation: 最终输出层的激活函数
        :param use_softmax: 输出权重是否做softmax
        :param inverse: 权重参数是否反向加权
        :param is_reduce: 是否对输出的axis=1维度做reduce
        :param kwargs: 其他参数
    """

    def __init__(self, scope_name, cape_unit=64, cape_len=7, emb_size=64, dropout_rate=0.1,
                 use_batch_norm=False, layer_units=None, use_cape=True, hidden_activation='relu',
                 output_activation='relu', use_softmax=False, gather_method="gather",
                 inverse=False, is_reduce=False, **kwargs):
        self.scope_name = scope_name
        self.cape_unit = cape_unit
        self.cape_len = cape_len
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.bn_cfg = Config({"type": "batch_norm"}) if self.use_batch_norm else None
        self.layer_units = layer_units
        self.use_cape = use_cape
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_softmax = use_softmax
        self.gather_method = gather_method
        self.inverse = inverse
        self.is_reduce = is_reduce
        if self.layer_units is None:
            self.layer_units = [128]
        self.mlp_layer_dims = self.layer_units + [1]
        self.dnn = None
        self.position = None
        super(CapeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dnn = DNN(hidden_dims=self.mlp_layer_dims,
                       hidden_activation=self.hidden_activation,
                       output_activation=self.output_activation,
                       dropout_rate=self.dropout_rate,
                       norm=self.bn_cfg)
        if self.use_cape:
            self.position = CAPE(self.scope_name, self.cape_unit, cape_len=self.cape_len,
                                 emb_size=self.emb_size,
                                 gather_method=self.gather_method,
                                 inverse=self.inverse)
        super(CapeLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # item特征和序列特征
        target, query = inputs
        item_eb = target
        item_his_eb = query
        seq_len = item_his_eb.get_shape().as_list()[1]
        item_eb_tile = tf.tile(item_eb, [1, seq_len, 1])
        att_layer_1 = tf.concat([item_eb_tile, item_his_eb, item_eb_tile - item_his_eb,
                                 item_eb_tile * item_his_eb], axis=-1)
        att_layer_3 = self.dnn(att_layer_1)
        scores = tf.reshape(att_layer_3, [-1, seq_len])
        if self.use_cape:
            pe = self.position((item_eb_tile, tf.tile(tf.expand_dims(scores, axis=1), [1, seq_len, 1])))
            scores += tf.reduce_mean(pe, axis=-1)
        if self.use_softmax:
            scores = tf.nn.softmax(scores)
        output = tf.multiply(tf.expand_dims(scores, axis=-1), item_his_eb)
        if self.is_reduce:
            output = tf.reduce_mean(output, axis=1)
        return output
