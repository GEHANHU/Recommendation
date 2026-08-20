#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2025. All rights reserved.
from typing import Dict, Iterable

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding

from modelflow.common import Config, logger
from modelflow.data import FeatureManager
from modelflow.integrations.tensorflow.modules.module import Module


@Module.register("dmr")
class DMR(layers.Layer):
    def __init__(self, name, dmr_specific_params: Config, is_u2i_use_pos_emb=False, is_i2i_use_pos_emb=False, **kwargs):
        ''' DMR 输出用户的兴趣emb
            {
            name: 'DMR_embedding',
            type: 'dmr',
            parameters: {
              dmr_specific_params: {
                target_features: [
                        "discrete1"
                ],
                sequence_features: [
                        "seq_discrete1"
                ],
                u2i_config: {
                    type: 'dnn',
                    hidden_dims: [64, 32, 1],
                    hidden_activation: ['p_relu','p_relu'],
                    output_activation: "tanh"
                },
                i2i_config: {
                    type: 'dnn',
                    hidden_dims: [64, 32, 1],
                    hidden_activation: ['p_relu','p_relu'],
                    output_activation: "tanh"
                }
              },
              is_u2i_use_pos_emb: false,
              is_i2i_use_pos_emb: false
            },
            inputs:{
               inputs: 'ref::eta.embedding_short'
            },
            outputs: ["i2i_output", "u2i_output"]
          }

        '''
        super().__init__(name=name)

        self.target_features = dmr_specific_params['target_features']
        self.long_sequence_features = dmr_specific_params['sequence_features']
        self.u2i_config = dmr_specific_params.get('u2i_config')
        self.i2i_config = dmr_specific_params.get('i2i_config')
        self.is_u2i_use_pos_emb = is_u2i_use_pos_emb
        self.is_i2i_use_pos_emb = is_i2i_use_pos_emb
        self.MASK_FILL = -1e9
        self.u2i_network = {}
        self.i2i_network = {}
        self.i2i_pos_encoder = {}
        self.u2i_pos_encoder = {}
        self.positions = {}
        self.embedding_size = {}

    def build(self, input_shape):
        self.embedding_size = self.get_embedding_size(input_shape)
        self._init_dmr_layer(self.long_sequence_features)
        self._init_pos_encoder(self.long_sequence_features, input_shape)

    def _init_pos_encoder(self, long_seq_feats, input_shape):
        if isinstance(input_shape, dict):
            for long_seq_feat in long_seq_feats:
                if isinstance(input_shape[long_seq_feat], list):
                    seq_len = input_shape[long_seq_feat][0][-2]
                else:
                    seq_len = input_shape[long_seq_feat][-2]
                self.positions[long_seq_feat] = tf.range(1, seq_len + 1, dtype=tf.int32)
                if self.is_u2i_use_pos_emb:
                    self.u2i_pos_encoder[long_seq_feat] = Embedding(input_dim=seq_len + 1,
                                                                    output_dim=self.embedding_size[long_seq_feat],
                                                                    mask_zero=False)
                if self.is_i2i_use_pos_emb:
                    self.i2i_pos_encoder[long_seq_feat] = Embedding(input_dim=seq_len + 1,
                                                                    output_dim=self.embedding_size[long_seq_feat],
                                                                    mask_zero=False)

    def _init_dmr_layer(self, long_sequence_features):
        for long_sequence_feature in long_sequence_features:
            self.u2i_network[long_sequence_feature] = Module.from_config(self.u2i_config)
            self.i2i_network[long_sequence_feature] = Module.from_config(self.i2i_config)

    def get_embedding_size(self, input_shape):
        embedding_size = {}
        if isinstance(input_shape, dict):
            for target_feat, long_seq_feat in zip(self.target_features, self.long_sequence_features):
                if isinstance(input_shape[target_feat], list):
                    tmp_target_size = input_shape[target_feat][0][-1]
                else:
                    tmp_target_size = input_shape[target_feat][-1]
                if isinstance(input_shape[long_seq_feat], list):
                    tmp_long_seq_size = input_shape[long_seq_feat][0][-1]
                else:
                    tmp_long_seq_size = input_shape[long_seq_feat][-1]
                embedding_size[long_seq_feat] = min(tmp_target_size, tmp_long_seq_size)
        else:
            for long_seq_feat in self.long_sequence_features:
                embedding_size[long_seq_feat] = input_shape[0][-1]
        return embedding_size

    def call(self, inputs, mask=None, **kwargs):
        '''
            计算DMR(Deep Match to Rank)的 i2i 和 u2i embedding
            支持dict / list of dict 类型,建议使用 但必须包含x_target x_long_seq 对应维度如下
            x_target: (B, 1, D) or (B, D)
            x_long_seq: (B, 1, L, D) or (B, L, D)
            reduced_long_seq_mask: (B, L)
            outputs shape:  i2i_outputs[(B, D)n], u2i_outputs[(B, 1)n]
        '''
        i2i_outputs = []
        u2i_outputs = []
        if isinstance(inputs, list):
            features = {}
            for input_dict in inputs:
                if not isinstance(input_dict, dict):
                    logger.error("inputs must be dict or list of dict")
                features.update(input_dict)
        else:
            features = inputs
        for target_feat, long_seq_feat in zip(self.target_features, self.long_sequence_features):
            x_target, x_long_seq, reduced_long_seq_mask, i2i_pos_emb, u2i_pos_emb = self.get_emb(target_feat,
                                                                                                 long_seq_feat,
                                                                                                 features, mask)
            if len(x_target.shape) == 3:
                x_target = tf.squeeze(x_target, axis=1)
            if len(x_target.shape) != 2:
                logger.error("target_feat shape must be [batch_size, D]")
            if len(x_long_seq.shape) == 4:
                x_long_seq = tf.squeeze(x_long_seq, axis=1)
            if len(x_long_seq.shape) != 3:
                logger.error("x_long_seq shape must be [batch_size, L, D]")


            i2i_out, u2i_out = self.get_dmr_output(reduced_long_seq_mask, long_seq_feat, x_long_seq,
                                                   x_target, i2i_pos_emb, u2i_pos_emb)
            i2i_outputs.append(i2i_out)
            u2i_outputs.append(u2i_out)

        return i2i_outputs, u2i_outputs

    def get_dmr_output(self, reduced_long_seq_mask, long_seq_feat, x_long_seq, x_target, i2i_pos_emb, u2i_pos_emb):
        seq_length = x_long_seq.shape[1]
        query = x_target
        x_target = tf.tile(tf.expand_dims(x_target, axis=1), [1, seq_length, 1])  # [B, L, D]

        # 计算item之间的attention，item to item网络
        if i2i_pos_emb is not None:
            i2i_attention_input = tf.concat([x_target, x_long_seq, i2i_pos_emb, x_long_seq - i2i_pos_emb,
                                             x_target - x_long_seq, x_target * x_long_seq, x_long_seq * i2i_pos_emb],
                                            axis=-1)  # [B, L, k]
        else:
            i2i_attention_input = tf.concat([x_target, x_long_seq,
                                             x_target - x_long_seq, x_target * x_long_seq], axis=-1)
        i2i_weight = tf.transpose(self.i2i_network[long_seq_feat](i2i_attention_input), [0, 2, 1])  # [B, 1, L]
        reduced_long_seq_mask = tf.cast(reduced_long_seq_mask, tf.float32)
        reduced_long_seq_mask = (1 - reduced_long_seq_mask) * self.MASK_FILL  # (B, L)
        i2i_weight = i2i_weight + tf.expand_dims(reduced_long_seq_mask, axis=-2)  # [B, 1, L]
        i2i_weight = tf.nn.softmax(i2i_weight, axis=-1)  # [B, 1, L]
        i2i_output_emb = tf.matmul(i2i_weight, x_long_seq)  # [B, 1, D]
        i2i_output_emb = tf.squeeze(i2i_output_emb, axis=1)  # 形状: [B, D]
        i2i_output = tf.reshape(i2i_output_emb, [-1, self.embedding_size[long_seq_feat]])

        # 计算用户与兴趣之间的attention， user to item网络
        if u2i_pos_emb is not None:
            u2i_attention_input = tf.concat([x_long_seq, u2i_pos_emb,
                                             x_long_seq - u2i_pos_emb, x_long_seq * u2i_pos_emb], axis=-1)  # [B, L, k]
        else:
            u2i_attention_input = tf.concat([x_long_seq], axis=-1)  # [B, L, k]
        u2i_weight = tf.transpose(self.u2i_network[long_seq_feat](u2i_attention_input), [0, 2, 1])  # [B, 1, L]
        u2i_weight = u2i_weight + tf.expand_dims(reduced_long_seq_mask, axis=-2)   # [B, 1, L]
        u2i_weight = tf.nn.softmax(u2i_weight, axis=-1)  # [B, 1, L]
        u2i_output_emb = tf.matmul(u2i_weight, x_long_seq)  # [B, 1, D]
        u2i_output_emb = tf.squeeze(u2i_output_emb, axis=1)  # 形状: [B, D]
        u2i_output_emb = tf.reshape(u2i_output_emb, [-1, self.embedding_size[long_seq_feat]])
        u2i_output = tf.reduce_sum(u2i_output_emb * query, axis=-1, keepdims=True)  # 形状: [B, 1]

        return i2i_output, u2i_output

    def get_emb(self, target_feat, long_seq_feat, inputs, mask):
        x_target, _ = tf.split(inputs[target_feat],
                               num_or_size_splits=[self.embedding_size[long_seq_feat], -1], axis=-1)
        x_long_seq, _ = tf.split(inputs[long_seq_feat],
                              num_or_size_splits=[self.embedding_size[long_seq_feat], -1], axis=-1)
        i2i_pos_emb = None
        u2i_pos_emb = None

        # 生成位置向量
        batch_positions = tf.tile(tf.expand_dims(self.positions[long_seq_feat], axis=0),
                                  [tf.shape(x_long_seq)[0], 1])  # [B, L]
        if self.is_i2i_use_pos_emb:
            i2i_pos_emb = self.i2i_pos_encoder[long_seq_feat](batch_positions)
        if self.is_u2i_use_pos_emb:
            u2i_pos_emb = self.u2i_pos_encoder[long_seq_feat](batch_positions)

        if mask is not None:
            reduced_long_seq_mask = mask[long_seq_feat]
        else:
            reduced_long_seq_mask = tf.cast(tf.ones(tf.shape(x_long_seq)[:-1]), tf.bool)

        return x_target, x_long_seq, reduced_long_seq_mask, i2i_pos_emb, u2i_pos_emb



