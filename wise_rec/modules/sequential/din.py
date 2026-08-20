from typing import Dict, Iterable

from tensorflow.keras import layers

from modelflow.common import Config, logger
from modelflow.data import FeatureManager
from modelflow.integrations.tensorflow.modules.module import Module
import tensorflow as tf


@Module.register("din")
class DIN(layers.Layer):

    def __init__(self, name, din_specific_params: Config, use_same_dnn: bool = False, **kwargs):
        ''' DIN序列模型 返回序列emb
          {
            name: 'din_embedding',
            type: 'din',
            parameters: {
              din_specific_params: {
                // target序列和点击序列特征数要一致 且保证序列长度一致
                target_feature: [
                        "discrete1", "discrete2",
                ],
                sequence_features: [
                        ["seq_discrete1", "seq_discrete2"], ["seq_discrete3", "seq_discrete4"]
                ],
                dnn_config: {
                    type: 'dnn',
                    hidden_dims: [32, 1],
                    hidden_activation: ['dice']
                }
              },
              use_same_dnn: false
            },
            inputs:{
               inputs: 'ref::eta.embedding_short'  // adapt two input, look at `build` `call` for detail
            },
            outputs: ["embedding"]
          }

        '''
        super().__init__(name=name, **kwargs)

        self.target_features = din_specific_params['target_feature']
        self.long_sequence_features = din_specific_params['sequence_features']
        self.dnn_config = din_specific_params.get('dnn_config')
        self.use_same_din = use_same_dnn
        self.MASK_FILL = -32768.0
        self.dnn = {}
        self.embedding_size = None

    def build(self, input_shape):
        # dict 形式传递 {target_feature: [tensor_target, tensor_longseq, ...]} or {feature: feature_tensor}

        self.embedding_size = self.get_embedding_size(input_shape)
        self._init_dnn_layer(self.target_features)

    def _init_dnn_layer(self, target_features):
        if self.use_same_din:  # 不同组的长序列 采用不同的DIN
            dnn = Module.from_config(self.dnn_config)
            for target_feat in target_features:
                if target_feat in self.target_features:
                    self.dnn[target_feat] = dnn
        else:
            for target_feat in target_features:
                if target_feat in self.target_features:
                    self.dnn[target_feat] = Module.from_config(self.dnn_config)

    def get_embedding_size(self, input_shape):
        if isinstance(input_shape, dict):
            for target_feature in input_shape.keys():
                if isinstance(input_shape[target_feature], list):
                    return input_shape[target_feature][0][-1]
                else:
                    return input_shape[target_feature][-1]
        else:
            return input_shape[0][-1]

    def call(self, inputs, mask=None, **kwargs):
        '''
            计算DIN(deep interest network)的序列表征
            支持dict/list 类型 但必须包含x_target x_long_seq 对应维度如下
            x_target: (B, 1, H)
            x_long_seq: (B, N, L, H) or (B, L, H)
            reduced_long_seq_mask: (B, N, L)
            outputs shape:  (B, N*H)
        '''

        output = []
        for target_feat, long_seq_feats in zip(self.target_features, self.long_sequence_features):
            if isinstance(inputs[target_feat], list):
                x_target, x_long_seq = inputs[target_feat][:2]
                if mask is not None:
                    reduced_long_seq_mask = mask[target_feat][1]
                else:
                    reduced_long_seq_mask = tf.cast(tf.ones(tf.shape(x_long_seq)[:-1]), tf.bool)
            else:
                x_target, x_long_seq, reduced_long_seq_mask = self.get_emb(target_feat, long_seq_feats, inputs, mask)

            if len(x_target.shape) == 2:
                x_target = tf.expand_dims(x_target, axis=1)
            if len(x_target.shape) != 3:
                logger.error("target_feat shape must be [batch_size, 1, H]")

            if len(x_long_seq.shape) == 3:
                x_long_seq = tf.expand_dims(x_long_seq, axis=1)
            if len(x_long_seq.shape) != 4:
                logger.error("x_long_seq shape must be [batch_size, N, L, H]")

            din_output_emb = self.get_din_pooling(reduced_long_seq_mask, target_feat, x_long_seq, x_target)
            output.append(din_output_emb)

        return tf.concat(output, axis=-1)

    def get_din_pooling(self, reduced_long_seq_mask, target_feat, x_long_seq, x_target):
        n_seq = x_long_seq.shape[1]
        seq_length = x_long_seq.shape[2]
        x_target = tf.tile(tf.expand_dims(x_target, axis=1), [1, n_seq, seq_length, 1])  # [B, N, L, D]
        attention_input = tf.concat([x_target, x_long_seq, x_target - x_long_seq, x_target * x_long_seq], axis=-1)
        weight = tf.transpose(self.dnn[target_feat](attention_input), [0, 1, 3, 2])  # [B, N, 1, k]
        reduced_long_seq_mask = tf.cast(reduced_long_seq_mask, x_long_seq.dtype)
        reduced_long_seq_mask = (1 - reduced_long_seq_mask) * self.MASK_FILL  # (B, N, L)
        weight = weight + tf.expand_dims(reduced_long_seq_mask, axis=-2)  # [B, N, 1, L]
        weight = tf.nn.softmax(weight, axis=-1)  # [B, N, 1, L]
        din_output_emb = tf.matmul(weight, x_long_seq)  # [B, N, 1, D]
        din_output_emb = tf.squeeze(din_output_emb, axis=2)  # 形状: [B, N, D]
        din_output_emb = tf.reshape(din_output_emb, [-1, n_seq * self.embedding_size])

        return din_output_emb

    def get_emb(self, target_feat, long_seq_feats, inputs, mask):
        x_target = inputs[target_feat]
        x_long_seq = []
        reduced_long_seq_mask = []
        for long_seq_feat in long_seq_feats:
            x_long_seq.append(inputs[long_seq_feat])
            if mask is not None and mask[long_seq_feat] is not None:
                mask_tensor = mask[long_seq_feat]
            else:
                mask_tensor = tf.cast(tf.ones(tf.shape(inputs[long_seq_feat])[:-1]), tf.bool)
            reduced_long_seq_mask.append(mask_tensor)
        x_long_seq = tf.stack(x_long_seq, axis=1)
        reduced_long_seq_mask = tf.stack(reduced_long_seq_mask, axis=1)

        return x_target, x_long_seq, reduced_long_seq_mask


@Module.register("din_v2")
class DIN_V2(DIN):

    def __init__(self, name, din_specific_params: Config, use_same_dnn: bool = False):
        ''' DIN序列模型 返回序列emb
          {
            name: 'din_embedding',
            type: 'din',
            parameters: {
              din_specific_params: {
                // target序列和点击序列特征数要一致 且保证序列长度一致
                target_feature: [
                        "discrete1", "discrete2",
                ],
                sequence_features: [
                        ["seq_discrete1", "seq_discrete2"], ["seq_discrete3", "seq_discrete4"]
                ],
                dnn_config: {
                    type: 'dnn',
                    hidden_dims: [32, 1],
                    hidden_activation: ['dice']
                }
              },
              use_same_dnn: false
            },
            inputs:{
               inputs: 'ref::eta.embedding_short'  // adapt two input, look at `build` `call` for detail
            },
            outputs: ["embedding"]
          }

        '''
        super(DIN_V2, self).__init__(name=name, din_specific_params=din_specific_params, use_same_dnn=use_same_dnn)

    def build(self, input_shape):
        super(DIN_V2, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        '''
            计算DIN(deep interest network)的序列表征
            支持dict/list 类型 但必须包含x_target x_long_seq 对应维度如下
            x_target: (B, 1, H)
            x_long_seq: (B, N, L, H) or (B, L, H)
            reduced_long_seq_mask: (B, N, L)
            outputs shape:  (B, N*H)
        '''

        x_target = inputs[0]
        x_long_seq = inputs[1:]
        reduced_long_seq_mask = []
        for x in x_long_seq:
            reduced_long_seq_mask.append(tf.cast(tf.ones(tf.shape(x)[:-1]), tf.bool))
        reduced_long_seq_mask = tf.stack(reduced_long_seq_mask, axis=1)
        x_long_seq = tf.stack(x_long_seq, axis=1)

        if len(x_target.shape) == 2:
            x_target = tf.expand_dims(x_target, axis=1)
        if len(x_target.shape) != 3:
            logger.error("target_feat shape must be [batch_size, 1, H]")

        if len(x_long_seq.shape) == 3:
            x_long_seq = tf.expand_dims(x_long_seq, axis=1)
        if len(x_long_seq.shape) != 4:
            logger.error("x_long_seq shape must be [batch_size, N, L, H]")

        din_output_emb = self.get_din_pooling(reduced_long_seq_mask, self.target_features[0], x_long_seq, x_target)

        return din_output_emb
