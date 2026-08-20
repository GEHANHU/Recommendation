import time
from typing import Dict, Iterable
import re
import keras
from tensorflow.keras import layers

from modelflow.common import Config, logger
from modelflow.common.distribute import get_rank_id
from modelflow.data import FeatureManager
from modelflow.integrations.tensorflow.modules.module import Module
from modelflow.integrations.tensorflow.loss.utils import InBatchNegativesSampler, SimilarityLayer
import tensorflow as tf
import numpy as np
import os

try:
    import horovod.tensorflow.keras as hvd
except ImportError:
    logger.warning("now images not found horovod")


def reverse_non_zero_elements(click_behavior_emebedding, mask_seq_valid_index):
    """根据下标变换序列，使得变换前后的序列形如
        1,2,3,4,0,0
        1,2,3,0,0,0
        --->
        0,0,4,3,2,1
        0,0,0,3,2,1
        --->
        4,3,2,1,0,0
        3,2,1,0,0,0

        前后下标形如：
        0 1 2 3 4 5
        0 1 2 3 4 5
        --->
        3 2 1 0 4 5
        2 1 0 3 4 5
    """
    # 动态获取B,N维度
    batch_size = tf.shape(click_behavior_emebedding)[0]
    seq_length = tf.shape(click_behavior_emebedding)[1]

    # 获取有效序列倒排后的索引
    indices = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])
    seq_length_expanded = tf.tile(tf.expand_dims(tf.expand_dims(seq_length - 1, 0), 0), [batch_size, seq_length])
    valid_reverse_indices_offset = seq_length_expanded - \
                                   tf.reshape(tf.expand_dims(mask_seq_valid_index, 0), [batch_size, 1])
    valid_reverse_indices = seq_length_expanded - indices - valid_reverse_indices_offset
    final_indices = tf.where(valid_reverse_indices >= 0, valid_reverse_indices, indices)

    # 使用 tf.gather 对 click_behavior_emebedding 进行重排
    result = tf.gather(click_behavior_emebedding, final_indices, batch_dims=1)

    return result


@Module.register("dien")
class DIEN(layers.Layer):

    def __init__(self, name, dien_specific_params: Config):
        """ DIEN序列模型 返回序列emb与辅助loss
            dien_specific_params:
                click_behavior_list: 记录user_behavior部分所有点击输入特征的训练数据;
                noclick_behavior_list: 记录user_behavior部分所有未点击输入特征的训练数据;
                target_item_list: 记录target_item部分输入特征的训练数据;
                original_feature_list: 如果存在，则使用离线dien 计算相似度用于 D ETA_topk;
                contrastive_aux_loss_flag: dien 的辅助loss，是否使用基于对比学习的next_item_prediction;
                num_negs: 使用基于对比学习的辅助loss时负采样的个数;
              {
                name: 'dien_embedding',
                type: 'dien',
                parameters: {
                  dien_specific_params: {
                    // target序列和点击序列特征数要一致 且保证序列长度一致, 为了使得concnat之后最后一个维度保持一致
                    target_item_list: [
                            "discrete1", "discrete1", "discrete2", "discrete2",
                    ],
                    click_behavior_list: [
                            "seq_discrete1", "seq_discrete2", "seq_discrete3", "seq_discrete4"
                    ],
                    original_feature_list:[
                            "ori_feature1","ori_feature2","ori_feature3","ori_feature4"
                    ]
                    reverse_sequence: true,
                    aux_layer_config: {
                        type: 'dnn',
                        hidden_dims: [128, 64, 2],
                        hidden_activation: ['sigmoid', 'sigmoid']
                    }
                  },
                },
                inputs:{
                   inputs: 'ref::eta.embedding_short'  // input shape is as said in `build`
                },
                outputs: ["embedding", "aux_loss"]
              },
            需进一步配置辅助loss
            extra_losses: [{type:'mean_predict',predict: 'ref::dien_embedding.aux_loss',label:
                    'ref::dien_embedding.aux_loss', weight: 0.5}],
            extra_metrics: [{type:'reduce_mean',predict: 'ref::dien_embedding.aux_loss',label:
                    'ref::dien_embedding.aux_loss'}]
        """
        super().__init__(name=name)

        infos = ['click_behavior_list', 'target_item_list', 'noclick_behavior_list']

        self.feature_dict = {}
        for info in infos:
            self.feature_dict[info] = dien_specific_params.get(info, None)

        self.activation = dien_specific_params.get('activation')
        self.return_loss = dien_specific_params.get('return_loss', True)
        self.reverse_sequence = dien_specific_params.get('reverse_sequence', False)
        self.aux_layer_config = dien_specific_params.get('aux_layer_config')
        self.is_sub_module = dien_specific_params.get('is_sub_module', False)
        self.original_feature_list = dien_specific_params.get('original_feature_list', [])
        self.seq_maxsplit = dien_specific_params.get('seq_maxsplit', 100)
        self.seq_topk = dien_specific_params.get('seq_topk', 3)
        self.hdfs_delimiter = dien_specific_params.get('delimiter', '\u0001')
        self.hdfs_newline = dien_specific_params.get('newline', "\n")
        self.file_dir = dien_specific_params.get('file_dir', None)
        self.dien_file_name = dien_specific_params.get('dien_file_name', "intermediate")
        self.dien_specific_params = dien_specific_params

        self.contrastive_aux_loss_flag = dien_specific_params.get("contrastive_aux_loss_flag", False)
        self.num_negs = dien_specific_params.get("num_negs", None)
        self.similarity_method = dien_specific_params.get("similarity_method", "cosine")
        self.cut_gradient_flag = dien_specific_params.get("cut_gradient_flag", False)
        self.use_batch_normalization_inside_flag = dien_specific_params.get("use_batch_normalization_inside_flag",
                                                                            False)

        self.embedding_dim_dict = {}
        self.user_behavior_gru = None
        self.attention_layer = None
        self.AuxNet = None
        self.user_behavior_augru = None

        self.file_cnt = 0

        if self.file_dir is not None:
            self.dien_file_name = os.path.join(self.file_dir, self.dien_file_name)
            if not os.path.exists(self.file_dir):
                os.makedirs(self.file_dir, mode=0o755, exist_ok=True)

    def build(self, input_shape):
        '''
            输入字典类型的input: {feat: feat_tensor}
        '''
        if len(input_shape) == 2:
            ori_input_shape = input_shape[1]
            input_shape = input_shape[0]
        self.embedding_dim_dict = self._get_emb_size(input_shape)
        unit = self.get_GRU_input_dim(self.embedding_dim_dict,
                                      self.feature_dict.get('click_behavior_list'))

        # Init GRU Layer
        self.user_behavior_gru = layers.GRU(unit, return_sequences=True)
        # Init Attention Layer
        self.attention_layer = layers.Softmax()
        # Init Auxiliary Layer
        self.AuxNet = AuxLayer(self.aux_layer_config)
        # Init similarity layer
        if self.contrastive_aux_loss_flag:
            self.similarity_layer = SimilarityLayer(similarity_method=self.similarity_method)
        # Init AUGRU Layer
        self.user_behavior_augru = AUGRU(unit)
        self.attention_dien = AttentionDien(nums_head=8)

        if self.use_batch_normalization_inside_flag:
            self.batch_normalization_item = layers.BatchNormalization()
            self.batch_normalization_behavior = layers.BatchNormalization()
            self.batch_normalization_gru_hidden = layers.BatchNormalization()
            self.batch_normalization_output = layers.BatchNormalization()

    def _get_emb_size(self, input_shape: Dict[str, tf.TensorShape]):
        embedding_dim_dict = dict()
        for feat_name, feat_emb in input_shape.items():
            embedding_size = int(feat_emb[-1])
            embedding_dim_dict[feat_name] = embedding_size
        return embedding_dim_dict

    def get_GRU_input_dim(self, embedding_dim_dict, user_behavior_features):
        '''
            保证点击序列的GRU隐藏层维度 与target物品训练emb维度一致
        '''
        rst = 0
        for feature in user_behavior_features:
            rst += embedding_dim_dict[feature]
        return rst

    def get_emb(self, inputs, mask, click_behavior_list, target_item_list, noclick_behavior_list):
        '''

        Args:
            inputs: Dict 输入特征embedding字典
            mask: Dict 输入特征的mask
            click_behavior_list: List 用户交互行为特征list
            target_item_list: List 与用户行为做交叉的item特征list

        Return: 拼接后的行为序列、target序列与掩码信息
            mask_seq: tf.bool [B, L]
            mask_seq_valid_length: tf.int32  [B, ]
            target_embedding:  [B, 1, H]
            behavior_emebedding:  [B, L, H]
        '''

        mask_seq = None
        target_item_feature_embedding = []
        for feature in target_item_list:
            target_item_feature_embedding.append(inputs[feature])

        click_behavior_embedding = []
        for feature in click_behavior_list:
            longseq_embedding = inputs[feature]
            click_behavior_embedding.append(longseq_embedding)

            # 有效长度取决于长板  最好是行为序列的长度保持一致
            if mask is None:
                mask_seq = tf.cast(tf.ones(shape=tf.shape(longseq_embedding)[:2]), tf.bool)
            else:
                # 只有所有序列特征都为0的地方,mask_seq对应位置才为false
                mask_seq = tf.math.logical_or(tf.cast(tf.ones_like(mask[feature]), tf.bool)
                                              if mask_seq is None else mask_seq,
                                              tf.cast(mask[feature], tf.bool))

        mask_seq_valid_length = tf.reduce_sum(tf.cast(mask_seq, tf.int32), axis=-1)

        # 仅去获取embedding, 把梯度回传截断
        if self.cut_gradient_flag:
            target_embedding = tf.stop_gradient(tf.concat(target_item_feature_embedding, axis=-1))
            behavior_emebedding = tf.stop_gradient(tf.concat(click_behavior_embedding, axis=-1))
        else:
            target_embedding = tf.concat(target_item_feature_embedding, axis=-1)
            behavior_emebedding = tf.concat(click_behavior_embedding, axis=-1)

        return (mask_seq,
                mask_seq_valid_length,
                target_embedding,
                behavior_emebedding,
                '')

    def split_and_take_first_n(self, s, n, mask=None, sep=','):
        """
        将字符串按分隔符分割，并取前N个元素。

        参数:
            s: 输入的字符串张量，可以是标量或批次。
            n: 要获取的元素数量。
            sep: 分隔符，默认为逗号。

        返回:
            RaggedTensor: 包含前N个分割元素的张量。
        """
        # 分割字符串
        split_tensor = tf.strings.split(s, sep=sep, maxsplit=n)
        split_tensor = tf.squeeze(split_tensor, axis=1)

        # 确保不超过实际分割的元素数量
        actual_length = split_tensor.row_lengths()
        indices = tf.minimum(actual_length, n)
        indices_B = tf.ragged.range(indices)

        # 提取前N个元素
        split_tensor_actual = tf.gather(split_tensor, indices_B, axis=1, batch_dims=1)
        return split_tensor_actual

    def call(self, inputs, mask=None, training=False, **kwargs):

        if len(inputs) == 2:
            ori_inputs = inputs[1]
            inputs = inputs[0]

            if mask is not None and len(mask) > 0:
                mask = mask[0]

        dien_offline = len(inputs) > 1 and len(self.original_feature_list) >= 1
        mask_flag = mask is not None and len(mask) > 0

        mask_seq, mask_seq_valid_length, target_item_embedding, click_behavior_emebedding, \
            noclick_behavior_embedding = self.get_emb(inputs, mask, **self.feature_dict)

        if self.use_batch_normalization_inside_flag:
            target_item_embedding = self.batch_normalization_item(target_item_embedding, training=training)
            click_behavior_emebedding = self.batch_normalization_behavior(click_behavior_emebedding, training=training)

        mask_seq_valid_index = mask_seq_valid_length - 1

        if self.reverse_sequence:
            click_behavior_emebedding = reverse_non_zero_elements(click_behavior_emebedding, mask_seq_valid_index)

        # GRU Layer 基于RNN的兴趣抽取
        # click_gru_emb shape (B,L,H)
        click_gru_emb = self.user_behavior_gru(click_behavior_emebedding, mask=mask_seq)

        if self.use_batch_normalization_inside_flag:
            click_gru_emb = self.batch_normalization_gru_hidden(click_gru_emb, training=training)

        # Auxiliary Loss  一跳监督信号辅助训练, 使得当前token表征的用户兴趣状态向量与下一token相似
        aux_loss = self.auxiliary_loss(click_gru_emb[:, :-1, :], click_behavior_emebedding[:, 1:, :],
                                       mask=mask_seq[:, 1:], num_negs=self.num_negs)

        # Attention Layer  目标物品与序列隐藏层做交叉
        mask_att = tf.expand_dims(mask_seq, 1)
        #  (B, 1, H) (B, L, H) -> (B, 1, L)

        if dien_offline and mask_flag:
            dien_file_name = tf.cast(self.dien_file_name, tf.string)

            # 两部分，1 过滤小于topk的   2.mask*score
            attn_score = self.attention_dien([click_gru_emb, click_gru_emb, click_gru_emb])

            click_emb_mask = mask[self.original_feature_list[-1]]

            indices = tf.range(self.seq_topk)  # 要选择的列索引
            B_mask = tf.gather(click_emb_mask, indices, axis=1)
            B_mask_not = tf.logical_not(B_mask)

            user_id_ori = ori_inputs[self.original_feature_list[0]]
            click_id_ori = ori_inputs[self.original_feature_list[1]]
            pt_d = ori_inputs[self.original_feature_list[2]]
            pt_h = ori_inputs[self.original_feature_list[3]]

            split_tensor = self.split_and_take_first_n(click_id_ori, n=self.seq_maxsplit, mask=B_mask_not)

            negative_infinity = tf.constant(-float('inf'), dtype=tf.float32)
            attn_score = tf.where(click_emb_mask, attn_score, negative_infinity)
            attn_score = tf.expand_dims(attn_score, axis=1)

            topk_score, topk_index = tf.nn.top_k(attn_score, self.seq_topk)
            topk_index = tf.squeeze(topk_index, axis=1)

            B_mask_index = tf.ragged.boolean_mask(
                topk_index,
                B_mask
            )

            click_id_topk = tf.gather(split_tensor, B_mask_index, axis=1, batch_dims=1)

            self.update_data(user_id_ori, click_id_topk, pt_d, pt_h, file_name=dien_file_name)

        # target_item_embedding shape (B,1,H), click_gru_emb shape (B,L,H), hist_attn shape (B,1,L)
        hist_attn = self.attention_layer(tf.matmul(target_item_embedding, click_gru_emb, transpose_b=True),
                                         mask=mask_att)

        # AUGRU Layer 结合交叉注意力的第二层RNN输出
        augru_hidden_state = tf.zeros_like(click_gru_emb[:, 0, :])

        outputs = []
        for i in range(click_gru_emb.shape[1]):
            in_emb, in_att = click_gru_emb[:, i, :], hist_attn[:, :, i]
            # (B, H) (B, 1) (B, 1)-> (B, H)
            augru_hidden_state = self.user_behavior_augru(in_emb, augru_hidden_state, in_att)
            outputs.append(augru_hidden_state)

        batch_indices = tf.range(tf.shape(mask_seq_valid_index)[0])
        # 收集每个序列中最后一个有效位置的输出
        mask_index_pair = tf.stack([batch_indices, mask_seq_valid_index], axis=1)

        outputs = tf.stack(outputs, axis=1)

        dien_output_emb = tf.gather_nd(outputs, mask_index_pair)
        embedding_out = tf.gather_nd(click_behavior_emebedding, mask_index_pair)
        hidden_states = tf.gather_nd(click_gru_emb, mask_index_pair)

        if self.use_batch_normalization_inside_flag:
            dien_output_emb = self.batch_normalization_output(dien_output_emb, training=training)

        return dien_output_emb, aux_loss

    def update_data(self, param_user_id, param_click_id, param_pt_d, param_pt_h, file_name):
        # 保存到文件
        tf.py_function(
            func=self.save_file,
            inp=[param_user_id, param_click_id, param_pt_d, param_pt_h, file_name],
            Tout=[]
        )

    def save_file(self, param_user_id, param_click_id, param_pt_d, param_pt_h, file_name):
        self.file_cnt += 1
        file_save_txt = f"{str(file_name.numpy().decode('utf-8'))}{str(self.file_cnt)}_{get_rank_id()}"
        vec_func = np.vectorize(lambda x: '^'.join(x.astype(str)))

        user_id = param_user_id.numpy()[:, 0]
        click_id = param_click_id.numpy()
        click_id_str = vec_func(click_id)
        pt_d = param_pt_d.numpy()[:, 0]
        pt_h = param_pt_h.numpy()[:, 0]
        file_cnt_list = [self.file_cnt] * len(user_id)

        res = list(zip(user_id, click_id_str, pt_d, pt_h, file_cnt_list))
        np.savetxt(file_save_txt, res, fmt='%s', delimiter=self.hdfs_delimiter, newline=self.hdfs_newline)

    def auxiliary_loss(self, hidden_states, embedding_out, mask=None, num_negs=10, tolerance=1e-8):
        """Auxiliary Loss Function

        contrastive_aux_loss_flag 为 False 时:
            通过hidden state与点击序列concate后进一个全连接神经网络，通过softmax得到最终二分类结果与点击序列和展现序列求解log_loss的到最终aux loss。
        contrastive_aux_loss_flag 为 True 时:
            hidden state 与下一步实际交互的 item embedding(正样本), 采样得到的其他item embedding (负样本) 做对比学习

        Args:
            hidden_states: gru产出的所有hidden state,从h(0)到h(n-1) shape:(B,N-1,D)
            embedding_out: gru输入的embedding特征,从e(1)到e(n)
            mask: 用户交互item 序列的mask
            num_negs: 使用对比学习时负采样的个数
            tolerance: 负样本与正样本对比时的最小容差

        """

        if not self.contrastive_aux_loss_flag:

            click_input_ = tf.concat([hidden_states, embedding_out], -1)
            click_prop_ = self.AuxNet(click_input_)[:, :, 0]  # 默认第一位为label
            click_loss_ = - tf.reshape(tf.math.log(click_prop_), [-1, tf.shape(embedding_out)[1]])
            if mask is not None:
                mask = tf.cast(mask, tf.float32)
                mask_inputs = click_loss_ * mask
                sum_inputs = tf.reduce_sum(mask_inputs, axis=-1)
                sum_masks = tf.reduce_sum(mask, axis=-1) + 1e-9
                click_loss_ = tf.divide(sum_inputs, sum_masks)

            aux_loss = tf.reduce_mean(click_loss_)

        else:

            sampler = InBatchNegativesSampler(
                l2_norm=False,
                l2_norm_eps=1e-6,
                dedup_embeddings=False
            )

            sampler.process_batch(embedding_out, presences=mask)
            # 获取所有 ID 和嵌入
            all_ids, all_embeddings = sampler.get_all_ids_and_embeddings()
            # 采样负样本
            sampled_ids, sampled_negs = sampler(embedding_out, num_to_sample=num_negs)

            positive_expanded = tf.expand_dims(embedding_out, axis=-2)

            vector_diff = tf.abs(positive_expanded - sampled_negs)
            # 使用逻辑或操作，只要有一个维度不满足就不行
            sampled_negatives_valid_mask = tf.reduce_any(vector_diff > tolerance, axis=-1)

            positive_mask = tf.expand_dims(tf.ones(tf.shape(sampled_negatives_valid_mask)[:-1]), -1)
            aux_seq_mask = tf.reshape(
                tf.concat([positive_mask, tf.cast(sampled_negatives_valid_mask, tf.float32)], axis=-1),
                [-1, num_negs + 1])

            ###### similarity layer
            positive_input = embedding_out
            negative_inputs = sampled_negs

            # aux_net_inputs shape (B,N-1,1+num_neg,D)
            aux_net_inputs = tf.concat([tf.expand_dims(positive_input, axis=-2), negative_inputs], axis=-2)

            click_logits = self.similarity_layer(tf.expand_dims(hidden_states, axis=-2), aux_net_inputs)

            click_logits = tf.reshape(click_logits, shape=[-1, tf.shape(click_logits)[-1]])
            indices = tf.zeros([tf.shape(click_logits)[0]], dtype=tf.int32)
            labels = tf.one_hot(indices, depth=tf.shape(click_logits)[-1], dtype=tf.float32)

            neg_inf = tf.constant(-1e9, dtype=click_logits.dtype)

            click_logits = tf.where(
                tf.equal(aux_seq_mask, 1),  # 条件：保留的类别
                click_logits,  # 满足条件时使用原logits
                neg_inf  # 不满足条件时使用负无穷
            )

            # 计算loss
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=click_logits)

            if mask is not None:
                loss = loss * tf.reshape(tf.cast(mask, tf.float32), [-1])
                valid_count = tf.reduce_sum(tf.cast(mask, tf.float32))
                aux_loss = tf.reduce_sum(loss) / (valid_count + 1e-8)

            else:
                aux_loss = tf.reduce_mean(loss)

        return aux_loss


class AuxLayer(layers.Layer):
    def __init__(self, aux_layer_config):
        super().__init__()
        self.aux_layer_config = aux_layer_config

    def build(self, input_shape):
        self.fc = tf.keras.Sequential()
        self.fc.add(layers.BatchNormalization())
        self.fc.add(Module.from_config(self.aux_layer_config, name=None))
        self.softmax = layers.Softmax()

    def call(self, x):
        logit = tf.squeeze(self.fc(x))
        return self.softmax(logit)


class GRU_GATES(tf.keras.layers.Layer):
    def __init__(self, units):
        super(GRU_GATES, self).__init__()
        self.linear_act = layers.Dense(units, activation=None, use_bias=True)
        self.linear_noact = layers.Dense(units, activation=None, use_bias=False)

    def call(self, a, b, gate_b=None):
        if gate_b is None:
            return tf.keras.activations.sigmoid(self.linear_act(a) + self.linear_noact(b))
        else:
            return tf.keras.activations.tanh(self.linear_act(a) + tf.math.multiply(gate_b, self.linear_noact(b)))


class AUGRU(layers.Layer):
    '''
        GRU的更新门中加入注意力得分
    '''

    def __init__(self, units):
        super(AUGRU, self).__init__()
        self.u_gate = GRU_GATES(units)
        self.r_gate = GRU_GATES(units)
        self.c_memo = GRU_GATES(units)

    def call(self, inputs, state, att_score):
        u = self.u_gate(inputs, state)
        r = self.r_gate(inputs, state)
        c = self.c_memo(inputs, state, r)
        u_ = att_score * u
        state_next = (1 - u_) * state + u_ * c
        return state_next


class AttentionDien(layers.Layer):
    def __init__(self, nums_head, **kwargs):
        self.nums_head = nums_head
        super(AttentionDien, self).__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        self.batch_size = input_shape[-1][0]
        self.seq_len = input_shape[-1][1]
        self.d_model = input_shape[-1][-1]

        self.depth = self.d_model // self.nums_head
        self.wq = layers.Dense(self.d_model)
        self.wk = layers.Dense(self.d_model)
        self.wv = layers.Dense(self.d_model)
        self.dense = layers.Dense(self.d_model)
        super().build(input_shape, **kwargs)

    def call(self, inputs, mask=None, **kwargs):
        q = inputs[0]
        k = inputs[1]
        v = inputs[2]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = tf.reshape(q, [-1, self.seq_len, self.nums_head, self.depth])
        k = tf.reshape(k, [-1, self.seq_len, self.nums_head, self.depth])
        v = tf.reshape(v, [-1, self.seq_len, self.nums_head, self.depth])

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # [b, num_head, seq_len, self.depth] -> [b, num_head, seq_len, seq_len]
        qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        qk = tf.divide(qk, tf.math.sqrt(dk))

        # [b  seq_len, seq_len]
        qk_sum = tf.reduce_sum(qk, axis=[1, 3])
        attention_weight = tf.nn.softmax(qk_sum, axis=-1)

        return attention_weight


@Module.register("dien_aux_loss")
class DienAuxLoss(layers.Layer):
    """
        即插即用 构建序列建模中前后emb的相似度损失
        :register name: dien_aux_loss
        :config example:
        .. code-block:: jsonnet

          using dnn model
          {
            name: 'aux_loss',
            type: 'dien_aux_loss',
            parameters: {
              loss_specific_params: {
                click_behavior: "seq_discrete3",
                non_click_behavior: "seq_discrete4",
                reverse_sequence: true,
                aux_layer_config: {
                    type: 'dnn',
                    hidden_dims: [128, 64, 2],
                    hidden_activation: ['sigmoid', 'sigmoid'],
                    output_activation: {"type": "sigmoid", "alpha": 0.1}
                }
              },
            },
            inputs:{
               inputs: 'ref::eta.embedding_short'
            },
            outputs: "aux_loss"
          }

          using mathematical model
          {
            name: 'aux_loss_2',
            type: 'dien_aux_loss',
            parameters: {
              loss_specific_params: {
                aux_layer_config: {
                    type: 'string_expression',
                    expression: "reduce_sum(x * y, axis=-1)"
                }
              },
            },
            inputs:{
               inputs: ['ref::eta.embedding_short.seq_discrete3[:,:-1,:]',
                        'ref::eta.embedding_short.seq_discrete3[:,1:,:]']
            },
            outputs: "aux_loss"
          }
    """

    def __init__(self, loss_specific_params, name="proj_dist"):
        super().__init__(name=name)

        aux_layer_config = loss_specific_params.get('aux_layer_config')
        self.click_behavior = loss_specific_params.get('click_behavior')
        self.non_click_behavior = loss_specific_params.get('non_click_behavior')
        self.reverse_sequence = loss_specific_params.get('reverse_sequence', False)
        self.pos_label = loss_specific_params.get('pos_label', 0)
        self.neg_label = loss_specific_params.get('neg_label', 1)
        self.use_gru = loss_specific_params.get('use_gru', False)
        unit = loss_specific_params.get('unit', 64)

        self.dnn_model = aux_layer_config.get('type') == 'dnn'
        if self.dnn_model:
            self.model = AuxLayer(aux_layer_config)
        else:
            self.model = Module.from_config(aux_layer_config)

        if self.use_gru:
            self.user_behavior_gru = layers.GRU(unit, return_sequences=True)

    def call(self, inputs, mask=None, **kwargs):
        """
            法1：DNN学习序列元素之间的转移概率 计算二分类loss  input: dict
            法2：传入算数表达式计算loss  input: [x,y]
        """
        if self.dnn_model:
            click_loss_ = self.get_loss_by_dnn(inputs, mask, self.click_behavior, label=self.pos_label)
            click_loss_ += self.get_loss_by_dnn(inputs, mask, self.non_click_behavior, label=self.neg_label)
        else:
            click_loss_ = self.get_loss_by_regular(inputs)
            click_loss_ += self.get_loss_by_regular(inputs)

        return tf.reduce_mean(click_loss_)

    def get_loss_by_regular(self, inputs):
        x, y = inputs
        model_input = {
            "x": x,
            "y": y,
        }
        click_loss_ = self.model(model_input)

        return click_loss_

    def get_loss_by_dnn(self, inputs, mask, behavior_feature, label):
        if behavior_feature is None or behavior_feature not in inputs:
            return tf.constant(0, dtype=tf.float32)

        click_behavior_emebedding = inputs[behavior_feature]
        mask_flag = mask is not None and len(mask) > 0
        if mask_flag:
            mask_seq = mask[self.click_behavior]
        else:
            mask_seq = tf.ones(shape=tf.shape(click_behavior_emebedding)[:2])
        mask_seq_valid_index = tf.reduce_sum(tf.cast(mask_seq, tf.int32), axis=-1) - 1
        if self.reverse_sequence:
            click_behavior_emebedding = reverse_non_zero_elements(click_behavior_emebedding, mask_seq_valid_index)

        if self.use_gru:
            click_gru_emb = self.user_behavior_gru(click_behavior_emebedding, mask=mask_seq)
            click_input_ = tf.concat([click_gru_emb[:, :-1, :], click_behavior_emebedding[:, 1:, :]], -1)
        else:
            click_input_ = tf.concat([click_behavior_emebedding[:, :-1, :], click_behavior_emebedding[:, 1:, :]], -1)

        click_prop_ = self.model(click_input_)[:, :, label]  # 默认第一位为label
        click_loss_ = - tf.reshape(tf.math.log(click_prop_), [-1, tf.shape(click_behavior_emebedding)[1] - 1])

        if mask_flag:
            click_loss_ = self.get_valid_loss(click_loss_, mask_seq)

        return click_loss_

    def get_valid_loss(self, click_loss_, mask_seq):
        mask = tf.cast(mask_seq[:, :-1], tf.float32)
        mask_inputs = click_loss_ * mask
        sum_inputs = tf.reduce_sum(mask_inputs, axis=-1)
        sum_masks = tf.reduce_sum(mask, axis=-1) + 1e-9
        click_loss_ = tf.divide(sum_inputs, sum_masks)
        return click_loss_
