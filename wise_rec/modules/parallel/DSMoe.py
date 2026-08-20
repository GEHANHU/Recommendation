import tensorflow as tf
from typing import Tuple, List, Dict, Union
from modelflow.integrations.tensorflow.modules import Module
from modelflow.integrations.tensorflow.sfpswrapper import sfps
from modelflow.integrations.tensorflow.modules.parallel.linear import ColumnParallelLinear, RowParallelLinear, Linear
from modelflow.integrations.tensorflow.modules.normalization import RMSNorm
from modelflow.common.distribute import get_rank_size, get_rank_id
import tensorflow.keras.activations as F
from .moe import MoE, Gate
import logging
try:
    import horovod.tensorflow as hvd
except ModuleNotFoundError:
    logging.warning("now images not found horovod")


@Module.register("DSMoE", tags=["parallel"])
class DSMoE(MoE):
    """
    稠密-稀疏混合专家模块
    通过稠密训练、稀疏推理的方法达到负载均衡的效果
    ::param dim: embedding维度
    ::param n_routed_experts: 路由专家总数
    ::param n_activated_experts: 激活路由专家个数
    ::param moe_inter_dim: 专家网络中间层节点数
    ::n_shared_experts: 共享专家个数
    ::group_size: 默认为-1。若有4张卡，
      当group_size=-1时，4张卡上的专家参数都相互独立；
      当group_size=1时，关闭模型并行；
      当group_size=2时，0卡和2卡的专家参数相同，1卡和3卡的专家参数相同。
    ::use_residual: 是否使用残差输出
    ::is_flatten: 若为True, 输出维度为[batch_size, seq_len*emb_dim], 若为False,输出维度为[batch_size, seq_len, emb_dim]
    """

    def __init__(self, dim, n_routed_experts, moe_inter_dim, n_shared_experts=1, group_size=1, is_flatten=True,
                 name="ds_moe", **kwargs):

        super().__init__(dim, n_routed_experts, moe_inter_dim, n_shared_experts=n_shared_experts, group_size=group_size,
                         is_flatten=is_flatten, name=name, **kwargs)

        self.shared_experts = FFN(dim, n_shared_experts * moe_inter_dim)
        self.experts = [
            FFN(dim, moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx
            else None for i in range(self.n_routed_experts)
        ]
        self.gate = DenseGate(dim, n_routed_experts)

        self.sparse_gate = SparseGate(dim, n_routed_experts)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> Dict[str, tf.Tensor]:
        """输出：【稠密门控路由计算的embedding， 稀疏门控路由计算的embedding，稀疏门控正则化项】"""
        if isinstance(inputs, List):
            inputs_dense, inputs_sparse = inputs
        else:
            inputs_dense = inputs_sparse = inputs
        y_dense = super().call(inputs_dense)

        seq_len = inputs_sparse.shape[1]
        x = tf.reshape(inputs_sparse, (-1, self.dim))

        # 获取路由权重和索引
        weights, indices, sparse_reg_loss = self.sparse_gate(x)

        y = tf.zeros_like(x)

        # 遍历当前分片负责的所有专家
        for i in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[i]
            # 生成当前专家的mask
            mask = tf.equal(i, indices)
            idx = tf.where(mask)
            selected_x = tf.gather(x, idx[:, 0])
            weighted_output = tf.stop_gradient(expert(selected_x)) * tf.gather_nd(weights, idx)[..., tf.newaxis]
            y = tf.tensor_scatter_nd_add(y, idx[:, 0][:, tf.newaxis], weighted_output)

        # 共享专家计算
        z = self.shared_experts(x)

        y = tf.reshape(y + z, [-1, seq_len, self.dim])
        y = y + inputs_sparse if self.use_residual else y
        y_sparse = self.normalize(y, seq_len)
        return {"y_dense": y_dense, "y_sparse": y_sparse, "sparse_reg_loss": sparse_reg_loss}

    def normalize(self, y: tf.Tensor, seq_len: int) -> tf.Tensor:
        y = tf.reshape(y, [-1, seq_len * self.dim])
        y = self.norm(y)
        return y if self.is_flatten else tf.reshape(y, [-1, seq_len, self.dim])


class DenseGate(Gate):
    def call(self, x: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        # 计算得分 (batch_size, n_experts)
        scores = tf.matmul(x, self.weight, transpose_b=True)

        # relu激活
        scores = tf.nn.relu(scores)

        # 计算专家索引
        indices = tf.where(scores > 0, tf.range(scores.shape[1], dtype=tf.int32), self.n_routed_experts)

        return scores, indices


class SparseGate(Gate):
    def call(self, x: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # 计算得分 (batch_size, n_experts)
        scores = tf.matmul(tf.stop_gradient(x), self.weight, transpose_b=True)

        # relu激活
        scores = tf.nn.relu(scores)

        # 正则化项 --> 稀疏化推理
        reg_loss = tf.reduce_mean(scores)

        # 计算专家索引
        indices = tf.where(scores > 0, tf.range(scores.shape[1], dtype=tf.int32), self.n_routed_experts)

        return scores, indices, reg_loss


class FFN(tf.keras.layers.Layer):
    def __init__(self, dim: int, inter_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.w1 = Linear(dim, inter_dim, bias=True)
        self.w2 = Linear(inter_dim, dim, bias=True)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        return self.w2(tf.nn.gelu(self.w1(x)))
