import tensorflow as tf
from tensorflow.keras import layers
from modelflow.integrations.tensorflow.modules.normalization import RMSNorm
from modelflow.integrations.tensorflow.modules import Module
from modelflow.integrations.tensorflow.modules.parallel.linear import ColumnParallelLinear, Linear, RowParallelLinear


@Module.register("MLA")
class MLA(layers.Layer):
    """
    多头潜在注意力
    ::param dim: embedding维度
    ::param qk_head_dim: key、query的维度
    ::param n_heads: 注意力头个数
    ::param q_lora_rank: query低秩分解维度
    ::param kv_lora_rank: key、value低秩分解维度
    ::param v_head_dim: value的维度
    ::group_size: 默认为-1。若有4张卡，
      当group_size=-1时，4张卡上的专家参数都相互独立；
      当group_size=1时，关闭模型并行；
      当group_size=2时，0卡和2卡的专家参数相同，1卡和3卡的专家参数相同。
    ::param use_residual: 是否使用残差输出
    """
    def __init__(self, dim, qk_head_dim, n_heads, q_lora_rank, kv_lora_rank, v_head_dim, group_size=-1,
                 use_residual=True, multi_head_return=False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_local_heads = n_heads  # 需要根据实际分布式设置调整
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.use_residual = use_residual
        self.multi_head_return = multi_head_return
        # 初始化投影层
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * qk_head_dim, group_size=group_size)  # [90, 4*64]
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * qk_head_dim, group_size=group_size)

        # KV 投影层
        self.wkv_a = Linear(self.dim, self.kv_lora_rank)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (qk_head_dim + v_head_dim),
                                          group_size=group_size)
        # 输出投影
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim, group_size=group_size)

        self.softmax_scale = self.qk_head_dim ** -0.5
        self.norm = RMSNorm(self.dim)

    def call(self, inputs, start_pos=0, freqs_cis=None, mask=None):
        # Query 投影
        if self.q_lora_rank == 0:
            q = self.wq(inputs)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(inputs)))

        seq_len = inputs.shape[1]

        q = tf.reshape(q, [-1, seq_len, self.n_local_heads, self.qk_head_dim])

        kv = self.wkv_a(inputs)
        wkv_b = self.wkv_b.weight
        wkv_b = tf.reshape(wkv_b, [self.n_local_heads, -1, self.kv_lora_rank])

        q = tf.einsum('bshd,hdc->bshc', q, wkv_b[:, :self.qk_head_dim])  # b:bs s:seqlen h:head_num d:dim c:lora_rank

        kv = self.kv_norm(kv)

        scores = tf.einsum('bshc,btc->bsht', q, kv) * self.softmax_scale
        scores = tf.nn.softmax(scores, axis=-1)
        x = tf.einsum('bsht,btc->bshc', scores, kv)
        x = tf.einsum('bshc,hdc->bshd', x, wkv_b[:, -self.v_head_dim:])
        if self.multi_head_return:
            return tf.reshape(x, [-1, seq_len, self.n_heads*self.v_head_dim])
        x = self.wo(tf.reshape(x, [-1, seq_len, self.n_heads * self.v_head_dim]))
        x = x + inputs if self.use_residual else x
        return self.norm(x)
