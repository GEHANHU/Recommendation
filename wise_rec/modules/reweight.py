#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from typing import List, Dict
from typing import Union
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers

from modelflow.common.config import convert_config_to_dict
from modelflow.integrations.tensorflow.modules.module import Module
from modelflow.common.log import logger
from modelflow.data import FeatureManager


@Module.register("explicit_weights")
class ExplicitWeightsLayer(layers.Layer):
    """ 借鉴lhuc模块，显式分场景

    :register name: explicit_weights

    :config example:
    .. code-block:: jsonnet
      {
        name: 'domain_feature_process',
        type: 'explicit_weights',
        parameters: {
          lhuc_num: 4,
          normal_feature: 'region!="domain"',
          explicit_dim: 16,
          non_domain_hidden_units: 64,
          use_bias_in_explicit_weight: true
        },
        inputs: {
          inputs: 'ref::domain_feature_concat.output',
        },
        outputs: ['explicit_weight', 'explicit_weight_bias', 'lhuc_weight'],
      }

    :param lhuc_num：lhuc数量
    :param normal_feature：非领域特征，用来计算非领域特征数量
    :param explicit_dim：显式分场景尺寸
    :param non_domain_hidden_units：非领域特征的输出维度
    :param lhuc_activation_fn：lhuc计算式使用的激活函数
    :param use_bias_in_explicit_weight：显式层是否加入bias
    :param name: 层名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 lhuc_num: int,
                 normal_feature: str,
                 explicit_dim: int,
                 non_domain_hidden_units: int,
                 feature_manager: FeatureManager,
                 lhuc_activation_fn: Optional[str] = None,
                 use_bias_in_explicit_weight: bool = True,
                 name="explicit_weights",
                 **kwargs):
        self.lhuc_num = lhuc_num
        self.explicit_dim = explicit_dim
        self.feature_manager = feature_manager
        self.normal_feature_num = len(self.feature_manager.query_feature(normal_feature))
        self.non_domain_hidden_units = non_domain_hidden_units
        self.lhuc_activation_fn = lhuc_activation_fn
        self.use_bias_in_explicit_weight = use_bias_in_explicit_weight
        self.dense1, self.dense2, self.dense3 = None, None, None
        super().__init__(name=name, **kwargs)

    def build(self, input_shape: tf.TensorShape):
        self.dense1 = layers.Dense(self.non_domain_hidden_units * self.explicit_dim, activation=None)
        self.dense2 = layers.Dense(self.normal_feature_num * self.lhuc_num, activation=self.lhuc_activation_fn)
        if self.use_bias_in_explicit_weight:
            self.dense3 = layers.Dense(self.explicit_dim, activation=None)

    def call(self, inputs: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        # inputs shape: B * F * D
        explicit_weight = self.dense1(inputs)
        explicit_weight = tf.reshape(explicit_weight, [-1, self.non_domain_hidden_units, self.explicit_dim])

        lhuc_weight = self.dense2(inputs)
        lhuc_weight = tf.reshape(lhuc_weight, [-1, self.lhuc_num, self.normal_feature_num])
        lhuc_weight = tf.nn.softmax(lhuc_weight, axis=-1)  # batch_size, lhuc_num, normal_feature_num
        # batch*lhuc_num*field_nums
        lhuc_weight = tf.expand_dims(lhuc_weight, axis=-1)

        if self.use_bias_in_explicit_weight:
            explicit_weight_bias = self.dense3(inputs)
            explicit_weight_bias = tf.reshape(explicit_weight_bias, [-1, 1, self.explicit_dim])
            # explicit_weight  batch_size, non_domain_hidden_units, explicit_dim
            # explicit_weight_bias  batch_size, 1, explicit_dim
            # lhuc_weight  batch_size, lhuc_num, normal_feature_num, 1
            return [explicit_weight, explicit_weight_bias, lhuc_weight]
        else:
            # explicit_weight  batch_size, non_domain_hidden_units, explicit_dim
            # lhuc_weight  batch_size, lhuc_num, normal_feature_num, 1
            return [explicit_weight, lhuc_weight]


@Module.register("implicit_weights")
class ImplicitWeightsLayer(layers.Layer):
    """ 隐式分场景

    :register name: explicit_weights

    :config example:
    .. code-block:: jsonnet
      {
        name: 'normal_feature_process',
        type: 'implicit_weights',
        parameters: {
          lhuc_num: 4,
          explicit_dim: 16,
          implicit_dim: 16,
          use_bias_in_implicit_weight: true,
          normal_feature: 'region!="domain"',
          embedding_size: 16,
          dnn_config: {
            type: 'dnn',
            hidden_dims: [1024, 256, 128, 64],
            hidden_activation: 'relu',
          },
          //attention_config: {}
        },
        inputs: {
          inputs: ['ref::normal_feature_concat.output', 'ref::domain_feature_process.lhuc_weight']
        },
        outputs: ['implicit_weight', 'implicit_weight_bias'],
      }

    :param lhuc_num：lhuc数量
    :param implicit_dim：隐式分场景尺寸
    :param explicit_dim：显式分场景尺寸
    :param normal_feature：非领域特征，用来计算非领域特征数量
    :param embedding_size：特征embedding_dim
    :param feature_manager：特征管理器
    :param use_bias_in_implicit_weight：隐式层是否加入bias
    :param dnn_config：隐藏层MLP配置
    :param attention_config：注意力层配置
    :param name: 层名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 lhuc_num,
                 implicit_dim,
                 explicit_dim,
                 normal_feature: str,
                 embedding_size,
                 feature_manager: FeatureManager,
                 use_bias_in_implicit_weight=True,
                 dnn_config=None,
                 attention_config=None,
                 name="implicit_weights",
                 **kwargs):
        self.lhuc_num = lhuc_num
        self.implicit_dim = implicit_dim
        self.explicit_dim = explicit_dim

        self.use_bias_in_implicit_weight = use_bias_in_implicit_weight
        self.attention_layer = None
        self.feature_manager = feature_manager
        self.feature_nums = len(self.feature_manager.query_feature(normal_feature))
        self.embedding_size = embedding_size

        self.dnn = None
        self.dnn_config = dnn_config
        self.attention_config = attention_config
        self.dense1, self.dense2 = None, None
        super().__init__(name=name, **kwargs)

    def build(self, input_shape: List[tf.TensorShape]):

        if self.attention_config:
            self.attention_layer = Module.from_config(self.attention_config)
            self.dnn_config["hidden_dims"] = [self.feature_nums * self.embedding_size,
                                              self.feature_nums * self.embedding_size]
            logger.info(f"now param use_self_attention_in_lhuc set True,"
                        f"hidden units will customed to {self.dnn_config['hidden_dims']}")
            self.dnn = Module.from_config(self.dnn_config)
        else:
            self.dnn = Module.from_config(self.dnn_config)

        self.dense1 = layers.Dense(self.explicit_dim * self.implicit_dim, activation=None)
        self.dense2 = layers.Dense(self.implicit_dim, activation=None)

    def call(self, inputs: List[tf.Tensor], **kwargs):
        # feature_tensor  batch_size, normal_feature_num*embedding_dim
        feature_tensor, lhuc_weight = inputs

        reshaped_embedding_flat_input = tf.reshape(feature_tensor, [-1, self.feature_nums, self.embedding_size])
        reshaped_embedding_flat_input = tf.expand_dims(reshaped_embedding_flat_input, axis=1)
        reshaped_embedding_flat_input = tf.tile(reshaped_embedding_flat_input, [1, self.lhuc_num, 1, 1])
        # reshaped_embedding_flat_input  batch_size, lhuc_num, normal_feature_num, embedding_size
        # lhuc_weight                    batch_size, lhuc_num, normal_feature_num, 1
        lhuc_hidden = lhuc_weight * reshaped_embedding_flat_input
        if self.attention_config:
            # batch*field*embedding
            lhuc_hidden = tf.reshape(lhuc_hidden, [-1, self.feature_nums, self.embedding_size])
            lhuc_hidden = self.attention_layer([lhuc_hidden, lhuc_hidden])
            # batch*lhuc_num*field_nums*embedding
            lhuc_hidden = tf.reshape(lhuc_hidden,
                                     [-1, self.lhuc_num, self.feature_nums, self.embedding_size])
            lhuc_hidden = tf.reshape(lhuc_hidden,
                                     [-1, self.lhuc_num, self.feature_nums * self.embedding_size])
            lhuc_hidden_dense = self.dnn(lhuc_hidden)
            lhuc_hidden = lhuc_hidden + lhuc_hidden_dense
        else:
            lhuc_hidden = tf.reshape(lhuc_hidden,
                                     [-1, self.lhuc_num, self.feature_nums * self.embedding_size])
            lhuc_hidden = self.dnn(lhuc_hidden)
        # lhuc_hidden batch_size, lhuc_num, dnn.shape[-1]
        implicit_weight = self.dense1(lhuc_hidden)
        implicit_weight = tf.reshape(implicit_weight, [-1, self.lhuc_num, self.explicit_dim, self.implicit_dim])

        if self.use_bias_in_implicit_weight:
            # implicit_weight_bias batch_size, lhuc_num, implicit_dim
            implicit_weight_bias = self.dense2(lhuc_hidden)
            # implicit_weight_bias batch_size, lhuc_num, 1, implicit_dim
            implicit_weight_bias = tf.reshape(implicit_weight_bias, [-1, self.lhuc_num, 1, self.implicit_dim])
            return [implicit_weight, implicit_weight_bias]
        else:
            # implicit_weight batch_size, lhuc_num, explicit_dim, implicit_dim
            return [implicit_weight]


@Module.register("domain_feature_residual")
class DomainFeatureResidual(layers.Layer):
    """ 领域特征残差连接层

    :register name: domain_feature_residual

    :config example:
    .. code-block:: jsonnet
      {
        name: 'domain_feature_residual',
        type: 'domain_feature_residual',
        parameters: {
          attention_config: {
            type: "dot_product_attention",
            num_heads: 1,
            dropout_rate: 0.5,
          },
          domain_feature: 'region=="domain"',
          embedding_size: 16,
          use_bias_in_weight: true,
          domain_feature_residual_type: 'dense'
        },
        inputs: {
          inputs: [
            'ref::mlp1.output',
            'ref::domain_feature_process.explicit_weight',
            'ref::normal_feature_process.implicit_weight',
            'ref::domain_feature_process.explicit_weight_bias',
            'ref::normal_feature_process.implicit_weight_bias',
            //'ref::domain_feature_concat.output'
          ]
        },
        outputs: 'output',
      }

    :param domain_feature：领域特征，用来计算领域特征数量
    :param feature_manager：特征管理器
    :param embedding_size：特征embedding_dim
    :param attention_config：注意力层配置
    :param use_bias_in_weight：显隐式分场景是否加入bias
    :param domain_feature_residual_type：残差类型，当前支持dense和attention两种，当类型是attention时，有6个输入
    :param name: 层名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 domain_feature,
                 feature_manager: FeatureManager,
                 embedding_size,
                 attention_config,
                 use_bias_in_weight=True,
                 domain_feature_residual_type="dense",
                 name='domain_feature_residual',
                 **kwargs):
        self.domain_feature_residual_type = domain_feature_residual_type
        self.feature_nums = len(feature_manager.query_feature(domain_feature))
        self.embedding_size = embedding_size
        self.use_bias_in_weight = use_bias_in_weight
        self.attention_config = attention_config

        self.attention_layer = None
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.attention_layer = Module.from_config(self.attention_config)

    def call(self, inputs: List[tf.Tensor], **kwargs):
        non_domain_hidden, explicit_weight, implicit_weight = inputs[0], inputs[1], inputs[2]
        # non_domain_hidden  batch_size, non_domain_hidden_units
        # explicit_weight    batch_size, non_domain_hidden_units, explicit_dim
        # implicit_weight    batch_size, lhuc_num, explicit_dim, implicit_dim
        # explicit_weight_bias  batch_size, 1, explicit_dim
        # implicit_weight_bias  batch_size, lhuc_num, 1, implicit_dim

        non_domain_hidden = tf.expand_dims(non_domain_hidden, axis=1)
        hidden = tf.matmul(non_domain_hidden, explicit_weight)  # hidden  batch_size, 1, explicit_dim

        if self.use_bias_in_weight:
            explicit_weight_bias = inputs[3]
            hidden = hidden + explicit_weight_bias  # hidden  batch_size, 1, explicit_dim
        # batch*1*1*explcit_dim
        hidden = tf.expand_dims(hidden, axis=1)  # hidden  batch*1*1*explcit_dim
        # batch*lhuc_num*1*implicit_num
        hidden = tf.matmul(hidden, implicit_weight)

        if self.use_bias_in_weight:
            implicit_weight_bias = inputs[4]
            hidden = hidden + implicit_weight_bias
        # batch*lhuc_num*implicit_num
        hidden = tf.squeeze(hidden, axis=2)
        if self.domain_feature_residual_type == "attention":
            domain_feature_embedding = tf.reshape(inputs[5], [-1, self.feature_nums, self.embedding_size])
            hidden = self.attention_layer([domain_feature_embedding, hidden, hidden])
        elif self.domain_feature_residual_type == "dense":
            hidden = self.attention_layer([hidden, hidden])
        hidden = tf.reduce_mean(hidden, axis=1)
        return hidden


@Module.register("senet")
class SENet(layers.Layer):
    """SENet是FiBiNET的子结构，用于动态学习特征的重要性

    * FiBiNET论文：`FiBiNET: Combining Feature Importance and Bilinear feature Interaction for
                   Click-Through Rate Prediction <https://arxiv.org/pdf/1905.09433.pdf>`_

    :register name: senet

    :param reduction_ratio: 缩放比例
    :param kernel_initializer: 权重初始化类型
    :param first_activation: 第一个全连接层的激活函数
    :param last_activation: 第二个全连接层的激活函数
    :param need_gate_output: 是否需要权重打分
    :param name: 层名称
    :param kwargs: class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 reduction_ratio: int = 3,
                 senet_type: Union[str, None] = "fields",
                 kernel_initializer: Union[str, None] = 'glorot_uniform',
                 first_activation: str = 'relu',
                 last_activation: str = 'sigmoid',
                 need_gate_output: bool = False,
                 name: str = "senet",
                 **kwargs):
        self.reduction_ratio = reduction_ratio
        self.senet_type = senet_type
        self.first_activation = first_activation
        self.last_activation = last_activation
        self.kernel_initializer = convert_config_to_dict(kernel_initializer)
        self.feature_size, self.dense1, self.dense2 = None, None, None
        self.need_gate_output = need_gate_output
        self.senet_gate_output = None

        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        # 支持多种格式以及shape输入
        flatten_shape = []
        for x in input_shape:
            if isinstance(x, List):
                flatten_shape.append(len(x))
            elif isinstance(x, Dict):
                tf.print(f"senet dict input keys:{x.keys()}")
                flatten_shape.append(len(x))
            elif len(x) == 2:
                flatten_shape.append(1)
            # 输入序列特征，则将序列特征切分开
            elif len(x) == 3:
                if self.split_feature_flag:
                    flatten_shape.append(x[1])
                else:
                    flatten_shape.append(1)
            else:
                raise NotImplementedError("input_shape is {x}, length is len(x), which is not Implemented ")
        self.feature_size = sum(flatten_shape)

        reduction_size = max(1, self.feature_size // self.reduction_ratio)

        self.dense1 = layers.Dense(reduction_size, self.first_activation, False, self.kernel_initializer)
        self.dense2 = layers.Dense(self.feature_size, self.last_activation, False, self.kernel_initializer)

    def get_gate_output(self, gate_output):
        self.senet_gate_output = gate_output
        return 0

    def call(self, inputs: List[Union[tf.Tensor, List[tf.Tensor]]], **kwargs) -> List[tf.Tensor]:
        """ 计算 senet 层

        :param inputs: 必须为列表形式，输入列表内容为Tensor或者嵌套列表List[Tensor]，Tensor的维度必须大等于2，
               不同Tensor最后一维可以不等长，但是要求列表以及嵌套列表中Tensor的维度相同
        :param kwargs: 额外参数
        :return: senet层计算结果，返回是list，会将嵌套列表extend到原列表的结果中进行field-wise加权，如果原列表中有嵌套列表，
                 返回列表的长度会发生变化，输出list中的Tensor维度和输入一致
        """

        flatten_inputs = []
        for x in inputs:
            if isinstance(x, List):
                for item in x:
                    if len(item.shape) == 2:
                        flatten_inputs.append(item)
                    elif len(item.shape) == 3:
                        flatten_inputs.append(tf.reshape(item, shape=[-1, tf.shape(item)[-1] * tf.shape(item)[-2]]))
                    else:
                        raise NotImplementedError(f"len(x.shape) is {len(item.shape)}, which is not supported in senet")
            elif isinstance(x, Dict):
                for _, item in x.items():
                    if len(item.shape) == 2:
                        flatten_inputs.append(item)
                    elif len(item.shape) == 3:
                        flatten_inputs.append(tf.reshape(item, shape=[-1, tf.shape(item)[-1] * tf.shape(item)[-2]]))
                    else:
                        raise NotImplementedError(f"len(x.shape) is {len(item.shape)}, which is not supported in senet")
            else:
                if len(x.shape) == 2:
                    flatten_inputs.append(x)
                elif len(x.shape) == 3:

                    if self.split_feature_flag:
                        # 如果输入的是一个序列特征，则将序列特征split成一个2维特征list
                        x_split = tf.split(x, x.shape[1], axis=1)
                        flatten_inputs.extend([tf.squeeze(x, axis=1) for x in x_split])

                    else:
                        # 如果输入的是一个序列特征，则将序列特征concat成一个2维特征
                        flatten_inputs.append(tf.reshape(x, shape=[-1, tf.shape(x)[-1] * tf.shape(x)[-2]]))

        if self.senet_type == "fields":
            inputs_mean = tf.concat([tf.reduce_mean(x, axis=-1, keepdims=True) for x in flatten_inputs], axis=-1)
        elif self.senet_type == "elements":
            inputs_mean = tf.concat(flatten_inputs, axis=-1)
        else:
            raise ValueError('parameter senet_type must be fields or elements')
        output = self.dense2(self.dense1(inputs_mean))

        if self.need_gate_output:  # make true the first-dim is batchsize
            tf.py_function(self.get_gate_output, [output], Tout=[tf.int64])

        output = tf.split(output, self.feature_size, axis=-1)
        output = [x[0] * x[1] for x in zip(flatten_inputs, output)]
        return output
