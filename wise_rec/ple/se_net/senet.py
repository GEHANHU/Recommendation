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
