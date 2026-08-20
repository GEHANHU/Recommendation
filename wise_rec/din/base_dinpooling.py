#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
import logging
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K, layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.python.keras.initializers.initializers_v2 import Constant
from tensorflow.python.keras.utils import tf_utils

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import Zeros

from modelflow.common import Config
from modelflow.common import logger
from modelflow.common.dynamic import safe_eval
from modelflow.common.exception import ConfigError
from modelflow.common.files import load_json, save_json
from modelflow.common.global_vars import set_global_var
from modelflow.integrations.tensorflow import TF_TENSOR_FUNCTIONS
from modelflow.integrations.tensorflow.modules.module import Module
from modelflow.common.digits import weird_division


@Module.register("ref_call")
def ref_call(ref: Callable, **kwargs) -> Any:
    """ 执行 `ref`， 其余参数作为输入参数

    :register name: ref_call

    :config example:
    .. code-block:: jsonnet

       {
         name: "attention_dense",
         type: "ref_call",
         inputs: {
           // 使用concat_attention层中的projection_layer
           ref: "ref::detail_user_seq_attention.self.loop_layers.projection_layer",
           inputs: "ref::concat_detail_target_cross.output"
         },
         outputs: "score",
       }

    :param ref: 函数或已经实现 `__call__` 方法的类实例，例如： :class:`tf.keras.layers.Layer`
    :param kwargs: `ref` 的输入参数
    :return: 执行 `ref` 的返回结果
    """
    return ref(**kwargs)


@Module.register("sum")
class SumLayer(Layer):
    """ 求和层，用于在指定轴上求和降维（不支持带mask的求和）

    :register name: sum

    >>> initializer = Constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    >>> embedding = tf.keras.layers.Embedding(4, 3, input_length=4, embeddings_initializer=initializer, mask_zero=True)
    >>> embedding_output = embedding(tf.convert_to_tensor([[1, 2, 0, 0], [2, 1, 3, 0], [0, 0, 0, 0]]))
    >>> avg_with_sum = SumLayer(axis=1)(embedding_output)
    >>> avg_with_sum.numpy().tolist()
    [[5.0, 7.0, 9.0], [12.0, 15.0, 18.0], [0.0, 0.0, 0.0]]
    >>> avg_with_sum._keras_mask.numpy().tolist()
    [True, True, False]
    >>> avg_with_sum = SumLayer(axis=1, mask=False)(embedding_output)
    >>> avg_with_sum.numpy().tolist()
    [[7.0, 11.0, 15.0], [13.0, 17.0, 21.0], [4.0, 8.0, 12.0]]

    :param axis: 所要求和的轴
    :param keep_dims: 结果是否和输入保持同一维度
    :param mask: 是否执行mask。如果为 ``True``，那么只会取非mask部分的均值作为计算结果
    :param name: layer名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等

    .. versionchanged:: 23.9.1
       新增 `mask` 参数，支持带mask的求和

    """

    def __init__(self,
                 axis: Optional[Union[int, List[int]]] = None,
                 keep_dims: bool = False,
                 mask: bool = True,
                 name: Optional[str] = None,
                 **kwargs):
        super(SumLayer, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.keep_dims = keep_dims
        self.mask = mask

    def call(self, inputs: tf.Tensor, mask: tf.Tensor = None, **kwargs) -> tf.Tensor:
        reduce_inputs = inputs
        if self.mask and mask is not None:
            expand_mask = tf.cast(mask, K.floatx())
            expand_mask = tf.repeat(tf.expand_dims(expand_mask, -1), inputs.shape[-1], axis=-1)
            reduce_inputs = inputs * expand_mask
        return tf.reduce_sum(reduce_inputs, axis=self.axis, keepdims=self.keep_dims)

    def compute_mask(self, inputs, mask=None):
        if self.mask and mask is not None:
            # 根据已有实现，mask比inputs少最后一维，因此需要扩充最后一维
            expand_mask = tf.repeat(tf.expand_dims(mask, -1), inputs.shape[-1], axis=-1)
            export_mask = K.any(expand_mask, axis=self.axis, keepdims=self.keep_dims)
            return K.any(export_mask, axis=-1)
        return None


@Module.register("average")
class AverageLayer(Layer):
    """ 求平均层，用于在指定轴上求平均值降维（支持带mask的求平均）

    :register name: average

    >>> initializer = Constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    >>> embedding = tf.keras.layers.Embedding(4, 3, input_length=4, embeddings_initializer=initializer, mask_zero=True)
    >>> embedding_output = embedding(tf.convert_to_tensor([[1, 2, 0, 0], [2, 1, 3, 0], [0, 0, 0, 0]]))
    >>> avg_with_mask = AverageLayer(axis=1)(embedding_output)
    >>> avg_with_mask.numpy().tolist()
    [[2.5, 3.5, 4.5], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]
    >>> avg_with_mask._keras_mask.numpy().tolist()
    [True, True, False]
    >>> avg_without_mask = AverageLayer(axis=1, mask=False)(embedding_output)
    >>> avg_without_mask.numpy().tolist()
    [[1.75, 2.75, 3.75], [3.25, 4.25, 5.25], [1.0, 2.0, 3.0]]

    :param axis: 所要求平均的轴
    :param keep_dims: 结果是否和输入保持同一维度
    :param mask: 是否执行mask。如果为 ``True``，那么只会取非mask部分的均值作为计算结果
    :param name: layer名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等

    .. versionchanged:: 1.0.1
       新增 `mask` 参数，支持带mask的求平均

    """

    def __init__(self,
                 axis: Optional[Union[int, List[int]]] = None,
                 keep_dims: bool = False,
                 mask: bool = True,
                 name: Optional[str] = None,
                 **kwargs):
        super(AverageLayer, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.keep_dims = keep_dims
        self.mask = mask

    def call(self, inputs: tf.Tensor, mask: tf.Tensor = None, **kwargs) -> tf.Tensor:
        if self.mask and mask is not None:
            expand_mask = tf.cast(mask, K.floatx())
            expand_mask = tf.repeat(tf.expand_dims(expand_mask, -1), inputs.shape[-1], axis=-1)
            mask_inputs = inputs * expand_mask
            sum_inputs = tf.reduce_sum(mask_inputs, axis=self.axis, keepdims=self.keep_dims)
            sum_masks = tf.reduce_sum(expand_mask, axis=self.axis, keepdims=self.keep_dims) + 1e-9
            return tf.divide(sum_inputs, sum_masks)
        return tf.reduce_mean(inputs, axis=self.axis, keepdims=self.keep_dims)

    def compute_mask(self, inputs, mask=None):
        if self.mask and mask is not None:
            # 根据已有实现，mask比inputs少最后一维，因此需要扩充最后一维
            expand_mask = tf.repeat(tf.expand_dims(mask, -1), inputs.shape[-1], axis=-1)
            export_mask = K.any(expand_mask, axis=self.axis, keepdims=self.keep_dims)
            return K.any(export_mask, axis=-1)
        return None


@Module.register("min")
class MinLayer(Layer):
    """ 求最小值层，用于在指定轴上求最小值降维（不支持带mask的求最小值）

    :register name: min

    :param axis: 所要求最小值的轴
    :param keep_dims: 结果是否和输入保持同一维度
    :param name: layer名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 axis: Optional[Union[int, List[int]]] = None,
                 keep_dims: bool = False,
                 name: Optional[str] = None,
                 **kwargs):
        super(MinLayer, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.keep_dims = keep_dims

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.reduce_min(inputs, axis=self.axis, keepdims=self.keep_dims)


@Module.register("max")
class MaxLayer(Layer):
    """ 求最大值层，用于在指定轴上求最大值降维（不支持带mask的求最大值）

    :register name: max

    :param axis: 所要求最大值的轴
    :param keep_dims: 结果是否和输入保持同一维度
    :param name: layer名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 axis: Optional[Union[int, List[int]]] = None,
                 keep_dims: bool = False,
                 name: Optional[str] = None,
                 **kwargs):
        super(MaxLayer, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.keep_dims = keep_dims

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.reduce_max(inputs, axis=self.axis, keepdims=self.keep_dims)


@Module.register("abs")
class AbsLayer(Layer):
    """ 求绝对值层

    :register name: abs

    """

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.abs(inputs)

    def compute_mask(self, inputs, mask=None):
        return mask


@Module.register("einsum")
class EinSumLayer(Layer):
    """万能矩阵运算-爱因斯坦求和层

    :register name: einsum

    :param equation: 表达式
    :param name: 层名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self, equation: str,
                 name: Optional[str] = None, **kwargs):
        super(EinSumLayer, self).__init__(name=name, **kwargs)
        self.equation = equation

    def call(self, inputs: List[tf.Tensor], **kwargs):
        return tf.einsum(self.equation, *inputs)


@Module.register("norm_dropout")
class NormDropoutLayer(Layer):
    """ 执行激活函数，归一化和dropout的层

    :register name: norm_dropout

    :param norm: 归一化配置，目前支持配置type是batch_norm和layer_norm的归一化
    :param dropout_rate: dropout比率
    :param activation: 激活函数配置，支持的激活函数包含标签的 `activation` 的Module
    :param orders: activation, dropout, norm的执行顺序，分别使用 `a`, `d`, `n` 来表示。
       例如 `nad` 表示执行顺序是norm, activation, dropout
    :param name: layer名
    :param kwargs:  :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等

    .. versionchanged:: 23.9.1
       移除参数 `norm_first`, 新增激活函数配置参数 `activation` 和执行顺序参数 `orders`

    """

    def __init__(self,
                 norm: Optional[Union[str, Config]] = None,
                 dropout_rate: float = 0.0,
                 activation: Union[str, Config, None] = None,
                 orders: str = "adn",
                 name="norm_dropout",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.norm = norm
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.orders = self._validate_order(orders.lower())
        self.dropout_layer = Dropout(rate=self.dropout_rate) if self.dropout_rate != 0 else None
        self.norm_layer = Module.from_config(self.norm, tags=["norm"])
        self.activation_layer = Module.from_config(self.activation, tags=["activation"])

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """ 执行归一化和dropout

        :param inputs: 输入tensor
        :param kwargs: 额外参数
        :return: 经过归一化和dropout的tensor
        """
        return self.norm_dropout(inputs)

    def norm_dropout(self, inputs: tf.Tensor) -> tf.Tensor:
        """ 执行归一化和dropout的方法，方便子类调用

        :param inputs: 输入tensor
        :return: 经过归一化和dropout的tensor
        """
        outputs = inputs
        for order in self.orders:
            if order == "a":
                outputs = self.activate(outputs)
            elif order == "d":
                outputs = self.dropout(outputs)
            else:
                outputs = self.normalize(outputs)
        return outputs

    def dropout(self, inputs: tf.Tensor) -> tf.Tensor:
        """ 执行dropout的方法，方便子类调用

        :param inputs: 输入tensor
        :return: 经过dropout的tensor
        """
        outputs = inputs
        if self.dropout_layer is not None:
            outputs = self.dropout_layer(outputs)
        return outputs

    def normalize(self, inputs: tf.Tensor) -> tf.Tensor:
        """ 执行归一化的方法，方便子类调用

        :param inputs: 输入tensor
        :return: 经过归一化的tensor
        """
        outputs = inputs
        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)
        return outputs

    def activate(self, inputs: tf.Tensor) -> tf.Tensor:
        """ 执行激活方法，方便子类调用

        :param inputs: 输入tensor
        :return: 经过归一化的tensor
        """
        outputs = inputs
        if self.activation_layer is not None:
            outputs = self.activation_layer(outputs)
        return outputs

    def _validate_order(self, orders: str) -> str:
        valid_orders = {"a", "d", "n"}
        filtered_orders = ""
        for order in orders:
            if order in valid_orders:
                filtered_orders += order
            else:
                logger.warning("Failed to parse order `%s` in NormDropoutLayer, valid orders are %s.",
                               order, valid_orders)
        return filtered_orders


@Module.register("l2_normalize")
class L2NormalizeLayer(Layer):
    """ L2归一化

    :register name: l2_normalize

    :param axis: 需要进行L2归一化的轴
    :param name: layer名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 axis: Optional[Union[int, List[int]]] = None,
                 name: Optional[str] = None,
                 **kwargs):
        super(L2NormalizeLayer, self).__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.nn.l2_normalize(inputs, axis=self.axis)


@Module.register("loop")
class LoopLayer(Layer):
    """遍历网络层

    遍历某个列表，对列表中的每个元素初始化相同的Module，并进行调用。

    :register name: loop

    :param loop: Module配置，列表中每个元素都将通过该模块
    :param loop_index: 遍历的列表在call函数的inputs中的索引，如果call函数的inputs是单值，可以不进行配置（如下面第一个样例）
    :param share: Module是否共享
    :param name: 遍历网络层名字
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等

    **样例：**

    >>> module_config = {"type": "dense", "units": 16}
    >>> loop_layer = LoopLayer(loop=Config(module_config), share=False)
    >>> input_tensors = [tf.random.normal((3, 4))] * 3
    >>> outputs = loop_layer(input_tensors)
    >>> tf_utils.get_shapes(outputs)
    [TensorShape([3, 16]), TensorShape([3, 16]), TensorShape([3, 16])]


    >>> module_config = {"type": "concat_attention", "project_query": True}
    >>> loop_layer = LoopLayer(loop=Config(module_config), share=False, loop_index=1)
    >>> input_tensors = [tf.random.normal((3, 4)), [tf.random.normal((3, 3, 5)), tf.random.normal((3, 4, 6))]]
    >>> outputs = loop_layer(input_tensors)
    >>> tf_utils.get_shapes(outputs)
    [TensorShape([3, 5]), TensorShape([3, 6])]

    """

    def __init__(self,
                 loop: Config,
                 loop_index: Optional[int] = None,
                 share: bool = False,
                 name: Optional[str] = None,
                 **kwargs):
        super(LoopLayer, self).__init__(name=name, **kwargs)
        self.loop = loop
        self.loop_index = loop_index
        self.share = share

        self.loop_layers: Optional[Union[Layer, List[Layer]]] = None

    def build(self, input_shape):
        """构建遍历网络层

        :param input_shape: 和 ``call`` 函数中的inputs结构相同，标记 ``call`` 函数中每个tensor的维度
        """
        if not isinstance(input_shape, list):
            raise ConfigError(f"LoopLayer's inputs must be with list/tuple/sequence type.")

        if self.loop_index is not None and not isinstance(input_shape[self.loop_index], list):
            raise ConfigError(f"inputs[{self.loop_index}] in LoopLayer must be with list/tuple/sequence type.")

        loop_length = len(input_shape[self.loop_index]) if self.loop_index is not None else len(input_shape)

        if self.share:
            self.loop_layers = Module.from_config(self.loop, name=None)
        else:
            self.loop_layers = [Module.from_config(self.loop, name=None) for _ in range(loop_length)]

        super(LoopLayer, self).build(input_shape)

    def call(self, inputs: Tuple[List[tf.Tensor], ...], **kwargs) -> List[Any]:
        """执行遍历网络层

        :param inputs: 输入的tensor，必须为列表格式，除了遍历的列表之外，其余参数和配置的网络层中的输入参数相同，位置也保持一致
        :param kwargs: 额外参数
        :return: 遍历的列表中每个元素经过所配置的网络层之后的输出
        """
        output_list: List[Any] = []

        if self.loop_index is None:
            if isinstance(self.loop_layers, list):
                for loop_value, loop_layer in zip(inputs, self.loop_layers):
                    loop_output = loop_layer(loop_value)
                    output_list.append(loop_output)
            else:
                for loop_value in inputs:
                    loop_output = self.loop_layers(loop_value)
                    output_list.append(loop_output)
        else:
            input_list = list(inputs)
            loop_list = input_list[self.loop_index]
            if isinstance(self.loop_layers, list):
                for loop_value, loop_layer in zip(loop_list, self.loop_layers):
                    input_list[self.loop_index] = loop_value
                    loop_output = loop_layer(input_list)
                    output_list.append(loop_output)
            else:
                for loop_value in loop_list:
                    input_list[self.loop_index] = loop_value
                    loop_output = self.loop_layers(input_list)
                    output_list.append(loop_output)

        return output_list


@Module.register("math_expression")
class MathExpressionLayer(Layer):
    """数学表达式计算层，公式为：

    .. math::
       result = weight \\times (\\alpha \\times inputs + \\beta)^{\\gamma}

    :register name: math_expression

    :param weight: 公式中的参数 :math:`weight`
    :param alpha: 公式中的参数 :math:`\\alpha`
    :param beta: 公式中的参数 :math:`\\beta`
    :param gamma: 公式中的参数 :math:`\\gamma`
    :param name: 层的名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 weight: float = 1.0,
                 alpha: float = 1.0,
                 beta: float = 0.0,
                 gamma: float = 1.0,
                 name: str = 'math_expression',
                 **kwargs):
        super(MathExpressionLayer, self).__init__(name=name, **kwargs)
        self.weight = weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def build(self, input_shape):
        if not isinstance(input_shape, tf.TensorShape):
            raise ConfigError(f"Unexpected inputs dimensions {input_shape} in MathExpression layer")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        radix = self.alpha * inputs + self.beta
        expression = self.weight * tf.pow(radix, self.gamma)
        return expression


@Module.register("string_expression")
class StringExpressionLayer(Layer):
    """字符串表达式计算层

    :register name: string_expression

    >>> input_tensor = tf.convert_to_tensor([0.0, 1.0, 3.0, 0.0, 2.0])
    >>> expression = "cast(pay > 0, 'float32') * pay + (1 - cast(pay > 0, 'float32')) * ones_like(pay)"
    >>> sel = StringExpressionLayer(expression=expression)
    >>> sel({"pay": input_tensor}).numpy().tolist()
    [1.0, 1.0, 3.0, 1.0, 2.0]

    :param expression: 表达式，支持 `tf基本函数 <https://www.tensorflow.org/versions/r2.3/api_docs/python/tf>`_
       **在表达式中使用函数时，只需要写函数名即可，不需要写引用名。** 如： `ones_like` 而不是 `tf.ones_like`
       若想使用tf.math.log类似的复杂逐级调用，则需要将math和log写入Attribute的SAFE_TF_FUNCTIONS列表中
       比如 expression:'where(tf.strings.to_number("0.7")>0.8,0.8,where(feature3>0.6,0.6,where(feature3>0.4,0.4,0.2)))'
    :param name: 层的名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 expression: str,
                 name: str = 'string_expression',
                 **kwargs):
        super(StringExpressionLayer, self).__init__(name=name, **kwargs)
        self.expression = expression

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs) -> tf.Tensor:
        """ 使用字符串表达式计算并返回结果

        :param inputs: 输入的变量字典。其中key是变量名，对应于 `expression` 中出现的变量名。
        :param kwargs: 其他参数
        :return: 字符串表达式的计算结果
        """
        valid_value_dict = dict(TF_TENSOR_FUNCTIONS, **inputs)
        result = safe_eval(self.expression, valid_value_dict)
        return result


@Module.register("string_expression_assignment")
class StringExpressionAssignmentLayer(Layer):
    """字符串表达式赋值层

    :register name: string_expression_assignment
    :config example:
        .. code-block:: json
            {
                name: 'f4_add_f2',
                type: 'string_expression_assignment',
                inputs: {
                    inputs: {
                        fc: "ref::inputs.`type == 'continuous'`",
                        f2: "ref::inputs.`type == 'discrete'`.feature2",
                    }
                },
                parameters: {
                    expression: "fc['feature4'] * 2 + cast(f2, 'float32')",
                    assignment_object: "fc.feature3",
                },
                outputs: 'output'
            },
            ...
            {
                ...
                inputs: {
                    inputs: "ref::f4_add_f2.output.fc"
                },
                ...
            }

    :param expression: 表达式，支持 `tf基本函数 <https://www.tensorflow.org/versions/r2.3/api_docs/python/tf>`_
       **在表达式中使用函数时，只需要写函数名即可，不需要写引用名。** 如： `ones_like` 而不是 `tf.ones_like`
    :param assignment_object: 赋值对象，如果输入tensor，直接使用变量名，如果输入dict，使用变量名.key（支持解析多层嵌套key）
    :param name: 层的名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 expression: str,
                 assignment_object: str,
                 name: str = 'string_expression_assignment',
                 **kwargs):
        super(StringExpressionAssignmentLayer, self).__init__(name=name, **kwargs)
        self.string_expression_layer = StringExpressionLayer(expression=expression, name=name)
        self.assignment_object = assignment_object

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs) -> Dict[str, tf.Tensor]:
        """ 使用字符串表达式计算，并将计算结果赋值给指定的输入对象

        :param inputs: 输入的变量字典。其中key是变量名，对应于 `expression` 中出现的变量名。
        :param kwargs: 其他参数
        :return: 赋值后的变量字典，其中key是变量名，在输入给其他layer时需要注意使用类似ref::assign_layer.output.key1的格式
        """
        result = self.string_expression_layer(inputs, **kwargs)
        # 将字符串表达式的计算结果作为inputs的指定key对应的value，循环处理可能存在的key嵌套
        tmp_dict = inputs
        assignment_objects = self.assignment_object.split('.')
        for index, key in enumerate(assignment_objects):
            if key in tmp_dict:
                # 循环结束，将结果赋值给对应的key
                if index == len(assignment_objects) - 1:
                    tmp_dict[key] = result
                # 循环深入dict的下一层
                else:
                    tmp_dict = tmp_dict[key]
            else:
                raise KeyError(f"the following key in assignment_object not exist: {key}")
        return inputs


@Module.register("fifo")
class FIFOLayer(Layer):
    """ 先进先出队列层

    将输入的数据放入队列，并返回 **输入数据放入队列前的结果**。队列的数据不能被训练。

    :register name: fifo

    :config example:
    .. code-block:: jsonnet

       {
         name: "fifo",
         type: "fifo",
         parameters: {
           queue_size: 1024,
           queue_dtype: 'float',
         },
         inputs: {
           inputs: ["ref::item_embedding.output"],
         },
         outputs: "output",
       }

    >>> from tensorflow.python.keras.utils import tf_utils
    >>> fifo = FIFOLayer(10)
    >>> output = fifo(tf.fill((1,5), np.nan))
    >>> tf_utils.get_shapes(output)
    TensorShape([0, 5])
    >>> insert_batch1 = fifo(tf.random.normal((4,5)))
    >>> tf_utils.get_shapes(insert_batch1)
    TensorShape([0, 5])
    >>> insert_batch2 = fifo(tf.random.normal((4,5)))
    >>> tf_utils.get_shapes(insert_batch2)
    TensorShape([4, 5])
    >>> insert_batch3 = fifo(tf.random.normal((4,5)))
    >>> tf_utils.get_shapes(insert_batch3)
    TensorShape([8, 5])
    >>> tf.reduce_all(tf.equal(insert_batch2, insert_batch3[:4, :])).numpy()
    True
    >>> insert_batch4 = fifo(tf.random.normal((4,5)))
    >>> tf_utils.get_shapes(insert_batch4)
    TensorShape([10, 5])
    >>> tf.reduce_all(tf.equal(insert_batch3[2:, :], insert_batch4[:6, :])).numpy()
    True

    :param queue_size: 队列长度，指的是样本数量
    :param queue_dtype: 队列的数据类型
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self, queue_size: int, queue_dtype: Union[str, None] = None, **kwargs):
        super(FIFOLayer, self).__init__(**kwargs)
        self.queue_dtype = queue_dtype
        self.queue_size = queue_size
        self.queue = None

    def build(self, input_shape):
        self.queue = self.add_weight(
            name="queue",
            shape=(self.queue_size, input_shape[-1]),
            dtype=self.queue_dtype,
            initializer=tf.keras.initializers.Constant(value=np.nan),
            trainable=False,
        )
        super(FIFOLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """ 执行fifo层，将 `inputs` 放入队列，并返回输入数据放入队列前的结果

        :param inputs: 要放入队列的数据，维度是[batch_size, input_embedding_size]
        :param kwargs: 额外参数
        :return: 输入数据放入队列前的结果，维度是[valid_queue_size, input_embedding_size]
        """
        # 构建返回结果
        output_mask = tf.reduce_any(~tf.math.is_nan(self.queue), axis=-1)
        result = tf.boolean_mask(self.queue, output_mask, axis=0)

        # 过滤有效的输入，即去除包含nan的行，并取top queue_size 行
        inputs_mask = tf.reduce_any(~tf.math.is_nan(inputs), axis=-1)
        valid_inputs = tf.boolean_mask(inputs, inputs_mask, axis=0)
        valid_inputs = valid_inputs[:self.queue_size]

        # 更新队列
        input_count = tf.shape(valid_inputs)[0]
        self.queue.assign(tf.concat([self.queue[input_count:, :], valid_inputs], axis=0))

        return result

    def clear(self):
        """ 重置队列 """
        self.queue.assign(tf.fill(self.queue.shape, np.nan))


@Module.register("expand_dims_class")
class ExpandDims(layers.Layer):
    def __init__(self, axis: int, **kwargs):
        """包装 tf.expand_dims 以类注册，增加张量维度
        :param axis:指定展开维度索引的整数“axis”必须在范围内`[-(D+1), D]`（包含）。
        """
        self.axis = axis
        super(ExpandDims, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ 增加输入张量的维度

         :param inputs:输入张量
         :param kwargs: 额外参数
         :return: 增加维度后的张量
        """
        output = tf.expand_dims(inputs, axis=self.axis)
        return output


@Module.register("flatten_list")
class FlattenList(layers.Layer):
    def __init__(self, **kwargs):
        """用于将输入嵌套list或多个list组合展平

        """

        super(FlattenList, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ 将list内嵌套结构展平，如输入：[[1, 2, 3], [4, 5]] --> 输出：[1, 2, 3, 4, 5]

         :param inputs:输入为列表形式
         :param kwargs: 额外参数
         :return: 展平后的列表
        """
        flatten_input = []
        for x in inputs:
            if isinstance(x, List):
                flatten_input.extend(x)
            else:
                flatten_input.append(x)
        return flatten_input


@Module.register("flatten_dict")
class FlattenDict(layers.Layer):
    def __init__(self, **kwargs):
        """用于将字典中的各tensor数据进行展开
        """
        super(FlattenDict, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ 将字典中的各tensor数据进行展开， 原形状为[batch_size, x, y]调整为 [batch_size, x*y]
         :param inputs:输入为列表形式
         :param kwargs: 额外参数
         :return: 改变后的列表
        """
        tensor_dict = {}
        test = []
        for feature, tensor in inputs.items():
            new_tensor = tf.reshape(tensor, [-1, tensor.shape[1] * tensor.shape[2]])
            tensor_dict[feature] = new_tensor
            test.append(feature)
        return tensor_dict


def adjust_list(inputs, adjusted_shape):
    if len(adjusted_shape) == 1:
        if len(inputs) != adjusted_shape[0]:
            raise ValueError(f"adjusted_shape: {adjusted_shape} not match len(inputs) {len(inputs)}")
        return inputs
    length = adjusted_shape[0]
    sub_length = len(inputs) // length
    output_list = []
    for idx in range(length):
        output_list.append(adjust_list(inputs[idx * sub_length: (idx + 1) * sub_length], adjusted_shape[1:]))
    return output_list


@Module.register("reshape_list")
class ReshapeList(layers.Layer):
    def __init__(self, adjusted_shape, **kwargs):
        """用于将输入list转化成对应嵌套形状
        :param adjusted_shape: 调整后的列表形状，如[5, 1]，注意调整后的列表元素要与之前的列表元素数量相同
        """
        self.adjusted_shape = list(adjusted_shape)
        super(ReshapeList, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ 将输入list形状调整为 adjusted_shape，如输入：[[1, 2, 3], [4, 5]] adjusted_shape 形状：[5, 1]--> 输出：[[1], [2], [3], [4], [5]]

         :param inputs:输入为列表形式
         :param kwargs: 额外参数
         :return: 改变后的列表
        """
        flatten_input = FlattenList()(inputs)
        reshape_list = adjust_list(flatten_input, self.adjusted_shape)
        return reshape_list


@Module.register("repeat_to_list")
class Repeat2List(layers.Layer):
    def __init__(self, n, **kwargs):
        """用于将输入元素重复输出一个包含N个该元素的列表
        :param n: 元素重复个数
        """
        self.n = n
        super(Repeat2List, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ 将输入元素重复输出一个包含N个该元素的列表，如输入：ts 当n=4时 --> 输出：[ts, ts, ts, ts]

         :param inputs:输入元素
         :param kwargs: 额外参数
         :return: 重复元素后的列表
        """
        output_list = [inputs] * self.n
        return output_list


@Module.register("transpose_list")
class TransposeList(layers.Layer):
    """ 转置列表的列表

    将List[List]的对象类似矩阵一样进行转置，List列表的对位元素组成一个新的List

    """
    def __init__(self, name: str = "transpose_list", **kwargs):
        super(TransposeList, self).__init__(name=name, **kwargs)

    def call(self, inputs: List[List[tf.Tensor]], **kwargs) -> List[List[tf.Tensor]]:
        """ 转置列表

         :param inputs:输入List[List]
         :param kwargs: 额外参数
         :return: 转置后的List[List[tf.Tensor]]
        """
        output_list = []
        length = len(inputs)
        for i in range(len(inputs[0])):
            row = []
            for j in range(length):
                row.append(inputs[j][i])
            output_list.append(row)

        return output_list


@Module.register("parallel")
class Parallel(Layer):
    """并行网络

    对对同一个输入，创建相同模块的不同实例，并输出列表

    :register name: parallel

    :param model_config: Module配置，列表中每个元素都将通过该模块
    :param nums: 实例数
    :param name: 遍历网络层名字
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 model_config: Union[str, Config, None] = None,
                 nums: int = 1,
                 name="parallel",
                 **kwargs):
        super(Parallel, self).__init__(name=name, **kwargs)
        self.model_config = model_config
        self.nums = nums

        self.model_list = None

    def build(self, input_shape):
        """构建遍历网络层
        """
        self.model_list = [Module.from_config(self.model_config) for _ in range(self.nums)]

    def call(self, inputs: Union[List[tf.Tensor], tf.Tensor], **kwargs) -> List[Any]:
        """并行网络

        :param inputs: 每个模型输入,
        :param kwargs: 额外参数
        :return: 每个模型实例的输出列表
        """
        output_list: List[Any] = []
        for i in range(self.nums):
            output_list.append(self.model_list[i](inputs))
        return output_list


@Module.register("position_encode")
class PositionEncodeLayer(Layer):
    """ 位置嵌入层

    :register name: position_encode

    :param seq_dim: 序列长度
    :param embed_dim: emb维度
    :param name: layer名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self, seq_dim, embed_dim, name="position_encode", **kwargs):
        super(PositionEncodeLayer, self).__init__(name=name, **kwargs)
        self.pos_embed = tf.Variable(
            initial_value=lambda: tf.keras.initializers.glorot_uniform(seed=1024)
            (shape=[1, seq_dim, embed_dim], dtype="float32"),
            trainable=True)

    def call(self, inputs):
        return inputs + tf.tile(self.pos_embed, [tf.shape(inputs)[0], 1, 1])


@Module.register("seq_mask")
class SeqMask(Layer):
    """ 序列mask层

    :register name: seq_mask

    :param pad: 序列padding的值
    :param name: layer名
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self, pad, name="seq_mask", **kwargs):
        super(SeqMask, self).__init__(name=name, **kwargs)
        self.pad = tf.constant(pad)

    def call(self, inputs):
        return tf.math.not_equal(inputs, tf.cast(self.pad, dtype=inputs.dtype))


@Module.register("concat_dict")
class ConcatDict(layers.Layer):
    def __init__(self, **kwargs):
        """用于将两个feature_dict进行合并，用后者更新前者
        """
        super(ConcatDict, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ 将list内嵌套结构展平，如输入：[{f1:1,f2:2},{f1:3,f3:3}] --> 输出：{f1:3,f2:2,f3:3}
         :param inputs:输入为列表形式
         :param kwargs: 额外参数
         :return: 合并后的feature_dict
        """
        output = {}
        for feature_dict in inputs:
            for feature in feature_dict:
                output[feature] = feature_dict[feature]
        return output


@Module.register("vec_proj")
class VectorProjection(Layer):
    """向量投影

    将源向量投影到目标向量上去

    :register name: vec_proj

    :param name: vec_proj
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self, name="vec_proj", **kwargs):
        super(VectorProjection, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        """
         :param inputs:输入为列表形式
            * inputs[0]: 源向量
            * inputs[1]: 目标向量
         :param kwargs: 额外参数
         :return: 投影后向量
        """
        source_emb = inputs[0]
        target_emb = inputs[1]
        source_target_product = tf.reduce_sum(source_emb * target_emb, axis=-1)
        target_norm = tf.reduce_sum(target_emb * target_emb, axis=-1)

        projection_coefficients = tf.math.divide_no_nan(source_target_product, target_norm)
        projection_coefficients = K.expand_dims(projection_coefficients, axis=1)

        output = projection_coefficients * target_emb
        return output


@Module.register("dice", tags=["activation"], is_custom=False)
class Dice(Layer):
    def __init__(self, axis=-1, epsilon=1e-8, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        self.bn = None
        self.alpha = None
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = BatchNormalization(axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alpha = self.add_weight(shape=(input_shape[-1],), initializer=Zeros(), dtype=tf.float32, name='alpha')
        super(Dice, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        x_p = tf.sigmoid(inputs_normed)
        return x_p * inputs + self.alpha * (1.0 - x_p) * inputs

    def get_config(self):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


@Module.register("fmcr_interpolate")
class FMCRInterpolate(layers.Layer):
    """fmcr_interpolate，根据条件对输入的embedding字典进行插值

    :register name: fmcr_interpolate
    :config example:
    .. code-block:: json
        local model_structure = [
        ...
        {
            name: 'concat_dict_fmcr',
            type: 'fmcr_interpolate',
            inputs: { inputs: ['ref::postprocess.embedding', 'ref::input2.embedding']},
            outputs: 'output',
        },
        {
            name:'next layer'
            ....
            inputs: {
                inputs: 'ref::values(concat_dict_fmcr.embedding[0])',
            },
        }
        ...
        ]
        local dcn_model = {
            ...
          outputs: {
            ...
            fmcr:'ref::concat_dict_fmcr.embedding[1]'
          }
        };

    :param name: 层名称
    :param kwargs: class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """

    def __init__(self,
                 name: str = "fmcr_interpolate",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.interpolate_flag = False
        self.interpolation_num = 1
        self.feature_names = []
        self.dims = []

    def build(self, input_shape):
        for feature_dict in input_shape:
            for _, feature_shape in feature_dict.items():
                self.dims.append(feature_shape[-1])
        super(FMCRInterpolate, self).build(input_shape)

    def call(self, inputs: List[Dict[str, tf.Tensor]], **kwargs):
        """ 计算 FMCRInterpolate 层

        :param inputs: 字典列表，每个字典中包含需要插值的特征以及对应的embedding
        :param kwargs: 额外参数
        :return: output, interpolate_emb,output为统一的合并字典，interpolate_emb在callback阶段使用用于梯度计算
        """
        output = {}
        for feature_dict in inputs:
            for feature in feature_dict:
                output[feature] = feature_dict[feature]
        self.feature_names = output.keys()
        interpolate_emb = tf.concat(list(output.values()), axis=-1)
        restore_list = tf.split(interpolate_emb, self.dims, axis=-1)
        output = {key: value for key, value in zip(list(output.keys()), restore_list)}

        if self.interpolate_flag:
            emb_expectation = tf.reduce_mean(interpolate_emb, axis=0, name='emb_expectation')
            interpolate_list = [
                (self.interpolation_num - n) / self.interpolation_num
                * emb_expectation + n / self.interpolation_num * interpolate_emb
                for n in range(1, self.interpolation_num + 1)]
            interpolate_emb = tf.concat(interpolate_list, axis=0)
            emb_list = tf.split(interpolate_emb, self.dims, axis=-1)
            output = {key: value for key, value in zip(list(output.keys()), emb_list)}
        return output, interpolate_emb


@Module.register("layer_seq")
class LayersSequential(layers.Layer):
    """layer组合器  将多个layer包装成一个layer
    方便在loop等只能传入单个layer的算子中使用，让配置更加灵活
    :register name: layer_seq
    :config example:
    .. code-block:: json
    {
    name: 'activate_multi_head_tower',
    type: "loop",
    parameters: {
      loop: {
        type: "layer_seq",
        config_list:[{
            type:'dnn',
            hidden_dims: [256,128,64],
            hidden_activation: "relu",
            output_activation: "relu",
        },
        {
            type:'dense',
            units: 1,
            activation: 'sigmoid',
        }
        ]
      },
    },
    inputs: {
      inputs: ["ref::mmoe_out.activate","ref::mmoe_out.activate"],
    },
    outputs: ["activate","activate_self"],
    },

    :param config_list: layer序列配置
    :param name: 输出层名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 config_list: List[Config],
                 name="layer_seq",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.config_list = config_list
        self.layer_list = None

    def build(self, input_shape):
        self.layer_list = [Module.from_config(config) for config in self.config_list]

    def call(self, inputs, **kwargs):
        outputs = inputs
        for layer in self.layer_list:
            outputs = layer(outputs)
        return outputs


@Module.register("list_multiply")           
class ListMultiply(layers.Layer):
    """ 对两个数组进行点乘

    :register name: list_multiply

    :param name: 输出层名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 name: str = "list_multiply",
                 **kwargs):
        super(ListMultiply, self).__init__(name=name, **kwargs)
        
    def build(self, input_shape):
        super().build(input_shape) 

    def call(self, list1: List, list2: List, **kwargs):
        multiplied = []
        for value1, value2 in zip(list1, list2):
            multiplied.append(value1 * value2)
        return multiplied


@Module.register("tower_combine")
class TowerCombine(layers.Layer):
    """ 对分塔数据进行叠加合并

    :register name: tower_combine

    :param name: 输出层名称
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 name: str = "tower_combine",
                 **kwargs):
        super(TowerCombine, self).__init__(name=name, **kwargs)
        
    def build(self, input_shape):
        super().build(input_shape) 

    def call(self, tower_output: List, tower_mask: List, **kwargs):
        combined_tower_output = tf.reduce_sum(tf.stack(
            [tower_output[i] * tf.reshape(tf.cast(tower_mask[i], tower_output[i].dtype),
                                          [-1, 1]) for i in range(len(tower_output))]), axis=0)
        return combined_tower_output


@Module.register("din_pooling")
class DINPooling(Layer):
    """
    din建模序列embed，支持三种风格建模方式
    out为模型原始方式，inner使用内积方式计算embed相关性，attn使用多头自注意力模块计算embed相关性
    craft使用多层的cross-attention模块计算相关性

    Args: params
        style: out、inner、attn、craft
        use_pre_mlp: 对输入的seq以及query是否使用mlp处理成统一维度
        mlp_ratio: DIN结构中隐藏层的expansion_ratio
        act_layer: DIN结构中隐藏层使用的激活函数
        use_softmax: 计算得分后采用softmax处理还是sigmoid处理
        use_proj：加权求和时使用原本embed特征还是带有side_info的embed特征
        head: 多头自注意力模块的头数
        use_bias: 多头子注意力模块的全连接是否采用bias
        embed_dim: 序列建模后输出的embed维度
        craft_layer_num: craft模块层数
        use_craft_pe: craft是否使用位置编码
        craft_ffn_dropout: craft模式下dropout比例
        craft_ffn_activation: craft模式下前馈网络激活函数
        craft_expansion_ratio: craft模式下前馈网络隐层维度扩展因子

    Inputs:
        seq_embed: (batch_size, sequence_len, embedding_size * n)
        query_embed: (batch_size, 1, embedding_size * m)

    Outputs:
        output shape list of '3D tensor' (batch_size, 1, embedding_size)

    参考配置：
    {
        name: 'din_pooling',
        type: 'din_pooling',
        parameters: {
          pooling_config: {
            style: 'out',
            use_pre_mlp: false,
            mlp_ratio: 1,
            act_layer: 'relu',
            use_softmax: true,
            use_proj: true,
            head: 1,
            use_bias: false,
            embed_dim: 128
          }
        },
        inputs: {
          inputs: "ref::values(input.embedding.`name == 'feature2'`)[0]",      # 长序列特征tensor
        },
        outputs: 'embedding',
    }
    """

    def __init__(self, pooling_config, **kwargs):
        self.style = pooling_config.get("style", "out")
        self.use_pre_mlp = pooling_config.get("use_pre_mlp", False)
        self.mlp_ratio = pooling_config.get("mlp_ratio", 1)
        self.act_layer = pooling_config.get("act_layer", 'relu')
        self.use_softmax = pooling_config.get("use_softmax", True)
        self.use_proj = pooling_config.get("use_proj", True)

        self.head = pooling_config.get("head", 1)
        self.use_bias = pooling_config.get("use_bias", False)
        self.embed_dim = pooling_config.get('embed_dim')

        # Craft params
        self.craft_layer_num = pooling_config.get("craft_layer_num", 3)
        self.use_craft_pe = pooling_config.get("use_craft_pe", True)
        self.craft_ffn_dropout = pooling_config.get("craft_ffn_dropout", 0.1)
        self.craft_ffn_activation = pooling_config.get("craft_ffn_activation", "gelu")
        self.craft_expansion_ratio = pooling_config.get("craft_expansion_ratio", 4)

        self.seq_pre_mlp = None
        self.query_pre_mlp = None
        self.dnn = None
        self.proj = None
        self.attn = None
        self.craft_layers = None
        self.pe = None
        super(DINPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        seq_embed_shape, query_embed_shape = input_shape
        min_dim = min(seq_embed_shape[-1], query_embed_shape[-1])
        if self.use_pre_mlp:
            self.seq_pre_mlp = tf.keras.layers.Dense(min_dim, use_bias=False)
            self.query_pre_mlp = tf.keras.layers.Dense(min_dim, use_bias=False)

        # din输出维度默认为输入维度
        if self.embed_dim is None:
            self.embed_dim = seq_embed_shape[-1]

        if self.style == 'out':
            hidden_units = [min_dim * self.mlp_ratio, min_dim, 1]
            self.dnn = Module.from_config(Config(
                {'type': 'dnn', 'hidden_dims': hidden_units, 'hidden_activation': self.act_layer,
                 'output_activation': None, 'orders': 'a'}))
            self.proj = Dense(self.embed_dim, use_bias=False) if self.use_proj else Lambda(lambda x: x)
        elif self.style == 'inner':
            self.proj = Dense(self.embed_dim, use_bias=False) if self.use_proj else Lambda(lambda x: x)
        elif self.style == 'attn':
            self.attn = tf.keras.layers.MultiHeadAttention(
                num_heads=self.head,
                key_dim=int(weird_division(self.embed_dim, self.head)),
                output_shape=self.embed_dim,
                use_bias=self.use_bias
            )
        elif self.style == 'craft':
            self.pe = SinusoidalPositionEncoding(self.embed_dim)
            self.craft_layers = [
                CrossAttentionLayer(
                    head=self.head,
                    embed_dim=self.embed_dim,
                    use_bias=self.use_bias,
                    dropout=self.craft_ffn_dropout,
                    expansion_ratio=self.craft_expansion_ratio,
                    ffn_activation=self.craft_ffn_activation
                ) for _ in range(self.craft_layer_num)
            ]
            logging.info("Din use craft pooling, craft_layer_num=%s, use_craft_pe=%s, craft_ffn_dropout=%s, "
                         "craft_ffn_activation=%s, craft_expansion_ratio=%s",
                         self.craft_layer_num,
                         self.use_craft_pe,
                         self.craft_ffn_dropout,
                         self.craft_ffn_activation,
                         self.craft_expansion_ratio
                         )
        else:
            raise ValueError("The parameter 'style' does not support '%s'.", self.style)
        super(DINPooling, self).build(input_shape)

    def call(self, inputs):
        seq_embed, query_embed = inputs
        # 判断结合side_info后的query和seq的embed_dim是否相同
        # 若不相同且不使用前置nlp处理，则报错告知dim不同
        _, max_len, seq_embed_dim = seq_embed.get_shape().as_list()
        _, _, query_embed_dim = query_embed.get_shape().as_list()
        if seq_embed_dim != query_embed_dim and not self.use_pre_mlp:
            raise ValueError("Sequence embed_dim is not equal to query embed_dim, \
                             check the feature's attention params")
        if self.use_pre_mlp:
            seq = self.seq_pre_mlp(seq_embed)
            query = self.query_pre_mlp(query_embed)
        else:
            seq = seq_embed
            query = query_embed
        if self.style == 'out':
            query = tf.tile(query, [1, max_len, 1])  # [batch, max_len, embed_dim]
            concat_embed = K.concatenate([
                query,
                seq,
                seq - query,
                seq * query
            ],
                axis=2)  # [batch_size, max_len, embed_dimm * 4]
            weight = self.dnn(concat_embed)  # [batch_size, max_len, 1]

            if self.use_softmax:
                weight = weight / (self.embed_dim ** 0.5)
                weight = tf.nn.softmax(weight, axis=-1)
            else:
                weight = tf.nn.sigmoid(weight)
            if not self.use_proj:
                seq_embed = seq_embed[:, :, :self.embed_dim]
            outputs = tf.matmul(weight, seq_embed, transpose_a=True)
            outputs = self.proj(outputs)
        elif self.style == 'inner':
            weight = tf.matmul(seq, query, transpose_b=True)  # bs, max_len, 1
            if self.use_softmax:
                weight = weight / (self.embed_dim ** 0.5)
                weight = tf.nn.softmax(weight, axis=-1)
            else:
                weight = tf.nn.sigmoid(weight)
            if not self.use_proj:
                seq_embed = seq_embed[:, :, :self.embed_dim]
            outputs = tf.matmul(weight, seq_embed, transpose_a=True)
            outputs = self.proj(outputs)
        elif self.style == 'attn':
            outputs = self.attn(query=query, value=seq)
        elif self.style == 'craft':
            if self.use_craft_pe:
                seq = self.pe(seq)

            # Initialize hidden state of layer_0 as query
            outputs = query

            # Use CrossAttentionLayer
            for layer in self.craft_layers:
                outputs = layer([outputs, seq])
        else:
            raise ValueError("The parameter 'style' does not support '%s'.", self.style)
        return outputs

    def compute_output_shape(self, input_shape):
        return (int(input_shape[0]), 1, self.embed_dim)

    def get_config(self):
        config = {
            'style': self.style,
            'use_pre_mlp': self.use_pre_mlp,
            'mlp_ratio': self.mlp_ratio,
            'act_layer': self.act_layer,
            'use_softmax': self.use_softmax,
            'use_proj': self.use_proj,
            'head': self.head,
            'use_bias': self.use_bias,
            'embed_dim': self.embed_dim,
            'craft_layer_num': self.craft_layer_num,
            'use_craft_pe': self.use_craft_pe,
            'craft_ffn_dropout': self.craft_ffn_dropout,
            'craft_ffn_activation': self.craft_ffn_activation,
            'craft_expansion_ratio': self.craft_expansion_ratio
        }
        base_config = super(DINPooling, self).get_config()
        base_config.update(config)
        return base_config


@Module.register("sinusoidal_position_encoder")
class SinusoidalPositionEncoding(Layer):
    """
    SinusoidalPositionEncoding in Transformer
    Args:
        embedding_size: 输入embedding维度
        trainable: 参数是否可训练
    Inputs:
        input shape '3D tensor' (batch_size, feature_length, embedding_size)
    Outputs:
        output shape '3D tensor' (batch_size, feature_length, embedding_size)
    参考配置：
    {
        name: 'sinusoidal_position_encoder',
        type: 'sinusoidal_position_encoder',
        parameters: {
          embedding_size: 128,
          trainable: false,
        },
        inputs: {
          inputs: "ref::values(input.embedding.`name == 'feature2'`)[0]",      # 长序列特征tensor
        },
        outputs: 'embedding',
    },
    """

    def __init__(self, embedding_size, trainable=True):
        self.embedding_size = embedding_size
        self.trainable = trainable
        self.lookup_table = None
        super(SinusoidalPositionEncoding, self).__init__()

    def build(self, input_shape):
        _, seq_length, _ = input_shape.as_list()

        self.lookup_table = tf.Variable(tf.zeros((seq_length, self.embedding_size)), trainable=self.trainable)

        for pos in range(seq_length):
            for i in range(self.embedding_size):
                try:
                    # transformer中位置编码公式，其中10000.和2.是公式固定参数，i//2表示i在奇数偶数中的顺序位置
                    self.lookup_table[pos, i].assign(pos / tf.pow(10000., 2. * (i // 2) / self.embedding_size))
                except ZeroDivisionError:
                    logging.error('Divide by 0 while position encodings')

        self.lookup_table[:, 0::2].assign(tf.sin(self.lookup_table[:, 0::2]))  # 偶数位置取sin编码
        self.lookup_table[:, 1::2].assign(tf.cos(self.lookup_table[:, 1::2]))  # 奇数位置取cos编码

        super(SinusoidalPositionEncoding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        _, seq_length, _ = inputs.get_shape().as_list()
        position_index = tf.expand_dims(tf.range(seq_length), 0)
        outputs = tf.nn.embedding_lookup(self.lookup_table, position_index)

        return outputs + inputs

    def get_config(self):
        config = {'embedding_size': self.embedding_size}
        base_config = super(SinusoidalPositionEncoding, self).get_config()
        base_config.update(config)
        return config


@Module.register("cross_attention")
class CrossAttentionLayer(Layer):
    """
    DIN craft-style建模序列
    Args:
        head: 注意力模块头数
        embed_dim: 嵌入维度
        use_bias: 注意力模块的线性变换层是否采用bias
        dropout: FFN中dropout参数
        expansion_ratio: 前馈网络隐藏层维度的扩展因子, 默认为 4
        ffn_activation: 前馈网络激活函数
    参考配置：
    {
        name: 'cross_attention',
        type: 'cross_attention',
        parameters: {
          head: 1,
          embed_dim: 128,
          use_bias: false,
          dropout: 0.1,
          expansion_ratio: 2,
          ffn_activation: 'relu',
        },
        inputs: {
          inputs: ["ref::values(input.embedding.`name == 'feature2'`)[0]",   # query
                   "ref::values(input.embedding.`name == 'feature2'`)[0]"],  # value
        },
        outputs: 'embedding',
    },
    """

    def __init__(self, head=1, embed_dim=None, use_bias=False, dropout=0.1,
                 expansion_ratio=4, ffn_activation='gelu', **kwargs):
        super(CrossAttentionLayer, self).__init__(**kwargs)
        self.head = head
        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.dropout = dropout
        self.expansion_ratio = expansion_ratio
        self.ffn_activation = ffn_activation

        self.attn = None
        self.ffn = None
        self.layer_norm1 = None
        self.layer_norm2 = None

    def build(self, input_shape):
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.head,
            key_dim=int(weird_division(self.embed_dim, self.head)),
            output_shape=self.embed_dim,
            use_bias=self.use_bias)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embed_dim * self.expansion_ratio, activation=self.ffn_activation),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(self.embed_dim)])
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12)

        super(CrossAttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        query, value = inputs
        attn_outputs = self.attn(query=query, value=value)
        norm_outputs = self.layer_norm1(attn_outputs + query)
        ffn_outputs = self.ffn(norm_outputs)
        outputs = self.layer_norm2(ffn_outputs + norm_outputs)
        return outputs

    def get_config(self):
        config = {
            'head': self.head,
            'embed_dim': self.embed_dim,
            'use_bias': self.use_bias,
            'dropout': self.dropout,
            'expansion_ratio': self.expansion_ratio,
            'ffn_activation': self.ffn_activation
        }
        base_config = super(CrossAttentionLayer, self).get_config()
        base_config.update(config)
        return base_config


@Module.register("feature_keys_values_sizes")
class FeatureKeysValuesSizes(layers.Layer):
    def __init__(self, save_path=None, features_name='values', **kwargs):
        """实现red::values的功能， 返回字段的values，同时记录keys
        """
        self.features_name = features_name
        if save_path is not None:
            # 对save_path进行规范化处理
            self.save_path = os.path.normpath(save_path)
            # 将save_path转换为绝对路径
            self.save_path = os.path.realpath(self.save_path)
            # 检查路径是否在允许的目录范围内
            allowed_dir = '/opt/huawei'
            if not self.save_path.startswith(allowed_dir):
                raise ValueError(f"save_path must be within {allowed_dir}")
        else:
            self.save_path = None

        super(FeatureKeysValuesSizes, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ 实现red::values的功能， 返回字段的values，同时记录keys
         :param inputs:输入为字典表形式 或 列表形式，
         :param kwargs: 额外参数， keys保存路径
         :return: 转成列表的list
        """
        # 检查inputs是否为None
        if inputs is None:
            raise ValueError("inputs cannot be None")

        # 检查inputs是否为字典或可迭代类型
        if isinstance(inputs, dict):
            inputs = [inputs]
        elif not isinstance(inputs, (list, tuple)):
            raise TypeError("inputs must be a dictionary or an iterable (list/tuple)")

        feature_name = []
        emb_size = []
        tensor_list = []
        for feature_input in inputs:
            if not isinstance(feature_input, dict):
                raise TypeError("Each element in inputs must be a dictionary")
            feature_name.extend(list(feature_input.keys()))
            emb_size.extend([value.shape[-1] for value in feature_input.values()])
            tensor_list.extend(list(feature_input.values()))

        if self.save_path:
            set_global_var('values_path', self.save_path)
            self._save_feature_metadata(feature_name, emb_size)

        return tensor_list

    def _save_feature_metadata(self, feature_name, emb_size):
        """Save feature metadata to a JSON file."""
        try:
            if os.path.exists(self.save_path):
                json_file = load_json(self.save_path)
            else:
                json_file = {}

            json_file[f"{self.features_name}_name"] = feature_name
            json_file[f"{self.features_name}_size"] = emb_size

            save_json(self.save_path, json_file, indent=4)
        except Exception as e:
            logger.error(f"Error saving feature metadata: {e}")
