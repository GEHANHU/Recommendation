from typing import List, Tuple, Any, Optional
from typing import Union

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import backend

from modelflow.common import Config
from modelflow.common.config import convert_config_to_dict
from modelflow.integrations.tensorflow.modules import Module


@Module.register("cgc")
class CGC(layers.Layer):
    """PLE模型中Customized Gate Control结构，通过gate控制共享网络和专业网络的输出

    原始论文：Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations

    :register name: cgc

    :param task_num: 任务数量
    :param expert_num: 各任务的专家网络数
    :param shared_num: 共享网络数
    :param expert_net_config: 专家网络配置
    :param shared_net_config: 共享网络配置
    :param gate_config: gate网络配置
    :param name: 层的名字
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 task_num: int = 4,
                 expert_num: Union[int, List[int]] = 2,
                 shared_num: int = 2,
                 expert_net_config: Union[str, Config, None] = None,
                 shared_net_config: Union[str, Config, None] = None,
                 gate_config: Union[str, Config, None] = None,
                 use_shared_status: bool = False,
                 name: str = "cgc",
                 **kwargs):
        super(CGC, self).__init__(name=name, **kwargs)

        self.task_num = task_num
        self.gate_num = task_num + 1 if use_shared_status else task_num
        self.expert_num = expert_num if isinstance(expert_num, list) else [expert_num] * task_num
        self.shared_num = shared_num
        self.expert_net_config = expert_net_config
        self.shared_net_config = shared_net_config
        self.gate_config = gate_config
        self.use_shared_status = use_shared_status

        self.expert_net_list = []
        self.shared_net_list = []
        self.gate_net_list = []

    def build(self, input_shape):

        # expert network
        for i in range(self.task_num):
            expert_list = []
            for _ in range(self.expert_num[i]):
                expert_list.append(Module.from_config(self.expert_net_config))
            self.expert_net_list.append(expert_list)

        # shared network
        if self.shared_num:
            for _ in range(self.shared_num):
                self.shared_net_list.append(Module.from_config(self.shared_net_config))

        # gate fusion
        for _ in range(self.gate_num):
            self.gate_net_list.append(Module.from_config(self.gate_config))

    def call(self, inputs: List[Union[tf.Tensor, List[tf.Tensor]]], **kwargs) -> Tuple[List[Any], Optional[Any]]:
        """调用CGC层

        :param inputs:
            * inputs[0]: key,用于gate生成weight，维度是[batch_size, key_embedding_size]
            * inputs[1]: value，作为专家网络的输入，维度[batch_size, value_embedding_size]
            * inputs[2]: shared，作为共享网络的输入，维度[batch_size, value_embedding_size]
        :param kwargs: 额外参数
        :return: 多任务embedding [batch_size, task_num, embedding_size]
        """
        key = inputs[0] if isinstance(inputs[0], List) else [inputs[0]] * self.task_num
        value = inputs[1] if isinstance(inputs[1], List) and len(inputs[1]) == self.task_num \
            else [inputs[1]] * self.task_num
        shared = inputs[2] if isinstance(inputs[2], List) else [inputs[2]] * 2

        # expert network
        task_output_list = []
        all_task_output_list = []
        shared_output_list = []
        for i in range(self.task_num):
            expert_list = []
            for j in range(self.expert_num[i]):
                output = tf.expand_dims(self.expert_net_list[i][j](value[i]), axis=-2)
                expert_list.append(output)
                all_task_output_list.append(output)
            task_output_list.append(expert_list)

        # shared network
        if self.shared_num:
            for i in range(self.shared_num):
                output = tf.expand_dims(self.shared_net_list[i](shared[1]), axis=-2)
                shared_output_list.append(output)

        # gate fusion
        final_outputs = []
        for i in range(self.task_num):
            task_output = self.gate_net_list[i]([key[i], tf.concat(task_output_list[i] + shared_output_list, axis=-2)])
            final_outputs.append(task_output)

        shared_outputs = []
        # use for PLE
        if self.shared_num and self.use_shared_status:
            shared_outputs.append(
                self.gate_net_list[-1]([shared[0], tf.concat(all_task_output_list + shared_output_list, axis=-2)]))

        return final_outputs, shared_outputs


@Module.register("ple")
class PLE(layers.Layer):
    """PLE主体结构， 串行CGC, 各自任务的CGC输出输入到各自任务的CGC专家网络

    原始论文：Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations

    :register name: ple

    :param layer_num: ple层数 即CGC
    :param cgc_config: cgc网络配置
    :param name: 层的名字
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 layer_num: int = 2,
                 cgc_config: Union[str, Config, None] = None,
                 name: str = "ple",
                 **kwargs):
        super(PLE, self).__init__(name=name, **kwargs)

        self.layer_num = layer_num
        self.cgc_config = cgc_config

        self.cgc_net_list = []

    def build(self, input_shape):

        # expert network
        for i in range(self.layer_num):
            if i == self.layer_num - 1:
                self.cgc_config['use_shared_status'] = False
            self.cgc_net_list.append(Module.from_config(self.cgc_config))

    def call(self, inputs: List[Union[tf.Tensor, List[tf.Tensor]]], **kwargs) -> tf.Tensor:
        """调用PLE层

        :param inputs:
            * inputs[0]: key,用于gate生成weight，维度是[batch_size, key_embedding_size]
            * inputs[1]: value，作为专家网络的输入，维度[batch_size, value_embedding_size]
        :param kwargs: 额外参数
        :return: 多任务embedding列表 List[Tensor[batch_size, embedding_size] * task_num]
        """
        key = inputs[0]
        value = inputs[1]

        task_output_list, shard_output = self.cgc_net_list[0]([key, value, value], **kwargs)
        for i in range(1, self.layer_num):
            task_output_list, shard_output = self.cgc_net_list[i]([task_output_list, task_output_list, shard_output[0]],
                                                                  **kwargs)
        return task_output_list


@Module.register("simpe_tower")
class SimpleTower(layers.Layer):
    """SimpleTower 简单塔结构

    :register name: simpe_tower

    :param tower_net_config: 塔网络配置
    :param name: 层的名字
    :param kwargs: :class:`tf.keras.layers.Layer` 中的额外参数，例如trainable等
    """
    def __init__(self,
                 tower_net_config: Union[str, Config, None] = None,
                 name: str = "simpe_tower",
                 **kwargs):
        super(SimpleTower, self).__init__(name=name, **kwargs)

        self.tower_net_config = tower_net_config

        self.tower_net = None

    def build(self, input_shape):
        self.tower_net = tf.keras.Sequential()
        self.tower_net.add(Module.from_config(self.tower_net_config))
        self.tower_net.add(tf.keras.layers.Dense(units=1))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """调用SimgleTower层

        :param inputs: 塔输入 [batch_size, embedding_size]
        :param kwargs: 额外参数
        :return: logit [batch_size, 1]
        """
        return self.tower_net(inputs)
