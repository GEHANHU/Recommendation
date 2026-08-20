#  !/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

"""
保存tensorflow模块类，用于构建tensorflow模型
"""

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers as kl

from modelflow.integrations.tensorflow.modules.multi_dim_embedding import MultiDimEmbedding
from modelflow.integrations.tensorflow.modules.wukong import WuKong
from modelflow.integrations.tensorflow.modules.interaction import CanLayer
from modelflow.integrations.tensorflow.modules.sequence import ResNet, CaPE
from modelflow.integrations.tensorflow.modules.order_concate_layer import OrderConCateLayer
from modelflow.integrations.tensorflow.modules.base import AbsLayer
from modelflow.integrations.tensorflow.modules.base import AverageLayer
from modelflow.integrations.tensorflow.modules.base import EinSumLayer
from modelflow.integrations.tensorflow.modules.base import FIFOLayer
from modelflow.integrations.tensorflow.modules.base import L2NormalizeLayer
from modelflow.integrations.tensorflow.modules.base import LoopLayer
from modelflow.integrations.tensorflow.modules.base import MathExpressionLayer
from modelflow.integrations.tensorflow.modules.base import MaxLayer
from modelflow.integrations.tensorflow.modules.base import MinLayer
from modelflow.integrations.tensorflow.modules.base import NormDropoutLayer
from modelflow.integrations.tensorflow.modules.base import StringExpressionLayer
from modelflow.integrations.tensorflow.modules.base import SumLayer
from modelflow.integrations.tensorflow.modules.base import ref_call
from modelflow.integrations.tensorflow.modules.base import ExpandDims
from modelflow.integrations.tensorflow.modules.base import FlattenList
from modelflow.integrations.tensorflow.modules.base import FlattenDict
from modelflow.integrations.tensorflow.modules.base import ReshapeList
from modelflow.integrations.tensorflow.modules.base import PositionEncodeLayer
from modelflow.integrations.tensorflow.modules.base import SeqMask
from modelflow.integrations.tensorflow.modules.base import DINPooling
from modelflow.integrations.tensorflow.modules.dnn import DNN
from modelflow.integrations.tensorflow.modules.dcn import DCN
from modelflow.integrations.tensorflow.modules.can_feature_selection import CanFeatureSelection
from modelflow.integrations.tensorflow.modules.embedding import Embedding
from modelflow.integrations.tensorflow.modules.embedding import NormEmbedding
from modelflow.integrations.tensorflow.modules.embedding import PretrainedEmbedding
from modelflow.integrations.tensorflow.modules.hinet import ScenarioAttentiveNetwork
from modelflow.integrations.tensorflow.modules.interaction import Cross
from modelflow.integrations.tensorflow.modules.interaction import FM
from modelflow.integrations.tensorflow.modules.uditsr import UDITSR
from modelflow.integrations.tensorflow.modules.aitm import AITM
from modelflow.integrations.tensorflow.modules.uditsr import InfoNCE
from modelflow.integrations.tensorflow.modules.sesrec import Interest_Contrast
from modelflow.integrations.tensorflow.modules.sesrec import Target_Attention
from modelflow.integrations.tensorflow.modules.module import Module
from modelflow.integrations.tensorflow.modules.rnn import BiRNN
from modelflow.integrations.tensorflow.modules.rnn import CustomRNN
from modelflow.integrations.tensorflow.modules.rnn_cell import StackedRNNCells
from modelflow.integrations.tensorflow.modules.stack_layer import StackLayer
from modelflow.integrations.tensorflow.modules.interaction import GateNU
from modelflow.integrations.tensorflow.modules.interaction import EPNet
from modelflow.integrations.tensorflow.modules.interaction import PPNET
from modelflow.integrations.tensorflow.modules.sdm import SDMLongPreference
from modelflow.integrations.tensorflow.modules.sdm import SDMShortPreference
from modelflow.integrations.tensorflow.modules.srn import SRN
from modelflow.integrations.tensorflow.modules.mind import MINDMultiInterestExtractor
from modelflow.integrations.tensorflow.modules.gate import Gate
from modelflow.integrations.tensorflow.modules.domain_net import DFFMInteractingLayer
from modelflow.integrations.tensorflow.modules.mask_net import MaskBlock
from modelflow.integrations.tensorflow.modules.multitask.mldag import MLDAG
from modelflow.integrations.tensorflow.modules.multitask.mldag import MLDAGExpertLayer
from modelflow.integrations.tensorflow.modules.base import Dice
from modelflow.integrations.tensorflow.modules.dnn_split import SerialsplitDNN
from modelflow.integrations.tensorflow.modules.dnn_split_v2 import SerialsplitDNNV2
from modelflow.integrations.tensorflow.modules.routing import SceneRouting
from modelflow.integrations.tensorflow.modules.mrl import PreMRLEmbedding, PostMrlEmbedding
from modelflow.integrations.tensorflow.modules.interaction import SpeLayer
from modelflow.integrations.tensorflow.modules.time2period_embd import Time2PeriodEmbd
from modelflow.integrations.tensorflow.modules.dcn_v3 import (ExponentialCrossNetwork,
                                                              LinearCrossNetwork, EmbeddingChuckReshape)
from modelflow.integrations.tensorflow.modules.vector_quantizer import ResidualQuantizer
from modelflow.integrations.tensorflow.modules.vector_quantizer import SimVectorQuantizer
from modelflow.integrations.tensorflow.modules.vector_quantizer import VectorQuantizer

# 添加tensorflow自带的layers
Module.register("add", is_custom=False)(kl.Add)
Module.register("additive_attention", is_custom=False)(kl.AdditiveAttention)
Module.register("attention", is_custom=False)(kl.Attention)
Module.register("concatenate", is_custom=False)(kl.Concatenate)
Module.register("dense", is_custom=False)(kl.Dense)
Module.register("conv1d", is_custom=False)(kl.Conv1D)
Module.register("dot", is_custom=False)(kl.Dot)
Module.register("dropout", is_custom=False)(kl.Dropout)
Module.register("flatten", is_custom=False)(kl.Flatten)
Module.register("masking", is_custom=False)(kl.Masking)
Module.register("multiply", is_custom=False)(kl.Multiply)
Module.register("permute", is_custom=False)(kl.Permute)
Module.register("reshape", is_custom=False)(kl.Reshape)
Module.register("repeat_vector", is_custom=False)(kl.RepeatVector)
Module.register("einsum_dense", is_custom=False)(kl.experimental.EinsumDense)
Module.register("orderConCateLayer")(OrderConCateLayer)
# 添加 pooling
Module.register("average_pooling_1d", tags=["pooling"], is_custom=False)(kl.AveragePooling1D)
Module.register("average_pooling_2d", tags=["pooling"], is_custom=False)(kl.AveragePooling2D)
Module.register("average_pooling_3d", tags=["pooling"], is_custom=False)(kl.AveragePooling3D)
Module.register("max_pooling_1d", tags=["pooling"], is_custom=False)(kl.MaxPooling1D)
Module.register("max_pooling_2d", tags=["pooling"], is_custom=False)(kl.MaxPooling2D)
Module.register("max_pooling_3d", tags=["pooling"], is_custom=False)(kl.MaxPooling3D)
Module.register("global_average_pooling_1d", tags=["pooling"], is_custom=False)(kl.GlobalAveragePooling1D)
Module.register("global_average_pooling_2d", tags=["pooling"], is_custom=False)(kl.GlobalAveragePooling2D)
Module.register("global_average_pooling_3d", tags=["pooling"], is_custom=False)(kl.GlobalAveragePooling3D)
Module.register("global_max_pooling_1d", tags=["pooling"], is_custom=False)(kl.GlobalMaxPooling1D)
Module.register("global_max_pooling_2d", tags=["pooling"], is_custom=False)(kl.GlobalMaxPooling2D)
Module.register("global_max_pooling_3d", tags=["pooling"], is_custom=False)(kl.GlobalMaxPooling3D)

# 添加 activations
Module.register("activation", tags=["activation"], is_custom=False)(kl.Activation)
Module.register("elu", tags=["activation"], is_custom=False)(kl.ELU)
Module.register("exponential", tags=["activation"], is_custom=False)(tf.keras.activations.exponential)
Module.register("gelu", tags=["activation"], is_custom=False)(tfa.layers.GELU)
Module.register("hard_sigmoid", tags=["activation"], is_custom=False)(tf.keras.activations.hard_sigmoid)
Module.register("leaky_relu", tags=["activation"], is_custom=False)(kl.LeakyReLU)
Module.register("linear", tags=["activation"], is_custom=False)(tf.keras.activations.linear)
Module.register("p_relu", tags=["activation"], is_custom=False)(kl.PReLU)
Module.register("relu", tags=["activation"], is_custom=False)(kl.ReLU)
Module.register("selu", tags=["activation"], is_custom=False)(tf.keras.activations.selu)
Module.register("sigmoid", tags=["activation"], is_custom=False)(tf.keras.activations.sigmoid)
Module.register("softmax", tags=["activation"], is_custom=False)(kl.Softmax)
Module.register("softplus", tags=["activation"], is_custom=False)(tf.keras.activations.softplus)
Module.register("softsign", tags=["activation"], is_custom=False)(tf.keras.activations.softsign)
Module.register("swish", tags=["activation"], is_custom=False)(tf.keras.activations.swish)
Module.register("tanh", tags=["activation"], is_custom=False)(tf.keras.activations.tanh)
Module.register("threshold_relu", tags=["activation"], is_custom=False)(kl.ThresholdedReLU)

# 添加 embedding
Module.register("origin_embedding", tags=["embedding"], is_custom=False)(kl.Embedding)

# 添加 norm
Module.register("batch_norm", tags=["norm"], is_custom=False)(kl.BatchNormalization)
Module.register("layer_norm", tags=["norm"], is_custom=False)(kl.LayerNormalization)
Module.register("sync_batch_norm", tags=["norm"], is_custom=False)(kl.experimental.SyncBatchNormalization)

# 添加 rnn
Module.register("gru", tags=["rnn"], is_custom=False)(kl.GRU)
Module.register("lstm", tags=["rnn"], is_custom=False)(kl.LSTM)
Module.register("rnn", tags=["rnn"], is_custom=False)(kl.SimpleRNN)

# 添加 rnn cells
Module.register("gru_cell", tags=["rnn_cell"], is_custom=False)(kl.GRUCell)
Module.register("lstm_cell", tags=["rnn_cell"], is_custom=False)(kl.LSTMCell)
Module.register("rnn_cell", tags=["rnn_cell"], is_custom=False)(kl.SimpleRNNCell)

# 添加 noise
Module.register("alpha_dropout", is_custom=False)(kl.AlphaDropout)
Module.register("gaussian_noise", is_custom=False)(kl.GaussianNoise)
Module.register("gaussian_dropout", is_custom=False)(kl.GaussianDropout)

# 添加预处理
Module.register("discretization", tags=["preprocess"], is_custom=False)(kl.Discretization)

# 添加tensorflow自带的方法
Module.register("cast", is_custom=False)(tf.cast)
Module.register("clip_by_norm", is_custom=False)(tf.clip_by_norm)
Module.register("clip_by_value", is_custom=False)(tf.clip_by_value)
Module.register("expand_dims", is_custom=False)(tf.expand_dims)
Module.register("matmul", is_custom=False)(tf.matmul)
Module.register("one_hot", is_custom=False)(tf.one_hot)
Module.register("slice", is_custom=False)(tf.slice)
Module.register("split", is_custom=False)(tf.split)
Module.register("squeeze", is_custom=False)(tf.squeeze)
Module.register("stack", is_custom=False)(tf.stack)
Module.register("stop_gradient", is_custom=False)(tf.stop_gradient)
Module.register("strided_slice", is_custom=False)(tf.strided_slice)
Module.register("unstack", is_custom=False)(tf.unstack)
Module.register("tile", is_custom=False)(tf.tile)
Module.register("tf_multiply", is_custom=False)(tf.multiply)
Module.register("tf_reshape", is_custom=False)(tf.reshape)
