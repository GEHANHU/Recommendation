import abc
import logging
import math
import os
import stat
from typing import Dict, List, Callable, Union, Tuple, NamedTuple

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.generic.initialization import truncated_normal
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.embeddings.embedding_modules import EmbeddingModule, SIDEmbedding
from modeling.generic.sequential.features import TensorDict
from modeling.model_registry import ModelRegistry
from utils.common_utils import weird_division
from modeling.generic.utils.constants import Const, FeatConst
from modeling.generic.sequential.query_fusion_model import FusionModel
from modeling import HAS_FBGEMM_OPS, ENABLE_JAGGED_OPS
from modeling.generic.utils.jagged_utils import dense_to_jagged, jagged_to_padded_dense
from modeling.generic.sequential.transformers import GLUFFN
from modeling.generic.sequential.action_conditioning import ActionConditioningModule


class ProcessedInput(NamedTuple):
    seq_embeddings: torch.Tensor
    # shape [B, N, E], where N is final sequence length (e.g. U[ser] + H[istory] + C[andidates])
    past_embeddings: torch.Tensor  # shape [B, H, E]
    all_timestamps: torch.Tensor   # shape [B, N-U], U[ser] tokens do not have a timestamp
    past_lengths_after_input_processor: torch.Tensor  # Shape [B, ]
    

class TimeBasedPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, embedding_type: str = "learnable"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_type = embedding_type.strip().lower()

        if self.embedding_type == "learnable":
            emb_cls = nn.Embedding
        elif self.embedding_type == "sinusoidal":
            emb_cls = SinusoidalPositionalEmbedding
        else:
            raise ValueError('Invalid time_pos_emb_type. Valid values are: "learnable" and "sinusoidal"')
        
        self.day_emb = emb_cls(31 + 1, embedding_dim)
        self.month_emb = emb_cls(12 + 1, embedding_dim)
        self.weekday_emb = emb_cls(7 + 1, embedding_dim)
        self.hour_emb = emb_cls(24 + 1, embedding_dim)

    def reset_state(self) -> None:
        if self.embedding_type == "learnable":
            std = math.sqrt(weird_division(1.0, self.embedding_dim))
            for emb in [self.day_emb, self.month_emb, self.weekday_emb, self.hour_emb]:
                truncated_normal(emb.weight.data, mean=0.0, std=std)

    def forward(
            self,
            days: torch.Tensor,
            months: torch.Tensor,
            weekdays: torch.Tensor,
            hours: torch.Tensor,
    ) -> torch.Tensor:
        day_embeddings = self.day_emb(days)
        month_embeddings = self.month_emb(months)
        weekday_embeddings = self.weekday_emb(weekdays)
        hour_embeddings = self.hour_emb(hours)
        return day_embeddings + month_embeddings + weekday_embeddings + hour_embeddings
    

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, embedding_dim: int):
        super().__init__()
        
        position = torch.arange(max_len, dtype=torch.int).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        self.register_buffer("pe", torch.zeros(max_len, embedding_dim), persistent=True)  # (L, d)
        self.pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        self.pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe[positions]


class InputProcessorModule(BaseModel):
    def __init__(
            self,
            model_cfg: Dict,
            common_hp: Dict,
            model_cls_dict: Dict
    ):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    @abc.abstractmethod
    def get_embeddings(
            self,
            model_inputs: TensorDict,
            embedder: EmbeddingModule
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def generate_input_sequence(
            self,
            sequence_components: Dict[str, torch.Tensor],
            model_inputs: TensorDict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def forward(
            self,
            model_inputs: TensorDict,
            embedder: EmbeddingModule,
    ) -> ProcessedInput:
        pass

    @abc.abstractmethod
    def unpack_sequence(
            self,
            sequence_embeddings: torch.Tensor,
            num_rerank: int,
    ) -> Dict[str, torch.Tensor]:
        pass


@ModelRegistry.register()
class InterleavedItemActionInputProcessor(InputProcessorModule):
    """
    Interleaved Item-Action Sequence Format: user, item1, action1, ..., itemN, actionN
    Candidate Interleaved sequence is appended if candidates exist in the input
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        feature_conf = common_hp["feature_conf"]

        max_sequence_length = model_conf.get("max_sequence_length", 100)
        self.embedding_dim: int = model_conf.get("item_embedding_dim", 64)
        self.num_ratings = model_conf.get("num_ratings", 5)
        self.use_time_pos_emb: bool = model_conf.get("use_time_pos_emb", False)
        self.use_index_pos_emb: bool = model_conf.get("use_index_pos_emb", True)
        self.index_pos_emb_type: str = model_conf.get("index_pos_emb_type", "learnable").strip().lower()
        self.time_pos_emb_type: str = model_conf.get("time_pos_emb_type", "learnable").strip().lower()
        self.use_ratings_padding_index: bool = model_conf.get("use_ratings_padding_index", False)
        self.dropout_rate: float = model_conf.get("linear_dropout_rate", 0.3)
        self.infer_timestamps_key = feature_conf.get("infer_timestamps_key", "timestamps")
        self.infer_items_key = feature_conf.get("infer_items_key", "item_id")
        self.infer_ratings_key = feature_conf.get("infer_ratings_key", "ratings")

        if self.use_time_pos_emb:
            self._time_pos_emb = TimeBasedPositionalEmbedding(self.embedding_dim, self.time_pos_emb_type)
        if self.use_index_pos_emb:
            if self.index_pos_emb_type == "learnable":
                self._pos_emb = nn.Embedding(max_sequence_length * 2 + 2, self.embedding_dim)
                self.cand_ix_correction = -1
            elif self.index_pos_emb_type == "sinusoidal":
                self._pos_emb = SinusoidalPositionalEmbedding(max_sequence_length * 2 + 2, self.embedding_dim)
                self.cand_ix_correction = 0
            else:
                raise ValueError('Invalid index_pos_emb_type. Valid values are: "learnable" and "sinusoidal"')

        self._dropout = nn.Dropout(p=self.dropout_rate)

        padding_idx = self.num_ratings if self.use_ratings_padding_index else None
        self._rating_emb = nn.Embedding(self.num_ratings + 1, self.embedding_dim, padding_idx=padding_idx)
        self.reset_state()

    def reset_state(self) -> None:
        std = math.sqrt(weird_division(1.0, self.embedding_dim))
        truncated_normal(self._rating_emb.weight.data, mean=0.0, std=std)
        if self.use_time_pos_emb:
            self._time_pos_emb.reset_state()
        if self.use_index_pos_emb:
            truncated_normal(self._pos_emb.weight.data, mean=0.0, std=std)

    def get_preprocessed_masks(
            self,
            past_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        生成预处理后的掩码。

        :param past_ids: 历史ID张量。
        :return: 预处理后的掩码张量, 形状为(B, N * 2)。
        """
        B, N = past_ids.size()
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(B, N * 2)

    def get_embeddings(
            self,
            model_inputs: TensorDict,
            embedder: EmbeddingModule
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_rerank = model_inputs.get(f"{FeatConst.CAND_PFX}_{self.infer_items_key}", torch.tensor([[]])).shape[1]
    
        embedder.candidate_features.run_debug = num_rerank > 0
        past_embeddings = embedder.get_history_item_embeddings(
            {f"{FeatConst.HIST_PFX}_{k}": v for k, v in model_inputs.items()})

        # NOTE Don't need to append candidate prefix to features since features are generated by SequentialFeatures
        candidate_embeddings = (
            embedder.get_candidate_item_embeddings(model_inputs) if num_rerank > 0 else torch.tensor([[]])
        )

        user_feature_embs = embedder.get_user_embeddings(model_inputs)

        return user_feature_embs, past_embeddings, candidate_embeddings

    def generate_input_sequence(
            self,
            sequence_components: Dict[str, torch.Tensor],
            model_inputs: TensorDict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        past_ids = sequence_components[f"{FeatConst.HIST_PFX}_ids"]
        past_embeddings = sequence_components[f"{FeatConst.HIST_PFX}_embeddings"]
        past_lengths = sequence_components[f"{FeatConst.HIST_PFX}_lengths"]
        candidate_embeddings = sequence_components[f"{FeatConst.CAND_PFX}_embeddings"]
        user_feature_embs = sequence_components[f"{FeatConst.USER_PFX}_feature_embeddings"]

        B, N, D = past_embeddings.size()

        # 提取评分 embedding
        rating_emb = self._rating_emb(model_inputs[self.infer_ratings_key])

        # 拼接历史物品 embedding 与评分 embedding, (i1,i2,i3,...), (a1,a2,a3,...)->(i1,a1,i2,a2,i3,a3,...)
        seq_embeddings = torch.cat([past_embeddings, rating_emb], dim=2) * (self.embedding_dim**0.5)
        seq_embeddings = seq_embeddings.view(B, N * 2, D)

        if self.use_time_pos_emb:
            time_pos_embeddings = self._time_pos_emb(
                days=model_inputs["timestamps_day"],
                months=model_inputs["timestamps_month"],
                weekdays=model_inputs["timestamps_weekday"],
                hours=model_inputs["timestamps_hour"],
            )
            time_pos_embeddings = time_pos_embeddings.repeat_interleave(2, dim=1)
            seq_embeddings += time_pos_embeddings

        if self.use_index_pos_emb:
            index_pos_embeddings = self._pos_emb(torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(B, 1))
            seq_embeddings += index_pos_embeddings

        seq_embeddings = self._dropout(seq_embeddings)

        # 生成有效掩码并应用
        valid_mask = self.get_preprocessed_masks(past_ids).unsqueeze(2).float()
        seq_embeddings *= valid_mask
        seq_embeddings = torch.cat((user_feature_embs.unsqueeze(1), seq_embeddings), dim=1)
        # 为了适用于底层加速，需要使user_embeddings序列长度为偶数
        if candidate_embeddings.size(1) == 0:
            seq_embeddings = nn.functional.pad(seq_embeddings, (0, 0, 0, 1, 0, 0), "constant", 0.0)

        past_lengths_after_input_processor = 1 + past_lengths * 2
        return seq_embeddings, past_lengths_after_input_processor

    def forward(
            self,
            model_inputs: TensorDict,
            embedder: EmbeddingModule
    ) -> ProcessedInput:
        """
        Build sequence embedding from model inputs following user, item1, action1, ..., itemN, actionN format.
        Optionally, append the candidate sequence if num_rerank > 0 (eval phase)

        :param model_inputs: batch history information (optionally candidate information as well)
        :param embedder: embedding module that takes raw features and outputs embedding vector
        :return:
            seq_embeddings: Sequence embedding - Training shape (B, 1+N*2+1, E); Eval shape (B, 1+N'*2, E)
            past_embeddings: Historical Item sequence embedding (B, N, E)
            all_timestamps: Sequence of timestamps (historical during training, hist. + cand. during eval)
            past_lengths_after_input_processor: sequence token lengths
        """
        user_feature_embs, past_embeddings, candidate_embeddings = self.get_embeddings(model_inputs, embedder=embedder)
        # 如果是在推理时，将历史序列和候选集序列拼接后返回；如果在训练，则只返回历史序列。

        sequence_components = {
            f"{FeatConst.HIST_PFX}_ids": model_inputs[f"{FeatConst.HIST_PFX}_ids"],
            f"{FeatConst.HIST_PFX}_embeddings": past_embeddings,
            f"{FeatConst.HIST_PFX}_lengths": model_inputs[f"{FeatConst.HIST_PFX}_lengths"],
            f"{FeatConst.CAND_PFX}_embeddings": candidate_embeddings,
            f"{FeatConst.USER_PFX}_feature_embeddings": user_feature_embs,
        }

        seq_embeddings, past_lengths_after_input_processor = self.generate_input_sequence(
            sequence_components, model_inputs
        )

        if torch.onnx.is_in_onnx_export() or candidate_embeddings.size(1) > 0:
            rerank_embeddings = self.process_rerank_embs(
                rerank_embs=candidate_embeddings,
                past_lengths=past_lengths_after_input_processor,
                past_payloads=model_inputs,
            )
            seq_embeddings = torch.concat([seq_embeddings, rerank_embeddings], dim=1)
            all_timestamps = torch.concat(
                [
                    model_inputs[self.infer_timestamps_key],
                    model_inputs[f"{FeatConst.CAND_PFX}_{self.infer_timestamps_key}"],
                ],
                dim=1,
            )
        else:
            all_timestamps = None
            if self.infer_timestamps_key in model_inputs:
                all_timestamps = model_inputs[self.infer_timestamps_key].detach()

        return ProcessedInput(
            seq_embeddings=seq_embeddings,
            past_embeddings=past_embeddings,
            all_timestamps=all_timestamps,
            past_lengths_after_input_processor=past_lengths_after_input_processor
        )

    def process_rerank_embs(
            self,
            rerank_embs: torch.Tensor,  # Shape: (B, C, D)
            past_lengths: torch.Tensor,  # Shape: (B, 1)
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B, N, D = rerank_embs.shape
        rerank_embs = rerank_embs * (self.embedding_dim**0.5)
        if self.use_time_pos_emb:
            time_pos_embeddings = self._time_pos_emb(
                days=past_payloads[f"{FeatConst.CAND_PFX}_timestamps_day"],
                months=past_payloads[f"{FeatConst.CAND_PFX}_timestamps_month"],
                weekdays=past_payloads[f"{FeatConst.CAND_PFX}_timestamps_weekday"],
                hours=past_payloads[f"{FeatConst.CAND_PFX}_timestamps_hour"],
            )
            rerank_embs += time_pos_embeddings
        if self.use_index_pos_emb:
            positions = (past_lengths + self.cand_ix_correction).clamp(min=0)
            time_pos_embeddings = self._pos_emb(
                positions
            ).unsqueeze(1).repeat(1, N, 1)  # Shape: (B, C, D)
            rerank_embs += time_pos_embeddings
        rerank_embs = nn.functional.pad(rerank_embs, (0, 0, 0, 1, 0, 0), "constant", 0.0)
        return rerank_embs

    def get_num_ratings(self) -> int:
        return self.num_ratings

    def unpack_sequence(
            self,
            sequence_embeddings: torch.Tensor,
            num_rerank: int,
    ) -> Dict[str, torch.Tensor]:
        elements = {
            "user_embeddings": sequence_embeddings[:, :1, :],
            "item_embeddings": sequence_embeddings[:, 1: -num_rerank - 1: 2, :],
            "action_embeddings": sequence_embeddings[:, 2: -num_rerank - 1: 2, :],
        }

        if torch.onnx.is_in_onnx_export() or num_rerank > 0:
            elements[f"{FeatConst.CAND_PFX}_item_embeddings"] = sequence_embeddings[:, -num_rerank - 1: -1, :]

        return elements


@ModelRegistry.register()
class FusedItemActionInputProcessor(InputProcessorModule):
    """
    Fused Item-Action Sequence Format: user, item1[+action1], ...,
    itemN+[actionN], cand_item1[+mask_action], ..., cand_itemM[+mask_action]
    Candidates are assumed to exist in the input
    Action token is fused with Item token if `use_action_emb` is True. Mask action token is used for candidates.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        model_conf = common_hp.get("model_conf")
        feat_conf = common_hp.get("feature_conf")
        max_sequence_length = model_conf.get("max_sequence_length", 512)
        gr_output_length = model_conf.get("gr_output_length", 0)
        self.max_sequence_len = max_sequence_length + gr_output_length
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        num_ratings = model_conf.get("num_ratings", 5)

        self._dropout_rate: float = model_cfg[Const.HP].get("embedding_dropout_rate", 0.3)
        self._emb_dropout = nn.Dropout(p=self._dropout_rate)

        self.num_ratings = num_ratings
        self._hist_infer_ratings_key = feat_conf.get(
            f"{FeatConst.HIST_PFX}_ratings_column",
            FeatConst.DFLT_HIST_RATINGS_KEY
        )
        self._cand_infer_ratings_key = feat_conf.get(
            f"{FeatConst.CAND_PFX}_ratings_column",
            FeatConst.DFLT_CAND_RATINGS_KEY
        )
        self.hist_ts_key = feat_conf.get(f"{FeatConst.HIST_PFX}_timestamps_column", FeatConst.DFLT_HIST_TS_KEY)
        self.cand_ts_key = feat_conf.get(f"{FeatConst.CAND_PFX}_timestamps_column", FeatConst.DFLT_CAND_TS_KEY)
        self.hist_date_key = feat_conf.get(f"{FeatConst.HIST_PFX}_date_column", FeatConst.DFLT_HIST_DATE_KEY)
        self.cand_date_key = feat_conf.get(f"{FeatConst.CAND_PFX}_date_column", FeatConst.DFLT_CAND_DATE_KEY)
        self.hist_items_key = feat_conf.get(f"{FeatConst.HIST_PFX}_items_key", FeatConst.DFLT_HIST_ITEM_KEY)
        self.cand_items_key = feat_conf.get(f"{FeatConst.CAND_PFX}_items_key", FeatConst.DFLT_CAND_ITEM_KEY)

        self.use_time_pos_emb: bool = model_conf.get("use_time_pos_emb", True)
        self.use_index_pos_emb: bool = model_conf.get("use_index_pos_emb", True)
        self.index_pos_emb_type: str = model_conf.get("index_pos_emb_type", "learnable").strip().lower()
        self.time_pos_emb_type: str = model_conf.get("time_pos_emb_type", "learnable").strip().lower()

        logging.info(f"use_time_pos_emb:{self.use_time_pos_emb};self.use_index_pos_emb:{self.use_index_pos_emb}")

        if self.use_time_pos_emb:
            self._time_pos_emb = TimeBasedPositionalEmbedding(self._embedding_dim, self.time_pos_emb_type)
        if self.use_index_pos_emb:
            if self.index_pos_emb_type == "learnable":
                self._pos_emb = nn.Embedding(max_sequence_length + 1, self._embedding_dim)
                self.cand_ix_correction = -1
            elif self.index_pos_emb_type == "sinusoidal":
                self._pos_emb = SinusoidalPositionalEmbedding(max_sequence_length + 1, self._embedding_dim)
                self.cand_ix_correction = 0
            else:
                raise ValueError('Invalid index_pos_emb_type. Valid values are: "learnable" and "sinusoidal"')

        model_conf = common_hp.get("model_conf")
        logging.info(f"use_action_emb:{model_conf.get('use_action_emb')}")
        if model_conf.get("use_action_emb", True):
            self._rating_emb: nn.Embedding = nn.Embedding(
                num_ratings + 1, self._embedding_dim, padding_idx=num_ratings
            )
        else:
            self._rating_emb = None

        self.mask_emb_id = self.num_ratings

        self.use_sid = feat_conf.get("use_sid", False)
        logging.info("use sid %s", self.use_sid)
        if self.use_sid:
            self.sid_embedding_module = SIDEmbedding(feat_conf, model_conf)
        self.fusion_model = FusionModel()

        self.reset_state()

    def reset_state(self) -> None:
        if self.use_index_pos_emb and self.index_pos_emb_type.lower() == "learnable":
            truncated_normal(
                self._pos_emb.weight.data,
                mean=0.0,
                std=math.sqrt(weird_division(1.0, self._embedding_dim)),
            )
        if self._rating_emb:
            truncated_normal(
                self._rating_emb.weight.data,
                mean=0.0,
                std=math.sqrt(weird_division(1.0, self._embedding_dim)),
            )
        if self.use_time_pos_emb:
            self._time_pos_emb.reset_state()

    def get_preprocessed_masks(
        self,
        past_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        生成预处理后的掩码。

        :param past_ids: 历史ID张量。
        :return: 预处理后的掩码张量, 形状为(B, N)。
        """
        B, N = past_ids.size()
        return (past_ids != 0).reshape(B, N)

    def get_embeddings(
            self,
            model_inputs: TensorDict,
            embedder: EmbeddingModule,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        past_embeddings = embedder.get_history_item_embeddings(model_inputs)
        candidate_embeddings = embedder.get_candidate_item_embeddings(model_inputs)
        user_feature_embs = embedder.get_user_embeddings(model_inputs)
        try:
            query_feature_embs = embedder.get_query_item_embeddings(model_inputs)
        except Exception as e:
            query_feature_embs = None

        return user_feature_embs, past_embeddings, candidate_embeddings, query_feature_embs

    def generate_input_sequence(
            self,
            sequence_components: Dict[str, torch.Tensor],
            model_inputs: TensorDict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_embeddings = sequence_components.get(f"{FeatConst.HIST_PFX}_embeddings")
        past_lengths = sequence_components.get(f"{FeatConst.HIST_PFX}_lengths")
        past_ratings = sequence_components.get(self._hist_infer_ratings_key)
        candidate_embeddings = sequence_components.get(f"{FeatConst.CAND_PFX}_embeddings")
        candidate_ratings = sequence_components.get(self._cand_infer_ratings_key)
        user_feature_embs = sequence_components.get(f"{FeatConst.USER_PFX}_feature_embeddings")
        query_embeddings = sequence_components.get(f"{FeatConst.QUER_PFX}_embeddings")
        # 1. Normalize embeddings

        past_embeddings = past_embeddings * (self._embedding_dim**0.5)  # (B, N, D)
        candidate_embeddings = candidate_embeddings * (self._embedding_dim**0.5)  # (B, M ,D)
        if query_embeddings is not None:
            query_embeddings = query_embeddings * (self._embedding_dim ** 0.5)  # (B, M ,D)

        # 2. Add positional embeddings
        if self.use_time_pos_emb:
            time_pos_embeddings = self._time_pos_emb(
                days=model_inputs["timestamps_day"],
                months=model_inputs["timestamps_month"],
                weekdays=model_inputs["timestamps_weekday"],
                hours=model_inputs["timestamps_hour"],
            )
            past_embeddings += time_pos_embeddings

            candidate_time_pos_embeddings = self._time_pos_emb(
                days=model_inputs[f"{FeatConst.CAND_PFX}_timestamps_day"],
                months=model_inputs[f"{FeatConst.CAND_PFX}_timestamps_month"],
                weekdays=model_inputs[f"{FeatConst.CAND_PFX}_timestamps_weekday"],
                hours=model_inputs[f"{FeatConst.CAND_PFX}_timestamps_hour"],
            )
            candidate_embeddings += candidate_time_pos_embeddings

        if self.use_index_pos_emb:
            B, Nh, _ = past_embeddings.size()
            Nc = candidate_embeddings.size(1)
            past_embeddings += self._pos_emb(torch.arange(Nh, device=past_embeddings.device).unsqueeze(0).repeat(B, 1))
            if self.training:
                # Compare dates for position
                date_diff = (
                    model_inputs[self.cand_date_key].unsqueeze(1) -
                    model_inputs[self.hist_date_key].unsqueeze(2).repeat(1, 1, Nc)
                )
                date_mask = date_diff <= 0
                # In case a candidate has future date than all history items (e.g. negative candidates)
                last_pos_offset = (~date_mask).all(dim=1) * past_lengths.unsqueeze(1)
                candidate_embeddings += self._pos_emb(torch.argmax(date_mask.int(), 1) + last_pos_offset)
            else:
                positions = (past_lengths + self.cand_ix_correction).clamp(min=0)
                candidate_embeddings += self._pos_emb(
                    positions
                ).unsqueeze(1).repeat(1, Nc, 1)

        # 3. Concatenate history and candidate sequences
        seq_embeddings = torch.cat([past_embeddings, candidate_embeddings], dim=1)  # (B, N+M, D)

        # 4. Fuse action embedding into the sequence (candidates are fused with a mask action token)
        if self._rating_emb:
            past_rating_emb = self._rating_emb(past_ratings)
            fake_candidate_ratings = torch.ones_like(candidate_ratings) * self.mask_emb_id
            cand_rating_emb = self._rating_emb(fake_candidate_ratings)
            user_rating_embeddings = torch.cat([past_rating_emb, cand_rating_emb], dim=1)
            seq_embeddings = seq_embeddings + user_rating_embeddings

        # 5. dropout
        seq_embeddings = self._emb_dropout(seq_embeddings)

        # 6. Add user embedding at the beginning of the sequence
        if user_feature_embs.numel() == 0:
            B, _, D = seq_embeddings.shape
            user_feature_embs = torch.zeros(B, 1, D, device=seq_embeddings.device)
        seq_embeddings = torch.cat((user_feature_embs, seq_embeddings), dim=1)  # (B, 1+N, D)

        return seq_embeddings, 1 + past_lengths, query_embeddings

    def forward(
            self,
            model_inputs: TensorDict,
            embedder: EmbeddingModule,
    ) -> ProcessedInput:
        """
        Build sequence embedding from model inputs following user, item1, ...,
        itemN, cand_item1, ..., cand_itemM format.
        Optionally, append the candidate sequence if num_rerank > 0 (eval phase)

        :param model_inputs: batch history information with candidate information
        :param embedder: embedding module that takes raw features and outputs embedding vector
        :return:
            seq_embeddings: Sequence embedding - Shape (B, 1+N+M, E)
            past_embeddings: Historical Item sequence embedding (B, N, E)
            all_timestamps: Sequence of hist. + cand. timestamps
            past_lengths_after_input_processor: sequence token lengths
        """
        user_feature_embs, past_embeddings, candidate_embeddings, query_embeddings = \
            self.get_embeddings(model_inputs, embedder=embedder)
        # 融合sid特征
        if self.use_sid:
            (past_embeddings,
             candidate_embeddings) = self.sid_emb_module(model_inputs, past_embeddings, candidate_embeddings)

        sequence_components = {
            f"{FeatConst.HIST_PFX}_embeddings": past_embeddings,
            f"{FeatConst.HIST_PFX}_lengths": model_inputs.get(f"{FeatConst.HIST_PFX}_lengths"),
            self._hist_infer_ratings_key: model_inputs.get(self._hist_infer_ratings_key),
            f"{FeatConst.CAND_PFX}_embeddings": candidate_embeddings,
            self._cand_infer_ratings_key: model_inputs.get(self._cand_infer_ratings_key),
            f"{FeatConst.USER_PFX}_feature_embeddings": user_feature_embs,
            f"{FeatConst.QUER_PFX}_embeddings": query_embeddings,
        }

        seq_embeddings, past_lengths_after_input_processor, query_embeddings = self.generate_input_sequence(
            sequence_components, model_inputs
        )

        if query_embeddings is not None:
            # 确保维度匹配，防止 fusion_model 出错
            if seq_embeddings.size(1) != query_embeddings.size(1):
                logging.warning("seq_embeddings and query_embeddings length mismatch, skipping fusion.")
            else:
                seq_embeddings = self.fusion_model(seq_embeddings, query_embeddings)

        all_timestamps = torch.concat([model_inputs.get(self.hist_ts_key), model_inputs.get(self.cand_ts_key)], dim=1)

        return ProcessedInput(
            seq_embeddings=seq_embeddings,
            past_embeddings=past_embeddings,
            all_timestamps=all_timestamps,
            past_lengths_after_input_processor=past_lengths_after_input_processor
        )

    def unpack_sequence(
            self,
            sequence_embeddings: torch.Tensor,
            num_rerank: int,
    ) -> Dict[str, torch.Tensor]:
        elements = {
            "user_embeddings": sequence_embeddings[:, :1, :],
            "item_embeddings": sequence_embeddings[:, 1: -num_rerank, :],
        }

        if torch.onnx.is_in_onnx_export() or num_rerank > 0:
            elements[f"{FeatConst.CAND_PFX}_item_embeddings"] = sequence_embeddings[:, -num_rerank:, :]

        return elements


@ModelRegistry.register()
class UserItemInputFeaturePreprocessor(InputProcessorModule):
    """
    用户-物品输入特征预处理器，支持候选特征分组标记化。

    该类支持两种标记化模式：
    1. 手动分配：用户为每个组指定target_tokens
    2. 加权分配：基于组权重自动分配标记
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        data_loader_conf = common_hp.get("data_loader_conf")
        model_conf = common_hp.get("model_conf")
        feat_conf = common_hp.get("feature_conf")

        if hasattr(data_loader_conf, 'action_history_length'):
            history_length = data_loader_conf.get('action_history_length', 150)
        else:
            history_length = data_loader_conf.get('history_length', 400)
        gr_output_length = model_conf.get('gr_output_length', 0)
        self.max_sequence_len = history_length + gr_output_length
        self._history_embedding_dim = model_conf.get("item_embedding_dim", None)

        self.n_u = feat_conf.get("n_u", 1)
        self.n_c = feat_conf.get("n_c", 1)
        self.init_fusion_layers(feat_conf)

        self._use_action_emb: bool = model_conf.get('use_action_emb', False)
        self._use_pos_emb: bool = model_conf.get('use_pos_emb', False)
        num_ratings = feat_conf.get("num_ratings", 3)

        self._history_length = data_loader_conf.get("history_length", 150)
        self._num_rerank = data_loader_conf.get("num_rerank", 400)

        model_hp = model_cfg[Const.HP]
        self._dropout_rate: float = model_hp.get("embedding_dropout_rate", 0.3)
        self._emb_dropout = torch.nn.Dropout(p=self._dropout_rate)

        self.num_ratings = num_ratings
        self._hist_infer_ratings_key = feat_conf.get("history_ratings_column", "history_action_type")
        self._cand_infer_ratings_key = feat_conf.get("candidate_ratings_column", "candidate_action_type")

        if self._use_pos_emb:
            self._pos_aligned_side = model_hp.get('pos_aligned_side', 'right')
            self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
                self.max_sequence_len + 2, self._history_embedding_dim if self._history_embedding_dim else 256,
            )

        if self._use_action_emb:
            self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
                num_ratings + 2, self._history_embedding_dim if self._history_embedding_dim else 256
            )
        else:
            self._rating_emb = None

        self.mask_emb_id = num_ratings + 1

        self.reset_state()

        # 标记化设置
        self._setup_tokenization(feat_conf, model_conf)

        # 动作条件化模块 (PinRec启发)
        self.use_action_conditioning = model_conf.get("use_action_conditioning", False)
        if self.use_action_conditioning:
            self.action_conditioning = ActionConditioningModule(
                action_emb_dim=self._embedding_dim,
                token_emb_dim=self._embedding_dim,
                num_action_types=num_ratings,
                use_film=model_conf.get("use_film", True),
                use_gated_fusion=model_conf.get("use_gated_fusion", True),
                use_attention_biasing=model_conf.get("use_attention_biasing", True),
                film_hidden_dim=model_conf.get("film_hidden_dim", 128),
                gate_hidden_dim=model_conf.get("gate_hidden_dim", 64),
                attention_bias_scale=model_conf.get("attention_bias_scale", 0.1)
            )

    def _setup_tokenization(self, feat_conf, model_conf):
        # 获取标记化配置
        tokenization_conf = feat_conf.get("tokenization", {})
        self.base_dim = tokenization_conf.get("base_dim", 16)
        self.hidden_dim = tokenization_conf.get("hidden_dim", model_conf.get("item_embedding_dim", 128))
        self.tokenization_type = tokenization_conf.get("type", "manual")  # "manual" 或 "weighted"

        # 获取候选特征组配置
        self.candidate_feature_groups = feat_conf.get("candidate_feature_groups", {})
        if not self.candidate_feature_groups:
            logging.warning("未指定candidate_feature_groups，标记化将被禁用")
            self.enable_tokenization = False
        else:
            self.enable_tokenization = True
            if self.tokenization_type == "weighted":
                self._init_weighted_tokenization(feat_conf)
            else:
                self._init_manual_tokenization(feat_conf)

    def _init_common_tokenization_components(self, feat_conf):
        """初始化两种标记化类型都需要的通用组件。"""
        candidate_feature_columns = feat_conf.get("candidate_item_feature_columns", {})

        # 构建特征维度映射: feature_name -> (start_idx, end_idx, dim)
        self.feature_dim_map = {}
        current_idx = 0
        for feat_name, feat_info in candidate_feature_columns.items():
            if feat_info.get("enabled", True):
                feat_dim = feat_info.get("dim", 64)
                self.feature_dim_map[feat_name] = (current_idx, current_idx + feat_dim, feat_dim)
                current_idx += feat_dim

        self.total_candidate_dim = current_idx
        return candidate_feature_columns

    def _parse_feature_groups(self, candidate_feature_columns):
        """解析特征组并计算维度 - 通用逻辑。"""
        group_features = []
        group_feature_indices = []
        group_names = []
        group_total_dims = []
        group_feature_counts = []

        for group_name, group_config in self.candidate_feature_groups.items():
            features = group_config.get("features", [])

            group_start_idx = None
            group_end_idx = None
            group_total_dim = 0
            feature_count = 0

            for feat_name in features:
                if feat_name in self.feature_dim_map:
                    start_idx, end_idx, feat_dim = self.feature_dim_map[feat_name]
                    if group_start_idx is None:
                        group_start_idx = start_idx
                    group_end_idx = end_idx
                    group_total_dim += feat_dim
                    feature_count += 1
                else:
                    logging.warning(f"特征 {feat_name} 在candidate_feature_columns中未找到，跳过")

            if group_total_dim == 0:
                logging.warning(f"组 {group_name} 没有有效特征，跳过")
                continue

            group_features.append(group_total_dim // self.base_dim)
            group_feature_indices.append((group_start_idx, group_end_idx))
            group_names.append(group_name)
            group_total_dims.append(group_total_dim)
            group_feature_counts.append(feature_count)

        return group_features, group_feature_indices, group_names, group_total_dims, group_feature_counts

    def _build_tokenization_buffers(self, chunk_dims):
        # 构建全局静态收集索引和掩码
        gather_index = torch.zeros((self.total_tokens, self.c_max), dtype=torch.long)
        mask = torch.zeros((1, self.total_tokens, self.c_max), dtype=torch.float32)

        global_feat_ptr = 0
        token_idx = 0

        for g_feat, g_tok in zip(self.group_features, self.target_tokens):
            concat_dim = g_feat * self.base_dim
            c_dim = math.ceil(concat_dim / g_tok)

            for k in range(g_tok):
                start_ptr = global_feat_ptr + k * c_dim
                end_ptr = min(global_feat_ptr + (k + 1) * c_dim, global_feat_ptr + concat_dim)
                actual_len = end_ptr - start_ptr

                gather_index[token_idx, :actual_len] = torch.arange(start_ptr, end_ptr)
                mask[0, token_idx, :actual_len] = 1.0
                token_idx += 1

            global_feat_ptr += concat_dim

        self.register_buffer('gather_index', gather_index)
        self.register_buffer('mask', mask)

        # 融合批矩阵乘法权重
        self.token_weight = nn.Parameter(torch.Tensor(self.total_tokens, self.c_max, self.hidden_dim))
        self.token_bias = nn.Parameter(torch.Tensor(self.total_tokens, self.hidden_dim))

        # 初始化参数
        self._reset_tokenization_parameters(chunk_dims)

    def _init_manual_tokenization(self, feat_conf):
        """初始化手动标记化组件。"""
        self._init_common_tokenization_components(feat_conf)

        # 解析特征组并计算块大小
        self.group_features = []
        self.target_tokens = []
        self.group_feature_indices = []

        for group_name, group_config in self.candidate_feature_groups.items():
            features = group_config.get("features", [])
            target_tokens = group_config.get("target_tokens", 1)

            group_start_idx = None
            group_end_idx = None
            group_total_dim = 0

            for feat_name in features:
                if feat_name in self.feature_dim_map:
                    start_idx, end_idx, feat_dim = self.feature_dim_map[feat_name]
                    if group_start_idx is None:
                        group_start_idx = start_idx
                    group_end_idx = end_idx
                    group_total_dim += feat_dim
                else:
                    logging.warning(f"特征 {feat_name} 在candidate_feature_columns中未找到，跳过")

            if group_total_dim == 0:
                logging.warning(f"组 {group_name} 没有有效特征，跳过")
                continue

            self.group_features.append(group_total_dim // self.base_dim)
            self.target_tokens.append(target_tokens)
            self.group_feature_indices.append((group_start_idx, group_end_idx))

        self.total_features = sum(self.group_features)
        self.total_tokens = sum(self.target_tokens)

        # 预计算块维度并找到全局最大值
        chunk_dims = []
        for g_feat, g_tok in zip(self.group_features, self.target_tokens):
            concat_dim = g_feat * self.base_dim
            c_dim = math.ceil(concat_dim / g_tok)
            chunk_dims.extend([c_dim] * g_tok)

        self.c_max = max(chunk_dims) if chunk_dims else 1

        # 构建标记化组件
        self._build_tokenization_buffers(chunk_dims)

        # 更新n_c以匹配总标记数
        self.n_c = self.total_tokens
        logging.info("初始化标记化后，n_c变为: %s", self.n_c)

    def _init_weighted_tokenization(self, feat_conf):
        """初始化加权标记化组件，支持自动标记分配。"""
        self._init_common_tokenization_components(feat_conf)

        # 使用通用逻辑解析特征组
        (self.group_features, self.group_feature_indices, group_names,
         group_total_dims, group_feature_counts) = self._parse_feature_groups(
            self._init_common_tokenization_components(feat_conf)
        )

        # 计算权重并分配标记
        n_c = self.n_c
        num_groups = len(group_names)

        if num_groups == 0:
            logging.warning("未找到有效组，标记化将被禁用")
            self.enable_tokenization = False
            return

        # 计算每组的权重: (特征数量 * 总维度)
        weights = []
        for feat_count, total_dim in zip(group_feature_counts, group_total_dims):
            weights.append(feat_count * total_dim)

        total_weight = sum(weights)
        if total_weight == 0:
            logging.warning("总权重为0，使用相等标记分配")
            weights = [1.0] * num_groups
            total_weight = num_groups

        # 初始标记分配 (与权重成比例，每组至少1个标记)
        self.target_tokens = []
        for weight in weights:
            allocated = max(1, round(weight * n_c / total_weight))
            self.target_tokens.append(allocated)

        # 调整以精确匹配n_c
        current_total = sum(self.target_tokens)

        # 如果需要，向权重最高的组添加标记
        while current_total < n_c:
            max_weight_idx = max(range(num_groups), key=lambda i: weights[i])
            self.target_tokens[max_weight_idx] += 1
            current_total += 1

        # 如果需要，从权重最低的组移除标记(但至少保留1个)
        while current_total > n_c:
            min_weight_idx = min(range(num_groups),
                                 key=lambda i: weights[i] if self.target_tokens[i] > 1 else float('inf'))
            if self.target_tokens[min_weight_idx] > 1:
                self.target_tokens[min_weight_idx] -= 1
                current_total -= 1
            else:
                break

        # 记录标记分配
        for i, name in enumerate(group_names):
            logging.info(f"组 '{name}': {group_feature_counts[i]} 个特征, {group_total_dims[i]} 维度, "
                         f"分配 {self.target_tokens[i]} 个标记 (权重: {weights[i] / total_weight:.3f})")

        self.total_features = sum(self.group_features)
        self.total_tokens = sum(self.target_tokens)

        # 预计算块维度并找到全局最大值
        chunk_dims = []
        for g_feat, g_tok in zip(self.group_features, self.target_tokens):
            concat_dim = g_feat * self.base_dim
            c_dim = math.ceil(concat_dim / g_tok)
            chunk_dims.extend([c_dim] * g_tok)

        self.c_max = max(chunk_dims) if chunk_dims else 1

        # 构建标记化组件
        self._build_tokenization_buffers(chunk_dims)

        # 更新n_c以匹配总标记数
        self.n_c = self.total_tokens
        logging.info("初始化加权标记化后，n_c变为: %s", self.n_c)

    def reset_state(self) -> None:
        if self._use_pos_emb:
            dim = self._pos_emb.embedding_dim
            truncated_normal(
                self._pos_emb.weight.data, mean=0.0, std=math.sqrt(weird_division(1.0, dim)),
            )
        if self._rating_emb:
            dim = self._rating_emb.embedding_dim
            truncated_normal(
                self._rating_emb.weight.data, mean=0.0, std=math.sqrt(weird_division(1.0, dim)),
            )

    def get_feat_dims(self, feature_conf):
        candidate_feature_columns: Dict = feature_conf.get('candidate_item_feature_columns', None)
        history_feature_columns: Dict = feature_conf.get('history_item_feature_columns', None)
        user_feature_columns: Dict = feature_conf.get('user_feature_columns', None)
        if user_feature_columns is None or candidate_feature_columns is None or history_feature_columns is None:
            raise ValueError(
                "user_feature_columns, candidate_feature_columns 和 history_feature_columns 不能为 None")
        all_feature_columns = candidate_feature_columns | history_feature_columns | user_feature_columns  # 合并
        feats_dim = {}

        for feature_name, feature_info in all_feature_columns.items():
            feature_dtype = feature_info.get("dtype", FeatConst.DFLT_DTYPE)
            if feature_dtype == "con":
                feats_dim[feature_name] = 1
            elif feature_dtype == "int" or feature_dtype == "context":
                feats_dim[feature_name] = feature_info.get('dim', FeatConst.FEAT_DIM)
            elif feature_dtype == "multi":
                shared_feat_name = feature_info.get("shared_feat_name", "")
                if shared_feat_name not in feats_dim.keys():
                    feat_dim = all_feature_columns.get(shared_feat_name).get('dim', FeatConst.FEAT_DIM)
                    feats_dim[shared_feat_name] = feat_dim
                    feats_dim[feature_name] = feat_dim
                else:
                    feats_dim[feature_name] = feats_dim[shared_feat_name]
        return feats_dim

    def _calculate_input_dim(self, feature_names: List, feats_dim: Dict) -> int:
        input_dim = 0
        for feat_name in feature_names:
            input_dim += feats_dim.get(feat_name, 0)
        return input_dim

    def init_fusion_layers(self, feature_conf):
        feats_dim = self.get_feat_dims(feature_conf)
        feature_groups = feature_conf.get("feature_groups")

        user_group = feature_groups.get(FeatConst.USER_PFX)
        cand_group = feature_groups.get(FeatConst.CAND_PFX)
        user_dim = self._calculate_input_dim(user_group.get("features"), feats_dim)
        cand_dim = self._calculate_input_dim(cand_group.get("features"), feats_dim)

        self._user_mlp = torch.nn.Linear(user_dim, self.n_u * self._history_embedding_dim)
        logging.info(
            f"将使用MLP对齐用户MLP的维度: "
            f"{user_dim} -> {self.n_u * self._history_embedding_dim}")
        self._candidate_mlp = torch.nn.Linear(cand_dim, self.n_c * self._history_embedding_dim)
        logging.info(
            f"将使用MLP对齐候选MLP的维度: "
            f"{cand_dim} -> {self.n_c * self._history_embedding_dim}")

    def get_preprocessed_masks(
            self,
            history_ids: torch.Tensor,
            candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, N = history_ids.size()
        return (history_ids != 0).reshape(B, N)

    def _generate_history_position_ids(
            self,
            history_lengths: torch.Tensor,
            max_length: int
    ) -> torch.Tensor:
        B = history_lengths.shape[0]
        device = history_lengths.device

        if self._pos_aligned_side == 'left':
            pos_ids = torch.arange(max_length, dtype=torch.int64, device=device).unsqueeze(0)
        elif self._pos_aligned_side == 'right':
            history_pos_ids = torch.arange(max_length, dtype=torch.int64,
                                           device=device).unsqueeze(0)  # [L] -> [1, L]
            pos_ids = torch.maximum(
                history_lengths.unsqueeze(-1) - history_pos_ids,
                torch.tensor(0, dtype=history_pos_ids.dtype, device=device))  # [B, L]

            pos_ids = pos_ids.to(torch.int64)
        else:
            raise NotImplementedError

        return pos_ids

    def _process_history_embeddings(
            self,
            history_embeddings: torch.Tensor,
            history_ratings: torch.Tensor,
            history_lengths: torch.Tensor
    ) -> torch.Tensor:

        _, N, D = history_embeddings.shape

        history_embeddings = history_embeddings * (D ** 0.5)

        if self._use_action_emb and self._rating_emb is not None:
            if self._rating_emb.embedding_dim == D:
                hist_rating_emb = self._rating_emb(history_ratings)
                history_embeddings = history_embeddings + hist_rating_emb

        if self._use_pos_emb and self._pos_emb.embedding_dim == D:
            pos_ids = self._generate_history_position_ids(history_lengths, N)
            history_embeddings = history_embeddings + self._pos_emb(pos_ids)

        history_embeddings = self._emb_dropout(history_embeddings)

        return history_embeddings

    def _process_user_embeddings(
            self,
            user_feature_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        处理用户特征嵌入以匹配历史嵌入维度。

    参数:
        user_feature_embs: (B, D_user)

    返回:
        (B, n_u, D_history)
    """
        B, D_user = user_feature_embs.shape
        D_history = self._history_embedding_dim
        n_u = self.n_u

        # 存储原始维度
        self._user_original_dim = D_user
        # 应用MLP扩展维度
        user_expanded = self._user_mlp(user_feature_embs)  # (B, n_u * D_history)

        # 重塑为 (B, n_u, D_history)
        user_reshaped = user_expanded.view(B, 1, n_u, D_history)

        return user_reshaped

    def get_embeddings(
            self,
            model_inputs: TensorDict,
            embedder: EmbeddingModule,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从模型输入获取嵌入。"""
        past_embeddings = embedder.get_history_item_embeddings(model_inputs)
        candidate_embeddings = embedder.get_candidate_item_embeddings(model_inputs)
        user_feature_embs = embedder.get_user_embeddings(model_inputs)
        return user_feature_embs, past_embeddings, candidate_embeddings

    def generate_input_sequence(
            self,
            sequence_components: Dict[str, torch.Tensor],
            model_inputs: TensorDict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从组件生成输入序列。"""
        # 提取组件
        history_embeddings = sequence_components.get("history_embeddings")
        candidate_embeddings = sequence_components.get("candidate_embeddings")
        history_lengths = sequence_components.get("history_lengths")
        history_ratings = sequence_components.get("history_ratings")
        candidate_ratings = sequence_components.get("candidate_ratings")
        user_feature_embs = sequence_components.get("user_feature_embeddings")

        # 处理嵌入
        history_embeddings_processed = self._process_history_embeddings(
            history_embeddings, history_ratings, history_lengths
        )
        candidate_embeddings_processed = self._process_candidate_embeddings(candidate_embeddings)
        user_feature_embs_processed = self._process_user_embeddings(user_feature_embs)

        # 应用掩码
        history_ids = model_inputs.get("history_ids", torch.zeros_like(history_embeddings[:, :, 0]))
        candidate_ids = model_inputs.get("candidate_ids", torch.zeros_like(candidate_embeddings[:, :, 0]))
        valid_mask = self.get_preprocessed_masks(history_ids, candidate_ids).unsqueeze(2).float()
        history_embeddings_processed = history_embeddings_processed * valid_mask

        # 构建嵌入字典
        emb_dict = {
            'user': user_feature_embs_processed,
            'user_original_dim': self._user_original_dim,
            'history': history_embeddings_processed,
            'candidate': candidate_embeddings_processed,
            'candidate_original_dim': self._candidate_original_dim,
        }

        new_sequence_lengths = self.n_u + history_lengths
        return emb_dict, new_sequence_lengths

    def forward(
            self,
            model_inputs: TensorDict,
            embedder: EmbeddingModule,
    ) -> ProcessedInput:
        """处理模型输入以生成处理后的输入。"""
        # 获取嵌入
        user_feature_embs, past_embeddings, candidate_embeddings = self.get_embeddings(model_inputs, embedder)

        # 准备序列组件
        sequence_components = {
            "history_embeddings": past_embeddings,
            "candidate_embeddings": candidate_embeddings,
            "history_lengths": model_inputs.get("history_lengths"),
            "history_ratings": model_inputs.get(self._hist_infer_ratings_key),
            "candidate_ratings": model_inputs.get(self._cand_infer_ratings_key),
            "user_feature_embeddings": user_feature_embs,
        }

        # 生成输入序列
        emb_dict, new_sequence_lengths = self.generate_input_sequence(sequence_components, model_inputs)

        # 准备返回值 (类似于原始实现)
        B = model_inputs.get("history_ids").shape[0]
        device = model_inputs.get("history_ids").device
        numrerank_mask = torch.tensor([True, False], device=device).repeat(B)

        return ProcessedInput(
            seq_embeddings=emb_dict,
            past_lengths_after_input_processor=new_sequence_lengths,
            past_embeddings=past_embeddings,
            all_timestamps=model_inputs.get("history_timestamps_column")
        )

    def unpack_sequence(
            self,
            sequence_embeddings: torch.Tensor,
            num_rerank: int,
    ) -> Dict[str, torch.Tensor]:
        """将序列嵌入解包为组件。"""
        elements = {
            "user_embeddings": sequence_embeddings[:, :1, :],
            "item_embeddings": sequence_embeddings[:, 1: -num_rerank, :],
        }

        if torch.onnx.is_in_onnx_export() or num_rerank > 0:
            elements[f"candidate_item_embeddings"] = sequence_embeddings[:, -num_rerank:, :]

        return elements

    def _process_candidate_embeddings(
            self,
            candidate_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        处理候选嵌入，支持可选标记化。

        参数:
            candidate_embeddings: (B, M, D_candidate)

        返回:
            (B, M, n_c, D_history)
        """
        B, M, D = candidate_embeddings.shape
        D_history = self._history_embedding_dim
        n_c = self.n_c

        self._candidate_original_dim = D

        if self.enable_tokenization:
            # 应用标记化
            candidate_expanded = self._tokenize_candidate_features(candidate_embeddings)
        else:
            # 使用原始MLP方法
            candidate_expanded = self._candidate_mlp(candidate_embeddings)

        candidate_expanded = self._emb_dropout(candidate_expanded)

        # 重塑为 (B, M, n_c, D_history)
        candidate_reshaped = candidate_expanded.view(B, M, n_c, D_history)

        return candidate_reshaped

    def _tokenize_candidate_features(
            self,
            candidate_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        对候选特征应用标记化。

        参数:
            candidate_embeddings: 连接的候选嵌入，形状为 (B, M, D_total)

        返回:
            标记化的候选嵌入，形状为 (B, M, n_c, hidden_dim)
        """
        B, M, D = candidate_embeddings.shape

        # 展平: (B, M, D_total) -> (B * M, D_total)
        x_flat = candidate_embeddings.view(B * M, -1)

        gathered = x_flat[:, self.gather_index]

        # 应用掩码
        gathered = gathered * self.mask

        # 融合批矩阵乘法
        out = torch.einsum('btc, tcd -> btd', gathered, self.token_weight) + self.token_bias

        # 重塑: (B * M, n_c, hidden_dim) -> (B, M, n_c, hidden_dim)
        out = out.view(B, M, self.n_c, self.hidden_dim)

        return out

    def process_rerank_embs(
            self,
            rerank_embs: torch.Tensor,
            past_lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = rerank_embs.shape

        rerank_embs = rerank_embs * (D ** 0.5)

        if self._use_pos_emb and self._pos_emb.embedding_dim == D:
            position_embs = self._pos_emb(past_lengths - 1).unsqueeze(1).repeat(1, N, 1)
            rerank_embs = rerank_embs + position_embs

        rerank_embs = self._emb_dropout(rerank_embs)

        rerank_embs = torch.nn.functional.pad(rerank_embs, (0, 0, 0, 1, 0, 0), 'constant', 0.0)

        return rerank_embs

    def get_num_ratings(self):
        return self.num_ratings

    def _reset_tokenization_parameters(self, chunk_dims):
        """使用适当的填充维度处理初始化标记化权重。"""
        for t in range(self.total_tokens):
            actual_c = chunk_dims[t]
            nn.init.kaiming_uniform_(self.token_weight[t, :actual_c, :], a=math.sqrt(5))
            if actual_c < self.c_max:
                nn.init.zeros_(self.token_weight[t, actual_c:, :])

            fan_in = actual_c
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.token_bias[t], -bound, bound)

    def debug_str(self) -> str:
        tokenization_str = ""
        if self.enable_tokenization:
            if self.tokenization_type == "weighted":
                tokenization_str = "_加权标记化"
            else:
                tokenization_str = "_标记化"
        return f"用户物品输入特征预处理器{tokenization_str}_nu{self.n_u}_nc{self.n_c}_d{self._dropout_rate}"
