import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.generic.sequential.attn_mask_modules import AttentionMaskModule
from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.sequential.embeddings.embedding_modules import EmbeddingModule
from modeling.generic.sequential.input_features_preprocessors import InputProcessorModule, ProcessedInput
from modeling.generic.sequential.loss.loss_utils import LossAggregatorModule
from modeling.generic.sequential.negative_sampler import NegativesSamplerModule
from modeling.generic.sequential.output_postprocessors import OutputPostprocessorModule
from modeling.generic.sequential.prediction_modules import PredictionAggregatorModule, PredictionOutput
from modeling.generic.sequential.transformers import SequentialModule, TransformerCacheStates, FPNSequentialModule
from modeling.generic.sequential.dense_tokenizer import DenseTokenizer
from modeling.generic.utils.constants import Const, FeatConst
from modeling.model_registry import ModelRegistry


@ModelRegistry.register(
    req_subs={
        "EmbeddingModule",
        "InputProcessorModule",
        "SequentialModule",
        "AttentionMaskModule",
        "PredictionAggregatorModule",
        "OutputPostprocessorModule",
        "LossAggregatorModule",
    },
    opt_subs={"NegativesSamplerModule"},
    req_hp=True,
)
class GenerativeRecommender(BaseModel):
    def __init__(
            self,
            model_cfg: Dict,
            common_hp: Dict,
            model_cls_dict: Dict
    ):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self._verbose = model_cfg[Const.HP].get("verbose", True)
        model_conf = common_hp["model_conf"]
        feat_conf = common_hp["feature_conf"]

        self.use_enhanced_interest_embeddings = feat_conf.get("use_enhanced_interest_embeddings", False)
        self.use_multi_action = model_conf.get("use_multi_action", False)
        self.use_action_conditioning = model_conf.get("use_action_conditioning", False)

        # 获取温度系数字典
        self.action_temps = model_conf.get("action_temps", {})
        # 获取行为时间戳特征
        self.action_timestamps_keys = model_conf.get("action_timestamps_keys", {})

        self._max_sequence_length: Optional[int] = model_conf.get("max_sequence_length")
        self.infer_items_key = feat_conf.get("infer_items_key", "item_id")
        self.embedding_module: EmbeddingModule = self.init_sub_model("EmbeddingModule")
        self.input_processor_module: InputProcessorModule = self.init_sub_model(
            "InputProcessorModule"
        )
        self.sequential_module: SequentialModule | FPNSequentialModule = self.init_sub_model("SequentialModule")

        # rankmixer
        if 'RankMixerModule' in self.model_cfg['sub_models']:
            self.rankmixer_module = self.init_sub_model('RankMixerModule')

        self.negative_sampler: Optional[NegativesSamplerModule] = (
            None
            if "NegativesSamplerModule" not in model_cfg[Const.SUB_MODELS]
            else self.init_sub_model("NegativesSamplerModule")
        )
        if self.negative_sampler is not None:
            self.negative_sampler.load_embedding_module(self.embedding_module)

        self.loss_aggregator: LossAggregatorModule = self.init_sub_model("LossAggregatorModule")

        self.attention_mask_module: AttentionMaskModule = self.init_sub_model("AttentionMaskModule")
        self.prediction_modules: PredictionAggregatorModule = self.init_sub_model("PredictionAggregatorModule")
        self.output_processor_module: OutputPostprocessorModule = self.init_sub_model("OutputPostprocessorModule")
        self.reset_params()

        self.use_dense_tokenization = model_conf.get("use_dense_tokenization", False)
        if self.use_dense_tokenization:
            logging.info("Using dense model...")
            self.history_action_type = feat_conf.get("dense_hist_type", "history_action_type")
            self.num_actions = feat_conf.get("num_actions", 8)
            self.dense_model = DenseTokenizer(self.item_embedding_dim_0, self.num_actions)

    def reset_params(self):
        for name, params in self.named_parameters():
            if ("sequential_module" in name) or ("embedding_module" in name):
                if self._verbose:
                    logging.info("Skipping init for %s", name)
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    logging.info("Initialize %s as xavier normal: %s params", name, params.data.shape[0])
            except Exception as e:
                if self._verbose:
                    logging.info("Failed to initialize %s: %s params", name, params.data.shape[0])
                    logging.info("Error %s", e)

    def generate_user_embeddings(
            self,
            tensor_dict: dict,
            return_cache_states: bool = False,
            num_rerank: int = 0,
            action_bias: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        综合序列信息，生成用户 embedding.
        [B, N] -> [B, N, D].
        :param tensor_dict: 包含用户行为序列信息的字典，键包括：
            - 'past_lengths' (torch.Tensor): 用户历史行为的长度，形状为 [B]
            - 'all_timestamps' (torch.Tensor): 用户行为的时间戳，形状为 [B, N]
            - 'seq_embeddings' (torch.Tensor): 用户行为序列的嵌入表示，形状为 [B, N, D]
            - 'attn_mask' (torch.Tensor): 注意力掩码，用于屏蔽无效位置，形状为 [B, N]
            - 'cache_states' (dict): 缓存状态，用于加速推理
            - 'delta_x_offsets' (tuple): 行为序列的偏移量，包含两个张量
        :param return_cache_states: 是否返回缓存状态，默认为 False
        :param num_rerank: 重新排序的数量，默认为 0
        :param action_bias: 行为类型偏置，默认为None
        """
        past_lengths = tensor_dict.get('past_lengths')
        all_timestamps = tensor_dict.get('all_timestamps')
        seq_embeddings = tensor_dict.get('seq_embeddings')
        attn_mask = tensor_dict.get('attn_mask')
        cache_states = tensor_dict.get('cache_states')
        delta_x_offsets = tensor_dict.get('delta_x_offsets', (torch.tensor([]), torch.tensor([])))

        item_embeddings, _ = self.sequential_module(
            x=seq_embeddings,
            x_offsets=torch.cat(
                (
                    torch.full((1,), 0, dtype=past_lengths.dtype).to(past_lengths.device),
                    torch.cumsum(past_lengths, dim=0),
                ),
                dim=0,
            ),
            all_timestamps=all_timestamps,
            invalid_attn_mask=attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            delta_x_offsets=delta_x_offsets,
            cache_states=cache_states,
            return_cache_states=return_cache_states,
            action_bias=action_bias,
        )

        # NOTE(aar): Behavior for output_processor_module has changed. It is now applied to the sequence
        #  without concat_user_embeddings since this behavior was moved to the PredictionModules
        # NOTE(aar): OutputPostprocessorModule is deprecated. Input normalization at the PredictionModule
        #  should be done instead
        item_embeddings = self.output_processor_module(item_embeddings)

        return self.input_processor_module.unpack_sequence(
            sequence_embeddings=item_embeddings,
            num_rerank=num_rerank
        )

    def process_single_act_seq(self,
                               action: str,
                               num_rerank: int,
                               candidate_embeddings: torch.Tensor,
                               user_feature_embs: torch.Tensor,
                               model_input: dict
                               ) -> Dict[str, torch.Tensor]:
        """
        处理单个行为序列
        :param action: 行为类型
        :param num_rerank: 候选数量
        :param candidate_embeddings: 候选嵌入
        :param user_feature_embs: 用户嵌入
        :param model_input: 模型输入
        :return:
        """
        single_act_seq_input = {}
        # 获取行为对应的时间戳
        act_ts_key = self.action_timestamps_keys[action]
        act_ts = model_input.get(act_ts_key)

        batch_size = candidate_embeddings.shape[0]

        for feat_name in self.feature_groups[FeatConst.HIST_PFX][action]['features']:
            single_act_seq_input[feat_name] = model_input[feat_name]
        for feat_name in self.feature_groups[FeatConst.CAND_PFX]['features']:
            single_act_seq_input[feat_name] = model_input[feat_name]
        single_act_seq_input[self.cand_ratings_key] = model_input[self.cand_ratings_key]
        single_act_seq_input["action_dates_key"] = model_input[act_ts_key]
        single_act_seq_input["cand_dates_key"] = model_input[self.cand_ts_key]

        attn_mask = self.attention_mask_module(single_act_seq_input, self._max_sequence_length, num_rerank)

        past_embeddings = self.embedding_module.get_ui_embeddings(
            group_name=FeatConst.HIST_PFX,
            input_features=single_act_seq_input,
            action_type=action
        )  # [batch_size, hist_length, emb_dim]

        if self.use_dense_tokenization:
            act_type = model_input.get(self.history_action_type)
            cand_ts = model_input.get(self.cand_ts_key)


            past_embeddings = self.dense_model(
                past_embeddings, act_type, act_ts, cand_ts
            )

        # 使用SRN结构交叉用户历史序列和item特征
        if self.use_enhanced_interest_embeddings:
            candidate_emb_for_srn = candidate_embeddings.clone()  # [batch_size, num_rerank, emb_dim]

            max_seq_len = past_embeddings.shape[1]
            batch_size = past_embeddings.shape[0]

            sequence_mask = torch.arange(max_seq_len, device=past_embeddings.device)[None, :] < self.history_lengths
            sequence_mask = sequence_mask.expand(batch_size, -1)  # 扩展到批次大小

            try:
                enhanced_past_embeddings = self.srn_module(
                    [candidate_emb_for_srn, past_embeddings],
                    mask=sequence_mask
                )
                past_embeddings = past_embeddings + enhanced_past_embeddings
            except Exception as e:
                logging.info("SRN module error: %s", e)
                raise e

        past_lengths_after_input_processor, user_embeddings, _ = self.input_propcessor_module(
            history_embeddings=past_embeddings,
            candidate_embeddings=candidate_embeddings,
            history_lengths=self.history_lengths.expand(batch_size),
            history_ids=model_input.get(self.hist_items_key),
            candidate_ids=model_input.get(self.cand_items_key),
            user_feature_embs=user_feature_embs,
            history_ratings=self.action_mapping.get(action),
            candidate_ratings=model_input.get(self.cand_ratings_key),
            hist_times=act_ts,
        )

        # 每个行为序列用自己的date_diff_seq
        encoded_embeddings = self.generate_user_embeddings(   # STU
            tensor_dict=dict(
                past_lengths=past_lengths_after_input_processor,
                all_timestamps=act_ts,
                seq_embeddings=user_embeddings,
                attn_mask=attn_mask,
                cache_states=TransformerCacheStates(),
            ),
            num_rerank=num_rerank)

        return encoded_embeddings

    def process_multi_action_history_seq(self,
                                         num_rerank: int,
                                         candidate_embeddings: torch.Tensor,
                                         user_feature_embs: torch.Tensor,
                                         model_input: dict
                                         ) -> torch.Tensor:
        """
        综合处理所有行为序列
        :param num_rerank: 候选数量
        :param candidate_embeddings: 候选嵌入
        :param user_feature_embs: 用户嵌入
        :param model_input: 模型输入
        :return:
        """
        multi_action_history_seq_list = []
        action_types = self.action_types
        action_temperatures = self.action_temps

        for action in action_types:
            action_embedding = self.process_single_act_seq(action, num_rerank,
                                                           candidate_embeddings, user_feature_embs, model_input)
            action_embedding = action_embedding / action_temperatures.get(action)
            multi_action_history_seq_list.append(action_embedding)
            # 每个行为序列的emb都是[bs, 1, item_embedding_dim * 2] -> 开启了use_user_embeddings_for_rerank
            # 不开启的话是[bs, 1, item_embedding_dim]
        encoded_embeddings = torch.cat(multi_action_history_seq_list, dim=-1)
        return encoded_embeddings

    def forward(self, model_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_rerank = model_input.get(f"{FeatConst.CAND_PFX}_{self.infer_items_key}", torch.tensor([[]])).shape[1]

        attn_mask = self.attention_mask_module(model_input, self._max_sequence_length, num_rerank) # MTAttentionMask

        processed_input: ProcessedInput = self.input_processor_module(model_input, self.embedding_module) # FusedItemActionInputProcessor LocalEmbeddingModuleWithSideInfo

        if self.use_multi_action:
            encoded_embeddings = self.process_multi_action_history_seq(
                num_rerank=num_rerank,
                candidate_embeddings=processed_input.seq_embeddings,
                user_feature_embs=processed_input.seq_embeddings,
                model_input=model_input
            )
        else:
            action_bias = None
            if self.use_action_conditioning:
                if (hasattr(self.input_processor_module, 'use_action_conditioning') and
                        self.input_processor_module.use_action_conditioning and
                        hasattr(self.input_processor_module, 'action_conditioning')):
                    # 获取历史行为类型和候选行为类型
                    history_action_types = model_input.get("history_action_type")  # [B, N]
                    candidate_action_types = model_input.get("candidate_action_type")  # [B, M]
                    user_embeddings = processed_input.seq_embeddings
                    # 拼接历史和候选行为序列
                    combined_action_types = torch.cat([history_action_types, candidate_action_types], dim=1)

                    # 生成attention bias
                    seq_length = user_embeddings.shape[1]
                    action_bias = self.input_processor_module.action_conditioning.generate_attention_bias(
                        action_types=combined_action_types,
                        seq_length=seq_length,
                        has_user_token=self.concat_user_embeddings
                        # user token不需要action_bias
                    )
            encoded_embeddings = self.generate_user_embeddings(
                tensor_dict=dict(
                    past_lengths=processed_input.past_lengths_after_input_processor,
                    all_timestamps=processed_input.all_timestamps,
                    seq_embeddings=processed_input.seq_embeddings,
                    attn_mask=attn_mask,
                    cache_states=TransformerCacheStates(),
                ),
                num_rerank=num_rerank,
                action_bias=action_bias,
            )

        if self.rankmixer_module:
            # apply rankmixer
            encoded_embeddings, l1_loss = self.rankmixer_module(encoded_embeddings)

        predictions: Dict[str, PredictionOutput] = self.prediction_modules(encoded_embeddings=encoded_embeddings,
                                                                           model_input=model_input)

        if not self.training or torch.onnx.is_in_onnx_export():
            return {
                key: pred_output.predictions for key, pred_output in predictions.items()
            }
        else:
            losses = self.loss_aggregator(
                past_embeddings=processed_input.past_embeddings,
                encoded_embeddings=encoded_embeddings,
                predictions=predictions,
                model_inputs=model_input,
                negative_sampler=self.negative_sampler,
            )

            if self.rankmixer_module:
                losses += l1_loss

            return losses
