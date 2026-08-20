"""
# Attention Mask Modules

This file defines a modular system for computing the attention masks.
The model itself expects a single attention mask, but complex masks can be built from simpler, composable modules.
Each attention mask module is a subclass of `AttentionMaskModule`. These can be used independently or composed using
the `AttentionMaskIntersection` module, which combines multiple masks via element-wise multiplication (logical AND).
This composition design supports future expansion (e.g., unions or negations).

## Examples

### Simple Causal Mask

A single attention module can be used directly:

```yaml
AttentionMaskModule:
  name: CausalAttentionMask
````

### Composite Mask via Intersection

Multiple attention modules can be combined using `AttentionMaskIntersection`, which wraps and intersects their outputs:

```yaml
AttentionMaskModule:
  name: AttentionMaskIntersection
  sub_models:
    CausalAttentionMask:
      name: CausalAttentionMask
    TimeWindowAttentionMask:
      name: TimeWindowAttentionMask
```

## Notation
- B: batch size
- N: max sequence length, currently defined for each item/action pairs
- S: sequence length, currently defined as S = 2*N+2[+C-1]
- C: number of candidate items
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.utils.constants import Const, FeatConst
from modeling.model_registry import ModelRegistry


class AttentionMaskModule(BaseModel):
    """
    Abstract base class for attention mask modules.

    Defines the interface expected by all attention masking strategies used in generative recommendation models.
    Subclasses must implement the `forward` method, which returns a mask tensor of shape (B or 1, S, S).
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_candidates) -> torch.Tensor:
        pass


@ModelRegistry.register(multi_sel_multi_subs=[
    {"CausalAttentionMask", "TimeWindowAttentionMask", "TimeBucketAttentionMask"}
])
class AttentionMaskIntersection(AttentionMaskModule):
    """
    Composite attention mask module that intersects multiple attention masks.

    Each sub-model must be a subclass of `AttentionMaskModule`. Their outputs are intersected via element-wise
    multiplication to enforce joint constraints. This supports constructing more complex masking logic.

    Typically used as a top-level module to wrap and compose multiple independent attention rules.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.attention_mask_modules: torch.nn.ModuleList[AttentionMaskModule] = torch.nn.ModuleList(
            [self.init_sub_model(sub_key) for sub_key in model_cfg[Const.SUB_MODELS].keys()]
        )

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_candidates) -> torch.Tensor:
        init_mask = None
        for attn_module in self.attention_mask_modules:
            mask = attn_module(model_inputs=model_inputs, max_seq_len=max_seq_len, num_candidates=num_candidates)
            if init_mask is None:
                init_mask = mask
            else:
                init_mask = init_mask * mask
        attn_mask = init_mask.detach().clone()  # Shape: (B, S, S)

        return attn_mask


@ModelRegistry.register()
class CausalAttentionMask(AttentionMaskModule):
    """
    Implements a lower-triangular causal attention mask.

    Ensures that each token can only attend to previous positions in the sequence (autoregressive behavior).
    Handles variable past lengths and adapts accordingly when computing candidates.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feature_conf = common_hp.get("feature_conf")
        self._infer_timestamps_key: str = feature_conf.get("infer_timestamps_key", "timestamps")
        self._infer_items_key: str = feature_conf.get('infer_items_key', 'item_id')

    @staticmethod
    def init_mask_for_export(seq_len: int, num_candidates: int, device: torch.device) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = seq_len * 2 + 2 + num_candidates
        _pos_indices = torch.arange(max_len).repeat(max_len).view(max_len, max_len).to(device)  # Shape: (S, S)
        _base_mask = (_pos_indices.t() > _pos_indices).float()  # Shape: (S, S)
        _identity = (_pos_indices.t() == _pos_indices).float()  # Shape: (S, S)
        return _pos_indices, _identity, _base_mask

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_candidates) -> torch.Tensor:
        """
        Returns a (B, S, S) causal attention mask (lower triangular).
        """
        device = model_inputs[self._infer_timestamps_key].device
        max_len = max_seq_len * 2 + num_candidates + 2  #
        if num_candidates == 0:
            indices = torch.arange(max_len).to(device)  # Shape: (S,)
            t = indices.expand(max_len, max_len)  # Shape: (S, S)
            attn_mask = (t.t() >= indices).unsqueeze(0)  # Shape: (1, S, S)
        else:
            past_lengths = model_inputs[f'{FeatConst.HIST_PFX}_lengths']  # N
            past_lengths = 1 + past_lengths * 2  # S = 2*N+1, Shape: (B,)
            _past_lengths = past_lengths.unsqueeze(-1).unsqueeze(-1)  # Shape: (B, 1, 1)
            _pos_indices, _identity, _base_mask = self.init_mask_for_export(
                seq_len=max_seq_len, num_candidates=num_candidates, device=device
            )  # Shapes: (S, S), (S, S), (S, S)
            seq_mask = (_pos_indices < _past_lengths).int()  # Shape: (B, S, S)
            attn_mask = (seq_mask * _base_mask + _identity)
        return attn_mask  # Shape: (B or 1, S, S)


@ModelRegistry.register()
class TimeWindowAttentionMask(AttentionMaskModule):
    """
    Computes attention mask based on a time-difference threshold.

    Allows attention only between positions where timestamps differ by less than a configured threshold (in seconds).
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp.get("feature_conf")
        self._time_threshold: int = model_cfg.get(Const.HP, {}).get("timestamp_mask_threshold", 86400)
        self._infer_items_key: str = feat_conf.get("infer_items_key", "item_id")
        self._infer_timestamps_key: str = feat_conf.get("infer_timestamps_key", "timestamps")

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len: int, num_candidates: int) -> torch.Tensor:
        device = model_inputs[self._infer_timestamps_key].device
        bs = model_inputs.get(self._infer_items_key).shape[0]
        all_timestamps = model_inputs[self._infer_timestamps_key].detach()  # Shape: (B, S)

        time_differences = all_timestamps.unsqueeze(2) - all_timestamps.unsqueeze(1)  # Shape: (B, S, S)
        time_threshold_mask = (time_differences <= self._time_threshold).float()

        _pos_indices = torch.arange(max_seq_len).repeat(max_seq_len).view(max_seq_len, max_seq_len).to(device)
        _identity = torch.eq(_pos_indices.t(), _pos_indices).float()  # Shape: (N, N)
        time_threshold_mask = time_threshold_mask - _identity  # Shape: (B, S, S)
        time_threshold_mask = time_threshold_mask.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, 2, 2)  # Shape: (B,S,S,2,2)
        time_threshold_mask = time_threshold_mask.reshape(bs, max_seq_len * 2, max_seq_len * 2)  # Shape: (B, S, S)

        atten_mask_time = 1.0 - time_threshold_mask  # Shape: (B, S, S)
        atten_mask_time = F.pad(atten_mask_time, (1, 1 + num_candidates, 1, 1 + num_candidates), 'constant', 1.0)
        return atten_mask_time


@ModelRegistry.register()
class TimeBucketAttentionMask(AttentionMaskModule):
    """
    Computes attention mask by grouping timestamps into discrete time buckets.

    Positions that fall within the same bucket are allowed to attend to each other.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp.get("feature_conf")
        model_conf = common_hp.get("model_conf")
        self._time_offset: float = model_conf.get("time_offset", 0)  # Default: Seconds past midnight to split buckets
        self._time_bucket_interval: int = model_conf.get("time_bucket_interval", 86400)
        self._infer_items_key: str = feat_conf.get("infer_items_key", "item_id")
        self._infer_timestamps_key: str = feat_conf.get("infer_timestamps_key", "timestamps")

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len: int, num_candidates: int) -> torch.Tensor:
        device = model_inputs[self._infer_timestamps_key].device
        bs = model_inputs[self._infer_items_key].shape[0]
        all_timestamps = model_inputs[self._infer_timestamps_key].detach()  # Shape: (B, S)

        bucket_idx = torch.floor((all_timestamps - self._time_offset) / self._time_bucket_interval).long()  # Sh: (B,S)
        same_bucket = bucket_idx.unsqueeze(2).eq(bucket_idx.unsqueeze(1)).float()  # Shape: (B, S, S)

        _pos_indices = torch.arange(max_seq_len, device=device).repeat(max_seq_len).view(max_seq_len, max_seq_len)
        _identity = torch.eq(_pos_indices.t(), _pos_indices).float()  # Shape: (N, N)
        same_bucket = same_bucket - _identity  # Shape: (B, S, S)
        same_bucket = same_bucket.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, 2, 2)  # Shape: (B, S, S, 2, 2)
        same_bucket = same_bucket.reshape(bs, max_seq_len * 2, max_seq_len * 2)  # Shape: (B, S, S)

        attn_mask_bucket = 1.0 - same_bucket  # Shape: (B, S, S)
        attn_mask_bucket = F.pad(
            attn_mask_bucket,
            (1, 1 + num_candidates, 1, 1 + num_candidates),
            "constant",
            1.0
        )  # S:(B,S,S)
        return attn_mask_bucket


@ModelRegistry.register()
class MTAttentionMask(AttentionMaskModule):
    """
    Meituan方案mask，history部分causal，candidate互相不可见，仅可看见history时间位于其前面的item
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp.get("feature_conf")
        self.hist_dates_key = feat_conf.get(f"{FeatConst.HIST_PFX}_date_column", FeatConst.DFLT_HIST_DATE_KEY)
        self.cand_dates_key = feat_conf.get(f"{FeatConst.CAND_PFX}_date_column", FeatConst.DFLT_CAND_DATE_KEY)

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len: int, num_candidates: int) -> torch.Tensor:
        """
            Returns a bs x (h+p+1) x (h+p+1)  attn mask
        """
        hist_ts = model_inputs.get(self.hist_dates_key)
        cand_ts = model_inputs.get(self.cand_dates_key)
        bs, hist_len = hist_ts.shape
        _, cand_len = cand_ts.shape
        max_len = hist_len + cand_len + 1
        device = hist_ts.device

        # final_mask example (1=True, 0=False), h4 and p3 are paddings
        #
        #       |  u | h0 | h1 | h2 | h3 | h4 | p0 | p1 | p2 | p3 |
        #       --------------------------------------------------
        #    u  |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h0  |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h1  |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h2  |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h3  |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |
        #   h4  |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   p0  |  1 |  1 |  1 |  1 |  0 |  0 |  1 |  0 |  0 |  0 |
        #   p1  |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  1 |  0 |  0 |
        #   p2  |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  1 |  0 |
        #   p3  |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |

        user_col = torch.zeros(bs, 1, device=device, dtype=hist_ts.dtype)
        seq_ts = torch.cat([user_col, hist_ts, cand_ts], dim=1)
        pad_len = max_len - seq_ts.shape[1]
        ts_pad = F.pad(seq_ts, (0, pad_len), mode='constant', value=0)

        idx = torch.arange(max_len, device=device)
        row_idx = idx.view(1, max_len, 1)
        col_idx = idx.view(1, 1, max_len)
        ts_i = ts_pad.unsqueeze(2)
        ts_j = ts_pad.unsqueeze(1)
        valid_row = (ts_i > 0) | (row_idx == 0)
        valid_col = (ts_j > 0) | (col_idx == 0)

        # 4 regions
        user_mask = (col_idx == 0)
        hist_region = (col_idx >= 1) & (col_idx < 1 + hist_len) & (row_idx < hist_len + 1)
        hist_mask = hist_region & (row_idx >= col_idx)
        cand_region = (row_idx >= 1 + hist_len) & (col_idx >= 1) & (col_idx < 1 + hist_len)
        cand_hist_mask = cand_region & (ts_i > ts_j)
        pred_diag_mask = (row_idx == col_idx) & (row_idx >= 1 + hist_len)

        # merge
        mask = ((user_mask | hist_mask | cand_hist_mask | pred_diag_mask) & valid_row & valid_col).float()

        return mask


@ModelRegistry.register()
class MTQueryAttentionMask(BaseModel):
    """
    query作为单独序列的mask
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp.get("feature_conf")
        self.hist_dates_key = feat_conf.get("history_date_column", FeatConst.DFLT_HIST_DATE_KEY)
        self.cand_dates_key = feat_conf.get("candidate_date_column", FeatConst.DFLT_CAND_DATE_KEY)
        self.query_dates_key = feat_conf.get("query_date_column", FeatConst.DFLT_CAND_DATE_KEY)
        self.cand_timestamps_key = feat_conf.get("candidate_timestamps_column", FeatConst.DFLT_QUERY_TS_KEY)
        self.query_timestamps_key = feat_conf.get("query_timestamps_column", FeatConst.DFLT_QUERY_TS_KEY)

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len: int, num_rerank: int) -> torch.Tensor:
        """
            Returns a bs x (h+p+1) x (h+p+1)  attn mask
        """
        hist_dt = model_inputs[self.hist_dates_key]
        cand_dt = model_inputs[self.cand_dates_key]
        cand_ts = model_inputs[self.cand_timestamps_key]
        query_dt = model_inputs[self.query_dates_key]
        query_ts = model_inputs[self.query_timestamps_key]
        bs, hist_len = hist_dt.shape
        _, cand_len = cand_dt.shape
        _, q_len = query_ts.shape

        max_len = hist_len + q_len + cand_len + 1  # query训练时与cand等长,推理长度为1
        device = hist_dt.device

        user_pos = 0
        q_start = hist_len + 1
        cand_start = q_start + q_len

        # final_mask 示意图 (1=True, 0=False), 无 padding
        #
        #       | h0 | h1 | h2 | h3 | h4 |  u | q0 | q1 | q2 | p0 | p1 | p2 |
        #       -------------------------------------------------------------
        #   h0  |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h1  |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h2  |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h3  |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   h4  |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   u   |  1 |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |
        #   q0  |  1 |  0 |  0 |  0 |  0 |  1 |  1 |  0 |  0 |  0 |  0 |  0 |
        #   q1  |  1 |  1 |  1 |  0 |  0 |  1 |  1 |  1 |  0 |  0 |  0 |  0 |
        #   q2  |  1 |  1 |  1 |  1 |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  0 |
        #   p0  |  1 |  0 |  0 |  0 |  0 |  1 |  1 |  0 |  0 |  1 |  0 |  0 |
        #   p1  |  1 |  1 |  1 |  0 |  0 |  1 |  0 |  1 |  0 |  0 |  1 |  0 |
        #   p2  |  1 |  1 |  1 |  1 |  1 |  1 |  0 |  0 |  1 |  0 |  0 |  1 |

        user_col = torch.zeros(bs, 1, device=device, dtype=hist_dt.dtype)
        seq_ts = torch.cat([user_col, hist_dt, query_dt, cand_dt], dim=1)  # h,u,q,p

        idx = torch.arange(max_len, device=device)
        row_idx = idx.view(1, max_len, 1)
        col_idx = idx.view(1, 1, max_len)
        ts_i = seq_ts.unsqueeze(2)
        ts_j = seq_ts.unsqueeze(1)

        row_is_hist = (row_idx < hist_len + 1) & (row_idx > 0)
        col_is_hist = (col_idx < hist_len + 1) & (col_idx > 0)

        row_is_user = (row_idx == user_pos)
        col_is_user = (col_idx == user_pos)

        row_is_query = (row_idx >= q_start) & (row_idx < q_start + q_len)
        col_is_query = (col_idx >= q_start) & (col_idx < q_start + q_len)

        row_is_cand = (row_idx >= cand_start) & (row_idx < cand_start + cand_len)

        valid_row = (ts_i > 0) | row_is_user
        valid_col = (ts_j > 0) | col_is_user

        # user列
        user_mask = col_is_user

        # history 下三角
        hist_region = row_is_hist & col_is_hist
        hist_mask = hist_region & (row_idx >= col_idx)

        # 新增query区域
        q_hist_region = row_is_query & col_is_hist
        q_hist_mask = q_hist_region & (ts_i > ts_j)

        # cand考虑query
        cand_hist_region = row_is_cand & col_is_hist
        cand_hist_mask = cand_hist_region & (ts_i > ts_j)

        # 对角线
        diag_mask = (row_idx == col_idx)
        diag_query_mask = diag_mask & row_is_query
        diag_cand_mask = diag_mask & row_is_cand

        # cand看见query
        row_rel_cand_idx = (row_idx - cand_start).clamp(0, cand_len - 1)  # [1, cand_len, 1]
        row_rel_cand_idx = row_rel_cand_idx.expand(bs, -1, -1).squeeze(2)  # [bs, cand_len]
        row_cq_ts_full = cand_ts.gather(1, row_rel_cand_idx).unsqueeze(2)  # [bs, cand_len, 1]

        col_rel_query_idx = (col_idx - q_start).clamp(0, q_len - 1)  # [1, 1, q_len]
        col_rel_query_idx = col_rel_query_idx.expand(bs, -1, -1).squeeze(1)  # [bs, q_len]
        col_cq_ts_full = query_ts.gather(1, col_rel_query_idx).unsqueeze(1)  # [bs, 1, q_len]

        cand_query_region = row_is_cand & col_is_query
        cand_query_mask = cand_query_region & (row_cq_ts_full == col_cq_ts_full)

        # 合并
        mask_bool = (
                user_mask
                | hist_mask
                | q_hist_mask
                | cand_hist_mask
                | diag_query_mask
                | diag_cand_mask
                | cand_query_mask
        )
        mask = (mask_bool & valid_row & valid_col).float()

        return mask

