import abc
import math
import torch

from typing import Dict, Tuple, Optional

from modeling.generic.sequential.base_model import BaseModel
from modeling.generic.utils.constants import Const
from modeling.generic.sequential.negative_sampler import NegativesSamplerModule
from modeling.generic.sequential.loss.loss_mask_modules import LossMaskModule
from modeling.generic.sequential.prediction_modules import PredictionOutput
from modeling.model_registry import ModelRegistry


class LossModule(BaseModel):
    """
    Abstract base class for a loss module used in generative recommender systems.
    """

    _allowed_prediction_names: str
    _allowed_label_names: str

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        self.name = model_cfg.get("name", None)
        if Const.SUB_MODELS in model_cfg:
            self._loss_weight_modules: torch.nn.Module[LossMaskModule] = torch.nn.ModuleList([
                self.init_sub_model(sub_key)
                for sub_key in model_cfg[Const.SUB_MODELS].keys()
            ])
        else:
            self._loss_weight_modules = None

        self._use_loss_clamp = model_cfg[Const.HP].get("use_loss_clamp", False)

    def get_scores(self, predictions: Dict[str, PredictionOutput]) -> torch.Tensor:
        """Extracts prediction scores from the model output, optionally clamped to avoid numerical instability.
        :raises KeyError: If expected prediction is missing.
        """
        pred_output: Optional[PredictionOutput] = predictions.get(self._allowed_prediction_names, None)

        if pred_output is None:
            raise KeyError(
                f"No predictions named '{self._allowed_prediction_names}' found in predictions dict. "
                f"Available keys: {list(predictions.keys())}"
            )

        min_value = Const.EPS if self._use_loss_clamp else -math.inf
        max_value = 1 - Const.EPS if self._use_loss_clamp else math.inf

        return pred_output.get_prediction_score(min_value, max_value)

    def get_logits(self, predictions: Dict[str, PredictionOutput]) -> torch.Tensor:
        """Extracts prediction logits from the model output
        :raises KeyError: If expected prediction is missing.
        """
        pred_output: Optional[PredictionOutput] = predictions.get(self._allowed_prediction_names, None)

        if pred_output is None:
            raise KeyError(
                f"No predictions named '{self._allowed_prediction_names}' found in predictions dict. "
                f"Available keys: {list(predictions.keys())}"
            )

        return pred_output.logits

    def get_masks(self, model_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes loss masks using the configured loss weighting modules."""

        mask_dct = {}
        if isinstance(self._loss_weight_modules, torch.nn.ModuleList):
            init_mask = None
            for loss_weight in self._loss_weight_modules:
                name = loss_weight.name
                mask = loss_weight(model_inputs)
                mask_dct[name] = mask
                init_mask = mask if init_mask is None else init_mask * mask

        else:
            # No weighting module provided; default to all-ones mask
            score_shape: torch.Tensor = model_inputs.get(self._allowed_label_names, None)  # Shape: (B, S)
            init_mask = torch.ones_like(score_shape)

        return init_mask, mask_dct

    @abc.abstractmethod
    def forward(
            self,
            past_embeddings: torch.Tensor,
            encoded_embeddings: Dict[str, torch.Tensor],
            predictions: Dict[str, PredictionOutput],
            model_inputs: Dict[str, torch.Tensor],
            negative_sampler: NegativesSamplerModule,
            offsets: Optional[torch.Tensor] = None,
            candi_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the loss.

        :param past_embeddings: Past interaction embeddings [B, T, D]
        :param encoded_embeddings: Dictionary containing encoded representations from the model [B, T, D]
        :param predictions: Model output containing predictions
        :param model_inputs: Input features including labels and masks
        :param negative_sampler: Negative sampling module
        :param offsets: Offsets for jagged tensor conversion (optional)
        :param candi_length: Maximum length for jagged tensor conversion (optional)
        :return: A scalar loss tensor.
        """
        pass


class LossAggregatorModule(BaseModel):
    """
    Base class for aggregating multiple task-specific losses.

    Submodules must subclass `LossModule`. Uses `.forward()` to get individual losses, and `.aggregate()` to compute a
    final scalar loss.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        sub_models = model_cfg[Const.SUB_MODELS]
        self.loss_modules: torch.nn.ModuleList[LossModule] = torch.nn.ModuleList(
            [self.init_sub_model(sub_model) for sub_model in sub_models]
        )

    def aggregate(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for loss_name, loss_value in losses.items():
            total_loss += self.task_weights.get(loss_name, 1.) * loss_value
        return total_loss  # Shape: scalar

    def forward(
            self,
            past_embeddings: torch.Tensor,
            encoded_embeddings: Dict[str, torch.Tensor],
            predictions: Dict[str, PredictionOutput],
            model_inputs: Dict[str, torch.Tensor],
            negative_sampler: NegativesSamplerModule,
    ) -> Dict[str, torch.Tensor]:
        losses = {
            loss_module.name: loss_module(
                past_embeddings=past_embeddings,
                encoded_embeddings=encoded_embeddings,
                predictions=predictions,
                model_inputs=model_inputs,
                negative_sampler=negative_sampler,
            )
            for loss_module in self.loss_modules
        }

        # Add the aggregated loss for backpropagation
        losses["loss"] = self.aggregate(losses)

        return losses


@ModelRegistry.register(
    multi_sel_multi_subs=[{"LossModule",
                           "BinaryCrossEntropyLossForRerankScore",
                           "BinaryCrossEntropyLoss",
                           "FocalLossForRerankScore",
                           "AsymmetricBinaryCrossEntropyLossForRerankScore",
                           "CrossEntropyLossForNextActionPred",
                           "SampledSoftmaxLossForNextItemPred",
                           }]
)
class SumLossAggregator(LossAggregatorModule):
    """
    Aggregates multiple losses using a weighted sum.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.task_weights = model_cfg.get(Const.HP, {}).get("loss_weight", {})


@ModelRegistry.register(
    multi_sel_multi_subs=[{"LossModule",
                           "BinaryCrossEntropyLossForRerankScore",
                           "BinaryCrossEntropyLoss",
                           "FocalLossForRerankScore",
                           "AsymmetricBinaryCrossEntropyLossForRerankScore",
                           "CrossEntropyLossForNextActionPred",
                           "SampledSoftmaxLossForNextItemPred",
                           }]
)
class AverageLossAggregator(LossAggregatorModule):
    """
    Aggregates multiple losses using a weighted average.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

        task_weights = model_cfg.get(Const.HP, {}).get("loss_weight", {})
        norm_weight = max(sum(task_weights.values()) or 1.0, 1.0)
        self.task_weights = {task: weight / norm_weight for task, weight in task_weights.items()}
