"""
Pretraining:
    L_pretrain = L_contrast + lambda_p * L_pyro
Fine-tuning:
    L_cls = weighted cross entropy over ordinal risk levels {0, 1, 2}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Pixel-wise weighted cross entropy for ordinal wildfire risk labels."""

    def __init__(self, class_weights: torch.Tensor | list[float] | None = None, ignore_index: int = -100):
        super().__init__()
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"logits must be [B,C,H,W], got {list(logits.shape)}")
        if targets.ndim == 4 and targets.size(1) == 1:
            targets = targets[:, 0]
        if targets.ndim != 3:
            raise ValueError(f"targets must be [B,H,W] or [B,1,H,W], got {list(targets.shape)}")
        return F.cross_entropy(
            logits,
            targets.long(),
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )


class FireTrendLoss(nn.Module):
    """
    Combined stage-aware FireTrend loss.

    stage="pretrain":  L_pretrain
    stage="finetune":  L_cls
    stage="joint":     L_cls + L_pretrain
    """

    def __init__(
        self,
        class_weights: torch.Tensor | list[float] | None = None,
        lambda_pyro: float = 1.0,
        lambda_cls: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.lambda_pyro = float(lambda_pyro)
        self.lambda_cls = float(lambda_cls)
        self.register_buffer("_device_ref", torch.empty(0))
        self.cls_loss = WeightedCrossEntropyLoss(class_weights=class_weights, ignore_index=ignore_index)

    def set_class_weights(self, class_weights: torch.Tensor | list[float] | None) -> None:
        if class_weights is None:
            self.cls_loss.class_weights = None
            return
        weights = torch.as_tensor(class_weights, dtype=torch.float32, device=self._device_ref.device)
        self.cls_loss.class_weights = weights

    def forward(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        y_true: torch.Tensor | None = None,
        stage: str = "joint",
    ) -> dict[str, torch.Tensor]:
        stage = stage.lower()
        if stage not in {"pretrain", "finetune", "joint"}:
            raise ValueError(f"Unsupported training stage: {stage}")

        logits = outputs.get("logits")
        L_contrast = outputs.get("L_contrast")
        L_pyro = outputs.get("L_pyro")
        if not isinstance(L_contrast, torch.Tensor):
            reference = logits if isinstance(logits, torch.Tensor) else y_true
            L_contrast = reference.sum() * 0.0
        if not isinstance(L_pyro, torch.Tensor):
            reference = logits if isinstance(logits, torch.Tensor) else y_true
            L_pyro = reference.sum() * 0.0

        L_pretrain = L_contrast + self.lambda_pyro * L_pyro

        if y_true is not None:
            if not isinstance(logits, torch.Tensor):
                raise ValueError("outputs['logits'] is required for classification loss")
            L_cls = self.cls_loss(logits, y_true)
        else:
            reference = logits if isinstance(logits, torch.Tensor) else L_pretrain
            L_cls = reference.sum() * 0.0

        if stage == "pretrain":
            L_total = L_pretrain
        elif stage == "finetune":
            L_total = self.lambda_cls * L_cls
        else:
            L_total = self.lambda_cls * L_cls + L_pretrain

        contrast_losses = outputs.get("contrast_losses", {})
        if not isinstance(contrast_losses, dict):
            contrast_losses = {}

        return {
            "L_total": L_total,
            "L_pretrain": L_pretrain,
            "L_cls": L_cls,
            "L_contrast": L_contrast,
            "L_pyro": L_pyro,
            "L_cross": contrast_losses.get("cross", L_contrast.sum() * 0.0),
            "L_spatial": contrast_losses.get("spatial", L_contrast.sum() * 0.0),
            "L_temporal": contrast_losses.get("temporal", L_contrast.sum() * 0.0),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 6, 7
    outputs = {
        "logits": torch.randn(B, C, H, W),
        "L_contrast": torch.tensor(1.2),
        "L_pyro": torch.tensor(0.3),
        "contrast_losses": {
            "cross": torch.tensor(0.4),
            "spatial": torch.tensor(0.5),
            "temporal": torch.tensor(0.3),
        },
    }
    y = torch.randint(0, C, (B, H, W))
    loss_fn = FireTrendLoss(class_weights=[1.0, 2.0, 4.0], lambda_pyro=0.5)
    print({k: float(v.detach()) for k, v in loss_fn(outputs, y, stage="joint").items()})
