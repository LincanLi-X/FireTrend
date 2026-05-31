from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _prepare_targets(targets) -> np.ndarray:
    targets_np = _to_numpy(targets)
    if targets_np.ndim == 4 and targets_np.shape[1] == 1:
        targets_np = targets_np[:, 0]
    return targets_np.reshape(-1).astype(np.int64)


def _prepare_probabilities(preds, probabilities=None, num_classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if probabilities is not None:
        probs = _to_numpy(probabilities)
    else:
        if isinstance(preds, torch.Tensor):
            tensor = preds.detach()
            if tensor.ndim == 4 and tensor.size(1) == num_classes:
                probs = F.softmax(tensor, dim=1).cpu().numpy()
            else:
                pred_cls = tensor.cpu().numpy()
                pred_cls = pred_cls[:, 0] if pred_cls.ndim == 4 and pred_cls.shape[1] == 1 else pred_cls
                pred_cls = pred_cls.reshape(-1).astype(np.int64)
                probs_flat = np.zeros((pred_cls.size, num_classes), dtype=np.float32)
                probs_flat[np.arange(pred_cls.size), np.clip(pred_cls, 0, num_classes - 1)] = 1.0
                return pred_cls, probs_flat
        else:
            preds_np = np.asarray(preds)
            if preds_np.ndim == 4 and preds_np.shape[1] == num_classes:
                exp = np.exp(preds_np - preds_np.max(axis=1, keepdims=True))
                probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-8, None)
            else:
                pred_cls = preds_np.reshape(-1).astype(np.int64)
                probs_flat = np.zeros((pred_cls.size, num_classes), dtype=np.float32)
                probs_flat[np.arange(pred_cls.size), np.clip(pred_cls, 0, num_classes - 1)] = 1.0
                return pred_cls, probs_flat

    if probs.ndim != 4:
        raise ValueError(f"probabilities/logits must be [B,C,H,W], got {probs.shape}")
    pred_cls = probs.argmax(axis=1).reshape(-1).astype(np.int64)
    probs_flat = np.moveaxis(probs, 1, -1).reshape(-1, num_classes)
    return pred_cls, probs_flat


def compute_multiclass_iou(y_true_cls, y_pred_cls, num_classes: int = 3, eps: float = 1e-8) -> float:
    ious = []
    for cls in range(num_classes):
        pred = y_pred_cls == cls
        true = y_true_cls == cls
        union = np.logical_or(pred, true).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred, true).sum()
        ious.append((inter + eps) / (union + eps))
    return float(np.mean(ious)) if ious else float("nan")


def compute_class_ious(y_true_cls, y_pred_cls, num_classes: int = 3, eps: float = 1e-8) -> dict[str, float]:
    out = {}
    for cls in range(num_classes):
        pred = y_pred_cls == cls
        true = y_true_cls == cls
        union = np.logical_or(pred, true).sum()
        inter = np.logical_and(pred, true).sum()
        out[f"IoU_class_{cls}"] = float((inter + eps) / (union + eps)) if union > 0 else float("nan")
    return out


def compute_multiclass_auprc(y_true_cls, y_score, num_classes: int = 3) -> float:
    y_true_onehot = np.zeros((y_true_cls.shape[0], num_classes), dtype=np.int64)
    valid = (y_true_cls >= 0) & (y_true_cls < num_classes)
    y_true_onehot[np.arange(y_true_cls.shape[0])[valid], y_true_cls[valid]] = 1
    try:
        return float(average_precision_score(y_true_onehot, y_score, average="macro"))
    except ValueError:
        return float("nan")


def compute_metrics(preds, targets, probabilities=None, num_classes: int = 3) -> dict[str, float]:
    """
    Compute ordinal risk-level classification metrics.

    Args:
        preds: logits/probabilities [B,C,H,W] or hard labels.
        targets: class labels [B,H,W] or [B,1,H,W].
        probabilities: optional probability tensor [B,C,H,W].
    """
    y_true = _prepare_targets(targets)
    y_pred, y_score = _prepare_probabilities(preds, probabilities=probabilities, num_classes=num_classes)

    metrics = {
        "IoU": compute_multiclass_iou(y_true, y_pred, num_classes=num_classes),
        "F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "AUPRC": compute_multiclass_auprc(y_true, y_score, num_classes=num_classes),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }
    metrics.update(compute_class_ious(y_true, y_pred, num_classes=num_classes))
    return metrics


def compute_temporal_drift_metrics(prob_t, prob_next, true_t, true_next, eps: float = 1e-8) -> dict[str, float]:
    """
    Optional prediction-drift diagnostics from the appendix: TDE and TCS.
    Inputs are probability maps [B,C,H,W] and class maps [B,H,W].
    """
    p0 = torch.as_tensor(prob_t).float()
    p1 = torch.as_tensor(prob_next).float()
    y0 = F.one_hot(torch.as_tensor(true_t).long(), num_classes=p0.size(1)).permute(0, 3, 1, 2).float()
    y1 = F.one_hot(torch.as_tensor(true_next).long(), num_classes=p0.size(1)).permute(0, 3, 1, 2).float()
    dp = (p1 - p0).reshape(p0.size(0), -1)
    dy = (y1 - y0).reshape(p0.size(0), -1)
    tde = torch.mean(torch.abs(dp - dy)).item()
    tcs = F.cosine_similarity(dp, dy, dim=1, eps=eps).mean().item()
    return {"TDE": float(tde), "TCS": float(tcs)}
