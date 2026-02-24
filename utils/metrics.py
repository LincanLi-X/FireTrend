import torch
import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def compute_mae(preds, targets):
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    return float(np.mean(np.abs(preds_np - targets_np)))


def compute_mse(preds, targets):
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    return float(np.mean((preds_np - targets_np) ** 2))

def _as_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1).astype(np.float64)


def _derive_risk_thresholds(targets_flat, method="quantile", quantiles=(1 / 3, 2 / 3), value_range=(0.0, 100.0)):
    lo, hi = float(value_range[0]), float(value_range[1])
    y = np.clip(targets_flat, lo, hi)

    method = str(method).lower()
    if method in ("quantile", "quantiles", "stat", "statistical"):
        q1, q2 = float(quantiles[0]), float(quantiles[1])
        q1 = min(max(q1, 0.0), 1.0)
        q2 = min(max(q2, 0.0), 1.0)
        if q2 <= q1:
            q1, q2 = 1 / 3, 2 / 3
        t1 = float(np.quantile(y, q1))
        t2 = float(np.quantile(y, q2))
    elif method in ("fixed", "manual"):
        t1, t2 = float(quantiles[0]), float(quantiles[1])
    else:
        raise ValueError(f"Unsupported risk threshold method: {method}")

    t1 = min(max(t1, lo), hi)
    t2 = min(max(t2, lo), hi)
    if t2 <= t1:
        # keep valid split
        mid = (lo + hi) * 0.5
        t1 = (lo + mid) * 0.5
        t2 = (mid + hi) * 0.5
    return t1, t2


def _risk_classes(x_flat, t1, t2):
    # 0=low, 1=mid, 2=high
    cls = np.zeros_like(x_flat, dtype=np.int64)
    cls[x_flat >= t2] = 2
    cls[(x_flat >= t1) & (x_flat < t2)] = 1
    return cls


def _risk_scores(x_flat, t1, t2, value_range=(0.0, 100.0)):
    """
    Build smooth class scores from scalar wildfire_risk.
    Output shape: [N, 3], rows sum to 1.
    """
    lo, hi = float(value_range[0]), float(value_range[1])
    x = np.clip(x_flat, lo, hi)
    eps = 1e-8

    mid = (t1 + t2) * 0.5
    width = max((t2 - t1) * 0.5, eps)

    low_score = np.clip((t2 - x) / max(t2 - lo, eps), 0.0, 1.0)
    high_score = np.clip((x - t1) / max(hi - t1, eps), 0.0, 1.0)
    mid_score = np.clip(1.0 - np.abs(x - mid) / width, 0.0, 1.0)

    scores = np.stack([low_score, mid_score, high_score], axis=1)
    scores = scores + eps
    scores = scores / np.sum(scores, axis=1, keepdims=True)
    return scores


def compute_multiclass_iou(y_true_cls, y_pred_cls, num_classes=3, eps=1e-6):
    ious = []
    for c in range(num_classes):
        pred_c = (y_pred_cls == c)
        true_c = (y_true_cls == c)
        inter = np.logical_and(pred_c, true_c).sum()
        union = np.logical_or(pred_c, true_c).sum()
        if union > 0:
            ious.append((inter + eps) / (union + eps))
    if len(ious) == 0:
        return float("nan")
    return float(np.mean(ious))


def compute_multiclass_f1(y_true_cls, y_pred_cls):
    return float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))


def compute_multiclass_auprc(y_true_cls, y_score, num_classes=3):
    y_true_onehot = np.zeros((y_true_cls.shape[0], num_classes), dtype=np.int64)
    y_true_onehot[np.arange(y_true_cls.shape[0]), y_true_cls] = 1
    try:
        return float(average_precision_score(y_true_onehot, y_score, average="macro"))
    except ValueError:
        return float("nan")


def compute_metrics(
    preds,
    targets,
    risk_method="quantile",
    risk_quantiles=(1 / 3, 2 / 3),
    risk_value_range=(0.0, 100.0),
):
    """
    Compute regression + risk-level classification metrics.
    Risk level is derived from wildfire_risk into 3 classes: low / mid / high.
    """
    preds_flat = _as_numpy_1d(preds)
    targets_flat = _as_numpy_1d(targets)

    t1, t2 = _derive_risk_thresholds(
        targets_flat=targets_flat,
        method=risk_method,
        quantiles=risk_quantiles,
        value_range=risk_value_range,
    )

    y_true_cls = _risk_classes(targets_flat, t1, t2)
    y_pred_cls = _risk_classes(preds_flat, t1, t2)
    y_pred_score = _risk_scores(preds_flat, t1, t2, value_range=risk_value_range)

    mse = compute_mse(preds, targets)
    return {
        "MAE": compute_mae(preds, targets),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "IoU": compute_multiclass_iou(y_true_cls, y_pred_cls, num_classes=3),
        "F1": compute_multiclass_f1(y_true_cls, y_pred_cls),
        "AUPRC": compute_multiclass_auprc(y_true_cls, y_pred_score, num_classes=3),
        "RiskThresholds": {"low_mid": float(t1), "mid_high": float(t2)},
    }
