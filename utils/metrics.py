from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


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
    scores = []
    for cls in range(num_classes):
        y_true_bin = (y_true_cls == cls).astype(np.int64)
        if y_true_bin.sum() == 0:
            continue
        order = np.argsort(-y_score[:, cls])
        sorted_true = y_true_bin[order]
        tp = np.cumsum(sorted_true)
        precision = tp / np.maximum(np.arange(1, sorted_true.size + 1), 1)
        ap = (precision * sorted_true).sum() / max(sorted_true.sum(), 1)
        scores.append(float(ap))
    return float(np.mean(scores)) if scores else float("nan")


def compute_macro_f1(y_true_cls, y_pred_cls, num_classes: int = 3, eps: float = 1e-8) -> float:
    f1s = []
    for cls in range(num_classes):
        tp = np.logical_and(y_pred_cls == cls, y_true_cls == cls).sum()
        fp = np.logical_and(y_pred_cls == cls, y_true_cls != cls).sum()
        fn = np.logical_and(y_pred_cls != cls, y_true_cls == cls).sum()
        denom = 2 * tp + fp + fn
        f1s.append(float((2 * tp + eps) / (denom + eps)) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def compute_kappa(y_true_cls, y_pred_cls, num_classes: int = 3, eps: float = 1e-8) -> float:
    conf = np.zeros((num_classes, num_classes), dtype=np.float64)
    valid = (y_true_cls >= 0) & (y_true_cls < num_classes) & (y_pred_cls >= 0) & (y_pred_cls < num_classes)
    for true, pred in zip(y_true_cls[valid], y_pred_cls[valid]):
        conf[int(true), int(pred)] += 1.0
    total = conf.sum()
    if total <= 0:
        return float("nan")
    p_o = np.trace(conf) / total
    p_e = (conf.sum(axis=1) * conf.sum(axis=0)).sum() / (total * total)
    if abs(1.0 - p_e) < eps:
        return 1.0 if abs(p_o - 1.0) < eps else 0.0
    return float((p_o - p_e) / (1.0 - p_e))


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
        "F1": compute_macro_f1(y_true, y_pred, num_classes=num_classes),
        "AUPRC": compute_multiclass_auprc(y_true, y_score, num_classes=num_classes),
        "Accuracy": float(np.mean(y_true == y_pred)),
        "Kappa": compute_kappa(y_true, y_pred, num_classes=num_classes),
    }
    metrics.update(compute_class_ious(y_true, y_pred, num_classes=num_classes))
    return metrics


def compute_pds(probabilities, wind_u, wind_v, class_index: int = 2, eps: float = 1e-8) -> float:
    """
    Propagation Direction Similarity between high-risk probability gradient
    and ERA5 wind vectors.
    """
    probs = _to_numpy(probabilities)
    if probs.ndim != 4:
        raise ValueError(f"probabilities must be [N,C,H,W], got {probs.shape}")
    field = probs[:, class_index]
    u = _to_numpy(wind_u)
    v = _to_numpy(wind_v)
    if u.ndim == 4 and u.shape[1] == 1:
        u = u[:, 0]
    if v.ndim == 4 and v.shape[1] == 1:
        v = v[:, 0]
    if u.shape != field.shape or v.shape != field.shape:
        raise ValueError(f"wind shape mismatch: field={field.shape}, u={u.shape}, v={v.shape}")

    grad_y, grad_x = np.gradient(field, axis=(-2, -1))
    grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
    wind_norm = np.sqrt(u ** 2 + v ** 2)
    mask = (grad_norm > eps) & (wind_norm > eps)
    if not mask.any():
        return float("nan")
    cosine = (grad_x * u + grad_y * v) / np.maximum(grad_norm * wind_norm, eps)
    return float(np.mean(cosine[mask]))


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


def compute_temporal_drift_sequence(probabilities, targets, target_indices=None, eps: float = 1e-8) -> dict[str, float]:
    probs = torch.as_tensor(probabilities).float()
    y = torch.as_tensor(targets).long()
    if probs.ndim != 4 or y.ndim != 3:
        raise ValueError(f"Expected probabilities [N,C,H,W] and targets [N,H,W], got {probs.shape}, {y.shape}")
    if probs.size(0) < 2:
        return {"TDE": float("nan"), "TCS": float("nan")}

    if target_indices is not None:
        idx = torch.as_tensor(target_indices).long()
        order = torch.argsort(idx)
        probs = probs[order]
        y = y[order]
        idx = idx[order]
        valid_pairs = (idx[1:] - idx[:-1]) == 1
    else:
        valid_pairs = torch.ones(probs.size(0) - 1, dtype=torch.bool)

    if not bool(valid_pairs.any()):
        return {"TDE": float("nan"), "TCS": float("nan")}

    p0 = probs[:-1][valid_pairs]
    p1 = probs[1:][valid_pairs]
    y0 = y[:-1][valid_pairs]
    y1 = y[1:][valid_pairs]
    return compute_temporal_drift_metrics(p0, p1, y0, y1, eps=eps)
