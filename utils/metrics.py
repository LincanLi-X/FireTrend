import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def compute_iou(preds, targets, threshold=0.5, eps=1e-6):
    """
    Compute Intersection over Union (IoU) for binary wildfire masks.
    Args:
        preds (torch.Tensor): predicted fire map [B, 1, H, W]
        targets (torch.Tensor): ground truth fire map [B, 1, H, W]
    """
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > threshold).float()

    intersection = (preds_bin * targets_bin).sum(dim=(1, 2, 3))
    union = (preds_bin + targets_bin - preds_bin * targets_bin).sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def compute_f1(preds, targets, threshold=0.5):
    preds_bin = (preds > threshold).cpu().numpy().astype(np.uint8)
    targets_bin = (targets > threshold).cpu().numpy().astype(np.uint8)
    f1 = f1_score(targets_bin.flatten(), preds_bin.flatten())
    return f1


def compute_auc(preds, targets):
    """
    Compute AUROC for continuous probability predictions.
    """
    preds_np = preds.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()
    try:
        auc = roc_auc_score(targets_np, preds_np)
    except ValueError:
        auc = np.nan
    return auc

## trend correlation is not a commonly used metrics in spatial-temporal task
# def compute_trend_correlation(preds_seq, targets_seq):
#     """
#     Compute temporal trend correlation across sequence.
#     Args:
#         preds_seq: [B, T, H, W]
#         targets_seq: [B, T, H, W]
#     """
#     preds_flat = preds_seq.reshape(preds_seq.shape[0], preds_seq.shape[1], -1)
#     targets_flat = targets_seq.reshape(targets_seq.shape[0], targets_seq.shape[1], -1)

#     corrs = []
#     for b in range(preds_flat.shape[0]):
#         corr = np.corrcoef(preds_flat[b].mean(axis=1), targets_flat[b].mean(axis=1))[0, 1]
#         corrs.append(corr)
#     return np.nanmean(corrs)


def compute_metrics(preds, targets, seq_preds=None, seq_targets=None, threshold=0.5):
    """
    Compute all evaluation metrics and return as dict.
    """
    metrics = {
        "IoU": compute_iou(preds, targets, threshold),
        "F1": compute_f1(preds, targets, threshold),
        "AUC": compute_auc(preds, targets),
    }

    # if seq_preds is not None and seq_targets is not None:
    #     metrics["TrendCorr"] = compute_trend_correlation(seq_preds, seq_targets)

    return metrics
