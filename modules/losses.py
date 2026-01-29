"""
losses.py
----------------------------------------
FireTrend Stage 7: Define Loss Functions

Implements:
    - L_pred: wildfire prediction loss (BCE or Focal Loss)
    - L_contrast: InfoNCE loss
    - L_phys: physics consistency loss (wind-direction smoothness)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# Shape Checking Utility
# -----------------------------------------------------------
def check_shape(tensor, expected_dims, context=""):
    if tensor.ndim != expected_dims:
        raise ValueError(f"[ShapeError:{context}] Expected {expected_dims}D tensor, got shape {tensor.shape}")
    return True


# -----------------------------------------------------------
# 1. Prediction Loss (BCE + optional Focal)
# -----------------------------------------------------------
class PredictionLoss(nn.Module):
    """
    Wildfire prediction loss using BCE or Focal Loss.
    """

    def __init__(self, loss_type="bce", alpha=0.25, gamma=2.0):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [B, 1, H, W]
            y_true: [B, 1, H, W]
        Returns:
            scalar loss
        """
        check_shape(y_pred, 4, "PredictionLoss:y_pred")
        check_shape(y_true, 4, "PredictionLoss:y_true")

        if self.loss_type == "bce":
            return self.bce(y_pred, y_true)

        elif self.loss_type == "focal":
            # Focal loss: focus on hard samples
            bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
            prob = torch.sigmoid(y_pred)
            p_t = y_true * prob + (1 - y_true) * (1 - prob)
            focal_term = (1 - p_t) ** self.gamma
            loss = self.alpha * focal_term * bce_loss
            return loss.mean()

        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")


# -----------------------------------------------------------
# 2. Contrastive Loss (InfoNCE)
# -----------------------------------------------------------
class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for multi-view contrastive learning.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1, z2: [B, D] embeddings
        Returns:
            scalar loss
        """

        check_shape(z1, 2, "InfoNCE:z1")
        check_shape(z2, 2, "InfoNCE:z2")

        B, D = z1.shape
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        logits = torch.matmul(z1, z2.T) / self.temperature  # [B, B]
        labels = torch.arange(B, device=z1.device)

        loss = F.cross_entropy(logits, labels)
        return loss


# -----------------------------------------------------------
# 3. Physics Consistency Loss
# -----------------------------------------------------------
class PhysicsConsistencyLoss(nn.Module):
    """
    Physics-based consistency constraint:
    Encourage smooth risk propagation along wind direction.
    """

    def __init__(self, lambda_smooth=0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth


    def forward(self, Y_pred, wind_u, wind_v):
        # --- Safety fix: handle 5D input [B, 1, 6, H, W] ---
        if wind_u.ndim == 5:
            wind_u = wind_u.mean(dim=2)
        if wind_v.ndim == 5:
            wind_v = wind_v.mean(dim=2)
    
        check_shape(Y_pred, 4, "PhysLoss:Y_pred")
        check_shape(wind_u, 4, "PhysLoss:wind_u")
        check_shape(wind_v, 4, "PhysLoss:wind_v")

        # Compute spatial gradients
        grad_y = Y_pred[:, :, 1:, :] - Y_pred[:, :, :-1, :]   # [B, 1, H-1, W]
        grad_x = Y_pred[:, :, :, 1:] - Y_pred[:, :, :, :-1]   # [B, 1, H, W-1]

        # Compute wind direction
        wind_angle = torch.atan2(wind_v, wind_u + 1e-6)
        cos_w, sin_w = torch.cos(wind_angle), torch.sin(wind_angle)

        # Align field shapes to overlap region [B, 1, H-1, W-1]
        cos_w = cos_w[:, :, :-1, :-1]
        sin_w = sin_w[:, :, :-1, :-1]
        grad_x = grad_x[:, :, :-1, :]    # 去掉底部一行
        grad_y = grad_y[:, :, :, :-1]    # 去掉最右一列

        assert grad_x.shape == grad_y.shape == cos_w.shape, (
            f"[ShapeMismatch] grad_x:{grad_x.shape}, grad_y:{grad_y.shape}, cos_w:{cos_w.shape}"
        )

        # Weighted smoothness loss
        grad_along_wind = cos_w * grad_x + sin_w * grad_y
        loss = self.lambda_smooth * torch.mean(grad_along_wind**2)

        return loss

# -----------------------------------------------------------
# ⚙️ Combined FireTrend Loss Wrapper
# -----------------------------------------------------------
class FireTrendLoss(nn.Module):
    """
    Combined loss for FireTrend:
        L_total = λ_pred * L_pred + λ_contrast * L_contrast + λ_phys * L_phys
    """

    def __init__(self, w_pred=1.0, w_contrast=0.5, w_phys=0.2, loss_type="bce"):
        super().__init__()
        self.w_pred = w_pred
        self.w_contrast = w_contrast
        self.w_phys = w_phys

        self.pred_loss = PredictionLoss(loss_type=loss_type)
        self.contrast_loss = InfoNCELoss()
        self.phys_loss = PhysicsConsistencyLoss()

    def forward(self, y_pred, y_true, embeddings=None, L_contrast=None, wind_u=None, wind_v=None):
        """
        Compute total loss (with optional components).

        Args:
            y_pred, y_true: [B, 1, H, W]
            z1, z2: embeddings for contrastive loss [B, D]
            wind_u, wind_v: [B, 1, H, W]
        """
        # --- Extract embeddings safely ---
        if isinstance(embeddings, (tuple, list)) and len(embeddings) == 2:
            z1, z2 = embeddings
        else:
            z1 = embeddings
            z2 = None

        # --- contrastive loss ---
        if z1 is None or z2 is None or not isinstance(z1, torch.Tensor) or not isinstance(z2, torch.Tensor):
            L_contrast = torch.tensor(0.0, device=y_pred.device)
        else:
            if z1.ndim > 2:
                z1 = z1.view(z1.size(0), -1)
            if z2.ndim > 2:
                z2 = z2.view(z2.size(0), -1)
            L_contrast = self.contrast_loss(z1, z2)


        L_pred = self.pred_loss(y_pred, y_true)
        #L_contrast = self.contrast_loss(z1, z2) if (z1 is not None and z2 is not None) else torch.tensor(0.0, device=y_pred.device)
        L_phys = self.phys_loss(y_pred, wind_u, wind_v) if (wind_u is not None and wind_v is not None) else torch.tensor(0.0, device=y_pred.device)

        L_total = self.w_pred * L_pred + self.w_contrast * L_contrast + self.w_phys * L_phys

        return {
            "L_total": L_total,
            "L_pred": L_pred,
            "L_contrast": L_contrast,
            "L_phys": L_phys
        }


# -----------------------------------------------------------
# Unit Test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Simulated inputs
    B, H, W, D = 4, 16, 16, 64
    y_pred = torch.randn(B, 1, H, W)
    y_true = torch.rand(B, 1, H, W)
    z1 = torch.randn(B, D)
    z2 = torch.randn(B, D)
    wind_u = torch.randn(B, 1, H, W)
    wind_v = torch.randn(B, 1, H, W)

    loss_fn = FireTrendLoss(loss_type="focal")
    losses = loss_fn(y_pred, y_true, z1, z2, wind_u, wind_v)

    print(f"   L_total: {losses['L_total']:.4f}")
    print(f"   L_pred: {losses['L_pred']:.4f}")
    print(f"   L_contrast: {losses['L_contrast']:.4f}")
    print(f"   L_phys: {losses['L_phys']:.6f}")
