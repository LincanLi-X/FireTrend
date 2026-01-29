"""
contrastive_learning.py
----------------------------------------
FireTrend Stage 2 — Multi-View Contrastive Learning

Implements three InfoNCE contrastive mechanisms:
1. Temporal-view (Z_t ↔ Z_{t+1})
2. Spatial-view  (Z_i ↔ Z_j)
3. Cross-view    (Z^m1 ↔ Z^m2)

Outputs:
    refined_embeddings, total_contrastive_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------------------------------------
# Utility: Shape Checking
# -----------------------------------------------------------
def check_tensor_shape(tensor: torch.Tensor, expected_min_dims: int, context: str = ""):
    if tensor.ndim < expected_min_dims:
        msg = (f"\033[91m[ShapeError in {context}] Expected ≥{expected_min_dims} dims, "
               f"but got {tensor.ndim}. Tensor shape = {list(tensor.shape)}\033[0m")
        raise ValueError(msg)
    return True


# -----------------------------------------------------------
# InfoNCE Loss
# -----------------------------------------------------------
class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss with cosine similarity.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i, z_j: [B, D]
        Returns:
            scalar loss
        """
        check_tensor_shape(z_i, 2, "InfoNCE:z_i")
        check_tensor_shape(z_j, 2, "InfoNCE:z_j")
        assert z_i.shape == z_j.shape, (
            f"[ShapeMismatch] z_i {z_i.shape} vs z_j {z_j.shape}"
        )

        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        logits = torch.mm(z_i, z_j.t()) / self.temperature
        labels = torch.arange(z_i.size(0), device=z_i.device)
        loss = F.cross_entropy(logits, labels)
        return loss


# -----------------------------------------------------------
# Projection Head
# -----------------------------------------------------------
class ProjectionHead(nn.Module):
    """
    Small MLP projection head for contrastive embeddings.
    """

    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        check_tensor_shape(x, 2, "ProjectionHead:x")
        return self.net(x)


# -----------------------------------------------------------
# 🔥 Main Module: Multi-View Contrastive Learning
# -----------------------------------------------------------
class MultiViewContrastive(nn.Module):
    """
    Implements FireTrend Stage 2 contrastive objectives:
    - Temporal-view contrast
    - Spatial-view contrast
    - Cross-view contrast
    """

    def __init__(self, embed_dim=128, temperature=0.07, spatial_neighbors=4, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.temperature = temperature
        self.spatial_neighbors = spatial_neighbors

        self.projection_head = ProjectionHead(embed_dim, hidden_dim=embed_dim * 2, out_dim=embed_dim)
        self.loss_fn = InfoNCELoss(temperature=temperature)

    # -------------------------------------------------------
    # Temporal-view Contrast (Z_t ↔ Z_{t+1})
    # -------------------------------------------------------
    def temporal_contrast(self, Z):
        """
        Args:
            Z: [B, T, D, H, W]
        Returns:
            temporal_loss (scalar)
        """
        check_tensor_shape(Z, 5, "TemporalContrast:Z")
        B, T, D, H, W = Z.shape
        if T < 2:
            return torch.tensor(0.0, device=Z.device)

        Z_t = Z[:, :-1].mean(dim=[3, 4])  # [B, T-1, D]
        Z_next = Z[:, 1:].mean(dim=[3, 4])  # [B, T-1, D]

        Z_t = rearrange(Z_t, "b t d -> (b t) d")
        Z_next = rearrange(Z_next, "b t d -> (b t) d")

        z1 = self.projection_head(Z_t)
        z2 = self.projection_head(Z_next)

        loss = self.loss_fn(z1, z2)
        if self.verbose:
            print(f"[Contrast] Temporal loss = {loss.item():.4f}")
        return loss

    # -------------------------------------------------------
    # Spatial-view Contrast (Z_i ↔ Z_j)
    # -------------------------------------------------------
    def spatial_contrast(self, Z):
        """
        Args:
            Z: [B, T, D, H, W]
        Returns:
            spatial_loss (scalar)
        """
        check_tensor_shape(Z, 5, "SpatialContrast:Z")
        B, T, D, H, W = Z.shape
        Z_t = Z.mean(dim=1)  # average over time [B, D, H, W]

        # sample spatial_neighbors coordinates
        i_coords = torch.randint(0, H - 1, (B, self.spatial_neighbors), device=Z.device)
        j_coords = torch.randint(0, W - 1, (B, self.spatial_neighbors), device=Z.device)

        # Extract features per sample correctly
        z_i_list, z_j_list = [], []
        for b in range(B):
            z_i_b = Z_t[b, :, i_coords[b], j_coords[b]]  # [D, n]
            z_j_b = Z_t[b, :, (i_coords[b] + 1).clamp(max=H - 1),
                            (j_coords[b] + 1).clamp(max=W - 1)]  # [D, n]
            z_i_list.append(z_i_b)
            z_j_list.append(z_j_b)

        # Stack and rearrange correctly
        z_i = torch.stack(z_i_list, dim=0)  # [B, D, n]
        z_j = torch.stack(z_j_list, dim=0)  # [B, D, n]
        z_i = rearrange(z_i, "b d n -> (b n) d")
        z_j = rearrange(z_j, "b d n -> (b n) d")

        # Projection & InfoNCE loss
        z1 = self.projection_head(z_i)
        z2 = self.projection_head(z_j)

        loss = self.loss_fn(z1, z2)
        if self.verbose:
            print(f"[Contrast] Spatial loss = {loss.item():.4f}")
        return loss


    # -------------------------------------------------------
    # Cross-view Contrast (Z^m1 ↔ Z^m2)
    # -------------------------------------------------------
    def cross_view_contrast(self, Z_m1, Z_m2):
        """
        Args:
            Z_m1, Z_m2: [B, T, D, H, W]
        Returns:
            cross_loss (scalar)
        """
        check_tensor_shape(Z_m1, 5, "CrossView:Z_m1")
        check_tensor_shape(Z_m2, 5, "CrossView:Z_m2")
        assert Z_m1.shape == Z_m2.shape, "[ShapeMismatch] Z_m1 and Z_m2 must match"

        Z1 = Z_m1.mean(dim=[1, 3, 4])  # [B, D]
        Z2 = Z_m2.mean(dim=[1, 3, 4])  # [B, D]

        z1 = self.projection_head(Z1)
        z2 = self.projection_head(Z2)

        loss = self.loss_fn(z1, z2)
        if self.verbose:
            print(f"[Contrast] Cross-view loss = {loss.item():.4f}")
        return loss

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------
    def forward(self, Z, Z_m1=None, Z_m2=None):
        """
        Args:
            Z: [B, T, D, H, W] — base embedding
            Z_m1, Z_m2: optional cross-modal embeddings
        Returns:
            refined embedding, total contrastive loss
        """
        check_tensor_shape(Z, 5, "MultiViewContrastive:Z")

        temporal_loss = self.temporal_contrast(Z)
        spatial_loss = self.spatial_contrast(Z)

        cross_loss = 0.0
        if Z_m1 is not None and Z_m2 is not None:
            cross_loss = self.cross_view_contrast(Z_m1, Z_m2)

        total_loss = temporal_loss + spatial_loss + cross_loss

        refined = self.projection_head(rearrange(Z.mean(dim=[1, 3, 4]), "b d -> b d"))
        if self.verbose:
            print(f"[Contrast] Total Loss = {total_loss.item():.4f}")

        return refined, total_loss


# -----------------------------------------------------------
# Unit Test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, H, W = 2, 5, 64, 8, 8
    Z = torch.randn(B, T, D, H, W)
    Z_m1 = torch.randn(B, T, D, H, W)
    Z_m2 = torch.randn(B, T, D, H, W)

    contrastive = MultiViewContrastive(embed_dim=D, verbose=True)
    refined, loss = contrastive(Z, Z_m1, Z_m2)

    print(f"Refined embedding: {refined.shape}")
    print(f"Total contrastive loss: {loss.item():.4f}")
