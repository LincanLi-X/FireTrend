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
        # Define grid-neighborhood radius for N(i). radius=1 means 8-connected neighborhood.
        self.spatial_radius = 1

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
    def _neighbor_offsets(self, radius):
        offsets = []
        for dh in range(-radius, radius + 1):
            for dw in range(-radius, radius + 1):
                if dh == 0 and dw == 0:
                    continue
                offsets.append((dh, dw))
        return offsets

    def spatial_contrast(self, Z):
        """
        Args:
            Z: [B, T, D, H, W]
        Returns:
            spatial_loss (scalar)
        """
        check_tensor_shape(Z, 5, "SpatialContrast:Z")
        B, T, D, H, W = Z.shape
        radius = self.spatial_radius
        if H < 2 or W < 2:
            return torch.tensor(0.0, device=Z.device)

        # Treat each (batch, time) slice as an independent spatial field.
        Z_bt = rearrange(Z, "b t d h w -> (b t) d h w")  # [BT, D, H, W]
        BT = Z_bt.shape[0]
        n_anchor = max(1, self.spatial_neighbors)

        h_idx = torch.randint(0, H, (BT, n_anchor), device=Z.device)
        w_idx = torch.randint(0, W, (BT, n_anchor), device=Z.device)

        # Anchor embeddings z_i: [BT, n_anchor, D]
        bt_ids = torch.arange(BT, device=Z.device)[:, None]
        z_i = Z_bt[bt_ids, :, h_idx, w_idx]

        # Build neighborhood N(i) with local grid offsets.
        offsets = self._neighbor_offsets(radius)
        K = len(offsets)
        z_j_all = torch.zeros(BT, n_anchor, K, D, device=Z.device, dtype=Z.dtype)
        valid_mask = torch.zeros(BT, n_anchor, K, device=Z.device, dtype=torch.bool)

        for k, (dh, dw) in enumerate(offsets):
            nh = h_idx + dh
            nw = w_idx + dw
            valid = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
            nh = nh.clamp(0, H - 1)
            nw = nw.clamp(0, W - 1)

            z_j_all[:, :, k, :] = Z_bt[bt_ids, :, nh, nw]
            valid_mask[:, :, k] = valid

        # Spatial-guided attention alpha_ij = softmax(sim(z_i, z_j)/tau), j in N(i)
        z_i_n = F.normalize(z_i, dim=-1)
        z_j_n = F.normalize(z_j_all, dim=-1)
        sim = (z_i_n.unsqueeze(2) * z_j_n).sum(dim=-1)  # [BT, n_anchor, K]
        sim = sim / self.temperature
        sim = sim.masked_fill(~valid_mask, -1e9)

        alpha = torch.softmax(sim, dim=2)
        alpha = alpha * valid_mask.float()
        alpha = alpha / (alpha.sum(dim=2, keepdim=True) + 1e-8)

        # Refined spatial embedding: z~_i = sum_{j in N(i)} alpha_ij * z_j
        z_tilde = (alpha.unsqueeze(-1) * z_j_all).sum(dim=2)  # [BT, n_anchor, D]

        z_i = rearrange(z_i, "bt n d -> (bt n) d")
        z_tilde = rearrange(z_tilde, "bt n d -> (bt n) d")

        # Projection & InfoNCE loss between original and neighborhood-refined embeddings.
        z1 = self.projection_head(z_i)
        z2 = self.projection_head(z_tilde)

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

    def cross_modal_contrast(self, Z_modalities):
        """
        Args:
            Z_modalities:
                - dict[str, Tensor[B,T,D,H,W]] or
                - list/tuple of Tensor[B,T,D,H,W]
        Returns:
            averaged pairwise cross-view loss (scalar tensor)
        """
        if isinstance(Z_modalities, dict):
            names = list(Z_modalities.keys())
            tensors = [Z_modalities[n] for n in names]
        elif isinstance(Z_modalities, (list, tuple)):
            tensors = list(Z_modalities)
            names = [f"m{i}" for i in range(len(tensors))]
        else:
            raise ValueError("[CrossModal] Z_modalities must be dict/list/tuple.")

        if len(tensors) < 2:
            device = tensors[0].device if len(tensors) == 1 else next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        total = None
        n_pairs = 0
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                pair_loss = self.cross_view_contrast(tensors[i], tensors[j])
                total = pair_loss if total is None else (total + pair_loss)
                n_pairs += 1
                if self.verbose:
                    print(f"[Contrast] Cross-modal pair ({names[i]}, {names[j]})")

        return total / max(n_pairs, 1)

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------
    def forward(self, Z, Z_m1=None, Z_m2=None, Z_modalities=None):
        """
        Args:
            Z: [B, T, D, H, W] — base embedding
            Z_m1, Z_m2: optional cross-modal embeddings (legacy pair mode)
            Z_modalities: optional dict/list of modality embeddings
        Returns:
            refined embedding, total contrastive loss
        """
        check_tensor_shape(Z, 5, "MultiViewContrastive:Z")

        temporal_loss = self.temporal_contrast(Z)
        spatial_loss = self.spatial_contrast(Z)

        cross_loss = torch.tensor(0.0, device=Z.device)
        if Z_modalities is not None:
            cross_loss = self.cross_modal_contrast(Z_modalities)
        elif Z_m1 is not None and Z_m2 is not None:
            cross_loss = self.cross_view_contrast(Z_m1, Z_m2)

        total_loss = temporal_loss + spatial_loss + cross_loss

        refined = self.projection_head(rearrange(Z.mean(dim=[1, 3, 4]), "b d -> b d"))
        if self.verbose:
            print(
                f"[Contrast] Total Loss = {total_loss.item():.4f} "
                f"(temporal={temporal_loss.item():.4f}, spatial={spatial_loss.item():.4f}, cross={cross_loss.item():.4f})"
            )

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
