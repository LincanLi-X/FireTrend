"""
Multi-view contrastive learning for FireTrend.

The losses follow the NeurIPS 2026 manuscript:
    L_contrast = L_cross + lambda_s * L_spat + lambda_t * L_temp

All objectives are label-free and operate on latent fire-state maps
Z with shape [B, T, D, H, W].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _sample_indices(total: int, max_samples: int | None, device: torch.device) -> torch.Tensor:
    if max_samples is None or max_samples <= 0 or total <= max_samples:
        return torch.arange(total, device=device)
    return torch.randperm(total, device=device)[:max_samples]


def _coerce_valid_mask(valid_region_mask: torch.Tensor | None, Z: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask [B,H,W] aligned with a latent tensor."""
    B, _, _, H, W = Z.shape
    if valid_region_mask is None:
        return torch.ones(B, H, W, dtype=torch.bool, device=Z.device)
    mask = torch.as_tensor(valid_region_mask, device=Z.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).expand(B, -1, -1)
    if mask.shape != (B, H, W):
        raise ValueError(f"valid_region_mask must be [B,H,W] or [H,W], got {mask.shape}")
    return mask


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE over paired embeddings [N, D]."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, symmetric: bool = True) -> torch.Tensor:
        if z_a.ndim != 2 or z_b.ndim != 2:
            raise ValueError(f"InfoNCE expects [N, D] tensors, got {z_a.shape} and {z_b.shape}")
        if z_a.shape != z_b.shape:
            raise ValueError(f"InfoNCE shape mismatch: {z_a.shape} vs {z_b.shape}")
        if z_a.size(0) <= 1:
            return _zero_like_loss(z_a)

        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        logits = z_a @ z_b.t() / self.temperature
        labels = torch.arange(z_a.size(0), device=z_a.device)
        loss = F.cross_entropy(logits, labels)
        if symmetric:
            loss = 0.5 * (loss + F.cross_entropy(logits.t(), labels))
        return loss


class ProjectionHead(nn.Module):
    """Small MLP projection head used by all contrastive views."""

    def __init__(self, in_dim: int, hidden_dim: int | None = None, out_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 2
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"ProjectionHead expects [N, D], got {x.shape}")
        return self.net(x)


class MultiViewContrastive(nn.Module):
    """
    Temporal, spatial, and cross-modal contrastive objectives.

    The implementation uses bounded sampling for memory safety on large grids
    while preserving the pair definitions from the manuscript.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        temperature: float = 0.07,
        lambda_temporal: float = 1.0,
        lambda_spatial: float = 1.0,
        lambda_cross: float = 1.0,
        spatial_radius: int = 1,
        max_temporal_cells: int = 512,
        max_spatial_anchors: int = 128,
        max_cross_samples: int = 1024,
        verbose: bool = False,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_temporal = float(lambda_temporal)
        self.lambda_spatial = float(lambda_spatial)
        self.lambda_cross = float(lambda_cross)
        self.spatial_radius = int(spatial_radius)
        self.max_temporal_cells = int(max_temporal_cells)
        self.max_spatial_anchors = int(max_spatial_anchors)
        self.max_cross_samples = int(max_cross_samples)
        self.verbose = verbose

        self.projection_head = ProjectionHead(embed_dim)
        self.infonce = InfoNCELoss(temperature=temperature)

    def _project_nd(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        projected = self.projection_head(flat)
        return projected.reshape(*x.shape[:-1], projected.shape[-1])

    def temporal_contrast(
        self,
        Z: torch.Tensor,
        valid_region_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Align consecutive latent states of the same grid cell.

        For each sampled cell, anchor z_{i,t} is contrasted against all
        time steps of that same cell except itself; z_{i,t+1} is positive.
        """
        if Z.ndim != 5:
            raise ValueError(f"Temporal contrast expects [B,T,D,H,W], got {Z.shape}")
        B, T, D, H, W = Z.shape
        if T < 2:
            return _zero_like_loss(Z)

        cells = rearrange(Z, "b t d h w -> (b h w) t d")
        valid_flat = rearrange(_coerce_valid_mask(valid_region_mask, Z), "b h w -> (b h w)")
        valid_indices = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
        if valid_indices.numel() <= 1:
            return _zero_like_loss(Z)
        subset = _sample_indices(valid_indices.numel(), self.max_temporal_cells, Z.device)
        idx = valid_indices[subset]
        cells = self._project_nd(cells[idx])
        cells = F.normalize(cells, dim=-1)

        losses = []
        for t in range(T - 1):
            anchor = cells[:, t]  # [N, D]
            logits = torch.einsum("nd,nkd->nk", anchor, cells) / self.temperature
            logits[:, t] = -torch.finfo(logits.dtype).max
            labels = torch.full((cells.size(0),), t + 1, device=Z.device, dtype=torch.long)
            losses.append(F.cross_entropy(logits, labels))
        loss = torch.stack(losses).mean()
        if self.verbose:
            print(f"[Contrast] temporal={loss.item():.4f}")
        return loss

    def _neighbor_offsets(self) -> list[tuple[int, int]]:
        offsets = []
        radius = max(1, self.spatial_radius)
        for dh in range(-radius, radius + 1):
            for dw in range(-radius, radius + 1):
                if dh == 0 and dw == 0:
                    continue
                offsets.append((dh, dw))
        return offsets

    @staticmethod
    def _shift_grid(x: torch.Tensor, dh: int, dw: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return x at neighbor offset (dh, dw) for every anchor cell."""
        BT, D, H, W = x.shape
        out = torch.zeros_like(x)
        mask = torch.zeros(BT, 1, H, W, dtype=torch.bool, device=x.device)

        dst_h0 = max(0, -dh)
        dst_h1 = min(H, H - dh)
        dst_w0 = max(0, -dw)
        dst_w1 = min(W, W - dw)
        if dst_h0 >= dst_h1 or dst_w0 >= dst_w1:
            return out, mask

        src_h0 = dst_h0 + dh
        src_h1 = dst_h1 + dh
        src_w0 = dst_w0 + dw
        src_w1 = dst_w1 + dw
        out[:, :, dst_h0:dst_h1, dst_w0:dst_w1] = x[:, :, src_h0:src_h1, src_w0:src_w1]
        mask[:, :, dst_h0:dst_h1, dst_w0:dst_w1] = True
        return out, mask

    def _spatial_guided_embedding(
        self,
        Z_bt: torch.Tensor,
        valid_mask_bt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute q_i,t = sum_j alpha_ij,t z_j,t over local neighbors."""
        offsets = self._neighbor_offsets()
        neighbors, masks = [], []
        for dh, dw in offsets:
            shifted, mask = self._shift_grid(Z_bt, dh, dw)
            if valid_mask_bt is not None:
                shifted_state, _ = self._shift_grid(valid_mask_bt[:, None].to(Z_bt.dtype), dh, dw)
                mask = mask & shifted_state.bool() & valid_mask_bt[:, None]
            neighbors.append(shifted)
            masks.append(mask)

        neighbor_tensor = torch.stack(neighbors, dim=1)  # [BT, K, D, H, W]
        valid_mask = torch.stack(masks, dim=1)  # [BT, K, 1, H, W]

        anchor = Z_bt.unsqueeze(1)
        sim = (F.normalize(anchor, dim=2) * F.normalize(neighbor_tensor, dim=2)).sum(dim=2)
        sim = sim / self.temperature
        sim = sim.masked_fill(~valid_mask.squeeze(2), -torch.finfo(sim.dtype).max)
        alpha = torch.softmax(sim, dim=1).unsqueeze(2)
        alpha = alpha * valid_mask.to(alpha.dtype)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        return (alpha * neighbor_tensor).sum(dim=1)

    def spatial_contrast(
        self,
        Z: torch.Tensor,
        valid_region_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Contrast each sampled spatial-guided embedding q_i,t with its
        neighboring q_j,t positives and all grid-cell negatives.
        """
        if Z.ndim != 5:
            raise ValueError(f"Spatial contrast expects [B,T,D,H,W], got {Z.shape}")
        B, T, D, H, W = Z.shape
        if H * W <= 1:
            return _zero_like_loss(Z)

        Z_bt = rearrange(Z, "b t d h w -> (b t) d h w")
        spatial_mask = _coerce_valid_mask(valid_region_mask, Z)
        valid_mask_bt = rearrange(
            spatial_mask[:, None].expand(B, T, H, W),
            "b t h w -> (b t) h w",
        )
        q_bt = self._spatial_guided_embedding(Z_bt, valid_mask_bt=valid_mask_bt)
        q_flat = rearrange(q_bt, "bt d h w -> bt (h w) d")
        q_flat = self._project_nd(q_flat)
        q_flat = F.normalize(q_flat, dim=-1)

        BT, N, _ = q_flat.shape
        mask_flat = valid_mask_bt.reshape(BT, N)
        batch_losses = []
        for bt_index in range(BT):
            valid_global = torch.nonzero(mask_flat[bt_index], as_tuple=False).squeeze(1)
            if valid_global.numel() <= 1:
                continue
            n_anchor = min(max(1, self.max_spatial_anchors), valid_global.numel())
            anchor_subset = _sample_indices(valid_global.numel(), n_anchor, Z.device)
            anchor_global = valid_global[anchor_subset]
            anchors = q_flat[bt_index, anchor_global]
            valid_embeddings = q_flat[bt_index, valid_global]
            logits = anchors @ valid_embeddings.t() / self.temperature

            global_to_local = torch.full((N,), -1, dtype=torch.long, device=Z.device)
            global_to_local[valid_global] = torch.arange(valid_global.numel(), device=Z.device)
            h_idx = anchor_global // W
            w_idx = anchor_global % W
            positive_local = []
            positive_valid = []
            for dh, dw in self._neighbor_offsets():
                nh = h_idx + dh
                nw = w_idx + dw
                in_bounds = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
                neighbor_global = nh.clamp(0, H - 1) * W + nw.clamp(0, W - 1)
                neighbor_local = global_to_local[neighbor_global]
                positive_local.append(neighbor_local.clamp_min(0).unsqueeze(-1))
                positive_valid.append((in_bounds & (neighbor_local >= 0)).unsqueeze(-1))
            positive_local_tensor = torch.cat(positive_local, dim=-1)
            positive_valid_tensor = torch.cat(positive_valid, dim=-1)
            positive_logits = torch.gather(logits, dim=1, index=positive_local_tensor)
            positive_logits = positive_logits.masked_fill(
                ~positive_valid_tensor,
                -torch.finfo(positive_logits.dtype).max,
            )
            valid_anchor = positive_valid_tensor.any(dim=-1)
            if not bool(valid_anchor.any()):
                continue
            numerator = torch.logsumexp(positive_logits[valid_anchor], dim=-1)
            denominator = torch.logsumexp(logits[valid_anchor], dim=-1)
            batch_losses.append(-(numerator - denominator).mean())
        if not batch_losses:
            return _zero_like_loss(Z)
        loss = torch.stack(batch_losses).mean()
        if self.verbose:
            print(f"[Contrast] spatial={loss.item():.4f}")
        return loss

    def cross_modal_contrast(
        self,
        Z_modalities: dict[str, torch.Tensor] | list[torch.Tensor] | tuple[torch.Tensor, ...],
        valid_region_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Align modality-specific embeddings at matching space-time positions."""
        if isinstance(Z_modalities, dict):
            names = list(Z_modalities.keys())
            tensors = [Z_modalities[name] for name in names]
        else:
            tensors = list(Z_modalities)
            names = [f"m{i}" for i in range(len(tensors))]

        if len(tensors) < 2:
            return _zero_like_loss(tensors[0]) if tensors else torch.tensor(0.0)

        base_shape = tensors[0].shape
        for tensor in tensors:
            if tensor.shape != base_shape:
                raise ValueError(f"All modality tensors must share shape; got {base_shape} and {tensor.shape}")

        flat_modalities = [rearrange(t, "b t d h w -> (b t h w) d") for t in tensors]
        spatial_mask = _coerce_valid_mask(valid_region_mask, tensors[0])
        B, T, _, H, W = tensors[0].shape
        valid_flat = rearrange(
            spatial_mask[:, None].expand(B, T, H, W),
            "b t h w -> (b t h w)",
        )
        valid_indices = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
        if valid_indices.numel() <= 1:
            return _zero_like_loss(tensors[0])
        subset = _sample_indices(valid_indices.numel(), self.max_cross_samples, flat_modalities[0].device)
        idx = valid_indices[subset]
        flat_modalities = [self.projection_head(t[idx]) for t in flat_modalities]

        losses = []
        for i in range(len(flat_modalities)):
            for j in range(i + 1, len(flat_modalities)):
                losses.append(self.infonce(flat_modalities[i], flat_modalities[j], symmetric=True))
                if self.verbose:
                    print(f"[Contrast] cross pair=({names[i]}, {names[j]})")
        loss = torch.stack(losses).mean()
        if self.verbose:
            print(f"[Contrast] cross={loss.item():.4f}")
        return loss

    def forward(
        self,
        Z: torch.Tensor,
        Z_m1: torch.Tensor | None = None,
        Z_m2: torch.Tensor | None = None,
        Z_modalities: dict[str, torch.Tensor] | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        valid_region_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if Z.ndim != 5:
            raise ValueError(f"MultiViewContrastive expects [B,T,D,H,W], got {Z.shape}")

        spatial_mask = _coerce_valid_mask(valid_region_mask, Z)
        temporal = self.temporal_contrast(Z, valid_region_mask=spatial_mask)
        spatial = self.spatial_contrast(Z, valid_region_mask=spatial_mask)

        if Z_modalities is not None:
            cross = self.cross_modal_contrast(Z_modalities, valid_region_mask=spatial_mask)
        elif Z_m1 is not None and Z_m2 is not None:
            B, T, _, H, W = Z_m1.shape
            valid_flat = rearrange(
                spatial_mask[:, None].expand(B, T, H, W),
                "b t h w -> (b t h w)",
            )
            cross = self.infonce(
                self.projection_head(rearrange(Z_m1, "b t d h w -> (b t h w) d")[valid_flat]),
                self.projection_head(rearrange(Z_m2, "b t d h w -> (b t h w) d")[valid_flat]),
                symmetric=True,
            )
        else:
            cross = _zero_like_loss(Z)

        total = self.lambda_cross * cross + self.lambda_spatial * spatial + self.lambda_temporal * temporal
        mask_weights = spatial_mask[:, None].to(Z.dtype)
        pooled = (Z[:, -1] * mask_weights).sum(dim=(2, 3))
        pooled = pooled / mask_weights.sum(dim=(2, 3)).clamp_min(1.0)
        refined = self.projection_head(pooled)
        losses = {
            "temporal": temporal,
            "spatial": spatial,
            "cross": cross,
            "total": total,
        }
        if self.verbose:
            print(
                "[Contrast] total="
                f"{total.item():.4f} cross={cross.item():.4f} "
                f"spatial={spatial.item():.4f} temporal={temporal.item():.4f}"
            )
        return refined, losses


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, H, W = 2, 5, 32, 6, 7
    Z = torch.randn(B, T, D, H, W)
    modalities = {
        "score": torch.randn(B, T, D, H, W),
        "meteo": torch.randn(B, T, D, H, W),
        "geo": torch.randn(B, T, D, H, W),
    }
    module = MultiViewContrastive(embed_dim=D, max_cross_samples=128, verbose=True)
    refined, loss_dict = module(Z, Z_modalities=modalities)
    print(refined.shape, {k: float(v.detach()) for k, v in loss_dict.items()})
