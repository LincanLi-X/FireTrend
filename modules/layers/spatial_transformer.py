"""
spatial_transformer.py
----------------------------------------
Spatial Transformer Layer for FireTrend.

Performs spatial self-attention across (H, W) grid for each time step T.
Model learns spatial dependencies between nearby cells.

Input shape : [B, T, C, H, W]
Output shape: [B, T, C, H, W]
"""

import torch
import torch.nn as nn
from einops import rearrange


# -----------------------------------------------------------
# 🧩 Shape Checking Utility
# -----------------------------------------------------------
def check_tensor_shape(tensor: torch.Tensor, expected_dims: int, context: str = ""):
    """
    Validate tensor dimensionality and print debug info if mismatch.
    """
    if tensor.ndim != expected_dims:
        msg = (f"\033[91m[ShapeError in {context}] Expected {expected_dims} dims, "
               f"but got {tensor.ndim}. Tensor shape = {list(tensor.shape)}\033[0m")
        raise ValueError(msg)
    return True


# -----------------------------------------------------------
# 📦 Modules
# -----------------------------------------------------------

class SpatialPositionalEncoding(nn.Module):
    """
    Learnable 2D positional encoding for spatial grid (H, W).
    Adds spatial awareness to each grid cell.
    """

    def __init__(self, dim, height, width):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, dim, height, width))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        check_tensor_shape(x, 4, "SpatialPositionalEncoding")
        if x.shape[-2:] != self.pos_emb.shape[-2:]:
            raise ValueError(f"[ShapeMismatch] Input spatial size {x.shape[-2:]} "
                             f"!= positional encoding size {self.pos_emb.shape[-2:]}")
        return x + self.pos_emb


class SpatialSelfAttention(nn.Module):
    """
    Multi-head self-attention across spatial grid for one time step with
    geospatial relative-distance/adjacency bias P_geo.
    """

    def __init__(self, dim, height, width, num_heads=4, dropout=0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.height = int(height)
        self.width = int(width)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.distance_scale = nn.Parameter(torch.tensor(0.1))
        self.adjacency_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("spatial_distance", self._build_distance(self.height, self.width), persistent=False)
        self.register_buffer("spatial_adjacency", self._build_adjacency(self.height, self.width), persistent=False)

    @staticmethod
    def _build_distance(height: int, width: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).float()
        dist = torch.cdist(coords, coords, p=2)
        return dist / dist.max().clamp_min(1.0)

    @staticmethod
    def _build_adjacency(height: int, width: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
        delta = (coords[:, None, :] - coords[None, :, :]).abs()
        return ((delta[..., 0] <= 1) & (delta[..., 1] <= 1) & (delta.sum(dim=-1) > 0)).float()

    def _ensure_spatial_bias(self, height: int, width: int, device: torch.device) -> None:
        if self.height == height and self.width == width and self.spatial_distance.device == device:
            return
        self.height = int(height)
        self.width = int(width)
        self.spatial_distance = self._build_distance(height, width).to(device)
        self.spatial_adjacency = self._build_adjacency(height, width).to(device)

    def forward(self, x, valid_region_mask=None):
        """
        Args:
            x: [B, H*W, C]
            valid_region_mask: optional boolean mask [B, H*W], where True
                marks cells that may participate as spatial keys/values
        Returns:
            [B, H*W, C]
        """
        check_tensor_shape(x, 3, "SpatialSelfAttention")
        n_cells = self.height * self.width
        if x.size(1) != n_cells:
            raise ValueError(f"Expected {n_cells} spatial cells, got {x.size(1)}")
        x_norm = self.norm(x)
        B, N, C = x_norm.shape
        valid_flat = None
        if valid_region_mask is not None:
            valid_flat = torch.as_tensor(valid_region_mask, device=x.device, dtype=torch.bool)
            if valid_flat.ndim == 1:
                valid_flat = valid_flat.unsqueeze(0).expand(B, -1)
            if tuple(valid_flat.shape) != (B, N):
                raise ValueError(
                    f"valid_region_mask must have shape {(B, N)}, got {tuple(valid_flat.shape)}"
                )
            if not bool(valid_flat.any(dim=1).all()):
                raise ValueError("Every sample must contain at least one valid spatial cell.")
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        p_geo = (
            -torch.relu(self.distance_scale) * self.spatial_distance
            + torch.relu(self.adjacency_scale) * self.spatial_adjacency
        ).to(dtype=logits.dtype, device=logits.device)
        logits = logits + p_geo.view(1, 1, N, N)
        if valid_flat is not None:
            # Invalid cells cannot contribute as attention keys or values.
            logits = logits.masked_fill(
                ~valid_flat[:, None, None, :],
                torch.finfo(logits.dtype).min,
            )
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj(attn_out)
        out = x + self.dropout(attn_out)
        if valid_flat is not None:
            # Invalid query positions are not part of the modeled state space.
            out = out * valid_flat.unsqueeze(-1).to(out.dtype)

        assert out.shape == x.shape, (
            f"[ShapeMismatch] SpatialSelfAttention output {out.shape} != input {x.shape}"
        )
        return out


class FeedForward(nn.Module):
    """
    Position-wise feedforward network (same as temporal counterpart).
    """

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, N, C]
        Returns:
            [B, N, C]
        """
        check_tensor_shape(x, 3, "FeedForward")
        x_norm = self.norm(x)
        out = x + self.net(x_norm)
        assert out.shape == x.shape, (
            f"[ShapeMismatch] FeedForward output {out.shape} != input {x.shape}"
        )
        return out


class SpatialTransformerLayer(nn.Module):
    """
    FireTrend Spatial Transformer block.

    Applies spatial self-attention for each temporal slice independently.
    """

    def __init__(self, dim, height, width, num_heads=4, hidden_dim=512, dropout=0.1, verbose=False):
        super().__init__()
        self.pos_encoding = SpatialPositionalEncoding(dim, height, width)
        self.attn = SpatialSelfAttention(dim, height, width, num_heads, dropout)
        self.ff = FeedForward(dim, hidden_dim, dropout)
        self.verbose = verbose

    def forward(self, x, valid_region_mask=None):
        """
        Args:
            x: [B, T, C, H, W]
            valid_region_mask: optional boolean mask [B,H,W] or [H,W]
        Returns:
            [B, T, C, H, W]
        """
        check_tensor_shape(x, 5, "SpatialTransformerLayer")

        B, T, C, H, W = x.shape
        spatial_mask = None
        mask_flat = None
        if valid_region_mask is not None:
            spatial_mask = torch.as_tensor(valid_region_mask, device=x.device, dtype=torch.bool)
            if spatial_mask.ndim == 2:
                spatial_mask = spatial_mask.unsqueeze(0).expand(B, -1, -1)
            if tuple(spatial_mask.shape) != (B, H, W):
                raise ValueError(
                    f"valid_region_mask must have shape {(B, H, W)}, got {tuple(spatial_mask.shape)}"
                )
            mask_flat = spatial_mask.reshape(B, H * W)
        if self.verbose:
            print(f"[SpatialTransformer] Input: B={B}, T={T}, C={C}, H={H}, W={W}")

        out_all = []

        for t in range(T):
            xt = x[:, t]  # [B, C, H, W]
            xt = self.pos_encoding(xt)  # [B, C, H, W]
            if spatial_mask is not None:
                xt = xt * spatial_mask[:, None].to(xt.dtype)
            self.attn._ensure_spatial_bias(H, W, xt.device)

            # Flatten spatial grid
            xt_flat = rearrange(xt, "b c h w -> b (h w) c")

            # Spatial attention
            xt_attn = self.attn(xt_flat, valid_region_mask=mask_flat)

            # Feedforward
            xt_out = self.ff(xt_attn)
            if mask_flat is not None:
                xt_out = xt_out * mask_flat.unsqueeze(-1).to(xt_out.dtype)

            # Reshape back
            xt_out = rearrange(xt_out, "b (h w) c -> b c h w", h=H, w=W)
            out_all.append(xt_out.unsqueeze(1))  # add temporal dim

        # Stack over time dimension
        out = torch.cat(out_all, dim=1)

        assert out.shape == (B, T, C, H, W), (
            f"[ShapeMismatch] Expected output {B,T,C,H,W}, got {out.shape}"
        )

        if self.verbose:
            print(f"[SpatialTransformer] Output: {list(out.shape)}")

        return out


# ---------------------------------------------------------------
# 🧪 Unit Test
# ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, H, W = 2, 4, 64, 8, 8
    x = torch.randn(B, T, C, H, W)

    layer = SpatialTransformerLayer(
        dim=C,
        height=H,
        width=W,
        num_heads=4,
        hidden_dim=256,
        verbose=True
    )
    out = layer(x)
    print(f"✔️ Forward successful: {out.shape}")
