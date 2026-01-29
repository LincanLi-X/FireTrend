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
    Multi-head self-attention across spatial grid for one time step.
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, H*W, C]
        Returns:
            [B, H*W, C]
        """
        check_tensor_shape(x, 3, "SpatialSelfAttention")
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        out = x + self.dropout(attn_out)

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
        self.attn = SpatialSelfAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, hidden_dim, dropout)
        self.verbose = verbose

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W]
        Returns:
            [B, T, C, H, W]
        """
        check_tensor_shape(x, 5, "SpatialTransformerLayer")

        B, T, C, H, W = x.shape
        if self.verbose:
            print(f"[SpatialTransformer] Input: B={B}, T={T}, C={C}, H={H}, W={W}")

        out_all = []

        for t in range(T):
            xt = x[:, t]  # [B, C, H, W]
            xt = self.pos_encoding(xt)  # [B, C, H, W]

            # Flatten spatial grid
            xt_flat = rearrange(xt, "b c h w -> b (h w) c")

            # Spatial attention
            xt_attn = self.attn(xt_flat)

            # Feedforward
            xt_out = self.ff(xt_attn)

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
