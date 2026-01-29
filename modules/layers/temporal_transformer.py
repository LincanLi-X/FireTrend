"""
temporal_transformer.py
----------------------------------------
Temporal Transformer Layer for FireTrend.

Performs intra-cell temporal attention across time dimension T
for each spatial position (H, W).

Input shape : [B, T, C, H, W]
Output shape: [B, T, C, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------------------------------------
# Shape Checking Utility
# -----------------------------------------------------------
def check_tensor_shape(tensor: torch.Tensor, expected_dims: int, context: str = ""):
    """
    Validate tensor dimensionality and print debug info if mismatch.
    Args:
        tensor (torch.Tensor): Tensor to check.
        expected_dims (int): Expected number of dimensions.
        context (str): Description for debug messages.
    """
    if tensor.ndim != expected_dims:
        msg = (f"\033[91m[ShapeError in {context}] Expected {expected_dims} dims, "
               f"but got {tensor.ndim}. Tensor shape = {list(tensor.shape)}\033[0m")
        raise ValueError(msg)
    return True



class TemporalPositionalEncoding(nn.Module):
    """
    Learnable temporal positional encoding.
    Adds temporal order awareness to each timestep.
    """

    def __init__(self, dim, max_len=512):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            x + positional_encoding[:, :T, :]
        """
        check_tensor_shape(x, 3, "TemporalPositionalEncoding")
        T = x.size(1)
        if T > self.pos_emb.size(1):
            raise ValueError(f"Input sequence length {T} exceeds max_len {self.pos_emb.size(1)}.")
        return x + self.pos_emb[:, :T, :]


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention across time dimension for each spatial cell.
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            out: [B, T, C]
        """
        check_tensor_shape(x, 3, "TemporalSelfAttention")
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        out = x + self.dropout(attn_out)

        # Shape validation
        assert out.shape == x.shape, (
            f"[ShapeMismatch] TemporalSelfAttention output {out.shape} != input {x.shape}"
        )
        return out


class FeedForward(nn.Module):
    """
    Position-wise feedforward network used after attention.
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
            x: [B, T, C]
        Returns:
            [B, T, C]
        """
        check_tensor_shape(x, 3, "FeedForward")
        x_norm = self.norm(x)
        out = x + self.net(x_norm)

        assert out.shape == x.shape, (
            f"[ShapeMismatch] FeedForward output {out.shape} != input {x.shape}"
        )
        return out


class TemporalTransformerLayer(nn.Module):
    """
    FireTrend Temporal Transformer block.
    Applies temporal self-attention for each spatial grid cell independently.
    """

    def __init__(self, dim, num_heads=4, hidden_dim=512, dropout=0.1, max_len=512, verbose=False):
        super().__init__()
        self.pos_encoding = TemporalPositionalEncoding(dim, max_len=max_len)
        self.attn = TemporalSelfAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, hidden_dim, dropout)
        self.verbose = verbose

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W]
        Returns:
            out: [B, T, C, H, W]
        """
        check_tensor_shape(x, 5, "TemporalTransformerLayer")

        B, T, C, H, W = x.shape
        if self.verbose:
            print(f"[TemporalTransformer] Input: B={B}, T={T}, C={C}, H={H}, W={W}")

        # Flatten spatial dimensions
        x_reshaped = rearrange(x, "b t c h w -> (b h w) t c")

        # Positional encoding
        x_reshaped = self.pos_encoding(x_reshaped)

        # Temporal self-attention
        x_attn = self.attn(x_reshaped)

        # Feedforward block
        out = self.ff(x_attn)

        # Reshape back
        out = rearrange(out, "(b h w) t c -> b t c h w", b=B, h=H, w=W)

        # Final dimension check
        assert out.shape == (B, T, C, H, W), (
            f"[ShapeMismatch] Expected output {B,T,C,H,W}, got {out.shape}"
        )

        if self.verbose:
            print(f"[TemporalTransformer] Output: {list(out.shape)} ✅")

        return out


# ---------------------------------------------------------------
# Unit Test
# ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, H, W = 2, 6, 64, 8, 8
    x = torch.randn(B, T, C, H, W)
    layer = TemporalTransformerLayer(dim=C, num_heads=4, hidden_dim=256, verbose=True)
    out = layer(x)
    print(f"✔️ Forward successful: {out.shape}")
