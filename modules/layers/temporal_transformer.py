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



class AdaptiveTemporalPositionalEncoding(nn.Module):
    """
    Adaptive Temporal Positional Encoding (ATPE):
        p_t = [sin(omega_1 t + phi_1), ..., sin(omega_K t + phi_K), W_p t + b_p]
    where omega/phi/W_p/b_p are learnable.
    """

    def __init__(self, dim, max_len=512, num_periodic=None):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim
        self.max_len = max_len

        if num_periodic is None:
            num_periodic = dim // 2
        num_periodic = int(max(1, min(num_periodic, dim)))
        self.num_periodic = num_periodic
        self.linear_dim = dim - num_periodic

        # Learnable periodic terms.
        self.omega = nn.Parameter(torch.randn(num_periodic))
        self.phi = nn.Parameter(torch.zeros(num_periodic))

        # Learnable non-periodic linear trend terms.
        if self.linear_dim > 0:
            self.W_p = nn.Parameter(torch.randn(self.linear_dim) * 0.01)
            self.b_p = nn.Parameter(torch.zeros(self.linear_dim))
        else:
            self.register_parameter("W_p", None)
            self.register_parameter("b_p", None)

    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            x + p_t, where p_t is ATPE [1, T, C].
        """
        check_tensor_shape(x, 3, "AdaptiveTemporalPositionalEncoding")
        B, T, C = x.shape
        if C != self.dim:
            raise ValueError(f"[ShapeMismatch] ATPE dim={self.dim}, input C={C}")
        if T > self.max_len:
            raise ValueError(f"Input sequence length {T} exceeds max_len {self.max_len}.")

        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(-1)  # [T, 1]

        omega = self.omega.to(device=x.device, dtype=x.dtype).unsqueeze(0)  # [1, K]
        phi = self.phi.to(device=x.device, dtype=x.dtype).unsqueeze(0)      # [1, K]
        periodic = torch.sin(t * omega + phi)  # [T, K]

        if self.linear_dim > 0:
            w_p = self.W_p.to(device=x.device, dtype=x.dtype).unsqueeze(0)  # [1, L]
            b_p = self.b_p.to(device=x.device, dtype=x.dtype).unsqueeze(0)  # [1, L]
            linear = t * w_p + b_p  # [T, L]
            pos = torch.cat([periodic, linear], dim=-1)  # [T, C]
        else:
            pos = periodic

        pos = pos.unsqueeze(0)  # [1, T, C]
        return x + pos


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention across time dimension for each spatial cell.
    """

    def __init__(self, dim, num_heads=4, dropout=0.1, causal=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.causal = bool(causal)

    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            out: [B, T, C]
        """
        check_tensor_shape(x, 3, "TemporalSelfAttention")
        x_norm = self.norm(x)
        attn_mask = None
        if self.causal:
            # True entries are blocked. Row t can therefore attend only to
            # keys 0..t and cannot read any future latent input.
            steps = x_norm.size(1)
            attn_mask = torch.triu(
                torch.ones(steps, steps, dtype=torch.bool, device=x_norm.device),
                diagonal=1,
            )
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_mask,
            need_weights=False,
        )
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

    def __init__(
        self,
        dim,
        num_heads=4,
        hidden_dim=512,
        dropout=0.1,
        max_len=512,
        causal_attention=True,
        verbose=False,
    ):
        super().__init__()
        self.pos_encoding = AdaptiveTemporalPositionalEncoding(dim, max_len=max_len)
        self.attn = TemporalSelfAttention(dim, num_heads, dropout, causal=causal_attention)
        self.ff = FeedForward(dim, hidden_dim, dropout)
        self.causal_attention = bool(causal_attention)
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
