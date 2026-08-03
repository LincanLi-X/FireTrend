"""
feedforward_norm.py
----------------------------------------
FeedForward and Normalization blocks for FireTrend.

This module provides reusable FFN and normalization layers
for transformer-based spatiotemporal modeling.

Input:
    [B, N, C] or [B, C, H, W]
Output:
    Same shape as input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# 🧩 Shape Checking Utility
# -----------------------------------------------------------
def check_tensor_shape(tensor: torch.Tensor, expected_min_dims: int, context: str = ""):
    """
    Validate tensor dimensionality and print debug info if mismatch.
    Allows flexible shape (e.g., 3D for sequence, 4D for image).
    """
    if tensor.ndim < expected_min_dims:
        msg = (f"\033[91m[ShapeError in {context}] Expected ≥{expected_min_dims} dims, "
               f"but got {tensor.ndim}. Tensor shape = {list(tensor.shape)}\033[0m")
        raise ValueError(msg)
    return True


# -----------------------------------------------------------
# Normalization Layer
# -----------------------------------------------------------
class NormalizationLayer(nn.Module):
    """
    General normalization layer supporting:
    - LayerNorm (sequence)
    - BatchNorm2d (spatial)
    - GroupNorm (custom grouping)
    """

    def __init__(self, dim, mode="layernorm", num_groups=8, eps=1e-6):
        """
        Args:
            dim (int): feature dimension
            mode (str): ['layernorm', 'batchnorm', 'groupnorm']
            num_groups (int): used only for groupnorm
        """
        super().__init__()
        mode = mode.lower()
        self.mode = mode

        if mode == "layernorm":
            self.norm = nn.LayerNorm(dim, eps=eps)
        elif mode == "batchnorm":
            self.norm = nn.BatchNorm2d(dim, eps=eps)
        elif mode == "groupnorm":
            self.norm = nn.GroupNorm(num_groups, dim, eps=eps)
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")

    def forward(self, x):
        check_tensor_shape(x, 3, f"NormalizationLayer[{self.mode}]")

        if self.mode == "layernorm":
            # Expect [B, N, C]
            assert x.ndim == 3, f"LayerNorm expects 3D input, got {x.shape}"
            out = self.norm(x)
        elif self.mode == "batchnorm":
            # Expect [B, C, H, W]
            assert x.ndim == 4, f"BatchNorm2d expects 4D input, got {x.shape}"
            out = self.norm(x)
        elif self.mode == "groupnorm":
            # Expect [B, C, *]
            assert x.ndim >= 3, f"GroupNorm expects ≥3D input, got {x.shape}"
            out = self.norm(x)
        else:
            raise RuntimeError(f"Invalid normalization mode: {self.mode}")

        assert out.shape == x.shape, (
            f"[ShapeMismatch] Normalization output {out.shape} != input {x.shape}"
        )

        return out


# -----------------------------------------------------------
# FeedForward Block
# -----------------------------------------------------------
class FeedForwardBlock(nn.Module):
    """
    Feed-forward network with normalization and residual connection.

    Compatible with both transformer sequence input [B, N, C]
    and spatial tensor [B, C, H, W].
    """

    def __init__(
        self,
        dim,
        hidden_dim=None,
        dropout=0.1,
        activation="gelu",
        norm_mode="layernorm",
        use_residual=True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (dim * 4)
        self.dim = dim
        self.use_residual = use_residual

        # Feedforward layers
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Normalization layer
        self.norm = NormalizationLayer(dim, mode=norm_mode)

    def forward(self, x):
        check_tensor_shape(x, 3, "FeedForwardBlock")

        residual = x

        # LayerNorm before feedforward (Pre-Norm style)
        x = self.norm(x)

        # Apply linear → activation → dropout → linear
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # Residual connection
        if self.use_residual:
            assert x.shape == residual.shape, (
                f"[ShapeMismatch] FFN residual mismatch: {x.shape} vs {residual.shape}"
            )
            x = x + residual

        return x


# -----------------------------------------------------------
# Unit Test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # === Test 1: Sequence input ===
    B, N, C = 2, 8, 64
    seq = torch.randn(B, N, C)
    ffn = FeedForwardBlock(dim=C, norm_mode="layernorm", dropout=0.1)
    out = ffn(seq)
    print("✅ Sequence FFN:", out.shape)

    # === Test 2: Spatial input with BatchNorm ===
    B, C, H, W = 2, 64, 16, 16
    spatial = torch.randn(B, C, H, W)
    norm = NormalizationLayer(dim=C, mode="batchnorm")
    out2 = norm(spatial)
    print("✅ Spatial BatchNorm:", out2.shape)

    # === Test 3: GroupNorm ===
    norm3 = NormalizationLayer(dim=C, mode="groupnorm")
    out3 = norm3(spatial)
    print("✅ GroupNorm:", out3.shape)
