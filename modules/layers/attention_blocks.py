"""
attention_blocks.py
----------------------------------------
General-purpose attention and normalization blocks for FireTrend.

Includes:
- MultiHeadAttentionBlock: standard MHA
- CrossAttentionBlock: for multimodal alignment
- ResidualAddNorm: residual + layer normalization
- check_tensor_shape: shape validation utility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# 🧠 Residual + Normalization Block
# -----------------------------------------------------------
class ResidualAddNorm(nn.Module):
    """
    Residual connection + LayerNorm wrapper.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_out):
        """
        Args:
            x: input tensor [B, N, C]
            sublayer_out: output of the sublayer [B, N, C]
        """
        check_tensor_shape(x, 3, "ResidualAddNorm:x")
        check_tensor_shape(sublayer_out, 3, "ResidualAddNorm:sublayer_out")

        assert x.shape == sublayer_out.shape, (
            f"[ShapeMismatch] ResidualAddNorm input {x.shape} != sublayer output {sublayer_out.shape}"
        )

        return x + self.dropout(self.norm(sublayer_out))


# -----------------------------------------------------------
# 🧩 Multi-Head Self-Attention
# -----------------------------------------------------------
class MultiHeadAttentionBlock(nn.Module):
    """
    Generic Multi-Head Attention block.
    Compatible with both spatial and temporal encoders.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, C]
            mask: optional attention mask [B, N, N]
        Returns:
            [B, N, C]
        """
        check_tensor_shape(x, 3, "MultiHeadAttentionBlock")

        B, N, C = x.shape
        assert C == self.dim, f"[ShapeError] Channel mismatch: expected {self.dim}, got {C}"

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to [B, num_heads, N, head_dim]
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention weights
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            check_tensor_shape(mask, 3, "MultiHeadAttentionBlock:mask")
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, V)

        # Restore shape [B, N, C]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)

        # Final shape check
        assert out.shape == (B, N, C), (
            f"[ShapeMismatch] Expected output {(B, N, C)}, got {out.shape}"
        )

        return out


# -----------------------------------------------------------
# 🧩 Cross-Attention Block
# -----------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between query (Q) and context (K, V).
    Used for multimodal alignment (e.g., satellite <-> meteorology).
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5

    def forward(self, q, kv):
        """
        Args:
            q: [B, Nq, C]
            kv: [B, Nk, C]
        Returns:
            [B, Nq, C]
        """
        check_tensor_shape(q, 3, "CrossAttentionBlock:q")
        check_tensor_shape(kv, 3, "CrossAttentionBlock:kv")

        B, Nq, Cq = q.shape
        B2, Nk, Ck = kv.shape

        assert B == B2, "[ShapeMismatch] Q and KV batch size mismatch"
        assert Cq == Ck == self.dim, "[ShapeMismatch] Feature dimension mismatch"

        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        # [B, heads, N, head_dim]
        Q = Q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.dim)
        out = self.out_proj(out)

        assert out.shape == (B, Nq, self.dim), (
            f"[ShapeMismatch] Expected output {(B, Nq, self.dim)}, got {out.shape}"
        )

        return out


# -----------------------------------------------------------
# 🧪 Unit Test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, C = 2, 16, 64
    mha = MultiHeadAttentionBlock(dim=C, num_heads=8)
    x = torch.randn(B, N, C)
    out = mha(x)
    print("✅ MultiHeadAttentionBlock:", out.shape)

    cross = CrossAttentionBlock(dim=C, num_heads=8)
    q = torch.randn(B, 10, C)
    kv = torch.randn(B, 20, C)
    out2 = cross(q, kv)
    print("✅ CrossAttentionBlock:", out2.shape)

    # Residual test
    res = ResidualAddNorm(dim=C)
    y = res(x, out)
    print("✅ ResidualAddNorm:", y.shape)
