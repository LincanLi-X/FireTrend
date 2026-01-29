"""
PyroCastPhysics — Patch-wise Physics-guided Fire Spread Module (Vectorized)
Efficient directional wildfire spread modeling using vectorized patch-wise anisotropic convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# Utility
# -----------------------------------------------------------
def check_tensor_shape(tensor, expected_dims, context=""):
    if tensor.ndim != expected_dims:
        raise ValueError(
            f"[ShapeError:{context}] Expected {expected_dims}D, got {list(tensor.shape)}"
        )


# -----------------------------------------------------------
# Vectorized Patch-wise Directional Convolution
# -----------------------------------------------------------
class PatchDirectionalConv2D(nn.Module):
    """
    Vectorized patch-wise directional convolution.
    Each P×P patch shares one wind-oriented convolution kernel.
    """

    def __init__(self, kernel_size=5, patch_size=3, verbose=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.pad = kernel_size // 2
        self.verbose = verbose

        # precompute kernel coordinate grid (static)
        p = self.pad
        y, x = torch.meshgrid(
            torch.arange(-p, p + 1),
            torch.arange(-p, p + 1),
            indexing="ij",
        )
        self.register_buffer("grid_x", x.float() / (p + 1))
        self.register_buffer("grid_y", y.float() / (p + 1))
        self.register_buffer(
            "distance_decay",
            torch.exp(-(self.grid_x ** 2 + self.grid_y ** 2) / 1.5),
        )

    def forward(self, Y_pred, wind_u, wind_v):
        check_tensor_shape(Y_pred, 4, "PatchConv:Y_pred")
        check_tensor_shape(wind_u, 4, "PatchConv:wind_u")
        check_tensor_shape(wind_v, 4, "PatchConv:wind_v")

        B, _, H, W = Y_pred.shape
        P = self.patch_size
        k = self.kernel_size

        # ---------------------------------------------------
        # 1. Pad input so H,W divisible by P
        # ---------------------------------------------------
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P

        Y_pad = F.pad(Y_pred, (0, pad_w, 0, pad_h), mode="replicate")
        u_pad = F.pad(wind_u, (0, pad_w, 0, pad_h), mode="replicate")
        v_pad = F.pad(wind_v, (0, pad_w, 0, pad_h), mode="replicate")

        Hp, Wp = Y_pad.shape[-2:]

        # ---------------------------------------------------
        # 2. Extract patches (KEEP 2D PATCH GRID)
        # ---------------------------------------------------
        # [B, 1, Hp//P, Wp//P, P, P]
        patches = Y_pad.unfold(2, P, P).unfold(3, P, P)
        B0, C0, Hp_, Wp_, _, _ = patches.shape

        # ---------------------------------------------------
        # 3. Compute mean wind per patch
        # ---------------------------------------------------
        u_patch = u_pad.unfold(2, P, P).unfold(3, P, P)
        v_patch = v_pad.unfold(2, P, P).unfold(3, P, P)

        u_mean = u_patch.mean(dim=(-1, -2))
        v_mean = v_patch.mean(dim=(-1, -2))

        angle = torch.atan2(v_mean, u_mean)  # [B, Hp_, Wp_]

        # ---------------------------------------------------
        # 4. Build directional kernels (vectorized)
        # ---------------------------------------------------
        cos_t = torch.cos(angle).view(B0, Hp_, Wp_, 1, 1)
        sin_t = torch.sin(angle).view(B0, Hp_, Wp_, 1, 1)

        kernel = (
            cos_t * self.grid_x + sin_t * self.grid_y
        ) * self.distance_decay
        kernel = F.relu(kernel)
        kernel = kernel / (kernel.sum(dim=(-1, -2), keepdim=True) + 1e-6)

        # reshape for batched conv
        kernel = kernel.view(B0 * Hp_ * Wp_, 1, k, k)

        # ---------------------------------------------------
        # 5. Convolution (single batched call)
        # ---------------------------------------------------
        # reshape patches into channel dimension
        patches = patches.contiguous().view(1,B0 * Hp_ * Wp_,P,P,)

        patches = F.pad(
            patches,
            (self.pad, self.pad, self.pad, self.pad),
            mode="replicate",
        )

        out = F.conv2d(
            patches,
            kernel,
            groups=B0 * Hp_ * Wp_
        )

        # reshape back to per-patch layout
        out = out.view(B0 * Hp_ * Wp_, 1, P, P)
        # ✅ 注意：这里不再做任何 crop


        # ---------------------------------------------------
        # 6. Fold back to image (CORRECT)
        # ---------------------------------------------------
        out = out.view(B0, Hp_, Wp_, 1, P, P)

        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(B0, 1, Hp_ * P, Wp_ * P)

        return out[:, :, :H, :W]


# -----------------------------------------------------------
# Main PyroCast Module
# -----------------------------------------------------------
class PyroCastPhysics(nn.Module):
    """
    Vectorized Patch-wise Physics-guided wildfire spread adjustment.
    """

    def __init__(
        self,
        kernel_size=5,
        patch_size=3,
        blend_beta=0.3,
        verbose=False,
    ):
        super().__init__()
        self.conv = PatchDirectionalConv2D(
            kernel_size=kernel_size,
            patch_size=patch_size,
            verbose=verbose,
        )
        self.blend_beta = blend_beta
        self.verbose = verbose

    def forward(self, Y_pred, wind_u, wind_v):
        check_tensor_shape(Y_pred, 4, "PyroCast:Y_pred")
        check_tensor_shape(wind_u, 4, "PyroCast:wind_u")
        check_tensor_shape(wind_v, 4, "PyroCast:wind_v")

        spread = self.conv(Y_pred, wind_u, wind_v)

        # energy normalization
        energy = spread.sum(dim=(2, 3), keepdim=True)
        spread = spread / (energy + 1e-6) * Y_pred.sum(dim=(2, 3), keepdim=True)

        Y_final = (1 - self.blend_beta) * Y_pred + self.blend_beta * spread

        if self.verbose:
            print(f"[PyroCast] Output mean={Y_final.mean().item():.4f}")

        return Y_final


# -----------------------------------------------------------
# Unit Test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, W = 2, 49, 53
    Y = torch.rand(B, 1, H, W)
    u = torch.randn(B, 1, H, W)
    v = torch.randn(B, 1, H, W)

    pyro = PyroCastPhysics(kernel_size=5, patch_size=3, verbose=True)
    out = pyro(Y, u, v)

    print("Test passed:", out.shape)
