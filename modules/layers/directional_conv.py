"""
directional_conv.py
----------------------------------------
Directional Convolution (PyroCast physics-inspired block) for FireTrend.

This module applies dynamic, direction-aware convolution
based on local wind fields (u, v). It mimics physical fire spread
in downwind directions.

Input:
    fire_map : [B, 1, H, W]  - wildfire risk map or fire mask
    wind_u   : [B, 1, H, W]  - east-west wind component
    wind_v   : [B, 1, H, W]  - north-south wind component

Output:
    spread_map: [B, 1, H, W] - physically guided fire propagation map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
# 🧠 Directional Convolution Layer
# -----------------------------------------------------------
class DirectionalConv2D(nn.Module):
    """
    Physics-inspired directional convolution.

    This module generates a spatially varying convolution kernel
    according to local wind direction and applies it to the fire map.
    """

    def __init__(self, kernel_size=5, sigma=1.5, blend_beta=0.3, padding_mode="replicate", verbose=False):
        """
        Args:
            kernel_size: base convolution kernel size (odd number)
            sigma: standard deviation for Gaussian base kernel
            blend_beta: residual blending factor (mix with input)
            padding_mode: padding behavior ('replicate' recommended)
            verbose: if True, print debug info
        """
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.blend_beta = blend_beta
        self.padding_mode = padding_mode
        self.verbose = verbose

        # Precompute Gaussian base kernel grid
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        base_gauss = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        self.register_buffer("base_kernel", base_gauss / base_gauss.sum())

    def compute_direction_kernel(self, wind_u, wind_v):
        """
        Compute direction-aware convolution kernel based on wind angle.

        Args:
            wind_u, wind_v: [B, 1, H, W]
        Returns:
            direction_kernel: [B, kernel_size, kernel_size, H, W]
        """

        check_tensor_shape(wind_u, 4, "DirectionalConv2D:wind_u")
        check_tensor_shape(wind_v, 4, "DirectionalConv2D:wind_v")

        # Step 1. Compute wind direction angle θ = atan2(v, u)
        theta = torch.atan2(wind_v, wind_u + 1e-8)  # radians, [-pi, pi]
        if self.verbose:
            print(f"[PyroCast] Computed wind angles θ ∈ [{theta.min().item():.2f}, {theta.max().item():.2f}]")

        # Step 2. Create base coordinate grid
        k = self.kernel_size
        ax = torch.arange(k, device=wind_u.device) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        xx, yy = xx.float(), yy.float()

        # Step 3. Compute directional weighting
        # Rotate base Gaussian according to wind direction
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        # For each pixel (x, y): project kernel offset along wind direction
        # W = exp(-((x*cosθ + y*sinθ)^2) / (2σ²))
        proj = (xx[None, None, :, :, None, None] * cos_t[:, :, None, None, :, :] +
                yy[None, None, :, :, None, None] * sin_t[:, :, None, None, :, :])
        weight = torch.exp(-(proj ** 2) / (2 * self.sigma ** 2))

        # Normalize per pixel
        weight = weight / (weight.sum(dim=(2, 3), keepdim=True) + 1e-8)

        # Remove dummy channel dimensions → [B, k, k, H, W]
        return weight.squeeze(1).squeeze(1)

    def forward(self, fire_map, wind_u, wind_v):
        """
        Args:
            fire_map: [B, 1, H, W]
            wind_u, wind_v: [B, 1, H, W]
        Returns:
            spread_map: [B, 1, H, W]
        """
        check_tensor_shape(fire_map, 4, "DirectionalConv2D:fire_map")
        B, C, H, W = fire_map.shape
        assert C == 1, "[ShapeError] fire_map must have 1 channel"

        # Step 1. Generate directional kernel for each location
        direction_kernel = self.compute_direction_kernel(wind_u, wind_v)  # [B, k, k, H, W]

        # Step 2. Unfold fire_map into local patches
        fire_patches = F.unfold(fire_map, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        # Shape: [B, k*k, H*W]
        fire_patches = fire_patches.view(B, self.kernel_size, self.kernel_size, H, W)

        # ✅ Step 3. Direction-weighted convolution (fix)
        spread_map = (fire_patches * direction_kernel).sum(dim=(1, 2))  # [B, H, W]
        spread_map = spread_map.unsqueeze(1)  # ✅ ensure [B, 1, H, W]

        # Step 4. Residual blending (physical correction + stability)
        spread_map = (1 - self.blend_beta) * spread_map + self.blend_beta * fire_map

        # Step 5. Clamp to [0, 1]
        spread_map = torch.clamp(spread_map, 0.0, 1.0)

        # Step 6. Shape check
        assert spread_map.shape == fire_map.shape, (
            f"[ShapeMismatch] Expected {fire_map.shape}, got {spread_map.shape}"
        )

        if self.verbose:
            print(f"[PyroCast] Output shape {spread_map.shape}, mean={spread_map.mean().item():.4f}")

        return spread_map


# -----------------------------------------------------------
# 🧪 Unit Test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 16, 16
    fire = torch.rand(B, 1, H, W)
    wind_u = torch.randn(B, 1, H, W)
    wind_v = torch.randn(B, 1, H, W)

    dconv = DirectionalConv2D(kernel_size=5, sigma=1.5, blend_beta=0.3, verbose=True)
    out = dconv(fire, wind_u, wind_v)
    print(f"✅ DirectionalConv2D output shape: {out.shape}")
