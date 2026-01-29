"""
PyroCastPhysics — FireTrend Stage 3
Physics-guided wildfire spread adjustment based on wind direction and speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# Utility: Shape Checking
# -----------------------------------------------------------
def check_tensor_shape(tensor: torch.Tensor, expected_dims: int, context: str = ""):
    if tensor.ndim != expected_dims:
        msg = (
            f"\033[91m[ShapeError in {context}] Expected {expected_dims} dims, "
            f"but got {tensor.ndim}. Tensor shape = {list(tensor.shape)}\033[0m"
        )
        raise ValueError(msg)
    return True


# -----------------------------------------------------------
# Directional Fire Spread Convolution
# -----------------------------------------------------------
class DirectionalConv2D(nn.Module):
    """
    Simulates fire spread influenced by wind direction using dynamic convolutional kernels.
    """

    def __init__(self, kernel_size=5, verbose=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.verbose = verbose
        self.pad_size = kernel_size // 2

    # def forward(self, Y_pred, wind_u, wind_v):
    #     """
    #     Args:
    #         Y_pred: [B, 1, H, W] — predicted fire risk map
    #         wind_u: [B, 1, H, W] — u-component of wind
    #         wind_v: [B, 1, H, W] — v-component of wind
    #     Returns:
    #         spread_map: [B, 1, H, W] — wind-influenced spread map
    #     """
    #     check_tensor_shape(Y_pred, 4, "DirectionalConv:Y_pred")
    #     check_tensor_shape(wind_u, 4, "DirectionalConv:wind_u")
    #     check_tensor_shape(wind_v, 4, "DirectionalConv:wind_v")

    #     B, _, H, W = Y_pred.shape
    #     k = self.kernel_size
    #     padding = k // 2

    #     # Generate dynamic kernel based on wind direction
    #     direction_kernel = self._generate_directional_kernel(wind_u, wind_v)

    #     # Apply grouped conv
    #     Y_expanded = Y_pred.view(B, 1, H, W)
    #     spread_map = torch.zeros_like(Y_expanded)

    #     for i in range(B):
    #         # each kernel_i now is [k, k]
    #         kernel_i = direction_kernel[i]
    #         kernel_i = kernel_i / (kernel_i.sum() + 1e-6)
    #         kernel_i = kernel_i.unsqueeze(0).unsqueeze(0)  # → [1,1,k,k]
    #         spread_map[i, 0] = F.conv2d(
    #             Y_expanded[i : i + 1], kernel_i, padding=padding
    #         )

    #     if self.verbose:
    #         print(f"[DirectionalConv2D] Spread_map shape = {spread_map.shape}")

    #     return spread_map

    def forward(self, Y_pred, wind_u, wind_v):
        """
        Forward wildfire spread simulation step.
        Args:
            Y_pred: [B, 1, H, W] predicted fire mask
            wind_u, wind_v: [B, 1, H, W]
        Returns:
            spread_map: [B, 1, H, W]
        """
        B, _, H, W = Y_pred.shape
        k = self.kernel_size
        padding = self.pad_size

        # --- 生成方向相关 kernel ---
        direction_kernel = self._generate_directional_kernel(wind_u, wind_v)  # [B, k, k, H, W]

        spread_map = torch.zeros_like(Y_pred)

        for i in range(B):
            # 对该 batch 求平均方向 kernel → [k, k]
            kernel_i = direction_kernel[i].mean(dim=(2, 3))
            kernel_i = kernel_i / (kernel_i.sum() + 1e-6)

            # 调整形状为 [1, 1, k, k] 供 conv2d 使用
            kernel_i = kernel_i.unsqueeze(0).unsqueeze(0)

            # 执行方向卷积
            spread_map[i : i + 1] = F.conv2d(
                Y_pred[i : i + 1],
                kernel_i,
                padding=padding,
            )

        return spread_map

    ### 原版的 def _generate_directional_kernal()
    # def _generate_directional_kernel(self, wind_u, wind_v):
    #     """
    #     Generates an anisotropic kernel oriented by wind direction.
    #     """
    #     check_tensor_shape(wind_u, 4, "PyroCast:wind_u")
    #     check_tensor_shape(wind_v, 4, "PyroCast:wind_v")

    #     B, _, H, W = wind_u.shape
    #     k = self.kernel_size
    #     center = k // 2

    #     # Compute wind direction (angle in radians)
    #     theta = torch.atan2(wind_v, wind_u + 1e-6)
    #     theta = torch.clamp(theta, -3.14, 3.14)

    #     # Precompute spatial coordinates for kernel
    #     y, x = torch.meshgrid(
    #         torch.arange(k, device=wind_u.device),
    #         torch.arange(k, device=wind_u.device),
    #         indexing="ij",
    #     )
    #     dx = x - center
    #     dy = y - center
    #     distance = torch.sqrt(dx**2 + dy**2) + 1e-6
    #     angle_map = torch.atan2(dy, dx + 1e-6)

    #     # Build kernel weights (cosine similarity between wind direction and spread direction)
    #     cos_weight = torch.cos(
    #         angle_map.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    #         - theta.unsqueeze(1).unsqueeze(2)
    #     )
    #     distance_decay = torch.exp(
    #         -distance.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) / center
    #     )
    #     direction_kernel = F.relu(cos_weight) * distance_decay

    #     # safer squeeze
    #     if direction_kernel.dim() == 6:
    #         direction_kernel = direction_kernel.squeeze(1)

    #     # relaxed shape validation
    #     if direction_kernel.shape[-3] != k:
    #         raise ValueError(
    #             f"[ShapeMismatch] Direction kernel shape mismatch: got {direction_kernel.shape}"
    #         )

    #     if self.verbose:
    #         print(
    #             f"[PyroCast] Generated kernel θ∈[{theta.min():.2f}, {theta.max():.2f}]"
    #         )

    #     return direction_kernel

    def _generate_directional_kernel(self, wind_u, wind_v):
        """
        Generate directional convolution kernels based on wind direction.
        Args:
            wind_u, wind_v: [B, C, H, W]
        Returns:
            direction_kernel: [B, k, k, H, W]
        """
        # --- Extract valid wind components ---
        if wind_u.shape[1] > 1:
            wind_u = wind_u[:, 0:1, :, :]
        if wind_v.shape[1] > 1:
            wind_v = wind_v[:, 1:2, :, :]

        B, _, H, W = wind_u.shape
        k = self.kernel_size
        pad = self.pad_size

        angle_map = torch.atan2(wind_v, wind_u)  # [B, 1, H, W]

        device = wind_u.device
        y, x = torch.meshgrid(
            torch.arange(-pad, pad + 1, device=device),
            torch.arange(-pad, pad + 1, device=device),
            indexing="ij"
        )
        grid_x = x / (pad + 1)
        grid_y = y / (pad + 1)
        distance_decay = torch.exp(-(grid_x ** 2 + grid_y ** 2) / 1.5)

        mean_angle = angle_map.mean(dim=(2, 3), keepdim=True)
        cos_t = torch.cos(mean_angle).view(B, 1, 1)
        sin_t = torch.sin(mean_angle).view(B, 1, 1)

        cos_weight = cos_t * grid_x + sin_t * grid_y
        cos_weight = cos_weight * distance_decay
        cos_weight = cos_weight.unsqueeze(-1).unsqueeze(-1).expand(B, k, k, H, W)

        direction_kernel = F.relu(cos_weight)

        if B > 1:
            print(f"[PyroCast] Generated kernel θ∈[{angle_map.min():.2f}, {angle_map.max():.2f}]")

        return direction_kernel

# -----------------------------------------------------------
# Main PyroCastPhysics Module
# -----------------------------------------------------------
class PyroCastPhysics(nn.Module):
    """
    Physics-informed wildfire spread adjustment module.
    Incorporates wind direction, energy conservation, and blending.
    """

    def __init__(self, kernel_size=5, blend_beta=0.3, use_physics_adjust=True, verbose=False):
        super().__init__()
        self.directional_conv = DirectionalConv2D(kernel_size=kernel_size, verbose=verbose)
        self.kernel_size = kernel_size
        self.blend_beta = blend_beta
        self.use_physics_adjust = use_physics_adjust
        self.verbose = verbose

    def _physics_adjustment(self, spread_map):
        """
        Regularizes the spread map to preserve energy and prevent blow-up.
        """
        total_energy = spread_map.sum(dim=[2, 3], keepdim=True)
        if torch.any(total_energy < 1e-8):
            total_energy = total_energy + 1e-6

        spread_map = spread_map / total_energy * spread_map.size(2) * spread_map.size(3)
        spread_map = torch.clamp(spread_map, 0, 1)

        if self.verbose:
            print(f"[PyroCast] Energy adjusted spread_map, mean={spread_map.mean():.4f}")

        return spread_map

    def forward(self, Y_pred, wind_u, wind_v):
        """
        Args:
            Y_pred: [B, 1, H, W]
            wind_u: [B, 1, H, W]
            wind_v: [B, 1, H, W]
        Returns:
            Y_final: [B, 1, H, W]
        """
        if wind_u.ndim == 5:
            wind_u = wind_u.squeeze(1)
        if wind_v.ndim == 5:
            wind_v = wind_v.squeeze(1)

        # Auto-handle extra dim (defensive)
        if Y_pred.ndim == 5:
            Y_pred = Y_pred.squeeze(1)

        check_tensor_shape(Y_pred, 4, "PyroCast:Y_pred")
        check_tensor_shape(wind_u, 4, "PyroCast:wind_u")
        check_tensor_shape(wind_v, 4, "PyroCast:wind_v")

        B, _, H, W = Y_pred.shape

        # --- Directional spread simulation ---
        spread_map = self.directional_conv(Y_pred, wind_u, wind_v)

        # --- Optional physical regularization ---
        if self.use_physics_adjust:
            spread_map = self._physics_adjustment(spread_map)

        # --- Blending predicted vs. physics-based map ---
        Y_final = (1 - self.blend_beta) * Y_pred + self.blend_beta * spread_map

        assert Y_final.shape == (B, 1, H, W), f"[ShapeMismatch] Expected {(B, 1, H, W)}, got {Y_final.shape}"

        if self.verbose:
            print(f"[PyroCast] Output shape {Y_final.shape}, mean={Y_final.mean().item():.4f}")

        return Y_final


# -----------------------------------------------------------
# Unit Test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 49, 53
    Y_pred = torch.rand(B, 1, H, W)
    wind_u = torch.rand(B, 1, H, W)
    wind_v = torch.rand(B, 1, H, W)

    pyro = PyroCastPhysics(kernel_size=5, blend_beta=0.3, verbose=True)
    Y_final = pyro(Y_pred, wind_u, wind_v)

    print(f"\n Test passed | Y_final shape = {list(Y_final.shape)} | mean={Y_final.mean().item():.4f}")
