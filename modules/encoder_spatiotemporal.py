"""
FireTrend Stage 1 — Two-Scale Multimodal Spatial-Temporal Encoder (Corrected)
Integrates wildfire history, meteorological, and dynamic geospatial inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.layers.temporal_transformer import TemporalTransformerLayer
from modules.layers.spatial_transformer import SpatialTransformerLayer


# -----------------------------------------------------------
# Shape Checking Utility
# -----------------------------------------------------------
def check_tensor_shape(tensor: torch.Tensor, expected_dims: int, context: str = ""):
    if tensor.ndim != expected_dims:
        raise ValueError(
            f"\033[91m[ShapeError in {context}] Expected {expected_dims} dims, "
            f"but got {tensor.ndim}. Tensor shape = {list(tensor.shape)}\033[0m"
        )
    return True


# -----------------------------------------------------------
# Multimodal Fusion Block
# -----------------------------------------------------------
class MultimodalFusion(nn.Module):
    """
    Fuses wildfire (daily), ERA5 (encoded daily), and dynamic geospatial inputs.
    """

    def __init__(self, in_dims, embed_dim):
        """
        Args:
            in_dims (dict): {'fire': C_f, 'meteo': C_m, 'geo': C_g}
        """
        super().__init__()
        self.fire_proj = nn.Conv3d(in_dims['fire'], embed_dim, kernel_size=1)
        self.meteo_proj = nn.Conv3d(in_dims['meteo'], embed_dim, kernel_size=1)
        # Temporal convolution to aggregate 4 hourly ERA5 slices → 1 daily representation
        # self.meteo_temporal_conv = nn.Conv3d(
        #     in_channels=in_dims['meteo'],  # 6 meteorological features
        #     out_channels=in_dims['meteo'],  # keep same feature size
        #     kernel_size=(4, 1, 1),          # aggregate 4 hourly slices
        #     stride=(4, 1, 1),
        #     padding=0,
        #     bias=True
        # )
        self.geo_proj = nn.Conv3d(in_dims['geo'], embed_dim, kernel_size=1)
        self.norm = nn.BatchNorm3d(embed_dim)

    # def forward(self, X_fire, X_meteo, X_geo):
    #     """
    #     Args:
    #         X_fire:  [B, T, C_f, H, W]
    #         X_meteo:[B, T, C_m, H, W]
    #         X_geo:  [B, T, C_g, H, W]
    #     Returns:
    #         fused: [B, T, D, H, W]
    #     """
    #     check_tensor_shape(X_fire, 5, "Fusion:X_fire")
    #     check_tensor_shape(X_meteo, 5, "Fusion:X_meteo")
    #     check_tensor_shape(X_geo, 5, "Fusion:X_geo")

    #     # Rearrange for Conv3D (C at dim=1, T at dim=2)
    #     X_fire = rearrange(X_fire, "b t c h w -> b c t h w")
    #     X_meteo = rearrange(X_meteo, "b t c h w -> b c t h w")
    #     X_geo = rearrange(X_geo, "b t c h w -> b c t h w")

    #     F_fire = self.fire_proj(X_fire)
    #     F_meteo = self.meteo_proj(X_meteo)
    #     F_geo = self.geo_proj(X_geo)

    #     # Fusion by summation (aligned across time)
    #     fused = F_fire + F_meteo + F_geo
    #     fused = self.norm(fused)
    #     fused = F.gelu(fused)

    #     # Restore to [B, T, D, H, W]
    #     fused = rearrange(fused, "b d t h w -> b t d h w")
    #     return fused


    def forward(self, X_fire, X_meteo, X_geo):
        """
        Args:
            X_fire:  [B, T, C_f, H, W]
            X_meteo:[B, T, 4, C_m, H, W]
            X_geo:  [B, T, C_g, H, W]
        Returns:
            fused: [B, T, D, H, W]
        """
        # check_tensor_shape(X_fire, 5, "Fusion:X_fire")
        # check_tensor_shape(X_meteo, 6, "Fusion:X_meteo")
        # check_tensor_shape(X_geo, 5, "Fusion:X_geo")

        # # --- Fire ---
        # X_fire = rearrange(X_fire, "b t c h w -> b c t h w")
        # F_fire = self.fire_proj(X_fire)

        # # --- Meteorology (ERA5 hourly aggregation) ---
        # B, T, hourly, C, H, W = X_meteo.shape
        # X_meteo = rearrange(X_meteo, "b t h c hh ww -> (b t) c h hh ww")
        # X_meteo = self.meteo_temporal_conv(X_meteo)
        # X_meteo = rearrange(X_meteo, "(b t) c 1 h w -> b t c h w", b=B, t=T)
        # X_meteo = rearrange(X_meteo, "b t c h w -> b c t h w")
        # F_meteo = self.meteo_proj(X_meteo)

        # # --- Geo ---
        # X_geo = rearrange(X_geo, "b t c h w -> b c t h w")
        # F_geo = self.geo_proj(X_geo)

        # # --- Fusion ---
        # fused = F_fire + F_meteo + F_geo
        # fused = self.norm(fused)
        # fused = F.gelu(fused)
        # fused = rearrange(fused, "b d t h w -> b t d h w")
        # return fused

# --- Shape checks (allow 5D/6D for robustness) ---
        check_tensor_shape(X_fire, 5, "Fusion:X_fire")
        if X_meteo.ndim not in [5, 6]:
            raise ValueError(f"[ShapeError:Fusion:X_meteo] Expected 5D or 6D tensor, got {X_meteo.shape}")
        check_tensor_shape(X_geo, 5, "Fusion:X_geo")

        # --- Fire ---
        X_fire = rearrange(X_fire, "b t c h w -> b c t h w")
        F_fire = self.fire_proj(X_fire)

        # --- Meteorology ---
        if X_meteo.ndim == 6:
            # (safety fallback only, normally already aggregated)
            X_meteo = X_meteo.mean(dim=2)
        X_meteo = rearrange(X_meteo, "b t c h w -> b c t h w")
        F_meteo = self.meteo_proj(X_meteo)

        # --- Geo ---
        X_geo = rearrange(X_geo, "b t c h w -> b c t h w")
        F_geo = self.geo_proj(X_geo)

        # --- Fusion ---
        fused = F_fire + F_meteo + F_geo
        fused = self.norm(fused)
        fused = F.gelu(fused)
        fused = rearrange(fused, "b d t h w -> b t d h w")
        return fused


# -----------------------------------------------------------
#  Main Encoder
# -----------------------------------------------------------
class SpatialTemporalEncoder(nn.Module):
    """
    FireTrend Multimodal Spatio-Temporal Encoder
    (Wildfire + ERA5 Meteorology + Dynamic Geo)
    """
       
    def __init__(self, in_dims, embed_dim=128, num_heads=4,
             hidden_dim=512, dropout=0.1, height=None, width=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.fusion = MultimodalFusion(in_dims, embed_dim)
        self.temporal_block = TemporalTransformerLayer(
            dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout
        )
        self.spatial_block = SpatialTransformerLayer(
            dim=embed_dim, height=height, width=width,
            num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout
        )
        self.meteo_temporal_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=6,   # 6个ERA5特征
                out_channels=6,
                kernel_size=(4, 1, 1),
                stride=(4, 1, 1),
                groups=6,        # depthwise conv，每个通道独立卷积
                padding=0
            ),
            nn.BatchNorm3d(6),
            nn.GELU()
        )
        self.proj_out = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.BatchNorm3d(embed_dim)


    def forward(self, X_fire, X_meteo, X_geo):
        """
        Args:
            X_fire:  [B, T, 1, H, W]
            X_meteo:[B, T, 6, H, W]
            X_geo:  [B, T, 2, H, W]
        Returns:
            Z: [B, T, D, H, W]
        """
        # --- Temporal Aggregation for ERA5 hourly ---
        if X_meteo.ndim == 6:  # [B, T, 4, 6, H, W]
            B, T, H4, C, H_, W_ = X_meteo.shape
            X_meteo = X_meteo.permute(0, 3, 1, 2, 4, 5).reshape(B, C, T * H4, H_, W_)
            X_meteo = self.meteo_temporal_conv(X_meteo)  # [B, 6, T, H, W]
            X_meteo = X_meteo.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, 6, H, W]
        
        # --- Optional Temporal Aggregation for dynamic geo ---
        if X_geo.ndim == 6:  # [B, T, 4, 2, H, W]
            #B, T, H4, C, H_, W_ = X_geo.shape
            X_geo = X_geo.mean(dim=2)  # 或用卷积聚合: self.geo_temporal_conv()
        check_tensor_shape(X_fire, 5, "Encoder:X_fire")
        check_tensor_shape(X_meteo, 5, "Encoder:X_meteo")
        check_tensor_shape(X_geo, 5, "Encoder:X_geo")

        if self.verbose:
            print(f"[Encoder] Inputs fire={X_fire.shape}, meteo={X_meteo.shape}, geo={X_geo.shape}")

        # Step 1. Multimodal Fusion
        fused = self.fusion(X_fire, X_meteo, X_geo)
        if self.verbose:
            print(f"[Encoder] After Fusion: {list(fused.shape)}")

        # Step 2. Temporal Transformer
        fused = self.temporal_block(fused)
        if self.verbose:
            print(f"[Encoder] After Temporal Transformer: {list(fused.shape)}")
        
        # --- 🔥 Dynamically fix positional encoding ---
        if hasattr(self.spatial_block, "pos_encoding"):
            h, w = fused.shape[-2:]
            pos_emb = self.spatial_block.pos_encoding
            if getattr(pos_emb, "H", None) != h or getattr(pos_emb, "W", None) != w:
                from modules.layers.spatial_transformer import SpatialPositionalEncoding
                self.spatial_block.pos_encoding = SpatialPositionalEncoding(fused.shape[2], h, w).to(fused.device)
                if self.verbose:
                    print(f"[Encoder] Reinitialized positional encoding to ({h}, {w})")
        # Step 3. Spatial Transformer
        fused = self.spatial_block(fused)
        if self.verbose:
            print(f"[Encoder] After Spatial Transformer: {list(fused.shape)}")

        # Step 4. Output Projection
        fused_3d = rearrange(fused, "b t d h w -> b d t h w")
        Z = self.proj_out(fused_3d)
        Z = self.norm(Z)
        Z = rearrange(Z, "b d t h w -> b t d h w")

        if self.verbose:
            print(f"[Encoder] Output Z: {list(Z.shape)}")

        return Z


# -----------------------------------------------------------
# Unit Test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 2      # batch size
    T = 6      # temporal sequence length (daily steps)
    H4 = 4     # 4-hourly resolution within each day
    H, W = 16, 16

    # --- Create dummy inputs ---
    X_fire = torch.randn(B, T, 1, H, W)        # wildfire daily
    X_meteo = torch.randn(B, T, H4, 6, H, W)   # ERA5 4-hourly data (6 features)
    X_geo = torch.randn(B, T, 2, H, W)         # dynamic geo features

    # --- Initialize model ---
    model = SpatialTemporalEncoder(
        in_dims={"fire": 1, "meteo": 6, "geo": 2},
        embed_dim=64,
        num_heads=4,
        hidden_dim=128,
        height=H,
        width=W,
        verbose=True
    )

    # --- Forward pass ---
    Z = model(X_fire, X_meteo, X_geo)

    print("\n Test passed!")
    print(f"Output Z shape: {list(Z.shape)}  (expected [B, T, D, H, W])")
