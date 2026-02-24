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


    def forward(self, X_fire, X_meteo, X_geo, return_modalities: bool = False):
        """
        Args:
            X_fire:  [B, T, C_f, H, W]
            X_meteo:[B, T, C_m, H, W]
            X_geo:  [B, T, C_g, H, W]
        Returns:
            fused: [B, T, D, H, W]
            modality_feats (optional): dict of modality tensors [B, T, D, H, W]
        """
        check_tensor_shape(X_fire, 5, "Fusion:X_fire")
        check_tensor_shape(X_meteo, 5, "Fusion:X_meteo")
        check_tensor_shape(X_geo, 5, "Fusion:X_geo")

        # --- Fire ---
        X_fire = rearrange(X_fire, "b t c h w -> b c t h w")
        F_fire = self.fire_proj(X_fire)

        # --- Meteorology ---
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

        if not return_modalities:
            return fused

        modality_feats = {
            "fire": rearrange(F.gelu(F_fire), "b d t h w -> b t d h w"),
            "meteo": rearrange(F.gelu(F_meteo), "b d t h w -> b t d h w"),
            "geo": rearrange(F.gelu(F_geo), "b d t h w -> b t d h w"),
        }
        return fused, modality_feats


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
        self.proj_out = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.BatchNorm3d(embed_dim)


    def _project_branch(self, branch_bt):
        """
        Project a modality branch into the same embedding space as Z.
        Args:
            branch_bt: [B, T, D, H, W]
        Returns:
            [B, T, D, H, W]
        """
        branch_3d = rearrange(branch_bt, "b t d h w -> b d t h w")
        branch_3d = self.proj_out(branch_3d)
        branch_3d = self.norm(branch_3d)
        return rearrange(branch_3d, "b d t h w -> b t d h w")

    def forward(self, X_fire, X_meteo, X_geo, return_modalities: bool = False):
        """
        Args:
            X_fire:  [B, T, 1, H, W]
            X_meteo:[B, T, C_m, H, W]
            X_geo:  [B, T, C_g, H, W]
        Returns:
            Z: [B, T, D, H, W]
            modality_Z (optional): dict of modality tensors [B, T, D, H, W]
        """
        check_tensor_shape(X_fire, 5, "Encoder:X_fire")
        check_tensor_shape(X_meteo, 5, "Encoder:X_meteo")
        check_tensor_shape(X_geo, 5, "Encoder:X_geo")

        if self.verbose:
            print(f"[Encoder] Inputs fire={X_fire.shape}, meteo={X_meteo.shape}, geo={X_geo.shape}")

        # Step 1. Multimodal Fusion
        if return_modalities:
            fused, modality_feats = self.fusion(X_fire, X_meteo, X_geo, return_modalities=True)
        else:
            fused = self.fusion(X_fire, X_meteo, X_geo, return_modalities=False)
            modality_feats = None
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
            pe_h, pe_w = pos_emb.pos_emb.shape[-2:]
            if pe_h != h or pe_w != w:
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

        if not return_modalities:
            return Z

        modality_Z = {name: self._project_branch(feat) for name, feat in modality_feats.items()}
        return Z, modality_Z


# -----------------------------------------------------------
# Unit Test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 2      # batch size
    T = 6      # temporal sequence length (daily steps)
    H, W = 16, 16

    # --- Create dummy inputs ---
    X_fire = torch.randn(B, T, 1, H, W)        # wildfire daily
    X_meteo = torch.randn(B, T, 6, H, W)       # meteorological features (daily)
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
