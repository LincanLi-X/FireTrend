"""
FireTrend: Full Integrated Model
Combining:
    - Stage 1: Multimodal Spatio-Temporal Encoder
    - Stage 2: Multi-View Contrastive Learning
    - Stage 3: PyroCast (Physics-Guided Fire Spread)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder_spatiotemporal import SpatialTemporalEncoder
from modules.contrastive_learning import MultiViewContrastive
from modules.pyrocast_physics import PyroCastPhysics


# -----------------------------------------------------------
# Utility — Shape Checking
# -----------------------------------------------------------
def check_shape(tensor, expected_dims, context=""):
    if tensor.ndim != expected_dims:
        raise ValueError(f"[ShapeError:{context}] Expected {expected_dims} dims, got {tensor.shape}")
    return True


# -----------------------------------------------------------
# FireTrend Model Definition
# -----------------------------------------------------------
class FireTrendModel(nn.Module):
    """
    Full FireTrend model integrating encoder, contrastive learning, and physics-based fire spread.
    """

    def __init__(self, in_dims, embed_dim=128, num_heads=4, hidden_dim=512,
                 height=None, width=None, kernel_size=5, beta=0.3,
                 temperature=0.07, verbose=False):
        super().__init__()
        self.verbose = verbose

        # Stage 1 — Encoder
        self.encoder = SpatialTemporalEncoder(
            in_dims=in_dims,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            height=height,
            width=width,
            verbose=verbose
        )

        # Stage 2 — Multi-View Contrastive
        self.contrastive = MultiViewContrastive(
            embed_dim=embed_dim,
            temperature=temperature,
            verbose=verbose
        )

        # Stage 3 — PyroCast Physics
        self.pyrocast = PyroCastPhysics(
            kernel_size=kernel_size,
            blend_beta=beta,
            verbose=verbose
        )

        # Final projection head (for risk prediction)
        self.proj_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1)
        )

    # -------------------------------------------------------
    #  Forward
    # -------------------------------------------------------
    def forward(self, X_fire, X_meteo, X_geo, wind_u=None, wind_v=None):
        """
        Args:
            X_fire:  [B, T, 1, H, W]
            X_meteo:[B, T, 6, H, W]
            X_geo:  [B, T, 2, H, W]
            wind_u, wind_v: [B, T, 1, H, W]  (dynamic wind fields)
        Returns:
            dict containing:
                - Y_pred:     [B, 1, H, W]
                - Y_final:    [B, 1, H, W]
                - embeddings: [B, D]
                - L_contrast: scalar
        """
        check_shape(X_fire, 5, "FireTrend:X_fire")
        # Allow both daily (5D) and hourly (6D) for ERA5 and Geo
        if X_meteo.ndim not in [5, 6]:
            raise ValueError(f"[ShapeError:FireTrend:X_meteo] Expected 5D or 6D tensor, got {X_meteo.shape}")

        if X_geo.ndim not in [5, 6]:
            raise ValueError(f"[ShapeError:FireTrend:X_geo] Expected 5D or 6D tensor, got {X_geo.shape}")

        if wind_u is not None:
            check_shape(wind_u, 5, "FireTrend:wind_u")
        if wind_v is not None:
            check_shape(wind_v, 5, "FireTrend:wind_v")
        
        if self.verbose:
            print("\n===== FireTrend Forward Pass =====")

        # ERA5 hourly → daily aggregation using temporal conv
# ERA5 hourly → daily temporal aggregation (conv over hourly dimension)
        if X_meteo.ndim == 6:
            B, T, H4, C, H_, W_ = X_meteo.shape  # [B, T, 4, 6, H, W]
            X_meteo = X_meteo.permute(0, 3, 1, 2, 4, 5).contiguous()  # [B, 6, T, 4, H, W]

            # Collapse T×4 into one longer temporal dimension for conv3d
            X_meteo = X_meteo.reshape(B, C, T * H4, H_, W_)  # [B, 6, T*4, H, W]

            # Apply temporal convolution along the time axis
            temporal_conv = nn.Conv3d(
                in_channels=C,
                out_channels=C,
                kernel_size=(4, 1, 1),
                stride=(4, 1, 1),
                padding=0,
                groups=C  # depthwise temporal conv per channel
            ).to(X_meteo.device)

            X_meteo = temporal_conv(X_meteo)  # [B, 6, T, H, W]
            X_meteo = X_meteo.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, 6, H, W]

        # Stage 1: Spatio-temporal encoding
        Z = self.encoder(X_fire, X_meteo, X_geo)  # [B, T, D, H, W]

        # Stage 2: Contrastive learning (self-supervised regularization)
        refined_embed, L_contrast = self.contrastive(Z)

        # Stage 3: Predict wildfire risk (using last time-step features)
        Z_last = Z[:, -1]  # [B, D, H, W]
        check_shape(Z_last, 4, "FireTrend:Z_last")

        Y_pred = self.proj_head(Z_last)  # [B, 1, H, W]
        check_shape(Y_pred, 4, "FireTrend:Y_pred")

        # # Physics-based adjustment (PyroCast) — keep full temporal context
        # if wind_u is not None and wind_v is not None:
        #     # pass full temporal wind tensors to physics module
        #     Y_final = self.pyrocast(Y_pred.unsqueeze(1), wind_u, wind_v)  # [B, 1, H, W]
        # else:
        #     Y_final = Y_pred

    
        if wind_u is not None and wind_v is not None:
            # Ensure we use only the last time-step and correct shape [B, 1, H, W]
            if wind_u.ndim == 5:  # [B, T, 1, H, W]
                wind_u_last = wind_u[:, -1]  # [B, 1, H, W]
            elif wind_u.ndim == 4:  # already [B, 1, H, W]
                wind_u_last = wind_u
            else:
                raise ValueError(f"[FireTrend] Unexpected wind_u shape {wind_u.shape}")

            if wind_v.ndim == 5:
                wind_v_last = wind_v[:, -1]
            elif wind_v.ndim == 4:
                wind_v_last = wind_v
            else:
                raise ValueError(f"[FireTrend] Unexpected wind_v shape {wind_v.shape}")

            # Correct call — no unsqueeze on Y_pred
            Y_final = self.pyrocast(Y_pred, wind_u_last, wind_v_last)
        else:
            Y_final = Y_pred
        
        check_shape(Y_final, 4, "FireTrend:Y_final")

        if self.verbose:
            print(f"[FireTrend] Encoder Z: {list(Z.shape)}")
            print(f"[FireTrend] Predicted Y_pred: {list(Y_pred.shape)}")
            print(f"[FireTrend] Physics-adjusted Y_final: {list(Y_final.shape)}")
            print(f"[FireTrend] Contrastive Loss: {L_contrast.item():.4f}")

        # -------------------------------------------------------
        # Derive contrastive embeddings (for InfoNCE)
        # -------------------------------------------------------
        z_temporal = Z.mean(dim=(1, 3, 4))   # [B, D]
        z_spatial = Z[:, -1].mean(dim=(2, 3))  # [B, D]

        return {
            "Y_pred": Y_pred,
            "Y_final": Y_final,
            "embeddings": (z_spatial, z_temporal),
            "L_contrast": L_contrast
        }

# -----------------------------------------------------------
# Unit Test
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, H, W = 2, 6, 49, 53
    X_fire = torch.randn(B, T, 1, H, W)
    X_meteo = torch.randn(B, T, 6, H, W)
    X_geo = torch.randn(B, T, 2, H, W)
    wind_u = torch.randn(B, T, 1, H, W)  # u10
    wind_v = torch.randn(B, T, 1, H, W)  # v10

    model = FireTrendModel(
        in_dims={"fire": 1, "meteo": 6, "geo": 2},
        embed_dim=64,
        num_heads=4,
        hidden_dim=128,
        height=H,
        width=W,
        kernel_size=5,
        verbose=True
    )

    out = model(X_fire, X_meteo, X_geo, wind_u, wind_v)

    print("\n Test Output Summary:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={list(v.shape)}")
        else:
            print(f"{k}: {v:.4f}")
