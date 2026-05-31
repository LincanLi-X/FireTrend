"""
FireTrend model for ordinal wildfire risk-level forecasting.

New-version pipeline:
1. Multimodal spatial-temporal encoder.
2. Label-free multi-view contrastive pretraining.
3. PyroCast-guided latent fire-state propagation.
4. Lightweight downstream classifier for low/mid/high risk levels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.contrastive_learning import MultiViewContrastive
from modules.encoder_spatiotemporal import SpatialTemporalEncoder
from modules.pyrocast_physics import PyroCastPhysics


def _check_shape(tensor: torch.Tensor, dims: int, name: str) -> None:
    if tensor.ndim != dims:
        raise ValueError(f"{name} must have {dims} dims, got {list(tensor.shape)}")


class FireTrendModel(nn.Module):
    """Physics-aware FireTrend with latent PyroCast and ordinal classifier."""

    def __init__(
        self,
        in_dims: dict[str, int],
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 512,
        height: int | None = None,
        width: int | None = None,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.1,
        kernel_size: int = 5,
        pyro_rho: float = 0.35,
        pyro_sigma_parallel: float = 1.50,
        pyro_sigma_perp: float = 0.75,
        temperature: float = 0.07,
        lambda_temporal: float = 1.0,
        lambda_spatial: float = 1.0,
        lambda_cross: float = 1.0,
        max_temporal_cells: int = 512,
        max_spatial_anchors: int = 128,
        max_cross_samples: int = 1024,
        meteo_driver_indices: dict[str, int] | None = None,
        detach_pyro_target: bool = True,
        verbose: bool = False,
        beta: float | None = None,
    ):
        super().__init__()
        self.verbose = verbose
        self.num_classes = int(num_classes)
        self.detach_pyro_target = bool(detach_pyro_target)

        # beta is accepted for backward compatibility with older config files.
        _ = beta

        self.encoder = SpatialTemporalEncoder(
            in_dims=in_dims,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            height=height,
            width=width,
            num_layers=num_layers,
            verbose=verbose,
        )

        self.contrastive = MultiViewContrastive(
            embed_dim=embed_dim,
            temperature=temperature,
            lambda_temporal=lambda_temporal,
            lambda_spatial=lambda_spatial,
            lambda_cross=lambda_cross,
            max_temporal_cells=max_temporal_cells,
            max_spatial_anchors=max_spatial_anchors,
            max_cross_samples=max_cross_samples,
            verbose=verbose,
        )

        self.pyrocast = PyroCastPhysics(
            kernel_size=kernel_size,
            rho=pyro_rho,
            sigma_parallel=pyro_sigma_parallel,
            sigma_perp=pyro_sigma_perp,
            verbose=verbose,
        )

        self.latent_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim, self.num_classes, kernel_size=1),
        )

        default_indices = {"u": 0, "v": 1, "temperature": 3, "humidity": 2}
        self.meteo_driver_indices = default_indices | (meteo_driver_indices or {})

    def encode(
        self,
        X_fire: torch.Tensor,
        X_meteo: torch.Tensor,
        X_geo: torch.Tensor,
        return_modalities: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        enc_out = self.encoder(X_fire, X_meteo, X_geo, return_modalities=return_modalities)
        if return_modalities:
            Z, Z_modalities = enc_out
            return Z, Z_modalities
        return enc_out, None

    def _extract_drivers(
        self,
        X_meteo: torch.Tensor,
        X_drivers: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Return PyroCast drivers [B, T, 1, H, W].

        X_drivers, when provided by the dataset, stores raw u/v wind and
        normalized temperature/humidity proxies in channels [u, v, temp, hum].
        This preserves wind direction after input min-max normalization.
        """
        if X_drivers is not None:
            _check_shape(X_drivers, 5, "X_drivers")
            if X_drivers.size(2) < 4:
                raise ValueError("X_drivers must contain [u, v, temperature, humidity] channels")
            return {
                "u": X_drivers[:, :, 0:1],
                "v": X_drivers[:, :, 1:2],
                "temperature": X_drivers[:, :, 2:3],
                "humidity": X_drivers[:, :, 3:4],
            }

        idx = self.meteo_driver_indices
        return {
            "u": X_meteo[:, :, idx["u"] : idx["u"] + 1],
            "v": X_meteo[:, :, idx["v"] : idx["v"] + 1],
            "temperature": X_meteo[:, :, idx["temperature"] : idx["temperature"] + 1],
            "humidity": X_meteo[:, :, idx["humidity"] : idx["humidity"] + 1],
        }

    def _propagate_one(
        self,
        H_t: torch.Tensor,
        drivers: dict[str, torch.Tensor],
        t: int,
    ) -> torch.Tensor:
        return self.pyrocast(
            H_t,
            drivers["u"][:, t],
            drivers["v"][:, t],
            drivers["temperature"][:, t],
            drivers["humidity"][:, t],
        )

    def physics_aware_representation(
        self,
        H_t: torch.Tensor,
        drivers: dict[str, torch.Tensor],
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H_phys = self._propagate_one(H_t, drivers, t)
        H_star = self.latent_fusion(torch.cat([H_t, H_phys], dim=1))
        return H_star, H_phys

    def pyrocast_latent_loss(self, Z: torch.Tensor, drivers: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Self-supervised loss: || F_pyro(H_t, K_t) - H_{t+1} ||_2^2."""
        B, T, D, H, W = Z.shape
        if T < 2:
            return Z.sum() * 0.0, None

        propagated = []
        for t in range(T - 1):
            propagated.append(self._propagate_one(Z[:, t], drivers, t).unsqueeze(1))
        H_phys_seq = torch.cat(propagated, dim=1)
        target = Z[:, 1:]
        if self.detach_pyro_target:
            target = target.detach()
        return F.mse_loss(H_phys_seq, target), H_phys_seq

    def forward(
        self,
        X_fire: torch.Tensor,
        X_meteo: torch.Tensor,
        X_geo: torch.Tensor,
        X_drivers: torch.Tensor | None = None,
        compute_pretrain: bool = True,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]]:
        _check_shape(X_fire, 5, "X_fire")
        _check_shape(X_meteo, 5, "X_meteo")
        _check_shape(X_geo, 5, "X_geo")

        Z, Z_modalities = self.encode(X_fire, X_meteo, X_geo, return_modalities=True)
        drivers = self._extract_drivers(X_meteo, X_drivers)

        zero_loss = Z.sum() * 0.0
        if compute_pretrain:
            refined_embed, contrast_losses = self.contrastive(Z, Z_modalities=Z_modalities)
            L_pyro, H_phys_seq = self.pyrocast_latent_loss(Z, drivers)
        else:
            refined_embed = Z[:, -1].mean(dim=(2, 3))
            contrast_losses = {
                "temporal": zero_loss,
                "spatial": zero_loss,
                "cross": zero_loss,
                "total": zero_loss,
            }
            L_pyro, H_phys_seq = zero_loss, None

        H_last = Z[:, -1]
        H_star, H_phys_last = self.physics_aware_representation(H_last, drivers, t=Z.size(1) - 1)
        logits = self.classifier(H_star)
        probabilities = torch.softmax(logits, dim=1)
        pred_classes = torch.argmax(probabilities, dim=1)

        if self.verbose:
            print(
                f"[FireTrend] Z={list(Z.shape)} H_star={list(H_star.shape)} "
                f"logits={list(logits.shape)}"
            )

        return {
            "logits": logits,
            "probabilities": probabilities,
            "pred_classes": pred_classes,
            "Z": Z,
            "H_star": H_star,
            "H_phys_last": H_phys_last,
            "H_phys_seq": H_phys_seq,
            "embeddings": refined_embed,
            "contrast_losses": contrast_losses,
            "L_contrast": contrast_losses["total"],
            "L_pyro": L_pyro,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, H, W = 2, 4, 8, 9
    model = FireTrendModel(
        in_dims={"fire": 1, "meteo": 8, "geo": 10},
        embed_dim=32,
        num_heads=4,
        hidden_dim=64,
        height=H,
        width=W,
        kernel_size=5,
    )
    x_fire = torch.randn(B, T, 1, H, W)
    x_meteo = torch.randn(B, T, 8, H, W)
    x_geo = torch.randn(B, T, 10, H, W)
    drivers = torch.randn(B, T, 4, H, W)
    out = model(x_fire, x_meteo, x_geo, drivers)
    print(out["logits"].shape, out["L_contrast"].item(), out["L_pyro"].item())
