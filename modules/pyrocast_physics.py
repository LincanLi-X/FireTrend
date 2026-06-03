"""
PyroCast latent propagation for FireTrend.

The NeurIPS 2026 version applies PyroCast in latent fire-state space,
not as a post-hoc correction on the predicted risk map. The operator
constructs a wind-conditioned anisotropic kernel per grid cell and applies
it channel-wise to latent maps H_t.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_4d(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be [B, C, H, W], got {list(tensor.shape)}")


def _positive_parameter(raw: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return F.softplus(raw) + eps


def _inverse_softplus(value: float, eps: float = 1e-4) -> torch.Tensor:
    value = max(float(value) - eps, eps)
    return torch.log(torch.expm1(torch.tensor(value)))


class DirectionalConv2D(nn.Module):
    """
    Spatially adaptive directional convolution.

    Inputs:
        latent_map: [B, C, H, W]
        wind_u/v:   [B, 1, H, W]
        temperature, humidity: optional [B, 1, H, W] normalized drivers
    Output:
        propagated latent map [B, C, H, W]
    """

    def __init__(
        self,
        kernel_size: int = 5,
        rho: float = 0.35,
        sigma_parallel: float = 1.50,
        sigma_perp: float = 0.75,
        normalize_kernel: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("PyroCast kernel_size must be odd.")
        self.kernel_size = int(kernel_size)
        self.pad_size = self.kernel_size // 2
        self.normalize_kernel = normalize_kernel
        self.verbose = verbose

        self.raw_rho = nn.Parameter(_inverse_softplus(float(rho)))
        self.raw_sigma_parallel = nn.Parameter(_inverse_softplus(float(sigma_parallel)))
        self.raw_sigma_perp = nn.Parameter(_inverse_softplus(float(sigma_perp)))

        # Propagation strength alpha = softplus(kappa*s + eta_t*T - eta_h*H)
        self.kappa = nn.Parameter(torch.tensor(1.0))
        self.eta_temperature = nn.Parameter(torch.tensor(0.25))
        self.eta_humidity = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def _coerce_single_channel(name: str, tensor: torch.Tensor) -> torch.Tensor:
        _check_4d(name, tensor)
        if tensor.size(1) != 1:
            tensor = tensor[:, :1]
        return tensor

    @staticmethod
    def _unit_like(reference: torch.Tensor, value: float) -> torch.Tensor:
        return torch.full_like(reference[:, :1], float(value))

    def generate_kernel(
        self,
        wind_u: torch.Tensor,
        wind_v: torch.Tensor,
        temperature: torch.Tensor | None = None,
        humidity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return K_spread with shape [B, K*K, H, W].
        """
        wind_u = self._coerce_single_channel("wind_u", wind_u)
        wind_v = self._coerce_single_channel("wind_v", wind_v)
        if temperature is None:
            temperature = self._unit_like(wind_u, 0.0)
        else:
            temperature = self._coerce_single_channel("temperature", temperature)
        if humidity is None:
            humidity = self._unit_like(wind_u, 0.0)
        else:
            humidity = self._coerce_single_channel("humidity", humidity)

        B, _, H, W = wind_u.shape
        device, dtype = wind_u.device, wind_u.dtype
        k = self.kernel_size
        pad = self.pad_size

        y, x = torch.meshgrid(
            torch.arange(-pad, pad + 1, device=device, dtype=dtype),
            torch.arange(-pad, pad + 1, device=device, dtype=dtype),
            indexing="ij",
        )
        dx = x.reshape(1, k, k, 1, 1)
        dy = y.reshape(1, k, k, 1, 1)

        speed = torch.sqrt(wind_u.square() + wind_v.square() + 1e-8)
        phi = torch.atan2(wind_v, wind_u + 1e-8)
        cos_phi = torch.cos(phi).unsqueeze(1)
        sin_phi = torch.sin(phi).unsqueeze(1)

        rho = _positive_parameter(self.raw_rho).to(dtype=dtype)
        sigma_parallel = _positive_parameter(self.raw_sigma_parallel).to(dtype=dtype)
        sigma_perp = _positive_parameter(self.raw_sigma_perp).to(dtype=dtype)

        mu_x = (-rho * speed * torch.cos(phi)).unsqueeze(1)
        mu_y = (-rho * speed * torch.sin(phi)).unsqueeze(1)
        centered_x = dx - mu_x
        centered_y = dy - mu_y

        parallel = centered_x * cos_phi + centered_y * sin_phi
        perpendicular = -centered_x * sin_phi + centered_y * cos_phi
        exponent = -0.5 * (
            parallel.square() / sigma_parallel.square()
            + perpendicular.square() / sigma_perp.square()
        )

        alpha = F.softplus(
            self.kappa.to(dtype=dtype) * speed
            + self.eta_temperature.to(dtype=dtype) * temperature
            - self.eta_humidity.to(dtype=dtype) * humidity
        ).unsqueeze(1)
        gaussian = torch.exp(exponent).reshape(B, k * k, H, W)
        if self.normalize_kernel:
            gaussian = gaussian / (gaussian.sum(dim=1, keepdim=True) + 1e-8)
        kernel = alpha.reshape(B, 1, H, W) * gaussian

        if self.verbose:
            print(
                "[PyroCast] kernel "
                f"speed=({speed.min().item():.3f},{speed.max().item():.3f}) "
                f"rho={rho.item():.3f} sigmas=({sigma_parallel.item():.3f},{sigma_perp.item():.3f})"
            )
        return kernel

    def forward(
        self,
        latent_map: torch.Tensor,
        wind_u: torch.Tensor,
        wind_v: torch.Tensor,
        temperature: torch.Tensor | None = None,
        humidity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_4d("latent_map", latent_map)
        B, C, H, W = latent_map.shape
        kernel = self.generate_kernel(wind_u, wind_v, temperature, humidity)  # [B, K*K, H, W]

        patches = F.unfold(latent_map, kernel_size=self.kernel_size, padding=self.pad_size)
        patches = patches.view(B, C, self.kernel_size * self.kernel_size, H, W)
        propagated = (patches * kernel.unsqueeze(1)).sum(dim=2)
        return propagated


class PyroCastPhysics(nn.Module):
    """
    Physics-guided latent propagation wrapper.

    The module is intentionally channel-agnostic, so it can propagate a
    one-channel map for diagnostics, but the FireTrend model uses it on
    latent feature maps [B, D, H, W].
    """

    def __init__(
        self,
        kernel_size: int = 5,
        rho: float = 0.35,
        sigma_parallel: float = 1.50,
        sigma_perp: float = 0.75,
        normalize_kernel: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.directional_conv = DirectionalConv2D(
            kernel_size=kernel_size,
            rho=rho,
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            normalize_kernel=normalize_kernel,
            verbose=verbose,
        )
        self.verbose = verbose

    def forward(
        self,
        latent_map: torch.Tensor,
        wind_u: torch.Tensor,
        wind_v: torch.Tensor,
        temperature: torch.Tensor | None = None,
        humidity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.directional_conv(latent_map, wind_u, wind_v, temperature, humidity)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, H, W = 2, 16, 8, 9
    H_t = torch.randn(B, D, H, W)
    wind_u = torch.randn(B, 1, H, W)
    wind_v = torch.randn(B, 1, H, W)
    temp = torch.rand(B, 1, H, W)
    hum = torch.rand(B, 1, H, W)
    pyro = PyroCastPhysics(kernel_size=5, verbose=True)
    out = pyro(H_t, wind_u, wind_v, temp, hum)
    print(out.shape)
