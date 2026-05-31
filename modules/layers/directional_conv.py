"""
Compatibility import for PyroCast directional convolution.

The active FireTrend implementation uses
`modules.pyrocast_physics.DirectionalConv2D`, which propagates latent
feature maps with spatially adaptive wind-conditioned kernels. This file is
kept only so older imports from `modules.layers.directional_conv` continue
to resolve.
"""

from modules.pyrocast_physics import DirectionalConv2D

__all__ = ["DirectionalConv2D"]
