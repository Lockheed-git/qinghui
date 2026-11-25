"""Inference-only generator compatible with Scenimefy's Shinkai checkpoint.

Architecture reference: https://github.com/Yuxinn-J/Scenimefy
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SceneNorm(nn.InstanceNorm2d):
    """Use one set of normalization statistics across a tiled scene."""

    def __init__(self, channels: int):
        super().__init__(channels)
        self.capture = False
        self.reference: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.capture:
            variance, mean = torch.var_mean(x.float(), dim=(-2, -1), correction=0)
            self.reference = (mean.flatten(), variance.flatten())
        elif self.reference is not None:
            mean, variance = self.reference
            return F.batch_norm(x, mean, variance, training=False, eps=self.eps)
        return super().forward(x)


class Resample(nn.Module):
    def __init__(self, channels: int, *, up: bool):
        super().__init__()
        self.channels = channels
        self.up = up
        taps = torch.tensor([1., 3., 3., 1.] if up else [1., 2., 1.])
        kernel = taps.outer(taps)
        kernel *= (4 if up else 1) / kernel.sum()
        self.register_buffer("filt", kernel.expand(channels, 1, -1, -1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.up:
            # The checkpoint's separable [1, 3, 3, 1] filter with
            # replicated borders is bilinear 2x sampling at pixel centers.
            return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(padded, self.filt, stride=2, groups=self.channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            SceneNorm(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            SceneNorm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            SceneNorm(64),
            nn.ReLU(inplace=True),
        ]
        for incoming, outgoing in [(64, 128), (128, 256)]:
            layers.extend((
                nn.Conv2d(incoming, outgoing, 3, padding=1),
                SceneNorm(outgoing),
                nn.ReLU(inplace=True),
                Resample(outgoing, up=False),
            ))
        layers.extend(ResidualBlock(256) for _ in range(9))
        for incoming, outgoing in [(256, 128), (128, 64)]:
            layers.extend((
                Resample(incoming, up=True),
                nn.Conv2d(incoming, outgoing, 3, padding=1),
                SceneNorm(outgoing),
                nn.ReLU(inplace=True),
            ))
        layers.extend((nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Tanh()))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_generator(path: Path, device: torch.device, dtype: torch.dtype) -> Generator:
    model = Generator()
    state = torch.load(path, map_location="cpu", weights_only=True)
    for name, layer in model.named_modules():
        if isinstance(layer, Resample) and layer.up:
            key = f"{name}.filt"
            if key not in state or not torch.equal(state[key], layer.filt):
                raise ValueError(f"Unexpected upsampling filter in checkpoint: {key}")
    model.load_state_dict(state, strict=True)
    return model.eval().to(device=device, dtype=dtype, memory_format=torch.channels_last)
