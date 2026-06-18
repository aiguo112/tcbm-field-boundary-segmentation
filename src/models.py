"""Segmentation model baselines for FTW / PASTIS field-boundary experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

IN_CHANNELS = 8
NUM_CLASSES = 3
BANDS_PER_DATE = 4


class UNetBaseline(nn.Module):
    """U-Net with EfficientNet-B3 encoder (8-channel input)."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        encoder_name: str = "efficientnet-b3",
    ) -> None:
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SegFormerBaseline(nn.Module):
    """SegFormer with MiT-B2 encoder (8-channel input)."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        encoder_name: str = "mit_b2",
    ) -> None:
        super().__init__()
        self.net = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAttention(nn.Module):
    """Lightweight temporal attention over a sequence of feature maps."""

    def __init__(self, channels: int, n_dates: int = 2) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels * n_dates, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, n_dates, kernel_size=1),
        )
        self.n_dates = n_dates

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # features: list of (B, C, H, W), length = n_dates
        stacked = torch.cat(features, dim=1)
        weights = self.attn(stacked)  # (B, n_dates, H, W)
        weights = torch.softmax(weights, dim=1)
        fused = sum(f * weights[:, i : i + 1] for i, f in enumerate(features))
        return fused


class TemporalConcat(nn.Module):
    """Fuse per-date features by channel concatenation + 1x1 projection."""

    def __init__(self, channels: int, n_dates: int = 2) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(channels * n_dates, channels, kernel_size=1)
        self.n_dates = n_dates

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.fuse(torch.cat(features, dim=1))


class TCBMFusion(nn.Module):
    """
    Temporal Contrast Boundary Module: fuse two date features via
    mean, absolute difference, normalized ratio, and raw concatenation.
    """

    def __init__(self, channels: int, n_dates: int = 2) -> None:
        super().__init__()
        if n_dates != 2:
            raise ValueError(f"TCBMFusion requires n_dates=2, got {n_dates}")
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 5, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        f1, f2 = features[0], features[1]
        input_dtype = f1.dtype
        f1 = f1.float()
        f2 = f2.float()
        mean = (f1 + f2) / 2
        diff = torch.abs(f2 - f1).clamp(0, 100)
        ratio = ((f2 - f1) / (f1 + f2 + 1e-3)).clamp(-10, 10)
        original = torch.cat([f1, f2], dim=1)
        stacked = torch.cat([mean, diff, ratio, original], dim=1)
        return self.fusion(stacked).to(input_dtype)


def _utae_conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UTAESegmentor(nn.Module):
    """U-TAE-style segmentor: shared per-date encoder + temporal attention + decoder."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        n_dates: int = 2,
        bands_per_date: int = BANDS_PER_DATE,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        if in_channels != n_dates * bands_per_date:
            raise ValueError(
                f"in_channels={in_channels} must equal n_dates*bands_per_date="
                f"{n_dates * bands_per_date}"
            )

        self.n_dates = n_dates
        self.bands_per_date = bands_per_date
        self.base_channels = base_channels

        self.date_encoder = nn.Sequential(
            _utae_conv_block(bands_per_date, base_channels),
            nn.MaxPool2d(2),
            _utae_conv_block(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            _utae_conv_block(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
        )
        self.temporal_attn = TemporalAttention(base_channels * 4, n_dates=n_dates)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _utae_conv_block(base_channels * 4, base_channels * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _utae_conv_block(base_channels * 2, base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.seg_head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def encode_decode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return decoder features (B, base_channels, H, W)."""
        b, _, h, w = x.shape
        x = x.view(b, self.n_dates, self.bands_per_date, h, w)
        date_features = [self.date_encoder(x[:, t]) for t in range(self.n_dates)]
        fused = self.temporal_attn(date_features)
        features = self.decoder(fused)
        if features.shape[-2:] != (h, w):
            features = F.interpolate(
                features, size=(h, w), mode="bilinear", align_corners=False
            )
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(self.encode_decode_features(x))


class UNetConcatSegmentor(nn.Module):
    """U-TAE-style segmentor with concat fusion instead of temporal attention."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        n_dates: int = 2,
        bands_per_date: int = BANDS_PER_DATE,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        if in_channels != n_dates * bands_per_date:
            raise ValueError(
                f"in_channels={in_channels} must equal n_dates*bands_per_date="
                f"{n_dates * bands_per_date}"
            )

        self.n_dates = n_dates
        self.bands_per_date = bands_per_date
        self.base_channels = base_channels

        self.date_encoder = nn.Sequential(
            _utae_conv_block(bands_per_date, base_channels),
            nn.MaxPool2d(2),
            _utae_conv_block(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            _utae_conv_block(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
        )
        self.temporal_fuse = TemporalConcat(base_channels * 4, n_dates=n_dates)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _utae_conv_block(base_channels * 4, base_channels * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _utae_conv_block(base_channels * 2, base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.seg_head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def encode_decode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return decoder features (B, base_channels, H, W)."""
        b, _, h, w = x.shape
        x = x.view(b, self.n_dates, self.bands_per_date, h, w)
        date_features = [self.date_encoder(x[:, t]) for t in range(self.n_dates)]
        fused = self.temporal_fuse(date_features)
        features = self.decoder(fused)
        if features.shape[-2:] != (h, w):
            features = F.interpolate(
                features, size=(h, w), mode="bilinear", align_corners=False
            )
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(self.encode_decode_features(x))


class TCBMUNet(nn.Module):
    """U-TAE encoder/decoder with TCBM temporal contrast fusion."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        n_dates: int = 2,
        bands_per_date: int = BANDS_PER_DATE,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        if in_channels != n_dates * bands_per_date:
            raise ValueError(
                f"in_channels={in_channels} must equal n_dates*bands_per_date="
                f"{n_dates * bands_per_date}"
            )

        self.n_dates = n_dates
        self.bands_per_date = bands_per_date
        self.base_channels = base_channels

        self.date_encoder = nn.Sequential(
            _utae_conv_block(bands_per_date, base_channels),
            nn.MaxPool2d(2),
            _utae_conv_block(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            _utae_conv_block(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
        )
        self.temporal_fuse = TCBMFusion(base_channels * 4, n_dates=n_dates)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _utae_conv_block(base_channels * 4, base_channels * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _utae_conv_block(base_channels * 2, base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.seg_head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def encode_decode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return decoder features (B, base_channels, H, W)."""
        b, _, h, w = x.shape
        x = x.view(b, self.n_dates, self.bands_per_date, h, w)
        date_features = [self.date_encoder(x[:, t]) for t in range(self.n_dates)]
        fused = self.temporal_fuse(date_features)
        features = self.decoder(fused)
        if features.shape[-2:] != (h, w):
            features = F.interpolate(
                features, size=(h, w), mode="bilinear", align_corners=False
            )
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(self.encode_decode_features(x))


class UTAESegmentorDist(UTAESegmentor):
    """U-TAE segmentor with auxiliary normalized distance-to-boundary head."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        n_dates: int = 2,
        bands_per_date: int = BANDS_PER_DATE,
        base_channels: int = 64,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            n_dates=n_dates,
            bands_per_date=bands_per_date,
            base_channels=base_channels,
        )
        self.dist_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_decode_features(x)
        return {
            "seg": self.seg_head(features),
            "dist": self.dist_head(features).squeeze(1),
        }


def segmentation_logits(
    output: torch.Tensor | dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return segmentation logits from a model forward pass."""
    if isinstance(output, dict):
        return output["seg"]
    return output


def get_model(
    name: str,
    *,
    in_channels: int = IN_CHANNELS,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Factory for supported segmentation models."""
    name = name.lower()
    if name == "unet":
        return UNetBaseline(in_channels=in_channels, num_classes=num_classes)
    if name == "segformer":
        return SegFormerBaseline(in_channels=in_channels, num_classes=num_classes)
    if name == "utae":
        return UTAESegmentor(in_channels=in_channels, num_classes=num_classes)
    if name == "utae_dist":
        return UTAESegmentorDist(in_channels=in_channels, num_classes=num_classes)
    if name == "unet_concat":
        return UNetConcatSegmentor(in_channels=in_channels, num_classes=num_classes)
    if name == "tcbm_unet":
        return TCBMUNet(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(
        f"Unknown model {name!r}; expected unet, segformer, utae, utae_dist, "
        f"unet_concat, or tcbm_unet."
    )


if __name__ == "__main__":
    model = get_model("tcbm_unet")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"tcbm_unet parameters: {n_params:,}")
