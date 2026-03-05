#!/usr/bin/env python3
"""CloneBlock-RT v0: causal STFT mask defense model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DefenseSTFTMaskConfig:
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    n_bands: int = 64
    hidden_channels: int = 128
    num_layers: int = 4
    kernel_size: int = 3
    lookahead: int = 0
    alpha: float = 0.15


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, lookahead: int = 0):
        super().__init__()
        self.left_pad = max(0, kernel_size - 1 - lookahead)
        self.right_pad = max(0, lookahead)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, self.right_pad))
        x = self.conv(x)
        x = self.norm(x)
        return F.silu(x)


class STFTMaskDefense(nn.Module):
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()
        raw = cfg or {}
        self.cfg = DefenseSTFTMaskConfig(
            n_fft=int(raw.get("n_fft", 512)),
            hop_length=int(raw.get("hop_length", 128)),
            win_length=int(raw.get("win_length", 512)),
            n_bands=int(raw.get("n_bands", 64)),
            hidden_channels=int(raw.get("hidden_channels", 128)),
            num_layers=int(raw.get("num_layers", 4)),
            kernel_size=int(raw.get("kernel_size", 3)),
            lookahead=int(raw.get("lookahead", 0)),
            alpha=float(raw.get("alpha", 0.15)),
        )
        self.register_buffer("window", torch.hann_window(self.cfg.win_length), persistent=False)

        layers = [
            CausalConvBlock(
                in_ch=self.cfg.n_bands,
                out_ch=self.cfg.hidden_channels,
                kernel_size=self.cfg.kernel_size,
                lookahead=self.cfg.lookahead,
            )
        ]
        for _ in range(self.cfg.num_layers - 1):
            layers.append(
                CausalConvBlock(
                    in_ch=self.cfg.hidden_channels,
                    out_ch=self.cfg.hidden_channels,
                    kernel_size=self.cfg.kernel_size,
                    lookahead=self.cfg.lookahead,
                )
            )
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv1d(self.cfg.hidden_channels, self.cfg.n_bands, kernel_size=1)

    def _to_band_features(self, log_mag: torch.Tensor) -> torch.Tensor:
        # log_mag: [B, F, N] -> [B, BANDS, N]
        bsz, n_freq, n_frames = log_mag.shape
        x = log_mag.permute(0, 2, 1).reshape(bsz * n_frames, 1, n_freq)
        x = F.adaptive_avg_pool1d(x, self.cfg.n_bands)
        x = x.reshape(bsz, n_frames, self.cfg.n_bands).permute(0, 2, 1)
        return x

    def _band_gain_to_bin_gain(self, band_gain: torch.Tensor, n_freq: int) -> torch.Tensor:
        # band_gain: [B, BANDS, N] -> [B, F, N]
        x = band_gain.permute(0, 2, 1)  # [B, N, BANDS]
        x = F.interpolate(x, size=n_freq, mode="linear", align_corners=True)
        return x.permute(0, 2, 1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: [B, T] or [T]
        Returns:
            defended waveform with same shape.
        """
        squeeze = False
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            squeeze = True
        if wav.dim() != 2:
            raise ValueError(f"Expected [B, T] or [T], got {tuple(wav.shape)}")

        length = wav.shape[-1]
        window = self.window.to(wav.device)
        spec = torch.stft(
            wav,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        mag = spec.abs().clamp_min(1e-6)
        log_mag = torch.log1p(mag)
        band_feat = self._to_band_features(log_mag)
        hidden = self.backbone(band_feat)
        raw_gain = self.head(hidden)
        band_gain = 1.0 + self.cfg.alpha * torch.tanh(raw_gain)
        bin_gain = self._band_gain_to_bin_gain(band_gain, n_freq=spec.size(1))
        defended_spec = spec * bin_gain
        defended = torch.istft(
            defended_spec,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            center=True,
            length=length,
        )
        return defended.squeeze(0) if squeeze else defended
