#!/usr/bin/env python3
"""Telephony channel simulation transform."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt


@dataclass
class TelephonyConfig:
    sample_rate: int = 16000
    bandlimit_enabled: bool = True
    bandlimit_low_hz: int = 300
    bandlimit_high_hz: int = 3400
    bandlimit_order: int = 6
    resample_enabled: bool = True
    narrowband_rate: int = 8000
    gain_enabled: bool = True
    gain_db_min: float = -6.0
    gain_db_max: float = 6.0
    compressor_enabled: bool = True
    compressor_threshold_db: float = -20.0
    compressor_ratio: float = 2.0
    noise_enabled: bool = True
    noise_snr_db_min: float = 20.0
    noise_snr_db_max: float = 35.0
    codec_enabled: bool = False
    codec_name: str = "opus"
    codec_bitrate: str = "16k"

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TelephonyConfig":
        return cls(
            sample_rate=int(cfg.get("sample_rate", 16000)),
            bandlimit_enabled=bool(cfg.get("bandlimit", {}).get("enabled", True)),
            bandlimit_low_hz=int(cfg.get("bandlimit", {}).get("low_hz", 300)),
            bandlimit_high_hz=int(cfg.get("bandlimit", {}).get("high_hz", 3400)),
            bandlimit_order=int(cfg.get("bandlimit", {}).get("order", 6)),
            resample_enabled=bool(cfg.get("resample", {}).get("enabled", True)),
            narrowband_rate=int(cfg.get("resample", {}).get("narrowband_rate", 8000)),
            gain_enabled=bool(cfg.get("gain", {}).get("enabled", True)),
            gain_db_min=float(cfg.get("gain", {}).get("db_min", -6.0)),
            gain_db_max=float(cfg.get("gain", {}).get("db_max", 6.0)),
            compressor_enabled=bool(cfg.get("agc_like", {}).get("compressor_enabled", True)),
            compressor_threshold_db=float(cfg.get("agc_like", {}).get("threshold_db", -20.0)),
            compressor_ratio=float(cfg.get("agc_like", {}).get("ratio", 2.0)),
            noise_enabled=bool(cfg.get("noise", {}).get("enabled", True)),
            noise_snr_db_min=float(cfg.get("noise", {}).get("snr_db_min", 20.0)),
            noise_snr_db_max=float(cfg.get("noise", {}).get("snr_db_max", 35.0)),
            codec_enabled=bool(cfg.get("codec", {}).get("enabled", False)),
            codec_name=str(cfg.get("codec", {}).get("name", "opus")),
            codec_bitrate=str(cfg.get("codec", {}).get("bitrate", "16k")),
        )


class TelephonyTransform:
    """Apply telephony style degradations to waveform."""

    def __init__(self, config: Dict[str, Any] | TelephonyConfig, seed: int = 42):
        self.cfg = config if isinstance(config, TelephonyConfig) else TelephonyConfig.from_dict(config)
        self.rng = np.random.default_rng(seed)

    def __call__(self, audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        params = self.sample_params()
        return self.apply_with_params(audio, sample_rate=sample_rate, params=params)

    def sample_params(self) -> Dict[str, Any]:
        return {
            "gain_db": float(self.rng.uniform(self.cfg.gain_db_min, self.cfg.gain_db_max)),
            "snr_db": float(self.rng.uniform(self.cfg.noise_snr_db_min, self.cfg.noise_snr_db_max)),
            "noise_seed": int(self.rng.integers(0, 2**31 - 1)),
        }

    def apply_with_params(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        params: Dict[str, Any] | None = None,
    ) -> np.ndarray:
        params = params or self.sample_params()
        x, sr = self._ensure_mono_float32(audio, sample_rate or self.cfg.sample_rate)

        if self.cfg.bandlimit_enabled:
            x = self._bandlimit(x, sr)
        if self.cfg.resample_enabled:
            x = self._narrowband_resample(x, sr)
        if self.cfg.gain_enabled:
            x = self._apply_gain(x, float(params.get("gain_db", 0.0)))
        if self.cfg.compressor_enabled:
            x = self._compress(x)
        if self.cfg.noise_enabled:
            x = self._add_noise(
                x,
                snr_db=float(params.get("snr_db", self.cfg.noise_snr_db_max)),
                noise_seed=int(params.get("noise_seed", 0)),
            )
        if self.cfg.codec_enabled:
            x = self._codec_roundtrip(x, sr)

        return np.clip(x, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _ensure_mono_float32(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 2:
            x = np.mean(x, axis=1)
        elif x.ndim != 1:
            raise ValueError(f"Expected 1D/2D waveform, got shape={x.shape}")
        return x, int(sample_rate)

    def _bandlimit(self, x: np.ndarray, sr: int) -> np.ndarray:
        low = self.cfg.bandlimit_low_hz
        high = min(self.cfg.bandlimit_high_hz, sr // 2 - 100)
        if not (0 < low < high):
            return x
        sos = butter(
            N=self.cfg.bandlimit_order,
            Wn=[low, high],
            btype="bandpass",
            fs=sr,
            output="sos",
        )
        return sosfiltfilt(sos, x).astype(np.float32)

    def _narrowband_resample(self, x: np.ndarray, sr: int) -> np.ndarray:
        nb = self.cfg.narrowband_rate
        if nb <= 0 or nb >= sr:
            return x
        down = resample_poly(x, up=nb, down=sr)
        up = resample_poly(down, up=sr, down=nb)
        if len(up) > len(x):
            up = up[: len(x)]
        elif len(up) < len(x):
            up = np.pad(up, (0, len(x) - len(up)))
        return up.astype(np.float32)

    def _apply_gain(self, x: np.ndarray, gain_db: float) -> np.ndarray:
        gain = 10.0 ** (gain_db / 20.0)
        return (x * gain).astype(np.float32)

    def _compress(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-8
        abs_x = np.maximum(np.abs(x), eps)
        sign = np.sign(x)
        db = 20.0 * np.log10(abs_x)
        over_db = np.maximum(db - self.cfg.compressor_threshold_db, 0.0)
        reduced_db = db - over_db * (1.0 - 1.0 / max(self.cfg.compressor_ratio, 1.0))
        y = sign * (10.0 ** (reduced_db / 20.0))
        return y.astype(np.float32)

    def _add_noise(self, x: np.ndarray, snr_db: float, noise_seed: int) -> np.ndarray:
        sig_power = float(np.mean(np.square(x)) + 1e-12)
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise_rng = np.random.default_rng(noise_seed)
        noise = noise_rng.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(np.float32)
        return (x + noise).astype(np.float32)

    def _codec_roundtrip(self, x: np.ndarray, sr: int) -> np.ndarray:
        codec_name = self.cfg.codec_name.lower()
        if codec_name not in {"opus", "g711", "pcm_mulaw", "mulaw"}:
            return x
        if shutil.which("ffmpeg") is None:
            if codec_name in {"g711", "pcm_mulaw", "mulaw"}:
                return self._mulaw_roundtrip_numpy(x)
            return x

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_wav = tmp / "in.wav"
            out_codec = tmp / ("tmp.opus" if codec_name == "opus" else "tmp.wav")
            out_wav = tmp / "out.wav"

            sf.write(in_wav, x, sr, subtype="PCM_16")
            if codec_name == "opus":
                enc = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(in_wav),
                    "-c:a",
                    "libopus",
                    "-b:a",
                    self.cfg.codec_bitrate,
                    str(out_codec),
                ]
            else:
                enc = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(in_wav),
                    "-c:a",
                    "pcm_mulaw",
                    str(out_codec),
                ]
            dec = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(out_codec),
                "-ar",
                str(sr),
                "-ac",
                "1",
                str(out_wav),
            ]
            try:
                subprocess.run(enc, check=True)
                subprocess.run(dec, check=True)
                y, _ = sf.read(out_wav, dtype="float32")
                if y.ndim == 2:
                    y = np.mean(y, axis=1)
                if len(y) > len(x):
                    y = y[: len(x)]
                elif len(y) < len(x):
                    y = np.pad(y, (0, len(x) - len(y)))
                return y.astype(np.float32)
            except Exception:
                if codec_name in {"g711", "pcm_mulaw", "mulaw"}:
                    return self._mulaw_roundtrip_numpy(x)
                return x

    @staticmethod
    def _mulaw_roundtrip_numpy(x: np.ndarray, mu: int = 255) -> np.ndarray:
        # Reproducible fallback without ffmpeg: mu-law compand and expand.
        x = np.clip(x, -1.0, 1.0).astype(np.float32)
        mag = np.log1p(mu * np.abs(x)) / np.log1p(mu)
        companded = np.sign(x) * mag
        q = np.round((companded + 1.0) * 0.5 * mu).astype(np.int32)
        q = np.clip(q, 0, mu)
        y = (q.astype(np.float32) / mu) * 2.0 - 1.0
        mag2 = (1.0 / mu) * ((1.0 + mu) ** np.abs(y) - 1.0)
        decoded = np.sign(y) * mag2
        return np.clip(decoded, -1.0, 1.0).astype(np.float32)
