#!/usr/bin/env python3
"""ASR metric with torchaudio pipeline + jiwer WER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torchaudio
from jiwer import wer


@dataclass
class ASRConfig:
    bundle_name: str = "WAV2VEC2_ASR_BASE_960H"
    device: str = "cpu"


class ASRMetric:
    """ASR transcription and WER computation."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.cfg = ASRConfig(
            bundle_name=str(cfg.get("bundle_name", "WAV2VEC2_ASR_BASE_960H")),
            device=str(cfg.get("device", "cpu")),
        )
        self.device = torch.device(self.cfg.device)
        if not hasattr(torchaudio.pipelines, self.cfg.bundle_name):
            raise ValueError(f"ASR bundle '{self.cfg.bundle_name}' not found.")
        self.bundle = getattr(torchaudio.pipelines, self.cfg.bundle_name)
        self.model = self.bundle.get_model().to(self.device).eval()
        self.labels = self.bundle.get_labels()
        self.sample_rate = int(self.bundle.sample_rate)

    @torch.inference_mode()
    def transcribe(self, wav: torch.Tensor, sample_rate: int) -> str:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() != 2:
            raise ValueError(f"Expected [T] or [1, T], got {wav.shape}")
        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)
        wav = wav.to(self.device)
        emission, _ = self.model(wav)
        token_ids = torch.argmax(emission[0], dim=-1).detach().cpu().tolist()
        return self._ctc_greedy_decode(token_ids)

    def _ctc_greedy_decode(self, token_ids: List[int]) -> str:
        blank = 0
        dedup = []
        prev = None
        for t in token_ids:
            if t != prev:
                dedup.append(t)
            prev = t
        tokens = [self.labels[t] for t in dedup if t != blank]
        text = "".join(tokens).replace("|", " ").strip()
        return " ".join(text.split())

    def compute_wer(self, references: Iterable[str], hypotheses: Iterable[str]) -> float:
        refs = list(references)
        hyps = list(hypotheses)
        if len(refs) == 0:
            return float("nan")
        return float(wer(refs, hyps))
