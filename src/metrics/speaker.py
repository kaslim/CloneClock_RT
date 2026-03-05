#!/usr/bin/env python3
"""Speaker embedding metrics: cosine similarity and optional EER."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


@dataclass
class SpeakerEncoderConfig:
    encoder_name: str = "WAV2VEC2_BASE"
    sample_rate: int = 16000
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints/speaker_encoders/speechbrain_ecapa"
    checkpoint_dir_xvector: str = "checkpoints/speaker_encoders/speechbrain_xvector"


def cosine_sim(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D embeddings."""
    x = F.normalize(x.flatten(), dim=0)
    y = F.normalize(y.flatten(), dim=0)
    return float(torch.dot(x, y).item())


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Equal Error Rate from score-label pairs."""
    # labels: 1 for same speaker, 0 for different speaker
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    pos = np.sum(sorted_labels == 1)
    neg = np.sum(sorted_labels == 0)
    if pos == 0 or neg == 0:
        return float("nan")

    fnr = np.cumsum(sorted_labels == 1) / pos
    fpr = (neg - np.cumsum(sorted_labels == 0)) / neg
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fnr[idx] + fpr[idx]) / 2.0)


class SpeakerMetric:
    """Speaker embedding extractor and similarity metric."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        repo_root = Path(__file__).resolve().parents[2]
        self.cfg = SpeakerEncoderConfig(
            encoder_name=str(cfg.get("encoder_name", "WAV2VEC2_BASE")),
            sample_rate=int(cfg.get("sample_rate", 16000)),
            device=str(cfg.get("device", "cpu")),
            checkpoint_dir=str(
                cfg.get("checkpoint_dir", str(repo_root / "checkpoints" / "speaker_encoders" / "speechbrain_ecapa"))
            ),
            checkpoint_dir_xvector=str(
                cfg.get("checkpoint_dir_xvector", str(repo_root / "checkpoints" / "speaker_encoders" / "speechbrain_xvector"))
            ),
        )
        self.device = torch.device(self.cfg.device)
        self.backend = "torchaudio"
        self.model = None
        self.sb_model = None
        self.bundle_sr = self.cfg.sample_rate
        self.model, self.bundle_sr = self._load_encoder(self.cfg.encoder_name)
        if self.model is not None:
            self.model = self.model.to(self.device).eval()

    def _load_encoder(self, encoder_name: str) -> Tuple[Optional[torch.nn.Module], int]:
        enc = encoder_name.lower()
        registry = {
            "speechbrain_ecapa": self._load_speechbrain_ecapa,
            "ecapa_tdnn": self._load_speechbrain_ecapa,
            "speechbrain/spkrec-ecapa-voxceleb": self._load_speechbrain_ecapa,
            "speechbrain_xvector": self._load_speechbrain_xvector,
            "xvector": self._load_speechbrain_xvector,
            "speechbrain/spkrec-xvect-voxceleb": self._load_speechbrain_xvector,
        }
        if enc in registry:
            return registry[enc]()
        if not hasattr(torchaudio.pipelines, encoder_name):
            raise ValueError(
                f"Encoder bundle '{encoder_name}' not found in torchaudio.pipelines. "
                "Try WAV2VEC2_BASE / WAV2VEC2_LARGE, speechbrain_ecapa, or speechbrain_xvector."
            )
        bundle = getattr(torchaudio.pipelines, encoder_name)
        model = bundle.get_model()
        sample_rate = int(bundle.sample_rate)
        return model, sample_rate

    def _load_speechbrain_ecapa(self) -> Tuple[Optional[torch.nn.Module], int]:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception as e:
            raise ImportError(
                "speechbrain is required for speechbrain_ecapa. "
                "Install with: pip install speechbrain huggingface_hub"
            ) from e

        run_opts = {"device": str(self.device)}
        savedir = str(Path(self.cfg.checkpoint_dir).resolve())
        self.sb_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts=run_opts,
        )
        self.backend = "speechbrain_ecapa"
        return None, 16000

    def _load_speechbrain_xvector(self) -> Tuple[Optional[torch.nn.Module], int]:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception as e:
            raise ImportError(
                "speechbrain is required for speechbrain_xvector. "
                "Install with: pip install speechbrain huggingface_hub"
            ) from e

        run_opts = {"device": str(self.device)}
        savedir = str(Path(self.cfg.checkpoint_dir_xvector).resolve())
        self.sb_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=savedir,
            run_opts=run_opts,
        )
        self.backend = "speechbrain_xvector"
        return None, 16000

    @torch.inference_mode()
    def embed(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Extract 1D embedding from waveform."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() != 2:
            raise ValueError(f"Expected [T] or [1, T], got {wav.shape}")

        if sample_rate != self.bundle_sr:
            wav = torchaudio.functional.resample(wav, sample_rate, self.bundle_sr)

        wav = wav.to(self.device)
        if self.backend in {"speechbrain_ecapa", "speechbrain_xvector"}:
            if self.sb_model is None:
                raise RuntimeError("SpeechBrain speaker model is not initialized.")
            emb = self.sb_model.encode_batch(wav)
            return F.normalize(emb.squeeze(0).squeeze(0).detach().cpu(), dim=0)

        # Wav2Vec2 model returns [B, T, C]; mean pooling for utterance embedding.
        if self.model is None:
            raise RuntimeError("Torchaudio speaker encoder is not initialized.")
        feats, _ = self.model.extract_features(wav)
        emb = feats[-1].mean(dim=1).squeeze(0).detach().cpu()
        return F.normalize(emb, dim=0)

    def cosine(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        return cosine_sim(emb_a, emb_b)

    @torch.inference_mode()
    def pair_cosine(
        self,
        wav_a: torch.Tensor,
        sr_a: int,
        wav_b: torch.Tensor,
        sr_b: int,
    ) -> float:
        emb_a = self.embed(wav_a, sr_a)
        emb_b = self.embed(wav_b, sr_b)
        return self.cosine(emb_a, emb_b)

    def batch_statistics(self, cosines: Iterable[float]) -> Dict[str, float]:
        values = np.asarray(list(cosines), dtype=np.float32)
        if values.size == 0:
            return {"count": 0, "mean": float("nan"), "std": float("nan")}
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    def batch_eer(
        self,
        positive_scores: List[float],
        negative_scores: List[float],
    ) -> float:
        scores = np.asarray(positive_scores + negative_scores, dtype=np.float32)
        labels = np.asarray([1] * len(positive_scores) + [0] * len(negative_scores), dtype=np.int32)
        return compute_eer(scores, labels)
