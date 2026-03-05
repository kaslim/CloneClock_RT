#!/usr/bin/env python3
"""run_train.py — CloneBlock-RT v0/v0.1/v0p2 training entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from speechbrain.inference.speaker import EncoderClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.defense_stftmask import STFTMaskDefense
from src.metrics.session_attack import pick_random_k


def get_data_paths(repo_root: Path, dataset: str):
    return {
        "processed_dir": repo_root / "data" / "processed" / dataset,
        "splits_dir": repo_root / "data" / "splits" / dataset,
    }


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != 16000:
        wav = torch.from_numpy(x)
        wav = torchaudio.functional.resample(wav, sr, 16000)
        x = wav.numpy()
    return x.astype(np.float32)


def pad_batch(wavs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([int(w.numel()) for w in wavs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = []
    for w in wavs:
        if w.numel() < max_len:
            w = F.pad(w, (0, max_len - w.numel()))
        padded.append(w)
    return torch.stack(padded, dim=0), lengths


class TorchTelephonyAug:
    """Torch-only telephony augmentation with shared sampled params."""

    def __init__(self, sample_rate: int = 16000, seed: int = 42):
        self.sample_rate = sample_rate
        self.nb_rate = 8000
        self.rng = np.random.default_rng(seed)

    def sample_params(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        gain_db = torch.from_numpy(
            self.rng.uniform(-4.0, 4.0, size=(batch_size, 1)).astype(np.float32)
        ).to(device=device)
        snr_db = torch.from_numpy(
            self.rng.uniform(22.0, 35.0, size=(batch_size, 1)).astype(np.float32)
        ).to(device=device)
        noise_seed = int(self.rng.integers(0, 2**31 - 1))
        return {
            "gain_db": gain_db,
            "snr_db": snr_db,
            "noise_seed": noise_seed,
        }

    def sample_shared_params(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        gain_db_val = float(self.rng.uniform(-4.0, 4.0))
        snr_db_val = float(self.rng.uniform(22.0, 35.0))
        noise_seed = int(self.rng.integers(0, 2**31 - 1))
        gain_db = torch.full((batch_size, 1), gain_db_val, device=device, dtype=torch.float32)
        snr_db = torch.full((batch_size, 1), snr_db_val, device=device, dtype=torch.float32)
        return {
            "gain_db": gain_db,
            "snr_db": snr_db,
            "noise_seed": noise_seed,
        }

    def apply_with_params(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        y = torchaudio.functional.highpass_biquad(x, self.sample_rate, cutoff_freq=300.0)
        y = torchaudio.functional.lowpass_biquad(y, self.sample_rate, cutoff_freq=3400.0)
        y = torchaudio.functional.resample(y, self.sample_rate, self.nb_rate)
        y = torchaudio.functional.resample(y, self.nb_rate, self.sample_rate)
        if y.shape[-1] > x.shape[-1]:
            y = y[..., : x.shape[-1]]
        elif y.shape[-1] < x.shape[-1]:
            y = F.pad(y, (0, x.shape[-1] - y.shape[-1]))
        gain = torch.pow(10.0, params["gain_db"] / 20.0)
        y = y * gain
        sig_power = y.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
        noise_power = sig_power / torch.pow(10.0, params["snr_db"] / 10.0)
        noise_gen = torch.Generator(device=x.device)
        noise_gen.manual_seed(int(params["noise_seed"]))
        noise = torch.randn(y.shape, generator=noise_gen, device=x.device, dtype=y.dtype)
        y = y + noise * torch.sqrt(noise_power)
        return y.clamp(-1.0, 1.0)


def speaker_embed_ecapa(model: EncoderClassifier, wav: torch.Tensor) -> torch.Tensor:
    lens = torch.ones((wav.shape[0],), device=wav.device)
    emb = model.encode_batch(wav, lens).squeeze(1)
    emb = F.normalize(emb, dim=-1)
    return emb


def stft_l1(a: torch.Tensor, b: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=a.device)
    sa = torch.stft(a, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, center=False, return_complex=True)
    sb = torch.stft(b, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, center=False, return_complex=True)
    return (sa.abs() - sb.abs()).abs().mean()


def multi_res_stft_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    losses = [
        stft_l1(a, b, n_fft=256, hop=64),
        stft_l1(a, b, n_fft=512, hop=128),
        stft_l1(a, b, n_fft=1024, hop=256),
    ]
    return sum(losses) / len(losses)


def sample_batch(rows: List[Dict[str, str]], batch_size: int) -> List[Dict[str, str]]:
    return random.sample(rows, k=batch_size) if len(rows) >= batch_size else random.choices(rows, k=batch_size)


def sample_same_speaker_group(
    speaker_rows: Dict[str, List[Dict[str, str]]],
    group_size: int,
) -> List[Dict[str, str]]:
    speakers = list(speaker_rows.keys())
    if not speakers:
        return []
    spk = random.choice(speakers)
    rows = speaker_rows[spk]
    if len(rows) >= group_size:
        return random.sample(rows, k=group_size)
    return random.choices(rows, k=group_size)


def sample_same_speaker_group_with_id(
    speaker_rows: Dict[str, List[Dict[str, str]]],
    group_size: int,
) -> Tuple[str, List[Dict[str, str]]]:
    speakers = list(speaker_rows.keys())
    if not speakers:
        return "", []
    spk = random.choice(speakers)
    rows = speaker_rows[spk]
    if len(rows) >= group_size:
        return spk, random.sample(rows, k=group_size)
    return spk, random.choices(rows, k=group_size)


def sample_negative_speaker_group(
    speaker_rows: Dict[str, List[Dict[str, str]]],
    exclude_spk: str,
    group_size: int,
) -> List[Dict[str, str]]:
    neg_speakers = [s for s in speaker_rows.keys() if s != exclude_spk]
    if not neg_speakers:
        return []
    spk = random.choice(neg_speakers)
    rows = speaker_rows[spk]
    if len(rows) >= group_size:
        return random.sample(rows, k=group_size)
    return random.choices(rows, k=group_size)


def choose_rank_k(rank_k_set: List[int], rank_k_probs: List[float]) -> int:
    if not rank_k_set:
        return 1
    if len(rank_k_probs) != len(rank_k_set):
        return int(random.choice(rank_k_set))
    return int(random.choices(rank_k_set, weights=rank_k_probs, k=1)[0])


def build_pseudo_targets(
    speaker_model: EncoderClassifier,
    speaker_rows: Dict[str, List[Dict[str, str]]],
    processed_wav_dir: Path,
    device: torch.device,
    max_rows_per_speaker: int = 12,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Build per-speaker centroid and pseudo-target speaker mapping."""
    centroids: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for spk, rows in speaker_rows.items():
            chosen = rows[:max_rows_per_speaker]
            embs = []
            for r in chosen:
                wav = read_wav(processed_wav_dir / r["path"])
                x = torch.from_numpy(wav).unsqueeze(0).to(device)
                z = speaker_embed_ecapa(speaker_model, x).squeeze(0)
                embs.append(z)
            if embs:
                c = F.normalize(torch.stack(embs, dim=0).mean(dim=0), dim=0)
                centroids[spk] = c.detach().clone()

    speakers = sorted(centroids.keys())
    pseudo_map: Dict[str, str] = {}
    if len(speakers) >= 2:
        C = torch.stack([centroids[s] for s in speakers], dim=0)
        sim = C @ C.t()
        eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e9)
        best_idx = torch.argmax(sim, dim=1)
        for i, s in enumerate(speakers):
            pseudo_map[s] = speakers[int(best_idx[i].item())]
    else:
        for s in speakers:
            pseudo_map[s] = s
    return centroids, pseudo_map


def resolve_wav_path(repo_root: Path, processed_wav_dir: Path, rel_path: str) -> Path:
    p1 = repo_root / rel_path
    if p1.exists():
        return p1
    p2 = processed_wav_dir / rel_path
    if p2.exists():
        return p2
    return p1


def load_session_pool(session_pool_csv: Path) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    by_utter: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    if not session_pool_csv.exists():
        return by_utter
    with open(session_pool_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        unit_id = r.get("session_id", r.get("utter_id", "unknown"))
        key = (r["speaker_id"], unit_id, r["path"])
        by_utter.setdefault(key, []).append(r)
    for k in list(by_utter.keys()):
        by_utter[k] = sorted(by_utter[k], key=lambda x: x["chunk_id"])
    return by_utter


def build_wave_batch(
    rows: List[Dict[str, str]],
    processed_wav_dir: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    wavs = []
    for r in rows:
        wav = read_wav(processed_wav_dir / r["path"])
        wavs.append(torch.from_numpy(wav))
    x, lengths = pad_batch(wavs)
    return x.to(device), lengths.to(device)


def compute_intra_loss(
    speaker_model: EncoderClassifier,
    wav_batch: torch.Tensor,
    lengths: torch.Tensor,
    sample_rate: int = 16000,
    win_sec: float = 1.0,
    hop_sec: float = 0.5,
    rms_thr: float = 0.008,
) -> Tuple[torch.Tensor, int]:
    win = int(win_sec * sample_rate)
    hop = int(hop_sec * sample_rate)
    per_sample = []
    used = 0
    for i in range(wav_batch.size(0)):
        L = int(lengths[i].item())
        x = wav_batch[i, :L]
        if L < win:
            continue
        windows = []
        for s in range(0, L - win + 1, hop):
            seg = x[s : s + win]
            rms = torch.sqrt(torch.mean(seg.pow(2)) + 1e-10)
            if float(rms.item()) < rms_thr:
                continue
            windows.append(seg)
        if len(windows) < 3:
            continue
        used += 1
        w = torch.stack(windows, dim=0)
        z = speaker_embed_ecapa(speaker_model, w)
        cos_adj = F.cosine_similarity(z[:-1], z[1:], dim=-1).mean()
        per_sample.append(cos_adj)
    if len(per_sample) == 0:
        return torch.tensor(0.0, device=wav_batch.device), 0
    return torch.stack(per_sample).mean(), used


def evaluate_speaker_sim_tel(
    model: STFTMaskDefense,
    speaker_model: EncoderClassifier,
    aug: TorchTelephonyAug,
    val_rows: List[Dict[str, str]],
    processed_wav_dir: Path,
    batch_size: int,
    val_batches: int,
    device: torch.device,
) -> float:
    model.eval()
    sims = []
    with torch.no_grad():
        for _ in range(val_batches):
            rows = sample_batch(val_rows, batch_size)
            x, _ = build_wave_batch(rows, processed_wav_dir, device)
            x_def = model(x)
            p = aug.sample_params(batch_size=x.shape[0], device=device)
            xt = aug.apply_with_params(x, p)
            xdt = aug.apply_with_params(x_def, p)
            z1 = speaker_embed_ecapa(speaker_model, xt)
            z2 = speaker_embed_ecapa(speaker_model, xdt)
            sims.append(float(F.cosine_similarity(z1, z2, dim=-1).mean().item()))
    model.train()
    return float(np.mean(sims)) if sims else float("nan")


def evaluate_targeted_k16_proxy(
    model: STFTMaskDefense,
    speaker_model: EncoderClassifier,
    aug: TorchTelephonyAug,
    session_pool: Dict[Tuple[str, str, str], List[Dict[str, str]]],
    processed_wav_dir: Path,
    device: torch.device,
    eot_p: float,
    att_candidate_windows: int,
    use_full_session: bool,
    att_beta: float,
    val_sessions: int,
    selection_source: str = "defended",
) -> float:
    if not session_pool:
        return float("nan")
    keys = list(session_pool.keys())
    random.shuffle(keys)
    keys = keys[: max(1, min(len(keys), int(val_sessions)))]
    vals: List[float] = []
    model.eval()
    with torch.no_grad():
        for key in keys:
            _, _, path = key
            win_rows = session_pool[key]
            if not win_rows:
                continue
            wav = read_wav(resolve_wav_path(REPO_ROOT, processed_wav_dir, path))
            if use_full_session:
                sel_rows = win_rows
            else:
                c = min(len(win_rows), max(4, int(att_candidate_windows)))
                sel_rows = random.sample(win_rows, k=c) if len(win_rows) > c else win_rows

            def build_segments(rows_in: List[Dict[str, str]]) -> List[torch.Tensor]:
                out = []
                for wr in rows_in:
                    s = int(round(float(wr["start_sec"]) * 16000))
                    d = int(round(float(wr["dur_sec"]) * 16000))
                    seg = wav[s : s + d]
                    if len(seg) < d:
                        seg = np.pad(seg, (0, max(0, d - len(seg))), mode="constant")
                    out.append(torch.from_numpy(seg.astype(np.float32)))
                return out

            segs_sel = build_segments(sel_rows)
            segs_all = build_segments(win_rows)
            if not segs_sel or not segs_all:
                continue
            x_sel, _ = pad_batch(segs_sel)
            x_all, _ = pad_batch(segs_all)
            x_sel = x_sel.to(device)
            x_all = x_all.to(device)
            x_sel_def = model(x_sel)

            if random.random() < eot_p:
                p_att = aug.sample_shared_params(batch_size=x_sel.shape[0], device=device)
                p_ref = {
                    "gain_db": p_att["gain_db"][:1].expand(x_all.shape[0], 1).clone(),
                    "snr_db": p_att["snr_db"][:1].expand(x_all.shape[0], 1).clone(),
                    "noise_seed": int(p_att["noise_seed"]),
                }
                x_sel_clean_t = aug.apply_with_params(x_sel, p_att)
                x_sel_def_t = aug.apply_with_params(x_sel_def, p_att)
                x_all_clean_t = aug.apply_with_params(x_all, p_ref)
            else:
                x_sel_clean_t = x_sel
                x_sel_def_t = x_sel_def
                x_all_clean_t = x_all

            z_sel_clean = speaker_embed_ecapa(speaker_model, x_sel_clean_t)
            z_sel_def_raw = speaker_embed_ecapa(speaker_model, x_sel_def)
            z_sel_def_t = speaker_embed_ecapa(speaker_model, x_sel_def_t)
            z_all_clean = speaker_embed_ecapa(speaker_model, x_all_clean_t)
            e_ref = F.normalize(z_all_clean.mean(dim=0), dim=0)

            if selection_source == "clean":
                z_sel = z_sel_clean
            elif selection_source == "tele_defended":
                z_sel = z_sel_def_t
            else:
                z_sel = z_sel_def_raw

            beta = float(max(1e-3, att_beta))
            scores = z_sel @ e_ref
            if use_full_session:
                k_eff = min(16, int(z_sel.size(0)))
                idx = torch.topk(scores, k=k_eff, largest=True).indices
                e_def = F.normalize(z_sel_def_t[idx].mean(dim=0), dim=0)
            else:
                w = torch.softmax(beta * scores, dim=0)
                e_def = F.normalize(torch.sum(z_sel_def_t * w.unsqueeze(1), dim=0), dim=0)
            vals.append(float(torch.dot(e_def, e_ref).item()))
    model.train()
    return float(np.mean(vals)) if vals else float("nan")


def parse_args():
    p = argparse.ArgumentParser(description="CloneBlock-RT training")
    p.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "train_v0.yaml"))
    p.add_argument("--dataset", type=str, choices=["librispeech", "vctk"], default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--run_name", type=str, default="v0")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--w_spk", type=float, default=None)
    p.add_argument("--w_rec", type=float, default=None)
    p.add_argument("--w_intra", type=float, default=None)
    p.add_argument("--w_agg", type=float, default=None)
    p.add_argument("--w_rank", type=float, default=None)
    p.add_argument("--w_ceiling", type=float, default=None)
    p.add_argument("--w_kagg", type=float, default=None)
    p.add_argument("--w_att", type=float, default=None)
    p.add_argument("--w_tgt", type=float, default=None)
    p.add_argument("--w_away", type=float, default=None)
    p.add_argument("--w_energy", type=float, default=None)
    p.add_argument("--eot_prob", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train_subset_size", type=int, default=None)
    p.add_argument("--stage1_steps", type=int, default=None)
    p.add_argument("--init_checkpoint", type=str, default="")
    p.add_argument("--agg_group_size", type=int, default=None)
    p.add_argument("--rank_margin", type=float, default=None)
    p.add_argument("--rank_k_set", type=str, default=None)
    p.add_argument("--rank_k_probs", type=str, default=None)
    p.add_argument("--ceiling_tau", type=float, default=None)
    p.add_argument(
        "--train_objective",
        type=str,
        choices=["rank", "kagg", "targetshift", "kagg_session", "targetedk", "targetedk_soft_def", "targetedk_hard16_margin"],
        default="rank",
    )
    p.add_argument("--session_pool_train_csv", type=str, default="")
    p.add_argument("--att_candidate_windows", type=int, default=48)
    p.add_argument("--att_beta", type=float, default=10.0)
    p.add_argument("--att_use_full_session", action="store_true")
    p.add_argument("--att_tau", type=float, default=0.85)
    p.add_argument("--att_gamma", type=float, default=20.0)
    p.add_argument("--att_margin_mode", type=str, choices=["hinge", "logistic"], default="hinge")
    p.add_argument(
        "--att_selection_source",
        type=str,
        choices=["clean", "defended", "tele_defended"],
        default="tele_defended",
    )
    p.add_argument("--best_metric", type=str, choices=["speaker_sim_tel", "val_k16_proxy"], default="speaker_sim_tel")
    p.add_argument("--val_session_pool_csv", type=str, default="")
    p.add_argument("--val_proxy_sessions", type=int, default=12)
    p.add_argument(
        "--val_proxy_selection_source",
        type=str,
        choices=["clean", "defended", "tele_defended"],
        default="defended",
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})

    dataset = args.dataset or data_cfg.get("dataset", "vctk")
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.alpha is not None:
        model_cfg["alpha"] = float(args.alpha)
    if args.w_spk is not None:
        loss_cfg["w_spk"] = float(args.w_spk)
    if args.w_rec is not None:
        loss_cfg["w_rec"] = float(args.w_rec)
    if args.w_intra is not None:
        loss_cfg["w_intra"] = float(args.w_intra)
    if args.w_agg is not None:
        loss_cfg["w_agg"] = float(args.w_agg)
    if args.w_rank is not None:
        loss_cfg["w_rank"] = float(args.w_rank)
    if args.w_ceiling is not None:
        loss_cfg["w_ceiling"] = float(args.w_ceiling)
    if args.w_kagg is not None:
        loss_cfg["w_kagg"] = float(args.w_kagg)
    if args.w_att is not None:
        loss_cfg["w_kagg"] = float(args.w_att)
    if args.w_tgt is not None:
        loss_cfg["w_tgt"] = float(args.w_tgt)
    if args.w_away is not None:
        loss_cfg["w_away"] = float(args.w_away)
    if args.w_energy is not None:
        loss_cfg["w_energy"] = float(args.w_energy)
    if args.eot_prob is not None:
        train_cfg["eot_prob"] = float(args.eot_prob)
    if args.train_subset_size is not None:
        data_cfg["train_subset_size"] = int(args.train_subset_size)
    if args.stage1_steps is not None:
        train_cfg["stage1_steps"] = int(args.stage1_steps)
    if args.agg_group_size is not None:
        train_cfg["agg_group_size"] = int(args.agg_group_size)
    if args.rank_margin is not None:
        train_cfg["rank_margin"] = float(args.rank_margin)
    if args.rank_k_set is not None:
        train_cfg["rank_k_set"] = str(args.rank_k_set)
    if args.rank_k_probs is not None:
        train_cfg["rank_k_probs"] = str(args.rank_k_probs)
    if args.ceiling_tau is not None:
        train_cfg["ceiling_tau"] = float(args.ceiling_tau)

    paths = get_data_paths(REPO_ROOT, dataset)
    processed_wav_dir = paths["processed_dir"] / "wav16k"
    train_rows = load_rows(paths["splits_dir"] / "train.csv")
    val_rows = load_rows(paths["splits_dir"] / "val.csv")

    subset_size = int(data_cfg.get("train_subset_size", 2000))
    if subset_size > 0:
        random.shuffle(train_rows)
        train_rows = train_rows[: min(subset_size, len(train_rows))]
    val_subset = int(data_cfg.get("val_subset_size", 400))
    if val_subset > 0:
        random.shuffle(val_rows)
        val_rows = val_rows[: min(val_subset, len(val_rows))]

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    max_steps = int(args.max_steps or train_cfg.get("max_steps", 1000))
    batch_size = int(train_cfg.get("batch_size", 8))
    lr = float(train_cfg.get("lr", 1e-3))
    eot_p = float(train_cfg.get("eot_prob", 0.7))
    log_every = int(train_cfg.get("log_every", 20))
    val_every = int(train_cfg.get("val_every", 100))
    val_batches = int(train_cfg.get("val_batches", 5))
    stage1_steps = int(train_cfg.get("stage1_steps", 200))
    agg_group_size = int(train_cfg.get("agg_group_size", 4))
    rank_margin = float(train_cfg.get("rank_margin", 0.1))
    rank_k_set_str = str(train_cfg.get("rank_k_set", "1,2,4,8,16"))
    rank_k_set = [int(x.strip()) for x in rank_k_set_str.split(",") if x.strip()]
    if not rank_k_set:
        rank_k_set = [1, 2, 4, 8, 16]
    rank_k_probs_str = str(train_cfg.get("rank_k_probs", "1:0.05,2:0.10,4:0.20,8:0.25,16:0.40"))
    rank_prob_map: Dict[int, float] = {}
    for tok in rank_k_probs_str.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        k_s, p_s = tok.split(":", 1)
        try:
            rank_prob_map[int(k_s.strip())] = float(p_s.strip())
        except ValueError:
            continue
    rank_k_probs = [max(0.0, float(rank_prob_map.get(k, 1.0))) for k in rank_k_set]
    if sum(rank_k_probs) <= 0.0:
        rank_k_probs = [1.0 for _ in rank_k_set]
    ceiling_tau = float(train_cfg.get("ceiling_tau", 0.8))

    model = STFTMaskDefense(model_cfg).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    aug = TorchTelephonyAug(sample_rate=16000, seed=seed)

    ecapa_dir = REPO_ROOT / "checkpoints" / "speaker_encoders" / "speechbrain_ecapa"
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ecapa_dir),
        run_opts={"device": str(device)},
    )
    speaker_model.eval()
    for p in speaker_model.parameters():
        p.requires_grad = False

    target_w_spk = float(loss_cfg.get("w_spk", 1.0))
    w_rec = float(loss_cfg.get("w_rec", 1.0))
    target_w_intra = float(loss_cfg.get("w_intra", 0.0))
    target_w_agg = float(loss_cfg.get("w_agg", 0.0))
    target_w_rank = float(loss_cfg.get("w_rank", target_w_agg))
    target_w_ceiling = float(loss_cfg.get("w_ceiling", 0.0))
    target_w_kagg = float(loss_cfg.get("w_kagg", 0.0))
    target_w_tgt = float(loss_cfg.get("w_tgt", 0.0))
    target_w_away = float(loss_cfg.get("w_away", 0.0))
    w_energy = float(loss_cfg.get("w_energy", 0.1))
    objective = str(args.train_objective).strip().lower()

    speaker_rows: Dict[str, List[Dict[str, str]]] = {}
    for rr in train_rows:
        speaker_rows.setdefault(rr["speaker_id"], []).append(rr)
    speaker_utt_rows: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for rr in train_rows:
        speaker_utt_rows.setdefault((rr["speaker_id"], rr["utter_id"]), []).append(rr)
    session_pool_train_csv = (
        Path(args.session_pool_train_csv)
        if args.session_pool_train_csv
        else (
            REPO_ROOT / "data" / "splits" / "session_pool_v3_train.csv"
            if objective in {"targetedk", "targetedk_soft_def", "targetedk_hard16_margin"}
            else (REPO_ROOT / "data" / "splits" / "session_pool_train.csv")
        )
    )
    session_pool = load_session_pool(session_pool_train_csv) if objective in {"kagg_session", "targetedk", "targetedk_soft_def", "targetedk_hard16_margin"} else {}
    session_pool_keys = [k for k, v in session_pool.items() if len(v) > 0]
    val_session_pool_csv = (
        Path(args.val_session_pool_csv)
        if args.val_session_pool_csv
        else (
            REPO_ROOT / "data" / "splits" / "session_pool_v3_val.csv"
            if objective in {"targetedk", "targetedk_soft_def", "targetedk_hard16_margin"}
            else (REPO_ROOT / "data" / "splits" / "session_pool_val.csv")
        )
    )
    val_session_pool = load_session_pool(val_session_pool_csv) if objective in {"targetedk", "targetedk_soft_def", "targetedk_hard16_margin"} else {}

    ckpt_dir = REPO_ROOT / "checkpoints" / "defense"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"{args.run_name}_best.pt"
    curve_csv = REPO_ROOT / "artifacts" / "tables" / f"train_{args.run_name}_curve.csv"
    curve_csv.parent.mkdir(parents=True, exist_ok=True)
    log_path = REPO_ROOT / "artifacts" / "logs" / f"train_{args.run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_metric_value = float("inf")
    best_val_proxy = float("inf")
    best_step = -1
    t0 = time.time()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pseudo_centroids: Dict[str, torch.Tensor] = {}
    pseudo_map: Dict[str, str] = {}
    if objective == "targetshift":
        pseudo_centroids, pseudo_map = build_pseudo_targets(
            speaker_model=speaker_model,
            speaker_rows=speaker_rows,
            processed_wav_dir=processed_wav_dir,
            device=device,
        )
        tables_dir = REPO_ROOT / "artifacts" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        map_csv = tables_dir / "pseudo_target_map.csv"
        with open(map_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["speaker_id", "target_speaker_id"])
            w.writeheader()
            for s in sorted(pseudo_map.keys()):
                w.writerow({"speaker_id": s, "target_speaker_id": pseudo_map[s]})
        speakers_sorted = sorted(pseudo_centroids.keys())
        centroid_mat = torch.stack([pseudo_centroids[s] for s in speakers_sorted], dim=0).cpu().numpy()
        np.save(str(tables_dir / "pseudo_centroids.npy"), {"speakers": speakers_sorted, "centroids": centroid_mat}, allow_pickle=True)

    with open(curve_csv, "w", newline="", encoding="utf-8") as cf, open(log_path, "w", encoding="utf-8") as lf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "step",
                "loss_total",
                "loss_spk_global",
                "loss_intra",
                "loss_rank",
                "loss_ceiling",
                "loss_kagg",
                "loss_tgt",
                "loss_away",
                "loss_rec",
                "loss_energy",
                "k16_proxy",
                "w_spk_eff",
                "w_intra_eff",
                "w_rank_eff",
                "w_ceiling_eff",
                "w_kagg_eff",
                "w_tgt_eff",
                "w_away_eff",
                "speaker_sim_tel",
                "n_windows_used",
                "rank_k",
                "val_speaker_sim_tel",
                "val_k16_proxy",
                "best_step",
                "best_val_speaker_sim_tel",
                "stoi_optional",
                "wer_pseudo_optional",
            ],
        )
        writer.writeheader()
        rank_k_hist: Dict[int, int] = {k: 0 for k in rank_k_set}

        for step in range(1, max_steps + 1):
            rows = sample_batch(train_rows, batch_size)
            x, lengths = build_wave_batch(rows, processed_wav_dir, device)
            x_def = model(x)

            if random.random() < eot_p:
                p = aug.sample_params(batch_size=x.shape[0], device=device)
                x_for_spk = aug.apply_with_params(x, p)
                x_def_for_spk = aug.apply_with_params(x_def, p)
            else:
                x_for_spk = x
                x_def_for_spk = x_def

            z_x = speaker_embed_ecapa(speaker_model, x_for_spk)
            z_def = speaker_embed_ecapa(speaker_model, x_def_for_spk)
            cos_tel_vec = F.cosine_similarity(z_x, z_def, dim=-1)
            loss_spk_global = cos_tel_vec.mean()
            loss_rec = multi_res_stft_loss(x_def, x)
            rms_x = torch.sqrt(torch.mean(x.pow(2), dim=-1) + 1e-8)
            rms_def = torch.sqrt(torch.mean(x_def.pow(2), dim=-1) + 1e-8)
            loss_energy = torch.mean(torch.abs(rms_x - rms_def))
            loss_intra, n_windows_used = compute_intra_loss(
                speaker_model=speaker_model,
                wav_batch=x_def_for_spk,
                lengths=lengths,
                sample_rate=16000,
            )

            rank_k = choose_rank_k(rank_k_set, rank_k_probs)
            if objective == "targetedk_hard16_margin":
                rank_k = 16
            rank_k_hist[rank_k] = rank_k_hist.get(rank_k, 0) + 1
            loss_kagg = torch.tensor(0.0, device=device)
            loss_tgt = torch.tensor(0.0, device=device)
            loss_away = torch.tensor(0.0, device=device)
            k16_proxy = torch.tensor(float("nan"), device=device)
            anchor_spk, pos_rows = sample_same_speaker_group_with_id(speaker_rows, rank_k)
            if objective in {"rank", "kagg"}:
                utt_cands = [k for k, vv in speaker_utt_rows.items() if k[0] == anchor_spk and len(vv) > 0]
                if utt_cands:
                    _, u = random.choice(utt_cands)
                    u_rows = speaker_utt_rows[(anchor_spk, u)]
                    if len(u_rows) >= rank_k:
                        pos_rows = random.sample(u_rows, k=rank_k)
                    else:
                        pos_rows = random.choices(u_rows, k=rank_k)
            if objective == "kagg_session" and session_pool_keys:
                anchor_spk, anchor_utt, anchor_path = random.choice(session_pool_keys)
                win_rows = session_pool[(anchor_spk, anchor_utt, anchor_path)]
                wav_full = read_wav(resolve_wav_path(REPO_ROOT, processed_wav_dir, anchor_path))
                idx = pick_random_k(n=len(win_rows), k=rank_k, rng=random.Random(seed + step))
                segs = []
                for ii in idx:
                    wr = win_rows[ii]
                    s = int(round(float(wr["start_sec"]) * 16000))
                    d = int(round(float(wr["dur_sec"]) * 16000))
                    seg = wav_full[s : s + d]
                    if len(seg) < d:
                        seg = np.pad(seg, (0, max(0, d - len(seg))), mode="constant")
                    segs.append(torch.from_numpy(seg.astype(np.float32)))
                if len(segs) > 0:
                    x_pos, _ = pad_batch(segs)
                    x_pos = x_pos.to(device)
                    x_pos_def = model(x_pos)
                    if random.random() < eot_p:
                        p_rank = aug.sample_shared_params(batch_size=x_pos.shape[0], device=device)
                        x_pos_clean_t = aug.apply_with_params(x_pos, p_rank)
                        x_pos_def_t = aug.apply_with_params(x_pos_def, p_rank)
                    else:
                        x_pos_clean_t = x_pos
                        x_pos_def_t = x_pos_def
                    z_pos_clean = speaker_embed_ecapa(speaker_model, x_pos_clean_t)
                    z_pos_def = speaker_embed_ecapa(speaker_model, x_pos_def_t)
                    e_ref = F.normalize(z_pos_clean.mean(dim=0), dim=0)
                    e_def = F.normalize(z_pos_def.mean(dim=0), dim=0)
                    loss_kagg = torch.dot(e_def, e_ref)
                loss_rank = torch.tensor(0.0, device=device)
                loss_ceiling = torch.tensor(0.0, device=device)
            elif objective == "targetedk" and session_pool_keys:
                anchor_spk, anchor_unit, anchor_path = random.choice(session_pool_keys)
                win_rows = session_pool[(anchor_spk, anchor_unit, anchor_path)]
                wav_full = read_wav(resolve_wav_path(REPO_ROOT, processed_wav_dir, anchor_path))
                if len(win_rows) > 0:
                    c = min(len(win_rows), max(4, int(args.att_candidate_windows)))
                    cand_idx = random.sample(range(len(win_rows)), k=c) if len(win_rows) > c else list(range(len(win_rows)))
                    segs = []
                    for ii in cand_idx:
                        wr = win_rows[ii]
                        s = int(round(float(wr["start_sec"]) * 16000))
                        d = int(round(float(wr["dur_sec"]) * 16000))
                        seg = wav_full[s : s + d]
                        if len(seg) < d:
                            seg = np.pad(seg, (0, max(0, d - len(seg))), mode="constant")
                        segs.append(torch.from_numpy(seg.astype(np.float32)))
                    if segs:
                        x_cand, _ = pad_batch(segs)
                        x_cand = x_cand.to(device)
                        x_cand_def = model(x_cand)
                        if random.random() < eot_p:
                            p_att = aug.sample_shared_params(batch_size=x_cand.shape[0], device=device)
                            x_cand_clean_t = aug.apply_with_params(x_cand, p_att)
                            x_cand_def_t = aug.apply_with_params(x_cand_def, p_att)
                        else:
                            x_cand_clean_t = x_cand
                            x_cand_def_t = x_cand_def
                        z_cand_clean = speaker_embed_ecapa(speaker_model, x_cand_clean_t)
                        z_cand_def = speaker_embed_ecapa(speaker_model, x_cand_def_t)
                        e_ref = F.normalize(z_cand_clean.mean(dim=0), dim=0)
                        s_clean = z_cand_clean @ e_ref
                        k_eff = min(int(rank_k), int(z_cand_clean.size(0)))
                        top_idx = torch.topk(s_clean, k=k_eff, largest=True).indices
                        e_def = F.normalize(z_cand_def[top_idx].mean(dim=0), dim=0)
                        loss_kagg = torch.dot(e_def, e_ref)
                loss_rank = torch.tensor(0.0, device=device)
                loss_ceiling = torch.tensor(0.0, device=device)
            elif objective == "targetedk_soft_def" and session_pool_keys:
                anchor_spk, anchor_unit, anchor_path = random.choice(session_pool_keys)
                win_rows = session_pool[(anchor_spk, anchor_unit, anchor_path)]
                wav_full = read_wav(resolve_wav_path(REPO_ROOT, processed_wav_dir, anchor_path))
                if len(win_rows) > 0:
                    c = min(len(win_rows), max(4, int(args.att_candidate_windows)))
                    cand_idx = random.sample(range(len(win_rows)), k=c) if len(win_rows) > c else list(range(len(win_rows)))
                    segs_cand: List[torch.Tensor] = []
                    segs_all: List[torch.Tensor] = []
                    for ii in cand_idx:
                        wr = win_rows[ii]
                        s = int(round(float(wr["start_sec"]) * 16000))
                        d = int(round(float(wr["dur_sec"]) * 16000))
                        seg = wav_full[s : s + d]
                        if len(seg) < d:
                            seg = np.pad(seg, (0, max(0, d - len(seg))), mode="constant")
                        segs_cand.append(torch.from_numpy(seg.astype(np.float32)))
                    for wr in win_rows:
                        s = int(round(float(wr["start_sec"]) * 16000))
                        d = int(round(float(wr["dur_sec"]) * 16000))
                        seg = wav_full[s : s + d]
                        if len(seg) < d:
                            seg = np.pad(seg, (0, max(0, d - len(seg))), mode="constant")
                        segs_all.append(torch.from_numpy(seg.astype(np.float32)))
                    if segs_cand and segs_all:
                        x_cand, _ = pad_batch(segs_cand)
                        x_all, _ = pad_batch(segs_all)
                        x_cand = x_cand.to(device)
                        x_all = x_all.to(device)
                        x_cand_def = model(x_cand)
                        # Use shared telephony params so train/eval threat model stays aligned.
                        if random.random() < eot_p:
                            p_att = aug.sample_shared_params(batch_size=x_cand.shape[0], device=device)
                            x_cand_clean_t = aug.apply_with_params(x_cand, p_att)
                            x_cand_def_t = aug.apply_with_params(x_cand_def, p_att)
                            # ref uses exactly the same sampled channel params as attacker branch
                            p_ref = {
                                "gain_db": p_att["gain_db"][:1].expand(x_all.shape[0], 1).clone(),
                                "snr_db": p_att["snr_db"][:1].expand(x_all.shape[0], 1).clone(),
                                "noise_seed": int(p_att["noise_seed"]),
                            }
                            x_all_clean_t = aug.apply_with_params(x_all, p_ref)
                        else:
                            x_cand_clean_t = x_cand
                            x_cand_def_t = x_cand_def
                            x_all_clean_t = x_all

                        z_cand_def = speaker_embed_ecapa(speaker_model, x_cand_def_t)
                        z_all_clean = speaker_embed_ecapa(speaker_model, x_all_clean_t)
                        e_ref = F.normalize(z_all_clean.mean(dim=0), dim=0)

                        s_def = z_cand_def @ e_ref
                        beta = float(max(1e-3, args.att_beta))
                        w_soft = torch.softmax(beta * s_def, dim=0)
                        e_def_soft = F.normalize(torch.sum(z_cand_def * w_soft.unsqueeze(1), dim=0), dim=0)
                        loss_kagg = torch.dot(e_def_soft, e_ref)

                        k_proxy = min(16, int(z_cand_def.size(0)))
                        idx_top = torch.topk(s_def, k=k_proxy, largest=True).indices
                        e_def_top = F.normalize(z_cand_def[idx_top].mean(dim=0), dim=0)
                        k16_proxy = torch.dot(e_def_top, e_ref)
                loss_rank = torch.tensor(0.0, device=device)
                loss_ceiling = torch.tensor(0.0, device=device)
            elif objective == "targetedk_hard16_margin" and session_pool_keys:
                anchor_spk, anchor_unit, anchor_path = random.choice(session_pool_keys)
                win_rows = session_pool[(anchor_spk, anchor_unit, anchor_path)]
                wav_full = read_wav(resolve_wav_path(REPO_ROOT, processed_wav_dir, anchor_path))
                if len(win_rows) > 0:
                    sel_rows = win_rows
                    if not bool(args.att_use_full_session):
                        c = min(len(win_rows), max(16, int(args.att_candidate_windows)))
                        sel_rows = random.sample(win_rows, k=c) if len(win_rows) > c else win_rows

                    def build_segments(rows_in: List[Dict[str, str]]) -> List[torch.Tensor]:
                        out = []
                        for wr in rows_in:
                            s = int(round(float(wr["start_sec"]) * 16000))
                            d = int(round(float(wr["dur_sec"]) * 16000))
                            seg = wav_full[s : s + d]
                            if len(seg) < d:
                                seg = np.pad(seg, (0, max(0, d - len(seg))), mode="constant")
                            out.append(torch.from_numpy(seg.astype(np.float32)))
                        return out

                    segs_sel = build_segments(sel_rows)
                    segs_all = build_segments(win_rows)
                    if segs_sel and segs_all:
                        x_sel, _ = pad_batch(segs_sel)
                        x_all, _ = pad_batch(segs_all)
                        x_sel = x_sel.to(device)
                        x_all = x_all.to(device)
                        x_sel_def = model(x_sel)

                        if random.random() < eot_p:
                            p_att = aug.sample_shared_params(batch_size=x_sel.shape[0], device=device)
                            p_ref = {
                                "gain_db": p_att["gain_db"][:1].expand(x_all.shape[0], 1).clone(),
                                "snr_db": p_att["snr_db"][:1].expand(x_all.shape[0], 1).clone(),
                                "noise_seed": int(p_att["noise_seed"]),
                            }
                            x_sel_clean_t = aug.apply_with_params(x_sel, p_att)
                            x_sel_def_t = aug.apply_with_params(x_sel_def, p_att)
                            x_all_clean_t = aug.apply_with_params(x_all, p_ref)
                        else:
                            x_sel_clean_t = x_sel
                            x_sel_def_t = x_sel_def
                            x_all_clean_t = x_all

                        z_sel_clean = speaker_embed_ecapa(speaker_model, x_sel_clean_t)
                        z_sel_def_raw = speaker_embed_ecapa(speaker_model, x_sel_def)
                        z_sel_def_t = speaker_embed_ecapa(speaker_model, x_sel_def_t)
                        z_all_clean = speaker_embed_ecapa(speaker_model, x_all_clean_t)
                        e_ref = F.normalize(z_all_clean.mean(dim=0), dim=0)

                        src = str(args.att_selection_source)
                        if src == "clean":
                            z_sel = z_sel_clean
                        elif src == "defended":
                            z_sel = z_sel_def_raw
                        else:
                            z_sel = z_sel_def_t

                        k_eff = min(16, int(z_sel.size(0)))
                        idx_top = torch.topk(z_sel @ e_ref, k=k_eff, largest=True).indices
                        e_def_k16 = F.normalize(z_sel_def_t[idx_top].mean(dim=0), dim=0)
                        cos_k16 = torch.dot(e_def_k16, e_ref)
                        k16_proxy = cos_k16

                        tau = float(args.att_tau)
                        if str(args.att_margin_mode) == "logistic":
                            gamma = float(max(1e-3, args.att_gamma))
                            loss_kagg = torch.log1p(torch.exp(gamma * (cos_k16 - tau)))
                        else:
                            loss_kagg = F.relu(cos_k16 - tau)
                loss_rank = torch.tensor(0.0, device=device)
                loss_ceiling = torch.tensor(0.0, device=device)
            elif len(pos_rows) > 0:
                x_pos, _ = build_wave_batch(pos_rows, processed_wav_dir, device)
                x_pos_def = model(x_pos)
                if random.random() < eot_p:
                    p_rank = aug.sample_shared_params(batch_size=x_pos.shape[0], device=device)
                    x_pos_clean_t = aug.apply_with_params(x_pos, p_rank)
                    x_pos_def_t = aug.apply_with_params(x_pos_def, p_rank)
                else:
                    x_pos_clean_t = x_pos
                    x_pos_def_t = x_pos_def
                z_pos_clean = speaker_embed_ecapa(speaker_model, x_pos_clean_t)
                z_pos_def = speaker_embed_ecapa(speaker_model, x_pos_def_t)
                e_ref = F.normalize(z_pos_clean.mean(dim=0), dim=0)
                e_def = F.normalize(z_pos_def.mean(dim=0), dim=0)
                cos_pos = torch.dot(e_def, e_ref)

                if objective in {"kagg", "kagg_session"}:
                    loss_kagg = cos_pos
                    loss_rank = torch.tensor(0.0, device=device)
                    loss_ceiling = torch.tensor(0.0, device=device)
                else:
                    # Hard negative pool: aggregate clean embeddings from other speakers in current batch.
                    spk_to_embs: Dict[str, List[torch.Tensor]] = {}
                    for i, rr in enumerate(rows):
                        sid = str(rr["speaker_id"])
                        if sid == anchor_spk:
                            continue
                        spk_to_embs.setdefault(sid, []).append(z_x[i])
                    neg_pool: List[torch.Tensor] = []
                    for sid, embs in spk_to_embs.items():
                        if not embs:
                            continue
                        e_neg_sid = F.normalize(torch.stack(embs, dim=0).mean(dim=0), dim=0)
                        neg_pool.append(e_neg_sid)

                    # Fallback only when current mini-batch has no usable negative speaker.
                    if len(neg_pool) == 0 and anchor_spk:
                        neg_rows = sample_negative_speaker_group(speaker_rows, anchor_spk, rank_k)
                        if len(neg_rows) > 0:
                            x_neg, _ = build_wave_batch(neg_rows, processed_wav_dir, device)
                            if random.random() < eot_p:
                                p_neg = aug.sample_shared_params(batch_size=x_neg.shape[0], device=device)
                                x_neg_clean_t = aug.apply_with_params(x_neg, p_neg)
                            else:
                                x_neg_clean_t = x_neg
                            z_neg_clean = speaker_embed_ecapa(speaker_model, x_neg_clean_t)
                            neg_pool.append(F.normalize(z_neg_clean.mean(dim=0), dim=0))

                    if len(neg_pool) > 0:
                        neg_scores = torch.stack([torch.dot(e_def, en) for en in neg_pool], dim=0)
                        cos_neg = torch.max(neg_scores)
                    else:
                        cos_neg = torch.tensor(-1.0, device=device)
                    loss_rank = F.relu(cos_pos - cos_neg + rank_margin)
                    loss_ceiling = F.relu(cos_pos - ceiling_tau)
            else:
                loss_rank = torch.tensor(0.0, device=device)
                loss_ceiling = torch.tensor(0.0, device=device)

            if objective == "targetshift":
                z_def_single = z_def
                tgt_embs = []
                ref_embs = []
                for i, rr in enumerate(rows):
                    sid = str(rr["speaker_id"])
                    tgt_sid = pseudo_map.get(sid, "")
                    if tgt_sid in pseudo_centroids and sid in pseudo_centroids:
                        tgt_embs.append(torch.dot(z_def_single[i], pseudo_centroids[tgt_sid]))
                        ref_embs.append(torch.dot(z_def_single[i], z_x[i]))
                if tgt_embs:
                    cos_to_tgt = torch.stack(tgt_embs, dim=0).mean()
                    cos_to_ref = torch.stack(ref_embs, dim=0).mean()
                    loss_tgt = 1.0 - cos_to_tgt
                    loss_away = cos_to_ref

            if step <= stage1_steps:
                w_spk_eff = 0.0
                w_intra_eff = 0.0
                w_rank_eff = 0.0
                w_ceiling_eff = 0.0
                w_kagg_eff = 0.0
                w_tgt_eff = 0.0
                w_away_eff = 0.0
            else:
                denom = max(1, max_steps - stage1_steps)
                ratio = float(step - stage1_steps) / float(denom)
                ratio = max(0.0, min(1.0, ratio))
                if objective in {"kagg", "kagg_session", "targetedk", "targetedk_soft_def", "targetedk_hard16_margin"}:
                    w_spk_eff = 0.0
                    w_intra_eff = 0.0
                    w_rank_eff = 0.0
                    w_ceiling_eff = 0.0
                    if objective == "targetedk_soft_def":
                        warm = int(max(0, stage1_steps))
                        if step <= warm:
                            w_kagg_eff = 0.0
                        else:
                            denom2 = max(1, max_steps - warm)
                            ratio2 = float(step - warm) / float(denom2)
                            ratio2 = max(0.0, min(1.0, ratio2))
                            w_kagg_eff = target_w_kagg * ratio2
                    else:
                        w_kagg_eff = target_w_kagg * ratio
                    w_tgt_eff = 0.0
                    w_away_eff = 0.0
                elif objective == "targetshift":
                    w_spk_eff = 0.0
                    w_intra_eff = 0.0
                    w_rank_eff = 0.0
                    w_ceiling_eff = 0.0
                    w_kagg_eff = 0.0
                    w_tgt_eff = target_w_tgt * ratio
                    w_away_eff = target_w_away * ratio
                else:
                    w_spk_eff = target_w_spk * ratio
                    w_intra_eff = target_w_intra * ratio
                    w_rank_eff = target_w_rank * ratio
                    w_ceiling_eff = target_w_ceiling * ratio
                    w_kagg_eff = 0.0
                    w_tgt_eff = 0.0
                    w_away_eff = 0.0

            loss = (
                w_spk_eff * loss_spk_global
                + w_rec * loss_rec
                + w_energy * loss_energy
                + w_intra_eff * loss_intra
                + w_rank_eff * loss_rank
                + w_ceiling_eff * loss_ceiling
                + w_kagg_eff * loss_kagg
                + w_tgt_eff * loss_tgt
                + w_away_eff * loss_away
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            row = {
                "step": step,
                "loss_total": float(loss.item()),
                "loss_spk_global": float(loss_spk_global.item()),
                "loss_intra": float(loss_intra.item()),
                "loss_rank": float(loss_rank.item()),
                "loss_ceiling": float(loss_ceiling.item()),
                "loss_kagg": float(loss_kagg.item()),
                "loss_tgt": float(loss_tgt.item()),
                "loss_away": float(loss_away.item()),
                "loss_rec": float(loss_rec.item()),
                "loss_energy": float(loss_energy.item()),
                "k16_proxy": float(k16_proxy.item()) if torch.isfinite(k16_proxy) else float("nan"),
                "w_spk_eff": w_spk_eff,
                "w_intra_eff": w_intra_eff,
                "w_rank_eff": w_rank_eff,
                "w_ceiling_eff": w_ceiling_eff,
                "w_kagg_eff": w_kagg_eff,
                "w_tgt_eff": w_tgt_eff,
                "w_away_eff": w_away_eff,
                "speaker_sim_tel": float(loss_spk_global.item()),
                "n_windows_used": int(n_windows_used),
                "rank_k": int(rank_k),
                "val_speaker_sim_tel": float("nan"),
                "val_k16_proxy": float("nan"),
                "best_step": best_step,
                "best_val_speaker_sim_tel": best_val,
                "stoi_optional": float("nan"),
                "wer_pseudo_optional": float("nan"),
            }

            if step % 50 == 0 or step == 1:
                msg = (
                    f"step={step} loss={row['loss_total']:.4f} "
                    f"L_spk={row['loss_spk_global']:.4f} L_intra={row['loss_intra']:.4f} "
                    f"L_rank={row['loss_rank']:.4f} L_ceiling={row['loss_ceiling']:.4f} "
                    f"L_kagg={row['loss_kagg']:.4f} L_tgt={row['loss_tgt']:.4f} L_away={row['loss_away']:.4f} "
                    f"L_rec={row['loss_rec']:.4f} L_energy={row['loss_energy']:.4f} "
                    f"w_spk={w_spk_eff:.3f} w_intra={w_intra_eff:.3f} "
                    f"w_rank={w_rank_eff:.3f} w_ceiling={w_ceiling_eff:.3f} "
                    f"w_kagg={w_kagg_eff:.3f} w_tgt={w_tgt_eff:.3f} w_away={w_away_eff:.3f} "
                    f"cos_tel={row['speaker_sim_tel']:.4f} nwin={n_windows_used} rank_k={rank_k}"
                )
                print(msg)
                lf.write(msg + "\n")
                for i in range(min(5, len(rows))):
                    pmsg = (
                        f"[pair_check] step={step} utter_id={rows[i]['utter_id']} "
                        f"spk_id={rows[i]['speaker_id']} cos_tel={float(cos_tel_vec[i].item()):.4f}"
                    )
                    print(pmsg)
                    lf.write(pmsg + "\n")
                lf.flush()
            elif step % log_every == 0:
                msg = (
                    f"step={step} loss={row['loss_total']:.4f} "
                    f"L_spk={row['loss_spk_global']:.4f} L_intra={row['loss_intra']:.4f} "
                    f"L_rank={row['loss_rank']:.4f} L_ceiling={row['loss_ceiling']:.4f} "
                    f"L_kagg={row['loss_kagg']:.4f} L_tgt={row['loss_tgt']:.4f} L_away={row['loss_away']:.4f}"
                )
                print(msg)
                lf.write(msg + "\n")
                lf.flush()

            if objective == "targetedk_hard16_margin" and step % 25 == 0:
                amsg = (
                    f"[att_diag] step={step} "
                    f"cos_k16={row['k16_proxy']:.4f} L_att={row['loss_kagg']:.4f} w_att={w_kagg_eff:.3f}"
                )
                print(amsg)
                lf.write(amsg + "\n")
                lf.flush()

            if step % 200 == 0:
                hist_msg = "[rank_k_hist] step={} {}".format(
                    step,
                    " ".join([f"K{k}:{rank_k_hist.get(k, 0)}" for k in sorted(rank_k_hist.keys())]),
                )
                print(hist_msg)
                lf.write(hist_msg + "\n")
                lf.flush()
                rank_k_hist = {k: 0 for k in rank_k_set}

            if step % val_every == 0 or step == max_steps:
                val_sim = evaluate_speaker_sim_tel(
                    model=model,
                    speaker_model=speaker_model,
                    aug=aug,
                    val_rows=val_rows,
                    processed_wav_dir=processed_wav_dir,
                    batch_size=batch_size,
                    val_batches=val_batches,
                    device=device,
                )
                vmsg = f"[val] step={step} speaker_sim_tel={val_sim:.4f}"
                print(vmsg)
                lf.write(vmsg + "\n")
                val_k16_proxy = float("nan")
                if objective in {"targetedk_soft_def", "targetedk_hard16_margin"} and val_session_pool:
                    val_k16_proxy = evaluate_targeted_k16_proxy(
                        model=model,
                        speaker_model=speaker_model,
                        aug=aug,
                        session_pool=val_session_pool,
                        processed_wav_dir=processed_wav_dir,
                        device=device,
                        eot_p=eot_p,
                        att_candidate_windows=int(args.att_candidate_windows),
                        use_full_session=bool(args.att_use_full_session),
                        att_beta=float(args.att_beta),
                        val_sessions=int(args.val_proxy_sessions),
                        selection_source=str(args.val_proxy_selection_source),
                    )
                    pmsg = f"[val_proxy] step={step} k16_proxy={val_k16_proxy:.4f}"
                    print(pmsg)
                    lf.write(pmsg + "\n")

                current_metric = val_sim
                if str(args.best_metric) == "val_k16_proxy" and np.isfinite(val_k16_proxy):
                    current_metric = float(val_k16_proxy)
                if current_metric < best_metric_value:
                    best_metric_value = float(current_metric)
                    best_val = val_sim
                    if np.isfinite(val_k16_proxy):
                        best_val_proxy = float(val_k16_proxy)
                    best_step = step
                    payload = {
                        "model_state_dict": model.state_dict(),
                        "model_config": model_cfg,
                        "dataset": dataset,
                        "step": step,
                        "run_name": args.run_name,
                        "best_val_speaker_sim_tel": best_val,
                        "best_val_k16_proxy": best_val_proxy,
                        "best_metric": str(args.best_metric),
                        "best_metric_value": best_metric_value,
                    }
                    torch.save(payload, best_ckpt)
                    smsg = f"[ckpt] saved best checkpoint at step={step} to {best_ckpt}"
                    print(smsg)
                    lf.write(smsg + "\n")
                lf.flush()
                row["val_speaker_sim_tel"] = float(val_sim)
                row["val_k16_proxy"] = float(val_k16_proxy)
                row["best_step"] = best_step
                row["best_val_speaker_sim_tel"] = best_val

            writer.writerow(row)

    elapsed = time.time() - t0
    peak_mem = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        peak_mem = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

    summary = {
        "dataset": dataset,
        "steps": max_steps,
        "batch_size": batch_size,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "run_name": args.run_name,
        "best_step": best_step,
        "best_val_speaker_sim_tel": best_val,
        "best_val_k16_proxy": best_val_proxy,
        "best_metric": str(args.best_metric),
        "best_metric_value": best_metric_value,
        "elapsed_sec": elapsed,
        "device": str(device),
        "peak_gpu_mem_mb": peak_mem,
        "outputs": {
            "checkpoint": str(best_ckpt),
            "curve_csv": str(curve_csv),
            "log_file": str(log_path),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
