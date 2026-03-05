#!/usr/bin/env python3
"""run_train_v2.py — CloneBlock-RT v0/v0.1/v0p2 training entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        return {"gain_db": gain_db, "snr_db": snr_db, "noise_seed": noise_seed}

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
    return F.normalize(emb, dim=-1)


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


def build_wave_batch(rows: List[Dict[str, str]], processed_wav_dir: Path, device: torch.device):
    wavs = [torch.from_numpy(read_wav(processed_wav_dir / r["path"])) for r in rows]
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
        per_sample.append(F.cosine_similarity(z[:-1], z[1:], dim=-1).mean())
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
            z1 = speaker_embed_ecapa(speaker_model, aug.apply_with_params(x, p))
            z2 = speaker_embed_ecapa(speaker_model, aug.apply_with_params(x_def, p))
            sims.append(float(F.cosine_similarity(z1, z2, dim=-1).mean().item()))
    model.train()
    return float(np.mean(sims)) if sims else float("nan")


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
    p.add_argument("--w_energy", type=float, default=None)
    p.add_argument("--eot_prob", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train_subset_size", type=int, default=None)
    p.add_argument("--stage1_steps", type=int, default=None)
    p.add_argument("--init_checkpoint", type=str, default="")
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
    if args.w_energy is not None:
        loss_cfg["w_energy"] = float(args.w_energy)
    if args.eot_prob is not None:
        train_cfg["eot_prob"] = float(args.eot_prob)
    if args.train_subset_size is not None:
        data_cfg["train_subset_size"] = int(args.train_subset_size)
    if args.stage1_steps is not None:
        train_cfg["stage1_steps"] = int(args.stage1_steps)

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

    model = STFTMaskDefense(model_cfg).to(device)
    if args.init_checkpoint:
        ck = torch.load(args.init_checkpoint, map_location=device)
        if isinstance(ck, dict) and "model_state_dict" in ck:
            model.load_state_dict(ck["model_state_dict"], strict=False)
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
    w_energy = float(loss_cfg.get("w_energy", 0.1))

    ckpt_dir = REPO_ROOT / "checkpoints" / "defense"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"{args.run_name}_best.pt"
    curve_csv = REPO_ROOT / "artifacts" / "tables" / f"train_{args.run_name}_curve.csv"
    log_path = REPO_ROOT / "artifacts" / "logs" / f"train_{args.run_name}.log"
    curve_csv.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_step = -1
    t0 = time.time()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with open(curve_csv, "w", newline="", encoding="utf-8") as cf, open(log_path, "w", encoding="utf-8") as lf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "step",
                "loss_total",
                "loss_spk_global",
                "loss_intra",
                "loss_rec",
                "loss_energy",
                "w_spk_eff",
                "w_intra_eff",
                "speaker_sim_tel",
                "n_windows_used",
                "val_speaker_sim_tel",
                "best_step",
                "best_val_speaker_sim_tel",
                "stoi_optional",
                "wer_pseudo_optional",
            ],
        )
        writer.writeheader()

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
            loss_intra, n_windows_used = compute_intra_loss(speaker_model, x_def_for_spk, lengths)

            if step <= stage1_steps:
                w_spk_eff = 0.0
                w_intra_eff = 0.0
            else:
                denom = max(1, max_steps - stage1_steps)
                ratio = max(0.0, min(1.0, float(step - stage1_steps) / float(denom)))
                w_spk_eff = target_w_spk * ratio
                w_intra_eff = target_w_intra * ratio

            loss = (
                w_spk_eff * loss_spk_global
                + w_rec * loss_rec
                + w_energy * loss_energy
                + w_intra_eff * loss_intra
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
                "loss_rec": float(loss_rec.item()),
                "loss_energy": float(loss_energy.item()),
                "w_spk_eff": w_spk_eff,
                "w_intra_eff": w_intra_eff,
                "speaker_sim_tel": float(loss_spk_global.item()),
                "n_windows_used": int(n_windows_used),
                "val_speaker_sim_tel": float("nan"),
                "best_step": best_step,
                "best_val_speaker_sim_tel": best_val,
                "stoi_optional": float("nan"),
                "wer_pseudo_optional": float("nan"),
            }

            if step % 50 == 0 or step == 1:
                msg = (
                    f"step={step} loss={row['loss_total']:.4f} "
                    f"L_spk={row['loss_spk_global']:.4f} L_intra={row['loss_intra']:.4f} "
                    f"L_rec={row['loss_rec']:.4f} L_energy={row['loss_energy']:.4f} "
                    f"w_spk={w_spk_eff:.3f} w_intra={w_intra_eff:.3f} "
                    f"cos_tel={row['speaker_sim_tel']:.4f} nwin={n_windows_used}"
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
                msg = f"step={step} loss={row['loss_total']:.4f} L_spk={row['loss_spk_global']:.4f}"
                print(msg)
                lf.write(msg + "\n")
                lf.flush()

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
                if val_sim < best_val:
                    best_val = val_sim
                    best_step = step
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model_config": model_cfg,
                            "dataset": dataset,
                            "step": step,
                            "run_name": args.run_name,
                            "best_val_speaker_sim_tel": best_val,
                        },
                        best_ckpt,
                    )
                    smsg = f"[ckpt] saved best checkpoint at step={step} to {best_ckpt}"
                    print(smsg)
                    lf.write(smsg + "\n")
                lf.flush()
                row["val_speaker_sim_tel"] = float(val_sim)
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
