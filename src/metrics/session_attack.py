#!/usr/bin/env python3
"""Session attack helpers: random/bestK selection + sanity diagnostics."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def pick_random_k(n: int, k: int, rng: random.Random) -> List[int]:
    if n <= 0:
        return []
    if n >= k:
        return rng.sample(range(n), k=k)
    return [rng.randrange(0, n) for _ in range(k)]


def _greedy_bestk_by_clean(
    clean_z: torch.Tensor,
    k: int,
    return_debug: bool = False,
    debug_topn: int = 5,
) -> List[int] | Tuple[List[int], List[Dict[str, float]]]:
    """Centroid-consistency selection: choose windows closest to clean centroid."""
    n = int(clean_z.size(0))
    if n <= 0:
        return ([], []) if return_debug else []
    e_cent = F.normalize(clean_z.mean(dim=0), dim=0)
    cent_scores = clean_z @ e_cent
    seed = int(torch.argmax(cent_scores).item())
    trace: List[Dict[str, float]] = []
    if k <= 1:
        if return_debug:
            trace.append(
                {
                    "step": 1.0,
                    "pick": float(seed),
                    "gain": float(cent_scores[seed].item()),
                    "cand_top1_idx": float(seed),
                    "cand_top1_score": float(cent_scores[seed].item()),
                }
            )
            return [seed], trace
        return [seed]
    sorted_idx = [int(i) for i in torch.argsort(cent_scores, descending=True).tolist()]
    pool_size = min(n, max(k, 3 * k))
    cand_pool = sorted_idx[:pool_size]
    selected: List[int] = []
    remaining = set(cand_pool)
    while len(selected) < min(k, n) and remaining:
        cand_scores: List[Tuple[int, float]] = []
        for i in remaining:
            cand = selected + [i]
            e_cand = F.normalize(clean_z[cand].mean(dim=0), dim=0)
            v = float(torch.dot(e_cand, e_cent).item())
            cand_scores.append((int(i), v))
        cand_scores.sort(key=lambda x: x[1], reverse=True)
        best_i, best_v = cand_scores[0]
        if return_debug:
            dbg = {"step": float(len(selected) + 1), "pick": float(best_i), "gain": float(best_v)}
            for j, (ci, cv) in enumerate(cand_scores[: max(1, int(debug_topn))]):
                dbg[f"cand_top{j + 1}_idx"] = float(ci)
                dbg[f"cand_top{j + 1}_score"] = float(cv)
            trace.append(dbg)
        selected.append(best_i)
        remaining.remove(best_i)
    if n < k:
        while len(selected) < k:
            selected.append(seed)
    return (selected, trace) if return_debug else selected


def _greedy_bestk_by_ref_similarity(
    clean_z: torch.Tensor,
    k: int,
    return_debug: bool = False,
    debug_topn: int = 5,
) -> List[int] | Tuple[List[int], List[Dict[str, float]]]:
    """Greedy targeted attack: maximize cos(mean(selected), clean-ref-mean)."""
    n = int(clean_z.size(0))
    if n <= 0:
        return ([], []) if return_debug else []
    e_ref = F.normalize(clean_z.mean(dim=0), dim=0)
    if n <= k:
        out = list(range(n))
        if n == 0:
            return (out, []) if return_debug else out
        best_idx = int(torch.argmax(clean_z @ e_ref).item())
        while len(out) < k:
            out.append(best_idx)
        return (out, []) if return_debug else out

    selected: List[int] = []
    remaining = set(range(n))
    trace: List[Dict[str, float]] = []
    while len(selected) < k and remaining:
        cand_scores: List[Tuple[int, float]] = []
        for i in remaining:
            cand = selected + [i]
            e_cand = F.normalize(clean_z[cand].mean(dim=0), dim=0)
            v = float(torch.dot(e_cand, e_ref).item())
            cand_scores.append((int(i), v))
        cand_scores.sort(key=lambda x: x[1], reverse=True)
        best_i, best_v = cand_scores[0]
        if return_debug:
            dbg = {
                "step": float(len(selected) + 1),
                "pick": float(best_i),
                "gain": float(best_v),
            }
            for j, (ci, cv) in enumerate(cand_scores[: max(1, int(debug_topn))]):
                dbg[f"cand_top{j + 1}_idx"] = float(ci)
                dbg[f"cand_top{j + 1}_score"] = float(cv)
            trace.append(dbg)
        selected.append(best_i)
        remaining.remove(best_i)
    return (selected, trace) if return_debug else selected


def pick_indices(
    strategy: str,
    clean_z: torch.Tensor,
    k: int,
    rng: random.Random,
    selection_z: torch.Tensor | None = None,
) -> List[int]:
    n = int(clean_z.size(0))
    if strategy == 'bestK_by_clean_consistency':
        return _greedy_bestk_by_clean(clean_z, k)
    if strategy == "bestK_by_ref_similarity":
        z_sel = selection_z if selection_z is not None else clean_z
        return _greedy_bestk_by_ref_similarity(z_sel, k)
    return pick_random_k(n=n, k=k, rng=rng)


def aggregate_cos_to_ref(def_z: torch.Tensor, clean_z: torch.Tensor, idx: List[int]) -> float:
    if len(idx) == 0 or def_z.numel() == 0 or clean_z.numel() == 0:
        return float('nan')
    e_def = F.normalize(def_z[idx].mean(dim=0), dim=0)
    e_ref = F.normalize(clean_z.mean(dim=0), dim=0)  # fixed ref: clean window mean
    return float(torch.dot(e_def, e_ref).item())


def sanity_compare_k(
    def_z: torch.Tensor,
    clean_z: torch.Tensor,
    rng: random.Random,
    n_random_trials: int = 32,
) -> dict:
    """Sanity values for one session on baseline/defended embeddings."""
    n_trials = max(1, int(n_random_trials))
    k1_vals = []
    k16_vals = []
    for _ in range(n_trials):
        idx_k1 = pick_indices("random_K", clean_z, 1, rng)
        idx_k16_r = pick_indices("random_K", clean_z, 16, rng)
        k1_vals.append(aggregate_cos_to_ref(def_z, clean_z, idx_k1))
        k16_vals.append(aggregate_cos_to_ref(def_z, clean_z, idx_k16_r))
    idx_k16_c = pick_indices("bestK_by_clean_consistency", clean_z, 16, rng)
    idx_k16_t = pick_indices("bestK_by_ref_similarity", clean_z, 16, rng)
    return {
        "cos_K1_random": float(np.nanmean(k1_vals)),
        "cos_K16_random": float(np.nanmean(k16_vals)),
        "cos_K16_bestK_consistency": aggregate_cos_to_ref(def_z, clean_z, idx_k16_c),
        "cos_K16_bestK_targeted": aggregate_cos_to_ref(def_z, clean_z, idx_k16_t),
    }


def sanity_debug_topk(
    def_z: torch.Tensor,
    clean_z: torch.Tensor,
    k: int = 16,
    debug_topn: int = 5,
) -> dict:
    """Return per-step top candidates for consistency/targeted greedy picks."""
    idx_c, trace_c = _greedy_bestk_by_clean(clean_z, k, return_debug=True, debug_topn=debug_topn)
    idx_t, trace_t = _greedy_bestk_by_ref_similarity(clean_z, k, return_debug=True, debug_topn=debug_topn)
    return {
        "k": int(k),
        "consistency": {
            "indices": [int(i) for i in idx_c],
            "cos": aggregate_cos_to_ref(def_z, clean_z, idx_c),
            "trace": trace_c,
        },
        "targeted": {
            "indices": [int(i) for i in idx_t],
            "cos": aggregate_cos_to_ref(def_z, clean_z, idx_t),
            "trace": trace_t,
        },
    }
