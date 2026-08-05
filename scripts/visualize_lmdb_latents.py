"""
Visualize ODE latent data stored in an LMDB dataset.

Each LMDB entry has latents of shape:
    (num_slots, num_frames, C, H, W)  -- ordered from pure noise to clean

  - Index 0: noisy latent at t=1000 (pure noise)
  - Index 1: noisy latent at t=750
  - Index 2: noisy latent at t=500
  - Index 3: noisy latent at t=250
  - Index 4: x0_pred  (final denoised output)
  - Index 5: clean_latent (ground truth)

C dimension determines VAE:
  - C=48 -> Wan2.2 VAE
  - C=16 -> Wan2.1 VAE

Usage:
    python scripts/visualize_lmdb_latents.py \
        --lmdb_path /ytech_milm_intern/data_share/Causal-Forcing-data/ODE6KCausal_chunkwise \
        --index 0 \
        --indices 4 5 \
        --output_dir vis/lmdb_vis

    # visualize all samples when --index is omitted
    python scripts/visualize_lmdb_latents.py \
        --lmdb_path /ytech_milm_intern/data_share/Causal-Forcing-data/ODE6KCausal_chunkwise \
        --indices 4 5 \
        --output_dir vis/lmdb_vis

    # uniformly sample 16 samples from the whole dataset
    python scripts/visualize_lmdb_latents.py \
        --lmdb_path /ytech_milm_intern/data_share/Causal-Forcing-data/ODE6KCausal_chunkwise \
        --limit 16 \
        --indices 4 5 \
        --output_dir vis/lmdb_vis
"""

import argparse
import os
import sys

import lmdb
import numpy as np
import torch
from torchvision.io import write_video
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from utils.wan_wrapper import WanVAEWrapper
from utils.wan_wrapper_22 import Wan22_VAEWrapper

LABEL_NAMES = [
    "t1000_noisy",
    "t750_noisy",
    "t500_noisy",
    "t250_noisy",
    "x0_pred",
    "clean_latent",
]


def decode_latent(vae, latent, device, dtype=torch.float32):
    """Decode a single latent [1, T, C, H, W] -> pixel frames [T, H, W, C] uint8."""
    latent = latent.to(device=device, dtype=dtype)
    with torch.no_grad():
        pixel = vae.decode_to_pixel(latent)  # [1, T, C, H, W], range [-1, 1]
    pixel = (pixel * 0.5 + 0.5).clamp(0, 1)
    pixel = (pixel * 255).byte()
    pixel = rearrange(pixel[0], "t c h w -> t h w c").cpu()
    return pixel


def main():
    parser = argparse.ArgumentParser(description="Visualize LMDB ODE latent entry")
    parser.add_argument("--lmdb_path", type=str,
                        default="/ytech_milm_intern/data_share/Causal-Forcing-data/ODE6KCausal_chunkwise",
                        help="Path to the LMDB directory")
    parser.add_argument("--index", type=int, default=None,
                        help="Index of the sample to visualize. Default: visualize all samples.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of samples to uniformly draw from the whole dataset when --index is not set.")
    parser.add_argument("--output_dir", type=str, default="/tmp/lmdb_vis",
                        help="Directory to save output videos")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Which latent slots to decode (e.g. 4 5). Default: all.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run VAE on")
    args = parser.parse_args()

    if args.index is not None and args.limit is not None:
        parser.error("--index and --limit cannot be used together")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Open LMDB and read entry ───────────────────────────────────────────────
    print(f"Opening LMDB: {args.lmdb_path}")
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False,
                    readahead=False, meminit=False)

    latents_shape = get_array_shape_from_lmdb(env, "latents")
    # latents_shape: (N, num_slots, T, C, H, W)
    print(f"LMDB latents shape: {latents_shape}  (total {latents_shape[0]} samples)")

    if args.index is not None:
        assert 0 <= args.index < latents_shape[0], f"Index {args.index} out of range (dataset size={latents_shape[0]})"
        sample_indices = [args.index]
    elif args.limit is not None and args.limit < latents_shape[0]:
        assert args.limit > 0, "--limit must be a positive integer"
        sample_indices = np.linspace(0, latents_shape[0] - 1, num=args.limit, dtype=int).tolist()
    else:
        sample_indices = list(range(latents_shape[0]))

    c_dim = latents_shape[3] if len(latents_shape) == 6 else latents_shape[2]

    # ── Select VAE ─────────────────────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if c_dim == 48:
        print(f"C={c_dim} -> using Wan2.2 VAE")
        vae = Wan22_VAEWrapper().to(device).to(torch.float32)
    elif c_dim == 16:
        print(f"C={c_dim} -> using Wan2.1 VAE")
        vae = WanVAEWrapper().to(device).to(torch.float32)
    else:
        raise ValueError(f"Unexpected C={c_dim}, expected 16 (Wan2.1) or 48 (Wan2.2)")
    vae.eval()

    # ── Decode & save ──────────────────────────────────────────────────────────
    for idx in sample_indices:
        latents = retrieve_row_from_lmdb(
            env, "latents", np.float16, idx, shape=latents_shape[1:]
        )
        # latents: numpy (num_slots, T, C, H, W) or (T, C, H, W) for old format
        if latents.ndim == 4:
            latents = latents[None, ...]  # add slot dim -> (1, T, C, H, W)

        prompt = retrieve_row_from_lmdb(env, "prompts", str, idx)

        print(f"\n[{idx}/{latents_shape[0] - 1}] Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"Latents shape: {list(latents.shape)}")

        num_slots = latents.shape[0]
        print(f"num_slots={num_slots}, C={latents.shape[2]}")

        slot_indices = args.indices if args.indices is not None else list(range(num_slots))
        for slot_idx in slot_indices:
            if slot_idx >= num_slots:
                print(f"[WARN] Slot {slot_idx} out of range (num_slots={num_slots}), skipping.")
                continue

            label = LABEL_NAMES[slot_idx] if slot_idx < len(LABEL_NAMES) else f"slot{slot_idx}"
            out_path = os.path.join(args.output_dir, f"{idx:08d}_{label}.mp4")

            print(f"Decoding slot {slot_idx} ({label}) -> {out_path}")
            latent = torch.tensor(latents[slot_idx], dtype=torch.float32).unsqueeze(0)  # [1, T, C, H, W]
            frames = decode_latent(vae, latent, device)

            write_video(out_path, frames, fps=args.fps)
            print(f"  Saved {frames.shape[0]} frames @ {args.fps}fps  shape={list(frames.shape)}")

    print("Done.")


if __name__ == "__main__":
    main()
