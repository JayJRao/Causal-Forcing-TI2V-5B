"""
Visualize ODE latent data stored in .pt files from get_causal_ode_data_chunkwise.py.

Data format (stored tensor shape): [1, 6, num_frames, C, H, W]
  - Index 0: noisy latent at t=1000 (pure noise)
  - Index 1: noisy latent at t=750
  - Index 2: noisy latent at t=500
  - Index 3: noisy latent at t=250
  - Index 4: x0_pred  (final denoised output)
  - Index 5: clean_latent (ground truth)

Usage:
    python scripts/visualize_ode_latents.py \
        --pt_path datasets/ODE_Causal_chunkwise_latents/ar_diffusion_tf_chunkwise_5b_720P_koala36m_merged_20260224011730_279043_lr2e-6/ar_diffusion_tf_chunkwise_5b_720P_koala36m_merged_20260224011730_279043_lr2e-6-checkpoint_model_015000-gc-5.0/split_1of12/00000001.pt \
        --indices 4 5 \
        --output_dir tmp/ode_vis
"""

import argparse
import os
import sys

import torch
from torchvision.io import write_video
from einops import rearrange

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """Decode a single latent [1, T, C, H, W] -> pixel video [1, T, H, W, C] uint8."""
    latent = latent.to(device=device, dtype=dtype)
    with torch.no_grad():
        pixel = vae.decode_to_pixel(latent)  # [1, T, C, H, W], range [-1, 1]
    pixel = (pixel * 0.5 + 0.5).clamp(0, 1)  # [0, 1]
    pixel = (pixel * 255).byte()             # uint8
    # [1, T, C, H, W] -> [T, H, W, C]
    pixel = rearrange(pixel[0], "t c h w -> t h w c").cpu()
    return pixel


def main():
    parser = argparse.ArgumentParser(description="Visualize ODE latent .pt file")
    parser.add_argument("--pt_path", type=str, required=True,
                        help="Path to the .pt file (e.g. 00000000.pt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output videos (default: same dir as .pt)")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Which of the 6 latent slots to decode (0-5). Default: all.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run VAE on")
    args = parser.parse_args()

    pt_path = args.pt_path
    assert os.path.isfile(pt_path), f"File not found: {pt_path}"

    output_dir = args.output_dir or os.path.dirname(pt_path)
    os.makedirs(output_dir, exist_ok=True)

    indices = args.indices if args.indices is not None else list(range(6))

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading {pt_path} ...")
    data = torch.load(pt_path, map_location="cpu")

    # data is {prompt_str: tensor [1, 6, T, C, H, W]}
    prompt = list(data.keys())[0]
    tensor = list(data.values())[0]   # [1, 6, T, C, H, W]

    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"Tensor shape: {list(tensor.shape)}")
    assert tensor.ndim == 6, f"Expected 6-dim tensor, got {tensor.ndim}"
    num_slots = tensor.shape[1]
    c_dim = tensor.shape[3]  # [1, 6, T, C, H, W]
    print(f"Number of latent slots: {num_slots} (expected 6), C={c_dim}")

    # ── Select VAE based on channel dim ────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if c_dim == 48:
        print(f"C={c_dim} -> using Wan2.2 VAE")
        vae = Wan22_VAEWrapper().to(device).to(torch.float32)
    elif c_dim == 16:
        print(f"C={c_dim} -> using Wan2.1 VAE")
        vae = WanVAEWrapper().to(device).to(torch.float32)
    else:
        raise ValueError(f"Unexpected channel dim C={c_dim}, expected 16 (Wan2.1) or 48 (Wan2.2)")
    vae.eval()

    # ── Decode & save ──────────────────────────────────────────────────────────
    base_name = os.path.splitext(os.path.basename(pt_path))[0]

    for idx in indices:
        if idx >= num_slots:
            print(f"[WARN] Index {idx} out of range (num_slots={num_slots}), skipping.")
            continue

        label = LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else f"slot{idx}"
        out_path = os.path.join(output_dir, f"{base_name}_{label}.mp4")

        print(f"Decoding slot {idx} ({label}) -> {out_path}")
        latent = tensor[:, idx]  # [1, T, C, H, W]
        frames = decode_latent(vae, latent, device)  # [T, H, W, C] uint8

        write_video(out_path, frames, fps=args.fps)
        print(f"  Saved {frames.shape[0]} frames @ {args.fps}fps  shape={list(frames.shape)}")

    print("Done.")


if __name__ == "__main__":
    main()
