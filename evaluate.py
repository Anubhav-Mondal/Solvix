#!/usr/bin/env python3
"""
Standalone evaluation / inference script for the KLA image-restoration model.

Usage:
    python evaluate.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/outputs

What it does:
    1. Loads the trained model from CHECKPOINT_PATH (edit the variable below,
       or override with --checkpoint on the command line).
    2. Reads every .npy file in --input_dir (each a single-channel degraded
       image, e.g. 128x128, values roughly in [0, 1]).
    3. Runs inference (with test-time augmentation by default) to produce the
       restored, 2x-super-resolved output.
    4. Writes each restored image to --output_dir in TWO forms:
         - <name>.npy   (float32, HxW, values in [0,1])            <- primary
         - <name>.png   (8-bit grayscale preview, for quick viewing)
    5. Prints per-image and total inference timing.

Runs standalone: no training-loop code, no pytorch_msssim, no drive mounting.
Only depends on torch, numpy, and pillow.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CHECKPOINT_PATH = "model/model_best.pt"

# =============================================================================
# Model definition (must exactly match the architecture used in training.py)
# =============================================================================

from model.architecture import DenseBlock, RRDB, RestorationNet

# =============================================================================
# Test-time augmentation (matches training.py's tta_predict)
# =============================================================================

def tta_predict(model, x):
    outs = []
    for hflip in (False, True):
        for k in range(4):
            xi = torch.rot90(x, k, dims=(-2, -1))
            if hflip:
                xi = torch.flip(xi, dims=(-1,))
            with torch.no_grad():
                yi = model(xi)
            if hflip:
                yi = torch.flip(yi, dims=(-1,))
            yi = torch.rot90(yi, -k, dims=(-2, -1))
            outs.append(yi)
    return torch.stack(outs, 0).mean(0).clamp(0, 1)


# =============================================================================
# Checkpoint loading -- robust to several possible save formats
# =============================================================================

def load_model(checkpoint_path: str, device: str) -> RestorationNet:
    """
    Loads RestorationNet weights from a checkpoint file, tolerating several
    formats that may have been produced across training runs:
      - a raw state_dict (torch.save(model.state_dict(), path))
      - a full dict with a "model_state_dict" key (as saved by training.py)
      - a full dict with a "state_dict" key (common alternate convention)
      - a dict that also has "ema_state_dict" (we prefer EMA weights, since
        they are typically smoother/better than the raw trained weights)

    Model hyperparameters (feat_ch, num_blocks, scale) are read from the
    checkpoint's saved "args" if present, otherwise fall back to the
    training defaults (feat_ch=48, num_blocks=8, scale=2).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            f"Pass the correct path with --checkpoint, or edit "
            f"CHECKPOINT_PATH at the top of evaluate.py."
        )

    ckpt = torch.load(ckpt_path, map_location=device)

    # Figure out hyperparameters
    feat_ch, num_blocks, scale = 48, 8, 2
    if isinstance(ckpt, dict) and "args" in ckpt:
        saved_args = ckpt["args"]
        feat_ch = saved_args.get("feat_ch", feat_ch)
        num_blocks = saved_args.get("num_blocks", num_blocks)
        scale = saved_args.get("scale", scale)

    model = RestorationNet(in_ch=1, feat_ch=feat_ch, num_blocks=num_blocks, scale=scale)

    # Figure out which key holds the actual weights
    state_dict = None
    prefer_ema = False
    if isinstance(ckpt, dict):
        if "ema_state_dict" in ckpt:
            state_dict = ckpt["ema_state_dict"]
            prefer_ema = True
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            # looks like a raw state_dict that happens to be a dict
            state_dict = ckpt
    if state_dict is None:
        raise RuntimeError(
            f"Could not find model weights inside checkpoint '{checkpoint_path}'. "
            f"Expected a raw state_dict, or a dict containing one of "
            f"'model_state_dict', 'state_dict', 'ema_state_dict'. "
            f"Found top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  weights used     : {'EMA' if prefer_ema else 'raw model'}")
    print(f"  feat_ch={feat_ch}  num_blocks={num_blocks}  scale={scale}")
    print(f"  parameters       : {sum(p.numel() for p in model.parameters()):,}")
    return model


# =============================================================================
# I/O helpers
# =============================================================================

def load_npy_image(path: Path) -> np.ndarray:
    """Load a single degraded image .npy file as a 2D (H, W) float32 array."""
    arr = np.load(path).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array after squeeze, got shape {arr.shape} for {path}")
    return arr


def save_outputs(restored: np.ndarray, out_dir: Path, stem: str):
    """Save restored image as both .npy (primary) and .png (preview)."""
    np.save(out_dir / f"{stem}.npy", restored.astype(np.float32))
    png = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(png, mode="L").save(out_dir / f"{stem}.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with the trained restoration model over a "
                    "directory of degraded .npy test images."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Directory containing degraded input .npy files.")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Directory to write restored .npy/.png outputs to. "
                              "Created if it does not already exist.")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                         help=f"Path to the trained model checkpoint "
                              f"(default: {CHECKPOINT_PATH}).")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu",
                         help="Device to run inference on (default: cuda if available).")
    parser.add_argument("--no_tta", action="store_true",
                         help="Disable test-time augmentation (8x flip/rotate averaging). "
                              "TTA is ON by default for best quality; disable it for "
                              "faster, single-pass inference timing.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"--input_dir does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in --input_dir: {input_dir}")

    print(f"Found {len(npy_files)} input images in {input_dir}")
    print(f"Device: {args.device}")

    model = load_model(args.checkpoint, args.device)

    total_start = time.time()
    per_image_times = []

    for i, path in enumerate(npy_files, 1):
        img = load_npy_image(path)
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(args.device)  # (1,1,H,W)

        t0 = time.time()
        with torch.no_grad():
            if args.no_tta:
                pred = model(x)
            else:
                pred = tta_predict(model, x)
        if args.device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        per_image_times.append(dt)

        restored = pred.squeeze(0).squeeze(0).cpu().numpy()
        save_outputs(restored, output_dir, path.stem)

        if i % 20 == 0 or i == len(npy_files):
            print(f"  [{i}/{len(npy_files)}] {path.name} -> "
                  f"{path.stem}.npy / .png  ({dt*1000:.1f} ms)")

    total_time = time.time() - total_start
    avg_ms = 1000.0 * sum(per_image_times) / len(per_image_times)
    print("\nDone.")
    print(f"  Images processed   : {len(npy_files)}")
    print(f"  Total wall time    : {total_time:.2f}s")
    print(f"  Avg inference time : {avg_ms:.2f} ms/image "
          f"({'with' if not args.no_tta else 'without'} TTA)")
    print(f"  Outputs written to : {output_dir}")


if __name__ == "__main__":
    main()
