"""
Compute PSNR / SSIM / LPIPS between restored outputs and ground truth.

Usage:
    python compute_metrics.py \
        --pred_dir outputs/ \
        --gt_dir data/val/GT \
        --csv_out metrics_report.csv

Prints per-image PSNR/SSIM/LPIPS plus overall means, and optionally
writes a CSV.

Depends on: numpy, scikit-image, lpips, torch. These are training/eval
-reporting-only dependencies -- intentionally NOT required by evaluate.py.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import lpips
except ImportError:
    lpips = None


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array after squeeze, got shape {arr.shape} for {path}")
    return arr


def to_lpips_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    """(H,W) float32 [0,1] -> (1,3,H,W) float32 in [-1,1], as LPIPS expects."""
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)
    t = t.repeat(1, 3, 1, 1) * 2.0 - 1.0
    return t


def main():
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM/LPIPS for restored outputs vs GT.")
    parser.add_argument("--pred_dir", type=str, required=True,
                         help="Directory of restored .npy outputs (from evaluate.py).")
    parser.add_argument("--gt_dir", type=str, required=True,
                         help="Directory of matching ground-truth .npy files.")
    parser.add_argument("--csv_out", type=str, default=None,
                         help="Optional path to write a per-image CSV report.")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_lpips", action="store_true",
                         help="Skip LPIPS (e.g. if the lpips package isn't installed).")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    pred_files = sorted(pred_dir.glob("*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No .npy files found in --pred_dir: {pred_dir}")

    use_lpips = not args.no_lpips
    if use_lpips and lpips is None:
        print("WARNING: 'lpips' package not installed -- skipping LPIPS. "
              "Install with: pip install lpips")
        use_lpips = False

    lpips_net = None
    if use_lpips:
        lpips_net = lpips.LPIPS(net="alex").to(args.device)
        lpips_net.eval()

    rows = []
    missing = []

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            missing.append(pred_path.name)
            continue

        pred = np.clip(load_npy(pred_path), 0.0, 1.0)
        gt = np.clip(load_npy(gt_path), 0.0, 1.0)

        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {pred_path.name}: pred={pred.shape}, gt={gt.shape}")

        psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim = structural_similarity(gt, pred, data_range=1.0)

        if use_lpips:
            with torch.no_grad():
                p_t = to_lpips_tensor(pred, args.device)
                g_t = to_lpips_tensor(gt, args.device)
                lp = float(lpips_net(p_t, g_t).item())
        else:
            lp = float("nan")

        rows.append({"name": pred_path.stem, "PSNR_dB": psnr, "SSIM": ssim, "LPIPS": lp})

    if missing:
        print(f"WARNING: {len(missing)} predictions had no matching GT file and were skipped "
              f"(e.g. {missing[0]}).")

    if not rows:
        raise RuntimeError("No matching prediction/GT pairs found -- nothing to score.")

    mean_psnr = float(np.mean([r["PSNR_dB"] for r in rows]))
    mean_ssim = float(np.mean([r["SSIM"] for r in rows]))

    print(f"Scored {len(rows)} images.")
    for r in rows:
        lp_str = f"{r['LPIPS']:.4f}" if use_lpips else "n/a"
        print(f"  {r['name']:<30} PSNR={r['PSNR_dB']:.3f} dB  SSIM={r['SSIM']:.4f}  LPIPS={lp_str}")

    print("\n--- Summary ---")
    print(f"Mean PSNR : {mean_psnr:.3f} dB")
    print(f"Mean SSIM : {mean_ssim:.4f}")
    if use_lpips:
        mean_lpips = float(np.mean([r["LPIPS"] for r in rows]))
        print(f"Mean LPIPS: {mean_lpips:.4f}")
    else:
        print("Mean LPIPS: skipped (--no_lpips or package missing)")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "PSNR_dB", "SSIM", "LPIPS"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-image CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
