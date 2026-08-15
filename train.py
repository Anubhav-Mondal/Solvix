"""Train the KLA image restoration model.

Expected paired dataset layout:
    <degraded_dir>/*.npy   # e.g. 128x128 degraded images
    <gt_dir>/*.npy         # matching filenames, e.g. 256x256 GT images

Example:
    python train.py \
        --degraded_dir data/train/NoisyLR \
        --gt_dir data/train/GT \
        --out_dir checkpoints \
        --epochs 20 \
        --batch_size 8

The resulting best checkpoint is written as:
    <out_dir>/model_best.pt
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from pytorch_msssim import MS_SSIM

try:
    import lpips
except ImportError:
    lpips = None

from model.architecture import RestorationNet, count_parameters, icnr_init


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Smooth approximation of L1 loss."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SobelGradientLoss(nn.Module):
    """L1 loss between Sobel edge-magnitude maps."""

    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _grad_mag(self, img: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(img, self.kx, padding=1)
        gy = F.conv2d(img, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._grad_mag(pred), self._grad_mag(target))


class FFTLoss(nn.Module):
    """L1 loss between Fourier magnitude spectra."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pf = torch.fft.rfft2(pred, norm="ortho")
        tf = torch.fft.rfft2(target, norm="ortho")
        return F.l1_loss(torch.abs(pf), torch.abs(tf))


class LPIPSLoss(nn.Module):
    """LPIPS loss wrapper for single-channel [0,1] images."""

    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        if lpips is None:
            raise ImportError(
                "LPIPS is enabled but the 'lpips' package is not installed. "
                "Run: pip install -r requirements.txt"
            )
        self.lpips_net = lpips.LPIPS(net=net)
        for p in self.lpips_net.parameters():
            p.requires_grad = False
        self.lpips_net.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred3 = pred.repeat(1, 3, 1, 1) * 2.0 - 1.0
        target3 = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
        return self.lpips_net(pred3, target3).mean()


class CombinedRestorationLoss(nn.Module):
    """Weighted pixel + MS-SSIM + edge + FFT + optional LPIPS loss."""

    def __init__(
        self,
        w_pixel: float = 1.0,
        w_ssim: float = 0.5,
        w_edge: float = 0.1,
        w_fft: float = 0.05,
        w_lpips: float = 0.1,
        use_lpips: bool = True,
        lpips_net: str = "alex",
    ) -> None:
        super().__init__()
        self.pixel_loss = CharbonnierLoss()
        self.ssim_loss = MS_SSIM(data_range=1.0, size_average=True, channel=1)
        self.edge_loss = SobelGradientLoss()
        self.fft_loss = FFTLoss()

        self.use_lpips = use_lpips and w_lpips > 0.0
        self.lpips_loss = LPIPSLoss(net=lpips_net) if self.use_lpips else None
        self.w_pixel = w_pixel
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.w_fft = w_fft
        self.w_lpips = w_lpips

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_pixel = self.pixel_loss(pred, target)
        l_ssim = 1.0 - self.ssim_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        l_fft = self.fft_loss(pred, target)
        total = (
            self.w_pixel * l_pixel
            + self.w_ssim * l_ssim
            + self.w_edge * l_edge
            + self.w_fft * l_fft
        )

        if self.use_lpips:
            assert self.lpips_loss is not None
            l_lpips = self.lpips_loss(pred, target)
            total = total + self.w_lpips * l_lpips
            lpips_value = float(l_lpips.detach().item())
        else:
            lpips_value = 0.0

        values = {
            "pixel": float(l_pixel.detach().item()),
            "ssim": float(l_ssim.detach().item()),
            "edge": float(l_edge.detach().item()),
            "fft": float(l_fft.detach().item()),
            "lpips": lpips_value,
            "total": float(total.detach().item()),
        }
        return total, values


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

NPY_EXTENSIONS = {".npy"}


def list_npy(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in NPY_EXTENSIONS)


def load_npy(path: str | Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got {arr.shape} for {path}")
    return arr


def random_crop_pair(
    degraded: np.ndarray,
    gt: np.ndarray,
    lr_patch: int,
    scale: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = degraded.shape
    if h < lr_patch or w < lr_patch:
        raise ValueError(
            f"Image too small ({h}x{w}) for requested LR patch size {lr_patch}."
        )

    top = random.randint(0, h - lr_patch)
    left = random.randint(0, w - lr_patch)
    d_patch = degraded[top:top + lr_patch, left:left + lr_patch]
    gt_size = lr_patch * scale
    gt_top, gt_left = top * scale, left * scale
    g_patch = gt[gt_top:gt_top + gt_size, gt_left:gt_left + gt_size]
    return d_patch, g_patch


def augment_pair(
    d: np.ndarray,
    g: np.ndarray,
    extra_noise_prob: float = 0.0,
    extra_noise_level_range: Tuple[float, float] = (0.01, 0.05),
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        d, g = np.fliplr(d).copy(), np.fliplr(g).copy()
    if random.random() < 0.5:
        d, g = np.flipud(d).copy(), np.flipud(g).copy()

    k = random.choice([0, 1, 2, 3])
    if k:
        d, g = np.rot90(d, k).copy(), np.rot90(g, k).copy()

    if extra_noise_prob > 0.0 and random.random() < extra_noise_prob:
        level = float(np.random.uniform(*extra_noise_level_range))
        d = d + np.random.randn(*d.shape).astype(np.float32) * level

    return d, g


class PairedRestorationDataset(Dataset):
    """Paired NoisyLR/GT dataset with optional random-crop training."""

    def __init__(
        self,
        degraded_dir: str,
        gt_dir: str,
        lr_patch: int | None = None,
        scale: int = 2,
        augment: bool = True,
        extra_noise_prob: float = 0.0,
        extra_noise_level_range: Tuple[float, float] = (0.01, 0.05),
    ) -> None:
        self.degraded_paths = list_npy(degraded_dir)
        if not self.degraded_paths:
            raise FileNotFoundError(f"No .npy files found in {degraded_dir}")

        gt_dir_path = Path(gt_dir)
        self.gt_paths = [gt_dir_path / p.name for p in self.degraded_paths]
        missing = [p for p in self.gt_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} NoisyLR files have no matching GT file; "
                f"example: {missing[0]}"
            )

        self.lr_patch = lr_patch
        self.scale = scale
        self.augment = augment
        self.extra_noise_prob = extra_noise_prob
        self.extra_noise_level_range = extra_noise_level_range

    def __len__(self) -> int:
        return len(self.degraded_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        degraded = load_npy(self.degraded_paths[idx])
        gt = load_npy(self.gt_paths[idx])

        expected_gt_shape = (
            degraded.shape[0] * self.scale,
            degraded.shape[1] * self.scale,
        )
        if gt.shape != expected_gt_shape:
            raise ValueError(
                f"Shape mismatch for {self.degraded_paths[idx].name}: "
                f"degraded={degraded.shape}, expected GT={expected_gt_shape}, got={gt.shape}"
            )

        if self.lr_patch is not None:
            degraded, gt = random_crop_pair(
                degraded, gt, self.lr_patch, self.scale
            )

        if self.augment:
            degraded, gt = augment_pair(
                degraded,
                gt,
                extra_noise_prob=self.extra_noise_prob,
                extra_noise_level_range=self.extra_noise_level_range,
            )

        degraded_t = torch.from_numpy(degraded.copy()).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt.copy()).unsqueeze(0).float()
        return degraded_t, gt_t


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }

    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.clone()

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def build_dataset(args: argparse.Namespace) -> Tuple[Dataset, Dataset]:
    full_train = PairedRestorationDataset(
        args.degraded_dir,
        args.gt_dir,
        lr_patch=args.lr_patch,
        scale=args.scale,
        augment=True,
        extra_noise_prob=args.extra_noise_prob,
    )
    full_val = PairedRestorationDataset(
        args.degraded_dir,
        args.gt_dir,
        lr_patch=args.lr_patch,
        scale=args.scale,
        augment=False,
    )

    n_val = max(1, int(0.1 * len(full_train)))
    n_train = len(full_train) - n_val
    if n_train < 1:
        raise ValueError("Dataset must contain at least 2 images for train/validation split")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(full_train), generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    return Subset(full_train, train_indices), Subset(full_val, val_indices)


def load_resume(
    checkpoint_path: str,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> Tuple[int, float, float]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint does not contain a 'model_state_dict'")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if "ema_state_dict" in ckpt:
        ema.shadow = ckpt["ema_state_dict"]
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_loss = float(ckpt.get("best_loss", float("inf")))
    best_val_ssim = float(ckpt.get("best_val_ssim", -1.0))
    return start_epoch, best_loss, best_val_ssim


def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ssims: list[float] = []
    psnrs: list[float] = []

    with torch.no_grad():
        for degraded, gt in loader:
            degraded = degraded.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            pred = model(degraded)

            pred_np = pred.squeeze().detach().cpu().numpy()
            gt_np = gt.squeeze().detach().cpu().numpy()
            ssims.append(structural_similarity(gt_np, pred_np, data_range=1.0))
            psnrs.append(peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0))

    return float(np.mean(ssims)), float(np.mean(psnrs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the KLA restoration model")
    parser.add_argument("--degraded_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--lr_patch",
        type=int,
        default=None,
        help="LR crop size; default None means full image training.",
    )
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--feat_ch", type=int, default=48)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--growth_ch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extra_noise_prob",
        type=float,
        default=0.3,
        help="Probability of additional Gaussian noise augmentation.",
    )
    parser.add_argument(
        "--no_lpips",
        action="store_true",
        help="Disable the LPIPS perceptual loss term.",
    )
    parser.add_argument("--w_lpips", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Dataset: {args.degraded_dir} <-> {args.gt_dir}")

    train_set, val_set = build_dataset(args)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_set) >= args.batch_size,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = RestorationNet(
        in_ch=1,
        feat_ch=args.feat_ch,
        num_blocks=args.num_blocks,
        growth_ch=args.growth_ch,
        scale=args.scale,
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    for module in model.upsample:
        if isinstance(module, nn.Conv2d):
            icnr_init(module, scale=2)

    ema = EMA(model, decay=args.ema_decay)
    criterion = CombinedRestorationLoss(
        use_lpips=not args.no_lpips,
        w_lpips=args.w_lpips,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    start_epoch = 1
    best_loss = float("inf")
    best_val_ssim = -1.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_loss, best_val_ssim = load_resume(
            args.resume,
            model,
            ema,
            optimizer,
            scheduler,
            device,
        )
        print(
            f"Resuming at epoch {start_epoch}; "
            f"best SSIM={best_val_ssim:.4f}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = {
            "pixel": 0.0,
            "ssim": 0.0,
            "edge": 0.0,
            "fft": 0.0,
            "lpips": 0.0,
            "total": 0.0,
        }

        for step, (degraded, gt) in enumerate(train_loader, start=1):
            degraded = degraded.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(degraded)
            loss, parts = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            ema.update(model)

            for key in running:
                running[key] += parts[key]

            if step % 20 == 0 or step == len(train_loader):
                n = step
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"total={running['total']/n:.4f} "
                    f"pixel={running['pixel']/n:.4f} "
                    f"ssim={running['ssim']/n:.4f} "
                    f"edge={running['edge']/n:.4f} "
                    f"lpips={running['lpips']/n:.4f}"
                )

        scheduler.step()

        # Evaluate EMA weights, matching the original training workflow.
        eval_model = copy.deepcopy(model)
        ema.apply_to(eval_model)
        eval_model.to(device)
        val_ssim, val_psnr = validate(eval_model, val_loader, device)
        eval_model.cpu()
        del eval_model

        avg_loss = running["total"] / max(len(train_loader), 1)
        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"  [val] SSIM={val_ssim:.4f} PSNR={val_psnr:.2f}dB\n"
            f"Epoch {epoch}/{args.epochs} done in {elapsed:.1f}s -- "
            f"avg_total_loss={avg_loss:.4f} lr={current_lr:.2e}"
        )

        if epoch % args.save_every == 0:
            latest_path = out_dir / "model_latest.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.shadow,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "best_val_ssim": best_val_ssim,
                    "args": vars(args),
                },
                latest_path,
            )
            print(f"Saved checkpoint: {latest_path}")

        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            best_loss = min(best_loss, avg_loss)
            best_path = out_dir / "model_best.pt"

            # Save the EMA weights as the main model_state_dict so the file is
            # immediately usable by evaluate.py without special handling.
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in ema.shadow.items()
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "ema_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "best_val_ssim": best_val_ssim,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"New best SSIM {best_val_ssim:.4f} -> {best_path}")

    # Final EMA checkpoint.
    ema_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
    final_path = out_dir / "model_final.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": ema_state,
            "args": vars(args),
        },
        final_path,
    )
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
