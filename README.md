# KLA Image Restoration — Solvix

Joint denoising (speckle + Gaussian) and 2x super-resolution of degraded
grayscale images, submitted for KLA PS01 ("AI-Based Restoration of
Degraded Images").

## Repository structure

```
root
├── model/
│   ├── architecture.py     # RestorationNet model definition
│   └── model.pt             # trained weights (see "Model weights" below)
├── notebooks/
│   ├── training.ipynb       # thin wrapper that runs train.py (Colab-friendly)
│   └── visualize_results.ipynb  # qualitative + quantitative result inspection
├── evaluate.py               # standalone inference script
├── train.py                  # training script (reproduces training from scratch)
├── compute_metrics.py        # PSNR/SSIM/LPIPS reporting on a val split with GT
├── requirements.txt           # minimal runtime deps for evaluate.py
├── requirements-freeze.txt    # full `pip freeze` from the training environment
├── outputs/                   # restored outputs on the test set
└── README.md
```

## Setup

```bash
git clone https://github.com/Anubhav-Mondal/Solvix.git
cd Solvix
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers everything `evaluate.py` needs (torch, numpy,
pillow). If you also want to run `train.py` or `compute_metrics.py`, install
the extra training/reporting dependencies:

```bash
pip install scikit-image pytorch-msssim lpips
```

(or just `pip install -r requirements-freeze.txt` to match our exact training
environment.)

## Model weights

The trained model is at `model/model.pt`.

- If the file is committed directly in this repo: nothing further to do.
- If it's tracked via Git LFS: run `git lfs pull` after cloning.
- If it's hosted externally (Drive / HuggingFace) because of size limits:
  download it from **[LINK]** and place it at `model/model.pt` before running
  inference.

`evaluate.py` expects the checkpoint at `model/model.pt` by default; override
with `--checkpoint /path/to/other.pt` if needed.

## Running inference (required script)

```bash
python evaluate.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/outputs
```

- `--input_dir`: directory of degraded `.npy` files (single-channel, e.g.
  128x128, values roughly in `[0, 1]`). One image per file.
- `--output_dir`: created if it doesn't exist. For every `<name>.npy` input,
  writes:
  - `<name>.npy` — restored image, float32, `[0, 1]`, 2x the input resolution
    (**primary output, used for scoring**)
  - `<name>.png` — 8-bit grayscale preview, for quick visual inspection only

Optional flags:

```bash
python evaluate.py --input_dir ... --output_dir ... \
    --checkpoint model/model.pt \
    --device cuda \
    --no_tta          # disable 8x test-time augmentation for faster single-pass inference
```

TTA (flip/rotate averaging, 8 forward passes per image) is **on by default**
for best quality. Pass `--no_tta` for faster, single-pass timing — see
"Timing" below for numbers with and without it.

This script only depends on `torch`, `numpy`, and `pillow` (everything in
`requirements.txt`) — no training-only packages, no notebook, no Drive
mounting. It runs standalone on a fresh machine.

## Reproducing training
 
```bash
python train.py \
    --degraded_dir data/train/NoisyLR \
    --gt_dir data/train/GT \
    --out_dir checkpoints \
    --epochs 20 \
    --batch_size 8
```
 
This expects a paired dataset layout:
 
```
data/train/NoisyLR/*.npy   # degraded inputs, e.g. 128x128
data/train/GT/*.npy         # matching filenames, ground truth, e.g. 256x256
```
 
A 10% split of the training set is automatically held out for validation.
Checkpoints are written to `<out_dir>/`:
 
- `model_latest.pt` — saved every `--save_every` epochs (default: every epoch)
- `model_best.pt` — best validation SSIM so far (EMA weights)
- `model_final.pt` — final EMA weights at the end of training
Our submitted `model/model.pt` is `model_best.pt` from a run resumed across
multiple sessions (v1 → v2 → v3 checkpoints), totaling roughly 60 epochs.
Full loss weights, learning rate, and other hyperparameters are the
`argparse` defaults in `train.py` unless noted otherwise in
`notebooks/training.ipynb`.
 
`notebooks/training.ipynb` is a thin Colab wrapper around this same script —
it mounts Drive, sets dataset paths, and calls `train.py`'s `main()`
directly, so it is not a second implementation to keep in sync.
 
## Computing PSNR / SSIM / LPIPS
 
`evaluate.py` intentionally never computes quality metrics — it only takes
degraded inputs and writes restored outputs, since KLA's hidden test set has
no ground truth available to us. To reproduce the numbers reported in our
submission slides, run `evaluate.py` on our own held-out validation split
(which does have matching GT), then score it:
 
```bash
python evaluate.py --input_dir data/val/NoisyLR --output_dir val_outputs/
python compute_metrics.py --pred_dir val_outputs/ --gt_dir data/val/GT --csv_out metrics_report.csv
```
 
Prints per-image and mean PSNR / SSIM / LPIPS, and writes a CSV. Requires
`scikit-image` and `lpips` (not required by `evaluate.py` itself).
 
**Note:** these numbers are computed on our internal validation split, not
on KLA's private/hidden test set.
 
## Restored test outputs
 
`outputs/` contains this model's restored `.npy`/`.png` outputs on the
provided test set, generated with:
 
```bash
python evaluate.py --input_dir <test_set_dir> --output_dir outputs/
```
 
## Model architecture
 
`RestorationNet` (`model/architecture.py`): an RRDB-style (Residual-in-
Residual Dense Block) convolutional network.
 
- Input: `(B, 1, H, W)`, single-channel, roughly `[0, 1]`
- Output: `(B, 1, scale*H, scale*W)`, clamped to `[0, 1]`
- Default config: `feat_ch=48`, `num_blocks=8`, `growth_ch=32`, `scale=2`
- A bicubic-upsampled skip connection gives the network an easy
  low-frequency path, so the learned branch focuses on denoising and
  high-frequency detail recovery.
- Upsampling via PixelShuffle with ICNR-initialized convolutions (reduces
  checkerboard artifacts vs. random init).
Loss (`train.py`): weighted combination of Charbonnier (smooth L1), MS-SSIM,
Sobel gradient (edge) loss, FFT magnitude loss, and LPIPS perceptual loss —
see `CombinedRestorationLoss` for weights.
 
## Timing
 
Measured on 400 test images, NVIDIA RTX 3050:
 
| | Total (400 images) | Per-image | Notes |
|---|---|---|---|
| With TTA (default) | 176 s | **429 ms** | 8x flip/rotate averaged, best quality |
| Without TTA (`--no_tta`) | 38.8 s | **70.5 ms** | single forward pass |
 
TTA gives an ~6x slowdown for the quality gain of averaging 8 augmented
predictions. Use `--no_tta` if inference speed is the priority.
 
## Tech stack
 
- PyTorch 2.6 (CUDA 12.4)
- Trained on: Google Colab, NVIDIA T4
- Inference benchmarked on: NVIDIA RTX 3050
- Model size: 4,608,817 parameters (~70.6 MB on disk)
- Training time: ~8 minutes/epochs