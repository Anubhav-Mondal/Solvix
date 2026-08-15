#!/usr/bin/env python3
"""
Quick viewer for a single .npy image file.

Usage:
    python show_npy.py path/to/image.npy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Display a .npy image file.")
    parser.add_argument("path", type=str, help="Path to the .npy file.")
    args = parser.parse_args()

    arr = np.load(args.path).astype(np.float32)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array after squeeze, got shape {arr.shape}")

    plt.imshow(np.clip(arr, 0.0, 1.0), cmap="gray", vmin=0, vmax=1)
    plt.title(args.path)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()