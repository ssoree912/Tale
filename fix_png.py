"""
Efficient PNG pre-processing script

What it does (same intent as your original fix_png.py):
1) Binarize location.png and segmentation.png (0/255) using a threshold.
2) Resize background/foreground to 512x512 with a smooth interpolator.
3) Resize location/segmentation to 512x512 with NEAREST (to avoid label mixing).

Default behavior:
- Processes examples/00, examples/01, ... (all subfolders under --root)
- Works in-place (overwrites the original pngs), like your script.

Usage:
  python fix_png_optimized.py
  python fix_png_optimized.py --root examples --size 512 --thr 128
  python fix_png_optimized.py --dirs 00 02 05

Note:
- If you *don't* want segmentation to be binarized (because it's a multi-class label map),
  run with: --no_binarize_seg
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def binarize_to_L(in_path: Path, out_path: Path, thr: int = 128) -> None:
    """Binarize an image to 0/255 single-channel (L) and save."""
    img = Image.open(in_path).convert("L")
    arr = np.asarray(img)
    mask = (arr >= thr).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(out_path)


def resize_image(
    in_path: Path,
    out_path: Path,
    size: int,
    mode: str,
    resample: int,
) -> None:
    """Load -> convert -> resize -> save."""
    img = Image.open(in_path).convert(mode)
    img = img.resize((size, size), resample=resample)
    img.save(out_path)


def iter_example_dirs(root: Path, only_dirs: Iterable[str] | None = None) -> list[Path]:
    """Return example directories under root."""
    if only_dirs:
        return [root / d for d in only_dirs]
    # Default: all subdirs under root (sorted)
    return sorted([p for p in root.iterdir() if p.is_dir()])


def process_dir(
    d: Path,
    size: int = 512,
    thr: int = 128,
    binarize_seg: bool = True,
) -> None:
    """Process one example directory in-place."""
    bg = d / "background.png"
    fg = d / "foreground.png"
    lc = d / "location.png"
    sg = d / "segmentation.png"

    # Basic existence checks (skip silently if missing)
    if not bg.exists() or not fg.exists() or not lc.exists() or not sg.exists():
        return

    # 1) Binarize masks first (in-place)
    binarize_to_L(lc, lc, thr=thr)
    if binarize_seg:
        binarize_to_L(sg, sg, thr=thr)

    # 2) Resize
    # RGB images: smooth resampling
    # resize_image(bg, bg, size=size, mode="RGB", resample=Image.BICUBIC)
    # resize_image(fg, fg, size=size, mode="RGB", resample=Image.BICUBIC)

    # Masks/labels: NEAREST to avoid mixing values
    # Keep as L (single-channel) to stay as a proper mask/label map.
    # resize_image(lc, lc, size=size, mode="L", resample=Image.NEAREST)
    # resize_image(sg, sg, size=size, mode="L", resample=Image.NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("examples"), help="Root folder containing example subfolders (e.g., examples/00)")
    ap.add_argument("--dirs", nargs="*", default=None, help="Specific subfolder names to process (e.g., 00 01 02). If omitted, processes all subfolders under --root.")
    ap.add_argument("--size", type=int, default=512, help="Target width/height (square).")
    ap.add_argument("--thr", type=int, default=128, help="Threshold for binarization.")
    ap.add_argument("--no_binarize_seg", action="store_true", help="Do NOT binarize segmentation.png (useful if it's multi-class labels).")
    args = ap.parse_args()

    root: Path = args.root
    dirs = iter_example_dirs(root, only_dirs=args.dirs)
    for d in dirs:
        process_dir(d, size=args.size, thr=args.thr, binarize_seg=(not args.no_binarize_seg))


if __name__ == "__main__":
    main()
