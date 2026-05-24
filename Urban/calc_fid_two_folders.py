#!/usr/bin/env python3
"""Compute FID for two image folders using UrbanGIRAFFE's internal FID code.

Example:
    python tools/calc_fid_two_folders.py \
        --real_dir /path/to/real \
        --fake_dir /path/to/fake \
        --device cuda \
        --model_path /path/to/inception_v3_google-0cc3c7bd.pth \
        --batch_size 16
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

from lib.evaluators.fid.calc_fid import (
    calculate_activation_statistics,
    calculate_frechet_distance,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(folder: str, recursive: bool = False, filename: str = "") -> List[str]:
    path = Path(folder)
    if not path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    pattern = "**/*" if recursive else "*"
    files = [
        str(p)
        for p in sorted(path.glob(pattern))
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and (not filename or p.name == filename)
    ]
    if not files:
        raise FileNotFoundError(
            f"No supported image files found in {folder}. "
            f"Filename filter: {filename or 'none'}. "
            f"Supported: {sorted(IMAGE_EXTS)}"
        )
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FID between a real-image folder and a fake-image folder."
    )
    parser.add_argument("--real_dir", required=True, help="Folder containing real images.")
    parser.add_argument("--fake_dir", required=True, help="Folder containing fake/generated images.")
    parser.add_argument(
        "--real_filename", default="",
        help="Optional exact filename filter for real images, e.g. background.png."
    )
    parser.add_argument(
        "--fake_filename", default="",
        help="Optional exact filename filter for fake/generated images."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for Inception feature extraction."
    )
    parser.add_argument(
        "--dims", type=int, default=2048, choices=[64, 192, 768, 2048],
        help="Inception feature dimension. 2048 is the standard FID setting."
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help="Device for feature extraction."
    )
    parser.add_argument(
        "--model_path", default="",
        help=(
            "Optional local torchvision Inception v3 weight path, e.g. "
            "inception_v3_google-0cc3c7bd.pth. If omitted, torchvision default "
            "cached/downloaded weights are used."
        )
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively search subfolders for images."
    )
    parser.add_argument(
        "--save_json", default="",
        help="Optional output JSON file for the results."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    use_cuda = False
    if args.device == "cuda":
        use_cuda = True
    elif args.device == "auto":
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False

    if args.model_path:
        args.model_path = os.path.expanduser(args.model_path)
        if not Path(args.model_path).is_file():
            raise FileNotFoundError(f"Model weight file does not exist: {args.model_path}")

    real_files = collect_images(
        args.real_dir, recursive=args.recursive, filename=args.real_filename
    )
    fake_files = collect_images(
        args.fake_dir, recursive=args.recursive, filename=args.fake_filename
    )

    if args.real_filename:
        print(f"Real filename filter: {args.real_filename}")
    if args.fake_filename:
        print(f"Fake filename filter: {args.fake_filename}")
    print(f"Found {len(real_files)} real images")
    print(f"Found {len(fake_files)} fake images")
    if len(real_files) < 100 or len(fake_files) < 100:
        print(
            "Warning: very small sample counts can make FID highly unstable. "
            "Use the result only as a rough reference."
        )

    mu_real, sigma_real = calculate_activation_statistics(
        real_files,
        batch_size=args.batch_size,
        dims=args.dims,
        cuda=use_cuda,
        model_path=args.model_path,
    )
    mu_fake, sigma_fake = calculate_activation_statistics(
        fake_files,
        batch_size=args.batch_size,
        dims=args.dims,
        cuda=use_cuda,
        model_path=args.model_path,
    )

    fid_value = float(
        calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake, eps=1e-6)
    )

    result = {
        "fid": fid_value,
        "real_dir": os.path.abspath(args.real_dir),
        "fake_dir": os.path.abspath(args.fake_dir),
        "real_filename": args.real_filename,
        "fake_filename": args.fake_filename,
        "num_real": len(real_files),
        "num_fake": len(fake_files),
        "batch_size": args.batch_size,
        "dims": args.dims,
        "device": "cuda" if use_cuda else "cpu",
        "model_path": os.path.abspath(args.model_path) if args.model_path else "",
    }

    print("\nResult")
    print(json.dumps(result, indent=2))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved JSON to: {out_path}")


if __name__ == "__main__":
    main()
