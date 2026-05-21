"""Build POSCO railway-object TALE inputs and optionally run diffusion.

This script expects normal POSCO CCTV frames under a train directory such as:

    ../anomaly_detection/datat/posco/train/02/*.jpg
    ../anomaly_detection/datat/posco/train/04/*.jpg

It creates TALE-style sample folders:

    <out>/<idx> <prompt>/
        background.png
        foreground.png
        segmentation.png
        location.png

`location.png` is sampled only from a railway mask. Mask files such as
`background/02_mask.*` are matched to CCTV frames named like `[CH002] ...jpg`.
Legacy `*_hl.*` files are still accepted. If `--require-channel-mask` is used,
every input frame must have a matching channel mask before diffusion runs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from posco_dataset_gen import extract_foreground


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHANNEL_PATTERN = re.compile(r"CH\s*0*(\d{1,3})", re.IGNORECASE)
NUMERIC_CHANNEL_PATTERN = re.compile(r"^(?:ch)?0*(\d{1,3})$", re.IGNORECASE)
MASK_CHANNEL_PATTERN = re.compile(r"^(?:ch)?0*(\d{1,3})(?:[_ -]*(?:mask|hl))$", re.IGNORECASE)
SAMPLE_NUMBER_PATTERN = re.compile(r"(?:^|_)(\d{6,7})(?:\D*$|$)")


@dataclass(frozen=True)
class ObjectSpec:
    label: str
    source_path: Path
    normalized_path: Path


def natural_key(path: Path) -> list[object]:
    parts: list[object] = []
    token = ""
    is_digit = False
    for ch in path.as_posix():
        ch_is_digit = ch.isdigit()
        if token and ch_is_digit != is_digit:
            parts.append(int(token) if is_digit else token.lower())
            token = ""
        token += ch
        is_digit = ch_is_digit
    if token:
        parts.append(int(token) if is_digit else token.lower())
    return parts


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=natural_key):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if ".ipynb_checkpoints" not in path.parts:
                yield path


def normalize_channel_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    match = CHANNEL_PATTERN.search(text)
    if match is None:
        match = MASK_CHANNEL_PATTERN.match(text)
    if match is None:
        match = NUMERIC_CHANNEL_PATTERN.match(text)
    if match is None:
        return None
    return f"CH{int(match.group(1)):03d}"


def extract_channel_id(path: Path) -> str | None:
    # Prefer explicit channel markers in the filename, e.g. "[CH002] 2026...jpg".
    for text in [path.name, path.stem, *reversed(path.parts)]:
        channel = normalize_channel_id(text.strip("[](){} _-"))
        if channel is not None:
            return channel
    return None


def is_channel_mask_image(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith(("_mask", "-mask", " mask", "_hl", "-hl", " hl"))


def is_highlight_image(path: Path) -> bool:
    return is_channel_mask_image(path)


def collect_backgrounds(train_dir: Path, camera_ids: set[str] | None) -> list[Path]:
    backgrounds = []
    for path in iter_images(train_dir):
        if is_channel_mask_image(path):
            continue
        channel = extract_channel_id(path)
        if camera_ids and channel not in camera_ids:
            continue
        backgrounds.append(path)
    return backgrounds


def green_highlight_mask(highlight_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(highlight_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    b, g, r = cv2.split(highlight_bgr)

    # The rail annotations are saved as jpg/jpeg in this project, so the green
    # paint can be muted by compression. Use both saturated-green and loose
    # green-dominance tests, then remove only tiny speckles.
    hsv_green = (h >= 35) & (h <= 95) & (s >= 35) & (v >= 45)
    dominant_green = (g.astype(np.int16) > r.astype(np.int16) + 5) & (g.astype(np.int16) > b.astype(np.int16) + 5) & (g > 40)
    # Also accept already-converted black/white highlight masks.
    white_mask = (r >= 180) & (g >= 180) & (b >= 180)
    mask = (hsv_green | dominant_green | white_mask).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    min_area = max(100, int(0.00005 * mask.shape[0] * mask.shape[1]))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def collect_highlight_masks(
    highlight_dir: Path | None,
    only_highlight_files: bool = False,
) -> dict[str, tuple[np.ndarray, str]]:
    masks: dict[str, tuple[np.ndarray, str]] = {}
    if highlight_dir is None or not highlight_dir.exists():
        return masks

    for path in iter_images(highlight_dir):
        if only_highlight_files and not is_channel_mask_image(path):
            continue
        channel = extract_channel_id(path)
        if channel is None:
            continue
        highlight_bgr = cv2.imread(str(path))
        if highlight_bgr is None:
            print(f"[skip-mask] failed to read {path}", flush=True)
            continue
        mask = green_highlight_mask(highlight_bgr)
        if mask.sum() == 0:
            print(f"[skip-mask] no rail mask pixels found in {path}", flush=True)
            continue
        masks[channel] = (mask, f"mask:{path}")
    return masks


def collect_channel_reference_masks(
    channel_background_dir: Path | None,
    highlight_dir: Path | None = None,
) -> dict[str, tuple[np.ndarray, str]]:
    masks = collect_highlight_masks(highlight_dir)
    if channel_background_dir is None or not channel_background_dir.exists():
        return masks

    inline_highlights = collect_highlight_masks(channel_background_dir, only_highlight_files=True)
    for channel, mask_spec in inline_highlights.items():
        masks[channel] = mask_spec

    for path in iter_images(channel_background_dir):
        if is_channel_mask_image(path):
            continue
        channel = extract_channel_id(path)
        if channel is None or channel in masks:
            continue
        bg_bgr = cv2.imread(str(path))
        if bg_bgr is None:
            print(f"[skip-channel-bg] failed to read {path}", flush=True)
            continue
        masks[channel] = (auto_rail_mask(bg_bgr), f"channel-bg:{path}")
    return masks


def default_highlight_dir() -> Path | None:
    for candidate in [ROOT / "backroung_highlight", ROOT / "background_highlight"]:
        if candidate.exists():
            return candidate
    return ROOT / "backroung_highlight"


def normalize_objects(object_dir: Path, dst_dir: Path | None, limit: int | None = None) -> list[ObjectSpec]:
    paths = list(iter_images(object_dir))
    if limit is not None:
        paths = paths[:limit]
    specs: list[ObjectSpec] = []
    if dst_dir is not None:
        dst_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(paths, start=1):
        label = f"object_{idx}"
        dst = src
        if dst_dir is not None:
            dst = dst_dir / f"{label}{src.suffix.lower()}"
            shutil.copy2(src, dst)
        specs.append(ObjectSpec(label=label, source_path=src, normalized_path=dst))
    return specs


def read_manual_rail_mask(
    bg_path: Path,
    train_dir: Path,
    mask_dir: Path | None,
    channel_id: str | None,
) -> np.ndarray | None:
    if mask_dir is None:
        return None

    rel = bg_path.relative_to(train_dir)
    candidates = [
        mask_dir / rel.with_suffix(".png"),
        mask_dir / rel.parent / f"{rel.stem}.png",
        mask_dir / f"{rel.parent.name}.png",
        mask_dir / f"{rel.stem}.png",
    ]
    if channel_id is not None:
        candidates.extend([
            mask_dir / f"{channel_id}.png",
            mask_dir / f"{channel_id.lower()}.png",
            mask_dir / f"{int(channel_id[2:])}.png",
            mask_dir / f"{int(channel_id[2:]):02d}.png",
        ])

    for candidate in candidates:
        if candidate.exists():
            mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"failed to read rail mask: {candidate}")
            return (mask > 128).astype(np.uint8) * 255
    return None


def auto_rail_mask(bg_bgr: np.ndarray) -> np.ndarray:
    """Estimate railway/track corridors from long rail-like lines.

    The mask is intentionally wider than the visible metal rail lines so a full
    object bounding box can be sampled on the track area.
    """
    h, w = bg_bgr.shape[:2]
    y0 = int(h * 0.32)
    y1 = int(h * 0.98)
    x0 = int(w * 0.03)
    x1 = int(w * 0.97)

    gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    roi = gray[y0:y1, x0:x1]
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(roi, 45, 140)

    min_len = max(140, int(w * 0.11))
    max_gap = max(18, int(w * 0.025))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=90,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    line_mask = np.zeros((h, w), dtype=np.uint8)
    scored_lines = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            lx1, ly1, lx2, ly2 = [int(v) for v in raw]
            lx1 += x0
            lx2 += x0
            ly1 += y0
            ly2 += y0
            dx = lx2 - lx1
            dy = ly2 - ly1
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx)))
            angle = min(angle, 180.0 - angle)
            if not (3.0 <= angle <= 82.0):
                continue
            mid_y = 0.5 * (ly1 + ly2)
            score = length * (1.0 + mid_y / h)
            scored_lines.append((score, lx1, ly1, lx2, ly2))

    scored_lines.sort(reverse=True)
    rail_thickness = max(45, int(w * 0.035))
    for _, lx1, ly1, lx2, ly2 in scored_lines[:18]:
        cv2.line(line_mask, (lx1, ly1), (lx2, ly2), 255, rail_thickness)

    # Join nearby parallel rail lines into a usable track corridor.
    close_w = max(35, int(w * 0.04))
    close_h = max(25, int(h * 0.035))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_w, close_h))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
    line_mask = cv2.dilate(line_mask, kernel, iterations=1)

    roi_mask = np.zeros_like(line_mask)
    roi_mask[y0:y1, x0:x1] = 255
    line_mask = cv2.bitwise_and(line_mask, roi_mask)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
    keep = np.zeros_like(line_mask)
    min_area = max(2000, int(0.0025 * h * w))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255

    if keep.sum() == 0:
        # Fail closed enough to stay near where rails usually appear, but still
        # keep generation possible for later manual inspection.
        pts = np.array(
            [
                [int(w * 0.20), int(h * 0.55)],
                [int(w * 0.80), int(h * 0.55)],
                [int(w * 0.92), int(h * 0.97)],
                [int(w * 0.08), int(h * 0.97)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(keep, [pts], 255)

    return keep


def rail_mask_for_background(
    bg_path: Path,
    train_dir: Path,
    bg_bgr: np.ndarray,
    mask_dir: Path | None,
    channel_masks: dict[str, tuple[np.ndarray, str]],
) -> tuple[np.ndarray, str, str | None]:
    channel_id = extract_channel_id(bg_path)
    mask = read_manual_rail_mask(bg_path, train_dir, mask_dir, channel_id)
    source = "manual" if mask is not None else "auto"

    if mask is None and channel_id in channel_masks:
        mask, source = channel_masks[channel_id]

    if mask is None:
        mask = auto_rail_mask(bg_bgr)

    h, w = bg_bgr.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 128).astype(np.uint8) * 255
    return mask, source, channel_id


def choose_mask_index(ys: np.ndarray, rng: random.Random, strategy: str) -> int:
    if strategy == "mixed":
        strategy = "uniform" if rng.random() < 0.70 else "depth-weighted"
    if strategy == "uniform":
        return rng.randrange(len(ys))

    # Depth-weighted keeps some perspective realism by preferring lower image
    # regions, but it can reduce placement diversity if used exclusively.
    weights = (ys - ys.min() + 1.0) ** 1.4
    return rng.choices(range(len(ys)), weights=weights, k=1)[0]


def placement_size_for(
    placement_idx: int,
    base_size_frac: tuple[float, float],
    large_size_frac: tuple[float, float] | None,
    large_every: int,
) -> tuple[tuple[float, float], str]:
    if large_size_frac is not None and large_every > 0 and placement_idx % large_every == 0:
        return large_size_frac, "large"
    return base_size_frac, "small" if large_size_frac is not None else "base"


def sample_location(
    mask: np.ndarray,
    obj_aspect: float,
    bg_wh: tuple[int, int],
    rng: random.Random,
    size_frac: tuple[float, float],
    placement_strategy: str,
) -> tuple[int, int, int, int]:
    bw, bh = bg_wh
    target_w = int(bw * rng.uniform(*size_frac))
    target_h = max(8, int(target_w / max(obj_aspect, 1e-3)))

    for scale in (1.0, 0.75, 0.55, 0.40):
        cur_w = max(32, int(target_w * scale))
        cur_h = max(32, int(target_h * scale))
        half_w = cur_w // 2
        half_h = cur_h // 2
        kernel_w = max(3, half_w)
        kernel_h = max(3, half_h)
        kernel = np.ones((kernel_h, kernel_w), np.uint8)
        eroded = cv2.erode(mask, kernel)
        ys, xs = np.where(eroded > 0)
        if ys.size:
            idx = choose_mask_index(ys, rng, placement_strategy)
            cx, cy = int(xs[idx]), int(ys[idx])
            x = min(max(0, cx - half_w), max(0, bw - cur_w))
            y = min(max(0, cy - half_h), max(0, bh - cur_h))
            return x, y, cur_w, cur_h

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise RuntimeError("rail mask is empty")
    idx = choose_mask_index(ys, rng, placement_strategy)
    cx, cy = int(xs[idx]), int(ys[idx])
    x = min(max(0, cx - target_w // 2), max(0, bw - target_w))
    y = min(max(0, cy - target_h // 2), max(0, bh - target_h))
    return x, y, target_w, target_h


def write_bundle(
    out: Path,
    bg_path: Path,
    fg_rgb: np.ndarray,
    seg: np.ndarray,
    loc_box: tuple[int, int, int, int],
    bg_wh: tuple[int, int],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    Image.open(bg_path).convert("RGB").save(out / "background.png")
    Image.fromarray(fg_rgb).save(out / "foreground.png")
    Image.fromarray(seg).save(out / "segmentation.png")

    bw, bh = bg_wh
    location = np.zeros((bh, bw), dtype=np.uint8)
    x, y, w, h = loc_box
    location[y : y + h, x : x + w] = 255
    Image.fromarray(location).save(out / "location.png")


def write_preview(
    out_path: Path,
    bg_bgr: np.ndarray,
    rail_mask: np.ndarray,
    loc_box: tuple[int, int, int, int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview = bg_bgr.copy()
    green = np.zeros_like(preview)
    green[..., 1] = 255
    preview = np.where((rail_mask > 0)[..., None], cv2.addWeighted(preview, 0.65, green, 0.35, 0), preview)
    x, y, w, h = loc_box
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.imwrite(str(out_path), preview)


def prompt_for_object(label: str, template: str) -> str:
    return template.format(object=label)


def sample_name_for(channel_id: str | None, object_label: str, counter: int, prompt: str, sample_name_format: str) -> str:
    if sample_name_format == "posco":
        channel = (channel_id or "unknown").lower()
        return f"{channel}_{object_label}_{counter:07d}"
    return f"{counter:06d} {prompt}"


def sample_number_from_name(name: str) -> int | None:
    matches = SAMPLE_NUMBER_PATTERN.findall(name)
    if not matches:
        return None
    return int(matches[-1])


def max_existing_sample_number(*roots: Path | None) -> int:
    max_number = 0
    for root in roots:
        if root is None or not root.exists():
            continue
        for path in root.iterdir():
            number = sample_number_from_name(path.stem if path.is_file() else path.name)
            if number is not None:
                max_number = max(max_number, number)
    return max_number


def run_diffusion(args: argparse.Namespace, data_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "posco_main.py"),
        "--model_path",
        str(args.model_path),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(args.result_dir),
        "--tprime",
        str(args.tprime),
        "--tau",
        str(args.tau),
        "--inv_guidance_scale",
        str(args.inv_guidance_scale),
        "--comp_guidance_scale",
        str(args.comp_guidance_scale),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--crop_padding",
        str(args.crop_padding),
        "--default_prompt",
        args.prompt_template,
    ]
    if args.flat_results:
        cmd.extend(["--flat_output", "--output_ext", ".jpg"])
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate railway-only POSCO object composites with TALE.")
    parser.add_argument("--train-dir", required=True, type=Path, help="POSCO normal train folder, e.g. ../anomaly_detection/datat/posco/train")
    parser.add_argument("--object-dir", type=Path, default=ROOT / "objects", help="Object image folder")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "examples" / "posco_background" / "inputs", help="TALE input dataset output")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "examples" / "posco_background" / "results", help="Diffusion output directory")
    parser.add_argument("--rail-mask-dir", type=Path, default=None, help="Optional manual rail masks")
    parser.add_argument("--channel-background-dir", type=Path, default=ROOT / "background", help="Channel masks such as background/02_mask.jpg; legacy *_hl files also work")
    parser.add_argument("--highlight-dir", type=Path, default=default_highlight_dir(), help="Optional separate folder of green-highlighted rail masks")
    parser.add_argument("--require-channel-mask", action="store_true", help="Fail if an input frame channel has no matching channel mask")
    parser.add_argument("--preview-dir", type=Path, default=ROOT / "examples" / "posco_background" / "preview", help="Rail/location overlay previews")
    parser.add_argument("--normalized-object-dir", type=Path, default=ROOT / "objects_normalized", help="Copied object_1, object_2, ... files")
    parser.add_argument("--no-copy-normalized-objects", action="store_true", help="Use source objects directly instead of writing object_1 files")
    parser.add_argument("--object-limit", type=int, default=None, help="Limit object files after sorting")
    parser.add_argument("--camera-ids", nargs="*", default=None, help="Optional camera folder ids such as 02 04 06 08")
    parser.add_argument("--n-per-object", type=int, default=3, help="Number of backgrounds sampled per object")
    parser.add_argument("--all-backgrounds-per-object", action="store_true", help="Use every background image for every object instead of random sampling")
    parser.add_argument("--background-limit", type=int, default=None, help="Limit available backgrounds after sorting")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--placement-size-frac", type=float, nargs=2, default=(0.10, 0.20), metavar=("MIN", "MAX"), help="Base object width fraction range")
    parser.add_argument("--large-placement-size-frac", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Optional large object width fraction range")
    parser.add_argument("--large-placement-every", type=int, default=0, help="Use large-placement-size-frac every Nth placement within each object/background pair")
    parser.add_argument("--placement-strategy", choices=("mixed", "uniform", "depth-weighted"), default="mixed", help="Rail-mask sampling strategy; uniform gives more diverse locations")
    parser.add_argument("--placements-per-pair", type=int, default=1, help="Number of random locations for each object/background pair")
    parser.add_argument("--prompt-template", default="an industrial object on a railway track at a steel mill")
    parser.add_argument("--sample-name-format", choices=("prompt", "posco"), default="prompt", help="Use prompt folders or POSCO ids such as ch002_object_1_0000001")
    parser.add_argument("--flat-results", action="store_true", help="Save diffusion results directly as output_dir/<sample>.jpg")
    parser.add_argument("--resume-numbering", action="store_true", help="Continue sample numbering from existing out/result/preview files")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing out/result/preview directories first")
    parser.add_argument("--run-diffusion", action="store_true", help="Run posco_main.py after building the dataset")

    parser.add_argument("--model_path", type=Path, default=ROOT / "stable-diffusion-2-1-base")
    parser.add_argument("--tprime", type=int, default=12)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--inv_guidance_scale", type=float, default=5.0)
    parser.add_argument("--comp_guidance_scale", type=float, default=10.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--crop_padding", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.train_dir.resolve()
    object_dir = args.object_dir.resolve()
    out_dir = args.out_dir.resolve()
    preview_dir = args.preview_dir.resolve() if args.preview_dir else None
    normalized_object_dir = None if args.no_copy_normalized_objects else args.normalized_object_dir.resolve()

    if not train_dir.exists():
        raise SystemExit(f"missing train dir: {train_dir}")
    if not object_dir.exists():
        raise SystemExit(f"missing object dir: {object_dir}")

    if args.overwrite and args.resume_numbering:
        raise SystemExit("--overwrite and --resume-numbering cannot be used together")

    for path in [out_dir, preview_dir, normalized_object_dir, args.result_dir if args.run_diffusion else None]:
        if path is not None and path.exists() and args.overwrite:
            shutil.rmtree(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    camera_ids = {normalize_channel_id(x) for x in args.camera_ids} if args.camera_ids else None
    if camera_ids is not None:
        camera_ids.discard(None)
    channel_masks = collect_channel_reference_masks(
        args.channel_background_dir.resolve() if args.channel_background_dir else None,
        args.highlight_dir.resolve() if args.highlight_dir else None,
    )
    print(f"channel_reference_masks={sorted(channel_masks)}", flush=True)
    backgrounds = collect_backgrounds(train_dir, camera_ids)
    if args.background_limit is not None:
        backgrounds = backgrounds[: args.background_limit]
    if not backgrounds:
        raise SystemExit(f"no background images found under: {train_dir}")
    if args.require_channel_mask:
        missing_masks: list[str] = []
        missing_channels: list[str] = []
        for bg_path in backgrounds:
            channel = extract_channel_id(bg_path)
            rel = bg_path.relative_to(train_dir)
            if channel is None:
                missing_channels.append(str(rel))
            elif channel not in channel_masks:
                missing_masks.append(f"{channel}:{rel}")
        if missing_channels or missing_masks:
            lines = ["channel mask check failed"]
            if missing_channels:
                lines.append("missing channel in filenames: " + ", ".join(missing_channels[:20]))
            if missing_masks:
                lines.append("missing masks: " + ", ".join(missing_masks[:20]))
            lines.append("expected mask names like background/02_mask.png or background/08_mask.jpg")
            raise SystemExit("\n".join(lines))

    objects = normalize_objects(object_dir, normalized_object_dir, args.object_limit)
    if not objects:
        raise SystemExit(f"no object images found under: {object_dir}")

    rng = random.Random(args.seed)
    meta_path = out_dir / "metadata.jsonl"
    result_dir_for_numbering = args.result_dir.resolve() if args.result_dir else None
    counter = max_existing_sample_number(out_dir, preview_dir, result_dir_for_numbering) if args.resume_numbering else 0
    start_counter = counter
    if args.resume_numbering:
        print(f"resume_numbering_start={counter + 1:07d}", flush=True)
    rail_cache: dict[str, tuple[np.ndarray, str, str | None]] = {}

    meta_mode = "a" if args.resume_numbering else "w"
    with meta_path.open(meta_mode, encoding="utf-8") as meta_f:
        for obj in objects:
            try:
                fg_rgb, seg = extract_foreground(obj.normalized_path)
            except Exception as exc:
                print(f"[skip-object] {obj.source_path}: {exc}", flush=True)
                continue

            ys, xs = np.where(seg > 0)
            if ys.size == 0:
                print(f"[skip-object] {obj.source_path}: empty segmentation", flush=True)
                continue
            obj_h = ys.max() - ys.min() + 1
            obj_w = xs.max() - xs.min() + 1
            obj_aspect = obj_w / max(obj_h, 1)

            if args.all_backgrounds_per_object:
                bg_pool = backgrounds
            else:
                sample_count = min(args.n_per_object, len(backgrounds))
                bg_pool = rng.sample(backgrounds, k=sample_count)

            for bg_path in bg_pool:
                bg_bgr = cv2.imread(str(bg_path))
                if bg_bgr is None:
                    print(f"[skip-bg] failed to read {bg_path}", flush=True)
                    continue
                bh, bw = bg_bgr.shape[:2]

                key = str(bg_path)
                if key not in rail_cache:
                    rail_cache[key] = rail_mask_for_background(bg_path, train_dir, bg_bgr, args.rail_mask_dir, channel_masks)
                rail_mask, mask_source, channel_id = rail_cache[key]

                for placement_idx in range(1, args.placements_per_pair + 1):
                    placement_size_frac, placement_size_label = placement_size_for(
                        placement_idx,
                        tuple(args.placement_size_frac),
                        tuple(args.large_placement_size_frac) if args.large_placement_size_frac else None,
                        args.large_placement_every,
                    )
                    loc = sample_location(
                        rail_mask,
                        obj_aspect,
                        (bw, bh),
                        rng,
                        placement_size_frac,
                        args.placement_strategy,
                    )

                    counter += 1
                    prompt = prompt_for_object(obj.label, args.prompt_template)
                    sample_name = sample_name_for(channel_id, obj.label, counter, prompt, args.sample_name_format)
                    sample_dir = out_dir / sample_name
                    write_bundle(sample_dir, bg_path, fg_rgb, seg, loc, (bw, bh))

                    if preview_dir is not None:
                        write_preview(preview_dir / f"{sample_name}.jpg", bg_bgr, rail_mask, loc)

                    record = {
                        "sample": sample_name,
                        "prompt": prompt,
                        "object_label": obj.label,
                        "object_source": str(obj.source_path),
                        "object_file": str(obj.normalized_path),
                        "background": str(bg_path),
                        "location_box_xywh": list(loc),
                        "placement_index": placement_idx,
                        "placement_strategy": args.placement_strategy,
                        "placement_size_label": placement_size_label,
                        "placement_size_frac": list(placement_size_frac),
                        "rail_mask_source": mask_source,
                        "channel_id": channel_id,
                    }
                    meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"[ok] {sample_name} bg={bg_path.relative_to(train_dir)} channel={channel_id} box={loc} placement={placement_idx}/{args.placements_per_pair} size={placement_size_label}:{placement_size_frac} mask={mask_source}", flush=True)

    print(f"done: wrote {counter - start_counter} new samples to {out_dir}", flush=True)
    print(f"metadata: {meta_path}", flush=True)
    if preview_dir is not None:
        print(f"previews: {preview_dir}", flush=True)

    if args.run_diffusion:
        run_diffusion(args, out_dir)


if __name__ == "__main__":
    main()
