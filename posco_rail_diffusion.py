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

`location.png` is sampled only from a railway mask. If `--rail-mask-dir` is
provided, masks from that directory are used first; otherwise a classical CV
rail-line heuristic is used.
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


def collect_backgrounds(train_dir: Path, camera_ids: set[str] | None) -> list[Path]:
    backgrounds = []
    for path in iter_images(train_dir):
        channel = extract_channel_id(path)
        if camera_ids and channel not in camera_ids:
            continue
        backgrounds.append(path)
    return backgrounds


def collect_channel_reference_masks(channel_background_dir: Path | None) -> dict[str, tuple[np.ndarray, str]]:
    masks: dict[str, tuple[np.ndarray, str]] = {}
    if channel_background_dir is None or not channel_background_dir.exists():
        return masks

    for path in iter_images(channel_background_dir):
        channel = extract_channel_id(path)
        if channel is None:
            continue
        bg_bgr = cv2.imread(str(path))
        if bg_bgr is None:
            print(f"[skip-channel-bg] failed to read {path}", flush=True)
            continue
        masks[channel] = (auto_rail_mask(bg_bgr), f"channel-bg:{path}")
    return masks


def normalize_objects(object_dir: Path, dst_dir: Path | None) -> list[ObjectSpec]:
    paths = list(iter_images(object_dir))
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


def sample_location(
    mask: np.ndarray,
    obj_aspect: float,
    bg_wh: tuple[int, int],
    rng: random.Random,
    size_frac: tuple[float, float],
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
            weights = (ys - ys.min() + 1.0) ** 1.4
            idx = rng.choices(range(len(xs)), weights=weights, k=1)[0]
            cx, cy = int(xs[idx]), int(ys[idx])
            x = min(max(0, cx - half_w), max(0, bw - cur_w))
            y = min(max(0, cy - half_h), max(0, bh - cur_h))
            return x, y, cur_w, cur_h

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise RuntimeError("rail mask is empty")
    idx = rng.randrange(len(xs))
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
    ]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate railway-only POSCO object composites with TALE.")
    parser.add_argument("--train-dir", required=True, type=Path, help="POSCO normal train folder, e.g. ../anomaly_detection/datat/posco/train")
    parser.add_argument("--object-dir", type=Path, default=ROOT / "objects", help="Object image folder")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "posco_rail_dataset", help="TALE input dataset output")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "posco_rail_results", help="Diffusion output directory")
    parser.add_argument("--rail-mask-dir", type=Path, default=None, help="Optional manual rail masks")
    parser.add_argument("--channel-background-dir", type=Path, default=ROOT / "background", help="Channel reference backgrounds such as background/CH002.jpg")
    parser.add_argument("--preview-dir", type=Path, default=ROOT / "posco_rail_preview", help="Rail/location overlay previews")
    parser.add_argument("--normalized-object-dir", type=Path, default=ROOT / "objects_normalized", help="Copied object_1, object_2, ... files")
    parser.add_argument("--no-copy-normalized-objects", action="store_true", help="Use source objects directly instead of writing object_1 files")
    parser.add_argument("--camera-ids", nargs="*", default=None, help="Optional camera folder ids such as 02 04 06 08")
    parser.add_argument("--n-per-object", type=int, default=3, help="Number of backgrounds sampled per object")
    parser.add_argument("--all-backgrounds-per-object", action="store_true", help="Use every background image for every object instead of random sampling")
    parser.add_argument("--background-limit", type=int, default=None, help="Limit available backgrounds after sorting")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--placement-size-frac", type=float, nargs=2, default=(0.10, 0.20), metavar=("MIN", "MAX"))
    parser.add_argument("--prompt-template", default="an industrial object on a railway track at a steel mill")
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

    for path in [out_dir, preview_dir, normalized_object_dir, args.result_dir if args.run_diffusion else None]:
        if path is not None and path.exists() and args.overwrite:
            shutil.rmtree(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    camera_ids = {normalize_channel_id(x) for x in args.camera_ids} if args.camera_ids else None
    if camera_ids is not None:
        camera_ids.discard(None)
    channel_masks = collect_channel_reference_masks(args.channel_background_dir.resolve() if args.channel_background_dir else None)
    print(f"channel_reference_masks={sorted(channel_masks)}", flush=True)
    backgrounds = collect_backgrounds(train_dir, camera_ids)
    if args.background_limit is not None:
        backgrounds = backgrounds[: args.background_limit]
    if not backgrounds:
        raise SystemExit(f"no background images found under: {train_dir}")

    objects = normalize_objects(object_dir, normalized_object_dir)
    if not objects:
        raise SystemExit(f"no object images found under: {object_dir}")

    rng = random.Random(args.seed)
    meta_path = out_dir / "metadata.jsonl"
    counter = 0
    rail_cache: dict[str, tuple[np.ndarray, str, str | None]] = {}

    with meta_path.open("w", encoding="utf-8") as meta_f:
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

                loc = sample_location(
                    rail_mask,
                    obj_aspect,
                    (bw, bh),
                    rng,
                    tuple(args.placement_size_frac),
                )

                counter += 1
                prompt = prompt_for_object(obj.label, args.prompt_template)
                sample_name = f"{counter:06d} {prompt}"
                sample_dir = out_dir / sample_name
                write_bundle(sample_dir, bg_path, fg_rgb, seg, loc, (bw, bh))

                if preview_dir is not None:
                    write_preview(preview_dir / f"{counter:06d}_{obj.label}.jpg", bg_bgr, rail_mask, loc)

                record = {
                    "sample": sample_name,
                    "object_label": obj.label,
                    "object_source": str(obj.source_path),
                    "object_file": str(obj.normalized_path),
                    "background": str(bg_path),
                    "location_box_xywh": list(loc),
                    "rail_mask_source": mask_source,
                    "channel_id": channel_id,
                }
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[ok] {sample_name} bg={bg_path.relative_to(train_dir)} channel={channel_id} box={loc} mask={mask_source}", flush=True)

    print(f"done: wrote {counter} samples to {out_dir}", flush=True)
    print(f"metadata: {meta_path}", flush=True)
    if preview_dir is not None:
        print(f"previews: {preview_dir}", flush=True)

    if args.run_diffusion:
        run_diffusion(args, out_dir)


if __name__ == "__main__":
    main()
