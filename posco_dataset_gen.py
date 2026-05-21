"""Generate TALE-style composition bundles by placing object foregrounds onto POSCO CCTV frames.

For each foreground (from ./object and ./result/*/foreground.png) we sample N posco backgrounds
and place the object on a valid ground area, writing the (background, foreground, segmentation,
location).png 4-tuple expected by TALE.

Outputs land in ./posco_dataset/<idx>/.
"""

import argparse
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OBJECT_DIR = ROOT / "object"
RESULT_DIR = ROOT / "result"
POSCO_DIR = ROOT / "posco"
OUT_DIR = ROOT / "posco_dataset"

FG_SIZE = 512  # output foreground/segmentation canvas size
N_PER_OBJECT = 3
PLACEMENT_SIZE_FRAC = (0.10, 0.20)  # object longest side as fraction of bg width
GROUND_Y_BAND = (0.55, 0.93)
GROUND_X_BAND = (0.08, 0.92)


# ---------- object → (foreground, segmentation) ---------------------------

def _border_color(img: np.ndarray) -> np.ndarray:
    """Estimate background color from the four image-corner patches."""
    h, w = img.shape[:2]
    p = max(6, min(h, w) // 40)
    patches = [
        img[:p, :p],
        img[:p, w - p :],
        img[h - p :, :p],
        img[h - p :, w - p :],
    ]
    return np.median(np.concatenate([x.reshape(-1, 3) for x in patches], axis=0), axis=0)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return (labels == keep).astype(np.uint8) * 255


def extract_foreground(img_path: Path):
    """Return (foreground_rgb_on_white, segmentation_uint8) at FG_SIZE x FG_SIZE.

    Tries alpha first (PNG with transparency), otherwise falls back to solid-bg keying
    using the dominant border color.
    """
    pil = Image.open(img_path)
    if pil.mode == "RGBA":
        arr = np.array(pil)
        rgb = arr[..., :3]
        alpha = arr[..., 3]
        if alpha.min() < 250 and alpha.max() > 5:  # has real transparency
            mask = (alpha > 32).astype(np.uint8) * 255
        else:
            rgb, mask = _solid_bg_key(rgb)
    else:
        rgb = np.array(pil.convert("RGB"))
        rgb, mask = _solid_bg_key(rgb)

    # clean up
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = _largest_component(mask)

    # bbox crop with padding
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise RuntimeError(f"empty mask for {img_path}")
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad = int(0.08 * max(y1 - y0, x1 - x0))
    h, w = mask.shape
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h, y1 + pad)
    x1 = min(w, x1 + pad)
    rgb_c = rgb[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]

    # pad to square centered
    ch, cw = mask_c.shape
    side = max(ch, cw)
    rgb_sq = np.full((side, side, 3), 255, dtype=np.uint8)
    mask_sq = np.zeros((side, side), dtype=np.uint8)
    oy = (side - ch) // 2
    ox = (side - cw) // 2
    # composite object onto white
    a = (mask_c.astype(np.float32) / 255.0)[..., None]
    rgb_sq[oy : oy + ch, ox : ox + cw] = (rgb_c.astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    mask_sq[oy : oy + ch, ox : ox + cw] = mask_c

    rgb_out = cv2.resize(rgb_sq, (FG_SIZE, FG_SIZE), interpolation=cv2.INTER_AREA)
    mask_out = cv2.resize(mask_sq, (FG_SIZE, FG_SIZE), interpolation=cv2.INTER_NEAREST)
    return rgb_out, mask_out


def _solid_bg_key(rgb: np.ndarray):
    """Background-color keying: any pixel close to estimated border color becomes background."""
    bg = _border_color(rgb)
    diff = np.linalg.norm(rgb.astype(np.float32) - bg[None, None], axis=-1)
    # adaptive threshold: ~ 18% of the dynamic range, min 25
    thr = max(25.0, 0.18 * 255.0)
    mask = (diff > thr).astype(np.uint8) * 255
    return rgb, mask


# ---------- POSCO background → ground mask --------------------------------

def ground_mask(bg_bgr: np.ndarray) -> np.ndarray:
    """Compute a permissive ground mask for a posco frame: gravel / track area in lower band."""
    h, w = bg_bgr.shape[:2]
    hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    v = hsv[..., 2]
    b, g, r = bg_bgr[..., 0], bg_bgr[..., 1], bg_bgr[..., 2]

    # low-saturation, mid-brightness pixels (gravel, concrete, asphalt, dirt)
    ground = (s < 70) & (v > 30) & (v < 225)
    # exclude red-dominant pillars / signs
    not_red = ~((r.astype(int) > g.astype(int) + 10) & (r.astype(int) > b.astype(int) + 10))
    # exclude bright sky-like blues
    not_sky = ~((b.astype(int) > r.astype(int) + 8) & (v > 180))
    mask = (ground & not_red & not_sky).astype(np.uint8) * 255

    # restrict to lower vertical band
    y_lo = int(h * GROUND_Y_BAND[0])
    y_hi = int(h * GROUND_Y_BAND[1])
    x_lo = int(w * GROUND_X_BAND[0])
    x_hi = int(w * GROUND_X_BAND[1])
    band = np.zeros_like(mask)
    band[y_lo:y_hi, x_lo:x_hi] = 255
    mask = cv2.bitwise_and(mask, band)

    # clean noise; only keep large blobs
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 0.01 * h * w:
            keep[labels == i] = 255

    if keep.sum() == 0:
        # fallback: pure band so we never fail to place
        keep[y_lo:y_hi, x_lo:x_hi] = 255
    return keep


# ---------- placement -----------------------------------------------------

def sample_location(gmask: np.ndarray, obj_aspect: float, bg_wh, rng: random.Random):
    """Pick a placement rectangle (x, y, w, h) inside the ground mask."""
    bw, bh = bg_wh
    target_w = int(bw * rng.uniform(*PLACEMENT_SIZE_FRAC))
    target_h = max(8, int(target_w / max(obj_aspect, 1e-3)))

    # candidate centers: erode mask by half-size so the rect stays inside
    half_w = target_w // 2
    half_h = target_h // 2
    erode_k = max(3, max(half_w, half_h))
    eroded = cv2.erode(gmask, np.ones((erode_k, erode_k), np.uint8))
    ys, xs = np.where(eroded > 0)
    if ys.size == 0:
        # fallback: shrink object
        target_w = max(40, target_w // 2)
        target_h = max(40, target_h // 2)
        half_w, half_h = target_w // 2, target_h // 2
        erode_k = max(3, max(half_w, half_h))
        eroded = cv2.erode(gmask, np.ones((erode_k, erode_k), np.uint8))
        ys, xs = np.where(eroded > 0)
    if ys.size == 0:
        ys, xs = np.where(gmask > 0)

    # prefer lower placements: weight y linearly
    weights = (ys - ys.min() + 1.0)
    idx = rng.choices(range(len(xs)), weights=weights, k=1)[0]
    cx, cy = int(xs[idx]), int(ys[idx])
    x = max(0, cx - half_w)
    y = max(0, cy - half_h)
    x = min(x, bw - target_w)
    y = min(y, bh - target_h)
    return x, y, target_w, target_h


def write_bundle(out: Path, bg_path: Path, fg_rgb: np.ndarray, seg: np.ndarray, loc_box, bg_wh):
    out.mkdir(parents=True, exist_ok=True)
    # background
    Image.open(bg_path).save(out / "background.png")
    # foreground (RGB on white)
    Image.fromarray(fg_rgb).save(out / "foreground.png")
    # segmentation
    Image.fromarray(seg).save(out / "segmentation.png")
    # location
    bw, bh = bg_wh
    canvas = np.zeros((bh, bw), dtype=np.uint8)
    x, y, w, h = loc_box
    canvas[y : y + h, x : x + w] = 255
    Image.fromarray(canvas).save(out / "location.png")


# ---------- main ---------------------------------------------------------

def collect_sources():
    """Return list of (label, image_path) tuples for every foreground."""
    items = []
    for p in sorted(OBJECT_DIR.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            items.append((f"object_{p.stem}", p))
    for fg in sorted(RESULT_DIR.glob("*/foreground.png")):
        items.append((f"result_{fg.parent.name}", fg))
    return items


def collect_backgrounds():
    bgs = []
    for folder in sorted(POSCO_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        frames = sorted(folder.glob("frame_*.jpg"))
        if frames:
            bgs.extend(frames)
    return bgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PER_OBJECT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_root = Path(args.out)
    if out_root.exists():
        # keep idempotent: overwrite per-folder
        pass
    out_root.mkdir(parents=True, exist_ok=True)

    sources = collect_sources()
    backgrounds = collect_backgrounds()
    print(f"sources={len(sources)} backgrounds={len(backgrounds)}")

    # precompute ground masks per background (small bg count → cheap)
    bg_cache = {}
    counter = 0
    for label, img_path in sources:
        try:
            fg_rgb, seg = extract_foreground(img_path)
        except Exception as e:
            print(f"[skip] {label}: {e}")
            continue

        ys, xs = np.where(seg > 0)
        if ys.size == 0:
            print(f"[skip] {label}: empty seg")
            continue
        obj_h = ys.max() - ys.min() + 1
        obj_w = xs.max() - xs.min() + 1
        obj_aspect = obj_w / obj_h

        # sample N distinct backgrounds for this object
        bg_pool = rng.sample(backgrounds, k=min(args.n, len(backgrounds)))
        for i, bg_path in enumerate(bg_pool):
            bg_bgr = cv2.imread(str(bg_path))
            if bg_bgr is None:
                continue
            bh, bw = bg_bgr.shape[:2]
            key = str(bg_path)
            if key not in bg_cache:
                bg_cache[key] = ground_mask(bg_bgr)
            gmask = bg_cache[key]
            loc = sample_location(gmask, obj_aspect, (bw, bh), rng)

            counter += 1
            out_dir = out_root / f"{counter:03d}_{label}_{i}"
            write_bundle(out_dir, bg_path, fg_rgb, seg, loc, (bw, bh))
            print(f"[ok] {out_dir.name}  bg={bg_path.parent.name}/{bg_path.name}  box={loc}")

    print(f"done: {counter} bundles in {out_root}")


if __name__ == "__main__":
    main()
