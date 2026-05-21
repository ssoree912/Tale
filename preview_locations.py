"""Render quick previews: background with the location box outlined in red."""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DS = ROOT / "posco_dataset"
OUT = ROOT / "posco_dataset_preview"

OUT.mkdir(exist_ok=True)

# Sample bundles: every 7th to cover the whole dataset, plus the first/last few
bundles = sorted([d for d in DS.iterdir() if d.is_dir()])
picks = bundles[::7][:18]

for b in picks:
    bg = cv2.imread(str(b / "background.png"))
    loc = cv2.imread(str(b / "location.png"), cv2.IMREAD_GRAYSCALE)
    if bg is None or loc is None:
        continue
    ys, xs = np.where(loc > 128)
    if ys.size == 0:
        continue
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    out = bg.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 4)
    # also light-shade interior
    overlay = out.copy()
    overlay[y0 : y1 + 1, x0 : x1 + 1] = (0, 0, 255)
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 3)
    cv2.imwrite(str(OUT / f"{b.name}.jpg"), out)

print(f"wrote {len(picks)} previews to {OUT}")
