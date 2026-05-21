"""Repack inputs (posco_dataset_tale/<idx prompt>/) + outputs (posco_results/<idx prompt>/results_highres.png)
into result/-style numeric folders: posco_final/<idx>/{background,foreground,segmentation,location,result}.png
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "posco_dataset_tale"
OUTPUTS = ROOT / "posco_results_sd"
DST = ROOT / "posco_final"

DST.mkdir(exist_ok=True)

for src in sorted(d for d in INPUTS.iterdir() if d.is_dir()):
    idx = src.name.split(" ", 1)[0]  # "001"
    out = DST / idx
    out.mkdir(exist_ok=True)

    for f in ("background.png", "foreground.png", "segmentation.png", "location.png"):
        shutil.copy(src / f, out / f)

    result = OUTPUTS / src.name / "results_highres.png"
    if result.exists():
        shutil.copy(result, out / "result.png")
    else:
        print(f"[no result] {idx}")

print(f"done → {DST}")
