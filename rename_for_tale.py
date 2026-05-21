"""Rename posco_dataset/<idx>_<label>_<n> → '<idx> <prompt>' so posco_main.py picks up prompts.

posco_main.py:115 does:  prompt = " ".join(sample.split(" ")[1:])
So the folder name must be: '<idx> <prompt words ...>'.
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "posco_dataset"
DST = ROOT / "posco_dataset_tale"  # safer: copy into new dir so we can re-run

SUFFIX = "on gravel ground at a steel mill"

# object/ filename stem (without trailing digits) → noun phrase
OBJECT_MAP = {
    "box": "a cardboard box",
    "brick": "a red brick",
    "rock": "a large rock",
    "tier": "an old tire",            # filename typo for "tire"
    "wooden-pallet": "a wooden pallet",
    "wooden-pattet": "a wooden pallet",  # filename typo
    "wrench": "a metal wrench",
    "shipping-crate": "a wooden shipping crate",
}

# result/<id> → noun phrase
RESULT_MAP = {
    "00": "red bricks",
    "01": "concrete blocks",
    "02": "a blue oil drum",
    "03": "a ratchet strap",
    "04": "a stack of wooden pallets",
    "05": "concrete debris",
    "06": "a stack of wooden planks",
    "07": "a rusty oil drum",
    "08": "a stack of worn tires",
    "09": "worn tires",
    "10": "a wooden pallet",
    "11": "a black plastic pallet",
    "12": "a large rock",
    "13": "a black plastic crate",
    "14": "black garbage bags",
    "15": "a rusty oil drum",
    "16": "a rusty wheel hub",
    "17": "a coil of black cable",
    "18": "a red brick",
    "19": "a large rock",
}


def label_to_prompt(label: str) -> str:
    """label looks like 'object_brick2' or 'result_07'."""
    kind, _, body = label.partition("_")
    if kind == "result":
        return RESULT_MAP.get(body, "an industrial object")
    # object_*: strip trailing digits from filename stem
    stem = re.sub(r"\d+$", "", body)
    return OBJECT_MAP.get(stem, "an industrial object")


def main():
    if not SRC.exists():
        raise SystemExit(f"missing source dir: {SRC}")
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    bundles = sorted([d for d in SRC.iterdir() if d.is_dir()])
    for b in bundles:
        m = re.match(r"^(\d+)_(.+?)_(\d+)$", b.name)
        if not m:
            print(f"[skip-format] {b.name}")
            continue
        idx, label, n = m.groups()
        prompt = f"{label_to_prompt(label)} {SUFFIX}"
        new_name = f"{idx} {prompt}"
        new_dir = DST / new_name
        shutil.copytree(b, new_dir)
    print(f"wrote {len(bundles)} renamed bundles to {DST}")


if __name__ == "__main__":
    main()
