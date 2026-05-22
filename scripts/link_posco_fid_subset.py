#!/usr/bin/env python3
"""Create symlinked FID folders from selected POSCO diffusion pairs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


REQUIRED_BUNDLE_FILES = (
    "background.png",
    "foreground.png",
    "segmentation.png",
    "location.png",
)


def load_metadata(metadata_path: Path) -> dict[str, dict]:
    records = {}
    if not metadata_path.exists():
        raise SystemExit(f"missing metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid json at {metadata_path}:{line_no}: {exc}") from exc
            sample = record.get("sample_id") or record.get("sample")
            if not sample:
                continue
            record["_line_no"] = line_no
            records.setdefault(sample, record)
    return records


def build_input_index(input_root: Path) -> dict[str, Path]:
    index = {}
    if not input_root.exists():
        return index

    required = set(REQUIRED_BUNDLE_FILES)
    for root, dirs, files in os.walk(input_root):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        if required.issubset(set(files)):
            path = Path(root)
            index.setdefault(path.name, path)
            dirs[:] = []
    return index


def build_result_index(result_root: Path, output_ext: str, flat_output: bool) -> dict[str, Path]:
    index = {}
    if not result_root.exists():
        return index

    ext = output_ext if output_ext.startswith(".") else f".{output_ext}"
    ext = ext.lower()
    for root, dirs, files in os.walk(result_root):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        root_path = Path(root)
        if flat_output:
            for name in files:
                if name.startswith("."):
                    continue
                path = root_path / name
                if path.suffix.lower() == ext:
                    index.setdefault(path.stem, path)
        elif "results_highres.png" in files:
            index.setdefault(root_path.name, root_path / "results_highres.png")
    return index


def size_frac_matches(record: dict, target_min_frac: float | None, tolerance: float) -> bool:
    if target_min_frac is None:
        return True
    value = record.get("placement_size_frac")
    if not isinstance(value, list) or not value:
        return False
    try:
        min_frac = float(value[0])
    except (TypeError, ValueError):
        return False
    return abs(min_frac - target_min_frac) <= tolerance


def size_label_matches(record: dict, size_label: str | None) -> bool:
    if size_label is None:
        return True
    return (record.get("placement_size_label") or record.get("placement_size_dir")) == size_label


def relative_target(src: Path, link_path: Path) -> str:
    return os.path.relpath(src.resolve(), start=link_path.parent.resolve())


def create_symlink(src: Path, link_path: Path, overwrite: bool) -> None:
    if link_path.is_symlink() or link_path.exists():
        if not overwrite:
            current = link_path.resolve() if link_path.exists() or link_path.is_symlink() else None
            if current == src.resolve():
                return
            raise FileExistsError(f"link path already exists: {link_path}")
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(relative_target(src, link_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Link selected POSCO pairs for FID evaluation.")
    parser.add_argument("--metadata", type=Path, default=Path("diffusion_result/input/metadata.jsonl"))
    parser.add_argument("--input-root", type=Path, default=Path("diffusion_result/input"))
    parser.add_argument("--result-root", type=Path, default=Path("diffusion_result/result"))
    parser.add_argument("--out-root", type=Path, default=Path("diffusion_result/fid_score"))
    parser.add_argument("--origin-name", default="origin")
    parser.add_argument("--result-name", default="result")
    parser.add_argument("--output-ext", default=".jpg")
    parser.add_argument("--flat-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--size-label", default="small", help="Metadata placement_size_label to select; use 'any' to disable")
    parser.add_argument("--target-min-frac", type=float, default=0.045, help="Select rows whose placement_size_frac[0] matches this value; use a negative value to disable")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--clear", action="store_true", help="Remove existing origin/result link folders before writing")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing links/files inside output folders")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    metadata_path = args.metadata.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    origin_dir = out_root / args.origin_name
    result_dir = out_root / args.result_name

    size_label = None if args.size_label.lower() == "any" else args.size_label
    target_min_frac = None if args.target_min_frac < 0 else args.target_min_frac

    metadata = load_metadata(metadata_path)
    input_index = build_input_index(input_root)
    result_index = build_result_index(result_root, args.output_ext, args.flat_output)

    selected = []
    skipped_no_input = 0
    skipped_no_result = 0
    skipped_size = 0
    skipped_missing_background = 0

    ext = args.output_ext if args.output_ext.startswith(".") else f".{args.output_ext}"
    for sample, record in sorted(metadata.items()):
        if not size_label_matches(record, size_label) or not size_frac_matches(record, target_min_frac, args.tolerance):
            skipped_size += 1
            continue

        input_dir = input_index.get(sample)
        if input_dir is None:
            skipped_no_input += 1
            continue
        background_path = input_dir / "background.png"
        if not background_path.exists():
            skipped_missing_background += 1
            continue

        anomaly_path = result_index.get(sample)
        if anomaly_path is None or not anomaly_path.exists():
            skipped_no_result += 1
            continue

        selected.append((sample, background_path, anomaly_path))

    if args.clear and not args.dry_run:
        for folder in (origin_dir, result_dir):
            if folder.exists() or folder.is_symlink():
                if folder.is_dir() and not folder.is_symlink():
                    shutil.rmtree(folder)
                else:
                    folder.unlink()

    if not args.dry_run:
        origin_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

    for sample, background_path, anomaly_path in selected:
        origin_link = origin_dir / f"{sample}.png"
        result_link = result_dir / f"{sample}{ext}"
        if not args.dry_run:
            create_symlink(background_path, origin_link, args.overwrite)
            create_symlink(anomaly_path, result_link, args.overwrite)

    print(f"metadata={metadata_path}")
    print(f"input_root={input_root}")
    print(f"result_root={result_root}")
    print(f"out_root={out_root}")
    print(f"filter=size_label:{size_label or 'any'} target_min_frac:{target_min_frac if target_min_frac is not None else 'any'}")
    print(
        f"metadata_rows={len(metadata)} input_bundles={len(input_index)} result_images={len(result_index)} "
        f"selected_pairs={len(selected)}"
    )
    print(
        f"skipped_size={skipped_size} skipped_no_input={skipped_no_input} "
        f"skipped_missing_background={skipped_missing_background} skipped_no_result={skipped_no_result}"
    )
    if args.dry_run:
        print("dry_run=true")
    else:
        print(f"origin_dir={origin_dir}")
        print(f"result_dir={result_dir}")

    for sample, background_path, anomaly_path in selected[: args.max_examples]:
        print(f"[pair] {sample} origin={background_path} result={anomaly_path}")
    if len(selected) > args.max_examples:
        print(f"[pair] ... {len(selected) - args.max_examples} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
