#!/usr/bin/env python3
"""Count POSCO normal-frame coverage from metadata.jsonl against a train folder."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_STEM_PATTERN = re.compile(r"(?:^|[_ -])(mask|hl)$", re.IGNORECASE)
OBJECT_PATTERN = re.compile(r"object_(\d+)")


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def is_mask_or_highlight(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith(("_mask", "-mask", " mask", "_hl", "-hl", " hl")) or bool(MASK_STEM_PATTERN.search(stem))


def canonical(path_value: str | Path) -> str:
    return str(Path(path_value).expanduser().resolve(strict=False))


def load_metadata_rows(metadata_path: Path) -> list[dict]:
    rows = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid json at {metadata_path}:{line_no}: {exc}") from exc
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def object_label(row: dict) -> str:
    explicit = row.get("object_label")
    if explicit:
        return str(explicit)
    sample = str(row.get("sample_id") or row.get("sample") or "")
    match = OBJECT_PATTERN.search(sample)
    return f"object_{match.group(1)}" if match else "<missing>"


def normal_path(row: dict) -> str | None:
    value = row.get("normal_path") or row.get("background")
    return str(value) if value else None


def collect_train_images(train_dir: Path) -> set[str]:
    images = set()
    for path in train_dir.rglob("*"):
        if not is_image(path):
            continue
        if ".ipynb_checkpoints" in path.parts:
            continue
        if is_mask_or_highlight(path):
            continue
        images.add(canonical(path))
    return images


def print_examples(title: str, values: list[str], limit: int) -> None:
    print(f"{title}={len(values)}")
    for value in values[:limit]:
        print(f"  {value}")
    if len(values) > limit:
        print(f"  ... {len(values) - limit} more")


def main() -> int:
    parser = argparse.ArgumentParser(description="Count metadata normal-frame usage and compare it with train images.")
    parser.add_argument("--metadata", type=Path, default=Path("diffusion_result/input/metadata.jsonl"))
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--object-label", default="object_1", help="Use object_1 as the full-normal-frame coverage reference")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    metadata_path = args.metadata.expanduser().resolve()
    train_dir = args.train_dir.expanduser().resolve()
    if not metadata_path.exists():
        raise SystemExit(f"missing metadata: {metadata_path}")
    if not train_dir.exists():
        raise SystemExit(f"missing train dir: {train_dir}")

    rows = load_metadata_rows(metadata_path)
    object_rows = [row for row in rows if object_label(row) == args.object_label]

    path_counter = Counter()
    missing_normal_path_rows = 0
    for row in object_rows:
        path = normal_path(row)
        if path is None:
            missing_normal_path_rows += 1
            continue
        path_counter[canonical(path)] += 1

    used_normals = set(path_counter)
    train_images = collect_train_images(train_dir)
    overlap = used_normals & train_images
    used_not_in_train = sorted(used_normals - train_images)
    train_not_used = sorted(train_images - used_normals)
    missing_files = sorted(path for path in used_normals if not Path(path).exists())

    object_counts = Counter(object_label(row) for row in rows)

    print(f"metadata={metadata_path}")
    print(f"train_dir={train_dir}")
    print(f"object_label={args.object_label}")
    print(f"metadata_rows_total={len(rows)}")
    print(f"metadata_rows_for_object={len(object_rows)}")
    print(f"unique_normals_used_by_object={len(used_normals)}")
    print(f"train_images_total={len(train_images)}")
    print(f"overlap_used_and_train={len(overlap)}")
    print(f"used_not_in_train={len(used_not_in_train)}")
    print(f"train_not_used={len(train_not_used)}")
    print(f"missing_normal_path_rows={missing_normal_path_rows}")
    print(f"used_paths_missing_on_disk={len(missing_files)}")

    if path_counter:
        total_uses = sum(path_counter.values())
        print(f"normal_use_rows={total_uses}")
        print(f"avg_rows_per_unique_normal={total_uses / max(len(used_normals), 1):.2f}")

    print("\n[object_row_counts]")
    for key, value in sorted(object_counts.items()):
        print(f"  {key}\t{value}")

    print("\n[most_reused_normals]")
    for path, count in path_counter.most_common(args.max_examples):
        print(f"  {count}\t{path}")

    print("\n[examples]")
    print_examples("used_not_in_train_examples", used_not_in_train, args.max_examples)
    print_examples("train_not_used_examples", train_not_used, args.max_examples)
    print_examples("used_paths_missing_on_disk_examples", missing_files, args.max_examples)

    return 1 if used_not_in_train or train_not_used or missing_files or missing_normal_path_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
