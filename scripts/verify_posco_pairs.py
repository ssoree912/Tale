#!/usr/bin/env python3
"""Verify POSCO normal/input/anomaly pair paths from metadata.jsonl."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REQUIRED_BUNDLE_FILES = (
    "background.png",
    "foreground.png",
    "segmentation.png",
    "location.png",
)


def existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def load_records(metadata_path: Path) -> list[dict]:
    records = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid json at {metadata_path}:{line_no}: {exc}") from exc
            record["_line_no"] = line_no
            records.append(record)
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


def resolve_input_dir(record: dict, input_root: Path, input_index: dict[str, Path]) -> Path | None:
    explicit = existing_path(record.get("input_dir"))
    if explicit is not None:
        return explicit

    sample = record.get("sample_id") or record.get("sample")
    size_dir = record.get("placement_size_dir") or record.get("placement_size_label")
    if sample and size_dir:
        candidate = input_root / size_dir / sample
        if candidate.exists():
            return candidate
    if sample:
        return input_index.get(sample)
    return None


def resolve_result_path(
    record: dict,
    result_root: Path,
    result_index: dict[str, Path],
    output_ext: str,
    flat_output: bool,
) -> Path | None:
    explicit = existing_path(record.get("anomaly_path"))
    if explicit is not None:
        return explicit

    sample = record.get("sample_id") or record.get("sample")
    size_dir = record.get("placement_size_dir") or record.get("placement_size_label")
    ext = output_ext if output_ext.startswith(".") else f".{output_ext}"
    if sample and size_dir:
        if flat_output:
            candidate = result_root / size_dir / f"{sample}{ext}"
        else:
            candidate = result_root / size_dir / sample / "results_highres.png"
        if candidate.exists():
            return candidate
    if sample:
        return result_index.get(sample)
    return None


def record_value_path(record: dict, key: str, fallback: Path | None = None) -> Path | None:
    value = existing_path(record.get(key))
    return value if value is not None else fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify POSCO metadata normal/input/anomaly path pairs.")
    parser.add_argument("--metadata", type=Path, default=Path("diffusion_result/input/metadata.jsonl"))
    parser.add_argument("--input-root", type=Path, default=Path("diffusion_result/input"))
    parser.add_argument("--result-root", type=Path, default=Path("diffusion_result/result"))
    parser.add_argument("--output-ext", default=".jpg")
    parser.add_argument("--flat-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--strict-warnings", action="store_true")
    args = parser.parse_args()

    metadata_path = args.metadata.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()

    if not metadata_path.exists():
        raise SystemExit(f"missing metadata: {metadata_path}")

    records = load_records(metadata_path)
    input_index = build_input_index(input_root)
    result_index = build_result_index(result_root, args.output_ext, args.flat_output)

    errors: list[str] = []
    warnings: list[str] = []
    seen_samples: dict[str, int] = {}
    size_counts: dict[str, int] = {}
    result_pairs = 0

    for record in records:
        line_no = record["_line_no"]
        sample = record.get("sample_id") or record.get("sample")
        if not sample:
            errors.append(f"line {line_no}: missing sample/sample_id")
            continue

        if sample in seen_samples:
            errors.append(f"line {line_no}: duplicate sample {sample!r}; first seen at line {seen_samples[sample]}")
        else:
            seen_samples[sample] = line_no

        size_dir = record.get("placement_size_dir") or record.get("placement_size_label") or "<none>"
        size_counts[size_dir] = size_counts.get(size_dir, 0) + 1

        normal_path = record_value_path(record, "normal_path", existing_path(record.get("background")))
        input_dir = resolve_input_dir(record, input_root, input_index)
        result_path = resolve_result_path(record, result_root, result_index, args.output_ext, args.flat_output)

        if "normal_path" not in record:
            warnings.append(f"line {line_no} {sample}: missing normal_path; using legacy background field")
        if "anomaly_path" not in record:
            warnings.append(f"line {line_no} {sample}: missing anomaly_path; inferred from result folder")
        if "input_dir" not in record:
            warnings.append(f"line {line_no} {sample}: missing input_dir; inferred from input folder")

        if normal_path is None:
            errors.append(f"line {line_no} {sample}: missing normal_path/background")
        elif not normal_path.exists():
            errors.append(f"line {line_no} {sample}: normal path does not exist: {normal_path}")

        if input_dir is None:
            errors.append(f"line {line_no} {sample}: input bundle not found under {input_root}")
        elif not input_dir.exists():
            errors.append(f"line {line_no} {sample}: input_dir does not exist: {input_dir}")
        else:
            for name in REQUIRED_BUNDLE_FILES:
                bundle_path = record_value_path(record, name.replace(".png", "_path"), input_dir / name)
                if bundle_path is None or not bundle_path.exists():
                    errors.append(f"line {line_no} {sample}: missing bundle file {name}: {bundle_path}")
            if size_dir in {"small", "large"} and input_dir.parent.name != size_dir:
                errors.append(f"line {line_no} {sample}: input size dir mismatch: metadata={size_dir}, path={input_dir}")

        if result_path is None:
            errors.append(f"line {line_no} {sample}: result image not found under {result_root}")
        elif not result_path.exists():
            errors.append(f"line {line_no} {sample}: result path does not exist: {result_path}")
        else:
            result_pairs += 1
            if result_path.stem != sample and args.flat_output:
                errors.append(f"line {line_no} {sample}: result filename mismatch: {result_path}")
            if size_dir in {"small", "large"} and result_path.parent.name != size_dir:
                errors.append(f"line {line_no} {sample}: result size dir mismatch: metadata={size_dir}, path={result_path}")

    print(f"metadata={metadata_path}")
    print(f"input_root={input_root}")
    print(f"result_root={result_root}")
    print(f"records={len(records)} input_bundles={len(input_index)} result_images={len(result_index)} paired_results={result_pairs}")
    if size_counts:
        print("size_counts=" + ", ".join(f"{key}:{value}" for key, value in sorted(size_counts.items())))
    print(f"warnings={len(warnings)} errors={len(errors)}")

    for message in warnings[: args.max_examples]:
        print(f"[warn] {message}")
    if len(warnings) > args.max_examples:
        print(f"[warn] ... {len(warnings) - args.max_examples} more")

    for message in errors[: args.max_examples]:
        print(f"[error] {message}")
    if len(errors) > args.max_examples:
        print(f"[error] ... {len(errors) - args.max_examples} more")

    if errors or (args.strict_warnings and warnings):
        return 1
    print("ok: metadata paths are paired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
