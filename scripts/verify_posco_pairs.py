#!/usr/bin/env python3
"""Verify POSCO TALE input/background and generated anomaly pairs."""

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
SIZE_DIRS = {"small", "large"}


def load_records(metadata_path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not metadata_path.exists():
        return records

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
    index: dict[str, Path] = {}
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
    index: dict[str, Path] = {}
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


def size_dir_for(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return "<outside>"
    return rel.parts[0] if len(rel.parts) > 1 else "<flat>"


def check_bundle(sample: str, input_dir: Path, errors: list[str]) -> Path | None:
    background_path = input_dir / "background.png"
    for name in REQUIRED_BUNDLE_FILES:
        path = input_dir / name
        if not path.exists():
            errors.append(f"{sample}: missing input bundle file: {path}")
    if not background_path.exists():
        return None
    return background_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify generated POSCO pairs using input/<sample>/background.png as the normal image."
    )
    parser.add_argument("--metadata", type=Path, default=Path("diffusion_result/input/metadata.jsonl"))
    parser.add_argument("--input-root", type=Path, default=Path("diffusion_result/input"))
    parser.add_argument("--result-root", type=Path, default=Path("diffusion_result/result"))
    parser.add_argument("--output-ext", default=".jpg")
    parser.add_argument("--flat-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--require-results", action="store_true", help="Treat input bundles without generated result images as errors")
    parser.add_argument("--strict-metadata", action="store_true", help="Also require metadata paths to agree with input/result paths")
    parser.add_argument("--pairs-jsonl", type=Path, default=None, help="Optional output manifest of completed normal/anomaly pairs")
    args = parser.parse_args()

    metadata_path = args.metadata.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()

    input_index = build_input_index(input_root)
    result_index = build_result_index(result_root, args.output_ext, args.flat_output)
    metadata_index = load_records(metadata_path)

    errors: list[str] = []
    warnings: list[str] = []
    completed_pairs: list[dict[str, str]] = []
    pending_results = 0
    size_counts: dict[str, int] = {}

    for sample, input_dir in sorted(input_index.items()):
        input_size = size_dir_for(input_dir, input_root)
        size_counts[input_size] = size_counts.get(input_size, 0) + 1

        background_path = check_bundle(sample, input_dir, errors)
        result_path = result_index.get(sample)
        if result_path is None or not result_path.exists():
            pending_results += 1
            if args.require_results:
                errors.append(f"{sample}: result image not found under {result_root}")
            continue

        if background_path is None:
            continue

        if args.flat_output and result_path.stem != sample:
            errors.append(f"{sample}: result filename mismatch: {result_path}")

        result_size = size_dir_for(result_path, result_root)
        if input_size in SIZE_DIRS and result_size in SIZE_DIRS and input_size != result_size:
            errors.append(f"{sample}: size dir mismatch: input={input_size}, result={result_size}, result_path={result_path}")

        record = metadata_index.get(sample)
        if record is None:
            warnings.append(f"{sample}: metadata row not found; using input/background.png pair only")
        elif args.strict_metadata:
            meta_input = record.get("input_dir")
            meta_result = record.get("anomaly_path")
            meta_normal = record.get("normal_path") or record.get("background")
            if meta_input and Path(meta_input).expanduser().resolve(strict=False) != input_dir.resolve(strict=False):
                errors.append(f"{sample}: metadata input_dir mismatch: {meta_input} != {input_dir}")
            if meta_result and Path(meta_result).expanduser().resolve(strict=False) != result_path.resolve(strict=False):
                errors.append(f"{sample}: metadata anomaly_path mismatch: {meta_result} != {result_path}")
            if meta_normal and not Path(meta_normal).expanduser().exists():
                errors.append(f"{sample}: metadata normal_path/background does not exist: {meta_normal}")

        completed_pairs.append(
            {
                "sample_id": sample,
                "normal_path": str(background_path),
                "anomaly_path": str(result_path),
                "input_dir": str(input_dir),
                "size_dir": input_size,
            }
        )

    extra_results = sorted(set(result_index) - set(input_index))
    for sample in extra_results[: args.max_examples]:
        warnings.append(f"{sample}: result exists without matching input bundle: {result_index[sample]}")
    if len(extra_results) > args.max_examples:
        warnings.append(f"... {len(extra_results) - args.max_examples} more results without matching input bundles")

    if args.pairs_jsonl is not None:
        out_path = args.pairs_jsonl.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for pair in completed_pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"metadata={metadata_path} exists={metadata_path.exists()} records={len(metadata_index)}")
    print(f"input_root={input_root}")
    print(f"result_root={result_root}")
    print(
        f"input_bundles={len(input_index)} result_images={len(result_index)} "
        f"paired_results={len(completed_pairs)} pending_results={pending_results} extra_results={len(extra_results)}"
    )
    if size_counts:
        print("size_counts=" + ", ".join(f"{key}:{value}" for key, value in sorted(size_counts.items())))
    if args.pairs_jsonl is not None:
        print(f"pairs_jsonl={args.pairs_jsonl.expanduser().resolve()}")
    print(f"warnings={len(warnings)} errors={len(errors)}")

    for message in warnings[: args.max_examples]:
        print(f"[warn] {message}")
    if len(warnings) > args.max_examples:
        print(f"[warn] ... {len(warnings) - args.max_examples} more")

    for message in errors[: args.max_examples]:
        print(f"[error] {message}")
    if len(errors) > args.max_examples:
        print(f"[error] ... {len(errors) - args.max_examples} more")

    if errors:
        return 1
    print("ok: completed result pairs match input/background.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
