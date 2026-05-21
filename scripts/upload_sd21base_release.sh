#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-ssoree912/Tale}"
TAG="${2:-sd21base-v1}"
TITLE="${3:-stable-diffusion-2-1-base local model}"
ASSET_DIR="${ASSET_DIR:-release_assets/model_sd21base}"

if ! command -v gh >/dev/null 2>&1; then
  echo "missing gh CLI. Install it and run: gh auth login" >&2
  exit 1
fi

if [ ! -f "$ASSET_DIR/sd21base.sha256" ]; then
  echo "missing $ASSET_DIR/sd21base.sha256" >&2
  exit 1
fi

if ! ls "$ASSET_DIR"/sd21base.tar.part-* >/dev/null 2>&1; then
  echo "missing $ASSET_DIR/sd21base.tar.part-* files" >&2
  exit 1
fi

gh auth status

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release upload "$TAG" \
    "$ASSET_DIR"/sd21base.tar.part-* \
    "$ASSET_DIR"/sd21base.sha256 \
    --repo "$REPO" \
    --clobber
else
  gh release create "$TAG" \
    "$ASSET_DIR"/sd21base.tar.part-* \
    "$ASSET_DIR"/sd21base.sha256 \
    --repo "$REPO" \
    --title "$TITLE" \
    --notes "Split tar archive of stable-diffusion-2-1-base for offline TALE inference."
fi

echo "uploaded model release assets to https://github.com/$REPO/releases/tag/$TAG"
