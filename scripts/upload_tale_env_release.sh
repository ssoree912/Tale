#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-ssoree912/Tale}"
TAG="${2:-tale-env-py310-cu121-v1}"
TITLE="${3:-tale conda-packed environment}"
ASSET_DIR="${ASSET_DIR:-release_assets/tale_env_py310_cu121}"

if ! command -v gh >/dev/null 2>&1; then
  echo "missing gh CLI. Run gh auth login after installing gh." >&2
  exit 1
fi

if ! ls "$ASSET_DIR"/tale_env_py310_cu121.tar.gz.part-* >/dev/null 2>&1; then
  echo "missing split env parts in $ASSET_DIR" >&2
  exit 1
fi

gh auth status

ASSETS=(
  "$ASSET_DIR"/tale_env_py310_cu121.tar.gz.part-*
  "$ASSET_DIR"/tale_env_py310_cu121.parts.sha256
  "$ASSET_DIR"/tale_env_py310_cu121.tar.gz.sha256
)

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release upload "$TAG" "${ASSETS[@]}" --repo "$REPO" --clobber
else
  gh release create "$TAG" "${ASSETS[@]}" \
    --repo "$REPO" \
    --title "$TITLE" \
    --notes "Split conda-pack archive of the TALE Python 3.10 / CUDA 12.1 environment for offline use."
fi

echo "uploaded tale env release assets to https://github.com/$REPO/releases/tag/$TAG"
