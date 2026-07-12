#!/usr/bin/env bash
# Fetches TransNetV2's real weights into weights/transnetv2/.
#
# The `transnetv2` PyPI package (installed via its git dependency in
# pyproject.toml) ships its bundled weights as unfetched Git LFS pointer
# files — a plain `pip`/`uv` git install does not run `git lfs pull`, so
# TransNetV2()'s default constructor fails against the installed package.
# This script clones the upstream repo (which does pull LFS content) and
# copies just the weights directory out.
set -euo pipefail

command -v git-lfs >/dev/null || { echo "git-lfs is required: apt install git-lfs / brew install git-lfs" >&2; exit 1; }

DEST="weights/transnetv2"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

git clone --depth 1 --filter=blob:none https://github.com/soCzech/TransNetV2.git "$TMP_DIR/TransNetV2"

mkdir -p "$DEST"
cp -r "$TMP_DIR/TransNetV2/inference/transnetv2-weights/." "$DEST/"

echo "TransNetV2 weights fetched -> $DEST"
