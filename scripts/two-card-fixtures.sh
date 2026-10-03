#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Fetch the DFlash2 draft fixtures and lay them out the way docker/two-card/qwen-c2-serving.Dockerfile expects:
#
#   scripts/two-card-fixtures.sh OUT_DIR
#
# Writes OUT_DIR/fixture/{attention,convolution,mlp,projection,selector,layer-1..4}/ and OUT_DIR/draft-config/config.json,
# the two directories stage 8 copies into the image (BUILD.md, section 6). Run it from the repository root, on a machine
# with network access to huggingface.co, python3 with the fixtures' requirements (torch, transformers, safetensors) and
# about 10 GB free in FIXTURE_CACHE.
#
#   FIXTURE_CACHE   a writable directory the downloads are cached in (default: ~/.cache/qwen-experiments)
#
# The fixtures come from the public repository incoai/Qwen3.8-27B-DFlash2 at the pinned revision below (its head has
# moved since). This script has not been run end to end by the maintainers from a clean checkout of the public branch.
set -euo pipefail

OUT=${1:?usage: two-card-fixtures.sh OUT_DIR}
CACHE=${FIXTURE_CACHE:-$HOME/.cache/qwen-experiments}
REVISION=dedf8df68adfb1afeaf7b7480c0a0243108177b4
[ -f scripts/ci/draft_attention_fixture.py ] || { echo "run me from the repository root" >&2; exit 1; }

mkdir -p "$CACHE" "$OUT/fixture" "$OUT/draft-config"
for component in attention convolution mlp projection selector; do
  fetcher="draft_${component}_fixture.py"
  if [ "$component" = projection ]; then fetcher=draft_projection_full_fixture.py; fi
  dest="$CACHE/dflash2-$component-$REVISION"
  timeout -k 10 1200 python3 "scripts/ci/$fetcher" --reuse-verified --output "$dest"
  rm -rf "$OUT/fixture/$component"; cp -a "$dest" "$OUT/fixture/$component"
done
stack="$CACHE/dflash2-stack-$REVISION"
timeout -k 10 4800 python3 -u scripts/ci/draft_remaining_layers_fixture.py --reuse-verified --output "$stack"
for layer in 1 2 3 4; do
  rm -rf "$OUT/fixture/layer-$layer"; cp -a "$stack/layer-$layer" "$OUT/fixture/layer-$layer"
done
curl -fsSL --max-time 60 "https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/resolve/$REVISION/config.json" \
  > "$OUT/draft-config/config.json"
echo "fixtures written to $OUT/fixture and $OUT/draft-config"
