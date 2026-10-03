#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Rewrite MANIFEST.txt after a deliberate change to a file in this directory.
# Format: "<sha256>  <path>" per file (sha256sum -c compatible); lines starting with # are comments.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
{
  echo "# tt-metal custom ops for Qwen3.8-27B: sha256 of every file in this directory except this one."
  echo "# base: tt-metal v0.77.0-rc1 = 9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9"
  echo "# prstack order: $(tr '\n' ' ' < prstack/ORDER | sed 's/ $//')"
  echo "# tiers: ops = prstack + tt-metal.diff + ops/ ; k64j = ops + k64j/"
  echo "# tools: bash, git >= 2.30, sha256sum, awk, sed, cmp"
  find . -type f ! -name MANIFEST.txt | sed 's#^\./##' | LC_ALL=C sort | while IFS= read -r f; do
    printf '%s  %s\n' "$(sha256sum "$f" | cut -c1-64)" "$f"
  done
} > MANIFEST.txt
echo "wrote MANIFEST.txt ($(grep -vc '^#' MANIFEST.txt) files)"
