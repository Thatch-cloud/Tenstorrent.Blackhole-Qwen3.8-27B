#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Apply the Qwen3.8-27B custom ttnn ops to a tt-metal checkout.
#
#   apply-to-tt-metal.sh --root <tt-metal checkout> [--tier prstack|ops|k64j] [--check]
#                        [--with-lazy-ccl] [--source-sha256 FILE] [--record FILE] [--prstack-applied FILE]
#
# The checkout must be tt-metal v0.77.0-rc1 (commit 9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9), unmodified.
# What is applied, in order:
#   1. prstack/ORDER: the upstream PR diffs 53314, 53319, 53320, then our one-line FIR fix (0004).
#   2. tt-metal.diff: the registration (sources.cmake, CMakeLists.txt, transformer_nanobind.cpp), the
#      nlp_concat_heads_decode / SDPA factory edits and the two model files the serving image pins.
#   3. ops/: the five op directories, copied into ttnn/cpp/ttnn/operations/transformer/ (refused if present).
#   4. tier k64j only: k64j/, the SDPA decode and prefill-chain kernels and the two factories they need.
#
# --tier prstack applies step 1 only (the Docker stage that builds the PR-stack image).
# --prstack-applied FILE skips step 1 because the tree already carries the stack; FILE must list the names in
# prstack/ORDER (the PR-stack image writes /opt/tt-prstack/APPLIED.txt for this).
#
# --check does not touch the working tree. It replays every patch into a temporary index and checks the
# destinations of the copies against it, so it also works on a no-checkout (blobless) clone.
#
# Exit status 0 = everything applied (or would apply) cleanly and every post-condition held.
set -euo pipefail

BASE=9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9
OPS_DIR=ttnn/cpp/ttnn/operations/transformer
MODEL_DIR=models/demos/blackhole/qwen36/tt

here=$(cd "$(dirname "$0")" && pwd)
root=""
tier="k64j"
check=0
lazy_ccl=0
source_sha=""
record=""
prstack_applied=""

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
die() { echo "apply-to-tt-metal: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --root) root="$2"; shift 2 ;;
    --tier) tier="$2"; shift 2 ;;
    --check) check=1; shift ;;
    --with-lazy-ccl) lazy_ccl=1; shift ;;
    --source-sha256) source_sha="$2"; shift 2 ;;
    --record) record="$2"; shift 2 ;;
    --prstack-applied) prstack_applied="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -n "$root" ] || usage
case "$tier" in prstack|ops|k64j) ;; *) die "--tier must be prstack, ops or k64j" ;; esac
root=$(cd "$root" && pwd)
[ -z "$record" ] && record="$root/.qwen-custom-ops-applied.json"

# 0. the bundle itself must be intact
( cd "$here" && grep -v '^#' MANIFEST.txt | grep -v '^$' | sha256sum -c --quiet ) \
  || die "MANIFEST.txt does not match the bundle files"
echo "bundle files match MANIFEST.txt"

# 1. the checkout must be the base commit, unmodified
head=$(git -C "$root" rev-parse HEAD)
[ "$head" = "$BASE" ] || die "HEAD is $head, expected $BASE (tt-metal v0.77.0-rc1)"
if [ "$check" -eq 0 ] && [ -z "$prstack_applied" ]; then
  git -C "$root" diff --quiet HEAD || die "tracked files in $root are modified; start from a clean checkout"
fi

# patches go through one helper so --check and apply share the order
tmp_index=""
cleanup() { [ -z "$tmp_index" ] || rm -f "$tmp_index"; }
trap cleanup EXIT
if [ "$check" -eq 1 ]; then
  tmp_index=$(mktemp "${TMPDIR:-/tmp}/qwen-ops-index.XXXXXX")
  GIT_INDEX_FILE="$tmp_index" git -C "$root" read-tree HEAD
fi

apply_patch() {
  local file="$1"
  if [ "$check" -eq 1 ]; then
    GIT_INDEX_FILE="$tmp_index" git -C "$root" apply --cached --whitespace=nowarn "$file" || die "does not apply: $file"
  else
    git -C "$root" apply --whitespace=nowarn "$file" || die "does not apply: $file"
  fi
  echo "  ok  $(basename "$file")"
}

exists_in_state() {
  if [ "$check" -eq 1 ]; then
    GIT_INDEX_FILE="$tmp_index" git -C "$root" ls-files --error-unmatch -- "$1" >/dev/null 2>&1
  else
    [ -e "$root/$1" ]
  fi
}

echo "== patches (tier $tier, $([ "$check" -eq 1 ] && echo check || echo apply))"
words() { tr -s "[:space:]" " " < "$1" | sed "s/ $//"; }
if [ -n "$prstack_applied" ]; then
  [ "$(words "$prstack_applied")" = "$(words "$here/prstack/ORDER")" ] || die "$prstack_applied does not list exactly prstack/ORDER"
  echo "  skip prstack (already applied: $(words "$prstack_applied"))"
else
  while read -r name; do
    [ -n "$name" ] || continue
    if   [ -f "$here/prstack/$name.diff" ];  then apply_patch "$here/prstack/$name.diff"
    elif [ -f "$here/prstack/$name.patch" ]; then apply_patch "$here/prstack/$name.patch"
    else die "prstack/ORDER names $name, which is not in prstack/"; fi
  done < "$here/prstack/ORDER"
fi
if [ "$tier" = "prstack" ]; then
  [ "$check" -eq 1 ] && echo "CHECK OK: the PR stack applies cleanly in order" || echo "APPLIED: PR stack only"
  exit 0
fi
apply_patch "$here/tt-metal.diff"
if [ "$lazy_ccl" -eq 1 ]; then apply_patch "$here/optional/lazy-ccl-links.patch"; fi

# 3. op directories: plain copies of directories that must not exist yet. A copy onto an existing
# directory would nest (cp -a / docker cp both do this), which is how a stray gdn_decay/gdn_decay appeared once.
echo "== op directories"
written=()
for dir in "$here"/ops/*/; do
  op=$(basename "$dir")
  dest="$OPS_DIR/$op"
  if exists_in_state "$dest" || [ -d "$root/$dest" ]; then die "$dest already exists"; fi
  if [ "$check" -eq 0 ]; then
    mkdir -p "$root/$dest"
    cp -a "$dir." "$root/$dest/"
  fi
  while IFS= read -r f; do written+=("$dest/$f"); done < <(cd "$dir" && find . -type f | sed 's#^\./##' | sort)
  echo "  ok  $op"
done

# 4. K64j overlay
if [ "$tier" = "k64j" ]; then
  echo "== k64j overlay"
  while IFS= read -r f; do
    dest="$OPS_DIR/$f"
    case "$f" in
      */device/*_program_factory.cpp)
        exists_in_state "$dest" || die "$dest should exist (patched by tt-metal.diff) before the overlay replaces it" ;;
      *)
        if exists_in_state "$dest"; then die "$dest already exists"; fi ;;
    esac
    if [ "$check" -eq 0 ]; then
      mkdir -p "$(dirname "$root/$dest")"
      cp -f "$here/k64j/$f" "$root/$dest"
    fi
    written+=("$dest")
    echo "  ok  $f"
  done < <(cd "$here/k64j" && find . -type f | sed 's#^\./##' | sort)
fi

# 5. post-conditions
echo "== post-conditions"
if [ "$check" -eq 0 ]; then
  for rel in "${written[@]}"; do
    case "$rel" in
      "$OPS_DIR"/sdpa/*|"$OPS_DIR"/sdpa_decode/*) src="$here/k64j/${rel#"$OPS_DIR"/}" ;;
      *) src="$here/ops/${rel#"$OPS_DIR"/}" ;;
    esac
    cmp -s "$src" "$root/$rel" || die "content mismatch after copy: $rel"
  done
  echo "  ok  ${#written[@]} copied files are byte-identical to the bundle"
fi

if [ -n "$source_sha" ]; then
  # The model graft installs over these files and refuses unless they carry the pinned originals.
  rc=0
  if [ "$check" -eq 1 ]; then
    while read -r sum path; do
      rel=$(printf '%s' "$path" | sed "s#^graft/\(.*\)\.orig\$#$MODEL_DIR/\1#")
      got=$(GIT_INDEX_FILE="$tmp_index" git -C "$root" show ":$rel" | sha256sum | cut -c1-64)
      [ "$got" = "$sum" ] || { echo "  FAIL $rel" >&2; rc=1; }
    done < "$source_sha"
  else
    sed "s#  graft/\(.*\)\.orig\$#  $root/$MODEL_DIR/\1#" "$source_sha" | sha256sum -c --quiet || rc=1
  fi
  [ "$rc" -eq 0 ] || die "model sources do not match $source_sha"
  echo "  ok  model sources match $(basename "$source_sha")"
fi

if [ "$check" -eq 1 ]; then
  echo "CHECK OK: every patch applies cleanly in order and every destination is free"
  exit 0
fi

# 6. a build system that tracks timestamps must not treat a unity-build TU as up to date
for rel in "${written[@]}"; do touch -c "$root/$rel"; done
for f in CMakeLists.txt sources.cmake transformer_nanobind.cpp; do touch -c "$root/$OPS_DIR/$f"; done

{
  printf '{\n  "base": "%s",\n  "tier": "%s",\n  "lazy_ccl": %s,\n  "files_copied": %d,\n' "$BASE" "$tier" \
    "$([ "$lazy_ccl" -eq 1 ] && echo true || echo false)" "${#written[@]}"
  printf '  "bundle_manifest_sha256": "%s"\n}\n' "$(sha256sum "$here/MANIFEST.txt" | cut -c1-64)"
} > "$record"
echo "APPLIED: tier $tier; record written to $record"
