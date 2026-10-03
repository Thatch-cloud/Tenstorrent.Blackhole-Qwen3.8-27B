#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Verify a tt-metal tree that has had the custom ops applied and been built.
#
#   verify.sh --root <tt-metal checkout> [--tier ops|k64j] [--no-python]
#
# Checks, in order:
#   1. both _ttnncpp.so copies exist; the set of QWEN_* strings in them is EXACTLY the set in markers.txt for the tier;
#   2. the Python extension (_ttnn.so) is the one that binds the five new ops (import ttnn, call-check each name);
#   3. the two _ttnncpp.so copies are the same build (warning only: copy one over the other if they differ).
# Prints the sha256 of the binary it checked. That value is specific to your build; the serving profiles that pin
# a binary hash take it as a build-time value (see BUILD.md).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=""
tier="k64j"
python_check=1
die() { echo "verify: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --root) root="$2"; shift 2 ;;
    --tier) tier="$2"; shift 2 ;;
    --no-python) python_check=0; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -n "$root" ] || die "usage: verify.sh --root DIR [--tier ops|k64j] [--no-python]"
case "$tier" in ops|k64j) ;; *) die "--tier must be ops or k64j" ;; esac

section() { awk -v s="[$1]" '$0==s{on=1;next} /^\[/{on=0} on && $0!~/^#/ && NF{print}' "$here/markers.txt"; }

lib_a="$root/build_Release/lib/_ttnncpp.so"
lib_b="$root/build_Release/ttnn/_ttnncpp.so"
for f in "$lib_a" "$lib_b"; do [ -f "$f" ] || die "missing $f (has the build run?)"; done

expected=$(section "$tier" | sort -u)
fail=0
for f in "$lib_a" "$lib_b"; do
  got=$(grep -aoE 'QWEN_[A-Z0-9_]+' "$f" | sort -u)
  if [ "$got" = "$expected" ]; then
    echo "ok   markers match for $(basename "$(dirname "$f")")/$(basename "$f") ($(printf '%s\n' "$expected" | wc -l) strings)"
  else
    echo "FAIL markers differ in $f" >&2
    diff <(printf '%s\n' "$expected") <(printf '%s\n' "$got") >&2 || true
    fail=1
  fi
done

if [ "$python_check" -eq 1 ]; then
  names=$(section bindings | tr '\n' ' ')
  ( cd "$root" && python3 - $names <<'PY'
import sys, ttnn
names = sys.argv[1:]
missing = [n for n in names if not callable(getattr(ttnn.transformer, n, None))]
assert not missing, "ttnn.transformer lacks: %s" % missing
print("ok   ttnn.transformer binds: " + ", ".join(names))
PY
  ) || fail=1
fi

sum_a=$(sha256sum "$lib_a" | cut -c1-64)
sum_b=$(sha256sum "$lib_b" | cut -c1-64)
if [ "$sum_a" != "$sum_b" ]; then
  echo "warn the two _ttnncpp.so copies differ (lib/ $sum_a, ttnn/ $sum_b); the serving image runs one binary at both paths" >&2
fi
echo "binary sha256 (lib/): $sum_a"
[ "$fail" -eq 0 ] || die "verification failed"
echo "VERIFY OK (tier $tier)"
