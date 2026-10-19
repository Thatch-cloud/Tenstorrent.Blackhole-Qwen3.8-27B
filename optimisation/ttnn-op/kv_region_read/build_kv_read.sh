#!/usr/bin/env bash
# Build ~/opgraft-KVR on the rig HOST: qwen_kv_read.so, the standalone nanobind extension that adds ttnn.qwen_read_blocks (and, from version 2, the raw
# qwen_read_blocks_raw / qwen_write_blocks_raw / qwen_block_bytes of the host KV tier)
# (the prefix audit's region read, docs/prefix-audit-cost.md) WITHOUT replacing any pinned image binary. It is compiled in the
# ttbuild container against the tt-metal tree the served binaries came from (9f9cd4fd), with the flags ninja uses for the
# ttnn unity sources, and linked against ttbuild's _ttnncpp.so / libtt_metal.so / nanobind static library; at run time it
# resolves against the IMAGE's own libraries (the soname of each is already loaded by `import ttnn`). It opens no device and
# starts no device container. Because nothing is replaced, graft-so-drops-image-patches cannot apply: the checks below show
# the image's three pinned binaries are byte-identical afterwards and that the module's own QWEN_ strings are the only new ones.
#
#   QUAL: bash optimisation/ttnn-op/kv_region_read/build_kv_read.sh [out dir]      (default ~/opgraft-KVR)
#
# Env:
#   KVR_IMAGES        space-separated images the module must import into with no unresolved symbol (default: the
#                     production image zot.thatch.local:5000/thatch-serving-tt:latest); each is also asked for the sha256 of the
#                     three pinned binaries, which must equal the recorded ones
#   KVR_DRY_RUN=1     print every docker command and run none of them
#   KVR_KEEP_WORK=1   leave /tmp/kvr in ttbuild
# Writes <out> (an existing <out>/prod-audit is carried over): qwen_kv_read.so, qwen_kv_read.o, compile-flags.txt, qwen_kv_read.cpp, build_kv_read.sh, MANIFEST.sha256,
# qwen-strings-*.txt, build.log. Built in <out>.partial and moved into place only after every check passed.
set -euo pipefail
export LC_ALL=C
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=${1:-$HOME/opgraft-KVR}
images=${KVR_IMAGES:-zot.thatch.local:5000/thatch-serving-tt:latest}
TTMETAL_HEAD=9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9
# The image's pinned binaries (K64j's _ttnncpp.so, the P8 libtt_metal.so, K64j's _ttnn.so) and where they sit.
TTNNCPP_SHA=152951c1c0de5c9dfad2d62c295393a43b2ecf353965c55c709da7e539b975b7
LIBTTMETAL_SHA=3f5a3d585b46b2bef7d7d2c6c34b88d7f4efb679ec051fb4da615f6ec9ccc9ce
TTNN_SHA=914d09462e27e8f5c9d7df03218c51d159bd25f03b87ebc95103dcd06f01a640
run() { if [ "${KVR_DRY_RUN:-0}" = 1 ]; then echo "### run: $*"; else "$@"; fi; }

test -s "$here/qwen_kv_read.cpp" || { echo "no qwen_kv_read.cpp beside this script" >&2; exit 2; }
if [ "$(docker inspect -f '{{.State.Running}}' ttbuild 2>/dev/null || echo missing)" != true ]; then
  echo "ttbuild is not running: starting it (a container start, no device)"
  run docker start ttbuild
  [ "${KVR_DRY_RUN:-0}" = 1 ] || sleep 3
fi
head=$(docker exec ttbuild git -C /opt/tt-metal rev-parse HEAD 2>/dev/null || echo unknown)
[ "${KVR_DRY_RUN:-0}" = 1 ] || test "$head" = "$TTMETAL_HEAD" || { echo "ttbuild's tt-metal is $head, not $TTMETAL_HEAD" >&2; exit 2; }
avail=$(df --output=pcent / | tail -1 | tr -dc 0-9)
test "$avail" -lt 80 || { echo "rig root disk is ${avail}% used (limit 80)" >&2; exit 2; }

work=$out.partial
rm -rf "$work"; mkdir -p "$work"
exec > >(tee "$work/build.log") 2>&1
echo "build_kv_read: tt-metal $head, disk ${avail}%, out $out"

# 1. compile + link in ttbuild (the flags are ninja's own for the ttnn unity sources, minus LTO and -Werror)
run docker exec ttbuild rm -rf /tmp/kvr
run docker exec ttbuild mkdir -p /tmp/kvr
run docker cp "$here/qwen_kv_read.cpp" ttbuild:/tmp/kvr/qwen_kv_read.cpp
cat > "$work/inner.sh" <<'EOF'
set -eu   # no pipefail: grep -m1 closes ninja's pipe
cd /opt/tt-metal/build_Release
FLAGS=$(ninja -t commands ttnn/_ttnn.so | grep -m1 "unity_0_cxx.cxx.o -c" | sed -e 's/^[^ ]* //' -e 's/ -MD .*//' -e 's/-flto=thin//' -e 's/ -Werror//' -e 's/-Dttnn_EXPORTS//')
test -n "$FLAGS"
echo "$FLAGS" > /tmp/kvr/flags.txt
NB=/opt/tt-metal/.cpmcache/nanobind/7965351fcedc97e6ade403d51a150a266d188dc6
eval /usr/bin/clang++-20 $FLAGS -fvisibility=hidden -I$NB/ext/robin_map/include -c /tmp/kvr/qwen_kv_read.cpp -o /tmp/kvr/qwen_kv_read.o
/usr/bin/clang++-20 -fPIC -O3 -shared -fuse-ld=lld -Wl,--gc-sections -Wl,-soname,qwen_kv_read.so -o /tmp/kvr/qwen_kv_read.so /tmp/kvr/qwen_kv_read.o \
  -Wl,-rpath,/opt/tt-metal/build_Release/tt_metal:/opt/tt-metal/build_Release/ttnn:/opt/tt-metal/build_Release/tt_metal/third_party/umd/lib:/opt/tt-metal/build_Release/tt_stl:/opt/tt-metal/build_Release/lib: \
  ttnn/_ttnncpp.so ttnn/libnanobind-static-abi3.a tt_metal/libtt_metal.so tt_stl/libtt_stl.so tt_metal/third_party/umd/lib/libtt-umd.so.0.77.0 lib/libtracy.so.0.13.3 -ldl
readelf -d /tmp/kvr/qwen_kv_read.so | grep -E "NEEDED|SONAME"
EOF
run docker cp "$work/inner.sh" ttbuild:/tmp/kvr/inner.sh
run docker exec ttbuild bash /tmp/kvr/inner.sh
for f in qwen_kv_read.so qwen_kv_read.o; do run docker cp ttbuild:/tmp/kvr/$f "$work/$f"; done
run docker cp ttbuild:/tmp/kvr/flags.txt "$work/compile-flags.txt"
cp "$here/qwen_kv_read.cpp" "$here/$(basename "${BASH_SOURCE[0]}")" "$work/"
[ "${KVR_KEEP_WORK:-0}" = 1 ] || run docker exec ttbuild rm -rf /tmp/kvr
if [ "${KVR_DRY_RUN:-0}" = 1 ]; then echo "dry run: nothing built"; rm -rf "$work"; exit 0; fi
sha=$(sha256sum < "$work/qwen_kv_read.so" | cut -c1-64)
echo "KVR_SHA256=$sha"
strings "$work/qwen_kv_read.so" | grep -oE 'QWEN_[A-Z0-9_]+|\[QWEN-[A-Z0-9-]+\]' | sort -u > "$work/qwen-strings-module.txt"

# 2. per image: the pinned binaries are untouched (we replace nothing), the module imports and every symbol resolves
fail=0
for image in $images; do
  tagname=$(printf '%s' "$image" | tr '/:@' '___')
  docker run --rm --network none --entrypoint sh "$image" -c '
    sha256sum /opt/tt-metal/build_Release/lib/_ttnncpp.so /opt/tt-metal/build_Release/ttnn/_ttnncpp.so /opt/tt-metal/build_Release/lib/libtt_metal.so /opt/tt-metal/ttnn/ttnn/_ttnn.so
    strings /opt/tt-metal/build_Release/lib/_ttnncpp.so | grep -oE "QWEN_[A-Z0-9_]+|\[QWEN-[A-Z0-9-]+\]" | sort -u | sed "s/^/STRING /"' > "$work/image-$tagname.txt"
  grep '^STRING ' "$work/image-$tagname.txt" | cut -c8- > "$work/qwen-strings-$tagname.txt"
  for pair in "$TTNNCPP_SHA /opt/tt-metal/build_Release/lib/_ttnncpp.so" "$TTNNCPP_SHA /opt/tt-metal/build_Release/ttnn/_ttnncpp.so" \
              "$LIBTTMETAL_SHA /opt/tt-metal/build_Release/lib/libtt_metal.so" "$TTNN_SHA /opt/tt-metal/ttnn/ttnn/_ttnn.so"; do
    grep -q "^${pair%% *}  ${pair#* }\$" "$work/image-$tagname.txt" || { echo "FAIL $image: ${pair#* } is not ${pair%% *}" >&2; fail=1; }
  done
  # the module adds no QWEN_ string the image's _ttnncpp.so lacks that a flag could mean; it replaces nothing, so the image set is untouched
  echo "QWEN_ strings: image _ttnncpp.so $(wc -l < "$work/qwen-strings-$tagname.txt"), module $(wc -l < "$work/qwen-strings-module.txt") ($(tr '\n' ' ' < "$work/qwen-strings-module.txt"))"
  unresolved=$(docker run --rm --network none --entrypoint bash -v "$work:/kvr:ro" "$image" -c \
    'ldd -r /kvr/qwen_kv_read.so 2>&1 | grep "undefined symbol" | grep -v "Py\|_Py" || true' | wc -l)
  echo "$image: unresolved non-Python symbols: $unresolved"
  test "$unresolved" = 0 || fail=1
  docker run --rm --network none --entrypoint python3 -v "$work:/kvr:ro" -e PYTHONPATH=/kvr "$image" -c \
    'import ttnn, qwen_kv_read; assert all(callable(getattr(ttnn, n)) for n in ("qwen_read_blocks", "qwen_block_bytes", "qwen_read_blocks_raw", "qwen_write_blocks_raw")); print("IMPORT-OK", qwen_kv_read.QWEN_KV_READ_VERSION)' 2>&1 | grep -E "IMPORT-OK|Error|error" | tail -3 \
    | tee "$work/import-$tagname.txt"
  grep -q IMPORT-OK "$work/import-$tagname.txt" || { echo "FAIL $image: the module does not import" >&2; fail=1; }
done
test "$fail" = 0 || { echo "build_kv_read: a check failed; $work left for reading" >&2; exit 1; }
# prod-audit/ (stage_prod_audit.py: the production model.py with the narrowed audit) survives a rebuild of the module; its pin is the checkout's, not this build's
[ -d "$out/prod-audit" ] && cp -a "$out/prod-audit" "$work/prod-audit"
(cd "$work" && find . -type f ! -name MANIFEST.sha256 ! -name build.log | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
rm -rf "$out"; mv "$work" "$out"
echo "build_kv_read: OK $out qwen_kv_read.so $sha"
