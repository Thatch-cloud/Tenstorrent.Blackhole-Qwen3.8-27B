#!/usr/bin/env bash
# Build ~/opgraft-K64e on the rig: ~/opgraft-K64d plus the [QWEN-SDPA] stage-1 sdpa_decode
# (tail-only mask, forced compact tree scratch, per-call q_chunk_size sentinel).
#
#   K64e = K64d's contents (attn_prep, nlp_concat_heads_decode, ...)
#        + _ttnncpp.so / _ttnn.so rebuilt in ttbuild over the F1-F8 factory
#        + sdpa_decode/  a copy of ttbuild's whole op directory, which is the image's
#                        directory plus reader_decode_qwen.cpp and sdpa_flash_decode_qwen.cpp.
#                        Its factory .cpp is put back to the audited 3e0a69af bytes: the
#                        compiled factory lives in the .so, and the serving path's
#                        sdpa_tree_scratch.audit(patched=True) hashes the on-disk one.
#
# lever_n_m3native_run_arm.sh mounts $KOPGRAFT64/sdpa_decode over the container's op
# directory only when the graft has one, so kernels JIT-compile from it.
#
# Usage (on the rig, after shipping this directory to $K64E_SRC):
#   bash ~/kwork64/k64e/build_k64e.sh
# Env: K64E_SRC (default ~/kwork64/k64e), K64E_BASE_GRAFT (~/opgraft-K64d), K64E_GRAFT
# (~/opgraft-K64e), K64E_IMAGE (the arm's default serving image), K64E_SKIP_IMAGE_COMPARE=1
# (only if that image is absent), K64E_KEEP_STAGED=1 (leave ttbuild on the qwen factory).
#
# ttbuild's sources are left as they were found (factory 3e0a69af, no qwen kernel files) unless
# K64E_KEEP_STAGED=1; everything it replaces is saved under $K64E_SRC/work-<stamp>/backup.
# build_Release is rebuilt from the restored factory (step 6) and checked to have lost the
# [QWEN-SDPA] branch: docker cp keeps the saved factory's OLD mtime, so without a touch and a
# rebuild ninja would call the unity TU up to date and the next graft built in ttbuild (a K64d
# rebuild, K64f, ...) would silently link the qwen factory. On a failure the EXIT trap restores
# and touches the sources but does not rebuild; it says so.
set -euo pipefail

S=${K64E_SRC:-$HOME/kwork64/k64e}
BASE=${K64E_BASE_GRAFT:-$HOME/opgraft-K64d}
GRAFT=${K64E_GRAFT:-$HOME/opgraft-K64e}
# Assembled beside the final name and moved into place only after every check passes, so a
# failed build never leaves a graft an arm could mount.
G=$GRAFT.partial
IMAGE=${K64E_IMAGE:-sha256:e41ef884f4c8e07ce6632511fac19364af35dc73d601f05f5191784a64a3a768}
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
XOPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer
D=$OPS/sdpa_decode
F=$D/device/sdpa_decode_program_factory.cpp
KD=$D/device/kernels
PREFILL=$OPS/sdpa/device/sdpa_program_factory.cpp

FACTORY_BASE=3e0a69af9563ae8899db1363286d6dc6cdc1744e0154887dc91a630b16084a4a
FACTORY_UNPATCHED=05708e6d9ddeddfdf13303d8f8fa391941d73b742ea3a380beeb0883ce8d4792
FACTORY_QWEN=1b54abd3fe466a058046939e2ec0366505f80fde631dba387c591985fa3da301
READER_ALL=49a05926b437e2ca90d7e01c60e85a6b11f333375a6159fa02c9ff6f78af764e
COMPUTE_ALL=d24769bdcbb8635f83f5f91a301fe0d89298d38263d4493a39c6d2decb57867f
WRITER_ALL=734c90c01c7a7174497133fae9df80110ead55275955faeb566d345bdccb60b8
DATAFLOW_COMMON=e4623a2254559eaec4450ebfab0f9c5732e02acfe4d8126bd5eeb7efe0fdc608
RT_ARGS_COMMON=1b52c60d78ada6f08effd326c2ed2407b3a74cf0db2353fadbe51b088610aec8
READER_QWEN=55d8fe5e1bc87d9ada56f4a6279afa523865028c163b36803bfd07013f488702
COMPUTE_QWEN=8776fcc7420c6f27a9c7ae06c54c391225a00ce78322c5397970d74a5063ca8a
PREFILL_COMBINED=fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783
K64D_TTNNCPP_PREFIX=06865d8e

W=$S/work-$(date +%Y%m%dT%H%M%S)
mkdir -p "$W/backup"
echo "### build_k64e $(date -Is) work=$W"

hsha() { sha256sum "$1" | cut -c1-64; }
csha() { docker exec ttbuild sha256sum "$1" | cut -c1-64; }
cexists() { docker exec ttbuild test -e "$1"; }
need() {  # label actual expected
  if [ "$2" != "$3" ]; then
    echo "FAIL: $1 is $2, expected $3" >&2
    exit 1
  fi
  echo "ok    $1 ${2:0:16}"
}
count() {  # file fixed-string -> number of matching strings (0 is not an error here)
  strings "$1" | grep -cF -- "$2" || true
}

# ---------- 0. inputs ----------
docker ps --format '{{.Names}}' | grep -qx ttbuild || { echo "FAIL: ttbuild container is not running" >&2; exit 1; }
for name in reader_decode_qwen.cpp sdpa_flash_decode_qwen.cpp apply_factory_qwen.py make_qwen_kernels.py; do
  test -s "$S/$name" || { echo "FAIL: $S/$name missing" >&2; exit 1; }
  if grep -q $'\r' "$S/$name"; then echo "FAIL: $S/$name has CRLF line endings" >&2; exit 1; fi
done
need "shipped reader_decode_qwen.cpp" "$(hsha "$S/reader_decode_qwen.cpp")" $READER_QWEN
need "shipped sdpa_flash_decode_qwen.cpp" "$(hsha "$S/sdpa_flash_decode_qwen.cpp")" $COMPUTE_QWEN

# ---------- 1. the base graft ----------
for part in _ttnncpp.so _ttnn.so attn_prep nlp_concat_heads_decode; do
  test -e "$BASE/$part" || { echo "FAIL: $BASE/$part missing" >&2; exit 1; }
done
test ! -e "$BASE/sdpa_decode" || { echo "FAIL: $BASE already carries sdpa_decode; expected K64d" >&2; exit 1; }
base_so=$(hsha "$BASE/_ttnncpp.so")
need "base graft _ttnncpp.so prefix" "${base_so:0:8}" $K64D_TTNNCPP_PREFIX
test "$(count "$BASE/_ttnncpp.so" QWEN_SDPA_TREE_SCRATCH_ROUNDS)" -ge 1 || { echo "FAIL: $BASE/_ttnncpp.so lacks the tree-scratch factory" >&2; exit 1; }

# ---------- 2. ttbuild sources the build depends on ----------
need "ttbuild prefill combined factory" "$(csha $PREFILL)" $PREFILL_COMBINED
need "ttbuild reader_decode_all.cpp" "$(csha $KD/dataflow/reader_decode_all.cpp)" $READER_ALL
need "ttbuild sdpa_flash_decode.cpp" "$(csha $KD/compute/sdpa_flash_decode.cpp)" $COMPUTE_ALL
need "ttbuild writer_decode_all.cpp" "$(csha $KD/dataflow/writer_decode_all.cpp)" $WRITER_ALL
need "ttbuild dataflow_common.hpp" "$(csha $KD/dataflow/dataflow_common.hpp)" $DATAFLOW_COMMON
need "ttbuild rt_args_common.hpp" "$(csha $KD/rt_args_common.hpp)" $RT_ARGS_COMMON
# The committed qwen kernels must regenerate from ttbuild's own originals, not just the dump.
docker cp ttbuild:$KD/dataflow/reader_decode_all.cpp "$W/reader_decode_all.cpp"
docker cp ttbuild:$KD/compute/sdpa_flash_decode.cpp "$W/sdpa_flash_decode.cpp"
python3 "$S/make_qwen_kernels.py" --reader "$W/reader_decode_all.cpp" --compute "$W/sdpa_flash_decode.cpp" --check "$S"
# attn_prep / nlp_concat_heads_decode in ttbuild must be what K64d's .so was built from.
rm -rf "$W/tb-attn_prep" "$W/tb-nlp_concat_heads_decode"
docker cp ttbuild:$OPS/attn_prep "$W/tb-attn_prep"
docker cp ttbuild:$XOPS/nlp_concat_heads_decode "$W/tb-nlp_concat_heads_decode"
diff -r "$W/tb-attn_prep" "$BASE/attn_prep" >/dev/null || { echo "FAIL: ttbuild attn_prep differs from $BASE/attn_prep" >&2; exit 1; }
diff -r "$W/tb-nlp_concat_heads_decode" "$BASE/nlp_concat_heads_decode" >/dev/null || { echo "FAIL: ttbuild nlp_concat_heads_decode differs from $BASE" >&2; exit 1; }
echo "ok    ttbuild attn_prep and nlp_concat_heads_decode equal $BASE's"

# ---------- 3. stage the factory and the two kernels (backing up anything replaced) ----------
# From here on ttbuild is modified; on any failure the EXIT trap puts it back (as step 6 does).
restored=0
built=0     # ninja has run over the qwen factory, so build_Release holds its objects
rebuilt=0   # step 6 rebuilt build_Release from the restored factory
armed=0   # set just before the first write into ttbuild; the refusals above it change nothing
restore_ttbuild() {
  if [ "$armed" = "0" ] || [ "$restored" = "1" ] || [ "${K64E_KEEP_STAGED:-}" = "1" ]; then
    return 0
  fi
  if [ -s "$S/sdpa_decode_program_factory.cpp.$FACTORY_BASE" ]; then
    docker cp "$S/sdpa_decode_program_factory.cpp.$FACTORY_BASE" ttbuild:$F
  fi
  for kernel in dataflow/reader_decode_qwen.cpp compute/sdpa_flash_decode_qwen.cpp; do
    if [ -e "$W/backup/$(basename "$kernel").replaced" ]; then
      docker cp "$W/backup/$(basename "$kernel").replaced" "ttbuild:$KD/$kernel"
    else
      docker exec ttbuild rm -f "$KD/$kernel"
    fi
  done
  # The restored file carries the saved copy's mtime, older than the objects ninja just built
  # from the qwen factory; touch it so ninja recompiles the unity TU that includes it.
  docker exec ttbuild touch "$F"
  restored=1
  echo "ttbuild restored: decode factory $(csha $F | cut -c1-16) (touched), qwen kernel files removed or put back"
}
on_exit() {
  restore_ttbuild || echo "WARN: ttbuild restore failed; check $F" >&2
  if [ "$built" = "1" ] && [ "$rebuilt" = "0" ] && [ "${K64E_KEEP_STAGED:-}" != "1" ]; then
    echo "WARN  ttbuild build_Release still holds the [QWEN-SDPA] objects; the next ninja run recompiles the touched factory, but do not copy a .so out of build_Release before one" >&2
  fi
}
trap on_exit EXIT
cur=$(csha $F)
docker cp ttbuild:$F "$W/backup/sdpa_decode_program_factory.cpp.${cur:0:8}"
if [ "$cur" = "$FACTORY_UNPATCHED" ]; then
  echo "FAIL: ttbuild decode factory is the unpatched 05708e6d; run the K64d build (tree-scratch) first" >&2
  exit 1
elif [ "$cur" = "$FACTORY_QWEN" ]; then
  echo "note  ttbuild factory already qwen-patched (a K64E_KEEP_STAGED run); rebuilding from it"
  test -s "$S/sdpa_decode_program_factory.cpp.$FACTORY_BASE" || { echo "FAIL: no saved 3e0a69af factory to restore from" >&2; exit 1; }
  armed=1
elif [ "$cur" = "$FACTORY_BASE" ]; then
  cp -n "$W/backup/sdpa_decode_program_factory.cpp.${cur:0:8}" "$S/sdpa_decode_program_factory.cpp.$FACTORY_BASE"
  python3 "$S/apply_factory_qwen.py" "$W/backup/sdpa_decode_program_factory.cpp.${cur:0:8}" --out "$W/sdpa_decode_program_factory.cpp"
  need "patched factory" "$(hsha "$W/sdpa_decode_program_factory.cpp")" $FACTORY_QWEN
  armed=1
  docker cp "$W/sdpa_decode_program_factory.cpp" ttbuild:$F
  docker exec ttbuild touch "$F"   # newer than every object, whatever mtime docker cp gave it
else
  echo "FAIL: unexpected ttbuild decode factory $cur" >&2
  exit 1
fi
need "saved base factory" "$(hsha "$S/sdpa_decode_program_factory.cpp.$FACTORY_BASE")" $FACTORY_BASE
need "ttbuild decode factory now" "$(csha $F)" $FACTORY_QWEN
for kernel in dataflow/reader_decode_qwen.cpp compute/sdpa_flash_decode_qwen.cpp; do
  if cexists "$KD/$kernel"; then
    docker cp "ttbuild:$KD/$kernel" "$W/backup/$(basename "$kernel").replaced"
    echo "note  ttbuild already had $kernel ($(hsha "$W/backup/$(basename "$kernel").replaced" | cut -c1-16)); backed up"
  fi
  docker cp "$S/$(basename "$kernel")" "ttbuild:$KD/$kernel"
done
need "ttbuild reader_decode_qwen.cpp" "$(csha $KD/dataflow/reader_decode_qwen.cpp)" $READER_QWEN
need "ttbuild sdpa_flash_decode_qwen.cpp" "$(csha $KD/compute/sdpa_flash_decode_qwen.cpp)" $COMPUTE_QWEN

# ---------- 4. build ----------
echo "### ninja start $(date -Is)"
built=1
docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
echo "### ninja done $(date -Is)"

# ---------- 5. assemble the graft ----------
rm -rf "$G"
mkdir -p "$G"
cp -a "$BASE/." "$G/"
docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so "$G/_ttnncpp.so"
docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnn.so "$G/_ttnn.so"
docker cp ttbuild:$D "$G/sdpa_decode"   # $G/sdpa_decode does not exist yet, so this does not nest
stray=$(find "$G/sdpa_decode" -type f \( -name '*.orig*' -o -name '*.bak*' -o -name '*.rej' -o -name '*~' \) | sort)
if [ -n "$stray" ]; then
  echo "note  dropping backup files from the graft's op directory:"; echo "$stray"
  echo "$stray" | while read -r path; do rm -f "$path"; done
fi
cp "$S/sdpa_decode_program_factory.cpp.$FACTORY_BASE" "$G/sdpa_decode/device/sdpa_decode_program_factory.cpp"

# ---------- 6. restore ttbuild ----------
if [ "${K64E_KEEP_STAGED:-}" = "1" ]; then
  echo "note  K64E_KEEP_STAGED=1: ttbuild keeps the qwen factory and kernels"
else
  restore_ttbuild
  need "ttbuild decode factory restored" "$(csha $F)" $FACTORY_BASE
  echo "### ninja (restore build_Release to the 3e0a69af factory) start $(date -Is)"
  docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
  echo "### ninja (restore) done $(date -Is)"
  TB_SO=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so
  tb_flags=$(docker exec ttbuild bash -c "grep -caF -- '[QWEN-SDPA] flags=' $TB_SO || true")
  tb_scratch=$(docker exec ttbuild bash -c "grep -caF -- QWEN_SDPA_TREE_SCRATCH_ROUNDS $TB_SO || true")
  test "$tb_flags" = "0" || { echo "FAIL: ttbuild's rebuilt _ttnncpp.so still has the [QWEN-SDPA] branch; build_Release is dirty" >&2; exit 1; }
  test "$tb_scratch" -ge 1 || { echo "FAIL: ttbuild's rebuilt _ttnncpp.so lost the tree-scratch factory" >&2; exit 1; }
  rebuilt=1
  echo "ok    ttbuild build_Release rebuilt from the 3e0a69af factory (no [QWEN-SDPA] branch, tree scratch present)"
fi

# ---------- 7. verify ----------
echo "### verify $G"
so=$G/_ttnncpp.so
flags=$(count "$so" '[QWEN-SDPA] flags=')
scratch=$(count "$so" QWEN_SDPA_TREE_SCRATCH_ROUNDS)
qreader=$(count "$so" reader_decode_qwen.cpp)
combined=$(count "$so" qwen_draft_fp32_intermediates)
echo "strings: '[QWEN-SDPA] flags=' $flags  QWEN_SDPA_TREE_SCRATCH_ROUNDS $scratch  reader_decode_qwen.cpp $qreader  qwen_draft_fp32_intermediates $combined"
test "$flags" -ge 1 || { echo "FAIL: the new _ttnncpp.so lacks the [QWEN-SDPA] factory branch" >&2; exit 1; }
test "$scratch" -ge 1 || { echo "FAIL: the new _ttnncpp.so lost the tree-scratch factory" >&2; exit 1; }
test "$qreader" -ge 1 || { echo "FAIL: the new _ttnncpp.so does not name reader_decode_qwen.cpp" >&2; exit 1; }
test "$(count "$BASE/_ttnncpp.so" '[QWEN-SDPA] flags=')" -eq 0 || { echo "FAIL: the base graft already had the branch?" >&2; exit 1; }
test "$(hsha "$so")" != "$base_so" || { echo "FAIL: the build produced K64d's _ttnncpp.so unchanged" >&2; exit 1; }
for op in attn_prep nlp_concat_heads_decode; do
  diff -r "$BASE/$op" "$G/$op" >/dev/null || { echo "FAIL: $G/$op differs from $BASE/$op" >&2; exit 1; }
done
echo "ok    attn_prep and nlp_concat_heads_decode identical to $BASE"
SD=$G/sdpa_decode/device
need "graft factory (on disk, audited)" "$(hsha $SD/sdpa_decode_program_factory.cpp)" $FACTORY_BASE
need "graft writer_decode_all.cpp" "$(hsha $SD/kernels/dataflow/writer_decode_all.cpp)" $WRITER_ALL
need "graft sdpa_flash_decode.cpp" "$(hsha $SD/kernels/compute/sdpa_flash_decode.cpp)" $COMPUTE_ALL
need "graft reader_decode_all.cpp" "$(hsha $SD/kernels/dataflow/reader_decode_all.cpp)" $READER_ALL
need "graft dataflow_common.hpp" "$(hsha $SD/kernels/dataflow/dataflow_common.hpp)" $DATAFLOW_COMMON
need "graft reader_decode_qwen.cpp" "$(hsha $SD/kernels/dataflow/reader_decode_qwen.cpp)" $READER_QWEN
need "graft sdpa_flash_decode_qwen.cpp" "$(hsha $SD/kernels/compute/sdpa_flash_decode_qwen.cpp)" $COMPUTE_QWEN
# Mounting the graft's op directory must change nothing in the container but add the two kernels.
if [ "${K64E_SKIP_IMAGE_COMPARE:-}" = "1" ]; then
  echo "WARN  K64E_SKIP_IMAGE_COMPARE=1: the op directory was NOT compared against $IMAGE"
else
  docker image inspect "$IMAGE" >/dev/null || { echo "FAIL: image $IMAGE not present; set K64E_IMAGE or K64E_SKIP_IMAGE_COMPARE=1" >&2; exit 1; }
  cid=$(docker create --network none --entrypoint true "$IMAGE")
  docker cp "$cid:$D" "$W/image-sdpa_decode"
  docker rm "$cid" >/dev/null
  differences=$(diff -rq "$W/image-sdpa_decode" "$G/sdpa_decode" || true)
  expected=$(printf 'Only in %s: %s\nOnly in %s: %s' \
    "$G/sdpa_decode/device/kernels/compute" sdpa_flash_decode_qwen.cpp \
    "$G/sdpa_decode/device/kernels/dataflow" reader_decode_qwen.cpp)
  if [ "$(echo "$differences" | sort)" != "$(echo "$expected" | sort)" ]; then
    echo "FAIL: the graft's op directory differs from the image's by more than the two qwen kernels:" >&2
    echo "$differences" >&2
    exit 1
  fi
  echo "ok    op directory = image ${IMAGE:7:12}'s + reader_decode_qwen.cpp + sdpa_flash_decode_qwen.cpp"
fi
(cd "$G" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum) > "$G/MANIFEST.sha256"
rm -rf "$GRAFT"
mv "$G" "$GRAFT"
G=$GRAFT
echo "### summary"
if [ "${K64E_KEEP_STAGED:-}" = "1" ]; then
  echo "ttbuild: KEPT STAGED - sources and build_Release carry the [QWEN-SDPA] factory"
else
  echo "ttbuild: sources restored (factory 3e0a69af, touched) and build_Release rebuilt from them"
fi
sha256sum "$G/_ttnncpp.so" "$G/_ttnn.so" "$BASE/_ttnncpp.so" "$BASE/_ttnn.so"
echo "graft files: $(wc -l < "$G/MANIFEST.sha256") (manifest $G/MANIFEST.sha256)"
echo "K64E_TTNNCPP_SHA256=$(hsha "$G/_ttnncpp.so")"
echo "### build_k64e done $(date -Is)"
