#!/usr/bin/env bash
# Build ~/opgraft-K64i on the rig: ~/opgraft-K64g (the served graft) plus the [QWEN-SDPA] stage-4 decode
# factory (K1: 0x4 head-sliced Q, 0x8 the share leader's K/V read-ahead) and its two kernels
# (k1-sdpa-head-slice-design.md section 8; optimisation/ttnn-op/sdpa_decode_slice).
#
#   K64i = K64g's contents (attn_prep, nlp_concat_heads_decode, sdpa_decode, sdpa, ...)
#        + _ttnncpp.so / _ttnn.so rebuilt in ttbuild over BOTH factories: the decode factory at stage 4
#          (apply_factory_slice.py: apply_factory_qwen --stage 3, then F13-F18; STAGE4_FACTORY) and the prefill
#          factory exactly as K64g built it (apply_factory_pf.py, PF_FACTORY), so [QWEN-SDPA-PF] survives
#        + sdpa_decode/device/kernels/dataflow/reader_decode_qwen_slice.cpp and writer_decode_qwen_slice.cpp.
#          Everything else in sdpa_decode/ is K64g's (the image's directory plus the two stage-3 qwen kernels,
#          its factory .cpp the audited 3e0a69af: the compiled factory lives in the .so). The compute kernel is
#          reused unchanged (8776fcc7).
#
# A graft .so replaces the WHOLE binary (the K64c lesson), so step 7 checks that every distinct QWEN_ / [QWEN-
# string of K64g's _ttnncpp.so AND of each image's is still present, plus the stage-4 literals.
#
# Layout (on the rig, LF endings): this directory and its two siblings shipped side by side,
#   ~/kwork64/k64i/sdpa_decode_slice/   (this script; the slice kernels, apply_factory_slice.py, make_slice_kernels.py)
#   ~/kwork64/k64i/sdpa_decode_qwen/    (apply_factory_qwen.py, make_qwen_kernels.py, stage3/)
#   ~/kwork64/k64i/sdpa_prefill_chain/  (apply_factory_pf.py, make_pf_reader.py, reader_interleaved_qwen_chain.cpp)
# and the saved base factories where the K64f / K64g builds left them: sdpa_decode_program_factory.cpp.<3e0a69af>
# in K64I_DECODE_BASE_DIR (~/kwork64/k64f) and sdpa_program_factory.cpp.<fd8c0676> in K64I_PF_BASE_DIR
# (~/kwork64/k64g). Either is also saved from ttbuild when ttbuild holds the base.
#
# Usage:  bash ~/kwork64/k64i/sdpa_decode_slice/build_k64i.sh
# Env: K64I_BASE_GRAFT (~/opgraft-K64g), K64I_BASE_SHA256 (K64g's _ttnncpp.so, 134bc834...; empty skips),
# K64I_GRAFT (~/opgraft-K64i), K64I_IMAGES (space-separated images whose sdpa_decode/ and sdpa/ the graft's must
# equal plus the qwen kernels, and whose _ttnncpp.so QWEN strings must survive; default: P3 70c27e75 (the
# v194-v199 gate image and run_card_b.sh's), P2 463964d5 (v191-v193), e41ef884 (lever_n_m3native_run_arm.sh's
# default); a subset must keep P3; every image is checked present at step 0, before the build),
# K64I_SKIP_IMAGE_COMPARE=1 (last resort), K64I_KEEP_STAGED=1 (leave ttbuild on the qwen factories and kernels; a
# rerun then patches from the saved bases).
#
# ttbuild's sources are left as they were found (decode factory 3e0a69af, prefill factory fd8c0676, no qwen
# kernel files) unless K64I_KEEP_STAGED=1; everything replaced is saved under $S/work-<stamp>/backup.
# build_Release is rebuilt from the restored factories (step 6) and checked to have lost both qwen branches:
# docker cp keeps the saved files' OLD mtimes, so without a touch and a rebuild ninja would call the unity TUs
# up to date. On a failure the EXIT trap restores and touches the sources but does not rebuild; it says so.
set -euo pipefail
# Byte-order collation for every sort/comm (the K64g lesson: comm under one locale, sort under another).
export LC_ALL=C

S=$(cd "$(dirname "$0")" && pwd)
DS=$(cd "$S/../sdpa_decode_qwen" && pwd)
PS=$(cd "$S/../sdpa_prefill_chain" && pwd)
KS=$DS/stage3
DBASE=${K64I_DECODE_BASE_DIR:-$HOME/kwork64/k64f}
PBASE=${K64I_PF_BASE_DIR:-$HOME/kwork64/k64g}
BASE=${K64I_BASE_GRAFT:-$HOME/opgraft-K64g}
GRAFT=${K64I_GRAFT:-$HOME/opgraft-K64i}
G=$GRAFT.partial
GATE_IMAGE=sha256:70c27e7539db757e3b166f0d14ccdc32edf86532017b6f2f50310fcaad855c3f   # P3: gate v194-v199, card B
IMAGES=${K64I_IMAGES:-$GATE_IMAGE sha256:463964d577321f2cebd5d29422edcaf848ade3ba4c5260c890497b6b2accfe98 sha256:e41ef884f4c8e07ce6632511fac19364af35dc73d601f05f5191784a64a3a768}
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
XOPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer
D=$OPS/sdpa_decode
F=$D/device/sdpa_decode_program_factory.cpp
KD=$D/device/kernels
P=$OPS/sdpa
PF=$P/device/sdpa_program_factory.cpp
PK=$P/device/kernels

# ---- recorded shas (test_sdpa_decode_slice_sources.py keeps these equal to the Python) ----
K64G_TTNNCPP=134bc8347d6533b3c0a5d2a1efaf14261e7f554f90579d179b1f1b88e20944f4
BASE_SHA=${K64I_BASE_SHA256-$K64G_TTNNCPP}
FACTORY_BASE=3e0a69af9563ae8899db1363286d6dc6cdc1744e0154887dc91a630b16084a4a
FACTORY_UNPATCHED=05708e6d9ddeddfdf13303d8f8fa391941d73b742ea3a380beeb0883ce8d4792
FACTORY_QWEN_STAGE1=1b54abd3fe466a058046939e2ec0366505f80fde631dba387c591985fa3da301
FACTORY_QWEN_STAGE3=06167779a979ba1f35c78d1c002f9956215ca52a4ccd4531e0f5f5fe4e44191d
FACTORY_QWEN_STAGE4=1634369677ae0247a387abb5610b8ee19f2b7a87d476534dd1e05bf6714529e3
PF_BASE=fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783
PF_FACTORY=bfab8558d889ad215f0e9ee732c75a4142be1a1e7f37810f4ca8c5e3c73bdf65
PF_READER_BASE=f97f5490cf476db92d575de33c85f8707f96d7474896ee086ee864c23a3efa27
PF_READER=eecc1166a209e61dc8498149b5a4338cc278d7106f4ca68d6434f620e942f8d8
READER_ALL=49a05926b437e2ca90d7e01c60e85a6b11f333375a6159fa02c9ff6f78af764e
COMPUTE_ALL=d24769bdcbb8635f83f5f91a301fe0d89298d38263d4493a39c6d2decb57867f
WRITER_ALL=734c90c01c7a7174497133fae9df80110ead55275955faeb566d345bdccb60b8
DATAFLOW_COMMON=e4623a2254559eaec4450ebfab0f9c5732e02acfe4d8126bd5eeb7efe0fdc608
RT_ARGS_COMMON=1b52c60d78ada6f08effd326c2ed2407b3a74cf0db2353fadbe51b088610aec8
READER_QWEN=280a847fae833891dffff1057d67b999288a183386cd058e3e1614755ce3499b
COMPUTE_QWEN=8776fcc7420c6f27a9c7ae06c54c391225a00ce78322c5397970d74a5063ca8a
READER_SLICE=0f5a019ccc06ca603bb4ed44c77cc66f9cb3e5bd35193810eb127f13c9f5631f
WRITER_SLICE=ac6cf815c34df85a9d39593d95f28eb2da0cb5b37c232633e65bf3ed116925f4
SHARE_MARKER='[QWEN-SDPA] KV-share twin bands'
STAGE1_REFUSAL='[QWEN-SDPA] KV share is not in this build'
SLICE_MARKER='[QWEN-SDPA] q-slice rows_per_kv='
STAGE4_MARKERS=("$SLICE_MARKER" '[QWEN-SDPA] KV read-ahead needs KV share' '[QWEN-SDPA] q-slice saves no tile'
                '[QWEN-SDPA] q-slice does not cover KV head' reader_decode_qwen_slice.cpp writer_decode_qwen_slice.cpp)
PF_MARKERS=('[QWEN-SDPA-PF] flags=' '[QWEN-SDPA-PF] kv_chain outside its qualified envelope' 'reader_interleaved_qwen_chain.cpp')

W=$S/work-$(date +%Y%m%dT%H%M%S)
mkdir -p "$W/backup"
echo "### build_k64i $(date -Is) work=$W base=$BASE"

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
qwen_strings() {  # every distinct QWEN_ / [QWEN- string of a binary, sorted
  strings "$1" | grep -E 'QWEN_|\[QWEN-' | sort -u || true
}

# ---------- 0. inputs ----------
docker ps --format '{{.Names}}' | grep -qx ttbuild || { echo "FAIL: ttbuild container is not running" >&2; exit 1; }
for path in "$S/reader_decode_qwen_slice.cpp" "$S/writer_decode_qwen_slice.cpp" "$S/apply_factory_slice.py" \
            "$S/make_slice_kernels.py" "$DS/apply_factory_qwen.py" "$DS/make_qwen_kernels.py" "$KS/reader_decode_qwen.cpp" \
            "$KS/sdpa_flash_decode_qwen.cpp" "$PS/apply_factory_pf.py" "$PS/make_pf_reader.py" \
            "$PS/reader_interleaved_qwen_chain.cpp"; do
  test -s "$path" || { echo "FAIL: $path missing" >&2; exit 1; }
  if grep -q $'\r' "$path"; then echo "FAIL: $path has CRLF line endings" >&2; exit 1; fi
done
need "shipped reader_decode_qwen_slice.cpp" "$(hsha "$S/reader_decode_qwen_slice.cpp")" $READER_SLICE
need "shipped writer_decode_qwen_slice.cpp" "$(hsha "$S/writer_decode_qwen_slice.cpp")" $WRITER_SLICE
need "shipped stage-3 reader" "$(hsha "$KS/reader_decode_qwen.cpp")" $READER_QWEN
need "shipped qwen compute" "$(hsha "$KS/sdpa_flash_decode_qwen.cpp")" $COMPUTE_QWEN
need "shipped reader_interleaved_qwen_chain.cpp" "$(hsha "$PS/reader_interleaved_qwen_chain.cpp")" $PF_READER
if [ "${K64I_SKIP_IMAGE_COMPARE:-}" != "1" ]; then
  case " $IMAGES " in
    *" $GATE_IMAGE "*) ;;
    *) echo "FAIL: K64I_IMAGES must include the gate / card-B image $GATE_IMAGE" >&2; exit 1 ;;
  esac
  for image in $IMAGES; do
    docker image inspect "$image" >/dev/null 2>&1 \
      || { echo "FAIL: image $image is not present; set K64I_IMAGES to the present images (keeping $GATE_IMAGE)" >&2; exit 1; }
  done
  echo "ok    images present: $(for image in $IMAGES; do printf '%s ' "${image:7:12}"; done)"
fi

# ---------- 1. the base graft (K64g: it carries sdpa/, the inverse of build_k64g.sh's step-1 check) ----------
for part in _ttnncpp.so _ttnn.so attn_prep nlp_concat_heads_decode sdpa_decode sdpa MANIFEST.sha256; do
  test -e "$BASE/$part" || { echo "FAIL: $BASE/$part missing (the base is K64g, which carries sdpa/)" >&2; exit 1; }
done
(cd "$BASE" && sha256sum -c --quiet MANIFEST.sha256) || { echo "FAIL: $BASE/MANIFEST.sha256 does not verify" >&2; exit 1; }
echo "ok    $BASE/MANIFEST.sha256 verifies"
base_so=$(hsha "$BASE/_ttnncpp.so")
if [ -n "$BASE_SHA" ]; then
  need "base graft _ttnncpp.so (K64g)" "$base_so" "$BASE_SHA"
fi
test "$(count "$BASE/_ttnncpp.so" QWEN_SDPA_TREE_SCRATCH_ROUNDS)" -ge 1 || { echo "FAIL: $BASE/_ttnncpp.so lacks the tree-scratch factory" >&2; exit 1; }
test "$(count "$BASE/_ttnncpp.so" "$SHARE_MARKER")" -ge 1 || { echo "FAIL: $BASE/_ttnncpp.so lacks '$SHARE_MARKER' (not K64f/K64g)" >&2; exit 1; }
test "$(count "$BASE/_ttnncpp.so" '[QWEN-SDPA-PF] flags=')" -ge 1 || { echo "FAIL: $BASE/_ttnncpp.so lacks [QWEN-SDPA-PF] (not K64g)" >&2; exit 1; }
test "$(count "$BASE/_ttnncpp.so" "$SLICE_MARKER")" -eq 0 || { echo "FAIL: $BASE/_ttnncpp.so already has the stage-4 factory" >&2; exit 1; }
for pair in "dataflow/reader_decode_qwen.cpp:$READER_QWEN" "compute/sdpa_flash_decode_qwen.cpp:$COMPUTE_QWEN"; do
  need "base graft ${pair%%:*}" "$(hsha "$BASE/sdpa_decode/device/kernels/${pair%%:*}")" "${pair#*:}"
done
for name in reader_decode_qwen_slice.cpp writer_decode_qwen_slice.cpp; do
  test ! -e "$BASE/sdpa_decode/device/kernels/dataflow/$name" || { echo "FAIL: $BASE already carries $name" >&2; exit 1; }
done
need "base graft reader_interleaved_qwen_chain.cpp" "$(hsha "$BASE/sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp")" $PF_READER
qwen_strings "$BASE/_ttnncpp.so" > "$W/base-qwen-strings.txt"
echo "ok    base graft ${base_so:0:16}: $(wc -l < "$W/base-qwen-strings.txt") distinct QWEN strings recorded"

# ---------- 2. ttbuild sources the build depends on ----------
dcur=$(csha $F)
case "$dcur" in
  "$FACTORY_BASE") ;;
  "$FACTORY_QWEN_STAGE1"|"$FACTORY_QWEN_STAGE3"|"$FACTORY_QWEN_STAGE4")
    echo "note  ttbuild decode factory is a qwen factory (${dcur:0:8}, a *_KEEP_STAGED run); step 3 patches from the saved 3e0a69af" ;;
  "$FACTORY_UNPATCHED") echo "FAIL: ttbuild decode factory is the unpatched 05708e6d; run the K64d build (tree-scratch) first" >&2; exit 1 ;;
  *) echo "FAIL: unexpected ttbuild decode factory $dcur" >&2; exit 1 ;;
esac
pcur=$(csha $PF)
case "$pcur" in
  "$PF_BASE") ;;
  "$PF_FACTORY") echo "note  ttbuild prefill factory is [QWEN-SDPA-PF]-patched (a *_KEEP_STAGED run); step 3 patches from the saved fd8c0676" ;;
  *) echo "FAIL: unexpected ttbuild prefill factory $pcur" >&2; exit 1 ;;
esac
need "ttbuild reader_decode_all.cpp" "$(csha $KD/dataflow/reader_decode_all.cpp)" $READER_ALL
need "ttbuild sdpa_flash_decode.cpp" "$(csha $KD/compute/sdpa_flash_decode.cpp)" $COMPUTE_ALL
need "ttbuild writer_decode_all.cpp" "$(csha $KD/dataflow/writer_decode_all.cpp)" $WRITER_ALL
need "ttbuild decode dataflow_common.hpp" "$(csha $KD/dataflow/dataflow_common.hpp)" $DATAFLOW_COMMON
need "ttbuild rt_args_common.hpp" "$(csha $KD/rt_args_common.hpp)" $RT_ARGS_COMMON
need "ttbuild reader_interleaved.cpp" "$(csha $PK/dataflow/reader_interleaved.cpp)" $PF_READER_BASE
# The committed kernels must regenerate from ttbuild's own originals, not just the dumps.
docker cp ttbuild:$KD/dataflow/reader_decode_all.cpp "$W/reader_decode_all.cpp"
docker cp ttbuild:$KD/compute/sdpa_flash_decode.cpp "$W/sdpa_flash_decode.cpp"
docker cp ttbuild:$KD/dataflow/writer_decode_all.cpp "$W/writer_decode_all.cpp"
docker cp ttbuild:$PK/dataflow/reader_interleaved.cpp "$W/reader_interleaved.cpp"
python3 "$DS/make_qwen_kernels.py" --stage 3 --reader "$W/reader_decode_all.cpp" --compute "$W/sdpa_flash_decode.cpp" --check "$KS"
python3 "$S/make_slice_kernels.py" --reader "$W/reader_decode_all.cpp" --writer "$W/writer_decode_all.cpp" --check "$S"
python3 "$PS/make_pf_reader.py" --reader "$W/reader_interleaved.cpp" --check "$PS"
# attn_prep / nlp_concat_heads_decode in ttbuild must be what the base graft's .so was built from.
rm -rf "$W/tb-attn_prep" "$W/tb-nlp_concat_heads_decode"
docker cp ttbuild:$OPS/attn_prep "$W/tb-attn_prep"
docker cp ttbuild:$XOPS/nlp_concat_heads_decode "$W/tb-nlp_concat_heads_decode"
diff -r "$W/tb-attn_prep" "$BASE/attn_prep" >/dev/null || { echo "FAIL: ttbuild attn_prep differs from $BASE/attn_prep" >&2; exit 1; }
diff -r "$W/tb-nlp_concat_heads_decode" "$BASE/nlp_concat_heads_decode" >/dev/null || { echo "FAIL: ttbuild nlp_concat_heads_decode differs from $BASE" >&2; exit 1; }
echo "ok    ttbuild attn_prep and nlp_concat_heads_decode equal $BASE's"

# ---------- 3. stage both factories and the five kernels (backing up anything replaced) ----------
restored=0
built=0
rebuilt=0
armed=0
DECODE_KERNELS=(dataflow/reader_decode_qwen.cpp compute/sdpa_flash_decode_qwen.cpp dataflow/reader_decode_qwen_slice.cpp
                dataflow/writer_decode_qwen_slice.cpp)
decode_source() {  # the shipped file staged at a kernel path
  case "$1" in
    dataflow/reader_decode_qwen.cpp) echo "$KS/reader_decode_qwen.cpp" ;;
    compute/sdpa_flash_decode_qwen.cpp) echo "$KS/sdpa_flash_decode_qwen.cpp" ;;
    *) echo "$S/$(basename "$1")" ;;
  esac
}
restore_ttbuild() {
  if [ "$armed" = "0" ] || [ "$restored" = "1" ] || [ "${K64I_KEEP_STAGED:-}" = "1" ]; then
    return 0
  fi
  docker cp "$DBASE/sdpa_decode_program_factory.cpp.$FACTORY_BASE" ttbuild:$F
  docker cp "$PBASE/sdpa_program_factory.cpp.$PF_BASE" ttbuild:$PF
  for kernel in "${DECODE_KERNELS[@]}"; do
    if [ -e "$W/backup/$(basename "$kernel").replaced" ]; then
      docker cp "$W/backup/$(basename "$kernel").replaced" "ttbuild:$KD/$kernel"
    else
      docker exec ttbuild rm -f "$KD/$kernel"
    fi
  done
  if [ -e "$W/backup/reader_interleaved_qwen_chain.cpp.replaced" ]; then
    docker cp "$W/backup/reader_interleaved_qwen_chain.cpp.replaced" "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp"
  else
    docker exec ttbuild rm -f "$PK/dataflow/reader_interleaved_qwen_chain.cpp"
  fi
  # The restored files carry the saved copies' mtimes, older than the objects ninja just built from the
  # qwen factories; touch them so ninja recompiles the unity TUs that include them.
  docker exec ttbuild touch "$F" "$PF"
  restored=1
  echo "ttbuild restored: decode factory $(csha $F | cut -c1-16), prefill factory $(csha $PF | cut -c1-16) (touched), qwen kernel files removed or put back"
}
on_exit() {
  restore_ttbuild || echo "WARN: ttbuild restore failed; check $F and $PF" >&2
  if [ "$built" = "1" ] && [ "$rebuilt" = "0" ] && [ "${K64I_KEEP_STAGED:-}" != "1" ]; then
    echo "WARN  ttbuild build_Release still holds the qwen objects; the next ninja run recompiles the touched factories, but do not copy a .so out of build_Release before one" >&2
  fi
}
trap on_exit EXIT

# 3a. the saved bases (from ttbuild when it holds them), then the decode factory at stage 4
docker cp ttbuild:$F "$W/backup/sdpa_decode_program_factory.cpp.${dcur:0:8}"
docker cp ttbuild:$PF "$W/backup/sdpa_program_factory.cpp.${pcur:0:8}"
mkdir -p "$DBASE" "$PBASE"
if [ "$dcur" = "$FACTORY_BASE" ]; then
  cp -n "$W/backup/sdpa_decode_program_factory.cpp.${dcur:0:8}" "$DBASE/sdpa_decode_program_factory.cpp.$FACTORY_BASE" || true
fi
if [ "$pcur" = "$PF_BASE" ]; then
  cp -n "$W/backup/sdpa_program_factory.cpp.${pcur:0:8}" "$PBASE/sdpa_program_factory.cpp.$PF_BASE" || true
fi
test -s "$DBASE/sdpa_decode_program_factory.cpp.$FACTORY_BASE" || { echo "FAIL: no saved 3e0a69af decode factory in $DBASE" >&2; exit 1; }
test -s "$PBASE/sdpa_program_factory.cpp.$PF_BASE" || { echo "FAIL: no saved fd8c0676 prefill factory in $PBASE" >&2; exit 1; }
need "saved decode base factory" "$(hsha "$DBASE/sdpa_decode_program_factory.cpp.$FACTORY_BASE")" $FACTORY_BASE
need "saved prefill base factory" "$(hsha "$PBASE/sdpa_program_factory.cpp.$PF_BASE")" $PF_BASE
python3 "$S/apply_factory_slice.py" "$DBASE/sdpa_decode_program_factory.cpp.$FACTORY_BASE" --out "$W/sdpa_decode_program_factory.cpp"
need "patched decode factory (stage 4)" "$(hsha "$W/sdpa_decode_program_factory.cpp")" $FACTORY_QWEN_STAGE4
# 3b. the prefill factory, exactly as build_k64g.sh step 3b
python3 "$PS/apply_factory_pf.py" "$PBASE/sdpa_program_factory.cpp.$PF_BASE" --out "$W/sdpa_program_factory.cpp"
need "patched prefill factory" "$(hsha "$W/sdpa_program_factory.cpp")" $PF_FACTORY

# 3c. stage (from here on ttbuild is modified; the EXIT trap puts it back)
armed=1
docker cp "$W/sdpa_decode_program_factory.cpp" ttbuild:$F
docker cp "$W/sdpa_program_factory.cpp" ttbuild:$PF
docker exec ttbuild touch "$F" "$PF"   # newer than every object, whatever mtime docker cp gave them
need "ttbuild decode factory now" "$(csha $F)" $FACTORY_QWEN_STAGE4
need "ttbuild prefill factory now" "$(csha $PF)" $PF_FACTORY
for kernel in "${DECODE_KERNELS[@]}"; do
  if cexists "$KD/$kernel"; then
    docker cp "ttbuild:$KD/$kernel" "$W/backup/$(basename "$kernel").replaced"
  fi
  docker cp "$(decode_source "$kernel")" "ttbuild:$KD/$kernel"
done
if cexists "$PK/dataflow/reader_interleaved_qwen_chain.cpp"; then
  docker cp "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp" "$W/backup/reader_interleaved_qwen_chain.cpp.replaced"
fi
docker cp "$PS/reader_interleaved_qwen_chain.cpp" "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp"
need "ttbuild reader_decode_qwen.cpp" "$(csha $KD/dataflow/reader_decode_qwen.cpp)" $READER_QWEN
need "ttbuild sdpa_flash_decode_qwen.cpp" "$(csha $KD/compute/sdpa_flash_decode_qwen.cpp)" $COMPUTE_QWEN
need "ttbuild reader_decode_qwen_slice.cpp" "$(csha $KD/dataflow/reader_decode_qwen_slice.cpp)" $READER_SLICE
need "ttbuild writer_decode_qwen_slice.cpp" "$(csha $KD/dataflow/writer_decode_qwen_slice.cpp)" $WRITER_SLICE
need "ttbuild reader_interleaved_qwen_chain.cpp" "$(csha $PK/dataflow/reader_interleaved_qwen_chain.cpp)" $PF_READER

# ---------- 4. build ----------
echo "### ninja start $(date -Is)"
built=1
docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
echo "### ninja done $(date -Is)"

# ---------- 5. assemble the graft: K64g plus the new .so files and the two slice kernels ----------
rm -rf "$G"
mkdir -p "$G"
cp -a "$BASE/." "$G/"
rm -f "$G/MANIFEST.sha256"
docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so "$G/_ttnncpp.so"
docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnn.so "$G/_ttnn.so"
for name in reader_decode_qwen_slice.cpp writer_decode_qwen_slice.cpp; do
  cp "$S/$name" "$G/sdpa_decode/device/kernels/dataflow/$name"
done

# ---------- 6. restore ttbuild ----------
if [ "${K64I_KEEP_STAGED:-}" = "1" ]; then
  echo "note  K64I_KEEP_STAGED=1: ttbuild keeps the qwen factories and kernels"
else
  restore_ttbuild
  need "ttbuild decode factory restored" "$(csha $F)" $FACTORY_BASE
  need "ttbuild prefill factory restored" "$(csha $PF)" $PF_BASE
  echo "### ninja (restore build_Release to the audited factories) start $(date -Is)"
  docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
  echo "### ninja (restore) done $(date -Is)"
  TB_SO=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so
  tb_pf=$(docker exec ttbuild bash -c "grep -caF -- '[QWEN-SDPA-PF]' $TB_SO || true")
  tb_flags=$(docker exec ttbuild bash -c "grep -caF -- '[QWEN-SDPA] flags=' $TB_SO || true")
  tb_scratch=$(docker exec ttbuild bash -c "grep -caF -- QWEN_SDPA_TREE_SCRATCH_ROUNDS $TB_SO || true")
  test "$tb_pf" = "0" || { echo "FAIL: ttbuild's rebuilt _ttnncpp.so still has the [QWEN-SDPA-PF] branch; build_Release is dirty" >&2; exit 1; }
  test "$tb_flags" = "0" || { echo "FAIL: ttbuild's rebuilt _ttnncpp.so still has the [QWEN-SDPA] branch; build_Release is dirty" >&2; exit 1; }
  test "$tb_scratch" -ge 1 || { echo "FAIL: ttbuild's rebuilt _ttnncpp.so lost the tree-scratch factory" >&2; exit 1; }
  rebuilt=1
  echo "ok    ttbuild build_Release rebuilt from the audited factories (no qwen branch, tree scratch present)"
fi

# ---------- 7. verify ----------
echo "### verify $G"
so=$G/_ttnncpp.so
test "$(hsha "$so")" != "$base_so" || { echo "FAIL: the build produced the base graft's _ttnncpp.so unchanged" >&2; exit 1; }
qwen_strings "$so" > "$W/graft-qwen-strings.txt"
lost=$(comm -23 "$W/base-qwen-strings.txt" "$W/graft-qwen-strings.txt")
if [ -n "$lost" ]; then
  echo "FAIL: QWEN strings of $BASE/_ttnncpp.so missing from the new .so (a graft .so replaces the whole binary):" >&2
  echo "$lost" >&2
  exit 1
fi
echo "ok    every one of the base's $(wc -l < "$W/base-qwen-strings.txt") distinct QWEN strings is in the new .so ($(wc -l < "$W/graft-qwen-strings.txt") now)"
for marker in "${STAGE4_MARKERS[@]}" "${PF_MARKERS[@]}" '[QWEN-SDPA] flags=' QWEN_SDPA_TREE_SCRATCH_ROUNDS reader_decode_qwen.cpp "$SHARE_MARKER"; do
  test "$(count "$so" "$marker")" -ge 1 || { echo "FAIL: the new _ttnncpp.so lacks '$marker'" >&2; exit 1; }
  echo "ok    strings: '$marker'"
done
test "$(count "$so" "$STAGE1_REFUSAL")" -eq 0 || { echo "FAIL: the new _ttnncpp.so refuses KV share (the stage-1 decode factory was linked)" >&2; exit 1; }
combined=$(count "$so" qwen_draft_fp32_intermediates)
test "$combined" = "$(count "$BASE/_ttnncpp.so" qwen_draft_fp32_intermediates)" \
  || { echo "FAIL: the combined prefill factory's qwen_draft_fp32_intermediates count moved from $BASE's" >&2; exit 1; }
echo "ok    qwen_draft_fp32_intermediates count $combined, as $BASE"
for op in attn_prep nlp_concat_heads_decode sdpa; do
  diff -r "$BASE/$op" "$G/$op" >/dev/null || { echo "FAIL: $G/$op differs from $BASE/$op" >&2; exit 1; }
done
echo "ok    attn_prep, nlp_concat_heads_decode and sdpa identical to $BASE"
differences=$(diff -rq "$BASE/sdpa_decode" "$G/sdpa_decode" || true)
expected=$(printf 'Only in %s: %s\nOnly in %s: %s' "$G/sdpa_decode/device/kernels/dataflow" reader_decode_qwen_slice.cpp \
  "$G/sdpa_decode/device/kernels/dataflow" writer_decode_qwen_slice.cpp)
if [ "$(echo "$differences" | sort)" != "$(echo "$expected" | sort)" ]; then
  echo "FAIL: the graft's sdpa_decode differs from $BASE's by more than the two slice kernels:" >&2
  echo "$differences" >&2
  exit 1
fi
echo "ok    sdpa_decode = $BASE's + reader_decode_qwen_slice.cpp + writer_decode_qwen_slice.cpp"
GK=$G/sdpa_decode/device/kernels
need "graft sdpa_decode factory (on disk, audited)" "$(hsha $G/sdpa_decode/device/sdpa_decode_program_factory.cpp)" $FACTORY_BASE
need "graft reader_decode_qwen.cpp" "$(hsha $GK/dataflow/reader_decode_qwen.cpp)" $READER_QWEN
need "graft sdpa_flash_decode_qwen.cpp" "$(hsha $GK/compute/sdpa_flash_decode_qwen.cpp)" $COMPUTE_QWEN
need "graft reader_decode_qwen_slice.cpp" "$(hsha $GK/dataflow/reader_decode_qwen_slice.cpp)" $READER_SLICE
need "graft writer_decode_qwen_slice.cpp" "$(hsha $GK/dataflow/writer_decode_qwen_slice.cpp)" $WRITER_SLICE
need "graft writer_decode_all.cpp" "$(hsha $GK/dataflow/writer_decode_all.cpp)" $WRITER_ALL
# Mounting the graft's two op directories must change nothing in the container but add the qwen kernels, in
# every image it will be mounted into (the gate's and card B's, the previous gate image, the arm's default).
if [ "${K64I_SKIP_IMAGE_COMPARE:-}" = "1" ]; then
  echo "WARN  K64I_SKIP_IMAGE_COMPARE=1: sdpa_decode/, sdpa/ and the image binaries were NOT compared against $IMAGES"
else
  for image in $IMAGES; do
    docker image inspect "$image" >/dev/null || { echo "FAIL: image $image not present; set K64I_IMAGES or K64I_SKIP_IMAGE_COMPARE=1" >&2; exit 1; }
    cid=$(docker create --network none --entrypoint true "$image")
    rm -rf "$W/image-sdpa" "$W/image-sdpa_decode" "$W/image-ttnncpp.so"
    docker cp "$cid:$P" "$W/image-sdpa"
    docker cp "$cid:$D" "$W/image-sdpa_decode"
    docker cp -L "$cid:/opt/tt-metal/build_Release/lib/_ttnncpp.so" "$W/image-ttnncpp.so"
    docker rm "$cid" >/dev/null
    qwen_strings "$W/image-ttnncpp.so" > "$W/image-qwen-strings.txt"
    image_lost=$(comm -23 "$W/image-qwen-strings.txt" "$W/graft-qwen-strings.txt")
    if [ -n "$image_lost" ]; then
      echo "FAIL: QWEN strings of image ${image:7:12}'s _ttnncpp.so missing from the new .so:" >&2
      echo "$image_lost" >&2
      exit 1
    fi
    echo "ok    every one of image ${image:7:12}'s $(wc -l < "$W/image-qwen-strings.txt") distinct QWEN strings is in the new .so"
    rm -f "$W/image-ttnncpp.so"
    differences=$(diff -rq "$W/image-sdpa_decode" "$G/sdpa_decode" || true)
    expected=$(printf 'Only in %s: %s\nOnly in %s: %s\nOnly in %s: %s\nOnly in %s: %s' \
      "$G/sdpa_decode/device/kernels/compute" sdpa_flash_decode_qwen.cpp \
      "$G/sdpa_decode/device/kernels/dataflow" reader_decode_qwen.cpp \
      "$G/sdpa_decode/device/kernels/dataflow" reader_decode_qwen_slice.cpp \
      "$G/sdpa_decode/device/kernels/dataflow" writer_decode_qwen_slice.cpp)
    if [ "$(echo "$differences" | sort)" != "$(echo "$expected" | sort)" ]; then
      echo "FAIL: the graft's sdpa_decode differs from image ${image:7:12}'s by more than the four qwen kernels:" >&2
      echo "$differences" >&2
      exit 1
    fi
    echo "ok    sdpa_decode = image ${image:7:12}'s + the four qwen kernels"
    differences=$(diff -rq "$W/image-sdpa" "$G/sdpa" || true)
    expected=$(printf 'Only in %s: %s' "$G/sdpa/device/kernels/dataflow" reader_interleaved_qwen_chain.cpp)
    if [ "$differences" != "$expected" ]; then
      echo "FAIL: the graft's sdpa directory differs from image ${image:7:12}'s by more than the chain reader:" >&2
      echo "$differences" >&2
      exit 1
    fi
    echo "ok    sdpa = image ${image:7:12}'s + reader_interleaved_qwen_chain.cpp"
  done
fi
(cd "$G" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum) > "$G/MANIFEST.sha256"
rm -rf "$GRAFT"
mv "$G" "$GRAFT"
G=$GRAFT
echo "### summary"
if [ "${K64I_KEEP_STAGED:-}" = "1" ]; then
  echo "ttbuild: KEPT STAGED - sources and build_Release carry the qwen factories and kernels"
else
  echo "ttbuild: sources restored (decode 3e0a69af, prefill fd8c0676, touched) and build_Release rebuilt from them"
fi
sha256sum "$G/_ttnncpp.so" "$G/_ttnn.so" "$BASE/_ttnncpp.so" "$BASE/_ttnn.so"
echo "graft files: $(wc -l < "$G/MANIFEST.sha256") (manifest $G/MANIFEST.sha256)"
echo "K64I_TTNNCPP_SHA256=$(hsha "$G/_ttnncpp.so")"
echo "### build_k64i done $(date -Is)"
