#!/usr/bin/env bash
# Build ~/opgraft-K64g on the rig: ~/opgraft-K64f plus the [QWEN-SDPA-PF] prefill K/V chain
# (sdpa-prefill-share-spec.md 5.2; optimisation/ttnn-op/sdpa_prefill_chain).
#
#   K64g = K64f's contents (attn_prep, nlp_concat_heads_decode, sdpa_decode, ...)
#        + _ttnncpp.so / _ttnn.so rebuilt in ttbuild over BOTH factories: the decode factory
#          exactly as K64f built it (apply_factory_qwen.py --stage 3, 06167779, with the stage-3
#          decode kernels) and the prefill factory with F0-F7 (apply_factory_pf.py, PF_FACTORY)
#        + sdpa/  a copy of ttbuild's whole prefill op directory, which is the image's directory plus
#                 reader_interleaved_qwen_chain.cpp (R1-R9). Its factory .cpp is put back to the
#                 audited fd8c0676 bytes: the compiled factory lives in the .so.
#
# A graft .so replaces the WHOLE binary (the K64c lesson: K64c silently lost the sdpa tree-scratch
# patch), so step 7 checks that every distinct QWEN_ / [QWEN- string of K64f's _ttnncpp.so is still
# present, plus the three new [QWEN-SDPA-PF] literals.
#
# The arm (lever_n_m3native_run_arm.sh) does NOT yet mount $KOPGRAFT64/sdpa (spec 5.2 step 8, a
# scripts/ci change: check both image copy lists first). Card M reaches the graft through
# ../sdpa_prefill_bench/run_m1.sh KOPGRAFT_PF=~/opgraft-K64g and run_card_m_pf.sh here. The factory
# log line '[QWEN-SDPA-PF] flags=' is the proof the chain ran, not the mount.
#
# Usage (on the rig, after shipping this directory to $K64G_SRC, LF endings):
#   bash ~/kwork64/k64g/build_k64g.sh
# Env: K64G_SRC (~/kwork64/k64g: this directory), K64G_DECODE_SRC (~/kwork64/k64f: the shipped
# sdpa_decode_qwen directory K64f was built from, stage3/ included, holding the saved 3e0a69af decode
# factory), K64G_BASE_GRAFT (~/opgraft-K64f), K64G_GRAFT (~/opgraft-K64g), K64G_DECODE_STAGE (3; 1
# rebases onto K64e: set K64G_BASE_GRAFT=~/opgraft-K64e too), K64G_IMAGES (space-separated images whose
# sdpa/ the graft's must equal plus the chain reader; default: A'' eceb2daa, the v128+ model-gate and
# card-M image; A' 1b9b6445, the K0 image; e41ef884, lever_n_m3native_run_arm.sh's own default; a
# subset must keep eceb2daa; every image is checked present at step 0, before the build, and at step 7
# each image's own _ttnncpp.so QWEN strings must also survive in the graft .so), K64G_SKIP_IMAGE_COMPARE=1
# (skips every image check; last resort), K64G_KEEP_STAGED=1 (leave ttbuild on the qwen factories; a
# rerun then accepts ttbuild's patched prefill factory at step 2 and re-patches from the saved fd8c0676).
#
# ttbuild's sources are left as they were found (decode factory 3e0a69af, prefill factory fd8c0676,
# no qwen kernel files) unless K64G_KEEP_STAGED=1; everything replaced is saved under
# $K64G_SRC/work-<stamp>/backup. build_Release is rebuilt from the restored factories (step 6) and
# checked to have lost both qwen branches: docker cp keeps the saved files' OLD mtimes, so without a
# touch and a rebuild ninja would call the unity TUs up to date. On a failure the EXIT trap restores
# and touches the sources but does not rebuild; it says so.
set -euo pipefail
# Byte-order collation for every sort/comm: the rig run died at step 7 with "comm: file 1 is not in
# sorted order" (qwen_strings sorted under one locale, comm checking under another) and set -e.
export LC_ALL=C

S=${K64G_SRC:-$HOME/kwork64/k64g}
DS=${K64G_DECODE_SRC:-$HOME/kwork64/k64f}
STAGE=${K64G_DECODE_STAGE:-3}
case "$STAGE" in
  3) KS=$DS/stage3; BASE=${K64G_BASE_GRAFT:-$HOME/opgraft-K64f} ;;
  1) KS=$DS; BASE=${K64G_BASE_GRAFT:-$HOME/opgraft-K64e} ;;
  *) echo "FAIL: K64G_DECODE_STAGE must be 3 or 1" >&2; exit 1 ;;
esac
GRAFT=${K64G_GRAFT:-$HOME/opgraft-K64g}
G=$GRAFT.partial
IMAGES=${K64G_IMAGES:-sha256:eceb2daa744c3345368a638488a804f0ffe8b76f6b0680945a47bb527bdb55a9 sha256:1b9b644549d4409c4fc80e2f92c183e665e7e693c37cccc95b4bb760a6d7d537 sha256:e41ef884f4c8e07ce6632511fac19364af35dc73d601f05f5191784a64a3a768}
GATE_IMAGE=sha256:eceb2daa744c3345368a638488a804f0ffe8b76f6b0680945a47bb527bdb55a9   # A'': model gate v128+ and card M
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
XOPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer
D=$OPS/sdpa_decode
F=$D/device/sdpa_decode_program_factory.cpp
KD=$D/device/kernels
P=$OPS/sdpa
PF=$P/device/sdpa_program_factory.cpp
PK=$P/device/kernels

# ---- recorded shas (test_sdpa_prefill_chain_sources.py keeps these equal to the Python) ----
PF_BASE=fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783
PF_FACTORY=bfab8558d889ad215f0e9ee732c75a4142be1a1e7f37810f4ca8c5e3c73bdf65
PF_READER_BASE=f97f5490cf476db92d575de33c85f8707f96d7474896ee086ee864c23a3efa27
PF_READER=eecc1166a209e61dc8498149b5a4338cc278d7106f4ca68d6434f620e942f8d8
PF_COMPUTE=a3f48af8ba0fd63b136c79a54c8b6f7b4b5b8fb0d7a209bf5701f081ed7fa3e0
PF_COMPUTE_COMMON=3fb5da2440c3bf90ebceb8acd55424c7739339de4c6b02db83836e1e3414fa19
PF_DATAFLOW_COMMON=554a0b282d2a36c7b129eef04df67a5becbb14f98220246e43e47fdd56118aa6
PF_CHAIN_LINK=43a24c466c97beb5d34269630fde6a7831a23246d9df99a03f8c88265044cd9f
PF_WRITER=2a0959ff7cce507eab39f2e063f4dbc264d8b51f4db9f479b8c2419435c745c2
FACTORY_BASE=3e0a69af9563ae8899db1363286d6dc6cdc1744e0154887dc91a630b16084a4a
FACTORY_UNPATCHED=05708e6d9ddeddfdf13303d8f8fa391941d73b742ea3a380beeb0883ce8d4792
FACTORY_QWEN_STAGE1=1b54abd3fe466a058046939e2ec0366505f80fde631dba387c591985fa3da301
FACTORY_QWEN_STAGE3=06167779a979ba1f35c78d1c002f9956215ca52a4ccd4531e0f5f5fe4e44191d
READER_ALL=49a05926b437e2ca90d7e01c60e85a6b11f333375a6159fa02c9ff6f78af764e
COMPUTE_ALL=d24769bdcbb8635f83f5f91a301fe0d89298d38263d4493a39c6d2decb57867f
WRITER_ALL=734c90c01c7a7174497133fae9df80110ead55275955faeb566d345bdccb60b8
DATAFLOW_COMMON=e4623a2254559eaec4450ebfab0f9c5732e02acfe4d8126bd5eeb7efe0fdc608
RT_ARGS_COMMON=1b52c60d78ada6f08effd326c2ed2407b3a74cf0db2353fadbe51b088610aec8
READER_QWEN_STAGE1=55d8fe5e1bc87d9ada56f4a6279afa523865028c163b36803bfd07013f488702
READER_QWEN_STAGE3=280a847fae833891dffff1057d67b999288a183386cd058e3e1614755ce3499b
COMPUTE_QWEN=8776fcc7420c6f27a9c7ae06c54c391225a00ce78322c5397970d74a5063ca8a
K64F_TTNNCPP_PREFIX=d59b3c99
if [ "$STAGE" = 3 ]; then
  FACTORY_QWEN=$FACTORY_QWEN_STAGE3; READER_QWEN=$READER_QWEN_STAGE3; BASE_PREFIX=${K64G_BASE_PREFIX:-$K64F_TTNNCPP_PREFIX}
else
  FACTORY_QWEN=$FACTORY_QWEN_STAGE1; READER_QWEN=$READER_QWEN_STAGE1; BASE_PREFIX=${K64G_BASE_PREFIX:-}
fi
SHARE_MARKER='[QWEN-SDPA] KV-share twin bands'
STAGE1_REFUSAL='[QWEN-SDPA] KV share is not in this build'
PF_MARKERS=('[QWEN-SDPA-PF] flags=' '[QWEN-SDPA-PF] kv_chain outside its qualified envelope' 'reader_interleaved_qwen_chain.cpp')

W=$S/work-$(date +%Y%m%dT%H%M%S)
mkdir -p "$W/backup"
echo "### build_k64g $(date -Is) work=$W decode_stage=$STAGE base=$BASE"

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
for path in "$S/reader_interleaved_qwen_chain.cpp" "$S/apply_factory_pf.py" "$S/make_pf_reader.py" \
            "$DS/apply_factory_qwen.py" "$DS/make_qwen_kernels.py" "$KS/reader_decode_qwen.cpp" "$KS/sdpa_flash_decode_qwen.cpp"; do
  test -s "$path" || { echo "FAIL: $path missing" >&2; exit 1; }
  if grep -q $'\r' "$path"; then echo "FAIL: $path has CRLF line endings" >&2; exit 1; fi
done
need "shipped reader_interleaved_qwen_chain.cpp" "$(hsha "$S/reader_interleaved_qwen_chain.cpp")" $PF_READER
need "shipped decode reader (stage $STAGE)" "$(hsha "$KS/reader_decode_qwen.cpp")" $READER_QWEN
need "shipped decode compute" "$(hsha "$KS/sdpa_flash_decode_qwen.cpp")" $COMPUTE_QWEN
# Every image step 7 compares against must be here BEFORE the ninja build (not after it), and the
# model-gate / card-M image is never left out; K64G_IMAGES may name a subset that keeps it.
if [ "${K64G_SKIP_IMAGE_COMPARE:-}" != "1" ]; then
  case " $IMAGES " in
    *" $GATE_IMAGE "*) ;;
    *) echo "FAIL: K64G_IMAGES must include the model-gate / card-M image $GATE_IMAGE" >&2; exit 1 ;;
  esac
  for image in $IMAGES; do
    docker image inspect "$image" >/dev/null 2>&1 \
      || { echo "FAIL: image $image is not present; set K64G_IMAGES to the present images (keeping $GATE_IMAGE)" >&2; exit 1; }
  done
  echo "ok    images present: $(for image in $IMAGES; do printf '%s ' "${image:7:12}"; done)"
fi

# ---------- 1. the base graft ----------
for part in _ttnncpp.so _ttnn.so attn_prep nlp_concat_heads_decode sdpa_decode MANIFEST.sha256; do
  test -e "$BASE/$part" || { echo "FAIL: $BASE/$part missing" >&2; exit 1; }
done
test ! -e "$BASE/sdpa" || { echo "FAIL: $BASE already carries sdpa/; expected K64f (or K64e)" >&2; exit 1; }
(cd "$BASE" && sha256sum -c --quiet MANIFEST.sha256) || { echo "FAIL: $BASE/MANIFEST.sha256 does not verify" >&2; exit 1; }
echo "ok    $BASE/MANIFEST.sha256 verifies"
base_so=$(hsha "$BASE/_ttnncpp.so")
if [ -n "$BASE_PREFIX" ]; then
  need "base graft _ttnncpp.so prefix" "${base_so:0:8}" "$BASE_PREFIX"
fi
test "$(count "$BASE/_ttnncpp.so" QWEN_SDPA_TREE_SCRATCH_ROUNDS)" -ge 1 || { echo "FAIL: $BASE/_ttnncpp.so lacks the tree-scratch factory" >&2; exit 1; }
if [ "$STAGE" = 3 ]; then
  test "$(count "$BASE/_ttnncpp.so" "$SHARE_MARKER")" -ge 1 || { echo "FAIL: $BASE/_ttnncpp.so lacks '$SHARE_MARKER' (not K64f)" >&2; exit 1; }
fi
test "$(count "$BASE/_ttnncpp.so" '[QWEN-SDPA-PF]')" -eq 0 || { echo "FAIL: $BASE/_ttnncpp.so already has [QWEN-SDPA-PF]" >&2; exit 1; }
qwen_strings "$BASE/_ttnncpp.so" > "$W/base-qwen-strings.txt"
echo "ok    base graft ${base_so:0:16}: $(wc -l < "$W/base-qwen-strings.txt") distinct QWEN strings recorded"

# ---------- 2. ttbuild sources the build depends on ----------
# fd8c0676, or PF_FACTORY after a K64G_KEEP_STAGED run (step 3b then re-patches from the saved fd8c0676).
pf_now=$(csha $PF)
if [ "$pf_now" = "$PF_FACTORY" ]; then
  test -s "$S/sdpa_program_factory.cpp.$PF_BASE" \
    || { echo "FAIL: ttbuild's prefill factory is the patched ${PF_FACTORY:0:8} and no saved fd8c0676 is in $S" >&2; exit 1; }
  echo "note  ttbuild prefill factory is [QWEN-SDPA-PF]-patched ${PF_FACTORY:0:16} (a K64G_KEEP_STAGED run); step 3b re-patches from the saved fd8c0676"
else
  need "ttbuild prefill factory" "$pf_now" $PF_BASE
fi
need "ttbuild reader_interleaved.cpp" "$(csha $PK/dataflow/reader_interleaved.cpp)" $PF_READER_BASE
need "ttbuild sdpa compute sdpa.cpp" "$(csha $PK/compute/sdpa.cpp)" $PF_COMPUTE
need "ttbuild compute_common.hpp" "$(csha $PK/compute/compute_common.hpp)" $PF_COMPUTE_COMMON
need "ttbuild prefill dataflow_common.hpp" "$(csha $PK/dataflow/dataflow_common.hpp)" $PF_DATAFLOW_COMMON
need "ttbuild chain_link.hpp" "$(csha $PK/dataflow/chain_link.hpp)" $PF_CHAIN_LINK
need "ttbuild writer_interleaved.cpp" "$(csha $PK/dataflow/writer_interleaved.cpp)" $PF_WRITER
if cexists "$PK/dataflow/reader_interleaved_qwen_chain.cpp"; then
  echo "note  ttbuild already has reader_interleaved_qwen_chain.cpp (a K64G_KEEP_STAGED run?); it is backed up and replaced"
fi
need "ttbuild reader_decode_all.cpp" "$(csha $KD/dataflow/reader_decode_all.cpp)" $READER_ALL
need "ttbuild sdpa_flash_decode.cpp" "$(csha $KD/compute/sdpa_flash_decode.cpp)" $COMPUTE_ALL
need "ttbuild writer_decode_all.cpp" "$(csha $KD/dataflow/writer_decode_all.cpp)" $WRITER_ALL
need "ttbuild decode dataflow_common.hpp" "$(csha $KD/dataflow/dataflow_common.hpp)" $DATAFLOW_COMMON
need "ttbuild rt_args_common.hpp" "$(csha $KD/rt_args_common.hpp)" $RT_ARGS_COMMON
# Spec Q-6: the chain reader's bounded waits use WAYPOINT / ASSERT without an extra include (DC pulls in
# api/debug/assert.h; WAYPOINT is expected through dataflow_api.h). Evidence only: the first JIT compile
# of the chain reader (the WATCHER=1 pass) is the proof, and it fails loudly.
waypoints=$(docker exec ttbuild bash -c "f=\$(find /opt/tt-metal/tt_metal/hw/inc -path '*api/dataflow/dataflow_api.h' | head -1); \
  test -n \"\$f\" && { grep -c 'WAYPOINT(' \"\$f\"; grep -c 'waypoint.h' \"\$f\"; } | paste -sd' '" || true)
echo "note  ttbuild api/dataflow/dataflow_api.h: WAYPOINT( uses / waypoint.h includes = ${waypoints:-not found}"
# The committed kernels must regenerate from ttbuild's own originals, not just the dumps.
docker cp ttbuild:$PK/dataflow/reader_interleaved.cpp "$W/reader_interleaved.cpp"
python3 "$S/make_pf_reader.py" --reader "$W/reader_interleaved.cpp" --check "$S"
docker cp ttbuild:$KD/dataflow/reader_decode_all.cpp "$W/reader_decode_all.cpp"
docker cp ttbuild:$KD/compute/sdpa_flash_decode.cpp "$W/sdpa_flash_decode.cpp"
python3 "$DS/make_qwen_kernels.py" --stage "$STAGE" --reader "$W/reader_decode_all.cpp" --compute "$W/sdpa_flash_decode.cpp" --check "$KS"
# attn_prep / nlp_concat_heads_decode in ttbuild must be what the base graft's .so was built from.
rm -rf "$W/tb-attn_prep" "$W/tb-nlp_concat_heads_decode"
docker cp ttbuild:$OPS/attn_prep "$W/tb-attn_prep"
docker cp ttbuild:$XOPS/nlp_concat_heads_decode "$W/tb-nlp_concat_heads_decode"
diff -r "$W/tb-attn_prep" "$BASE/attn_prep" >/dev/null || { echo "FAIL: ttbuild attn_prep differs from $BASE/attn_prep" >&2; exit 1; }
diff -r "$W/tb-nlp_concat_heads_decode" "$BASE/nlp_concat_heads_decode" >/dev/null || { echo "FAIL: ttbuild nlp_concat_heads_decode differs from $BASE" >&2; exit 1; }
echo "ok    ttbuild attn_prep and nlp_concat_heads_decode equal $BASE's"

# ---------- 3. stage both factories and the three kernels (backing up anything replaced) ----------
restored=0
built=0
rebuilt=0
armed=0
DECODE_KERNELS=(dataflow/reader_decode_qwen.cpp compute/sdpa_flash_decode_qwen.cpp)
restore_ttbuild() {
  if [ "$armed" = "0" ] || [ "$restored" = "1" ] || [ "${K64G_KEEP_STAGED:-}" = "1" ]; then
    return 0
  fi
  if [ -s "$DS/sdpa_decode_program_factory.cpp.$FACTORY_BASE" ]; then
    docker cp "$DS/sdpa_decode_program_factory.cpp.$FACTORY_BASE" ttbuild:$F
  fi
  if [ -s "$S/sdpa_program_factory.cpp.$PF_BASE" ]; then
    docker cp "$S/sdpa_program_factory.cpp.$PF_BASE" ttbuild:$PF
  fi
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
  # The restored files carry the saved copies' mtimes, older than the objects ninja just built from
  # the qwen factories; touch them so ninja recompiles the unity TUs that include them.
  docker exec ttbuild touch "$F" "$PF"
  restored=1
  echo "ttbuild restored: decode factory $(csha $F | cut -c1-16), prefill factory $(csha $PF | cut -c1-16) (touched), qwen kernel files removed or put back"
}
on_exit() {
  restore_ttbuild || echo "WARN: ttbuild restore failed; check $F and $PF" >&2
  if [ "$built" = "1" ] && [ "$rebuilt" = "0" ] && [ "${K64G_KEEP_STAGED:-}" != "1" ]; then
    echo "WARN  ttbuild build_Release still holds the qwen objects; the next ninja run recompiles the touched factories, but do not copy a .so out of build_Release before one" >&2
  fi
}
trap on_exit EXIT

# 3a. the decode factory, exactly as build_k64f.sh step 3 (or build_k64e.sh for stage 1)
cur=$(csha $F)
docker cp ttbuild:$F "$W/backup/sdpa_decode_program_factory.cpp.${cur:0:8}"
if [ "$cur" = "$FACTORY_UNPATCHED" ]; then
  echo "FAIL: ttbuild decode factory is the unpatched 05708e6d; run the K64d build (tree-scratch) first" >&2
  exit 1
elif [ "$cur" = "$FACTORY_BASE" ]; then
  cp -n "$W/backup/sdpa_decode_program_factory.cpp.${cur:0:8}" "$DS/sdpa_decode_program_factory.cpp.$FACTORY_BASE" || true
elif [ "$cur" = "$FACTORY_QWEN_STAGE1" ] || [ "$cur" = "$FACTORY_QWEN_STAGE3" ]; then
  echo "note  ttbuild decode factory is a qwen factory (${cur:0:8}, a *_KEEP_STAGED run); patching from the saved 3e0a69af"
else
  echo "FAIL: unexpected ttbuild decode factory $cur" >&2
  exit 1
fi
test -s "$DS/sdpa_decode_program_factory.cpp.$FACTORY_BASE" || { echo "FAIL: no saved 3e0a69af decode factory in $DS" >&2; exit 1; }
need "saved decode base factory" "$(hsha "$DS/sdpa_decode_program_factory.cpp.$FACTORY_BASE")" $FACTORY_BASE
python3 "$DS/apply_factory_qwen.py" "$DS/sdpa_decode_program_factory.cpp.$FACTORY_BASE" --stage "$STAGE" --out "$W/sdpa_decode_program_factory.cpp"
need "patched decode factory (stage $STAGE)" "$(hsha "$W/sdpa_decode_program_factory.cpp")" $FACTORY_QWEN

# 3b. the prefill factory
pcur=$(csha $PF)
docker cp ttbuild:$PF "$W/backup/sdpa_program_factory.cpp.${pcur:0:8}"
if [ "$pcur" = "$PF_BASE" ]; then
  cp -n "$W/backup/sdpa_program_factory.cpp.${pcur:0:8}" "$S/sdpa_program_factory.cpp.$PF_BASE" || true
elif [ "$pcur" = "$PF_FACTORY" ]; then
  echo "note  ttbuild prefill factory is already [QWEN-SDPA-PF]-patched (a K64G_KEEP_STAGED run); patching from the saved fd8c0676"
else
  echo "FAIL: unexpected ttbuild prefill factory $pcur" >&2
  exit 1
fi
test -s "$S/sdpa_program_factory.cpp.$PF_BASE" || { echo "FAIL: no saved fd8c0676 prefill factory in $S" >&2; exit 1; }
need "saved prefill base factory" "$(hsha "$S/sdpa_program_factory.cpp.$PF_BASE")" $PF_BASE
python3 "$S/apply_factory_pf.py" "$S/sdpa_program_factory.cpp.$PF_BASE" --out "$W/sdpa_program_factory.cpp"
need "patched prefill factory" "$(hsha "$W/sdpa_program_factory.cpp")" $PF_FACTORY

# 3c. stage (from here on ttbuild is modified; the EXIT trap puts it back)
armed=1
docker cp "$W/sdpa_decode_program_factory.cpp" ttbuild:$F
docker cp "$W/sdpa_program_factory.cpp" ttbuild:$PF
docker exec ttbuild touch "$F" "$PF"   # newer than every object, whatever mtime docker cp gave them
need "ttbuild decode factory now" "$(csha $F)" $FACTORY_QWEN
need "ttbuild prefill factory now" "$(csha $PF)" $PF_FACTORY
for kernel in "${DECODE_KERNELS[@]}"; do
  if cexists "$KD/$kernel"; then
    docker cp "ttbuild:$KD/$kernel" "$W/backup/$(basename "$kernel").replaced"
  fi
  docker cp "$KS/$(basename "$kernel")" "ttbuild:$KD/$kernel"
done
if cexists "$PK/dataflow/reader_interleaved_qwen_chain.cpp"; then
  docker cp "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp" "$W/backup/reader_interleaved_qwen_chain.cpp.replaced"
fi
docker cp "$S/reader_interleaved_qwen_chain.cpp" "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp"
need "ttbuild reader_decode_qwen.cpp" "$(csha $KD/dataflow/reader_decode_qwen.cpp)" $READER_QWEN
need "ttbuild sdpa_flash_decode_qwen.cpp" "$(csha $KD/compute/sdpa_flash_decode_qwen.cpp)" $COMPUTE_QWEN
need "ttbuild reader_interleaved_qwen_chain.cpp" "$(csha $PK/dataflow/reader_interleaved_qwen_chain.cpp)" $PF_READER

# ---------- 4. build ----------
echo "### ninja start $(date -Is)"
built=1
docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
echo "### ninja done $(date -Is)"

# ---------- 5. assemble the graft ----------
rm -rf "$G"
mkdir -p "$G"
cp -a "$BASE/." "$G/"
rm -f "$G/MANIFEST.sha256"
docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so "$G/_ttnncpp.so"
docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnn.so "$G/_ttnn.so"
docker cp ttbuild:$P "$G/sdpa"   # $G/sdpa does not exist yet (step 1 refused a base with one), so no nesting
stray=$(find "$G/sdpa" -type f \( -name '*.orig*' -o -name '*.bak*' -o -name '*.rej' -o -name '*~' \) | sort)
if [ -n "$stray" ]; then
  echo "note  dropping backup files from the graft's sdpa directory:"; echo "$stray"
  echo "$stray" | while read -r path; do rm -f "$path"; done
fi
cp "$S/sdpa_program_factory.cpp.$PF_BASE" "$G/sdpa/device/sdpa_program_factory.cpp"

# ---------- 6. restore ttbuild ----------
if [ "${K64G_KEEP_STAGED:-}" = "1" ]; then
  echo "note  K64G_KEEP_STAGED=1: ttbuild keeps the qwen factories and kernels"
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
for marker in "${PF_MARKERS[@]}"; do
  test "$(count "$so" "$marker")" -ge 1 || { echo "FAIL: the new _ttnncpp.so lacks '$marker'" >&2; exit 1; }
  echo "ok    strings: '$marker'"
done
for marker in '[QWEN-SDPA] flags=' QWEN_SDPA_TREE_SCRATCH_ROUNDS reader_decode_qwen.cpp; do
  test "$(count "$so" "$marker")" -ge 1 || { echo "FAIL: the new _ttnncpp.so lacks '$marker'" >&2; exit 1; }
done
if [ "$STAGE" = 3 ]; then
  test "$(count "$so" "$SHARE_MARKER")" -ge 1 || { echo "FAIL: the new _ttnncpp.so lacks the stage-3 KV-share branch" >&2; exit 1; }
  test "$(count "$so" "$STAGE1_REFUSAL")" -eq 0 || { echo "FAIL: the new _ttnncpp.so refuses KV share (the stage-1 decode factory was linked)" >&2; exit 1; }
fi
combined=$(count "$so" qwen_draft_fp32_intermediates)
test "$combined" = "$(count "$BASE/_ttnncpp.so" qwen_draft_fp32_intermediates)" \
  || { echo "FAIL: the combined prefill factory's qwen_draft_fp32_intermediates count moved from $BASE's" >&2; exit 1; }
echo "ok    qwen_draft_fp32_intermediates count $combined, as $BASE"
for op in attn_prep nlp_concat_heads_decode sdpa_decode; do
  diff -r "$BASE/$op" "$G/$op" >/dev/null || { echo "FAIL: $G/$op differs from $BASE/$op" >&2; exit 1; }
done
echo "ok    attn_prep, nlp_concat_heads_decode and sdpa_decode identical to $BASE"
SP=$G/sdpa/device
need "graft prefill factory (on disk, audited)" "$(hsha $SP/sdpa_program_factory.cpp)" $PF_BASE
need "graft reader_interleaved.cpp" "$(hsha $SP/kernels/dataflow/reader_interleaved.cpp)" $PF_READER_BASE
need "graft reader_interleaved_qwen_chain.cpp" "$(hsha $SP/kernels/dataflow/reader_interleaved_qwen_chain.cpp)" $PF_READER
need "graft sdpa.cpp" "$(hsha $SP/kernels/compute/sdpa.cpp)" $PF_COMPUTE
need "graft compute_common.hpp" "$(hsha $SP/kernels/compute/compute_common.hpp)" $PF_COMPUTE_COMMON
need "graft dataflow_common.hpp" "$(hsha $SP/kernels/dataflow/dataflow_common.hpp)" $PF_DATAFLOW_COMMON
need "graft chain_link.hpp" "$(hsha $SP/kernels/dataflow/chain_link.hpp)" $PF_CHAIN_LINK
need "graft writer_interleaved.cpp" "$(hsha $SP/kernels/dataflow/writer_interleaved.cpp)" $PF_WRITER
# Mounting the graft's sdpa/ must change nothing in the container but add the chain reader, in every
# image it will be mounted into (the model gate's, the card-M qualification one, the arm default).
if [ "${K64G_SKIP_IMAGE_COMPARE:-}" = "1" ]; then
  echo "WARN  K64G_SKIP_IMAGE_COMPARE=1: the sdpa directory was NOT compared against $IMAGES"
else
  for image in $IMAGES; do
    docker image inspect "$image" >/dev/null || { echo "FAIL: image $image not present; set K64G_IMAGES or K64G_SKIP_IMAGE_COMPARE=1" >&2; exit 1; }
    cid=$(docker create --network none --entrypoint true "$image")
    rm -rf "$W/image-sdpa" "$W/image-ttnncpp.so"
    docker cp "$cid:$P" "$W/image-sdpa"
    docker cp -L "$cid:/opt/tt-metal/build_Release/lib/_ttnncpp.so" "$W/image-ttnncpp.so"
    docker rm "$cid" >/dev/null
    # The K64c lesson against the IMAGE binary too (not only K64f's): the graft .so replaces the image's
    # whole _ttnncpp.so, so every QWEN_ / [QWEN- string the image's binary carries must survive in it.
    qwen_strings "$W/image-ttnncpp.so" > "$W/image-qwen-strings.txt"
    image_lost=$(comm -23 "$W/image-qwen-strings.txt" "$W/graft-qwen-strings.txt")
    if [ -n "$image_lost" ]; then
      echo "FAIL: QWEN strings of image ${image:7:12}'s _ttnncpp.so missing from the new .so:" >&2
      echo "$image_lost" >&2
      exit 1
    fi
    echo "ok    every one of image ${image:7:12}'s $(wc -l < "$W/image-qwen-strings.txt") distinct QWEN strings is in the new .so"
    rm -f "$W/image-ttnncpp.so"
    differences=$(diff -rq "$W/image-sdpa" "$G/sdpa" || true)
    expected=$(printf 'Only in %s: %s' "$G/sdpa/device/kernels/dataflow" reader_interleaved_qwen_chain.cpp)
    if [ "$differences" != "$expected" ]; then
      echo "FAIL: the graft's sdpa directory differs from image ${image:7:12}'s by more than the chain reader:" >&2
      echo "$differences" >&2
      exit 1
    fi
    echo "ok    sdpa directory = image ${image:7:12}'s + reader_interleaved_qwen_chain.cpp"
  done
fi
(cd "$G" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum) > "$G/MANIFEST.sha256"
rm -rf "$GRAFT"
mv "$G" "$GRAFT"
G=$GRAFT
echo "### summary"
if [ "${K64G_KEEP_STAGED:-}" = "1" ]; then
  echo "ttbuild: KEPT STAGED - sources and build_Release carry the qwen factories"
else
  echo "ttbuild: sources restored (decode 3e0a69af, prefill fd8c0676, touched) and build_Release rebuilt from them"
fi
sha256sum "$G/_ttnncpp.so" "$G/_ttnn.so" "$BASE/_ttnncpp.so" "$BASE/_ttnn.so"
echo "graft files: $(wc -l < "$G/MANIFEST.sha256") (manifest $G/MANIFEST.sha256)"
echo "K64G_TTNNCPP_SHA256=$(hsha "$G/_ttnncpp.so")"
echo "### build_k64g done $(date -Is)"
