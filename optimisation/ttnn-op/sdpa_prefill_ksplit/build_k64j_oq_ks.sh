#!/usr/bin/env bash
# Build ~/opgraft-K64j-OQ-KS on the rig HOST: the K64j-OQ graft (~/opgraft-K64j-OQ: K64j plus the oneq edits, the binary the oneq images bake) plus ONE change, the K-split of the prefill SDPA
# (apply_factory_pk.py PK0-PK13: flag 0x10 of the per-call word on top of the oneq word, and its three kernels), through the ttbuild container. A NEW directory beside the existing grafts:
# nothing existing is overwritten or moved. It is the experiment's graft, never a serving binary: the K-split card job (run_card_m_ks.sh) is what mounts it. It opens no device and launches no
# device container. NOT RUN: this script has never run (no build container was available); its dry run is the CPU test's evidence, the first real run is the first compile of the C++.
#
#   K64j-OQ-KS = K64j-OQ's contents, byte for byte (attn_prep, nlp_concat_heads_decode, sdpa_decode, _ttnn.so, the sdpa/ op directory)
#              + a _ttnncpp.so rebuilt in ttbuild from K64j-OQ's OWN source state (the decode factory at K64j, the four K64j kernels, the chain reader, the oneq prefill factory PS) with the
#                prefill factory replaced by the K-split one (apply_factory_pk.py over PS),
#              + three NEW kernel files in sdpa/device/kernels/ (reader_interleaved_qwen_ksplit.cpp, writer_interleaved_qwen_ksplit.cpp, sdpa_qwen_ksplit.cpp: JIT sources the factory selects only for
#                flag 0x10; make_ks_reader.py, make_ks_writer.py, make_ks_compute.py).
#   A graft .so replaces the WHOLE binary (memory graft-so-drops-image-patches), so this build starts from K64j-OQ's exact source tree, adds only the K-split edits, and REFUSES to publish
#   unless the result is K64j-OQ with that change and nothing else (the checks, in order):
#     1. the base is K64j-OQ: its MANIFEST verifies and its _ttnncpp.so is the sha packed_any_admission.K64J_OQ_TTNNCPP_SHA256 records;
#     2. the generators are the recorded files and their outputs are the recorded shas (PK factory over the saved fd8c0676, the three kernels from the pinned sources);
#     3. REPRODUCTION (unless KS_SKIP_REPRODUCE=1): K64j-OQ alone, rebuilt in this ttbuild, is compared with the base _ttnncpp.so. Byte-identical means the only delta of the result is the
#        patch; not identical is a WARNING (KS_REQUIRE_REPRODUCE=1 makes it fatal) and checks 4-6 carry the weight;
#     4. every distinct QWEN_ / [QWEN- string of the base survives in the result and the ONLY new ones are the four literals the patch adds (apply_factory_pk.NEW_QWEN_STRINGS);
#     5. the exported symbols (nm -D) of the result equal the base's, where nm exists;
#     6. the graft is the base directory with exactly one file replaced (_ttnncpp.so) and exactly three files added (the ksplit kernels); _ttnn.so and every other op directory are the
#        base's; the manifest differs from the base's by those four paths only.
#   Then ttbuild is restored (the factories from the saved bases, the kernels put back or removed), rebuilt clean, and checked.
#
# THROUGH CI (the owner is offsite): the cardm action of qwen-c2-serving.yml
#   C2_CARDM_HARNESS=optimisation/ttnn-op/sdpa_prefill_ksplit/build_k64j_oq_ks.sh
#   C2_CARDM_ENV=KS_BUILD_DRY_RUN=1      # first: every check the host can make, every command printed, nothing run
#   C2_CARDM_ENV=                        # then the build: about four ninja runs of 30 to 60 s
# runs it on the rig host from the checkout root with QUAL_CARD = card M and ALLOW_SERVING_CARD=1 (the card is context only: nothing here opens a device).
# BY HAND on the rig: QUAL_CARD=<board id> [ALLOW_SERVING_CARD=1] bash optimisation/ttnn-op/sdpa_prefill_ksplit/build_k64j_oq_ks.sh [graft dir]   (no default board, card B refused)
#
# Env (C2_CARDM_ENV carries NAME=value words without spaces):
#   KS_BASE_GRAFT        the K64j-OQ graft to layer on (~/opgraft-K64j-OQ). NEVER written, never moved. KS_BASE_SHA256: its _ttnncpp.so (default: the sha packed_any_admission.py
#                        records for K64j-OQ; empty skips, last resort)
#   KS_GRAFT             the graft to write (~/opgraft-K64j-OQ-KS; a first argument also sets it). Built in <it>.partial and moved into place only after every check; an existing one
#                        is refused (move it away yourself): no existing graft is ever overwritten
#   KS_WORK_ROOT         work-<stamp>/ (backups, copies out of ttbuild, strings lists): ~/kwork64/k64j-oq-ks
#   K64J_DECODE_BASE_DIR / K64J_PF_BASE_DIR   the saved 3e0a69af decode and fd8c0676 prefill factories (build_k64j.sh's: ~/kwork64/k64f, ~/kwork64/k64g)
#   KS_IMAGES            images (comma or space separated) whose _ttnncpp.so QWEN strings must also survive; default none (the base graft IS the image's binary)
#   KS_SKIP_REPRODUCE=1  skip the K64j-OQ-alone build (one ninja run less); KS_REQUIRE_REPRODUCE=1  a K64j-OQ rebuild that is not byte-identical is fatal
#   KS_ALLOW_SYMBOL_DELTA=1  a different exported-symbol set is a warning (it should be identical: the patch adds no function)
#   KS_KEEP_STAGED=1     leave ttbuild staged; KS_BUILD_DRY_RUN=1  print every docker, cp, mv, rm, touch, mkdir and generator command as '### run: ...' and run none of them
#   QUAL_CARD            only validated (qual_card_select): this build never opens a card
set -euo pipefail
export LC_ALL=C

S=$(cd "$(dirname "$0")" && pwd)
OPS_DIR=$(cd "$S/.." && pwd)
CHECKOUT=$(cd "$S/../../.." && pwd)
ONEQ_DIR=$OPS_DIR/sdpa_prefill_oneq
K64J_DIR=$OPS_DIR/k64j
K64J_BUILD=$K64J_DIR/build_k64j.sh
ONEQ_BUILD=$ONEQ_DIR/build_k64j_oq.sh
lib=$CHECKOUT/scripts/ci/qual_card.sh
for path in "$K64J_BUILD" "$ONEQ_BUILD" "$lib"; do
  test -s "$path" || { echo "refusing: $path missing (the checkout carries k64j, sdpa_prefill_oneq and sdpa_prefill_ksplit side by side, and scripts/ci/qual_card.sh)" >&2; exit 1; }
done
# ---- K64j's recorded shas, kernel lists and markers: ONE source, build_k64j.sh's own header (a copy would drift) ----
eval "$(sed -n '/^# ---- recorded shas/,/^# >>> qual/p' "$K64J_BUILD" | sed '$d')"
test -n "${FACTORY_K64J:-}" && test -n "${PF_FACTORY:-}" && [ "${#DECODE_KERNELS[@]}" = 4 ] \
  || { echo "refusing: could not read K64j's recorded shas from $K64J_BUILD" >&2; exit 1; }
# ---- the oneq build's recorded shas (the PS factory and script, the oneq literals): its header, not a copy ----
eval "$(sed -n '/^PS_FACTORY=/p;/^PS_SCRIPT=/p;/^OQ_LITERAL_1=/p;/^OQ_LITERAL_2=/p' "$ONEQ_BUILD")"
test -n "${PS_FACTORY:-}" && test -n "${PS_SCRIPT:-}" && test -n "${OQ_LITERAL_1:-}" \
  || { echo "refusing: could not read the oneq build's recorded shas from $ONEQ_BUILD" >&2; exit 1; }
OQ_LITERALS=("$OQ_LITERAL_1" "$OQ_LITERAL_2")

# ---- this build's own recorded shas (test_ksplit_build.py keeps them equal to the generators and their outputs) ----
PK_FACTORY=692c79ee9cbc2f29ac3ab88bb274828afcf3ce76461930be61bd6edd07c45e68
PK_SCRIPT=0000000000000000000000000000000000000000000000000000000000000000
KS_READER=16438818a762c699d929c579903925eba175b6f0e61171efa01dc0113a8dc128
KS_WRITER=9b9ef47f9af44cfe83dc82d587568cfd82c4881021306f7af677cb672fd589ee
KS_COMPUTE=fc9571d178639f139b1da0f6aa385b5b10963f647df83860e763a7caff893913
KS_LITERAL_1='[QWEN-SDPA-PF] ksplit needs at least 2 slices: {} q chunks on {} cores'
KS_LITERAL_2='[QWEN-SDPA-PF] ksplit needs a scale exact in bf16'
KS_LITERAL_3='[QWEN-SDPA-PF] ksplit needs fp32 dest accumulation and exact exp'
KS_LITERAL_4='[QWEN-SDPA-PF] ksplit=1 slices={} q_chunks={} cores={} busy={} chains={} members={}'
KS_LITERALS=("$KS_LITERAL_1" "$KS_LITERAL_2" "$KS_LITERAL_3" "$KS_LITERAL_4")

# The card selection is the canonical library SOURCED (the scripts/ci rig runners do the same), not embedded: the embedded copies carry board ids, which a new public file must not.
# shellcheck disable=SC1090
. "$lib"
qual_card_select
DRY=${KS_BUILD_DRY_RUN:-0}
case $DRY in
  0|1) ;;
  *) echo "refusing: KS_BUILD_DRY_RUN=$DRY (0 or 1)" >&2; exit 1 ;;
esac
if [ -n "${CARD_B_ARGS:-}" ]; then
  echo "refusing: CARD_B_ARGS='$CARD_B_ARGS': build_k64j_oq_ks.sh takes no arguments from the job file; set KS_* in C2_CARDM_ENV" >&2
  exit 1
fi
if [ $# -gt 1 ]; then
  echo "refusing: build_k64j_oq_ks.sh takes at most one argument, the graft directory (got $#)" >&2
  exit 1
fi

KJ=$K64J_DIR/kernels
CH=$OPS_DIR/sdpa_prefill_chain
PATCHER_PS=$ONEQ_DIR/apply_factory_ps.py
PATCHER=$S/apply_factory_pk.py
ADMISSION=$CHECKOUT/scripts/ci/packed_any_admission.py
DBASE=${K64J_DECODE_BASE_DIR:-$HOME/kwork64/k64f}
PBASE=${K64J_PF_BASE_DIR:-$HOME/kwork64/k64g}
BASE=${KS_BASE_GRAFT:-$HOME/opgraft-K64j-OQ}
GRAFT=${1:-${KS_GRAFT:-$HOME/opgraft-K64j-OQ-KS}}
G=$GRAFT.partial
WROOT=${KS_WORK_ROOT:-$HOME/kwork64/k64j-oq-ks}
IMAGES=${KS_IMAGES:-}
IMAGES=${IMAGES//,/ }
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
D=$OPS/sdpa_decode
F=$D/device/sdpa_decode_program_factory.cpp
KD=$D/device/kernels
P=$OPS/sdpa
PF=$P/device/sdpa_program_factory.cpp
PK=$P/device/kernels
TB_SO=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so
SAVED_DECODE=$DBASE/sdpa_decode_program_factory.cpp.$FACTORY_BASE
SAVED_PF=$PBASE/sdpa_program_factory.cpp.$PF_BASE
KS_FILES=(dataflow/reader_interleaved_qwen_ksplit.cpp dataflow/writer_interleaved_qwen_ksplit.cpp compute/sdpa_qwen_ksplit.cpp)
KS_SHAS=("$KS_READER" "$KS_WRITER" "$KS_COMPUTE")
KS_SOURCES=("$S/reader_interleaved_qwen_ksplit.cpp" "$S/writer_interleaved_qwen_ksplit.cpp" "$S/sdpa_qwen_ksplit.cpp")
# The expected base binary: the one the repo records for the oneq image, unless the caller gives another (empty skips).
if [ -n "${KS_BASE_SHA256+x}" ]; then
  BASE_SHA=$KS_BASE_SHA256
else
  BASE_SHA=$(grep -oE "^K64J_OQ_TTNNCPP_SHA256 = '[0-9a-f]{64}'" "$ADMISSION" 2>/dev/null | grep -oE '[0-9a-f]{64}' || true)
  test -n "$BASE_SHA" || { echo "refusing: no K64J_OQ_TTNNCPP_SHA256 in $ADMISSION (give KS_BASE_SHA256)" >&2; exit 1; }
fi

canon() {  # one absolute spelling of a path
  local path=$1
  if command -v cygpath >/dev/null 2>&1; then
    path=$(cygpath -u -- "$path")
  fi
  readlink -m -- "$path"
}
under() {  # path root -> true when path is root or inside it
  local path root
  path=$(canon "$1")
  root=$(canon "$2")
  case "$path/" in "$root/"*) return 0 ;; esac
  return 1
}
case $GRAFT in
  ''|/|.|..) echo "refusing: graft directory '$GRAFT'" >&2; exit 1 ;;
esac
if under "$GRAFT" "$BASE" || under "$BASE" "$GRAFT"; then
  echo "refusing: the graft $GRAFT and the base $BASE overlap (the base is K64j-OQ: it is copied, never written)" >&2
  exit 1
fi
for path in "$GRAFT" "$WROOT"; do
  if under "$path" "$CHECKOUT"; then
    echo "refusing: $path is inside the checkout $CHECKOUT (the next checkout cleans it)" >&2
    exit 1
  fi
done
if [ -e "$GRAFT" ]; then
  echo "refusing: $GRAFT exists (a graft is never overwritten: move it away, or name another with KS_GRAFT)" >&2
  exit 1
fi

stamp=$(date +%Y%m%dT%H%M%S)
exec 3>&1
x() {
  if [ "$DRY" = 1 ]; then
    echo "### run: $(printf '%q ' "$@")" >&3
    return 0
  fi
  "$@"
}
fail() {
  echo "FAIL: $*" >&2
  exit 1
}
need() {  # label actual expected (an empty actual in a dry run is a command that was not run)
  if [ "$DRY" = 1 ] && [ -z "$2" ]; then
    echo "dry   $1: must be ${3:0:16}"
    return 0
  fi
  if [ "$2" != "$3" ]; then
    echo "FAIL: $1 is ${2:-missing}, expected $3" >&2
    exit 1
  fi
  echo "ok    $1 ${2:0:16}"
}
hsha() { sha256sum "$1" 2>/dev/null | cut -c1-64 || true; }
csha() { x docker exec ttbuild sha256sum "$1" | cut -c1-64; }
cexists() { x docker exec ttbuild test -e "$1"; }
count() { strings "$1" | grep -cF -- "$2" || true; }
qwen_strings() { strings "$1" | grep -E 'QWEN_|\[QWEN-' | sort -u || true; }
symbols() { nm -D --defined-only "$1" 2>/dev/null | awk '{print $NF}' | sort -u || true; }
nearest() {
  local path
  path=$(canon "$1")
  while [ ! -e "$path" ] && [ "$path" != / ]; do
    path=$(dirname "$path")
  done
  echo "$path"
}
echo "### user $(id -un 2>/dev/null || echo unknown) HOME $HOME"
for path in "$GRAFT" "$WROOT" "$DBASE" "$PBASE"; do
  at=$(nearest "$path")
  free=$(df -Pk "$at" 2>/dev/null | awk 'NR == 2 { printf "%d MB free", $4 / 1024 }' || true)
  if [ -w "$at" ]; then
    echo "### $path: its nearest existing directory $at is writable, ${free:-free space unknown}"
  elif [ "$DRY" = 1 ]; then
    echo "### $path: its nearest existing directory $at is NOT WRITABLE by $(id -un 2>/dev/null) (the build would stop here)"
  else
    fail "$path: its nearest existing directory $at is not writable by $(id -un 2>/dev/null)"
  fi
done

if [ "$DRY" = 1 ]; then
  W=$(mktemp -d "${TMPDIR:-/tmp}/k64j-oq-ks-dry.XXXXXX")
  echo "### build_k64j_oq_ks DRY RUN $(date -Is) graft=$GRAFT base=$BASE card context $QUAL_CARD ($QUAL_TAG): nothing is built"
  x mkdir -p "$WROOT/work-$stamp/backup"
else
  command -v docker >/dev/null 2>&1 || fail "docker is not on PATH"
  command -v strings >/dev/null 2>&1 || fail "strings (binutils) is not on PATH: the QWEN-string checks need it"
  W=$WROOT/work-$stamp
  mkdir -p "$W/backup"
  echo "### build_k64j_oq_ks $(date -Is) work=$W graft=$GRAFT base=$BASE card context $QUAL_CARD ($QUAL_TAG): no device is opened"
fi
command -v python3 >/dev/null 2>&1 || fail "python3 is not on PATH: the factory generators are python"
HAVE_NM=0
command -v nm >/dev/null 2>&1 && HAVE_NM=1
[ "$HAVE_NM" = 1 ] || echo "WARN  nm is not on PATH: check 5 (the exported symbols) will be skipped"

# The EXIT trap: ttbuild's sources back (from the first staging copy on), and the dry run's scratch removed.
armed=0
restored=0
built=0
rebuilt=0
restore_ttbuild() {
  if [ "$armed" = "0" ] || [ "$restored" = "1" ] || [ "${KS_KEEP_STAGED:-}" = "1" ]; then
    return 0
  fi
  x docker cp "$SAVED_DECODE" "ttbuild:$F"
  x docker cp "$SAVED_PF" "ttbuild:$PF"
  for kernel in "${DECODE_KERNELS[@]}"; do
    if [ -e "$W/backup/$(basename "$kernel").replaced" ]; then
      x docker cp "$W/backup/$(basename "$kernel").replaced" "ttbuild:$KD/$kernel"
    else
      x docker exec ttbuild rm -f "$KD/$kernel"
    fi
  done
  if [ -e "$W/backup/reader_interleaved_qwen_chain.cpp.replaced" ]; then
    x docker cp "$W/backup/reader_interleaved_qwen_chain.cpp.replaced" "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp"
  else
    x docker exec ttbuild rm -f "$PK/dataflow/reader_interleaved_qwen_chain.cpp"
  fi
  for file in "${KS_FILES[@]}"; do
    if [ -e "$W/backup/$(basename "$file").replaced" ]; then
      x docker cp "$W/backup/$(basename "$file").replaced" "ttbuild:$PK/$file"
    else
      x docker exec ttbuild rm -f "$PK/$file"
    fi
  done
  # docker cp keeps the saved files' old mtimes: touch, or ninja calls the unity TUs up to date.
  x docker exec ttbuild touch "$F" "$PF"
  restored=1
  if [ "$DRY" = 1 ]; then
    echo "ttbuild restore: the commands above (the factories from the saved bases, the qwen kernel files removed or put back)"
  else
    echo "ttbuild restored: decode $(csha $F | cut -c1-16), prefill $(csha $PF | cut -c1-16) (touched)"
  fi
}
on_exit() {
  local status=$?
  restore_ttbuild || echo "WARN: ttbuild restore failed; check $F and $PF in ttbuild" >&2
  if [ "$DRY" != 1 ] && [ "$built" = "1" ] && [ "$rebuilt" = "0" ] && [ "${KS_KEEP_STAGED:-}" != "1" ]; then
    echo "WARN  ttbuild build_Release still holds the staged objects; the next ninja run recompiles the touched factories, but do not copy a .so out of build_Release before one" >&2
  fi
  if [ "$DRY" = 1 ]; then
    rm -rf "$W"
  elif [ "$status" != 0 ] && [ -e "$G" ]; then
    echo "note  $G is left for inspection; the next run removes it" >&2
  fi
}
trap on_exit EXIT

# ---------- 1. preflight: the checkout (the K64j sources, the generators, the three kernels) ----------
for path in "$KJ/dataflow/reader_decode_qwen.cpp" "$KJ/dataflow/reader_decode_qwen_slice.cpp" \
            "$KJ/compute/sdpa_flash_decode_qwen.cpp" "$KJ/dataflow/writer_decode_qwen_slice.cpp" \
            "$K64J_DIR/apply_factory_k64j.py" "$CH/apply_factory_pf.py" "$CH/reader_interleaved_qwen_chain.cpp" \
            "$PATCHER_PS" "$PATCHER" "$ADMISSION" "${KS_SOURCES[@]}"; do
  test -s "$path" || fail "$path missing"
  if grep -q $'\r' "$path"; then fail "$path has CRLF line endings"; fi
done
for kernel in "${DECODE_KERNELS[@]}"; do
  need "shipped K64j $kernel" "$(hsha "$KJ/$kernel")" "$(k64j_sha "$kernel")"
done
need "shipped reader_interleaved_qwen_chain.cpp" "$(hsha "$CH/reader_interleaved_qwen_chain.cpp")" $PF_READER
need "shipped apply_factory_ps.py" "$(hsha "$PATCHER_PS")" $PS_SCRIPT
need "shipped apply_factory_pk.py" "$(hsha "$PATCHER")" $PK_SCRIPT
for index in 0 1 2; do
  need "shipped ${KS_FILES[$index]}" "$(hsha "${KS_SOURCES[$index]}")" "${KS_SHAS[$index]}"
done

# ---------- 2. preflight: the base graft (K64j-OQ; read-only) ----------
base_so=
if [ "$DRY" = 1 ] && [ ! -e "$BASE" ]; then
  echo "### dry run: $BASE does not exist here; the base graft was not checked"
else
  for part in _ttnncpp.so _ttnn.so attn_prep nlp_concat_heads_decode sdpa_decode sdpa MANIFEST.sha256; do
    test -e "$BASE/$part" || fail "$BASE/$part missing (the base is K64j-OQ, build_k64j_oq.sh)"
  done
  (cd "$BASE" && sha256sum -c --quiet MANIFEST.sha256) || fail "$BASE/MANIFEST.sha256 does not verify"
  echo "ok    $BASE/MANIFEST.sha256 verifies"
  base_so=$(hsha "$BASE/_ttnncpp.so")
  if [ -n "$BASE_SHA" ]; then
    need "base graft _ttnncpp.so (K64j-OQ, the oneq image's)" "$base_so" "$BASE_SHA"
  else
    echo "WARN  KS_BASE_SHA256 is empty: the base graft is not checked against the recorded K64j-OQ binary"
  fi
  BK=$BASE/sdpa_decode/device/kernels
  for kernel in "${DECODE_KERNELS[@]}"; do
    need "base graft $kernel (K64j)" "$(hsha "$BK/$kernel")" "$(k64j_sha "$kernel")"
  done
  need "base graft sdpa/ chain reader" "$(hsha "$BASE/sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp")" $PF_READER
  for file in "${KS_FILES[@]}"; do
    test ! -e "$BASE/sdpa/device/kernels/$file" || fail "$BASE/sdpa/device/kernels/$file already exists (the base is not K64j-OQ)"
  done
  if command -v strings >/dev/null 2>&1; then
    for marker in "${K64J_MARKERS[@]}" "${STAGE4_MARKERS[@]}" "${PF_MARKERS[@]}" QWEN_SDPA_TREE_SCRATCH_ROUNDS; do
      test "$(count "$BASE/_ttnncpp.so" "$marker")" -ge 1 || fail "$BASE/_ttnncpp.so lacks '$marker' (not K64j-OQ)"
    done
    for literal in "${OQ_LITERALS[@]}"; do
      test "$(count "$BASE/_ttnncpp.so" "$literal")" -ge 1 || fail "$BASE/_ttnncpp.so lacks the oneq literal '$literal' (not K64j-OQ)"
    done
    for literal in "${KS_LITERALS[@]}"; do
      test "$(count "$BASE/_ttnncpp.so" "$literal")" -eq 0 || fail "$BASE/_ttnncpp.so already has '$literal' (it is not K64j-OQ)"
    done
    qwen_strings "$BASE/_ttnncpp.so" > "$W/base-qwen-strings.txt"
    echo "ok    base graft ${base_so:0:16}: K64j and oneq markers present, $(wc -l < "$W/base-qwen-strings.txt") distinct QWEN strings recorded"
  fi
  if [ "$HAVE_NM" = 1 ]; then
    symbols "$BASE/_ttnncpp.so" > "$W/base-symbols.txt"
    echo "ok    base graft: $(wc -l < "$W/base-symbols.txt") exported symbols recorded"
  fi
fi

# ---------- 3. preflight: the saved factory bases, ttbuild ----------
for pair in "decode|$SAVED_DECODE|$FACTORY_BASE" "prefill|$SAVED_PF|$PF_BASE"; do
  label=${pair%%|*}
  rest=${pair#*|}
  path=${rest%|*}
  if [ -e "$path" ]; then
    need "saved $label base factory" "$(hsha "$path")" "${rest##*|}"
  else
    echo "note  no saved $label base at $path yet: step 4 saves it from ttbuild when ttbuild holds the base, else fails"
  fi
done
if [ "$DRY" = 1 ]; then
  x docker ps --format '{{.Names}}'
else
  docker ps --format '{{.Names}}' | grep -qx ttbuild || fail "ttbuild container is not running"
  echo "ok    ttbuild is running"
fi
dcur=$(csha $F)
pcur=$(csha $PF)
if [ "$DRY" != 1 ]; then
  case "$dcur" in
    "$FACTORY_BASE"|"$FACTORY_QWEN_STAGE1"|"$FACTORY_QWEN_STAGE3"|"$FACTORY_QWEN_STAGE4"|"$FACTORY_K64J") ;;
    "$FACTORY_UNPATCHED") fail "ttbuild decode factory is the unpatched 05708e6d; run the K64d build (tree-scratch) first" ;;
    *) fail "unexpected ttbuild decode factory ${dcur:-missing}" ;;
  esac
  case "$pcur" in
    "$PF_BASE"|"$PF_FACTORY"|"$PS_FACTORY") ;;
    "$PK_FACTORY") echo "note  ttbuild prefill factory is already the ksplit one (a KS_KEEP_STAGED run); it is restored from the saved base at the end" ;;
    *) fail "unexpected ttbuild prefill factory ${pcur:-missing}" ;;
  esac
  echo "ok    ttbuild: decode ${dcur:0:16}, prefill ${pcur:0:16}"
fi
need "ttbuild reader_decode_all.cpp" "$(csha $KD/dataflow/reader_decode_all.cpp)" $READER_ALL
need "ttbuild sdpa_flash_decode.cpp" "$(csha $KD/compute/sdpa_flash_decode.cpp)" $COMPUTE_ALL
need "ttbuild writer_decode_all.cpp" "$(csha $KD/dataflow/writer_decode_all.cpp)" $WRITER_ALL
need "ttbuild decode dataflow_common.hpp" "$(csha $KD/dataflow/dataflow_common.hpp)" $DATAFLOW_COMMON
need "ttbuild rt_args_common.hpp" "$(csha $KD/rt_args_common.hpp)" $RT_ARGS_COMMON
need "ttbuild reader_interleaved.cpp" "$(csha $PK/dataflow/reader_interleaved.cpp)" $PF_READER_BASE
need "ttbuild writer_interleaved.cpp" "$(csha $PK/dataflow/writer_interleaved.cpp)" 2a0959ff7cce507eab39f2e063f4dbc264d8b51f4db9f479b8c2419435c745c2
need "ttbuild sdpa.cpp (compute)" "$(csha $PK/compute/sdpa.cpp)" a3f48af8ba0fd63b136c79a54c8b6f7b4b5b8fb0d7a209bf5701f081ed7fa3e0
need "ttbuild compute_common.hpp" "$(csha $PK/compute/compute_common.hpp)" 3fb5da2440c3bf90ebceb8acd55424c7739339de4c6b02db83836e1e3414fa19
if [ -n "$IMAGES" ]; then
  for image in $IMAGES; do
    if [ "$DRY" = 1 ]; then
      x docker image inspect "$image"
    else
      docker image inspect "$image" >/dev/null 2>&1 || fail "image $image is not present"
    fi
  done
fi

# ---------- 4. stage K64j-OQ exactly as build_k64j_oq.sh does (K64j, then the PS prefill factory), backing up anything replaced ----------
x docker cp ttbuild:$F "$W/backup/sdpa_decode_program_factory.cpp.${dcur:0:8}"
x docker cp ttbuild:$PF "$W/backup/sdpa_program_factory.cpp.${pcur:0:8}"
x mkdir -p "$DBASE" "$PBASE"
if [ "$dcur" = "$FACTORY_BASE" ] && [ ! -e "$SAVED_DECODE" ]; then
  x cp "$W/backup/sdpa_decode_program_factory.cpp.${dcur:0:8}" "$SAVED_DECODE"
fi
if [ "$pcur" = "$PF_BASE" ] && [ ! -e "$SAVED_PF" ]; then
  x cp "$W/backup/sdpa_program_factory.cpp.${pcur:0:8}" "$SAVED_PF"
fi
if [ "$DRY" != 1 ]; then
  test -s "$SAVED_DECODE" || fail "no saved 3e0a69af decode factory in $DBASE (and ttbuild does not hold it)"
  test -s "$SAVED_PF" || fail "no saved fd8c0676 prefill factory in $PBASE (and ttbuild does not hold it)"
fi
need "saved decode base factory" "$( [ "$DRY" = 1 ] || hsha "$SAVED_DECODE")" $FACTORY_BASE
need "saved prefill base factory" "$( [ "$DRY" = 1 ] || hsha "$SAVED_PF")" $PF_BASE
x python3 -B "$K64J_DIR/apply_factory_k64j.py" "$SAVED_DECODE" --out "$W/sdpa_decode_program_factory.cpp"
need "patched decode factory (K64j)" "$( [ "$DRY" = 1 ] || hsha "$W/sdpa_decode_program_factory.cpp")" $FACTORY_K64J
x python3 -B "$PATCHER_PS" "$SAVED_PF" --out "$W/sdpa_program_factory.ps.cpp"
need "oneq prefill factory (PS0-PS5 over the saved base)" "$( [ "$DRY" = 1 ] || hsha "$W/sdpa_program_factory.ps.cpp")" $PS_FACTORY
x python3 -B "$PATCHER" "$SAVED_PF" --out "$W/sdpa_program_factory.pk.cpp"
need "ksplit prefill factory (PK0-PK13 over PS over the saved base)" "$( [ "$DRY" = 1 ] || hsha "$W/sdpa_program_factory.pk.cpp")" $PK_FACTORY
x python3 -B "$PATCHER" "$W/sdpa_program_factory.ps.cpp" --out "$W/sdpa_program_factory.pk2.cpp"
need "ksplit prefill factory (PK0-PK13 over the PS factory)" "$( [ "$DRY" = 1 ] || hsha "$W/sdpa_program_factory.pk2.cpp")" $PK_FACTORY

armed=1
x docker cp "$W/sdpa_decode_program_factory.cpp" "ttbuild:$F"
x docker cp "$W/sdpa_program_factory.ps.cpp" "ttbuild:$PF"      # the oneq prefill factory first: build A is K64j-OQ alone
x docker exec ttbuild touch "$F" "$PF"
need "ttbuild decode factory now" "$(csha $F)" $FACTORY_K64J
need "ttbuild prefill factory now (oneq)" "$(csha $PF)" $PS_FACTORY
for kernel in "${DECODE_KERNELS[@]}"; do
  if cexists "$KD/$kernel"; then
    x docker cp "ttbuild:$KD/$kernel" "$W/backup/$(basename "$kernel").replaced"
  fi
  x docker cp "$KJ/$kernel" "ttbuild:$KD/$kernel"
done
if cexists "$PK/dataflow/reader_interleaved_qwen_chain.cpp"; then
  x docker cp "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp" "$W/backup/reader_interleaved_qwen_chain.cpp.replaced"
fi
x docker cp "$CH/reader_interleaved_qwen_chain.cpp" "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp"
for kernel in "${DECODE_KERNELS[@]}"; do
  need "ttbuild $kernel" "$(csha "$KD/$kernel")" "$(k64j_sha "$kernel")"
done
need "ttbuild reader_interleaved_qwen_chain.cpp" "$(csha $PK/dataflow/reader_interleaved_qwen_chain.cpp)" $PF_READER
for file in "${KS_FILES[@]}"; do   # not staged yet: build A must not see them (they are inert JIT sources, but the reproduction is K64j-OQ alone)
  if cexists "$PK/$file"; then
    x docker cp "ttbuild:$PK/$file" "$W/backup/$(basename "$file").replaced"
  fi
done

# ---------- 5. build A: K64j-OQ alone (the reproduction of the base) ----------
built=1
a_sha=
if [ "${KS_SKIP_REPRODUCE:-}" = "1" ]; then
  echo "WARN  KS_SKIP_REPRODUCE=1: K64j-OQ was not rebuilt alone; the delta to the base is judged by checks 4-6 only"
else
  echo "### ninja A (K64j-OQ alone) start $(date -Is)"
  x docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
  echo "### ninja A done $(date -Is)"
  x docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so "$W/A-ttnncpp.so"
  if [ "$DRY" != 1 ]; then
    a_sha=$(hsha "$W/A-ttnncpp.so")
    if [ "$a_sha" = "$base_so" ]; then
      echo "ok    REPRODUCED: K64j-OQ rebuilt alone in this ttbuild is byte-identical to the base graft's _ttnncpp.so (${a_sha:0:16}): the only delta of the result is the patch"
    elif [ "${KS_REQUIRE_REPRODUCE:-}" = "1" ]; then
      fail "K64j-OQ rebuilt alone (${a_sha:0:16}) is not byte-identical to the base graft's (${base_so:0:16}) and KS_REQUIRE_REPRODUCE=1"
    else
      echo "WARN  K64j-OQ rebuilt alone (${a_sha:0:16}) is NOT byte-identical to the base graft's (${base_so:0:16}): the toolchain, paths or timestamps differ; checks 4-6 carry the weight"
    fi
  fi
fi

# ---------- 6. build B: K64j-OQ + the ksplit prefill factory (+ its three kernels, which the build does not compile: they are JIT sources) ----------
x docker cp "$W/sdpa_program_factory.pk.cpp" "ttbuild:$PF"
x docker exec ttbuild touch "$PF"
need "ttbuild prefill factory now (ksplit)" "$(csha $PF)" $PK_FACTORY
for index in 0 1 2; do
  x docker cp "${KS_SOURCES[$index]}" "ttbuild:$PK/${KS_FILES[$index]}"
  need "ttbuild ${KS_FILES[$index]}" "$(csha "$PK/${KS_FILES[$index]}")" "${KS_SHAS[$index]}"
done
echo "### ninja B (K64j-OQ + ksplit) start $(date -Is)"
x docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
echo "### ninja B done $(date -Is)"
x docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so "$W/B-ttnncpp.so"
x docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnn.so "$W/B-ttnn.so"

# ---------- 7. assemble: the base directory with exactly one file replaced and exactly three added ----------
x rm -rf "$G"
x mkdir -p "$G"
x cp -a "$BASE/." "$G/"
x rm -f "$G/MANIFEST.sha256"
x cp "$W/B-ttnncpp.so" "$G/_ttnncpp.so"
for index in 0 1 2; do
  x cp "${KS_SOURCES[$index]}" "$G/sdpa/device/kernels/${KS_FILES[$index]}"
done

# ---------- 8. restore ttbuild (before the verify; the EXIT trap covers a failure) ----------
if [ "${KS_KEEP_STAGED:-}" = "1" ]; then
  echo "note  KS_KEEP_STAGED=1: ttbuild keeps the staged factories and kernels"
else
  restore_ttbuild
  need "ttbuild decode factory restored" "$(csha $F)" $FACTORY_BASE
  need "ttbuild prefill factory restored" "$(csha $PF)" $PF_BASE
  echo "### ninja (restore build_Release to the audited sources) start $(date -Is)"
  x docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
  echo "### ninja (restore) done $(date -Is)"
  tb_flags=$(x docker exec ttbuild bash -c "grep -caF -- '[QWEN-SDPA] flags=' $TB_SO || true")
  tb_extent=$(x docker exec ttbuild bash -c "grep -caF -- '$EXTENT_MARKER' $TB_SO || true")
  tb_pf=$(x docker exec ttbuild bash -c "grep -caF -- '[QWEN-SDPA-PF]' $TB_SO || true")
  tb_ks=$(x docker exec ttbuild bash -c "grep -caF -- 'ksplit needs at least 2 slices' $TB_SO || true")
  tb_scratch=$(x docker exec ttbuild bash -c "grep -caF -- QWEN_SDPA_TREE_SCRATCH_ROUNDS $TB_SO || true")
  if [ "$DRY" != 1 ]; then
    test "$tb_flags" = "0" || fail "ttbuild's rebuilt _ttnncpp.so still has the [QWEN-SDPA] branch; build_Release is dirty"
    test "$tb_extent" = "0" || fail "ttbuild's rebuilt _ttnncpp.so still has the K64j runtime extent; build_Release is dirty"
    test "$tb_pf" = "0" || fail "ttbuild's rebuilt _ttnncpp.so still has the [QWEN-SDPA-PF] branch; build_Release is dirty"
    test "$tb_ks" = "0" || fail "ttbuild's rebuilt _ttnncpp.so still has the ksplit edits; build_Release is dirty"
    test "$tb_scratch" -ge 1 || fail "ttbuild's rebuilt _ttnncpp.so lost the tree-scratch factory"
    echo "ok    ttbuild build_Release rebuilt from the audited sources (no qwen branch, no ksplit, tree scratch present)"
  fi
  rebuilt=1
fi

# ---------- 9. verify: K64j-OQ plus the ksplit edits and nothing else ----------
echo "### verify $G"
if [ "$DRY" = 1 ]; then
  echo "### dry run: step 9 reads the assembled graft; its checks were not run"
else
  so=$G/_ttnncpp.so
  test "$(hsha "$so")" != "$base_so" || fail "the build produced the base graft's _ttnncpp.so unchanged (the patch did not reach the binary)"
  if [ -n "$a_sha" ]; then
    test "$(hsha "$so")" != "$a_sha" || fail "build B equals build A: the patch did not reach the binary"
  fi
  qwen_strings "$so" > "$W/graft-qwen-strings.txt"
  lost=$(comm -23 "$W/base-qwen-strings.txt" "$W/graft-qwen-strings.txt")
  if [ -n "$lost" ]; then
    echo "FAIL: QWEN strings of $BASE/_ttnncpp.so missing from the new .so (a graft .so replaces the whole binary):" >&2
    echo "$lost" >&2
    exit 1
  fi
  echo "ok    every one of the base's $(wc -l < "$W/base-qwen-strings.txt") distinct QWEN strings is in the new .so ($(wc -l < "$W/graft-qwen-strings.txt") now)"
  for literal in "${KS_LITERALS[@]}"; do
    test "$(count "$so" "$literal")" -ge 1 || fail "the new _ttnncpp.so lacks '$literal'"
  done
  added=$(comm -13 "$W/base-qwen-strings.txt" "$W/graft-qwen-strings.txt")
  unexpected=
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    known=0
    for literal in "${KS_LITERALS[@]}"; do
      case $line in *"$literal"*) known=1 ;; esac
    done
    [ "$known" = 1 ] || unexpected="$unexpected$line"$'\n'
  done <<< "$added"
  if [ -n "$unexpected" ]; then
    echo "FAIL: the new .so's QWEN strings beyond the base's are not the ksplit literals:" >&2
    printf '%s' "$unexpected" >&2
    exit 1
  fi
  echo "ok    the only new QWEN strings are the four ksplit literals ($(printf '%s\n' "$added" | grep -c . || true) lines)"
  for marker in "${K64J_MARKERS[@]}" "${STAGE4_MARKERS[@]}" "${PF_MARKERS[@]}" '[QWEN-SDPA] flags=' QWEN_SDPA_TREE_SCRATCH_ROUNDS \
                reader_decode_qwen.cpp "$SHARE_MARKER" "${OQ_LITERALS[@]}"; do
    test "$(count "$so" "$marker")" -ge 1 || fail "the new _ttnncpp.so lacks '$marker'"
  done
  echo "ok    strings: every K64j, K64i-slice, prefill-chain, oneq and tree-scratch marker of the base is present"
  for path in sdpa_qwen_ksplit.cpp reader_interleaved_qwen_ksplit.cpp writer_interleaved_qwen_ksplit.cpp; do
    test "$(count "$so" "$path")" -ge 1 || fail "the new _ttnncpp.so does not name the kernel $path (the factory does not select it)"
  done
  echo "ok    the factory names the three ksplit kernels"
  test "$(count "$so" "$STAGE1_REFUSAL")" -eq 0 || fail "the new _ttnncpp.so refuses KV share (the stage-1 decode factory was linked)"
  test "$(count "$so" qwen_draft_fp32_intermediates)" = "$(count "$BASE/_ttnncpp.so" qwen_draft_fp32_intermediates)" \
    || fail "the combined prefill factory's qwen_draft_fp32_intermediates count moved from $BASE's"
  if [ "$HAVE_NM" = 1 ]; then
    symbols "$so" > "$W/graft-symbols.txt"
    if ! cmp -s "$W/base-symbols.txt" "$W/graft-symbols.txt"; then
      if [ "${KS_ALLOW_SYMBOL_DELTA:-}" = "1" ]; then
        echo "WARN  the exported symbols of the new .so differ from the base's (KS_ALLOW_SYMBOL_DELTA=1):"
        diff "$W/base-symbols.txt" "$W/graft-symbols.txt" | head -20 || true
      else
        echo "FAIL: the exported symbols of the new .so differ from $BASE's (the patch adds no function):" >&2
        diff "$W/base-symbols.txt" "$W/graft-symbols.txt" | head -20 >&2 || true
        exit 1
      fi
    else
      echo "ok    the $(wc -l < "$W/graft-symbols.txt") exported symbols equal the base's"
    fi
  fi
  # The directory is the base's with one file replaced and three added: _ttnn.so and every other op directory identical, sdpa/ differing by the three kernels only.
  cmp -s "$BASE/_ttnn.so" "$G/_ttnn.so" || fail "$G/_ttnn.so differs from $BASE/_ttnn.so"
  for op in attn_prep nlp_concat_heads_decode sdpa_decode; do
    diff -r "$BASE/$op" "$G/$op" >/dev/null || fail "$G/$op differs from $BASE/$op"
  done
  sdpa_delta=$(diff -rq "$BASE/sdpa" "$G/sdpa" | sed "s#^Only in $G/sdpa/device/kernels/##" | sort | tr '\n' ' ')
  want_delta=$(printf 'dataflow: reader_interleaved_qwen_ksplit.cpp\ndataflow: writer_interleaved_qwen_ksplit.cpp\ncompute: sdpa_qwen_ksplit.cpp\n' | sort | tr '\n' ' ')
  test "$sdpa_delta" = "$want_delta" || fail "$G/sdpa differs from $BASE/sdpa by more than the three ksplit kernels: $sdpa_delta"
  echo "ok    _ttnn.so, attn_prep, nlp_concat_heads_decode and sdpa_decode identical to $BASE; sdpa/ differs by the three ksplit kernels only"
  size_base=$(stat -c %s "$BASE/_ttnncpp.so")
  size_new=$(stat -c %s "$so")
  echo "ok    _ttnncpp.so size $size_base -> $size_new ($((size_new - size_base)) bytes)"
  for image in $IMAGES; do
    cid=$(docker create --network none --entrypoint true "$image")
    docker cp -L "$cid:/opt/tt-metal/build_Release/lib/_ttnncpp.so" "$W/image-ttnncpp.so"
    docker rm "$cid" >/dev/null
    qwen_strings "$W/image-ttnncpp.so" > "$W/image-${image//[^A-Za-z0-9]/_}-qwen-strings.txt"
    image_lost=$(comm -23 "$W/image-${image//[^A-Za-z0-9]/_}-qwen-strings.txt" "$W/graft-qwen-strings.txt")
    if [ -n "$image_lost" ]; then
      echo "FAIL: QWEN strings of image $image's _ttnncpp.so missing from the new .so:" >&2
      echo "$image_lost" >&2
      exit 1
    fi
    echo "ok    every QWEN string of image $image's _ttnncpp.so is in the new .so"
    rm -f "$W/image-ttnncpp.so"
  done
fi
if [ "$DRY" = 1 ]; then
  echo "### run: (cd $(printf '%q' "$G") && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum) > $(printf '%q' "$G/MANIFEST.sha256")"
else
  (cd "$G" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum) > "$G/MANIFEST.sha256"
  # The manifest differs from the base's in exactly four paths: ./_ttnncpp.so changed, the three kernels added.
  changed=$(diff <(sed 's/^[0-9a-f]\{64\}  //' "$BASE/MANIFEST.sha256" | sort) <(sed 's/^[0-9a-f]\{64\}  //' "$G/MANIFEST.sha256" | sort) | grep '^[<>]' | sort | tr '\n' ' ' || true)
  want_changed=$(printf '> ./sdpa/device/kernels/compute/sdpa_qwen_ksplit.cpp\n> ./sdpa/device/kernels/dataflow/reader_interleaved_qwen_ksplit.cpp\n> ./sdpa/device/kernels/dataflow/writer_interleaved_qwen_ksplit.cpp\n' | sort | tr '\n' ' ')
  test "$changed" = "$want_changed" || fail "the new manifest lists other files than the base's plus the three ksplit kernels: $changed"
  differing=$(join -j 2 <(sort -k 2 "$BASE/MANIFEST.sha256") <(sort -k 2 "$G/MANIFEST.sha256") | awk '$2 != $3 {print $1}')
  test "$differing" = "./_ttnncpp.so" || fail "the new manifest differs from the base's in more than ./_ttnncpp.so (and the added kernels): $differing"
  echo "ok    MANIFEST.sha256 differs from the base's in ./_ttnncpp.so and the three added kernels only"
fi
x mv "$G" "$GRAFT"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing built"
  exit 0
fi
so_sha=$(hsha "$GRAFT/_ttnncpp.so")
echo "### summary"
echo "ttbuild: sources restored (decode 3e0a69af, prefill fd8c0676, touched) and build_Release rebuilt from them"
sha256sum "$GRAFT/_ttnncpp.so" "$BASE/_ttnncpp.so"
echo "graft files: $(wc -l < "$GRAFT/MANIFEST.sha256") (manifest $GRAFT/MANIFEST.sha256); $BASE was not touched"
if [ -n "${RESULTS:-}" ]; then
  {
    mkdir -p "$RESULTS"
    cp "$GRAFT/MANIFEST.sha256" "$RESULTS/k64j-oq-ks-MANIFEST.sha256"
    cp "$W"/*-qwen-strings.txt "$RESULTS/"
    [ ! -e "$W/base-symbols.txt" ] || cp "$W/base-symbols.txt" "$W/graft-symbols.txt" "$RESULTS/"
    printf 'graft=%s\nbase=%s\nK64J_OQ_KS_TTNNCPP_SHA256=%s\nK64J_OQ_TTNNCPP_SHA256=%s\nreproduced=%s\nwork=%s\n' \
      "$GRAFT" "$BASE" "$so_sha" "$base_so" "$([ "$a_sha" = "$base_so" ] && echo yes || echo no)" "$W" > "$RESULTS/k64j-oq-ks-build-summary.txt"
  } || echo "WARN  could not copy the manifest and strings lists into $RESULTS" >&2
fi
echo "next: the K-split card job (run_card_m_ks.sh) mounts $GRAFT and checks its _ttnncpp.so sha256 $so_sha (EXPECT_TTNNCPP_SHA256)"
echo "K64J_OQ_KS_TTNNCPP_SHA256=$so_sha"
echo "### build_k64j_oq_ks done $(date -Is)"
