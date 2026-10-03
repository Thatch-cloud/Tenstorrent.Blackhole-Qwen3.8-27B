#!/usr/bin/env bash
# K64j K1-K4 (c2-serve-for-real-plan.md section 2.3): run k64j_card_b.py on the card QUAL_CARD names (no default;
# card B is refused) in the C2 gate image (P8) with graft K64j mounted exactly as lever_n_m3native_run_arm.sh mounts
# a KOPGRAFT64 graft: _ttnn.so, _ttnncpp.so (both paths), attn_prep, nlp_concat_heads_decode, sdpa_decode and sdpa.
# No build, no weights: the harness runs the runtime-extent (0x20) programs against K64i's compile-time served
# programs of the same binary. Queue it through the cardm action of qwen-c2-serving.yml (card M; see below) or run
# it by hand on the rig. Every run names its card: the examples below take QUAL_CARD and ALLOW_SERVING_CARD as
# "ON CARD M" sets them (there is no default target; card B is reserved for another project and refused).
#
#   WATCHER=1 bash run_card_b.sh         # THE FIRST HARDWARE PASS of the 0x20 kernels: TT_METAL_WATCHER=5,
#                                         # extents 2,304 and 33,024, seed 0, G4B3 0x21 / 0x23 and G8B2 0x27, 8 trace
#                                         # families, no timing, a per-call watchdog (WATCHDOG_S, default 120 s), a
#                                         # 2,700 s container timeout; the watcher's own log lands in
#                                         # $RESULTS/watcher-<stamp>/
#   bash run_card_b.sh                    # every section (N, X, M, K, L, T) at K1's six extents, seeds 0-2, every
#                                         # combo, and the timing; a 5,400 s container timeout, --deadline-s 600 s
#                                         # before it; CARD_B_ARGS="--seeds 0,1,2,3,4" is K1's full seed list
#   K64J_CARD_DRY_RUN=1 bash run_card_b.sh   # print the launch argv and exit: no node resolution, no holder check,
#                                         # no docker, nothing launched (the graft is checked when it exists)
#
# CB2a (s2-design.md W10a, 6.2: K2, X7 and Z, opt-in by --sections), watcher pass first:
#   WATCHER=1 CARD_B_ARGS="--sections K2,X7,Z" bash run_card_b.sh
#                                         # the watcher defaults above plus CB2a's reduced set: K2 tickets 232..263
#                                         # (families 256 and 512 and the cap), floor 120..127, families 2,304 and
#                                         # 131,328 at +0 / +240 / +255, Z at 256 / 512 / 2,304 / 3,840; all K2 rows
#                                         # equal reads k2_verdict=REDUCED-PASS k2_coverage=38/1980 (never the
#                                         # policy's PASS); a K2 FAIL decides at any coverage
#   CARD_B_ARGS="--sections K2,X7,Z --seeds 0,1,2,3,4 --variants normal,peaky --no-timing" bash run_card_b.sh
#                                         # the full pass: K2 every ticket 128..300 and the five families, X7, and Z
#                                         # at all 15 stale-writer families; K2 decides the exactness policy
#                                         # (k2_verdict=PASS only with k2_coverage=1980/1980)
#
# ONE KV HEAD PER CHIP (the four-card S2 evidence, CB1-TP4 and CB2a-TP4; K64J_HARNESS=card only): CARD_B_ARGS="--kv-heads 1
# ..." runs k64j_card_b.py at 6 query heads on ONE KV head (the four-card chip's geometry; the default, 2, is the pair's). The
# harness then defaults to the combos G4B3 / G8B2 at 0x21 / 0x23 (0x27 and 0x2F need a second KV head) and the trace combos
# G4B3 0x21 / G8B2 0x23, and the WATCHER=1 pass below runs those instead of G8B2 0x27; everything else is unchanged. The
# reports say kv_heads=1 and the verdict line 'kv_heads=1'. The jobs (scripts/ci/references/tp4-s2-serve-jobs):
#   EV-W1  WATCHER=1, --kv-heads 1 --sections N,X,M,K,L,T,K2,X7,Z --seeds 0 --extents 2304,131328 --variants normal,peaky
#          --no-timing                       (reduced scope: a REDUCED-PASS, the go/no-go and the watcher pass, never evidence)
#   EV-F1  --kv-heads 1 --seeds 0,1,2,3,4    (CB1-TP4: the default sections N,X,M,K,L,T at K1's six extents and starts)
#   EV-F2  --kv-heads 1 --sections K2,X7,Z --seeds 0,1,2,3,4 --variants normal,peaky --no-timing   (CB2a-TP4: 1980 tickets)
#
# ON CARD M (card B is reserved for another project): the same runner, by hand on the rig or through the cardm
# action, which sets both variables itself. Two variables select card M, both required:
#   QUAL_CARD=blackhole-CEF5729692C19E6D   card M's board id (the target is resolved by board id, never a node number)
#   ALLOW_SERVING_CARD=1                   lifts the serving-pair refusal, with a loud warning
# e.g. QUAL_CARD=blackhole-CEF5729692C19E6D ALLOW_SERVING_CARD=1 WATCHER=1 CARD_B_ARGS="--sections K2,X7,Z"
#      KOPGRAFT64=... EXPECT_TTNNCPP_SHA256=... bash run_card_b.sh
# Nothing else changes: the node resolved by board id and rechecked right before the launch, one --device (card M's
# node), the holder check (fuser on card M's and card A's nodes, card M being a serving card; and a container that
# can reach either refuses the launch - a --privileged one, one with /dev or /dev/tenstorrent bound, a CI gate arm -
# while one given only card B's node is not in the way), the per-call watchdog, the container timeout and the deadline. Results go to ~/kwork64/k64j/card-m, the container
# is qwen-k64j-card-card-m, and a hang's hint resets card M and card A TOGETHER (the Ethernet-linked pair). Check gh
# run list for the qwen-two-p150a-exclusive group before and after.
#
# CB2b (s2-design.md W10b, 6.2: R1, S, R2 and R4 through the REAL S2 extent readers): K64J_HARNESS=extent_reader runs
# extent_reader_card_b.py instead of k64j_card_b.py (K64J_HARNESS=card, the default). This checkout's scripts/ci,
# the code under test, is mounted read-only at /bench/ci, and QWEN_FAST_SDPA_MODES=tail,share,slice is set (the
# image's; the reader adds extent). The pinned sources the reader runs (attention_mask_replay.py/.cpp and
# attention_fold_dma.py/.cpp) are NEVER taken from the checkout. The harness loads them from the image's served tree,
# $SERVED_CI, as serving does. Before it opens the device, it refuses any whose sha256 is not the served one. The
# image's attention_mask_replay.py is the frozen recipe's stage (6a31981c), not the checkout's 3e431742. The container
# logs their sha256 with the binaries'. Report reader-<stamp>.json, container qwen-k64j-reader-<card tag>, verdict
# line K64J_READER. Everything else (the graft checks, the board, the holder check, the watchdog, the timeouts) is as
# above. CARD B IS RESERVED for another project (refused; there is no default QUAL_CARD), so CB2b runs on card M only:
# through the cardm action (.github/c2-serving-job.env), which sets card M itself -
#   C2_CARDM_HARNESS=optimisation/ttnn-op/k64j/run_card_b.sh
#   C2_CARDM_ENV=K64J_HARNESS=extent_reader [WATCHER=1] KOPGRAFT64=/home/thatch/opgraft-K64j
#     EXPECT_TTNNCPP_SHA256=<K64J_TTNNCPP_SHA256>
#   C2_CARDM_ARGS as CARD_B_ARGS below
# - or by hand with card M named, as every example here names it (never QUAL_CARD unset). Watcher pass first:
#   QUAL_CARD=blackhole-CEF5729692C19E6D ALLOW_SERVING_CARD=1 K64J_HARNESS=extent_reader WATCHER=1 bash run_card_b.sh
#                                         # seed 0, peaky; R1 at words 0 / 32 / 255 on G8B2, G4B3 and G4B1; S; R2 at
#                                         # the five named families (two assignments, one replay each, with the
#                                         # tables); R4 with segments 2 and 3 idle (starts 0 and 32) at the first
#                                         # assignment and at the one holding C; reads scope=reduced (never CB2b's
#                                         # evidence)
#   QUAL_CARD=blackhole-CEF5729692C19E6D ALLOW_SERVING_CARD=1 K64J_HARNESS=extent_reader bash run_card_b.sh
#                                         # the full pass, the harness's defaults (CARD_B_ARGS="--sections R1,S,R2,R4
#                                         # --seeds 0,1,2 --variants normal,peaky"): 56 families (more than 50)
#                                         # restaged twice each, four idle patterns; K64J_READER verdict=PASS
#                                         # scope=full is CB2b's evidence
#
# CB2b AT FOUR CARDS (the S2 fast path on the four-card mesh; the evidence jobs EV-W2 and EV-F3): K64J_HARNESS=extent_reader
# TP4_WIDTH=4 runs the same harness with --width 4 (TP4_WIDTH unset or 2 is the pair's run above, unchanged) and sets QWEN_FAST_TP=4
# and QWEN_FAST_SDPA_MODES=tail,share (the four-card profiles' modes, not the pair's tail,share,slice) in the container. The code
# under test is then also scripts/ci's extent_attention_replay_tp.py, tp_shapes.py, tp_kernels.py,
# tp_addresses.py and chip_view.py (/bench/ci), driven through ChipView(chips=4) at flags 0x23 on ONE KV head (six query heads per
# token); the pinned pair modules and the twin's siblings (attention_mask_replay_tp.py/.cpp, attention_fold_dma_tp.py/.cpp) come
# from the image's served tree, so IMAGE must be the four-card S2 image (the P8 default has no such siblings):
#   C2_CARDM_ENV=K64J_HARNESS=extent_reader TP4_WIDTH=4 [WATCHER=1] IMAGE=<the four-card image> KOPGRAFT64=... EXPECT_TTNNCPP_SHA256=...
#   C2_CARDM_ARGS=--sections R1,S,R2,R4 --seeds 0,1,2 --variants normal,peaky    (the full pass; the watcher pass: --seeds 0 --variants normal)
# Verdict line: K64J_READER verdict=PASS scope=full ... chips=1of4 ... extent_sha256=<extent_attention_replay_tp.py's sha256>.
#
# THE FOUR-CARD PORT'S ONE-CARD WINDOW (S2T-01 / S2T-10, HW-A): two more harnesses, both on card M through the cardm action
# with the graft K64j mounted and verified as above (QUAL_CARD, ALLOW_SERVING_CARD and RESULTS are set by the step):
#   K64J_HARNESS=nkv1_spike   k64j_nkv1_spike.py: K64j at ONE KV head per chip (6 query heads on 1 KV head, cache
#                             (N, 1, 64, 256)): the extent program 0x23 at G8B2 and 0x21 at G16B1 against its compile-time twin
#                             (bit-equal on the poisoned table), the legacy numerics, the factory lines and timings.
#                             Verdict line K64J_NKV1. WATCHER=1 first: one family, two starts, no timing.
#   K64J_HARNESS=gdn_tp4      scripts/ci/gdn_tp4_card_test.py (this checkout's scripts/ci at /bench/ci): the packed GDN pieces
#                             at the four-card widths on this ONE card with QWEN_FAST_TP=4 - user-batch and K5 at 12 value
#                             heads (against the per-user launch and the pinned pair launch's first 12 heads), the native
#                             twin, and the DMA sibling kernels at 192 / 80 pages. Verdict line GDN_TP4 (scope full needs all
#                             four sections and three seeds). TP4_WIDTH=2 runs its PARITY section instead (QWEN_FAST_TP
#                             unset: the pinned DMA kernels against their _tp siblings at the pair's page counts):
#                               C2_CARDM_ENV=K64J_HARNESS=gdn_tp4 TP4_WIDTH=2 ... C2_CARDM_ARGS=--sections PARITY
#   K64J_HARNESS=ordered_writer  scripts/ci/ordered_writer_tp4_card_test.py (E1, the 262k window's first card evidence): the four-card
#                             ordered K/V writers - the packed block's chained launch and the engines' 32-row tile - at page-table
#                             widths 2,052 (control) and 4,096 on this ONE card with QWEN_FAST_TP=4, eager and under trace replay
#                             (table rewritten in place, and untouched), the complete cache read back against the host prediction.
#                             Verdict line ORDERED_WRITER (scope full needs both writers, both widths and seeds 0,1,2). QWEN_FAST_TP=4
#                             is set by the script; TP4_WIDTH is not read.
# Before anything is launched the graft must verify against its MANIFEST.sha256, its _ttnncpp.so must be
# EXPECT_TTNNCPP_SHA256 (REQUIRED for a real run: K64j's sha is the K64J_TTNNCPP_SHA256 line build_k64j.sh prints)
# and carry the F22 literal '[QWEN-SDPA] runtime-extent entries=', its four qwen decode kernels must be K64j's
# (make_k64j_kernels.OUTPUTS) and its five STOCK decode sources the stock ones, so a wrong graft fails before it
# costs card time. Inside the container the harness checks the mapped binary and the kernels again.
#
# The container timeout SIGTERMs `docker run`, which forwards it to the harness (PID 1 in the container, by exec):
# the harness writes its partial report at once and unwinds. A 124 exit prints the hang hint below unless the log
# shows the harness finished that unwinding ('ERROR terminated by signal', printed after the device closed).
#
# Env: KOPGRAFT64 (default ~/opgraft-K64j), EXPECT_TTNNCPP_SHA256, IMAGE (default image P8), RESULTS
# (~/kwork64/k64j/<card tag>), CARD_B_ARGS (extra harness args, appended last, e.g. "--sections X,M --seeds 0"),
# WATCHER=1, WATCHDOG_S, K64J_CARD_DRY_RUN=1, K64J_HARNESS (card, the default, extent_reader, nkv1_spike, gdn_tp4 or ordered_writer; the
# last three are the four-card port's one-card window, see below; TP4_WIDTH=2 is gdn_tp4's PARITY run; TP4_WIDTH=4 is extent_reader's
# four-card run, unset being its pair run), QUAL_CARD (the
# target board id under /dev/tenstorrent/by-id; required, no default: card B is reserved for another project and refused; card M or card A, the
# serving pair, is refused unless ALLOW_SERVING_CARD=1, which prints a loud warning). QWEN_SDPA_TREE_SCRATCH_ROUNDS=1
# is set as the arm sets it. Every run gets a fresh kernel cache.
#
# ON A HANG (the WATCHDOG line and exit 3; exit 1 with 'Timeout (' in the log; or the timeout's 124 / 137): the EXIT
# trap removes the container. Then reset THE TARGET CARD ONLY with the hint printed below. This script never resets.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
probe_dir=$(cd "$here/../k64j_probe" 2>/dev/null && pwd || echo "$here/../k64j_probe")
qwen=$(cd "$here/../sdpa_decode_qwen" 2>/dev/null && pwd || echo "$here/../sdpa_decode_qwen")
G=${KOPGRAFT64:-$HOME/opgraft-K64j}
IMAGE=${IMAGE:-sha256:57cb699489436842d7e7bdd5ab917d509b4492fc95f6f083ef7d2865b488c2ef}
EXPECT=${EXPECT_TTNNCPP_SHA256:-}
EXTENT_MARKER='[QWEN-SDPA] runtime-extent entries='
# The K64j qwen kernels (make_k64j_kernels.OUTPUTS; k64j_card_b.K64J_KERNELS).
READER_QWEN=adb6091878ba3f0a0805846ff56f05352610d7fe779b5ae96320c437c095db49
READER_SLICE=518d8096e3cceb160eaef8ab4f0ae976ccbffd3904d31176b7f9d02828c37f8a
COMPUTE_QWEN=409a1aafc3ffaaca2c6afba0e999525b7d141491f0a52e5583efa70447ee5c0e
WRITER_SLICE=642c36f809be0f1ad1664deb405dc310dabaa32d5628710320a118eb37a6cc7a
# The stock decode sources (tt-metal 9f9cd4fd; probe_k64j_card_b.STOCK_KERNELS, the vendored ../k64j_probe/fixtures).
READER_ALL=49a05926b437e2ca90d7e01c60e85a6b11f333375a6159fa02c9ff6f78af764e
WRITER_ALL=734c90c01c7a7174497133fae9df80110ead55275955faeb566d345bdccb60b8
COMPUTE_ALL=d24769bdcbb8635f83f5f91a301fe0d89298d38263d4493a39c6d2decb57867f
DATAFLOW_COMMON=e4623a2254559eaec4450ebfab0f9c5732e02acfe4d8126bd5eeb7efe0fdc608
RT_ARGS_COMMON=1b52c60d78ada6f08effd326c2ed2407b3a74cf0db2353fadbe51b088610aec8
# K64J_HARNESS=extent_reader: the code under test it loads from this checkout's scripts/ci, and the pinned sources it
# runs from the image's served tree (extent_reader_card_b.SERVED_ROOT and PINNED; the harness checks their bytes).
CI_SOURCES='extent_attention_replay.py pooled_attention_replay.py serving_buffer_pool.py attention_head_fold.py
gdn_multitoken_conv.py'
SERVED_CI=/experiment-scripts/ci
SERVED_SOURCES='attention_mask_replay.py attention_mask_replay.cpp attention_fold_dma.py attention_fold_dma.cpp'
# ... and at TP4_WIDTH=4 (extent_reader_card_b.QUAD_RECORDED_SOURCES / QUAD_SIBLINGS): the four-card twin and its helpers from the
# checkout, the twin's sibling kernels and modules from the image.
QUAD_CI_SOURCES='extent_attention_replay_tp.py tp_shapes.py tp_kernels.py tp_addresses.py chip_view.py'
QUAD_SERVED_SOURCES='attention_mask_replay_tp.py attention_mask_replay_tp.cpp attention_fold_dma_tp.py attention_fold_dma_tp.cpp'

# >>> qual_card.sh: which board a qualification harness runs on (canonical copy scripts/ci/qual_card.sh)
# Every single-card harness under optimisation/ttnn-op embeds this block byte for byte (the scripts in
# scripts/ci source the file); scripts/ci/test_qual_card.py fails when a copy drifts, and
# `py -3.11 -B scripts/ci/test_qual_card.py --sync` re-copies the canonical text into every harness.
#
# The rig has three p150a. Card M (blackhole-CEF5729692C19E6D) and card A (blackhole-3707293C249A5E67)
# are the serving pair: Ethernet-linked, mounted by every CI gate arm (lever_n_m3native_run_arm.sh), and
# reset together by the gate. Card B (blackhole-F36F768B9A5CAFA0) is reserved for another project: no
# harness may select it, so there is no default target, and nothing here opens, maps, resets or waits
# on it. /dev/tenstorrent/N numbers change across resets and switch power-cycles, and tt-smi's own board
# index is a different numbering again, so nothing here hard-codes either: the target is a board id, its
# node is resolved with readlink -f at launch and again right before the container starts, and the
# reset hint prints commands that resolve the board id when they are run (never a bare index) plus the
# PCI address that identifies the board's row in tt-smi -ls.
#
# The qwen-* hardware workflows in the qwen-two-p150a-exclusive group act on card M and card A: a run on
# either must not overlap them - check gh run list for that group before and during it.
#
#   QUAL_CARD=<board id>    the target, a name under /dev/tenstorrent/by-id (required: there is no default)
#   ALLOW_SERVING_CARD=1    required to target card M or card A (half of the serving pair); loud warning
#
#   qual_card_select      sets QUAL_CARD, QUAL_BYID, QUAL_TAG, QUAL_SERVING; refuses an unset QUAL_CARD,
#                         card B, and a serving card without the override. Touches no device: dry runs
#                         call it too.
#   qual_card_resolve     sets QUAL_NODE (readlink -f, now) and QUAL_PCI; refuses a missing board and a
#                         board id that resolves to card B's node, and treats one that resolves to a
#                         serving card's node as that serving card.
#   qual_refuse_holders   refuses while a container or a host process can reach the target (for a serving
#                         target, either end of the pair).
#   qual_card_recheck     readlink -f again right before the container starts; refuses if the node moved.
#   qual_reset_hint       the recovery lines after a hang, on stdout; it never resets anything itself.
QUAL_RESERVED_CARD=blackhole-F36F768B9A5CAFA0
QUAL_SERVING_CARDS='blackhole-CEF5729692C19E6D blackhole-3707293C249A5E67'
QUAL_TT_ROOT=/dev/tenstorrent
QUAL_BYID_ROOT=$QUAL_TT_ROOT/by-id
QUAL_SYS_ROOT=/sys

qual_card_label() {
  case ${1:-$QUAL_CARD} in
    blackhole-CEF5729692C19E6D) echo 'card M, half of the serving pair' ;;
    blackhole-3707293C249A5E67) echo 'card A, half of the serving pair' ;;
    "$QUAL_RESERVED_CARD") echo 'card B, reserved for another project' ;;
    *) echo 'a board this harness does not name' ;;
  esac
}

qual_card_select() {
  if [ -z "${QUAL_CARD:-}" ]; then
    echo "refusing: QUAL_CARD is not set; name the target's board id under $QUAL_BYID_ROOT. There is no default:" >&2
    echo "  card B is reserved for another project, and card M or card A needs ALLOW_SERVING_CARD=1." >&2
    exit 1
  fi
  case $QUAL_CARD in
    .*|*/*|*[!A-Za-z0-9._-]*)
      echo "refusing: QUAL_CARD=$QUAL_CARD is not a board id under $QUAL_BYID_ROOT" >&2
      exit 1 ;;
  esac
  if [ "$QUAL_CARD" = "$QUAL_RESERVED_CARD" ]; then
    echo "refusing: QUAL_CARD=$QUAL_CARD is $(qual_card_label); no qualification harness may use it." >&2
    echo "  Run on card M instead (QUAL_CARD=card M's board id, ALLOW_SERVING_CARD=1: qwen-c2-serving.yml's cardm)." >&2
    exit 1
  fi
  QUAL_BYID=$QUAL_BYID_ROOT/$QUAL_CARD
  case $QUAL_CARD in
    blackhole-CEF5729692C19E6D) QUAL_TAG=card-m ;;
    blackhole-3707293C249A5E67) QUAL_TAG=card-a ;;
    *) QUAL_TAG=$QUAL_CARD ;;
  esac
  QUAL_SERVING=0
  case " $QUAL_SERVING_CARDS " in
    *" $QUAL_CARD "*) qual_serving_override ;;
  esac
}

qual_serving_override() {
  QUAL_SERVING=1
  if [ "${ALLOW_SERVING_CARD:-0}" != 1 ]; then
    echo "refusing: QUAL_CARD=$QUAL_CARD is $(qual_card_label), which the CI gate and the endpoint use." >&2
    echo "  ALLOW_SERVING_CARD=1 overrides (card B is reserved for another project: it is never an alternative)." >&2
    exit 1
  fi
  echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' >&2
  echo "!!! WARNING: ALLOW_SERVING_CARD=1: this run is on $QUAL_CARD, $(qual_card_label)." >&2
  echo '!!! A CI gate that starts meanwhile fails its holder check, or resets the pair under this run before' >&2
  echo '!!! it opens the card; a hang here keeps the pair down until card M and card A are reset together.' >&2
  echo '!!! Check gh run list for the qwen-two-p150a-exclusive group before and after.' >&2
  echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' >&2
}

qual_is_char() {
  test -c "$1"
}

qual_card_resolve() {
  local card node
  QUAL_NODE=$(readlink -f -- "$QUAL_BYID" 2>/dev/null || true)
  if [ -z "$QUAL_NODE" ] || ! qual_is_char "$QUAL_NODE"; then
    echo "refusing: $QUAL_CARD ($(qual_card_label)) has no device node here: $QUAL_BYID does not resolve to one" >&2
    exit 1
  fi
  node=$(readlink -f -- "$QUAL_BYID_ROOT/$QUAL_RESERVED_CARD" 2>/dev/null || true)
  if [ -n "$node" ] && [ "$node" = "$QUAL_NODE" ]; then
    echo "refusing: $QUAL_CARD resolves to the node of $(qual_card_label "$QUAL_RESERVED_CARD")" >&2
    exit 1
  fi
  if [ "$QUAL_SERVING" != 1 ]; then
    for card in $QUAL_SERVING_CARDS; do
      node=$(readlink -f -- "$QUAL_BYID_ROOT/$card" 2>/dev/null || true)
      if [ "$node" = "$QUAL_NODE" ]; then
        echo "### $QUAL_CARD resolves to $node, the node of $card ($(qual_card_label "$card"))" >&2
        qual_serving_override
      fi
    done
  fi
  QUAL_PCI=$(qual_pci_of "$QUAL_NODE")
  echo "### target card: $QUAL_CARD ($(qual_card_label)) -> $QUAL_NODE, PCI ${QUAL_PCI:-unknown}"
}

# A path's device numbers (major:minor, hex, following symlinks), or nothing.
qual_majmin_of() {
  stat -L -c '%t:%T' -- "$1" 2>/dev/null || true
}

# A device node's PCI address (0000:f4:00.0) from sysfs, or nothing.
qual_pci_of() {
  local majmin dev
  majmin=$(qual_majmin_of "${1:-}")
  case $majmin in
    *[!0-9a-f:]*|:*|*:|*:*:*) return 0 ;;
    *:*) ;;
    *) return 0 ;;
  esac
  dev=$(readlink -e -- "$QUAL_SYS_ROOT/dev/char/$((16#${majmin%%:*})):$((16#${majmin##*:}))/device" 2>/dev/null) || return 0
  case ${dev##*/} in
    [0-9a-f][0-9a-f][0-9a-f][0-9a-f]:[0-9a-f][0-9a-f]:[0-9a-f][0-9a-f].[0-7]) echo "${dev##*/}" ;;
  esac
}

# The nodes nothing else may be able to reach while the target runs (qual_refuse_holders sets them): the
# target's, and for a serving target its Ethernet partner's as well (a program on either end of the pair
# reaches the other across the link), each with what it is and its device numbers. Any other board's node
# (card B's included) is never among them. A serving target whose partner has no node here is refused:
# no container's devices could then be told apart from the partner's.
QUAL_GUARD_NODES=()
QUAL_GUARD_WHAT=()
QUAL_GUARD_MAJMIN=()

qual_guard_set() {
  local card node
  QUAL_GUARD_NODES=("$QUAL_NODE")
  QUAL_GUARD_WHAT=('the target')
  if [ "$QUAL_SERVING" = 1 ]; then
    for card in $QUAL_SERVING_CARDS; do
      node=$(readlink -f -- "$QUAL_BYID_ROOT/$card" 2>/dev/null || true)
      [ "$node" != "$QUAL_NODE" ] || continue
      if [ -z "$node" ] || ! qual_is_char "$node"; then
        echo "refusing: $card ($(qual_card_label "$card")), the target's Ethernet partner, has no device node here:" >&2
        echo "  a container's devices cannot be told apart from it" >&2
        exit 1
      fi
      QUAL_GUARD_NODES+=("$node")
      QUAL_GUARD_WHAT+=("$(qual_card_label "$card"), the target's Ethernet partner")
    done
  fi
  QUAL_GUARD_MAJMIN=()
  for node in "${QUAL_GUARD_NODES[@]}"; do
    QUAL_GUARD_MAJMIN+=("$(qual_majmin_of "$node")")
  done
}

# Why a container (qual_refuse_holders' docker inspect lines) can reach a guarded node; nothing when it
# cannot. It names: --privileged; any device cgroup rule; a device request naming Tenstorrent (CDI); a
# device or a mount that is a guarded node, a directory holding one (/dev, /dev/tenstorrent - every CI
# gate arm mounts /dev/tenstorrent read-only for its board-mapping check, beside card M and card A), or
# the host's root; a device with a guarded node's numbers; a /dev/tenstorrent path that is not a device
# node now (which board it was given is unknowable). Anything else - a container given only card B's node,
# or any other board's - cannot reach the target and is not looked at further. With no guarded node set,
# every container is refused.
qual_container_reach() {
  local head=${1%%$'\n'*} line kind path resolved i
  if [ "${#QUAL_GUARD_NODES[@]}" = 0 ]; then
    echo 'was inspected before any target node was set (qual_guard_set)'; return 0
  fi
  case $head in
    *' true') echo 'is --privileged (it can open every device)'; return 0 ;;
  esac
  while IFS= read -r line; do
    kind=${line%% *}
    path=${line#* }
    case $kind in
      rule) echo "has a device cgroup rule ($path)"; return 0 ;;
      req)
        case $path in
          *[Tt]enstorrent*) echo "has a device request for a Tenstorrent device ($path)"; return 0 ;;
        esac
        continue ;;
      dev|mnt) ;;
      *) continue ;;
    esac
    resolved=$(readlink -f -- "$path" 2>/dev/null || true)
    if [ "$resolved" = / ]; then
      echo "has $path among its ${kind}s, the host's root (it holds every device node)"; return 0
    fi
    for i in "${!QUAL_GUARD_NODES[@]}"; do
      if [ "$resolved" = "${QUAL_GUARD_NODES[$i]}" ]; then
        echo "has $path among its ${kind}s, which is ${QUAL_GUARD_NODES[$i]} (${QUAL_GUARD_WHAT[$i]})"; return 0
      fi
      if [ -n "$resolved" ]; then
        case ${QUAL_GUARD_NODES[$i]} in
          "$resolved"/*)
            echo "has $path among its ${kind}s, a directory holding ${QUAL_GUARD_NODES[$i]} (${QUAL_GUARD_WHAT[$i]})"
            return 0 ;;
        esac
      fi
      if [ "$kind" = dev ] && [ -n "${QUAL_GUARD_MAJMIN[$i]}" ] \
          && [ "$(qual_majmin_of "$path")" = "${QUAL_GUARD_MAJMIN[$i]}" ]; then
        echo "is given $path, a device with the numbers of ${QUAL_GUARD_NODES[$i]} (${QUAL_GUARD_WHAT[$i]}," \
          "${QUAL_GUARD_MAJMIN[$i]})"
        return 0
      fi
    done
    case $path in
      "$QUAL_TT_ROOT"|"$QUAL_TT_ROOT"/*)
        if [ -z "$resolved" ] || ! qual_is_char "$resolved"; then
          echo "has $path among its ${kind}s, which is not a device node now (it may be the target)"; return 0
        fi ;;
    esac
  done <<< "${1#*$'\n'}"
}

# Refuses while anything else can reach the target. Containers: qual_container_reach, against the guarded
# nodes (qual_guard_set). Host processes: fuser on the guarded nodes (with sudo -n when that works, else
# this user's processes, said so) - which also sees a process in a container holding one, however the
# container was given it - up to five tries two seconds apart (the rig's telemetry exporter holds every
# card for a moment every 30 s), refusing while any holder persists.
qual_refuse_holders() {
  local id info why st out try scope
  local pre=()
  qual_guard_set
  for id in $(docker ps -q); do
    info=$(docker inspect "$id" --format '{{.Name}} {{.HostConfig.Privileged}}{{println}}{{range .HostConfig.Devices}}dev {{println .PathOnHost}}{{end}}{{range .HostConfig.DeviceCgroupRules}}rule {{println .}}{{end}}{{range .HostConfig.DeviceRequests}}req {{.Driver}} {{println .DeviceIDs}}{{end}}{{range .Mounts}}mnt {{println .Source}}{{end}}') || continue
    why=$(qual_container_reach "$info")
    if [ -n "$why" ]; then
      echo "refusing: container ${info%% *} $why" >&2
      exit 1
    fi
  done
  echo "### containers: none can reach ${QUAL_GUARD_NODES[*]} ($QUAL_CARD)"
  if ! command -v fuser >/dev/null 2>&1; then
    echo "WARN: fuser is not installed; host processes holding ${QUAL_GUARD_NODES[*]} were not checked" >&2
    return 0
  fi
  scope="this user's processes only (no passwordless sudo)"
  if [ "$(id -u)" = 0 ]; then
    scope=all
  elif sudo -n true >/dev/null 2>&1; then
    pre=(sudo -n)
    scope=all
  fi
  for try in 1 2 3 4 5; do
    st=0
    out=$(${pre[@]+"${pre[@]}"} fuser -v "${QUAL_GUARD_NODES[@]}" 2>&1) || st=$?
    if [ "$st" != 0 ] && [ -z "$out" ]; then
      echo "### device holders on ${QUAL_GUARD_NODES[*]}: none ($scope)"
      return 0
    fi
    [ "$try" = 5 ] || sleep 2
  done
  echo "refusing: host processes hold ${QUAL_GUARD_NODES[*]} (or fuser failed):" >&2
  echo "$out" >&2
  exit 1
}

# readlink -f again right before the container starts: refuses when a board is not on the node the holder
# check cleared (the boards re-enumerated in between - a switch event, or the gate resetting the pair - so
# the old node may now be another board's). Args: [board id] [node]; the target by default.
qual_card_recheck() {
  local card=${1:-$QUAL_CARD} was=${2:-$QUAL_NODE} now
  now=$(readlink -f -- "$QUAL_BYID_ROOT/$card" 2>/dev/null || true)
  if [ -z "$now" ] || [ "$now" != "$was" ] || ! qual_is_char "$now"; then
    echo "refusing: $card ($(qual_card_label "$card")) was $was at the holder check and is ${now:-gone} now" >&2
    echo "  (the boards re-enumerated); nothing was launched - check the boards, then run again" >&2
    exit 1
  fi
  echo "### $card is still $now"
}

# The recovery lines after a hang. The node and PCI address are looked up again now (nodes renumber),
# and the reset command resolves the board id again when it is run: tt-smi -r takes a /dev/tenstorrent
# path, while a bare number there is tt-smi's own board index, which renumbers too - so none is printed.
# The command runs tt-smi only when readlink -e resolved every board id (an empty argument would make
# tt-smi -r reset every board). The PCI address is what identifies the board's row in tt-smi -ls.
qual_reset_hint() {
  local node pci card n p var vars='m a' cmd= args=
  node=$(readlink -f -- "$QUAL_BYID" 2>/dev/null || true)
  pci=$(qual_pci_of "$node")
  echo "HANG RECOVERY for $QUAL_CARD ($(qual_card_label)), once the container is gone; nothing here resets a card."
  echo "  It is ${node:-absent} now, PCI ${pci:-unknown}; nodes renumber, so the reset below resolves the board id"
  echo "  when it is run. Never a bare number: tt-smi -r reads one as tt-smi's own board index, which renumbers too."
  if [ "$QUAL_SERVING" = 1 ]; then
    for card in $QUAL_SERVING_CARDS; do
      n=$(readlink -f -- "$QUAL_BYID_ROOT/$card" 2>/dev/null || true)
      p=$(qual_pci_of "$n")
      echo "  $card ($(qual_card_label "$card")) is ${n:-absent} now, PCI ${p:-unknown}."
      var=${vars%% *}
      vars=${vars#* }
      cmd="$cmd$var=\$(readlink -e $QUAL_BYID_ROOT/$card) && "
      args="$args \"\$$var\""
    done
    echo "  It is half of the serving pair: one link end reset alone leaves the mesh at 1x1. Check gh run list for the"
    echo "  qwen-two-p150a-exclusive group, confirm both PCI addresses in ~/.local/bin/tt-smi -ls, then reset card M and"
    echo "  card A TOGETHER, in one call:"
    echo "    $cmd~/.local/bin/tt-smi -r$args"
    echo "  then a passing smoke run."
  else
    if [ -n "$pci" ]; then
      echo "  CONFIRM its row (PCI BDF $pci): ~/.local/bin/tt-smi -ls | grep -i '${pci#0000:}'; then reset it alone:"
    else
      echo "  CONFIRM in ~/.local/bin/tt-smi -ls which row is ${node:-this board} (its PCI address is unknown here); then reset it alone:"
    fi
    echo "    n=\$(readlink -e $QUAL_BYID) && ~/.local/bin/tt-smi -r \"\$n\""
    echo "  then a passing smoke run. Never card M or card A (the serving pair), and never card B (reserved)."
  fi
}
# <<< qual_card.sh
qual_card_select
R=${RESULTS:-$HOME/kwork64/k64j/$QUAL_TAG}
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations
KD=$OPS/transformer/sdpa_decode/device/kernels
DRY=${K64J_CARD_DRY_RUN:-0}
MAIN=${K64J_HARNESS:-card}
case $MAIN in
  card) stem=card; verdict_tag=K64J_CARD ;;
  extent_reader) stem=reader; verdict_tag=K64J_READER ;;
  nkv1_spike) stem=nkv1; verdict_tag=K64J_NKV1 ;;
  gdn_tp4) stem=gdn; verdict_tag=GDN_TP4 ;;
  ordered_writer) stem=ordered; verdict_tag=ORDERED_WRITER ;;
  *) echo "refusing: K64J_HARNESS=$MAIN is none of card (k64j_card_b.py), extent_reader" \
       "(extent_reader_card_b.py), nkv1_spike (k64j_nkv1_spike.py), gdn_tp4 (scripts/ci/gdn_tp4_card_test.py) and" \
       "ordered_writer (scripts/ci/ordered_writer_tp4_card_test.py)" >&2
     exit 1 ;;
esac
name=qwen-k64j-$stem-$QUAL_TAG
stamp=$(date +%Y%m%dT%H%M%S)
timeout_s=5400   # N, X (6 combos x 6 extents x 5 starts x 3 seeds), M, K, L, T (two 64-family traces), timing

if [ "$DRY" != 1 ] && [ -z "$EXPECT" ]; then
  echo "refusing: EXPECT_TTNNCPP_SHA256 is required (the K64J_TTNNCPP_SHA256 line build_k64j.sh printed for $G)" >&2
  exit 1
fi

# The target's node, resolved by board id now; never shared: refuse while a container or a host
# process can reach it (a container on another board - a CI gate on the serving pair - does not block).
if [ "$DRY" = 1 ]; then
  node=$QUAL_BYID
else
  qual_card_resolve
  node=$QUAL_NODE
  qual_refuse_holders
fi

HARNESS=("$here/k64j_card_b.py" "$probe_dir/probe_k64j_card_b.py" "$probe_dir/split_model.py"
         "$qwen/test_sdpa_decode_qwen_card_m.py" "$qwen/probe_k1_card_b.py")
if [ "$MAIN" = extent_reader ]; then
  HARNESS+=("$here/extent_reader_card_b.py")
fi
if [ "$MAIN" = nkv1_spike ]; then
  HARNESS+=("$here/k64j_nkv1_spike.py")
fi
for file in "${HARNESS[@]}"; do
  test -s "$file" || { echo "refusing: $file missing (ship k64j, k64j_probe and sdpa_decode_qwen side by side)" >&2; exit 1; }
done
XE=()
WIDTH_ARGS=()
if [ "$MAIN" = extent_reader ]; then
  # The width: the pair by default (the harness's own default), or TP4_WIDTH=4, the four-card reader twin on one KV head. The
  # launch carries it both ways, --width for the harness and QWEN_FAST_TP=4 for tp_shapes, which the harness checks.
  case ${TP4_WIDTH:-2} in
    2) ;;
    4) CI_SOURCES="$CI_SOURCES $QUAD_CI_SOURCES"
       SERVED_SOURCES="$SERVED_SOURCES $QUAD_SERVED_SOURCES"
       WIDTH_ARGS=(--width 4) ;;
    *) echo "refusing: TP4_WIDTH=${TP4_WIDTH} is neither 4 nor 2" >&2; exit 1 ;;
  esac
  # The code under test is this checkout's scripts/ci, mounted read-only. The pinned sources it runs are the image's
  # ($SERVED_CI). The harness loads them from there and checks their served bytes before it opens the device.
  ci=$(cd "$here/../../../scripts/ci" 2>/dev/null && pwd || echo "$here/../../../scripts/ci")
  for file in $CI_SOURCES; do
    test -s "$ci/$file" \
      || { echo "refusing: $ci/$file missing (the extent reader runs this checkout's scripts/ci)" >&2; exit 1; }
  done
  echo "### code under test: $ci; pinned sources: the image's $SERVED_CI"
  if [ "${TP4_WIDTH:-2}" = 4 ]; then
    # The four-card profiles' modes (c2-packed-tp4: tail,share; the slice needs a second KV head), and the width.
    # QWEN_C2_SERVING=0: IMAGE is the served S2 image, whose ENV turns on the C2 boot hook in every python process;
    # the hook applies the default serving profile's mesh (TT_MESH_GRAPH_DESC_PATH = the four-card ring descriptor), and
    # this ONE-card open then fails in tt-metal's topology mapper (run 36786351340). The harness reads the image's served
    # sources by path and needs none of the hook's serving setup.
    XE=(-e QWEN_FAST_SDPA_MODES=tail,share -e QWEN_FAST_TP=4 -e QWEN_C2_SERVING=0)
  else
    XE=(-e QWEN_FAST_SDPA_MODES=tail,share,slice)
  fi
fi
if [ "$MAIN" = gdn_tp4 ]; then
  # The code under test is this checkout's scripts/ci, first on PYTHONPATH; the four-card width is a launch variable
  # (tp_shapes reads it at import): 4 for the qualification sections, unset for the pair's PARITY section.
  ci=$(cd "$here/../../../scripts/ci" 2>/dev/null && pwd || echo "$here/../../../scripts/ci")
  for file in gdn_tp4_card_test.py chip_view.py tp_shapes.py tp_kernels.py tp_addresses.py gdn_user_batch_tp.py \
              gdn_multitoken_tp.py gdn_commit_dma_tp.py gdn_commit_dma_tp.cpp gdn_conv_windows_tp.cpp \
              gdn_conv_prefix_copy_tp.cpp; do
    test -s "$ci/$file" \
      || { echo "refusing: $ci/$file missing (the four-card GDN test runs this checkout's scripts/ci)" >&2; exit 1; }
  done
  echo "### code under test: $ci (QWEN_FAST_TP=${TP4_WIDTH:-4}, 2 meaning unset)"
  # /bench/ci goes FIRST and the image's own entries stay behind it: replacing the variable outright would drop
  # /opt/tt-metal/ttnn and /opt/tt-metal, and the harness's `import ttnn` (and `models`) would die on the card.
  XE=(-e PYTHONPATH=/bench/ci:/experiment-scripts/ci:/speculative-decoding/harness:/opt/tt-metal/ttnn:/opt/tt-metal)
  case ${TP4_WIDTH:-4} in
    4) XE+=(-e QWEN_FAST_TP=4) ;;
    2) ;;
    *) echo "refusing: TP4_WIDTH=${TP4_WIDTH} is neither 4 nor 2" >&2; exit 1 ;;
  esac
fi

if [ "$MAIN" = ordered_writer ]; then
  # E1: this checkout's scripts/ci first on PYTHONPATH (the pinned ordered_cache.py it loads is the checkout's too: the image's
  # copy is the same bytes, and the report records its sha256 for the recorder to compare with the live file).
  ci=$(cd "$here/../../../scripts/ci" 2>/dev/null && pwd || echo "$here/../../../scripts/ci")
  for file in ordered_writer_tp4_card_test.py ordered_cache_hw_plan.py ordered_cache.py ordered_cache_tp.py packed_ordered_cache.py \
              page_width_tp4.py ordered_writer_evidence_tp4.json chip_view.py tp_shapes.py verify_trace_t1.py verify_trace_t2.py; do
    test -s "$ci/$file" \
      || { echo "refusing: $ci/$file missing (the ordered-writer test runs this checkout's scripts/ci)" >&2; exit 1; }
  done
  echo "### code under test: $ci (QWEN_FAST_TP=4)"
  XE=(-e PYTHONPATH=/bench/ci:/experiment-scripts/ci:/speculative-decoding/harness:/opt/tt-metal/ttnn:/opt/tt-metal
      -e QWEN_FAST_TP=4 -e QWEN_C2_SERVING=0)
fi

# Graft K64j, checked before anything is launched (in a dry run, only when it exists here).
if [ "$DRY" = 1 ] && [ ! -e "$G" ]; then
  echo "### dry run: $G does not exist here; the graft was not checked"
else
  for part in _ttnn.so _ttnncpp.so attn_prep nlp_concat_heads_decode sdpa_decode sdpa MANIFEST.sha256; do
    test -e "$G/$part" || { echo "refusing: $G/$part missing (the harness needs graft K64j, build_k64j.sh)" >&2; exit 1; }
  done
  (cd "$G" && sha256sum -c --quiet MANIFEST.sha256 >&2) \
    || { echo "refusing: $G/MANIFEST.sha256 does not verify (not the graft its build script made)" >&2; exit 1; }
  so_sha=$(sha256sum "$G/_ttnncpp.so" | cut -c1-64)
  if [ -n "$EXPECT" ] && [ "$so_sha" != "$EXPECT" ]; then
    echo "refusing: $G/_ttnncpp.so is ${so_sha:0:16}, not ${EXPECT:0:16}" >&2
    exit 1
  fi
  grep -a -q -F -- "$EXTENT_MARKER" "$G/_ttnncpp.so" \
    || { echo "refusing: $G/_ttnncpp.so lacks '$EXTENT_MARKER' (not a K64j binary)" >&2; exit 1; }
  pairs=("dataflow/reader_decode_qwen.cpp:$READER_QWEN" "compute/sdpa_flash_decode_qwen.cpp:$COMPUTE_QWEN"
         "dataflow/reader_decode_qwen_slice.cpp:$READER_SLICE" "dataflow/writer_decode_qwen_slice.cpp:$WRITER_SLICE")
  stock=("dataflow/reader_decode_all.cpp:$READER_ALL" "dataflow/writer_decode_all.cpp:$WRITER_ALL"
         "compute/sdpa_flash_decode.cpp:$COMPUTE_ALL" "dataflow/dataflow_common.hpp:$DATAFLOW_COMMON"
         "rt_args_common.hpp:$RT_ARGS_COMMON")
  for pair in "${pairs[@]/#/K64j:}" "${stock[@]/#/stock:}"; do
    kind=${pair%%:*}
    pair=${pair#*:}
    file=$G/sdpa_decode/device/kernels/${pair%%:*}
    got=$(sha256sum "$file" 2>/dev/null | cut -c1-64 || true)
    if [ "$got" != "${pair#*:}" ]; then
      echo "refusing: $file is ${got:-missing}, not the $kind ${pair#*:}" >&2
      exit 1
    fi
  done
  echo "### graft $G: _ttnncpp.so ${so_sha:0:16} with the F22 literal, ${#pairs[@]} K64j qwen kernels, ${#stock[@]} stock" \
    "decode sources, manifest verified"
fi
KM=(--mount "type=bind,src=$G/_ttnn.so,dst=/opt/tt-metal/ttnn/ttnn/_ttnn.so,readonly"
    --mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so,readonly"
    --mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/lib/_ttnncpp.so,readonly")
for op in attn_prep:transformer/attn_prep nlp_concat_heads_decode:experimental/transformer/nlp_concat_heads_decode \
          sdpa_decode:transformer/sdpa_decode sdpa:transformer/sdpa; do
  KM+=(--mount "type=bind,src=$G/${op%%:*},dst=$OPS/${op#*:},readonly")
done

if [ "$DRY" != 1 ]; then
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "refusing: image $IMAGE is not present on this host" >&2; exit 1; }
  mkdir -p "$R" "$R/kcache-$stamp"
  chmod 0777 "$R" "$R/kcache-$stamp"
fi

args=(--out "/results/$stem-$stamp.json" --expect-binary-sha256 "$EXPECT")
if [ "$MAIN" = extent_reader ]; then
  args+=(${WIDTH_ARGS[@]+"${WIDTH_ARGS[@]}"})
fi
# One KV head per chip: --kv-heads 1 in CARD_B_ARGS (the last --kv-heads wins, as argparse reads it). The watcher pass below
# then runs the one-head combos: G8B2 0x27 (the q-slice) needs a second KV head and the harness refuses it as a combo.
ONE_HEAD=0
prev=
for word in ${CARD_B_ARGS:-}; do
  case $word in
    --kv-heads=1) ONE_HEAD=1 ;;
    --kv-heads=*) ONE_HEAD=0 ;;
  esac
  if [ "$prev" = --kv-heads ]; then
    if [ "$word" = 1 ]; then ONE_HEAD=1; else ONE_HEAD=0; fi
  fi
  prev=$word
done
BM=()
for file in "${HARNESS[@]}"; do
  BM+=(--mount "type=bind,src=$file,dst=/bench/$(basename "$file"),readonly")
done
if [ "$MAIN" = extent_reader ] || [ "$MAIN" = gdn_tp4 ] || [ "$MAIN" = ordered_writer ]; then
  BM+=(--mount "type=bind,src=$ci,dst=/bench/ci,readonly")
fi
WM=()
if [ "${WATCHER:-}" = "1" ]; then
  # One pass under the NoC sanitiser over every new program kind (0x21, share, slice) and the trace; timing is
  # meaningless here. CARD_B_ARGS, appended last, can widen it.
  timeout_s=2700
  if [ "$MAIN" = extent_reader ]; then
    args+=(--seeds 0 --variants peaky --r1-words 0,32,255 --r2-families 5 --r2-restages 1 --idle-patterns 2+3
           --watchdog "${WATCHDOG_S:-120}")
  elif [ "$MAIN" = nkv1_spike ]; then
    args+=(--extents 2304 --starts 128,255 --seeds 0 --iterations 0 --no-legacy --watchdog "${WATCHDOG_S:-120}")
  elif [ "$MAIN" = gdn_tp4 ]; then
    args+=(--seeds 17 --prefixes 0,16 --iterations 0 --watchdog "${WATCHDOG_S:-120}")
  elif [ "$MAIN" = ordered_writer ]; then
    args+=(--seeds 0 --modes eager,replay_changed --watchdog "${WATCHDOG_S:-120}")
  else
    if [ "$ONE_HEAD" = 1 ] && [ "$MAIN" = card ]; then
      watcher_combos=(--combos G4B3:0x21,G4B3:0x23,G8B2:0x23 --trace-combos G4B3:0x21,G8B2:0x23)
    else
      watcher_combos=(--combos G4B3:0x21,G4B3:0x23,G8B2:0x27 --trace-combos G4B3:0x21,G8B2:0x27)
    fi
    args+=(--extents 2304,33024 --seeds 0 --variants normal "${watcher_combos[@]}"
           --trace-families 8 --trace-references 2 --no-timing
           --k2-sweep 232:263 --k2-floor 120:127 --cb2-extents 2304,131328 --cb2-starts 0,240,255
           --z-families 256,512,2304,3840 --watchdog "${WATCHDOG_S:-120}")
  fi
  WM=(-e TT_METAL_WATCHER=5 --mount "type=bind,src=$R/watcher-$stamp,dst=/opt/tt-metal/generated/watcher")
  if [ "$DRY" != 1 ]; then
    mkdir -p "$R/watcher-$stamp"
    chmod 0777 "$R/watcher-$stamp"
  fi
  echo "WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog ${WATCHDOG_S:-120} s, container timeout ${timeout_s} s"
else
  args+=(--watchdog "${WATCHDOG_S:-300}")
fi
# Stop cleanly 600 s (DEADLINE_MARGIN_S) before the container timeout: the rest is listed, not lost.
args+=(--deadline-s "$((timeout_s - 600))")
# shellcheck disable=SC2206
extra=(${CARD_B_ARGS:-})

# The container's script: record the binaries and the decode sources it runs, then the harness. One line (printf %q
# of a newline is $'...', which the runner test's shlex cannot read).
inner='sha256sum /opt/tt-metal/build_Release/lib/_ttnncpp.so /opt/tt-metal/build_Release/ttnn/_ttnncpp.so '
inner+="$KD/dataflow/reader_decode_qwen.cpp $KD/dataflow/reader_decode_qwen_slice.cpp $KD/compute/sdpa_flash_decode_qwen.cpp "
inner+="$KD/dataflow/writer_decode_qwen_slice.cpp $KD/dataflow/reader_decode_all.cpp $KD/dataflow/writer_decode_all.cpp "
inner+="$KD/compute/sdpa_flash_decode.cpp $KD/dataflow/dataflow_common.hpp $KD/rt_args_common.hpp "
if [ "$MAIN" = extent_reader ]; then
  for file in $SERVED_SOURCES; do
    inner+="$SERVED_CI/$file "
  done
fi
inner+='2>&1; '
if [ "$MAIN" = extent_reader ]; then
  inner+='exec python3 -B /bench/extent_reader_card_b.py "$@"'
elif [ "$MAIN" = nkv1_spike ]; then
  inner+='exec python3 -B /bench/k64j_nkv1_spike.py "$@"'
elif [ "$MAIN" = gdn_tp4 ]; then
  inner+='exec python3 -B /bench/ci/gdn_tp4_card_test.py "$@"'
elif [ "$MAIN" = ordered_writer ]; then
  inner+='exec python3 -B /bench/ci/ordered_writer_tp4_card_test.py "$@"'
else
  inner+='exec python3 -B /bench/k64j_card_b.py "$@"'
fi

argv=(docker run --rm --name "$name" --network none
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g
  --device "$node"
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G
  "${BM[@]}"
  --mount "type=bind,src=$R,dst=/results"
  --mount "type=bind,src=$R/kcache-$stamp,dst=/kcache"
  "${KM[@]}"
  ${WM[@]+"${WM[@]}"}
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8
  -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1
  ${XE[@]+"${XE[@]}"}
  --entrypoint sh "$IMAGE" -c "$inner"
  card "${args[@]}" ${extra[@]+"${extra[@]}"})
echo "### k64j-card $stamp card=$QUAL_CARD ($QUAL_TAG) node=$node image=${IMAGE:7:12} graft=$G watcher=${WATCHER:-0}" \
  "harness=$MAIN"
echo "### argv: $(printf '%q ' "${argv[@]}")"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing launched"
  exit 0
fi
qual_card_recheck   # the board is still on the node the holder check cleared
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 "$timeout_s" "${argv[@]}" 2>&1 | tee "$R/$stem-$stamp.log"
status=${PIPESTATUS[0]}
log=$R/$stem-$stamp.log
echo "### exit $status; report $R/$stem-$stamp.json; native log $R/$stem-$stamp.json.native.log"
if [ "${WATCHER:-}" = "1" ]; then
  wlog=$R/watcher-$stamp/watcher.log
  if [ -s "$wlog" ]; then
    echo "### watcher log $wlog: $(grep -ciE 'error|assert|tripped|sanitiz' "$wlog" || true) error/assert lines"
    grep -iE 'error|assert|tripped|sanitiz' "$wlog" | head -20 || true
  else
    echo "### no watcher log at $wlog"
  fi
fi
# Anchored: the summary line 'SDPA_K64J_CARD passed=...' comes after the verdict and also contains 'K64J_CARD '
# (the extent reader's verdict line is K64J_READER's).
echo "### $(grep -E "^${verdict_tag:-K64J_CARD} " "$log" | tail -1 || echo "no ${verdict_tag:-K64J_CARD} line")"
# >>> hang: status, log -> hung
hung=0
case "$status" in
  3|124|137) hung=1 ;;
  1) grep -qF 'Timeout (' "$log" && hung=1 ;;   # the faulthandler backstop (GIL held)
esac
# The container timeout (124) after the harness caught SIGTERM and unwound: its 'ERROR terminated by signal' line is
# printed only after the device closed, so this is an overrun (see its partial report), not a hang.
if [ "$status" = 124 ] && grep -q '^ERROR terminated by signal' "$log" && ! grep -qF 'Timeout (' "$log"; then
  hung=0
  echo "### TIMEOUT (exit 124): the harness caught SIGTERM, wrote its partial report and closed the device; not a" \
    "hang. Raise the timeout or split the run (--sections)." >&2
fi
# <<< hang
if [ "$hung" = 1 ]; then
  echo "HANG SUSPECTED (exit $status): the container is removed on exit." >&2
  qual_reset_hint >&2
fi
exit "$status"
