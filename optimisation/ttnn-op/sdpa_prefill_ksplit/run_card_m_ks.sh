#!/usr/bin/env bash
# The K-split card-M job (prefill lever PF3-K): ksplit_card_m.py on the card QUAL_CARD names (no default; card B is refused), in the serving image, with graft K64j-OQ-KS
# (build_k64j_oq_ks.sh) mounted as the oneq runner mounts K64j-OQ: _ttnn.so, _ttnncpp.so (both paths), attn_prep, nlp_concat_heads_decode, sdpa_decode and sdpa (whose kernels directory now
# also holds the three ksplit kernels). No model, no weights: the harness calls the served causal chunked paged SDPA at the TP4 shape directly in three programs - stock (no program word),
# oneq (0x5EFA000B) and ksplit (0x5EFA001B) - and judges ksplit against oneq by an ERROR STATISTIC (ksplit_numerics.py), not by bytes: the K-split is a numerics change (docs/tp4-prefill3-ksplit.md).
# A TEMPLATE: it has never run on a card.
#
#   WATCHER=1 bash run_card_m_ks.sh     # THE FIRST HARDWARE PASS: TT_METAL_WATCHER=5 (NoC sanitiser, waypoints, asserts), 1,024 and 512 rows, five starts, one seed, no timing,
#                                       # a 2,700 s container timeout; the watcher log lands in $RESULTS/watcher-<stamp>/   (about 40 minutes)
#   bash run_card_m_ks.sh               # the statistics, refusals, cache, grid and timing passes, a 5,400 s container timeout   (about 75 minutes)
#   bash run_card_m_ks.sh hang          # THE PLANTED HANG, LAST in its session (it hangs the card on purpose): flag 0x41B with QWEN_SDPA_PF_TEST=1; PASS = the watchdog fires
#                                       # on the planted call or its read back (exit 3 + WATCHDOG, or exit 1 + 'Timeout ('); WATCHER=1 adds the sanitiser (PASS = QKDC or an assert)
#   KS_CARD_DRY_RUN=1 bash run_card_m_ks.sh    # print the launch argv and exit: no node resolution, no holder check, no docker (the graft is checked when it exists)
#
# Through CI (the cardm action of qwen-c2-serving.yml; it sets QUAL_CARD, ALLOW_SERVING_CARD and RESULTS itself):
#   C2_CARDM_HARNESS=optimisation/ttnn-op/sdpa_prefill_ksplit/run_card_m_ks.sh
#   C2_CARDM_ENV=IMAGE=<the serving image> KOPGRAFT64=<the K64j-OQ-KS graft dir> EXPECT_TTNNCPP_SHA256=<the K64J_OQ_KS_TTNNCPP_SHA256 line build_k64j_oq_ks.sh printed> [WATCHER=1]
#   C2_CARDM_ARGS=<extra ksplit_card_m.py arguments, e.g. --configs s2 --no-timing>
# By hand: QUAL_CARD=<board id> [ALLOW_SERVING_CARD=1] IMAGE=... KOPGRAFT64=... EXPECT_TTNNCPP_SHA256=... bash run_card_m_ks.sh [hang]
# Card M (the serving pair's half) needs ALLOW_SERVING_CARD=1 and a look at gh run list for the qwen-two-p150a-exclusive group before and after; a hang's hint resets card M and card A
# TOGETHER. The node is resolved by board id at launch and rechecked right before the docker run; the holder check refuses while a container or a host process can reach the card.
#
# The card selection is the canonical library scripts/ci/qual_card.sh SOURCED (as the scripts/ci rig runners do), not embedded as the older harnesses embed it: the embedded copies carry
# board ids, which a new file in a public repository must not. A checkout is therefore required (the runner refuses without scripts/ci/qual_card.sh beside the ops directories).
#
# Before anything is launched the graft must verify against its MANIFEST.sha256, its _ttnncpp.so must be EXPECT_TTNNCPP_SHA256 (REQUIRED for a real run) and carry the K64j runtime-extent
# literal, the chain factory's, the oneq edits' and the ksplit edits' literals, and its three ksplit kernels must be the recorded ones. Inside the container the harness checks the
# mapped binary again (the markers and the sha).
#
# Env: KOPGRAFT64 (default ~/opgraft-K64j-OQ-KS), EXPECT_TTNNCPP_SHA256, IMAGE (required for a real run: no default digest), RESULTS (~/kwork64/k64j-oq-ks/<card tag>), KS_ARGS (extra harness
# args, appended last), WATCHER=1, WATCHDOG_S, KS_CARD_DRY_RUN=1, QUAL_CARD, ALLOW_SERVING_CARD. QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is set as the K64j arm sets it. Every run gets a fresh kernel cache.
#
# ON A HANG (the WATCHDOG line and exit 3; exit 1 with 'Timeout (' in the log; or the timeout's 124 / 137): the EXIT trap removes the container. Then reset THE TARGET CARD ONLY with the hint
# printed below. This script never resets.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ops=$(cd "$here/.." && pwd)
repo=$(cd "$here/../../.." && pwd)
lib=$repo/scripts/ci/qual_card.sh
numerics=$repo/scripts/ci/ksplit_numerics.py
role=${1:-numerics}
case $role in
  numerics|hang) ;;
  *) echo "usage: run_card_m_ks.sh [hang]  (numerics is the default)" >&2; exit 1 ;;
esac
test -s "$lib" || { echo "refusing: $lib missing (this runner sources the canonical card-selection library; run it from a checkout)" >&2; exit 1; }
# shellcheck disable=SC1090
. "$lib"
G=${KOPGRAFT64:-$HOME/opgraft-K64j-OQ-KS}
IMAGE=${IMAGE:-}
EXPECT=${EXPECT_TTNNCPP_SHA256:-}
EXTENT_MARKER='[QWEN-SDPA] runtime-extent entries='
CHAIN_MARKER='[QWEN-SDPA-PF] flags='
ONEQ_MARKER='[QWEN-SDPA-PF] oneq needs one q chunk per core'
KS_MARKER='[QWEN-SDPA-PF] ksplit needs at least 2 slices'
KS_READER_SHA=16438818a762c699d929c579903925eba175b6f0e61171efa01dc0113a8dc128
KS_WRITER_SHA=9b9ef47f9af44cfe83dc82d587568cfd82c4881021306f7af677cb672fd589ee
KS_COMPUTE_SHA=fc9571d178639f139b1da0f6aa385b5b10963f647df83860e763a7caff893913
qual_card_select
R=${RESULTS:-$HOME/kwork64/k64j-oq-ks/$QUAL_TAG}
SDPA_OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations
KDIR=transformer/sdpa/device/kernels
DRY=${KS_CARD_DRY_RUN:-0}
case $DRY in
  0|1) ;;
  *) echo "refusing: KS_CARD_DRY_RUN=$DRY (0 or 1)" >&2; exit 1 ;;
esac
name=qwen-ksplit-card-$QUAL_TAG
stamp=$(date +%Y%m%dT%H%M%S)
timeout_s=5400   # 3 configs x 2 Q memories x 2 seeds x 2 variants x 15 starts x 2 programs, the float references, the grid pass and the timing
[ "$role" = hang ] && timeout_s=900

if [ "$DRY" != 1 ] && [ -z "$EXPECT" ]; then
  echo "refusing: EXPECT_TTNNCPP_SHA256 is required (the K64J_OQ_KS_TTNNCPP_SHA256 line build_k64j_oq_ks.sh printed for $G)" >&2
  exit 1
fi
if [ "$DRY" != 1 ] && [ -z "$IMAGE" ]; then
  echo "refusing: IMAGE is required (the serving image the graft is mounted into; there is no default digest)" >&2
  exit 1
fi
[ -n "$IMAGE" ] || IMAGE=dry-run-image

# The target's node, resolved by board id now; never shared: refuse while a container or a host process can reach it (a container on another board does not block).
if [ "$DRY" = 1 ]; then
  node=$QUAL_BYID
else
  qual_card_resolve
  node=$QUAL_NODE
  qual_refuse_holders
fi

for file in "$here/ksplit_card_m.py" "$here/ksplit_planner.py" "$here/apply_factory_pk.py" "$ops/sdpa_prefill_oneq/oneq_card_m.py" "$ops/sdpa_prefill_oneq/oneq_planner.py" \
            "$ops/sdpa_prefill_oneq/oneq_report.py" "$ops/sdpa_prefill_oneq/apply_factory_ps.py" "$ops/sdpa_prefill_chain/test_sdpa_prefill_chain_card_m.py" \
            "$ops/sdpa_prefill_chain/apply_factory_pf.py" "$numerics"; do
  test -s "$file" || { echo "refusing: $file missing (ship sdpa_prefill_ksplit, sdpa_prefill_oneq and sdpa_prefill_chain side by side, and scripts/ci/ksplit_numerics.py)" >&2; exit 1; }
done

# Graft K64j-OQ-KS, checked before anything is launched (in a dry run, only when it exists here).
if [ "$DRY" = 1 ] && [ ! -e "$G" ]; then
  echo "### dry run: $G does not exist here; the graft was not checked"
else
  for part in _ttnn.so _ttnncpp.so attn_prep nlp_concat_heads_decode sdpa_decode sdpa MANIFEST.sha256; do
    test -e "$G/$part" || { echo "refusing: $G/$part missing (the harness needs graft K64j-OQ-KS, build_k64j_oq_ks.sh)" >&2; exit 1; }
  done
  (cd "$G" && sha256sum -c --quiet MANIFEST.sha256 >&2) \
    || { echo "refusing: $G/MANIFEST.sha256 does not verify (not the graft its build script made)" >&2; exit 1; }
  so_sha=$(sha256sum "$G/_ttnncpp.so" | cut -c1-64)
  if [ -n "$EXPECT" ] && [ "$so_sha" != "$EXPECT" ]; then
    echo "refusing: $G/_ttnncpp.so is ${so_sha:0:16}, not ${EXPECT:0:16}" >&2
    exit 1
  fi
  for marker in "$EXTENT_MARKER" "$CHAIN_MARKER" "$ONEQ_MARKER" "$KS_MARKER"; do
    grep -a -q -F -- "$marker" "$G/_ttnncpp.so" \
      || { echo "refusing: $G/_ttnncpp.so lacks '$marker' (not a K64j-OQ-KS binary)" >&2; exit 1; }
  done
  for pair in "dataflow/reader_interleaved_qwen_ksplit.cpp:$KS_READER_SHA" "dataflow/writer_interleaved_qwen_ksplit.cpp:$KS_WRITER_SHA" "compute/sdpa_qwen_ksplit.cpp:$KS_COMPUTE_SHA"; do
    file=${pair%%:*}
    want=${pair##*:}
    got=$(sha256sum "$G/sdpa/device/kernels/$file" 2>/dev/null | cut -c1-64 || true)
    if [ "$got" != "$want" ]; then
      echo "refusing: $G/sdpa/device/kernels/$file is ${got:-missing}, not the recorded ksplit kernel $want" >&2
      exit 1
    fi
  done
  echo "### graft $G: _ttnncpp.so ${so_sha:0:16} with the K64j, chain, oneq and ksplit literals, the three recorded ksplit kernels, manifest verified"
fi
KM=(--mount "type=bind,src=$G/_ttnn.so,dst=/opt/tt-metal/ttnn/ttnn/_ttnn.so,readonly"
    --mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so,readonly"
    --mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/lib/_ttnncpp.so,readonly")
for op in attn_prep:transformer/attn_prep nlp_concat_heads_decode:experimental/transformer/nlp_concat_heads_decode \
          sdpa_decode:transformer/sdpa_decode sdpa:transformer/sdpa; do
  KM+=(--mount "type=bind,src=$G/${op%%:*},dst=$SDPA_OPS/${op#*:},readonly")
done

if [ "$DRY" != 1 ]; then
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "refusing: image $IMAGE is not present on this host" >&2; exit 1; }
  mkdir -p "$R" "$R/kcache-$stamp"
  chmod 0777 "$R" "$R/kcache-$stamp"
fi

args=(--out "/results/ksplit-$role-$stamp.json" --expect-binary-sha256 "$EXPECT")
# The harness' own directory layout, kept under /ops so that its relative imports (../sdpa_prefill_oneq, ../sdpa_prefill_chain) resolve as in the checkout.
BM=(--mount "type=bind,src=$here,dst=/ops/sdpa_prefill_ksplit,readonly"
    --mount "type=bind,src=$ops/sdpa_prefill_oneq,dst=/ops/sdpa_prefill_oneq,readonly"
    --mount "type=bind,src=$ops/sdpa_prefill_chain,dst=/ops/sdpa_prefill_chain,readonly"
    --mount "type=bind,src=$numerics,dst=/ops/ci/ksplit_numerics.py,readonly")
WM=()
ENVS=()
if [ "$role" = hang ]; then
  args+=(--role hang --watchdog "${WATCHDOG_S:-90}")
  ENVS+=(-e QWEN_SDPA_PF_TEST=1)
elif [ "${WATCHER:-}" = "1" ]; then
  timeout_s=2700
  args+=(--configs s2,s4 --starts 0,128,2048,65536,251904 --seeds 0 --variants normal --q-memory dram --no-timing --no-grids --reference-max-start 0 --no-anchor
         --watchdog "${WATCHDOG_S:-120}")
else
  args+=(--watchdog "${WATCHDOG_S:-300}")
fi
if [ "${WATCHER:-}" = "1" ]; then
  WM=(-e TT_METAL_WATCHER=5 --mount "type=bind,src=$R/watcher-$stamp,dst=/opt/tt-metal/generated/watcher")
  if [ "$DRY" != 1 ]; then
    mkdir -p "$R/watcher-$stamp"
    chmod 0777 "$R/watcher-$stamp"
  fi
  echo "WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog ${WATCHDOG_S:-120} s, container timeout ${timeout_s} s"
fi
# shellcheck disable=SC2206
extra=(${KS_ARGS:-} ${CARD_B_ARGS:-})

# The container's script: record the binaries and the harness it runs, then the harness. One line (printf %q of a newline is $'...', which the runner test's shlex cannot read).
# unset TT_MESH_GRAPH_DESC_PATH and QWEN_C2_SERVING=0: a served image's ENV turns on the C2 boot hook, which applies the four-card mesh descriptor, and this ONE-card open then dies
# in the topology mapper.
inner='unset TT_MESH_GRAPH_DESC_PATH; sha256sum /opt/tt-metal/build_Release/lib/_ttnncpp.so /opt/tt-metal/build_Release/ttnn/_ttnncpp.so '
inner+="$SDPA_OPS/$KDIR/dataflow/reader_interleaved_qwen_ksplit.cpp $SDPA_OPS/$KDIR/dataflow/writer_interleaved_qwen_ksplit.cpp $SDPA_OPS/$KDIR/compute/sdpa_qwen_ksplit.cpp "
inner+='/ops/sdpa_prefill_ksplit/ksplit_card_m.py /ops/sdpa_prefill_ksplit/ksplit_planner.py /ops/ci/ksplit_numerics.py 2>&1; exec python3 -B /ops/sdpa_prefill_ksplit/ksplit_card_m.py "$@"'

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
  ${ENVS[@]+"${ENVS[@]}"}
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8
  ${QUAL_TT_GRID:+-e "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=$QUAL_TT_GRID"}
  -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 -e QWEN_C2_SERVING=0
  --entrypoint sh "$IMAGE" -c "$inner"
  ksplit "${args[@]}" ${extra[@]+"${extra[@]}"})
echo "### ksplit-card $stamp role=$role card=$QUAL_CARD ($QUAL_TAG) node=$node image=${IMAGE:0:19} graft=$G watcher=${WATCHER:-0}"
echo "### argv: $(printf '%q ' "${argv[@]}")"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing launched"
  exit 0
fi
qual_card_recheck   # the board is still on the node the holder check cleared
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 "$timeout_s" "${argv[@]}" 2>&1 | tee "$R/ksplit-$role-$stamp.log"
status=${PIPESTATUS[0]}
log=$R/ksplit-$role-$stamp.log
echo "### exit $status; report $R/ksplit-$role-$stamp.json; native log $R/ksplit-$role-$stamp.json.native.log"
if [ "${WATCHER:-}" = "1" ]; then
  wlog=$R/watcher-$stamp/watcher.log
  if [ -s "$wlog" ]; then
    echo "### watcher log $wlog: $(grep -ciE 'error|assert|tripped|sanitiz' "$wlog" || true) error/assert lines"
    grep -iE 'error|assert|tripped|sanitiz' "$wlog" | head -20 || true
  else
    echo "### no watcher log at $wlog"
  fi
fi
if [ "$role" = hang ]; then
  if grep -qF 'planted ksplit hang RETURNED' "$log" "$R/ksplit-$role-$stamp.json" 2>/dev/null; then
    echo "HANG TEST FAIL: the planted ksplit hang returned a result" >&2
    exit 1
  fi
  if ! grep -qF 'planted ksplit hang: armed' "$log"; then
    echo "HANG TEST INCONCLUSIVE (exit $status): the planted hang was never armed - the device open, an upload or the ksplit pre-check stalled" >&2
    qual_reset_hint >&2
    exit 1
  fi
  if [ "$status" = 3 ] && grep -qE "WATCHDOG: '(planted ksplit hang 0x41b|planted ksplit hang read back)'" "$log"; then
    echo "HANG TEST PASS: the watchdog fired on the planted call (exit 3); reset THE TARGET CARD ONLY:" >&2
    qual_reset_hint >&2
    exit 0
  fi
  if [ "$status" = 1 ] && grep -qF 'Timeout (' "$log"; then
    echo "HANG TEST PASS (backstop): the faulthandler fired while the planted call held the GIL (exit 1); reset THE TARGET CARD ONLY:" >&2
    qual_reset_hint >&2
    exit 0
  fi
  echo "HANG TEST INCONCLUSIVE (exit $status): no watchdog line on the planted call" >&2
  qual_reset_hint >&2
  exit 1
fi
# The verdict and the timing lines (anchored: the harness prints them at column 0).
echo "### $(grep -E '^KSPLIT_CARD_M ' "$log" | tail -1 || echo 'no KSPLIT_CARD_M line')"
grep -E '^KSPLIT TIME_VERDICT ' "$log" || true
# A hang: the watchdog's exit 3, the container timeout (124 / 137), or the faulthandler backstop (exit 1 with 'Timeout (' when a blocking call held the GIL).
hung=0
case "$status" in
  3|124|137) hung=1 ;;
  1) grep -qF 'Timeout (' "$log" && hung=1 ;;
esac
if [ "$hung" = 1 ]; then
  echo "HANG SUSPECTED (exit $status): the container is removed on exit." >&2
  qual_reset_hint >&2
fi
exit "$status"
