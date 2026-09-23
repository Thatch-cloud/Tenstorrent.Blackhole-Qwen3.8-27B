#!/usr/bin/env bash
# Run test_sdpa_prefill_chain_card_m.py (Q1 of sdpa-prefill-share-spec.md 6.2) on card M, in the
# model-gate image, mirroring ../sdpa_decode_qwen/run_card_m.sh.
#
#   bash run_card_m_pf.sh reference            # stock image, no graft: every served output's sha
#   WATCHER=1 bash run_card_m_pf.sh candidate  # THE FIRST HARDWARE PASS of K64g (Q1.2): TT_METAL_WATCHER=5
#                                              # (NoC sanitiser, waypoints, asserts), flags 0x1/0x3/0x5/0x7,
#                                              # rows 2048, starts 0 and 2048, seed 0, no stress, 120 s
#                                              # per-call watchdog, 900 s container cap
#   bash run_card_m_pf.sh candidate            # the full Q1 sweep + cache + refusals + M-A + stress + trace
#   bash run_card_m_pf.sh hang                 # Q1.3, LAST in its session (it hangs card M on purpose):
#                                              # a served pre-check, then flags 0x203; PASS = the planted call
#                                              # never returns (WATCHDOG on it, exit 3; or the faulthandler
#                                              # backstop, exit 1 + 'Timeout ('; or the container cap)
#   WATCHER=1 bash run_card_m_pf.sh hang       # Q1.3 with the watcher: PASS = QWDC / QWDV, or an assert in
#                                              # reader_interleaved_qwen_chain, in the watcher log
#
# Env: PF_SRC (~/kwork64/k64g: this directory), KOPGRAFT_PF (~/opgraft-K64g), IMAGE (the model-gate
# image A'' eceb2daa, qwen-lever-n-m3native-gate.yml v128+; spec 6: the image the model gate will use,
# and re-take the stock reference in it), RESULTS ($PF_SRC/card-m), REFERENCE (a reference report to
# use instead of the automatic pick), CARD_M_ARGS (extra args, appended last), WATCHER=1, WATCHDOG_S
# (per-call seconds, default 120), PF_DRY_RUN=1 (print the docker argv only).
#
# The candidate's reference is the newest reference-*.json in $RESULTS that PASSED, ran in the same
# image, and (for the full candidate) covers the full matrix; the full candidate refuses to start
# without one. The test also fails every swept case the reference lacks.
#
# Card M only. Refuses if any running container can reach a Tenstorrent device (a tenstorrent device
# or mount, a /dev mount, or --privileged) or if a host process holds card M's node (fuser; with
# sudo -n when that works, else this user's processes only, said so). Every run gets a fresh kernel
# cache. The graft is mounted exactly as ../sdpa_prefill_bench/run_m1.sh KOPGRAFT_PF mounts it (the
# arm's targets). candidate and hang set QWEN_SDPA_PF_TEST=1 (M-A, the test-flag refusal, 0x203).
# ON A HANG (exit 3, 124 or 137, or 1 with 'Timeout (' in the log): the EXIT trap removes the
# container; then reset CARD M ONLY with ~/.local/bin/tt-smi -r <its tt-smi BOARD index> - not the
# /dev/tenstorrent number: the hint printed below resolves card M's PCI address to match against
# tt-smi -ls - then a passing stock smoke call (run_m1.sh with M1_ARGS="--arms baseline --starts
# 0,2048 --rounds 1") before the next run. Never the serving cards.
set -euo pipefail
role=${1:?usage: run_card_m_pf.sh reference|candidate|hang}
S=${PF_SRC:-$HOME/kwork64/k64g}
G=${KOPGRAFT_PF:-$HOME/opgraft-K64g}
IMAGE=${IMAGE:-sha256:eceb2daa744c3345368a638488a804f0ffe8b76f6b0680945a47bb527bdb55a9}
R=${RESULTS:-$S/card-m}
CARD_M=/dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations
DRY=${PF_DRY_RUN:-0}
name=qwen-sdpa-pf-card-m
stamp=$(date +%Y%m%dT%H%M%S)
timeout_s=7200   # the full candidate: ~8,600 matrix calls (4 flag sets), 1,000 stress calls, 200 replays, cold JIT
WD=${WATCHDOG_S:-120}

# Card M's reset hint: tt-smi -r takes the tt-smi BOARD index, which is not the /dev/tenstorrent
# number (nodes renumber after a hot-cycle); print the PCI address to match against tt-smi -ls.
card_m_reset_hint() {
  local node majmin bdf rank
  node=$(readlink -f "$CARD_M" 2>/dev/null || echo "$CARD_M")
  bdf=
  if majmin=$(stat -c '%t:%T' "$node" 2>/dev/null); then
    bdf=$(readlink -f "/sys/dev/char/$((16#${majmin%%:*})):$((16#${majmin##*:}))/device" 2>/dev/null || true)
    bdf=${bdf##*/}
  fi
  [ -n "$bdf" ] || bdf=unknown
  rank=$(ls /dev/tenstorrent 2>/dev/null | grep -E '^[0-9]+$' | sort -n | grep -nx "${node##*/}" | cut -d: -f1 || true)
  echo "RESET CARD M ONLY (CEF5729692C19E6D = $node, PCI $bdf): tt-smi -r takes the tt-smi BOARD index, not the" \
       "/dev/tenstorrent number. Probable index $([ -n "$rank" ] && echo $((rank - 1)) || echo '?') (its rank among the" \
       "/dev/tenstorrent nodes); CONFIRM with ~/.local/bin/tt-smi -ls that this index is PCI $bdf, check gh run list" \
       "for the qwen-two-p150a-exclusive group, then ~/.local/bin/tt-smi -r <that index>; then a passing stock smoke" \
       "call. Never the serving cards." >&2
}

# Refuse if anything else can reach a Tenstorrent device: a running container with a tenstorrent
# device or mount, a /dev mount or --privileged (scripts/ci/reset-cards.sh's checks), or a host
# process holding card M's node.
refuse_device_holders() {
  local node=$1 id info st out scope
  local pre=()
  for id in $(docker ps -q); do
    info=$(docker inspect "$id" --format '{{.Name}} privileged={{.HostConfig.Privileged}} {{json .HostConfig.Devices}} {{json .Mounts}}')
    if printf '%s' "$info" | grep -qE 'privileged=true|tenstorrent|"Source":"/dev"|"PathOnHost":"/dev"'; then
      echo "refusing: container ${info%% *} can reach a Tenstorrent device (a tenstorrent device or mount, a /dev mount, or --privileged)" >&2
      exit 1
    fi
  done
  if command -v fuser >/dev/null 2>&1; then
    scope="this user's processes only (no passwordless sudo)"
    if [ "$(id -u)" = 0 ]; then
      scope=all
    elif sudo -n true >/dev/null 2>&1; then
      pre=(sudo -n)
      scope=all
    fi
    st=0
    out=$(${pre[@]+"${pre[@]}"} fuser -v "$node" 2>&1) || st=$?
    if [ "$st" = 0 ] || [ -n "$out" ]; then
      echo "refusing: host processes hold $node (or fuser failed):" >&2
      echo "$out" >&2
      exit 1
    fi
    echo "### device holders on $node: none ($scope)"
  else
    echo "WARN: fuser is not installed; host processes holding $node were not checked" >&2
  fi
}

test -s "$S/test_sdpa_prefill_chain_card_m.py" || { echo "$S/test_sdpa_prefill_chain_card_m.py missing (set PF_SRC)" >&2; exit 1; }
if [ "$DRY" = 1 ]; then
  node=$CARD_M
else
  node=$(readlink -f "$CARD_M")
  test -c "$node" || { echo "card M ($CARD_M) is not a device node here" >&2; exit 1; }
  refuse_device_holders "$node"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "refusing: image $IMAGE is not present on this host" >&2; exit 1; }
  mkdir -p "$R" "$R/kcache-$role-$stamp"
  chmod 0777 "$R" "$R/kcache-$role-$stamp"
fi

args=(--role "$role" --out "/results/$role-$stamp.json" --watchdog "$WD" --image "$IMAGE")
KM=()
EM=()
case "$role" in
  reference)
    ;;
  candidate|hang)
    for part in _ttnn.so _ttnncpp.so sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp; do
      test -s "$G/$part" || { echo "$G/$part missing: build it with build_k64g.sh" >&2; exit 1; }
    done
    KM+=(--mount "type=bind,src=$G/_ttnn.so,dst=/opt/tt-metal/ttnn/ttnn/_ttnn.so,readonly")
    KM+=(--mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so,readonly")
    KM+=(--mount "type=bind,src=$G/_ttnncpp.so,dst=/opt/tt-metal/build_Release/lib/_ttnncpp.so,readonly")
    for op in attn_prep:transformer/attn_prep nlp_concat_heads_decode:experimental/transformer/nlp_concat_heads_decode \
              sdpa_decode:transformer/sdpa_decode sdpa:transformer/sdpa; do
      if [ -d "$G/${op%%:*}" ]; then
        KM+=(--mount "type=bind,src=$G/${op%%:*},dst=$OPS/${op#*:},readonly")
      fi
    done
    EM+=(-e QWEN_SDPA_PF_TEST=1)
    if [ "$role" = candidate ]; then
      # The newest reference that passed, ran in this image and (full candidate) covers the full matrix.
      reference=
      if [ -n "${REFERENCE:-}" ]; then
        reference=$REFERENCE
      elif compgen -G "$R/reference-*.json" >/dev/null; then
        reference=$("${PF_PYTHON:-python3}" -B - "$R" "$IMAGE" "${WATCHER:-0}" <<'PY' || true
import json
import sys
from pathlib import Path

root, image, watcher = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == '1'
paths = sorted(root.glob('reference-*.json'), key=lambda path: path.stat().st_mtime, reverse=True) if root.is_dir() else []
for path in paths:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    if data.get('passed') is True and data.get('image') == image and (watcher or data.get('narrow') is False):
        print(path.as_posix())
        break
PY
)
      fi
      if [ -n "$reference" ]; then
        test -s "$reference" || { echo "refusing: reference $reference is missing or empty" >&2; exit 1; }
        case "$(readlink -f "$reference")" in
          "$(readlink -f "$R")"/*) ;;
          *) echo "refusing: reference $reference is not in $R (the container sees /results only)" >&2; exit 1 ;;
        esac
        args+=(--reference "/results/$(basename "$reference")")
        echo "served outputs are compared against $reference"
      elif [ "$DRY" = 1 ]; then
        echo "no passing reference report for this image in $R (dry run: continuing)"
      else
        echo "refusing: no passing $( [ "${WATCHER:-}" = 1 ] || echo 'full-matrix ')reference report for image ${IMAGE:7:12} in $R;" \
             "run 'run_card_m_pf.sh reference' first (Q1.6 needs it) or set REFERENCE" >&2
        exit 1
      fi
    fi
    ;;
  *)
    echo "unknown role $role" >&2
    exit 1
    ;;
esac
if [ "$role" = hang ]; then
  timeout_s=900
fi
if [ "${WATCHER:-}" = "1" ]; then
  timeout_s=900
  if [ "$role" = candidate ]; then
    args+=(--rows 2048 --starts 0,2048 --seeds 0 --variants normal --tables perm --widths fit --q-memory dram
           --flags 0x1,0x3,0x5,0x7 --alternations 0 --trace-replays 0 --no-mutation --no-controls)
  elif [ "$role" = reference ]; then
    args+=(--rows 2048 --starts 0,2048 --seeds 0 --variants normal --tables perm --widths fit --q-memory dram)
  fi
  WM=(-e TT_METAL_WATCHER=5 --mount "type=bind,src=$R/watcher-$stamp,dst=/opt/tt-metal/generated/watcher")
  if [ "$DRY" != 1 ]; then
    mkdir -p "$R/watcher-$stamp"
    chmod 0777 "$R/watcher-$stamp"
  fi
  echo "WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog ${WD} s, container cap ${timeout_s} s, log $R/watcher-$stamp"
else
  WM=()
fi
# shellcheck disable=SC2206
extra=(${CARD_M_ARGS:-})

argv=(docker run --rm --name "$name" --network none
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g
  --device "$node"
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G
  --mount "type=bind,src=$S/test_sdpa_prefill_chain_card_m.py,dst=/bench/test_sdpa_prefill_chain_card_m.py,readonly"
  --mount "type=bind,src=$R,dst=/results"
  --mount "type=bind,src=$R/kcache-$role-$stamp,dst=/kcache"
  ${KM[@]+"${KM[@]}"}
  ${WM[@]+"${WM[@]}"}
  ${EM[@]+"${EM[@]}"}
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8
  --entrypoint sh "$IMAGE" -c
  "sha256sum /opt/tt-metal/build_Release/lib/_ttnncpp.so $OPS/transformer/sdpa/device/kernels/dataflow/reader_interleaved*.cpp $OPS/transformer/sdpa/device/sdpa_program_factory.cpp 2>&1; exec python3 -B /bench/test_sdpa_prefill_chain_card_m.py \"\$@\""
  card-m "${args[@]}" ${extra[@]+"${extra[@]}"})
echo "### card M pf $role $stamp node=$node image=${IMAGE:7:12} graft=$([ "$role" = reference ] && echo none || echo "$G") watcher=${WATCHER:-0}"
echo "### argv: $(printf '%q ' "${argv[@]}")"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing launched"
  exit 0
fi
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e
timeout -k 30 "$timeout_s" "${argv[@]}" 2>&1 | tee "$R/$role-$stamp.log"
status=${PIPESTATUS[0]}
log=$R/$role-$stamp.log
echo "### exit $status; report $R/$role-$stamp.json; native log $R/$role-$stamp.json.native.log"
wlog=$R/watcher-$stamp/watcher.log
if [ "${WATCHER:-}" = "1" ]; then
  if [ -s "$wlog" ]; then
    echo "### watcher log $wlog: $(grep -ciE 'error|assert|tripped|sanitiz' "$wlog" || true) error/assert lines; QWDC/QWDV lines: $(grep -cE 'QWD[CV]' "$wlog" || true)"
    grep -iE 'error|assert|tripped|sanitiz|QWD[CV]' "$wlog" | head -20 || true
  else
    echo "### no watcher log at $wlog"
  fi
fi
if [ "$role" = hang ]; then
  if grep -qF 'planted hang RETURNED' "$log" "$R/$role-$stamp.json" 2>/dev/null; then
    echo "HANG TEST FAIL (K-7): the planted hang returned a result" >&2
    card_m_reset_hint
    exit 1
  fi
  verdict=2
  if ! grep -qF 'planted hang: armed' "$log"; then
    echo "HANG TEST INCONCLUSIVE (exit $status): the planted hang was never armed - the device open, an upload or" \
         "the served pre-check did not complete, so card M is suspect before the test began" >&2
  elif [ "${WATCHER:-}" = "1" ]; then
    if grep -qE 'QWD[CV]' "$wlog" 2>/dev/null \
       || { grep -qF 'tripped an assert' "$wlog" 2>/dev/null && grep -qF reader_interleaved_qwen_chain "$wlog"; }; then
      echo "HANG TEST PASS: the bounded wait fired under the watcher (QWDC/QWDV, or an assert with the chain reader loaded); exit $status"
      verdict=0
    else
      echo "HANG TEST INCONCLUSIVE (exit $status): the watcher log names neither QWDC/QWDV nor a chain-reader assert" >&2
    fi
  elif [ "$status" = 3 ] && grep -qE "WATCHDOG: '(planted hang 0x203|planted hang read back)'" "$log"; then
    echo "HANG TEST PASS: the planted call never returned; the host watchdog exited 3"
    verdict=0
  elif [ "$status" = 1 ] && grep -qF 'Timeout (' "$log"; then
    echo "HANG TEST PASS: the planted call never returned; the faulthandler backstop exited 1 (the Python watchdog" \
         "thread never ran: a blocking ttnn call held the GIL)"
    verdict=0
  elif [ "$status" = 124 ] || [ "$status" = 137 ]; then
    echo "HANG TEST PASS: the planted call never returned (container cap, exit $status). WARN: neither the host" \
         "watchdog nor its faulthandler backstop fired - fix that before relying on them in a candidate run" >&2
    verdict=0
  else
    echo "HANG TEST INCONCLUSIVE (exit $status): the watchdog did not name the planted call or its read back" >&2
  fi
  echo "NOW RECOVER card M: the container is removed on exit." >&2
  card_m_reset_hint
  exit "$verdict"
fi
hung=0
case "$status" in
  3|124|137) hung=1 ;;
  1) grep -qF 'Timeout (' "$log" && hung=1 ;;
esac
if [ "$hung" = 1 ]; then
  echo "HANG SUSPECTED (exit $status): the container is removed on exit." >&2
  card_m_reset_hint
fi
exit "$status"
