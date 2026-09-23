#!/usr/bin/env bash
# Run gdn_prefill_conv_card_m.py (lever #2: the GDN prefill conv op) on card M, in the serving image.
#
#   bash run_card_m.sh                 # the full matrix, negative controls, chain, cache, microbenches
#   WATCHER=1 bash run_card_m.sh       # THE FIRST HARDWARE PASS of a new kernel: TT_METAL_WATCHER=5
#                                      # (NoC sanitiser, waypoints), the quick matrix, no timing, a
#                                      # per-call watchdog (WATCHDOG_S, default 120 s) that
#                                      # os._exit(3)s the test, and a 900 s container timeout
#   PROFILE=1 bash run_card_m.sh       # TT_METAL_DEVICE_PROFILER=1; the device CSV lands in
#                                      # $RESULTS/profile-<stamp> (the section 7 device times)
#   MODELS=1 bash run_card_m.sh        # also layers 0 and 47's real conv taps from the HF snapshot
#
# The op is mounted from THIS checkout, one file each, into /bench/pcx: the module and its three
# kernels, the list read from lever_n_m3native_patch.PREFILL_CONV_FILES (the table the graft and the
# arm use), so the files under test are exactly the ones a model run would mount. No image build,
# no .so: the kernels JIT from the mounted sources into a fresh per-run kernel cache.
#
# Env: REPO (this checkout; default three levels up), IMAGE (the 0648ca9a serving image),
# RESULTS (~/kwork64/pcx/card-m), CARD_M_ARGS (extra args, appended last, e.g. "--ts 32,64"),
# WATCHER=1, WATCHDOG_S, PROFILE=1, MODELS=1 (MODEL_DIR: the HF snapshot, default the arm's).
#
# ON A HANG (the WATCHDOG line, exit 3, or the timeout's 124): the EXIT trap removes the
# container; then reset CARD M ONLY with ~/.local/bin/tt-smi -r <card M's index> - never the
# serving cards. This script never resets anything itself.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$here/../../.." && pwd)}
IMAGE=${IMAGE:-sha256:0648ca9ad663acc72e7d8ea59d9cde0f9218b583ad74a60d58ff2f31bddd6fae}
R=${RESULTS:-$HOME/kwork64/pcx/card-m}
MODEL_DIR=${MODEL_DIR:-/home/thatch/hf-cache/hub/models--Qwen--Qwen3.8-27B}
CARD_M=/dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D
name=qwen-pcx-card-m
stamp=$(date +%Y%m%dT%H%M%S)
timeout_s=2400

node=$(readlink -f "$CARD_M")
test -c "$node" || { echo "card M ($CARD_M) is not a device node here" >&2; exit 1; }
# Never share the card: refuse if any running container can reach a Tenstorrent device.
for id in $(docker ps -q); do
  if docker inspect "$id" --format '{{json .HostConfig.Devices}} {{json .Mounts}}' | grep -q tenstorrent; then
    echo "refusing: container $(docker inspect "$id" --format '{{.Name}}') holds a Tenstorrent device" >&2
    exit 1
  fi
done

mapfile -t op_files < <(python3 -B -c 'import sys; sys.path.insert(0, sys.argv[1]); import lever_n_m3native_patch as p; print(chr(10).join(sorted(set(p.PREFILL_CONV_FILES.values()))))' "$REPO/scripts/ci" | tr -d '\r')
if [ "${#op_files[@]}" -eq 0 ]; then
  echo "lever_n_m3native_patch.PREFILL_CONV_FILES is empty or unreadable under $REPO/scripts/ci" >&2
  exit 1
fi
OM=()
for file in "${op_files[@]}"; do
  test -s "$REPO/scripts/ci/$file" || { echo "$REPO/scripts/ci/$file missing" >&2; exit 1; }
  OM+=(--mount "type=bind,src=$REPO/scripts/ci/$file,dst=/bench/pcx/$file,readonly")
done
test_file="$here/gdn_prefill_conv_card_m.py"
test -s "$test_file" || { echo "$test_file missing" >&2; exit 1; }

mkdir -p "$R" "$R/kcache-$stamp"
chmod 0777 "$R" "$R/kcache-$stamp"
args=(--out "/results/pcx-$stamp.json" --op-dir /bench/pcx)
WM=()
PM=()
MM=()
if [ "${WATCHER:-}" = "1" ]; then
  timeout_s=900
  args+=(--quick --no-timing --seeds 0 --watchdog "${WATCHDOG_S:-120}")
  mkdir -p "$R/watcher-$stamp"
  chmod 0777 "$R/watcher-$stamp"
  WM+=(-e TT_METAL_WATCHER=5 --mount "type=bind,src=$R/watcher-$stamp,dst=/opt/tt-metal/generated/watcher")
  echo "WATCHER=1: TT_METAL_WATCHER=5, per-call watchdog ${WATCHDOG_S:-120} s, container timeout ${timeout_s} s"
else
  args+=(--watchdog "${WATCHDOG_S:-300}")
fi
if [ "${PROFILE:-}" = "1" ]; then
  mkdir -p "$R/profile-$stamp"
  chmod 0777 "$R/profile-$stamp"
  PM+=(-e TT_METAL_DEVICE_PROFILER=1 --mount "type=bind,src=$R/profile-$stamp,dst=/opt/tt-metal/generated/profiler")
  args+=(--sections timing)
  echo "PROFILE=1: device profiler on, microbenches only; CSV in $R/profile-$stamp"
fi
if [ "${MODELS:-}" = "1" ]; then
  test -d "$MODEL_DIR" || { echo "MODELS=1 but $MODEL_DIR is not a directory" >&2; exit 1; }
  MM+=(--mount "type=bind,src=$MODEL_DIR,dst=/models,readonly")
  args+=(--taps-from /models)
fi
# shellcheck disable=SC2206
extra=(${CARD_M_ARGS:-})

echo "### card M pcx $stamp node=$node image=${IMAGE:7:12} op=${op_files[*]}"
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 "$timeout_s" docker run --rm --name "$name" --network none \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g \
  --device "$node" \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  "${OM[@]}" \
  --mount "type=bind,src=$test_file,dst=/bench/gdn_prefill_conv_card_m.py,readonly" \
  --mount "type=bind,src=$R,dst=/results" \
  --mount "type=bind,src=$R/kcache-$stamp,dst=/kcache" \
  "${WM[@]}" \
  "${PM[@]}" \
  "${MM[@]}" \
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8 \
  --entrypoint python3 "$IMAGE" -B /bench/gdn_prefill_conv_card_m.py "${args[@]}" "${extra[@]}" \
  2>&1 | tee "$R/pcx-$stamp.log"
status=${PIPESTATUS[0]}
echo "### exit $status; report $R/pcx-$stamp.json"
case "$status" in
  3|124|137)
    echo "HANG SUSPECTED (exit $status): the container is removed on exit; reset card M only:" \
         "~/.local/bin/tt-smi -r <card M index> (CEF5729692C19E6D). Never the serving cards." >&2
    ;;
esac
exit "$status"
