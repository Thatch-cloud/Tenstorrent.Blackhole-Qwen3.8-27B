#!/usr/bin/env bash
# Run test_sdpa_decode_qwen_card_m.py on card M, in the serving image.
#
#   bash run_card_m.sh reference   # stock image, no graft: legacy calls only, records output shas
#   bash run_card_m.sh candidate   # K64e graft mounted exactly as lever_n_m3native_run_arm.sh
#                                  # mounts it; compares tail vs legacy, and legacy vs reference
#
# Env: K64E_SRC (~/kwork64/k64e: this directory), KOPGRAFT64 (~/opgraft-K64e), IMAGE (the
# arm's default serving image), RESULTS (~/kwork64/k64e/card-m), CARD_M_ARGS (extra args,
# e.g. "--seeds 0 --capacities 33024" for a quick pass), WATCHER=1 (TT_METAL_WATCHER=5).
# Every run gets a fresh kernel cache. On a hang: docker rm -f qwen-sdpa-card-m, then
# ~/.local/bin/tt-smi -r for card M only.
set -euo pipefail
mode=${1:?usage: run_card_m.sh reference|candidate}
S=${K64E_SRC:-$HOME/kwork64/k64e}
G=${KOPGRAFT64:-$HOME/opgraft-K64e}
IMAGE=${IMAGE:-sha256:e41ef884f4c8e07ce6632511fac19364af35dc73d601f05f5191784a64a3a768}
R=${RESULTS:-$S/card-m}
CARD_M=/dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D
name=qwen-sdpa-card-m
stamp=$(date +%Y%m%dT%H%M%S)

node=$(readlink -f "$CARD_M")
test -c "$node" || { echo "card M ($CARD_M) is not a device node here" >&2; exit 1; }
# Never share the card: refuse if any running container can reach a Tenstorrent device.
for id in $(docker ps -q); do
  if docker inspect "$id" --format '{{json .HostConfig.Devices}} {{json .Mounts}}' | grep -q tenstorrent; then
    echo "refusing: container $(docker inspect "$id" --format '{{.Name}}') holds a Tenstorrent device" >&2
    exit 1
  fi
done
mkdir -p "$R" "$R/kcache-$mode-$stamp"
chmod 0777 "$R" "$R/kcache-$mode-$stamp"

args=(--out "/results/$mode-$stamp.json")
KM=()
case "$mode" in
  reference)
    args+=(--legacy-only)
    ;;
  candidate)
    for part in _ttnn.so _ttnncpp.so attn_prep nlp_concat_heads_decode sdpa_decode; do
      test -e "$G/$part" || { echo "$G/$part missing: build it with build_k64e.sh" >&2; exit 1; }
    done
    KM+=(-v "$G/_ttnn.so:/opt/tt-metal/ttnn/ttnn/_ttnn.so:ro")
    KM+=(-v "$G/_ttnncpp.so:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so:ro")
    KM+=(-v "$G/_ttnncpp.so:/opt/tt-metal/build_Release/lib/_ttnncpp.so:ro")
    KM+=(-v "$G/attn_prep:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/attn_prep:ro")
    KM+=(-v "$G/nlp_concat_heads_decode:/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode:ro")
    KM+=(-v "$G/sdpa_decode:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode:ro")
    reference=$(ls -t "$R"/reference-*.json 2>/dev/null | head -1 || true)
    if [ -n "$reference" ]; then
      args+=(--reference "/results/$(basename "$reference")")
      echo "legacy outputs are compared against $reference"
    else
      echo "no reference report in $R: run 'run_card_m.sh reference' first for the legacy-regression check"
    fi
    ;;
  *)
    echo "unknown mode $mode" >&2
    exit 1
    ;;
esac
# shellcheck disable=SC2206
extra=(${CARD_M_ARGS:-})

echo "### card M $mode $stamp node=$node image=${IMAGE:7:12}"
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 1800 docker run --rm --name "$name" --network none \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g \
  --device "$node" \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount "type=bind,src=$S/test_sdpa_decode_qwen_card_m.py,dst=/bench/test_sdpa_decode_qwen_card_m.py,readonly" \
  --mount "type=bind,src=$R,dst=/results" \
  --mount "type=bind,src=$R/kcache-$mode-$stamp,dst=/kcache" \
  "${KM[@]}" \
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8 \
  ${WATCHER:+-e TT_METAL_WATCHER=5} \
  --entrypoint python3 "$IMAGE" -B /bench/test_sdpa_decode_qwen_card_m.py "${args[@]}" "${extra[@]}" \
  2>&1 | tee "$R/$mode-$stamp.log"
status=${PIPESTATUS[0]}
echo "### exit $status; report $R/$mode-$stamp.json; native log $R/$mode-$stamp.json.native.log"
exit "$status"
