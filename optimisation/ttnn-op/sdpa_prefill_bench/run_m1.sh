#!/usr/bin/env bash
# M1 of the prefill ranking: the chunked-SDPA isolation bench (sdpa_prefill_bench.py) on card M,
# in the serving image, no model load, no graft.
#
#   bash run_m1.sh              # all seven arms at chunk_start 0 / 32k / 64k / 126k
#   M1_ARGS="--arms baseline,bf16_kv --rounds 3" bash run_m1.sh     # a quick pass
#
# Env: M1_SRC (the directory holding sdpa_prefill_bench.py; default ~/kwork-m1), IMAGE (the
# serving image the 131k prefill numbers were measured on, sha256:0648ca9a...), RESULTS
# (default $M1_SRC/results), M1_ARGS (extra args). Card M only; refuses if any running container
# can reach a Tenstorrent device; every run gets a fresh kernel cache. Before the bench it greps
# the image's own SDPA factory for the three facts the ranking read from the TT-Sim tree (the
# Q-chunk pair distribution, zigzag balancing, and the chain-forward gate), so the verdict and
# the lever #1 build plan rest on the served source, not the simulator's.
# On a hang: docker rm -f qwen-sdpa-m1, then ~/.local/bin/tt-smi -r for card M only.
set -euo pipefail
S=${M1_SRC:-$HOME/kwork-m1}
IMAGE=${IMAGE:-sha256:0648ca9ad663acc72e7d8ea59d9cde0f9218b583ad74a60d58ff2f31bddd6fae}
R=${RESULTS:-$S/results}
CARD_M=/dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D
FACTORY=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp
name=qwen-sdpa-m1
stamp=$(date +%Y%m%dT%H%M%S)

test -s "$S/sdpa_prefill_bench.py" || { echo "$S/sdpa_prefill_bench.py missing (set M1_SRC)" >&2; exit 1; }
node=$(readlink -f "$CARD_M")
test -c "$node" || { echo "card M ($CARD_M) is not a device node here" >&2; exit 1; }
# Never share the card: refuse if any running container can reach a Tenstorrent device.
for id in $(docker ps -q); do
  if docker inspect "$id" --format '{{json .HostConfig.Devices}} {{json .Mounts}}' | grep -q tenstorrent; then
    echo "refusing: container $(docker inspect "$id" --format '{{.Name}}') holds a Tenstorrent device" >&2
    exit 1
  fi
done
mkdir -p "$R" "$R/kcache-m1-$stamp"
chmod 0777 "$R" "$R/kcache-m1-$stamp"

echo "### factory facts in ${IMAGE:7:12} ($FACTORY)"
timeout -k 10 120 docker run --rm --network none --entrypoint sh "$IMAGE" -c \
  "sha256sum $FACTORY; grep -n -E 'global_q_pair_distribute|use_zigzag_balancing|is_causal && !is_chunked|q_num_chunks % 2' $FACTORY" \
  2>&1 | tee "$R/m1-$stamp.factory.txt" || true

# shellcheck disable=SC2206
extra=(${M1_ARGS:-})
echo "### M1 $stamp node=$node image=${IMAGE:7:12}"
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 1800 docker run --rm --name "$name" --network none \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g \
  --device "$node" \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount "type=bind,src=$S/sdpa_prefill_bench.py,dst=/bench/sdpa_prefill_bench.py,readonly" \
  --mount "type=bind,src=$R,dst=/results" \
  --mount "type=bind,src=$R/kcache-m1-$stamp,dst=/kcache" \
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8 \
  --entrypoint python3 "$IMAGE" -B /bench/sdpa_prefill_bench.py --out "/results/m1-$stamp.json" "${extra[@]}" \
  2>&1 | tee "$R/m1-$stamp.log"
status=${PIPESTATUS[0]}
echo "### exit $status; report $R/m1-$stamp.json; log $R/m1-$stamp.log"
grep -F 'M1 VERDICT:' "$R/m1-$stamp.log" || true
exit "$status"
