#!/usr/bin/env bash
# M1 of the prefill ranking: the chunked-SDPA isolation bench (sdpa_prefill_bench.py) on card M,
# in the serving image, no model load, no graft.
#
#   bash run_m1.sh              # all seven arms at chunk_start 0 / 32k / 64k / 126k
#   M1_ARGS="--arms baseline,bf16_kv --rounds 3" bash run_m1.sh     # a quick pass
#   M1_READER=$HOME/kwork-m1/k0/reader_k0a.cpp M1_ARGS="--arms baseline --sha" bash run_m1.sh   # a K0 probe
#
# Env: M1_SRC (the directory holding sdpa_prefill_bench.py; default ~/kwork-m1), IMAGE (the
# serving image the 131k prefill numbers were measured on, sha256:0648ca9a...), RESULTS
# (default $M1_SRC/results), M1_ARGS (extra args). Card M only; refuses if any running container
# can reach a Tenstorrent device; every run gets a fresh kernel cache. Before the bench it greps
# the image's own SDPA factory for the three facts the ranking read from the TT-Sim tree (the
# Q-chunk pair distribution, zigzag balancing, and the chain-forward gate), so the verdict and
# the lever #1 build plan rest on the served source, not the simulator's.
#
# K0 (sdpa-prefill-share-spec.md 3.7 / 7.1):
#   M1_READER=<file>       bind-mounts that ONE file, read-only, over the image's
#                          sdpa/device/kernels/dataflow/reader_interleaved.cpp (never a directory);
#                          the served factory JIT-compiles whatever is there. Inside the container,
#                          before the bench, the mounted file must hash to the host file (else exit 97).
#   M1_REQUIRE_SOURCES=1   also refuse (exit 97) unless the other six served sdpa sources equal the
#                          probe-v25 shas below (and the reader too when M1_READER is unset).
#   M1_DRY_RUN=1           print the docker argv and exit 0: no device, holder or image check, no docker.
# The in-container sha256 of all seven served sdpa sources and of the mounted bench is printed before
# every run (the bench must hash to the host file, else exit 97), and the full launched docker argv is
# echoed ("read the launched argv"). The '### M1 <stamp> node=' line is printed only once every
# pre-launch check has passed (k0_session.sh reads it as "a container was launched on card M").
# On a hang: docker rm -f qwen-sdpa-m1, then ~/.local/bin/tt-smi -r for card M only.
set -euo pipefail
S=${M1_SRC:-$HOME/kwork-m1}
IMAGE=${IMAGE:-sha256:0648ca9ad663acc72e7d8ea59d9cde0f9218b583ad74a60d58ff2f31bddd6fae}
R=${RESULTS:-$S/results}
CARD_M=/dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D
SDPA=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device
FACTORY=$SDPA/sdpa_program_factory.cpp
READER_DST=$SDPA/kernels/dataflow/reader_interleaved.cpp
# The seven served sdpa sources (probe-v25 = images 1b9b6445 / 0648ca9a); the reader is index 1.
SERVED=(
  "fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783  $FACTORY"
  "f97f5490cf476db92d575de33c85f8707f96d7474896ee086ee864c23a3efa27  $READER_DST"
  "a3f48af8ba0fd63b136c79a54c8b6f7b4b5b8fb0d7a209bf5701f081ed7fa3e0  $SDPA/kernels/compute/sdpa.cpp"
  "3fb5da2440c3bf90ebceb8acd55424c7739339de4c6b02db83836e1e3414fa19  $SDPA/kernels/compute/compute_common.hpp"
  "554a0b282d2a36c7b129eef04df67a5becbb14f98220246e43e47fdd56118aa6  $SDPA/kernels/dataflow/dataflow_common.hpp"
  "43a24c466c97beb5d34269630fde6a7831a23246d9df99a03f8c88265044cd9f  $SDPA/kernels/dataflow/chain_link.hpp"
  "2a0959ff7cce507eab39f2e063f4dbc264d8b51f4db9f479b8c2419435c745c2  $SDPA/kernels/dataflow/writer_interleaved.cpp"
)
DRY=${M1_DRY_RUN:-0}
name=qwen-sdpa-m1
stamp=$(date +%Y%m%dT%H%M%S)

test -s "$S/sdpa_prefill_bench.py" || { echo "$S/sdpa_prefill_bench.py missing (set M1_SRC)" >&2; exit 1; }
bench_sha=$(sha256sum "$S/sdpa_prefill_bench.py" | cut -d' ' -f1)
echo "### bench $S/sdpa_prefill_bench.py sha256=$bench_sha -> /bench/sdpa_prefill_bench.py (read-only)"

# M1_READER: exactly one regular file, mounted read-only over the image's reader. Never a directory.
expect=("${SERVED[@]}")
reader_mount=()
if [ -n "${M1_READER:-}" ]; then
  reader_src=$(readlink -f -- "$M1_READER") || { echo "refusing: M1_READER=$M1_READER does not resolve" >&2; exit 1; }
  if [ -d "$reader_src" ] || [ ! -f "$reader_src" ] || [ ! -s "$reader_src" ]; then
    echo "refusing: M1_READER=$M1_READER is not one non-empty regular file (a directory is never mounted)" >&2
    exit 1
  fi
  case "$reader_src" in
    *,*) echo "refusing: M1_READER path contains a comma (breaks --mount): $reader_src" >&2; exit 1 ;;
  esac
  reader_sha=$(sha256sum "$reader_src" | cut -d' ' -f1)
  expect[1]="$reader_sha  $READER_DST"
  reader_mount=(--mount "type=bind,src=$reader_src,dst=$READER_DST,readonly")
  echo "### M1_READER $reader_src sha256=$reader_sha -> $READER_DST (one file, read-only)"
fi

if [ "$DRY" = 1 ]; then
  node=$CARD_M
else
  node=$(readlink -f "$CARD_M")
  test -c "$node" || { echo "card M ($CARD_M) is not a device node here" >&2; exit 1; }
  # Never share the card: refuse if any running container can reach a Tenstorrent device.
  for id in $(docker ps -q); do
    if docker inspect "$id" --format '{{json .HostConfig.Devices}} {{json .Mounts}}' | grep -q tenstorrent; then
      echo "refusing: container $(docker inspect "$id" --format '{{.Name}}') holds a Tenstorrent device" >&2
      exit 1
    fi
  done
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "refusing: image $IMAGE is not present on this host" >&2; exit 1; }
  mkdir -p "$R" "$R/kcache-m1-$stamp"
  chmod 0777 "$R" "$R/kcache-m1-$stamp"
  # What the container checks before the bench: every source line, the mounted bench, and
  # (M1_READER) the mounted reader.
  printf '%s\n' "${expect[@]}" > "$R/m1-$stamp.sources.sha256"
  printf '%s\n' "$bench_sha  /bench/sdpa_prefill_bench.py" > "$R/m1-$stamp.bench.sha256"
  if [ -n "${M1_READER:-}" ]; then
    printf '%s\n' "${expect[1]}" > "$R/m1-$stamp.reader.sha256"
  fi
  chmod 0644 "$R/m1-$stamp.sources.sha256" "$R/m1-$stamp.bench.sha256" "$R/m1-$stamp.reader.sha256" 2>/dev/null || true

  echo "### factory facts in ${IMAGE:7:12} ($FACTORY)"
  timeout -k 10 120 docker run --rm --network none --entrypoint sh "$IMAGE" -c \
    "sha256sum $FACTORY; grep -n -E 'global_q_pair_distribute|use_zigzag_balancing|is_causal && !is_chunked|q_num_chunks % 2' $FACTORY" \
    2>&1 | tee "$R/m1-$stamp.factory.txt" || true
fi

# The in-container preamble: the seven shas (the reader line is the mounted file under M1_READER),
# the mount proof, then exec the bench so its exit status is the container's.
paths=("${SERVED[@]#*  }")
pre="echo '### in-container sha256 of the served sdpa sources (reader_interleaved.cpp = the M1_READER file when set) and the mounted bench'; "
pre+="sha256sum ${paths[*]} /bench/sdpa_prefill_bench.py || echo '### some sdpa sources are missing in this image'; "
pre+="sha256sum -c /results/m1-$stamp.bench.sha256 || { echo '### REFUSING: the in-container bench is not the host file'; exit 97; }; "
if [ -n "${M1_READER:-}" ]; then
  pre+="grep -F reader_interleaved.cpp /proc/self/mountinfo || echo '### no mountinfo line for reader_interleaved.cpp'; "
  pre+="sha256sum -c /results/m1-$stamp.reader.sha256 || { echo '### REFUSING: the in-container reader is not the M1_READER file'; exit 97; }; "
fi
pre+="if sha256sum -c --quiet /results/m1-$stamp.sources.sha256; then echo '### sdpa sources: as expected'; "
pre+="else echo '### sdpa sources DIFFER from the expected shas'; [ \"\${M1_REQUIRE_SOURCES:-0}\" = 1 ] && exit 97; fi; "
pre+="exec python3 -B /bench/sdpa_prefill_bench.py --out /results/m1-$stamp.json \"\$@\""

# shellcheck disable=SC2206
extra=(${M1_ARGS:-})
argv=(docker run --rm --name "$name" --network none
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges
  --pids-limit 1024 --memory 48g --cpus 8 --shm-size 4g
  --device "$node"
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G
  --mount "type=bind,src=$S/sdpa_prefill_bench.py,dst=/bench/sdpa_prefill_bench.py,readonly"
  ${reader_mount[@]+"${reader_mount[@]}"}
  --mount "type=bind,src=$R,dst=/results"
  --mount "type=bind,src=$R/kcache-m1-$stamp,dst=/kcache"
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8
  -e "M1_REQUIRE_SOURCES=${M1_REQUIRE_SOURCES:-0}"
  --entrypoint sh "$IMAGE" -c "$pre" m1 "${extra[@]}")
echo "### M1 $stamp node=$node image=${IMAGE:7:12} reader=${M1_READER:-served}"
echo "### argv: $(printf '%q ' "${argv[@]}")"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing launched"
  exit 0
fi
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 1800 "${argv[@]}" 2>&1 | tee "$R/m1-$stamp.log"
status=${PIPESTATUS[0]}
echo "### exit $status; report $R/m1-$stamp.json; log $R/m1-$stamp.log"
grep -F 'M1 VERDICT:' "$R/m1-$stamp.log" || true
exit "$status"
