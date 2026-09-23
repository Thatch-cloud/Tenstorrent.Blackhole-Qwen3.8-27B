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
#
# Prefill lever #1 (sdpa-prefill-share-spec.md 3.7, optimisation/ttnn-op/sdpa_prefill_chain):
#   KOPGRAFT_PF=<dir>      the K64g graft, mounted as lever_n_m3native_run_arm.sh mounts a graft: its
#                          _ttnn.so, its _ttnncpp.so (build_Release/ttnn and build_Release/lib) and every
#                          op directory it carries (attn_prep, nlp_concat_heads_decode, sdpa_decode, sdpa),
#                          read-only. Its sdpa/ must hold reader_interleaved_qwen_chain.cpp; inside the
#                          container the mounted .so files must hash to the host files (else exit 97), and
#                          the chain reader joins the source list (its recorded sha below). Refused together
#                          with M1_READER (a file mount inside a directory mount). IMAGE must be set
#                          explicitly to one of PF_GRAFT_IMAGES, the images build_k64g.sh compared the
#                          graft's sdpa/ against (never this script's 0648ca9a default), and the seven
#                          sdpa sources are then enforced (M1_REQUIRE_SOURCES=1).
#   WATCHER=1              TT_METAL_WATCHER=5 (NoC sanitiser, waypoints, asserts), the watcher log in
#                          $RESULTS/watcher-<stamp>/, and a 900 s container cap instead of 1800 s.
#   QWEN_SDPA_PF_TEST=1    passed into the container: the factory then accepts the test-only flags
#                          (0x100 mutation, 0x200 planted hang) of --program-word arms.
# Device holders: refuses if a running container can reach a Tenstorrent device (a tenstorrent
# device or mount, a /dev mount, or --privileged) or a host process holds card M's node (fuser).
set -euo pipefail
S=${M1_SRC:-$HOME/kwork-m1}
IMAGE_EXPLICIT=${IMAGE:+1}
IMAGE=${IMAGE:-sha256:0648ca9ad663acc72e7d8ea59d9cde0f9218b583ad74a60d58ff2f31bddd6fae}
# build_k64g.sh's K64G_IMAGES default (A'' eceb2daa, A' 1b9b6445, e41ef884): a KOPGRAFT_PF run is refused
# in any other image (the CPU tests keep the two lists equal).
PF_GRAFT_IMAGES="sha256:eceb2daa744c3345368a638488a804f0ffe8b76f6b0680945a47bb527bdb55a9 sha256:1b9b644549d4409c4fc80e2f92c183e665e7e693c37cccc95b4bb760a6d7d537 sha256:e41ef884f4c8e07ce6632511fac19364af35dc73d601f05f5191784a64a3a768"
REQUIRE_SOURCES=${M1_REQUIRE_SOURCES:-0}
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

# KOPGRAFT_PF: the K64g graft, exactly the arm's mount targets.
CHAIN_READER=$SDPA/kernels/dataflow/reader_interleaved_qwen_chain.cpp
CHAIN_READER_SHA=eecc1166a209e61dc8498149b5a4338cc278d7106f4ca68d6434f620e942f8d8
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations
graft_mounts=()
graft_lines=()
if [ -n "${KOPGRAFT_PF:-}" ]; then
  if [ -n "${M1_READER:-}" ]; then
    echo "refusing: M1_READER and KOPGRAFT_PF together (a file mount inside the graft's sdpa/ mount)" >&2
    exit 1
  fi
  if [ -z "$IMAGE_EXPLICIT" ]; then
    echo "refusing: KOPGRAFT_PF needs IMAGE set explicitly to one of the images build_k64g.sh compared: $PF_GRAFT_IMAGES" >&2
    exit 1
  fi
  case " $PF_GRAFT_IMAGES " in
    *" $IMAGE "*) ;;
    *) echo "refusing: KOPGRAFT_PF in image $IMAGE, which build_k64g.sh did not compare (allowed: $PF_GRAFT_IMAGES)" >&2; exit 1 ;;
  esac
  REQUIRE_SOURCES=1
  graft=$(readlink -f -- "$KOPGRAFT_PF") || { echo "refusing: KOPGRAFT_PF=$KOPGRAFT_PF does not resolve" >&2; exit 1; }
  case "$graft" in
    *,*) echo "refusing: KOPGRAFT_PF path contains a comma (breaks --mount): $graft" >&2; exit 1 ;;
  esac
  for part in _ttnn.so _ttnncpp.so sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp; do
    test -s "$graft/$part" || { echo "refusing: $graft/$part missing (build it with build_k64g.sh)" >&2; exit 1; }
  done
  graft_mounts+=(--mount "type=bind,src=$graft/_ttnn.so,dst=/opt/tt-metal/ttnn/ttnn/_ttnn.so,readonly")
  graft_mounts+=(--mount "type=bind,src=$graft/_ttnncpp.so,dst=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so,readonly")
  graft_mounts+=(--mount "type=bind,src=$graft/_ttnncpp.so,dst=/opt/tt-metal/build_Release/lib/_ttnncpp.so,readonly")
  for op in attn_prep:transformer/attn_prep nlp_concat_heads_decode:experimental/transformer/nlp_concat_heads_decode \
            sdpa_decode:transformer/sdpa_decode sdpa:transformer/sdpa; do
    if [ -d "$graft/${op%%:*}" ]; then
      graft_mounts+=(--mount "type=bind,src=$graft/${op%%:*},dst=$OPS/${op#*:},readonly")
    fi
  done
  so_sha=$(sha256sum "$graft/_ttnn.so" | cut -d' ' -f1)
  cpp_sha=$(sha256sum "$graft/_ttnncpp.so" | cut -d' ' -f1)
  graft_lines=("$so_sha  /opt/tt-metal/ttnn/ttnn/_ttnn.so"
               "$cpp_sha  /opt/tt-metal/build_Release/ttnn/_ttnncpp.so"
               "$cpp_sha  /opt/tt-metal/build_Release/lib/_ttnncpp.so")
  expect+=("$CHAIN_READER_SHA  $CHAIN_READER")
  echo "### KOPGRAFT_PF $graft _ttnn.so=${so_sha:0:16} _ttnncpp.so=${cpp_sha:0:16} (+ its op directories, read-only)"
fi

# WATCHER=1: the NoC sanitiser pass (spec 6.2 Q1.2); a shorter container cap.
timeout_s=1800
watcher=()
if [ "${WATCHER:-}" = 1 ]; then
  timeout_s=900
  watcher=(-e TT_METAL_WATCHER=5 --mount "type=bind,src=$R/watcher-$stamp,dst=/opt/tt-metal/generated/watcher")
  echo "### WATCHER=1: TT_METAL_WATCHER=5, watcher log $R/watcher-$stamp, container cap ${timeout_s} s"
fi
pf_test=()
if [ "${QWEN_SDPA_PF_TEST:-}" = 1 ]; then
  pf_test=(-e QWEN_SDPA_PF_TEST=1)
  echo "### QWEN_SDPA_PF_TEST=1: the factory accepts the test-only flags 0x100 / 0x200"
fi

if [ "$DRY" = 1 ]; then
  node=$CARD_M
else
  node=$(readlink -f "$CARD_M")
  test -c "$node" || { echo "card M ($CARD_M) is not a device node here" >&2; exit 1; }
  # Never share the card: refuse if any running container can reach a Tenstorrent device (a
  # tenstorrent device or mount, a /dev mount or --privileged: scripts/ci/reset-cards.sh's checks) or a
  # host process holds card M's node.
  for id in $(docker ps -q); do
    info=$(docker inspect "$id" --format '{{.Name}} privileged={{.HostConfig.Privileged}} {{json .HostConfig.Devices}} {{json .Mounts}}')
    if printf '%s' "$info" | grep -qE 'privileged=true|tenstorrent|"Source":"/dev"|"PathOnHost":"/dev"'; then
      echo "refusing: container ${info%% *} holds a Tenstorrent device (a tenstorrent device or mount, a /dev mount, or --privileged)" >&2
      exit 1
    fi
  done
  if command -v fuser >/dev/null 2>&1; then
    holder_pre=()
    holder_scope="this user's processes only (no passwordless sudo)"
    if [ "$(id -u)" = 0 ]; then
      holder_scope=all
    elif sudo -n true >/dev/null 2>&1; then
      holder_pre=(sudo -n)
      holder_scope=all
    fi
    holder_status=0
    holders=$(${holder_pre[@]+"${holder_pre[@]}"} fuser -v "$node" 2>&1) || holder_status=$?
    if [ "$holder_status" = 0 ] || [ -n "$holders" ]; then
      echo "refusing: host processes hold $node (or fuser failed):" >&2
      echo "$holders" >&2
      exit 1
    fi
    echo "### device holders on $node: none ($holder_scope)"
  else
    echo "WARN: fuser is not installed; host processes holding $node were not checked" >&2
  fi
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "refusing: image $IMAGE is not present on this host" >&2; exit 1; }
  mkdir -p "$R" "$R/kcache-m1-$stamp"
  chmod 0777 "$R" "$R/kcache-m1-$stamp"
  if [ "${WATCHER:-}" = 1 ]; then
    mkdir -p "$R/watcher-$stamp"
    chmod 0777 "$R/watcher-$stamp"
  fi
  # What the container checks before the bench: every source line, the mounted bench, and
  # (M1_READER) the mounted reader.
  printf '%s\n' "${expect[@]}" > "$R/m1-$stamp.sources.sha256"
  printf '%s\n' "$bench_sha  /bench/sdpa_prefill_bench.py" > "$R/m1-$stamp.bench.sha256"
  if [ -n "${M1_READER:-}" ]; then
    printf '%s\n' "${expect[1]}" > "$R/m1-$stamp.reader.sha256"
  fi
  if [ -n "${KOPGRAFT_PF:-}" ]; then
    printf '%s\n' "${graft_lines[@]}" > "$R/m1-$stamp.graft.sha256"
    chmod 0644 "$R/m1-$stamp.graft.sha256"
  fi
  chmod 0644 "$R/m1-$stamp.sources.sha256" "$R/m1-$stamp.bench.sha256" "$R/m1-$stamp.reader.sha256" 2>/dev/null || true

  echo "### factory facts in ${IMAGE:7:12} ($FACTORY)"
  timeout -k 10 120 docker run --rm --network none --entrypoint sh "$IMAGE" -c \
    "sha256sum $FACTORY; grep -n -E 'global_q_pair_distribute|use_zigzag_balancing|is_causal && !is_chunked|q_num_chunks % 2' $FACTORY" \
    2>&1 | tee "$R/m1-$stamp.factory.txt" || true
fi

# The in-container preamble: the seven shas (the reader line is the mounted file under M1_READER),
# the mount proof, then exec the bench so its exit status is the container's.
paths=("${expect[@]#*  }")
pre="echo '### in-container sha256 of the served sdpa sources (reader_interleaved.cpp = the M1_READER file when set) and the mounted bench'; "
pre+="sha256sum ${paths[*]} /bench/sdpa_prefill_bench.py || echo '### some sdpa sources are missing in this image'; "
pre+="sha256sum -c /results/m1-$stamp.bench.sha256 || { echo '### REFUSING: the in-container bench is not the host file'; exit 97; }; "
if [ -n "${KOPGRAFT_PF:-}" ]; then
  pre+="echo '### in-container sha256 of the mounted graft binaries'; "
  pre+="sha256sum -c /results/m1-$stamp.graft.sha256 || { echo '### REFUSING: the in-container graft .so files are not the KOPGRAFT_PF files'; exit 97; }; "
fi
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
  ${graft_mounts[@]+"${graft_mounts[@]}"}
  ${watcher[@]+"${watcher[@]}"}
  ${pf_test[@]+"${pf_test[@]}"}
  --mount "type=bind,src=$R,dst=/results"
  --mount "type=bind,src=$R/kcache-m1-$stamp,dst=/kcache"
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/kcache -e OMP_NUM_THREADS=8
  -e "M1_REQUIRE_SOURCES=$REQUIRE_SOURCES"
  --entrypoint sh "$IMAGE" -c "$pre" m1 "${extra[@]}")
echo "### M1 $stamp node=$node image=${IMAGE:7:12} reader=${M1_READER:-served} graft=${KOPGRAFT_PF:-none} watcher=${WATCHER:-0}"
echo "### argv: $(printf '%q ' "${argv[@]}")"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing launched"
  exit 0
fi
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set +e   # keep the exit status of the run itself, below
timeout -k 30 "$timeout_s" "${argv[@]}" 2>&1 | tee "$R/m1-$stamp.log"
status=${PIPESTATUS[0]}
echo "### exit $status; report $R/m1-$stamp.json; log $R/m1-$stamp.log"
grep -F 'M1 VERDICT:' "$R/m1-$stamp.log" || true
grep -F 'M1 Q2:' "$R/m1-$stamp.log" || true
# A hang: the watchdog's exit 3, the container cap (124 / 137), or the bench's faulthandler backstop
# (exit 1 with 'Timeout (' when a blocking ttnn call held the GIL and the watchdog thread never ran).
hung=0
case "$status" in
  3|124|137) hung=1 ;;
  1) grep -qF 'Timeout (' "$R/m1-$stamp.log" && hung=1 ;;
esac
if [ "$hung" = 1 ]; then
  # tt-smi -r takes the tt-smi BOARD index, not the /dev/tenstorrent number: print card M's PCI address.
  bdf=
  if majmin=$(stat -c '%t:%T' "$node" 2>/dev/null); then
    bdf=$(readlink -f "/sys/dev/char/$((16#${majmin%%:*})):$((16#${majmin##*:}))/device" 2>/dev/null || true)
    bdf=${bdf##*/}
  fi
  rank=$(ls /dev/tenstorrent 2>/dev/null | grep -E '^[0-9]+$' | sort -n | grep -nx "${node##*/}" | cut -d: -f1 || true)
  echo "HANG SUSPECTED (exit $status): the container is removed on exit. RESET CARD M ONLY (CEF5729692C19E6D =" \
       "$node, PCI ${bdf:-unknown}): tt-smi -r takes the tt-smi BOARD index, not the /dev/tenstorrent number;" \
       "probable index $([ -n "$rank" ] && echo $((rank - 1)) || echo '?'); CONFIRM with ~/.local/bin/tt-smi -ls" \
       "that it is PCI ${bdf:-unknown}, check gh run list for the qwen-two-p150a-exclusive group, then" \
       "~/.local/bin/tt-smi -r <that index>, then a passing stock smoke run." >&2
fi
exit "$status"
