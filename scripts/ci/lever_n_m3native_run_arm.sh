#!/usr/bin/env bash
# Serve the Lever N M3native gate: mount the native 64-row decode graft (model_config.py,
# attention/tp.py, gdn/tp.py, mlp.py) read-only over the pinned qwen36/tt sources and run
# the four-user packed gate against it.
#
# One arm only, unlike M1/M2's baseline/resumable split: native_m3 is a hasattr-gated
# overlay switch (model_batch.two_tile_bindings), not a scheduler edit with a stock
# counterpart worth serving separately. The same image with the graft mounted IS the
# thing under test; its own [PINDIAG] native_m3 marker and the retired binders' zero
# call counts (lever_n_m3native_gate.py) are the positive controls that the overlay
# actually engaged rather than silently falling back to the two-call path a stock
# image (no _64 attrs) would take unnoticed.
#
# $1 is the image sha, defaulting to the pinned qwen-fp2u-image.yml fast-serving image so
# the graft rides the exact image the four-user runs measure against.
#
# QWEN_FAST_FOUR_AS_TWO=0 is required: serving_runtime.py defaults four scheduler
# requests to TWO 32-row M1 blocks (QWEN_FAST_FOUR_AS_TWO, default ON at count 4), and
# under that default the single 64-row M3 block this gate exists to exercise is never
# built - two_tile_bindings(32) returns () and the [PINDIAG] native_m3 marker can never
# fire (gate 1, run 35556533480: a false negative - no decode round ever ran, even
# though the graft was mounted correctly). Setting it to 0 keeps the single M3 block.
set -euo pipefail
image="${1:-${M3NATIVE_IMAGE:-sha256:e41ef884f4c8e07ce6632511fac19364af35dc73d601f05f5191784a64a3a768}}"
target=/home/thatch/hf-cache/hub/models--Qwen--Qwen3.8-27B
cache=/home/thatch/.cache/qwen-experiments
revision=dedf8df68adfb1afeaf7b7480c0a0243108177b4
root=/opt/tt-metal/models/demos/blackhole/qwen36/tt

# Arm geometry: M3NATIVE_USERS/CONTEXT/PROMPT_TOKENS default to today's four-user 32768
# recipe, so an unset environment reproduces the exact prior invocation. A wider arm (e.g.
# the 131k one-user attach probe) overrides these three; QWEN_DSPARK_REQUEST_CONTEXT tracks
# prompt_tokens below because that is what the frozen recipe's request-context selector
# reads. Setting it past 32768 does NOT admit a wider T16 fast path today - the staged
# combined-runtime recipe is CLI-locked to context 32768 (frozen_recipe_context.py:196-198)
# and every downstream literal it emits is a hard 32768, not derived from this value; see
# docs/lever-n-131k-attach-arm.md for the exact refusal chain. This arm exists to measure
# attach and report, not to reach a decode round.
users="${M3NATIVE_USERS:-4}"
context="${M3NATIVE_CONTEXT:-33024}"
prompt_tokens="${M3NATIVE_PROMPT_TOKENS:-32768}"
dspark_context="$prompt_tokens"
# QWEN_FAST_MAX_POSITION raises dflash_prefill_window's absolute prefill-position ceiling
# (default 65504) so a wider-than-default context can reach prefill at all; only set when
# non-default, so the default arm's docker invocation stays byte-identical to before.
max_position=""
if [ "$context" != "33024" ]; then
  max_position="$context"
fi
# M3NATIVE_ALLOW_MISSING_REFERENCES=1 threads --allow-missing-references to the gate for
# arms with no single-stream reference yet (e.g. 131k): comparisons still record
# reference_present=false and the dram/packed_phase lines still print, but gate_passed no
# longer requires every user to have a reference. Unset by default (empty, no flag added).
allow_missing_references=""
if [ "${M3NATIVE_ALLOW_MISSING_REFERENCES:-}" = "1" ]; then
  allow_missing_references="--allow-missing-references"
fi
# M3NATIVE_PROMPT_SOURCE (synthetic | real-text) and M3NATIVE_EOS (ignore | stop) pass through as the
# gate's --prompt-source / --eos, the same way M3NATIVE_SEQUENTIAL_USERS does. real-text builds each
# user's prompt inside the container from the image's own vLLM source (real_text_prompts.py, mounted
# at /bench below); it has no single-stream reference, so it needs M3NATIVE_ALLOW_MISSING_REFERENCES=1,
# and it needs EOS honoured (the gate refuses --eos ignore with it). Refused here as well, before the
# draft download and the card checks. Unset: neither flag is passed and the gate keeps its defaults.
case "${M3NATIVE_PROMPT_SOURCE:-}" in
  ''|synthetic|real-text) ;;
  *) echo "M3NATIVE_PROMPT_SOURCE must be synthetic or real-text, got '$M3NATIVE_PROMPT_SOURCE'" >&2; exit 1 ;;
esac
case "${M3NATIVE_EOS:-}" in
  ''|ignore|stop) ;;
  *) echo "M3NATIVE_EOS must be ignore or stop, got '$M3NATIVE_EOS'" >&2; exit 1 ;;
esac
if [ "${M3NATIVE_PROMPT_SOURCE:-}" = "real-text" ]; then
  if [ "${M3NATIVE_ALLOW_MISSING_REFERENCES:-}" != "1" ]; then
    echo "M3NATIVE_PROMPT_SOURCE=real-text needs M3NATIVE_ALLOW_MISSING_REFERENCES=1 (no real-text prompt has a single-stream reference)" >&2
    exit 1
  fi
  if [ "${M3NATIVE_EOS:-stop}" != "stop" ]; then
    echo "M3NATIVE_PROMPT_SOURCE=real-text needs M3NATIVE_EOS=stop or unset (the fast path finishes at EOS whatever ignore_eos says)" >&2
    exit 1
  fi
fi

mkdir -p experiment-results draft-config
if [ ! -s draft-config/config.json ]; then
  curl -fsSL --max-time 30 \
    "https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/resolve/$revision/config.json" \
    > draft-config/config.json
fi

mounts=()
for component in attention convolution mlp projection selector; do
  mounts+=(--mount "type=bind,src=$cache/dflash2-$component-$revision,dst=/experiment-dflash-fixture/$component,readonly")
done
for layer in 1 2 3 4; do
  mounts+=(--mount "type=bind,src=$cache/dflash2-stack-$revision/layer-$layer,dst=/experiment-dflash-fixture/layer-$layer,readonly")
done
# The tracked references (scripts/ci/references/packed-gate) are what the runner has;
# a local runner-evidence.local copy, when present, is preferred (freshest local run).
refs="$PWD/scripts/ci/references/packed-gate"
if [ -d "$PWD/runner-evidence.local/packed-gate" ]; then refs="$PWD/runner-evidence.local/packed-gate"; fi
mounts+=(--mount "type=bind,src=$refs,dst=/bench/packed-gate-reference,readonly")

lever_n_mounts=()
if [ -n "${M3NATIVE_PREFILL_CHUNK_TOKENS:-}" ]; then
  plugin=/opt/qwen-fast-plugin/src/vllm_tt_plugin
  lever_n_mounts+=(--mount "type=bind,src=$PWD/graft/model.py,dst=$root/model.py,readonly")
  lever_n_mounts+=(--mount "type=bind,src=$PWD/graft/qwen36_vllm.py,dst=$root/qwen36_vllm.py,readonly")
  lever_n_mounts+=(--mount "type=bind,src=$PWD/graft/platform.py,dst=$plugin/platform.py,readonly")
  lever_n_mounts+=(--mount "type=bind,src=$PWD/graft/scheduler.py,dst=$plugin/scheduler.py,readonly")
  lever_n_mounts+=(--mount "type=bind,src=$PWD/graft/lane_scheduler.py,dst=$plugin/lane_scheduler.py,readonly")
  lever_n_mounts+=(-e TT_M1_FORCE_CHUNKED_PREFILL=1)
fi
# The gate's own results directory (its raw server.log, every line the stdout filter
# drops) persists under experiment-results/gate and ships with the artifact. The
# container runs as root with no CAP_DAC_OVERRIDE, so the directory must be world-writable
# (run 35579223088: the engine died with no Python traceback and the only copy of the
# native crash text was /tmp/m3native-gate/server.log inside the container).
mkdir -p experiment-results/gate
chmod 0777 experiment-results/gate
mounts+=(--mount "type=bind,src=$PWD/experiment-results/gate,dst=/experiment-results-gate")
# The serving pair, by board id: card M (PCI d1) and card A (PCI f3). Every node used to be mounted,
# which was right while only two boards were present; with the third board (B) back the container would
# see three devices. M3NATIVE_CARDS overrides the by-id list (space-separated); a missing card refuses.
# M3NATIVE_STAGGER (seconds, default 0) spaces the gate's request starts so vLLM admits the users in order
# 0,1,2,3. With 0 the four threads race and the admission order - which fixes each user's slot and pair row -
# changes from run to run; the pair drafter drafts row 1 differently from row 0 (h1a-draft-race.md), so draft
# sequences and acceptance then differ between identical runs. Timed ABAB arms should set it.
stagger="${M3NATIVE_STAGGER:-0}"
if ! printf '%s' "$stagger" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
  echo "M3NATIVE_STAGGER must be a non-negative number of seconds, got '$stagger'" >&2
  exit 1
fi

serving_cards="${M3NATIVE_CARDS:-blackhole-CEF5729692C19E6D blackhole-3707293C249A5E67}"
# The reset step hot-cycles the boards behind the PCIe switch and returns before pciehp has
# re-enumerated them, so a by-id link can be missing for a few seconds (run 35930349210 refused
# on card A one second after the reset). Wait up to M3NATIVE_CARD_WAIT_S for every link, then
# settle and resolve again: a node that moved while we waited is refused, not guessed.
card_wait_s="${M3NATIVE_CARD_WAIT_S:-120}"
resolve_serving_nodes() {
  nodes=()
  local card node
  for card in $serving_cards; do
    node=$(readlink -f "/dev/tenstorrent/by-id/$card" 2>/dev/null || true)
    if [ -z "$node" ] || [ ! -c "$node" ]; then
      missing_card="$card"
      return 1
    fi
    nodes+=("${node##*/}")
  done
  return 0
}
card_waited=0
missing_card=
until resolve_serving_nodes; do
  if [ "$card_waited" -ge "$card_wait_s" ]; then
    echo "serving card $missing_card has no device node under /dev/tenstorrent/by-id after ${card_waited}s; refusing to guess" >&2
    ls -la /dev/tenstorrent/by-id >&2 || true
    # The likely cause (run v190): the telemetry race after the reset. Named, never acted on here.
    case $missing_card in
      blackhole-CEF5729692C19E6D) missing_pci=0000:d1:00.0 ;;
      blackhole-3707293C249A5E67) missing_pci=0000:f3:00.0 ;;
      *) missing_pci='<its PCI address>' ;;
    esac
    {
      echo "hint: likely the telemetry race after a reset: the board re-enumerated before its ARC firmware was ready."
      echo "hint:   The driver logs 'tenstorrent $missing_pci: Telemetry not available'; its node exists and tt-smi -ls"
      echo "hint:   lists it, but udev never creates its by-id link, so no wait here brings it back. The gate's reset"
      echo "hint:   step re-probes the driver for this (serving_pair_heal, scripts/ci/serving_pair.sh); if it ran, see"
      echo "hint:   its [reset] lines in experiment-results/reset.log. By hand, once no process holds the node:"
      echo "hint:     echo $missing_pci | sudo -n tee /sys/bus/pci/drivers/tenstorrent/unbind; sleep 3"
      echo "hint:     echo $missing_pci | sudo -n tee /sys/bus/pci/drivers/tenstorrent/bind"
      telemetry=$({ dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null || true; } | grep -F 'Telemetry not available' | tail -n 4 || true)
      if [ -n "$telemetry" ]; then
        echo "hint: dmesg, last 'Telemetry not available' lines:"
        printf '%s\n' "$telemetry" | sed 's/^/hint:   /'
      else
        echo "hint: no 'Telemetry not available' line readable in dmesg here: the cause may be another."
      fi
    } >&2
    exit 1
  fi
  sleep 2
  card_waited=$((card_waited + 2))
done
if [ "$card_waited" -gt 0 ]; then
  echo "serving pair by-id links present after ${card_waited}s; settling"
  first_nodes="${nodes[*]}"
  sleep 5
  if ! resolve_serving_nodes || [ "${nodes[*]}" != "$first_nodes" ]; then
    echo "serving pair moved while settling (${first_nodes} -> ${nodes[*]:-missing}); refusing to guess" >&2
    exit 1
  fi
fi
if [ "${#nodes[@]}" -ne 2 ]; then
  echo "expected exactly two serving cards, got ${#nodes[@]} (${serving_cards})" >&2
  exit 1
fi
mapfile -t nodes < <(printf '%s\n' "${nodes[@]}" | sort -n)
devices=()
for node in "${nodes[@]}"; do devices+=(--device "/dev/tenstorrent/$node"); done
echo "serving pair: ${serving_cards} -> /dev/tenstorrent/{$(IFS=,; echo "${nodes[*]}")}"

# Optional K64 kernel graft (~/opgraft-K64: the batch-64 attn_decode_prep and
# nlp_concat_heads_decode C++ ops, both proved bit-exact on device). Unlike native_m3
# this patches no Python source, only the two ops and their bindings, so it is opt-in
# by env (KOPGRAFT64=<dir>) rather than hasattr-detected; default empty mounts nothing
# and passes no QWEN_FAST_NATIVE_ATTN, today's exact two-call behaviour
# (two_tile_decode.native_attn_enabled). Mirrors ~/kwork64/test-k64.sh on the rig.
#
# The grafted _ttnncpp.so also replaces the combined-runtime binary that
# dflash_combined_sim_runtime.BINARY_SHA256 pins, so its hash is passed through as
# QWEN_FAST_RUNTIME_BINARY_SHA256 for runtime_binary_override to admit at attach.
# Run 35558196471 mounted an earlier K64 graft binary and was refused at attach by
# that pin: it had been built over the ORIGINAL SDPA factory, not the COMBINED one
# the pin requires, so admitting it would have changed the qualified draft fp32 SDPA
# intermediates unnoticed. ~/opgraft-K64c on the rig is the same kernel graft rebuilt
# over the combined SDPA factory fd8c0676 on 2026-09-21 03:47 UTC, so it passes both
# runtime_binary_override checks (every pinned path matches, and the factory hash is
# the combined one) and is admitted for measurement without touching the pin itself.
#
# A graft that carries an sdpa_decode op directory (~/opgraft-K64e onward: the [QWEN-SDPA]
# factory branch in _ttnncpp.so, plus reader_decode_qwen.cpp / sdpa_flash_decode_qwen.cpp,
# see optimisation/ttnn-op/sdpa_decode_qwen) also mounts that directory over the image's.
# Device kernels are JIT-compiled from the op directory at dispatch, so without it the new
# kernel files do not exist in the container. The graft's copy is the image's directory
# plus those two files only - its factory .cpp is the audited 3e0a69af, not the compiled
# one, so sdpa_tree_scratch.audit(patched=True) still passes (build_k64e.sh checks this).
# K64c/K64d have no such directory, so their runs are unchanged. Such a graft also gets
# its own JIT cache, keyed by its kernels' bytes: the kernel cache hash is not known to
# cover file contents, so a revised kernel at the same path could otherwise reuse a stale
# binary (the default arm keeps /experiment-cache/kernels).
#
# Stage 4 (~/opgraft-K64i onward, optimisation/ttnn-op/sdpa_decode_slice) adds
# reader_decode_qwen_slice.cpp and writer_decode_qwen_slice.cpp, which the factory selects for
# flag 0x4 / 0x8 (M3NATIVE_SDPA_MODES slice / readahead). Both must be present when the graft
# carries either one, or when either mode is requested; those modes are refused before the
# run without a grafted sdpa_decode directory (no image serves the stage-4 factory) and without
# the stage-4 literal in the graft's _ttnncpp.so (pooled_attention_replay would refuse it too,
# but only after the model had loaded). The cache key covers EVERY *qwen*.cpp in the kernels
# tree: the two stage-3 kernels' bytes first - exactly the key's input before stage 4 - then
# one '<sha256> <path>' line per further *qwen*.cpp in byte order. A graft with no others
# (K64e..K64g) therefore keeps its key and its warm cache, and a revised slice kernel (or one
# added, removed or renamed) gets a fresh one.
KM=""
graft_binary_sha=""
kernel_cache=/experiment-cache/kernels
sdpa_kernels=""
sdpa_stage4_modes=""
sdpa_modes_compact="${M3NATIVE_SDPA_MODES:-}"
case ",${sdpa_modes_compact//[[:space:]]/}," in
  *,slice,*|*,readahead,*) sdpa_stage4_modes=1 ;;
esac
if [ -n "${KOPGRAFT64:-}" ]; then
  KM="$KM -v $KOPGRAFT64/_ttnn.so:/opt/tt-metal/ttnn/ttnn/_ttnn.so:ro"
  KM="$KM -v $KOPGRAFT64/_ttnncpp.so:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so:ro"
  KM="$KM -v $KOPGRAFT64/_ttnncpp.so:/opt/tt-metal/build_Release/lib/_ttnncpp.so:ro"
  KM="$KM -v $KOPGRAFT64/attn_prep:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/attn_prep:ro"
  KM="$KM -v $KOPGRAFT64/nlp_concat_heads_decode:/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode:ro"
  if [ -d "$KOPGRAFT64/sdpa_decode" ]; then
    sdpa_kernels="$KOPGRAFT64/sdpa_decode/device/kernels"
    sdpa_required=(dataflow/reader_decode_qwen.cpp compute/sdpa_flash_decode_qwen.cpp)
    sdpa_slice=(dataflow/reader_decode_qwen_slice.cpp dataflow/writer_decode_qwen_slice.cpp)
    if [ -n "$sdpa_stage4_modes" ] || [ -e "$sdpa_kernels/${sdpa_slice[0]}" ] || [ -e "$sdpa_kernels/${sdpa_slice[1]}" ]; then
      sdpa_required+=("${sdpa_slice[@]}")
    fi
    for kernel in "${sdpa_required[@]}"; do
      if [ ! -s "$sdpa_kernels/$kernel" ]; then
        echo "KOPGRAFT64 sdpa_decode directory lacks $kernel" >&2
        exit 1
      fi
    done
    if [ -n "$sdpa_stage4_modes" ] && ! grep -a -q -F -- '[QWEN-SDPA] q-slice rows_per_kv=' "$KOPGRAFT64/_ttnncpp.so"; then
      echo "M3NATIVE_SDPA_MODES='$M3NATIVE_SDPA_MODES' names slice or readahead, but $KOPGRAFT64/_ttnncpp.so lacks" \
        "'[QWEN-SDPA] q-slice rows_per_kv=' (not a K64i-or-later build)" >&2
      exit 1
    fi
    KM="$KM -v $KOPGRAFT64/sdpa_decode:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode:ro"
    sdpa_qwen_others=$(cd "$sdpa_kernels" && find . -type f -name '*qwen*.cpp' ! -path ./dataflow/reader_decode_qwen.cpp \
      ! -path ./compute/sdpa_flash_decode_qwen.cpp | LC_ALL=C sort | while IFS= read -r file; do
        printf '%s %s\n' "$(sha256sum < "$file" | cut -c1-64)" "$file"
      done)
    kernel_cache="/experiment-cache/kernels-qwen-$({ cat "$sdpa_kernels/dataflow/reader_decode_qwen.cpp" \
      "$sdpa_kernels/compute/sdpa_flash_decode_qwen.cpp"; printf '%s' "$sdpa_qwen_others"; } | sha256sum | cut -c1-12)"
    sdpa_qwen_named="none"
    [ -z "$sdpa_qwen_others" ] || sdpa_qwen_named=$(printf '%s\n' "$sdpa_qwen_others" | cut -d' ' -f2- | tr '\n' ' ')
    echo "sdpa_decode op directory grafted from $KOPGRAFT64; kernel cache $kernel_cache" \
      "(keyed on the stage-3 pair and further *qwen*.cpp: ${sdpa_qwen_named% })"
  elif [ -n "$sdpa_stage4_modes" ]; then
    echo "M3NATIVE_SDPA_MODES='$M3NATIVE_SDPA_MODES' names slice or readahead, but $KOPGRAFT64 has no sdpa_decode directory" \
      "(the slice kernels are JIT-compiled from a K64i-or-later graft's)" >&2
    exit 1
  fi
  graft_binary_sha=$(sha256sum "$KOPGRAFT64/_ttnncpp.so" | cut -c1-64)
elif [ -n "$sdpa_stage4_modes" ]; then
  echo "M3NATIVE_SDPA_MODES='$M3NATIVE_SDPA_MODES' names slice or readahead, which needs KOPGRAFT64 (a K64i-or-later graft:" \
    "no image serves the stage-4 factory or its slice kernels)" >&2
  exit 1
fi
# M3NATIVE_SDPA_PF=1 (prefill lever #1, optimisation/ttnn-op/sdpa_prefill_chain) becomes QWEN_FAST_SDPA_PF=1,
# read by the grafted attention/tp.py (lever_n_m3native_patch section I), which then puts the G6 K/V chain
# word on the chunked prefill SDPA calls card M qualified. It needs a K64g-or-later graft: the factory
# branch is in its _ttnncpp.so (mounted by the block above), and the chain reader is JIT-compiled from the
# op directory, so the graft's sdpa/ is mounted over the image's - ONE directory, the way sdpa_decode is,
# never anything under /experiment-scripts/ci. Every kernel the JIT builds from sdpa/ then comes from the
# graft's copy: the served prefill reader and writer, the draft SDPA, and sdpa_flash_decode_qwen.cpp's
# include of compute_common.hpp - and the attach pins cover only some of those files (not
# dataflow_common.hpp or chain_link.hpp). So, before the run, the graft's sdpa/ must equal THIS run's
# image's plus reader_interleaved_qwen_chain.cpp and nothing else (build_k64g.sh step 7's own test, here
# against $image, whichever image the tag names), and the graft must verify against its MANIFEST.sha256.
# Also refused here: a graft without the chain reader, or whose _ttnncpp.so lacks the factory's
# '[QWEN-SDPA-PF] flags=' literal (the model refuses such a binary too, but only after attach).
# M3NATIVE_SDPA_PF_FLAGS (default 0x3, the Q2 choice) must be a production flag set and is always passed,
# so the launched argv names the flags the gate then requires in the factory's log line; set without
# M3NATIVE_SDPA_PF=1 it is refused (it would pass nothing and an intended "on" arm would measure off).
# The JIT cache is also keyed on the whole mounted sdpa/ tree (every file's bytes: the reader includes
# its headers from that tree). Unset: nothing is mounted, passed or checked, and the cache is unchanged.
sdpa_pf_env=()
if [ -n "${M3NATIVE_SDPA_PF:-}" ]; then
  if [ "$M3NATIVE_SDPA_PF" != "1" ]; then
    echo "M3NATIVE_SDPA_PF must be 1 or unset, got '$M3NATIVE_SDPA_PF'" >&2
    exit 1
  fi
  if [ -z "${KOPGRAFT64:-}" ]; then
    echo "M3NATIVE_SDPA_PF=1 needs KOPGRAFT64 (a K64g-or-later graft: the chain factory is in its _ttnncpp.so)" >&2
    exit 1
  fi
  sdpa_pf_flags="${M3NATIVE_SDPA_PF_FLAGS:-0x3}"
  case "$sdpa_pf_flags" in
    0x1|0x3|0x5|0x7) ;;
    *) echo "M3NATIVE_SDPA_PF_FLAGS must be a production flag set (0x1, 0x3, 0x5, 0x7), got '$sdpa_pf_flags'" >&2; exit 1 ;;
  esac
  sdpa_pf_reader="$KOPGRAFT64/sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp"
  if [ ! -s "$sdpa_pf_reader" ]; then
    echo "M3NATIVE_SDPA_PF: $KOPGRAFT64 lacks sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp" >&2
    exit 1
  fi
  if ! grep -a -q -F -- '[QWEN-SDPA-PF] flags=' "$KOPGRAFT64/_ttnncpp.so"; then
    echo "M3NATIVE_SDPA_PF: $KOPGRAFT64/_ttnncpp.so lacks '[QWEN-SDPA-PF] flags=' (not a K64g-or-later build)" >&2
    exit 1
  fi
  if [ ! -s "$KOPGRAFT64/MANIFEST.sha256" ] || ! (cd "$KOPGRAFT64" && sha256sum -c --quiet MANIFEST.sha256 >&2); then
    echo "M3NATIVE_SDPA_PF: $KOPGRAFT64/MANIFEST.sha256 is missing or does not verify (not the graft build_k64g.sh made)" >&2
    exit 1
  fi
  sdpa_pf_target=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa
  sdpa_pf_work=$(mktemp -d)
  if ! sdpa_pf_cid=$(docker create --network none --entrypoint true "$image") || [ -z "$sdpa_pf_cid" ]; then
    rm -rf "$sdpa_pf_work"
    echo "M3NATIVE_SDPA_PF: could not create a container of $image to compare its sdpa/ with the graft's" >&2
    exit 1
  fi
  if ! docker cp "$sdpa_pf_cid:$sdpa_pf_target" "$sdpa_pf_work/image-sdpa" >/dev/null; then
    docker rm -f "$sdpa_pf_cid" >/dev/null 2>&1 || true
    rm -rf "$sdpa_pf_work"
    echo "M3NATIVE_SDPA_PF: could not copy $sdpa_pf_target out of $image" >&2
    exit 1
  fi
  docker rm -f "$sdpa_pf_cid" >/dev/null 2>&1 || true
  sdpa_pf_differences=$(diff -rq "$sdpa_pf_work/image-sdpa" "$KOPGRAFT64/sdpa" || true)
  rm -rf "$sdpa_pf_work"
  sdpa_pf_expected="Only in $KOPGRAFT64/sdpa/device/kernels/dataflow: reader_interleaved_qwen_chain.cpp"
  if [ "$sdpa_pf_differences" != "$sdpa_pf_expected" ]; then
    echo "M3NATIVE_SDPA_PF: $KOPGRAFT64/sdpa is not image $image's sdpa/ plus reader_interleaved_qwen_chain.cpp alone:" >&2
    echo "$sdpa_pf_differences" >&2
    exit 1
  fi
  echo "SDPA prefill K/V chain (lever #1): $KOPGRAFT64/sdpa = image's sdpa/ + reader_interleaved_qwen_chain.cpp; manifest verified"
  KM="$KM -v $KOPGRAFT64/sdpa:$sdpa_pf_target:ro"
  sdpa_pf_tree=$(cd "$KOPGRAFT64/sdpa" && find . -type f | LC_ALL=C sort | while IFS= read -r file; do
    printf '%s %s\n' "$(sha256sum < "$file" | cut -c1-64)" "$file"
  done | sha256sum | cut -c1-12)
  kernel_cache="$kernel_cache-pf-$sdpa_pf_tree"
  sdpa_pf_env=(-e QWEN_FAST_SDPA_PF=1 -e "QWEN_FAST_SDPA_PF_FLAGS=$sdpa_pf_flags")
  echo "SDPA prefill K/V chain (lever #1): sdpa op directory grafted from $KOPGRAFT64, flags $sdpa_pf_flags; kernel cache $kernel_cache"
elif [ -n "${M3NATIVE_SDPA_PF_FLAGS:-}" ]; then
  echo "M3NATIVE_SDPA_PF_FLAGS='$M3NATIVE_SDPA_PF_FLAGS' without M3NATIVE_SDPA_PF=1 passes nothing (the run would be the served prefill): set both or neither" >&2
  exit 1
fi
# M3NATIVE_SDPA_MODES (e.g. 'tail') becomes QWEN_FAST_SDPA_MODES, read by
# pooled_attention_replay.sdpa_modes inside the engine. That module is part of the baked
# evidence tree (/experiment-scripts/ci), so an image older than this change would never
# read the flag; the arm therefore also mounts this checkout's copy over that ONE file
# when the flag is set (it is free to edit - not a pinned source - and in both image copy
# lists). Unset, neither the env var nor the mount is added. 'slice' and 'readahead' (stage 4)
# were already checked against the graft above: K64i-or-later binary and slice kernels.
sdpa_mode_mounts=()
if [ -n "${M3NATIVE_SDPA_MODES:-}" ]; then
  sdpa_mode_mounts+=(--mount "type=bind,src=$PWD/scripts/ci/pooled_attention_replay.py,dst=/experiment-scripts/ci/pooled_attention_replay.py,readonly")
fi
# M3NATIVE_GDN_PREFILL_CONV=1 (lever #2) becomes QWEN_FAST_GDN_PREFILL_CONV=1, read by the grafted
# gdn/tp.py (lever_n_m3native_patch section H), and mounts the op beside it: the module and its
# three kernels, ONE FILE EACH (never a directory over the image's gdn/), from the graft job's
# staging. The list is lever_n_m3native_patch.PREFILL_CONV_FILES itself, so the mounts cannot
# drift from what the graft stages. M3NATIVE_GDN_PREFILL_CONV_AUDIT=<n> runs the FIR beside the
# first n calls (QWEN_FAST_GDN_PREFILL_CONV_AUDIT). Unset, nothing is mounted or passed.
prefill_conv_mounts=()
if [ -n "${M3NATIVE_GDN_PREFILL_CONV:-}" ]; then
  mapfile -t prefill_conv_files < <(python3 -B -c 'import sys; sys.path.insert(0, "scripts/ci"); import lever_n_m3native_patch as p; print(chr(10).join(sorted(p.PREFILL_CONV_FILES)))' | tr -d '\r')
  if [ "${#prefill_conv_files[@]}" -eq 0 ]; then
    echo "M3NATIVE_GDN_PREFILL_CONV: lever_n_m3native_patch.PREFILL_CONV_FILES is empty or unreadable" >&2
    exit 1
  fi
  for relative in "${prefill_conv_files[@]}"; do
    if [ ! -s "$PWD/graft/$relative" ]; then
      echo "M3NATIVE_GDN_PREFILL_CONV: graft/$relative was not staged" >&2
      exit 1
    fi
    prefill_conv_mounts+=(--mount "type=bind,src=$PWD/graft/$relative,dst=$root/$relative,readonly")
  done
  echo "GDN prefill conv (lever #2): mounting ${prefill_conv_files[*]}"
fi
# M3NATIVE_C1_EXACT=1 (C1e, lever_n_m3native_patch section J) becomes QWEN_FAST_C1_EXACT=1, read by the
# grafted mlp.py and layer.py: under QWEN_FAST_SINGLE_GATEUP=1 the prefill MLP runs the served fused op
# again, on a per-layer rebuild of its packed weight in one scratch. The op is mounted beside mlp.py,
# ONE FILE EACH, from the graft job's staging; the list is lever_n_m3native_patch.C1E_FILES itself.
# M3NATIVE_C1_EXACT_AUDIT=<n> (0..64) byte-checks the first n layers' served weight at load and the
# first n packs (QWEN_FAST_C1_EXACT_AUDIT; a correctness arm). C1e needs M3NATIVE_SINGLE_GATEUP (alone
# the served path runs and the arm would measure nothing new) and excludes M3NATIVE_C1_AGMM and
# M3NATIVE_C1_LEGACY (the graft refuses the pair at construction): refused here, before the docker run
# rather than minutes into the model load. Unset, nothing is mounted or passed.
c1e_mounts=()
if [ -n "${M3NATIVE_C1_EXACT:-}" ]; then
  if [ "$M3NATIVE_C1_EXACT" != "1" ]; then
    echo "M3NATIVE_C1_EXACT must be 1 or unset, got '$M3NATIVE_C1_EXACT'" >&2
    exit 1
  fi
  if [ -z "${M3NATIVE_SINGLE_GATEUP:-}" ]; then
    echo "M3NATIVE_C1_EXACT=1 needs M3NATIVE_SINGLE_GATEUP=1 (alone the served packed copy is built and C1e never runs)" >&2
    exit 1
  fi
  if [ -n "${M3NATIVE_C1_AGMM:-}" ] || [ -n "${M3NATIVE_C1_LEGACY:-}" ]; then
    echo "M3NATIVE_C1_EXACT=1 excludes M3NATIVE_C1_AGMM and M3NATIVE_C1_LEGACY (each replaces the same prefill gate/up)" >&2
    exit 1
  fi
  case "${M3NATIVE_C1_EXACT_AUDIT:-0}" in
    ''|*[!0-9]*) echo "M3NATIVE_C1_EXACT_AUDIT must be an integer 0..64, got '$M3NATIVE_C1_EXACT_AUDIT'" >&2; exit 1 ;;
  esac
  if [ "${M3NATIVE_C1_EXACT_AUDIT:-0}" -gt 64 ]; then
    echo "M3NATIVE_C1_EXACT_AUDIT must be an integer 0..64, got '$M3NATIVE_C1_EXACT_AUDIT'" >&2
    exit 1
  fi
  mapfile -t c1e_files < <(python3 -B -c 'import sys; sys.path.insert(0, "scripts/ci"); import lever_n_m3native_patch as p; print(chr(10).join(sorted(p.C1E_FILES)))' | tr -d '\r')
  if [ "${#c1e_files[@]}" -eq 0 ]; then
    echo "M3NATIVE_C1_EXACT: lever_n_m3native_patch.C1E_FILES is empty or unreadable" >&2
    exit 1
  fi
  for relative in "${c1e_files[@]}"; do
    if [ ! -s "$PWD/graft/$relative" ]; then
      echo "M3NATIVE_C1_EXACT: graft/$relative was not staged" >&2
      exit 1
    fi
    c1e_mounts+=(--mount "type=bind,src=$PWD/graft/$relative,dst=$root/$relative,readonly")
  done
  echo "C1e (exact prefill gate/up under the single copy): mounting ${c1e_files[*]}; audit ${M3NATIVE_C1_EXACT_AUDIT:-0}"
elif [ -n "${M3NATIVE_C1_EXACT_AUDIT:-}" ]; then
  echo "M3NATIVE_C1_EXACT_AUDIT='$M3NATIVE_C1_EXACT_AUDIT' without M3NATIVE_C1_EXACT=1 audits nothing: set both or neither" >&2
  exit 1
fi
# M3NATIVE_PADDED_BLOCK=1 (variable-user packed rounds M2) becomes QWEN_FAST_PADDED_BLOCK=1: the 64-row
# block also serves two or three live users as one pass, the missing segments idle on page 0
# (packed_verifier.py, VARIABLE-USER ROUNDS). M3NATIVE_PADDED_BLOCK_MIN_USERS=<2|3> becomes
# QWEN_FAST_PADDED_BLOCK_MIN_USERS (default 2: two idle segments are all page 0 holds). serving_runtime
# admits the flag at the four-user M3 block only and refuses it at attach anywhere else; refused here
# first, before the docker run. Unset, nothing is passed.
if [ -n "${M3NATIVE_PADDED_BLOCK:-}" ]; then
  if [ "$M3NATIVE_PADDED_BLOCK" != "1" ]; then
    echo "M3NATIVE_PADDED_BLOCK must be 1 or unset, got '$M3NATIVE_PADDED_BLOCK'" >&2
    exit 1
  fi
  if [ "$users" != "4" ] || [ -n "${M3NATIVE_SEQUENTIAL_USERS:-}" ]; then
    echo "M3NATIVE_PADDED_BLOCK=1 serves the four-user 64-row block only (users=$users, sequential '${M3NATIVE_SEQUENTIAL_USERS:-}')" >&2
    exit 1
  fi
  case "${M3NATIVE_PADDED_BLOCK_MIN_USERS:-2}" in
    2|3) ;;
    *) echo "M3NATIVE_PADDED_BLOCK_MIN_USERS must be 2 or 3, got '$M3NATIVE_PADDED_BLOCK_MIN_USERS'" >&2; exit 1 ;;
  esac
  echo "padded block (variable-user rounds M2): min_users ${M3NATIVE_PADDED_BLOCK_MIN_USERS:-2}"
elif [ -n "${M3NATIVE_PADDED_BLOCK_MIN_USERS:-}" ]; then
  echo "M3NATIVE_PADDED_BLOCK_MIN_USERS='$M3NATIVE_PADDED_BLOCK_MIN_USERS' without M3NATIVE_PADDED_BLOCK=1 pads nothing: set both or neither" >&2
  exit 1
fi
# Round-fence plan H1a (verify_prestage.py; every flag default off). M3NATIVE_PRESTAGE=1 becomes
# QWEN_FAST_PRESTAGE=1: the drafts' fence window pre-stages the next packed verify and the verify
# writes only what differs (value diff). M3NATIVE_PRESTAGE_AUDIT=1 becomes QWEN_FAST_PRESTAGE_AUDIT=1:
# a rotating 8 staged buffers read back after every verify-time write. M3NATIVE_ROUND_FENCES=1 becomes
# QWEN_FAST_ROUND_FENCES=1: the fence diet (F3, F8, the first-commit validate). The window exists only
# under the packed proposal coordinator, so the pre-stage needs M3NATIVE_PACKED_PROPOSAL=1 and
# M3NATIVE_PIPELINED_PROPOSALS=1; all three serve the packed block only. Refused here, before the
# docker run; unset, nothing is passed.
for h1a_flag in M3NATIVE_PRESTAGE M3NATIVE_PRESTAGE_AUDIT M3NATIVE_ROUND_FENCES; do
  h1a_value="${!h1a_flag:-}"
  if [ -n "$h1a_value" ] && [ "$h1a_value" != "1" ]; then
    echo "$h1a_flag must be 1 or unset, got '$h1a_value'" >&2
    exit 1
  fi
done
if [ -n "${M3NATIVE_PRESTAGE_AUDIT:-}" ] && [ -z "${M3NATIVE_PRESTAGE:-}" ]; then
  echo "M3NATIVE_PRESTAGE_AUDIT=1 without M3NATIVE_PRESTAGE=1 audits nothing: set both or neither" >&2
  exit 1
fi
if [ -n "${M3NATIVE_PRESTAGE:-}${M3NATIVE_ROUND_FENCES:-}" ]; then
  if [ "$users" -lt 2 ] || [ -n "${M3NATIVE_SEQUENTIAL_USERS:-}" ]; then
    echo "M3NATIVE_PRESTAGE / M3NATIVE_ROUND_FENCES serve the packed block only (users=$users, sequential '${M3NATIVE_SEQUENTIAL_USERS:-}')" >&2
    exit 1
  fi
  if [ -n "${M3NATIVE_PRESTAGE:-}" ] && { [ "${M3NATIVE_PACKED_PROPOSAL:-}" != "1" ] || [ "${M3NATIVE_PIPELINED_PROPOSALS:-}" != "1" ]; }; then
    echo "M3NATIVE_PRESTAGE=1 needs M3NATIVE_PACKED_PROPOSAL=1 and M3NATIVE_PIPELINED_PROPOSALS=1 (the coordinator's fence window is where it runs)" >&2
    exit 1
  fi
  echo "round-fence plan H1a: prestage ${M3NATIVE_PRESTAGE:-0} audit ${M3NATIVE_PRESTAGE_AUDIT:-0} round fences ${M3NATIVE_ROUND_FENCES:-0} pipelined publish ${M3NATIVE_PIPELINED_PUBLISH:-0}"
fi
# Round-fence plan H1b (fused_commit.py; every flag default off). M3NATIVE_FUSED_COMMIT=1 becomes
# QWEN_FAST_FUSED_COMMIT=1: each packed segment's feature and K/V projection as one captured trace
# (T_proj) and the publication overrides. M3NATIVE_FUSED_COMMIT_INPLACE=1 (QWEN_FAST_FUSED_COMMIT_INPLACE)
# slides the live K/V banks in place through per-(segment, prefix) traces; M3NATIVE_FUSED_COMMIT_LIVE_BANKS=1
# (QWEN_FAST_FUSED_COMMIT_LIVE_BANKS, needs _INPLACE) binds the packed pair traces to the live banks (F4);
# M3NATIVE_FUSED_COMMIT_AUDIT=1 (QWEN_FAST_FUSED_COMMIT_AUDIT) shadows every fused publication with today's
# eager one and compares (a correctness arm). The block serves the packed block only; the fused path needs a
# captured proposal (C7's condition: M3NATIVE_TRACED_PROPOSAL=1) and, because T_proj is one more trace that
# replays every round, the pair's per-round mask refresh (M3NATIVE_PAIR_MASK_REFRESH=1); F4 needs the packed
# pairs (M3NATIVE_PACKED_PROPOSAL=1). Refused here, before the docker run; unset, nothing is passed.
for h1b_flag in M3NATIVE_FUSED_COMMIT M3NATIVE_FUSED_COMMIT_INPLACE M3NATIVE_FUSED_COMMIT_LIVE_BANKS M3NATIVE_FUSED_COMMIT_AUDIT; do
  h1b_value="${!h1b_flag:-}"
  if [ -n "$h1b_value" ] && [ "$h1b_value" != "1" ]; then
    echo "$h1b_flag must be 1 or unset, got '$h1b_value'" >&2
    exit 1
  fi
done
if [ -z "${M3NATIVE_FUSED_COMMIT:-}" ] && [ -n "${M3NATIVE_FUSED_COMMIT_INPLACE:-}${M3NATIVE_FUSED_COMMIT_LIVE_BANKS:-}${M3NATIVE_FUSED_COMMIT_AUDIT:-}" ]; then
  echo "M3NATIVE_FUSED_COMMIT_INPLACE / _LIVE_BANKS / _AUDIT without M3NATIVE_FUSED_COMMIT=1 do nothing: set it or none" >&2
  exit 1
fi
if [ -n "${M3NATIVE_FUSED_COMMIT_LIVE_BANKS:-}" ] && [ -z "${M3NATIVE_FUSED_COMMIT_INPLACE:-}" ]; then
  echo "M3NATIVE_FUSED_COMMIT_LIVE_BANKS=1 needs M3NATIVE_FUSED_COMMIT_INPLACE=1 (out of place the live bank moves every commit)" >&2
  exit 1
fi
if [ -n "${M3NATIVE_FUSED_COMMIT:-}" ]; then
  if [ "$users" -lt 2 ] || [ -n "${M3NATIVE_SEQUENTIAL_USERS:-}" ]; then
    echo "M3NATIVE_FUSED_COMMIT serves the packed block only (users=$users, sequential '${M3NATIVE_SEQUENTIAL_USERS:-}')" >&2
    exit 1
  fi
  if [ "${M3NATIVE_TRACED_PROPOSAL:-}" != "1" ] || [ "${M3NATIVE_PAIR_MASK_REFRESH:-}" != "1" ]; then
    echo "M3NATIVE_FUSED_COMMIT=1 needs M3NATIVE_TRACED_PROPOSAL=1 (a captured proposal) and M3NATIVE_PAIR_MASK_REFRESH=1 (T_proj replays every round)" >&2
    exit 1
  fi
  if [ -n "${M3NATIVE_FUSED_COMMIT_LIVE_BANKS:-}" ] && [ "${M3NATIVE_PACKED_PROPOSAL:-}" != "1" ]; then
    echo "M3NATIVE_FUSED_COMMIT_LIVE_BANKS=1 needs M3NATIVE_PACKED_PROPOSAL=1 (it binds the packed pair traces)" >&2
    exit 1
  fi
  echo "round-fence plan H1b: fused commit 1 in place ${M3NATIVE_FUSED_COMMIT_INPLACE:-0} live banks ${M3NATIVE_FUSED_COMMIT_LIVE_BANKS:-0} audit ${M3NATIVE_FUSED_COMMIT_AUDIT:-0}"
fi
# Round-fence plan H2 (early_draft.py; every flag default off). M3NATIVE_EARLY_DRAFT=1 becomes
# QWEN_FAST_EARLY_DRAFT=1: the worker hook drafts the next round inside execute_model and hands vLLM the
# cached drafts at take_draft_token_ids. M3NATIVE_GDN_AFTER_PAIRS=1 (QWEN_FAST_GDN_AFTER_PAIRS, needs the early
# draft and M3NATIVE_ROUND_FENCES=1, whose owed fence the next replay pays) enqueues the packed block's GDN
# commit traces after the next round's pairs have been read back. Both serve the packed block only, through the
# packed proposal coordinator (M3NATIVE_PACKED_PROPOSAL=1, M3NATIVE_PIPELINED_PROPOSALS=1). Refused here,
# before the docker run; unset, nothing is passed.
for h2_flag in M3NATIVE_EARLY_DRAFT M3NATIVE_GDN_AFTER_PAIRS; do
  h2_value="${!h2_flag:-}"
  if [ -n "$h2_value" ] && [ "$h2_value" != "1" ]; then
    echo "$h2_flag must be 1 or unset, got '$h2_value'" >&2
    exit 1
  fi
done
if [ -n "${M3NATIVE_GDN_AFTER_PAIRS:-}" ] && [ -z "${M3NATIVE_EARLY_DRAFT:-}" ]; then
  echo "M3NATIVE_GDN_AFTER_PAIRS=1 without M3NATIVE_EARLY_DRAFT=1 defers nothing (nobody flushes inside the step): set both or neither" >&2
  exit 1
fi
if [ -n "${M3NATIVE_GDN_AFTER_PAIRS:-}" ] && [ -z "${M3NATIVE_ROUND_FENCES:-}" ]; then
  echo "M3NATIVE_GDN_AFTER_PAIRS=1 needs M3NATIVE_ROUND_FENCES=1 (the next replay pays the fence the deferred commits owe)" >&2
  exit 1
fi
if [ -n "${M3NATIVE_EARLY_DRAFT:-}" ]; then
  if [ "$users" -lt 2 ] || [ -n "${M3NATIVE_SEQUENTIAL_USERS:-}" ]; then
    echo "M3NATIVE_EARLY_DRAFT serves the packed block only (users=$users, sequential '${M3NATIVE_SEQUENTIAL_USERS:-}')" >&2
    exit 1
  fi
  if [ "${M3NATIVE_PACKED_PROPOSAL:-}" != "1" ] || [ "${M3NATIVE_PIPELINED_PROPOSALS:-}" != "1" ]; then
    echo "M3NATIVE_EARLY_DRAFT=1 needs M3NATIVE_PACKED_PROPOSAL=1 and M3NATIVE_PIPELINED_PROPOSALS=1 (the coordinator's window is where the pairs are read back)" >&2
    exit 1
  fi
  echo "round-fence plan H2: early draft 1 gdn after pairs ${M3NATIVE_GDN_AFTER_PAIRS:-0}"
fi
# The pair drafter's row-1 fix (pair_row_exact.py; default off). M3NATIVE_PAIR_ROW_EXACT=1 becomes
# QWEN_FAST_PAIR_ROW_EXACT=1: each packed pair's draft SDPA is folded, one KV head per user segment, so a user in
# pair row 1 drafts exactly as it would alone (h1a-draft-race.md sections 2-3). It folds the packed pairs only
# (M3NATIVE_PACKED_PROPOSAL=1, two or more concurrent users); the module is baked, so an image without it fails
# the gate's '[PINDIAG] pair row exact engaged' requirement. M3NATIVE_START_ORDER (user indices, a permutation
# of 0..users-1, comma separated; needs M3NATIVE_STAGGER > 0) starts the gate's requests in that order, and so
# fixes each user's slot and pair row to it: the field check runs one flag set in two orders and compares each
# user's packed fingerprints. Refused here, before the docker run; unset, nothing is passed.
pair_row_value="${M3NATIVE_PAIR_ROW_EXACT:-}"
if [ -n "$pair_row_value" ] && [ "$pair_row_value" != "1" ]; then
  echo "M3NATIVE_PAIR_ROW_EXACT must be 1 or unset, got '$pair_row_value'" >&2
  exit 1
fi
if [ -n "$pair_row_value" ]; then
  if [ "$users" -lt 2 ] || [ -n "${M3NATIVE_SEQUENTIAL_USERS:-}" ]; then
    echo "M3NATIVE_PAIR_ROW_EXACT folds the packed pairs only (users=$users, sequential '${M3NATIVE_SEQUENTIAL_USERS:-}')" >&2
    exit 1
  fi
  if [ "${M3NATIVE_PACKED_PROPOSAL:-}" != "1" ]; then
    echo "M3NATIVE_PAIR_ROW_EXACT=1 needs M3NATIVE_PACKED_PROPOSAL=1 (it folds the packed pair traces)" >&2
    exit 1
  fi
  echo "pair row exact 1"
fi
start_order="${M3NATIVE_START_ORDER:-}"
if [ -n "$start_order" ]; then
  if ! printf '%s' "$start_order" | grep -Eq '^[0-9]+(,[0-9]+)*$'; then
    echo "M3NATIVE_START_ORDER must be comma-separated user indices, got '$start_order'" >&2
    exit 1
  fi
  case "$stagger" in
    *[1-9]*) ;;
    *) echo "M3NATIVE_START_ORDER needs M3NATIVE_STAGGER > 0 (got '$stagger'): without a stagger the requests race" >&2
       exit 1 ;;
  esac
  if [ -n "${M3NATIVE_SEQUENTIAL_USERS:-}" ]; then
    echo "M3NATIVE_START_ORDER orders concurrent requests; a sequential arm has none" >&2
    exit 1
  fi
  echo "start order $start_order (stagger $stagger s)"
fi

# Proposals: the fp2u lane runs each request's draft proposal EAGERLY
# (QWEN_FAST_EAGER_PROPOSAL=1) because per-request proposal traces clobbered each
# other on one mesh (run 35477522469, serving_request_factory.prepare_proposal). At
# four users that is four sequential ~213 ms eager proposals per round (run
# 35559199392), more than half the round. M3NATIVE_TRACED_PROPOSAL=1 replaces that env
# with an inert marker so each request captures its own proposal trace again; the
# gate's byte-exact check against the single-stream references catches a clobbered trace.
if [ "${M3NATIVE_TRACED_PROPOSAL:-}" = "1" ]; then
  M3NATIVE_TRACED_PROPOSAL="-e M3NATIVE_TRACED_PROPOSAL=1"
else
  M3NATIVE_TRACED_PROPOSAL=""
fi

# Optional device-op profiling (M3NATIVE_PROFILE=1), off by default and byte-identical to
# today's invocation when unset. Mirrors dflash-request-profile.sh's own tracy wrapper
# exactly (the same five profiler env vars, the same `python3 -m tracy -p
# --check-exit-code --disable-device-data-dump-to-files --disable-device-data-push-to-tracy
# --dump-device-data-mid-run --op-support-count 20000 -o <dir>` invocation; the count is
# now M3NATIVE_PROFILE_OP_SUPPORT, default 20000), so the packed
# round this gate serves is attributed with the same method already qualified for the T8
# verifier profile (docs/current-verifier-profile-2026-09-09.md), not a new one. The
# mid-run dumps are large, so profiling caps the run at --max-tokens 48 (three or four
# packed rounds after the four prefills) instead of the usual 256; every other arg, mount
# and env var stays the same. The existing 2200 s outer timeout is kept as-is: profiling
# fewer rounds is less work than the unprofiled 256-token run it replaces, and the
# workflow's own 40-minute step / 45-minute job ceilings leave no room to grow it to
# dflash-request-profile.sh's 4200 s (that budget covers a much larger, unrelated run).
#
# Profile mode also sets QWEN_FAST_PROFILED_BLOCK_STREAM=1 so serving_runtime's
# profiled_block_stream_override admits TT_METAL_DEVICE_PROFILER into the mandatory
# block-stream recipe for this attribution measurement only (run 35561945903 refused it).
# M3NATIVE_MAX_TOKENS overrides the 256-token run so users finish at DIFFERENT rounds (the
# early-finish transition, runs 35564623068 / 35567165791); the gate compares a shorter
# stream as a prefix of its reference.
max_tokens="${M3NATIVE_MAX_TOKENS:-256}"
trace_region_bytes=1073741824
entry_args=(-B /bench/lever_n_m3native_gate.py)
if [ "${M3NATIVE_PROFILE:-}" = "1" ]; then
  # The container runs as root with every capability dropped (no CAP_DAC_OVERRIDE), so
  # it cannot create tracy's .logs inside a host directory owned by the runner user
  # (run 35561589158: 'rm -rf /experiment-results-profile/.logs; mkdir -p ...' exit 1).
  mkdir -p experiment-results/profile
  chmod 0777 experiment-results/profile
  mounts+=(--mount "type=bind,src=$PWD/experiment-results/profile,dst=/experiment-results-profile")
  # Profile mode still defaults to 48 tokens, but an explicit M3NATIVE_MAX_TOKENS now wins:
  # the single-user 131k prefill profile (prefill ranking M2) runs --max-tokens 1.
  max_tokens="${M3NATIVE_MAX_TOKENS:-48}"
  trace_region_bytes=268435456
  # M3NATIVE_PROFILE_OP_SUPPORT replaces the fixed --op-support-count 20000 (still the
  # default): one 131k prefill is ~64 chunks x several hundred ops per chunk per chip.
  op_support="${M3NATIVE_PROFILE_OP_SUPPORT:-20000}"
  case "$op_support" in
    ''|*[!0-9]*) echo "M3NATIVE_PROFILE_OP_SUPPORT must be a positive integer, got '$op_support'" >&2; exit 1 ;;
  esac
  if [ "$op_support" -le 0 ]; then
    echo "M3NATIVE_PROFILE_OP_SUPPORT must be a positive integer, got '$op_support'" >&2
    exit 1
  fi
  entry_args=(-m tracy -p --check-exit-code --disable-device-data-dump-to-files
              --disable-device-data-push-to-tracy --dump-device-data-mid-run
              --op-support-count "$op_support" -o /experiment-results-profile
              /bench/lever_n_m3native_gate.py)
  # The layer.py mid-prefill profiler drain (QWEN_PREFILL_PROFILE_FLUSH) is its own switch,
  # M3NATIVE_PROFILE_FLUSH=1: on hardware it segfaulted the dispatch thread's completion-queue
  # read at its first drain (v131, run 35844616271), so plain profile mode no longer sets it.
  echo "profile mode: max_tokens=$max_tokens op_support_count=$op_support QWEN_PREFILL_PROFILE_FLUSH=${M3NATIVE_PROFILE_FLUSH:-0}"
fi
# Traced proposals cost DRAM: each request uploads its own proposal-trace inputs (the
# (1,1,context+32,5120) history and per-layer cached K/V), 0.80 GB per engine against
# 0.58 GB eager, and the fourth user's admission ran out of memory with 716 MB largest
# free (run 35565478581). The traced arm halves the recipe's 1 GiB trace region for the
# headroom; decode traces are command streams and this repo's probes run at 256 MiB.
if [ -n "${M3NATIVE_TRACED_PROPOSAL:-}" ] || [ -n "${M3NATIVE_PACKED_PROPOSAL:-}" ]; then
  # A packed proposal trace uploads the SAME per-request placeholders as one
  # traced-proposal request (dflash_proposal_trace.PreparedPackedDFlashProposal
  # mirrors PreparedDFlashProposal's own placeholder set) but for TWO users at
  # once per pair, so QWEN_FAST_PACKED_PROPOSAL needs at least the traced arm's
  # own headroom trim and never less.
  trace_region_bytes=536870912
fi
# M3NATIVE_TRACE_REGION_BYTES overrides every default above, including the 512 MiB
# the traced/packed-proposal block just forced. It has to sit here, last: anywhere
# earlier it would be silently overwritten by that block. Part of the 4 x 131k DRAM
# plan (A2): 512 -> 256 MiB is worth ~0.27 GB per chip, but traced proposals, packed
# pair traces and traced publish have never been shown to fit in 256 MiB.
if [ -n "${M3NATIVE_TRACE_REGION_BYTES:-}" ]; then
  case "$M3NATIVE_TRACE_REGION_BYTES" in
    ''|*[!0-9]*) echo "M3NATIVE_TRACE_REGION_BYTES must be a positive integer, got '$M3NATIVE_TRACE_REGION_BYTES'" >&2; exit 1 ;;
  esac
  if [ "$M3NATIVE_TRACE_REGION_BYTES" -le 0 ] || [ $(( M3NATIVE_TRACE_REGION_BYTES % 1048576 )) -ne 0 ]; then
    echo "M3NATIVE_TRACE_REGION_BYTES must be a positive multiple of 1 MiB" >&2
    exit 1
  fi
  trace_region_bytes="$M3NATIVE_TRACE_REGION_BYTES"
  echo "trace region overridden: $trace_region_bytes bytes"
fi
# The tt-metal watcher (TT_METAL_WATCHER=20, inherited from the fp2u lane's hang diagnosis)
# compiles NoC sanitisation and waypoints into every kernel. The same folded SDPA decode
# call costs 0.46 ms on card M without it and 2.05 ms in the gate's device profile with
# it (run 35567165791 vs sdpa_verify_prefill_style_bench.py), so every packed-round
# trace_ms measured so far (1453, 1118, 1100 ms) carries watcher overhead the
# single-stream baselines never had. Off by default; M3NATIVE_WATCHER=1 re-enables it.
# Three env vars are read by the scripts mounted at /bench, not by this script, so
# they have to cross into the container or they are dead: M3NATIVE_PREFILL_CHUNK_TOKENS
# (lever_n_m3native_gate builds --enable-chunked-prefill from it) and the two TTFT
# ceilings m3native_ttft_profile asserts. Run 35679222511 set CHUNK_TOKENS=2048 on the
# HOST, mounted the three M1 files, passed TT_M1_FORCE_CHUNKED_PREFILL=1 - and the
# gate inside still saw it unset, launched the server with --no-enable-chunked-prefill
# --max-num-batched-tokens 33024, and prefilled each prompt whole. The resumable path
# never ran. test_m3native_arm_env.py now derives this list from the /bench mounts and
# fails if a read is not passed through.
name="qwen-m3native-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
trap 'timeout 20 docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
timeout -k 30 2200 docker run --rm --name "$name" --network none \
  --hostname qwen-m3native --add-host qwen-m3native:127.0.0.1 \
  --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
  --pids-limit 4096 --memory 96g --cpus 16 --shm-size 8g \
  "${devices[@]}" \
  --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount "type=bind,src=$target,dst=/models/hub/models--Qwen--Qwen3.8-27B,readonly" \
  --mount "type=bind,src=$PWD/draft-config,dst=/draft-config,readonly" \
  --mount "type=bind,src=$PWD/graft/model_config.py,dst=$root/model_config.py,readonly" \
  --mount "type=bind,src=$PWD/graft/attention/tp.py,dst=$root/attention/tp.py,readonly" \
  --mount "type=bind,src=$PWD/graft/gdn/tp.py,dst=$root/gdn/tp.py,readonly" \
  "${prefill_conv_mounts[@]}" \
  --mount "type=bind,src=$PWD/graft/mlp.py,dst=$root/mlp.py,readonly" \
  "${c1e_mounts[@]}" \
  --mount "type=bind,src=$PWD/graft/layer.py,dst=$root/layer.py,readonly" \
  "${lever_n_mounts[@]}" \
  --mount "type=bind,src=$PWD/scripts/ci/lever_n_m3native_gate.py,dst=/bench/lever_n_m3native_gate.py,readonly" \
  --mount "type=bind,src=$PWD/scripts/ci/longctx_cycle_bench.py,dst=/bench/longctx_cycle_bench.py,readonly" \
  --mount "type=bind,src=$PWD/scripts/ci/m3native_ttft_profile.py,dst=/bench/m3native_ttft_profile.py,readonly" \
  --mount "type=bind,src=$PWD/scripts/ci/acceptance_report.py,dst=/bench/acceptance_report.py,readonly" \
  --mount "type=bind,src=$PWD/scripts/ci/real_text_prompts.py,dst=/bench/real_text_prompts.py,readonly" \
  --mount type=volume,src=qwen-experiments-f1e9b1a64b4f,dst=/experiment-cache \
  "${mounts[@]}" \
  "${sdpa_mode_mounts[@]}" \
  $KM \
  "${sdpa_pf_env[@]}" \
  ${KOPGRAFT64:+-e QWEN_FAST_NATIVE_ATTN=1} \
  ${KOPGRAFT64:+-e QWEN_FAST_RUNTIME_BINARY_SHA256=$graft_binary_sha} \
  ${M3NATIVE_PIPELINED_COMMITS:+-e QWEN_FAST_PIPELINED_COMMITS=1} \
  ${M3NATIVE_PIPELINED_PROPOSALS:+-e QWEN_FAST_PIPELINED_PROPOSALS=1} \
  ${M3NATIVE_FAST_COMMIT:+-e QWEN_FAST_FAST_COMMIT=1} \
  ${M3NATIVE_PACKED_PROPOSAL:+-e QWEN_FAST_PACKED_PROPOSAL=1} \
  ${M3NATIVE_PIPELINED_PUBLISH:+-e QWEN_FAST_PIPELINED_PUBLISH=1} \
  ${M3NATIVE_TRACED_PUBLISH:+-e QWEN_FAST_TRACED_PUBLISH=1} \
  ${M3NATIVE_GDN_USER_BATCH:+-e QWEN_FAST_GDN_USER_BATCH=1} \
  ${M3NATIVE_REPLAY_GROUP_ROWS:+-e QWEN_FAST_REPLAY_GROUP_ROWS=$M3NATIVE_REPLAY_GROUP_ROWS} \
  ${M3NATIVE_SDPA_MODES:+-e QWEN_FAST_SDPA_MODES=$M3NATIVE_SDPA_MODES} \
  ${M3NATIVE_GDN_USER_BATCH_MIN_USERS:+-e QWEN_FAST_GDN_USER_BATCH_MIN_USERS=$M3NATIVE_GDN_USER_BATCH_MIN_USERS} \
  ${M3NATIVE_GDN_STATE_COPY_BATCH:+-e QWEN_FAST_GDN_STATE_COPY_BATCH=1} \
  ${M3NATIVE_MEMORY_LEDGER:+-e QWEN_FAST_MEMORY_LEDGER=1} \
  ${M3NATIVE_SKIP_BLOCK_STREAM:+-e QWEN_FAST_SKIP_BLOCK_STREAM=1} \
  ${M3NATIVE_SINGLE_GATEUP:+-e QWEN_FAST_SINGLE_GATEUP=1} \
  ${M3NATIVE_C1_AGMM:+-e QWEN_FAST_C1_AGMM=1} \
  ${M3NATIVE_C1_EXACT:+-e QWEN_FAST_C1_EXACT=1} \
  ${M3NATIVE_C1_EXACT_AUDIT:+-e QWEN_FAST_C1_EXACT_AUDIT=$M3NATIVE_C1_EXACT_AUDIT} \
  ${M3NATIVE_C1_LEGACY:+-e QWEN_FAST_C1_LEGACY=1} \
  ${M3NATIVE_ROUND_B1:+-e QWEN_FAST_ROUND_B1=1} \
  ${M3NATIVE_ROUND_B1_AUDIT:+-e QWEN_FAST_ROUND_B1_AUDIT=1} \
  ${M3NATIVE_VERIFY_T1:+-e QWEN_FAST_VERIFY_T1=1} \
  ${M3NATIVE_VERIFY_T1_AUDIT:+-e QWEN_FAST_VERIFY_T1_AUDIT=1} \
  ${M3NATIVE_VERIFY_T1_SKIP:+-e QWEN_FAST_VERIFY_T1_SKIP=$M3NATIVE_VERIFY_T1_SKIP} \
  ${M3NATIVE_VERIFY_T2:+-e QWEN_FAST_VERIFY_T2=1} \
  ${M3NATIVE_VERIFY_T2_AUDIT:+-e QWEN_FAST_VERIFY_T2_AUDIT=1} \
  ${M3NATIVE_VERIFY_T2_SKIP:+-e QWEN_FAST_VERIFY_T2_SKIP=$M3NATIVE_VERIFY_T2_SKIP} \
  ${M3NATIVE_VERIFY_T2_KV_ROWS:+-e QWEN_FAST_VERIFY_T2_KV_ROWS=$M3NATIVE_VERIFY_T2_KV_ROWS} \
  ${M3NATIVE_PAIR_MASK_REFRESH:+-e QWEN_FAST_PAIR_MASK_REFRESH=1} \
  ${M3NATIVE_PAIR_MASK_AUDIT:+-e QWEN_FAST_PAIR_MASK_AUDIT=1} \
  ${M3NATIVE_PAIRS_PACKED_ONLY:+-e QWEN_FAST_PAIRS_PACKED_ONLY=1} \
  ${M3NATIVE_PADDED_PROBE:+-e QWEN_FAST_PADDED_PROBE=1} \
  ${M3NATIVE_PADDED_BLOCK:+-e QWEN_FAST_PADDED_BLOCK=1} \
  ${M3NATIVE_PADDED_BLOCK_MIN_USERS:+-e QWEN_FAST_PADDED_BLOCK_MIN_USERS=$M3NATIVE_PADDED_BLOCK_MIN_USERS} \
  ${M3NATIVE_PRESTAGE:+-e QWEN_FAST_PRESTAGE=1} \
  ${M3NATIVE_PRESTAGE_AUDIT:+-e QWEN_FAST_PRESTAGE_AUDIT=1} \
  ${M3NATIVE_ROUND_FENCES:+-e QWEN_FAST_ROUND_FENCES=1} \
  ${M3NATIVE_FUSED_COMMIT:+-e QWEN_FAST_FUSED_COMMIT=1} \
  ${M3NATIVE_FUSED_COMMIT_INPLACE:+-e QWEN_FAST_FUSED_COMMIT_INPLACE=1} \
  ${M3NATIVE_FUSED_COMMIT_LIVE_BANKS:+-e QWEN_FAST_FUSED_COMMIT_LIVE_BANKS=1} \
  ${M3NATIVE_FUSED_COMMIT_AUDIT:+-e QWEN_FAST_FUSED_COMMIT_AUDIT=1} \
  ${M3NATIVE_EARLY_DRAFT:+-e QWEN_FAST_EARLY_DRAFT=1} \
  ${M3NATIVE_GDN_AFTER_PAIRS:+-e QWEN_FAST_GDN_AFTER_PAIRS=1} \
  ${M3NATIVE_PAIR_ROW_EXACT:+-e QWEN_FAST_PAIR_ROW_EXACT=1} \
  ${M3NATIVE_LEGACY_CONTINUATION_ORDER:+-e QWEN_FAST_LEGACY_CONTINUATION_ORDER=1} \
  ${M3NATIVE_GDN_PREFILL_CONV:+-e QWEN_FAST_GDN_PREFILL_CONV=1} \
  ${M3NATIVE_GDN_PREFILL_CONV_AUDIT:+-e QWEN_FAST_GDN_PREFILL_CONV_AUDIT=$M3NATIVE_GDN_PREFILL_CONV_AUDIT} \
  ${M3NATIVE_DRAFT_BF8:+-e QWEN_FAST_DRAFT_BF8=1} \
  ${M3NATIVE_PREFILL_CHUNK:+-e MAX_PREFILL_CHUNK_SIZE=$M3NATIVE_PREFILL_CHUNK} \
  ${M3NATIVE_INTERLEAVE:+-e TT_PREFILL_DECODE_INTERLEAVE=$M3NATIVE_INTERLEAVE} \
  ${M3NATIVE_DECODE_STEPS_PER_CHUNK:+-e TT_DECODE_STEPS_PER_PREFILL_CHUNK=$M3NATIVE_DECODE_STEPS_PER_CHUNK} \
  ${M3NATIVE_PREFILL_CHUNK_TOKENS:+-e M3NATIVE_PREFILL_CHUNK_TOKENS=$M3NATIVE_PREFILL_CHUNK_TOKENS} \
  ${M3NATIVE_TTFT_MAX_S:+-e M3NATIVE_TTFT_MAX_S=$M3NATIVE_TTFT_MAX_S} \
  ${M3NATIVE_TTFT_MAX_STALL_S:+-e M3NATIVE_TTFT_MAX_STALL_S=$M3NATIVE_TTFT_MAX_STALL_S} \
  ${M3NATIVE_PROFILE:+-e TTNN_OP_PROFILER=1} \
  ${M3NATIVE_PROFILE:+-e TT_METAL_DEVICE_PROFILER=1} \
  ${M3NATIVE_PROFILE:+-e TT_METAL_PROFILER_TRACE_TRACKING=1} \
  ${M3NATIVE_PROFILE:+-e TT_METAL_PROFILER_CPP_POST_PROCESS=1} \
  ${M3NATIVE_PROFILE:+-e TT_METAL_PROFILER_MID_RUN_DUMP=1} \
  ${M3NATIVE_PROFILE:+-e QWEN_FAST_PROFILE_DUMP_ROUND=4} \
  ${M3NATIVE_PROFILE:+-e QWEN_FAST_PROFILED_BLOCK_STREAM=1} \
  ${M3NATIVE_PROFILE_FLUSH:+-e QWEN_PREFILL_PROFILE_FLUSH=1} \
  -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e QWEN_PROJECTION_LINKS=4 \
  -e QWEN_FAST_FOUR_AS_TWO=0 \
  -e QWEN_FABRIC_LINK_PROBE=1 -e QWEN_FROZEN_COMBINED_RUNTIME=1 -e QWEN_DSPARK_REQUEST_CONTEXT=$dspark_context \
  ${max_position:+-e QWEN_FAST_MAX_POSITION=$max_position} \
  ${M3NATIVE_TRACED_PROPOSAL:--e QWEN_FAST_EAGER_PROPOSAL=1} -e QWEN_FAST_SHARD_CHECK=0 -e QWEN_FAST_PHASE_LOG=1 -e QWEN_FAST_CARRY_LOG=1 \
  -e QWEN_FAST_SHARED_CCL=1 -e QWEN_FAST_PACKED_STEP=1 -e QWEN_FAST_PACKED_AUDIT=1 -e QWEN_FAST_FAULTHANDLER=1 \
  ${M3NATIVE_WATCHER:+-e TT_METAL_WATCHER=20} ${M3NATIVE_WATCHER:+-e TT_METAL_WATCHER_APPEND=1} ${M3NATIVE_WATCHER:+-e TT_METAL_WATCHER_DISABLE_ASSERT=1} \
  -e QWEN_GDN_DIRECT_WINDOW=1 -e QWEN_GDN_SHARED_QK_EXPERIMENT=1 \
  -e QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1 -e QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1 \
  -e QWEN_SDPA_BF8=1 -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 \
  -e QWEN_SKIP_UNUSED_SINGLETON_POSITIONS=1 -e QWEN_FAST_PHASE_TIMING=1 \
  -e QWEN_GDN_GROUPED_GATHER_ABBA=0 -e QWEN_GDN_GATE_EXP_ABBA=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e TT_METAL_HOME=/opt/tt-metal -e MESH_DEVICE=P300 -e OMP_NUM_THREADS=8 \
  -e TT_CACHE_PATH=/experiment-cache/weights -e TT_METAL_CACHE=$kernel_cache \
  -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto \
  --entrypoint python3 "$image" "${entry_args[@]}" \
  --users "$users" --context "$context" --prompt-tokens "$prompt_tokens" --max-tokens "$max_tokens" --stream-timeout 600 --trace-region-bytes "$trace_region_bytes" \
  --prompt-base 1000 --prompt-user-offset 1 --stagger "$stagger" ${M3NATIVE_SEQUENTIAL_USERS:+--sequential-users $M3NATIVE_SEQUENTIAL_USERS} \
  ${M3NATIVE_START_ORDER:+--start-order $M3NATIVE_START_ORDER} \
  ${M3NATIVE_PROMPT_SOURCE:+--prompt-source $M3NATIVE_PROMPT_SOURCE} ${M3NATIVE_EOS:+--eos $M3NATIVE_EOS} \
  --references /bench/packed-gate-reference $allow_missing_references --results /experiment-results-gate \
  > experiment-results/m3native-gate-stdout.log 2>&1 || true

if [ "${M3NATIVE_PROFILE:-}" = "1" ]; then
  # tracy wrote its .logs as root inside the bind mount; the runner user could not
  # delete them and the next checkout died (run 35562881085, EACCES unlink). A container
  # with its default capabilities hands the tree back before anything else runs.
  timeout -k 10 120 docker run --rm --network none     --mount "type=bind,src=$PWD/experiment-results/profile,dst=/p"     --entrypoint sh "$image" -c 'chmod -R a+rwX /p' > /dev/null 2>&1 || true
fi
sed -n '/M3NATIVE_GATE_JSON_BEGIN/,/M3NATIVE_GATE_JSON_END/p' experiment-results/m3native-gate-stdout.log \
  | sed '1d;$d' > experiment-results/m3native-gate.json || true
sed -n '/M3NATIVE_GATE_LOG_BEGIN/,/M3NATIVE_GATE_LOG_END/p' experiment-results/m3native-gate-stdout.log \
  | sed '1d;$d' > experiment-results/m3native-server-tail.log || true
# Every '[PINDIAG] dram after' line, straight to the workflow log: an arm that never
# reaches a packed round (e.g. the 131k attach probe) still has this to read without
# downloading the artifact.
grep -F '[PINDIAG] dram after' experiment-results/m3native-server-tail.log || true
test -s experiment-results/m3native-gate.json
