#!/usr/bin/env bash
set -euo pipefail
mkdir -p /experiment/results
sampling_args=()
prefix_copy_args=()
[[ "${QWEN_LEARNED_STACK:-0}" = 0 || ( "${QWEN_LEARNED_STACK:-0}" = 1 && "${QWEN_RUN_MODE:-baseline}" = learned-attention ) ]]
if [ "${QWEN_PREFIX_ZERO_REUSE:-0}" != 0 ]; then
    [[ "$QWEN_PREFIX_ZERO_REUSE" = 1 && "${QWEN_RUN_MODE:-baseline}" = full-attention-tree ]]
    prefix_copy_args=(--prefix-zero-reuse)
fi
if [[ "${QWEN_RUN_MODE:-baseline}" = sampling || "${QWEN_RUN_MODE:-baseline}" = sampling-extended ]]; then
    export QWEN_TP2_SAMPLING_EXPERIMENT=1
fi
if [ "${QWEN_RUN_MODE:-baseline}" = sampling-extended ]; then sampling_args=(--extended); fi
unset TT_METAL_SIMULATOR TT_METAL_SLOW_DISPATCH_MODE TT_METAL_MOCK_CLUSTER_DESC_PATH
export PYTHONPATH=/opt/tt-metal/ttnn:/opt/tt-metal${PYTHONPATH:+:$PYTHONPATH}
python3 /experiment-scripts/ci/device-owners.py > /experiment/results/allocation.json
python3 /experiment-scripts/ci/hardware-correctness.py --suite audit --output /experiment/results/runtime-audit.json
if [ "${QWEN_DFLASH_LIVE_QUERY_ABBA:-0}" = 1 ]; then
    [[ "${QWEN_RUN_MODE:-baseline}" = full-norm-engine && "${QWEN_DFLASH_CONTEXT:-0}" = 4096 && "${QWEN_DFLASH_CAPTURE:-0}" = 1 && "${QWEN_DFLASH_DRAFTS:-0}" = 7 ]]
    python3 /experiment-scripts/ci/live_attention_request_gate.py --metal-root /opt/tt-metal \
        > /experiment/results/live-attention-preflight.json
fi
if [ "${QWEN_LIVE_QK:-0}" = 1 ]; then
    [[ "${QWEN_RUN_MODE:-baseline}" = baseline && "${QWEN_CCL_LAZY_BUILD:-0}" = 0 ]]
    PYTHONPATH="/experiment-scripts/ci:$PYTHONPATH" python3 -c \
        'import json; from live_qk_gate import native_hashes; print(json.dumps(native_hashes("/opt/tt-metal")))' \
        > /experiment/results/live-qk-native-hashes.json
    PYTHONPATH="/experiment-scripts/ci:$PYTHONPATH" python3 -c \
        'import json; from pathlib import Path; from live_qk_gate import native_hashes, source_hashes, qualify_simulator; sources=source_hashes(); native=native_hashes("/opt/tt-metal"); [qualify_simulator(json.loads(Path(f"/experiment-scripts/ci/live-qk-simulator-{context}.json").read_text()), context, sources, native) for context in (31, 2048)]'
    for context in 31 2048; do
        OMP_NUM_THREADS=1 timeout -k 15 600 python3 /experiment-scripts/ci/draft-live-qk-probe.py \
            --hardware --attention --context "$context" \
            --simulator-report "/experiment-scripts/ci/live-qk-simulator-$context.json" \
            --output "/experiment/results/live-qk-$context.json"
    done
    exit
fi
if [ "${QWEN_MTP_DRAFTS:-0}" != 0 ]; then
    PYTHONPATH="/experiment-speculative:$PYTHONPATH" python3 /experiment-scripts/ci/full_mtp_request.py \
        --weights /models/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
        --audit-output /experiment/results/mtp-checkpoint.json
fi
if [ "${QWEN_CCL_LAZY_BUILD:-0}" = 1 ]; then
    if [ "${QWEN_TENSIX_MLP:-0}" = 1 ]; then
        test "${QWEN_TINY_MLP:-0}" = 0
        python3 /experiment-scripts/ci/tensix-stream-mlp-hardware.py --preflight \
            --simulator-report /experiment-scripts/ci/tensix-mlp-simulator.json \
            --simulator-exit-status /experiment-scripts/ci/tensix-mlp-simulator.exit-status \
            --output /experiment/results/tensix-mlp-preflight.json
    fi
    if [ "${QWEN_TINY_MLP:-0}" = 1 ]; then
        python3 /experiment-scripts/ci/tiny-mlp-hardware.py --preflight \
            --simulator-report /experiment-scripts/ci/tiny-mlp-simulator.json \
            --output /experiment/results/tiny-mlp-preflight.json
    fi
    bash /experiment-scripts/ci/ccl-links-build.sh
    if [[ "${QWEN_MTP_DRAFTS:-0}" = 0 && "${QWEN_DFLASH_DRAFTS:-0}" = 0 ]]; then
        OMP_NUM_THREADS=1 timeout -k 15 900 python3 -u /experiment-scripts/ci/ccl-link-probe.py \
            --hardware --output /experiment/results/ccl-link-hardware.json 2>&1 | tee /experiment/results/ccl-link-hardware.log
        if grep -q 'Failed to discover available ethernet links' /experiment/results/ccl-link-hardware.log; then
            echo 'Explicit-link hardware collective still invoked fallback discovery' >&2
            exit 1
        fi
        if [[ "${QWEN_TINY_MLP:-0}" != 1 && "${QWEN_TENSIX_MLP:-0}" != 1 ]]; then exit 0; fi
    fi
fi
if [ "${QWEN_RUN_MODE:-baseline}" = learned-mlp ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    OMP_NUM_THREADS=1 timeout -k 15 600 python3 /experiment-scripts/ci/fused-batch-probe.py \
        --hardware --timing --device-weight-check --trace-replay --fixture /experiment-projection-fixture \
        --output /experiment/results/fused-batch.json
    OMP_NUM_THREADS=1 timeout -k 30 1800 python3 /experiment-scripts/ci/learned-mlp-probe.py \
        --hardware --fixture /experiment-projection-fixture --output /experiment/results/learned-mlp.json
    OMP_NUM_THREADS=1 timeout -k 30 1800 python3 /experiment-scripts/ci/learned-mlp-probe.py \
        --hardware --fixture /experiment-projection-fixture --convolution-fixture /experiment-convolution-fixture \
        --output /experiment/results/learned-mlp-integrated.json
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = learned-attention ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    if [ "${QWEN_LEARNED_STACK:-0}" = 1 ]; then
        OMP_NUM_THREADS=1 timeout -k 15 900 python3 /experiment-scripts/ci/draft-mlp-replay-probe.py \
            --hardware --timing --fixture /experiment-mlp-fixture \
            --convolution-fixture /experiment-convolution-fixture --output /experiment/results/learned-mlp-replay.json
        OMP_NUM_THREADS=1 timeout -k 15 900 python3 /experiment-scripts/ci/draft-mlp-trace-probe.py \
            --hardware --timing --fixture /experiment-mlp-fixture \
            --convolution-fixture /experiment-convolution-fixture --output /experiment/results/learned-mlp-trace.json
        OMP_NUM_THREADS=1 timeout -k 15 300 python3 /experiment-scripts/ci/draft-dot-probe.py \
            --hardware --timing --keys 32 --width 256 --cache-tiles \
            --selector-fixture /experiment-selector-fixture --output /experiment/results/learned-selector-dot.json
        OMP_NUM_THREADS=1 timeout -k 30 3600 python3 /experiment-scripts/ci/learned-attention-probe.py \
            --hardware --fp32-rope --explicit-softmax --fused-row-sum --fused-dots --cache-dot-tiles \
            --fixture /experiment-projection-fixture --convolution-fixture /experiment-convolution-fixture \
            --mlp-fixture /experiment-mlp-fixture --stack-fixtures /experiment-stack-fixture --stack-layers 5 \
            --selector-fixture /experiment-selector-fixture --output /experiment/results/learned-five-layers.json
        exit 0
    fi
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/learned-attention-probe.py \
        --hardware --fp32-rope --explicit-softmax --pairwise-softmax --pairwise-dots \
        --fixture /experiment-projection-fixture --output /experiment/results/learned-attention.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/draft-row-sum-probe.py \
        --hardware --timing --output /experiment/results/draft-row-sum.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/learned-attention-probe.py \
        --hardware --fp32-rope --explicit-softmax --fused-row-sum --pairwise-dots \
        --fixture /experiment-projection-fixture --output /experiment/results/learned-attention-fused-sum.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/draft-dot-probe.py \
        --hardware --timing --keys 32 --width 128 --output /experiment/results/draft-dot-short.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/draft-dot-probe.py \
        --hardware --timing --keys 2080 --width 128 --output /experiment/results/draft-dot-qk-long.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/draft-dot-probe.py \
        --hardware --timing --keys 128 --width 2080 --output /experiment/results/draft-dot-pv-long.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/learned-attention-probe.py \
        --hardware --fp32-rope --explicit-softmax --fused-row-sum --fused-dots \
        --fixture /experiment-projection-fixture --output /experiment/results/learned-attention-fused-dots.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/draft-dot-probe.py \
        --hardware --timing --cache-tiles --keys 32 --width 128 --output /experiment/results/draft-dot-cached-short.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/draft-dot-probe.py \
        --hardware --timing --cache-tiles --keys 2080 --width 128 --output /experiment/results/draft-dot-cached-qk-long.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/draft-dot-probe.py \
        --hardware --timing --cache-tiles --keys 128 --width 2080 --output /experiment/results/draft-dot-cached-pv-long.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/learned-attention-probe.py \
        --hardware --fp32-rope --explicit-softmax --fused-row-sum --fused-dots --cache-dot-tiles \
        --fixture /experiment-projection-fixture --output /experiment/results/learned-attention-cached-dots.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/draft-dot-probe.py \
        --hardware --timing --cache-tiles --workers 110 --keys 2080 --width 128 \
        --output /experiment/results/draft-dot-cached-qk-110.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/draft-dot-probe.py \
        --hardware --timing --cache-tiles --workers 110 --keys 128 --width 2080 --columns-per-task 8 \
        --output /experiment/results/draft-dot-cached-pv-split110.json
    OMP_NUM_THREADS=1 timeout -k 30 3600 python3 /experiment-scripts/ci/learned-attention-probe.py \
        --hardware --context 2048 --fp32-rope --explicit-softmax --fused-row-sum --fused-dots \
        --cache-dot-tiles --wide-dot-placement --fixture /experiment-projection-fixture \
        --output /experiment/results/learned-attention-long-wide.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/learned-attention-probe.py \
        --hardware --fp32-rope --explicit-softmax --fused-row-sum --fused-dots --cache-dot-tiles \
        --fixture /experiment-projection-fixture --convolution-fixture /experiment-convolution-fixture \
        --output /experiment/results/learned-attention-integrated.json
    OMP_NUM_THREADS=1 timeout -k 30 1800 python3 /experiment-scripts/ci/learned-attention-probe.py \
        --hardware --fp32-rope --explicit-softmax --fused-row-sum --fused-dots --cache-dot-tiles \
        --fixture /experiment-projection-fixture --convolution-fixture /experiment-convolution-fixture \
        --mlp-fixture /experiment-mlp-fixture --output /experiment/results/learned-layer-complete.json
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = learned-convolution ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/learned-convolution-probe.py \
        --hardware --fp32-intermediates --fixture /experiment-projection-fixture \
        --output /experiment/results/learned-convolution.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/draft-attention-probe.py \
        --hardware --composed --timing --output /experiment/results/draft-attention-composed.json
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = feature-projection-full ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/feature-projection-probe.py \
        --hardware --full-projection --rows 1 --fixture /experiment-projection-fixture \
        --reference blackhole-accumulation --k-block 4 --output /experiment/results/feature-projection-full.json
    OMP_NUM_THREADS=1 timeout -k 15 300 python3 /experiment-scripts/ci/feature-norm-probe.py \
        --hardware --fixture /experiment-projection-fixture \
        --projection-report /experiment/results/feature-projection-full.json --output /experiment/results/feature-norm.json
    OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/feature-projection-probe.py \
        --hardware --full-projection --device-tail --rows 1 --fixture /experiment-projection-fixture \
        --reference blackhole-accumulation --k-block 4 --output /experiment/results/feature-projection-device-tail.json
    OMP_NUM_THREADS=1 timeout -k 30 1200 python3 /experiment-scripts/ci/feature-projection-probe.py \
        --hardware --full-projection --device-tail --rows 8 --fixture /experiment-projection-fixture \
        --reference blackhole-accumulation --k-block 4 --output /experiment/results/feature-projection-device-tail-rows8.json
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = feature-projection ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    for block in 100 4; do
        OMP_NUM_THREADS=1 timeout -k 30 900 python3 /experiment-scripts/ci/feature-projection-probe.py \
            --hardware --fixture /experiment-projection-fixture --reference blackhole-accumulation \
            --k-block "$block" --output "/experiment/results/feature-projection-k$block.json"
    done
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = device-readback ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    exit 0
fi
python3 - <<'PY' > /experiment/results/token-protocol-fields.json
import ast
import importlib.util
import json
from pathlib import Path
root = Path(importlib.util.find_spec('vllm').origin).parent / 'entrypoints' / 'openai'
report = {}
for path in root.rglob('*protocol*.py'):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef) and ('Completion' in node.name or node.name in ('StreamOptions', 'PerRequestTimingMetrics')):
            fields = {field.target.id: ast.unparse(field.annotation) for field in node.body
                      if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)}
            report[str(path.relative_to(root)) + ':' + node.name] = fields
print(json.dumps(report, indent=2))
PY
revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export MODEL_WEIGHTS_DIR="/models/hub/models--Qwen--Qwen3.8-27B/snapshots/$revision"
export HF_MODEL="$MODEL_WEIGHTS_DIR"
python3 - <<'PY' > /experiment/results/model-manifest.json
import hashlib
import json
import os
from pathlib import Path
root = Path(os.environ['MODEL_WEIGHTS_DIR'])
index = json.loads((root / 'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    if not (root / name).is_file():
        raise RuntimeError(f'Missing weight shard: {name}')
files = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors.index.json']
print(json.dumps(dict(snapshot=root.name, files={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in files}, config=json.loads((root / 'config.json').read_text()),
    flags={name: value for name, value in os.environ.items() if name.startswith(('QWEN', 'TT_', 'MESH_DEVICE'))}), indent=2))
PY
if [ "${QWEN_RUN_MODE:-baseline}" = sampling-kernel ]; then
    if [ "${QWEN_FABRIC_LINK_PROBE:-0}" = 1 ]; then
        timeout -k 15 900 python3 /experiment-scripts/ci/sampling-links.py
    else
        timeout 900 python3 /experiment-scripts/ci/sampling-kernel.py
    fi
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-model-fusion ]; then
    timeout -k 30 3000 python3 /experiment-scripts/ci/full-model-fusion.py
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = attention-batch ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/attention-batch.py
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = attention-groups ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1800 python3 /experiment-scripts/ci/attention-group-timing.py
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = attention-group-layer ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1800 python3 /experiment-scripts/ci/attention-batch.py --timing --ordered-cache --grouped
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = attention-group-dma ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1800 python3 /experiment-scripts/ci/attention-group-timing.py --dma-layout
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = attention-dma-layer ]; then
    timeout -k 15 60 python3 /experiment-scripts/ci/sdpa-tree-audit.py
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1800 python3 /experiment-scripts/ci/attention-batch.py --timing --ordered-cache --grouped --dma-layout
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = attention-parallel-groups ]; then
    timeout -k 15 60 python3 /experiment-scripts/ci/sdpa-tree-audit.py
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1800 python3 /experiment-scripts/ci/attention-group-timing.py --parallel-groups
    timeout -k 30 1800 python3 /experiment-scripts/ci/attention-batch.py --timing --ordered-cache --grouped --dma-layout --parallel-groups
    exit 0
fi
if [[ "${QWEN_RUN_MODE:-baseline}" = attention-tree-scratch || "${QWEN_RUN_MODE:-baseline}" = attention-tree-parallel || "${QWEN_RUN_MODE:-baseline}" = attention-tree-layer ]]; then
    timeout -k 30 1920 bash /experiment-scripts/ci/sdpa-tree-build.sh
    export QWEN_SDPA_TREE_SCRATCH_ROUNDS=1
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    if [ "$QWEN_RUN_MODE" = attention-tree-layer ]; then
        timeout -k 30 1800 python3 /experiment-scripts/ci/attention-batch.py --timing --ordered-cache --grouped --dma-layout --parallel-groups --tree-parallel
        exit 0
    fi
    tree_option=--tree-scratch
    if [ "$QWEN_RUN_MODE" = attention-tree-parallel ]; then tree_option=--tree-parallel; fi
    timeout -k 30 1800 python3 /experiment-scripts/ci/attention-group-timing.py "$tree_option"
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = attention-mask-replay ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 600 python3 /experiment-scripts/ci/attention-mask-replay.py
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = attention-replay ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1800 python3 /experiment-scripts/ci/attention-replay.py
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = attention-tree-replay ]; then
    timeout -k 30 1920 bash /experiment-scripts/ci/sdpa-tree-build.sh
    export QWEN_SDPA_TREE_SCRATCH_ROUNDS=1
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 600 python3 /experiment-scripts/ci/attention-mask-replay.py --wide
    timeout -k 30 1800 python3 /experiment-scripts/ci/attention-replay.py --max-group-rows 8
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = attention-timing ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/attention-batch.py --timing --ordered-cache
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-prefix ]; then
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py
    exit 0
fi
if [[ "${QWEN_RUN_MODE:-baseline}" = target-features || "${QWEN_RUN_MODE:-baseline}" = target-feature-prefill ]]; then
    feature_options=()
    if [ "$QWEN_RUN_MODE" = target-feature-prefill ]; then feature_options+=(--target-feature-prefill); fi
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --target-features "${feature_options[@]}"
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-batch ]; then
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --batch
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-coding-cost ]; then
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --batch --coding-cost --serial-sdpa
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-compact-gdn ]; then
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --batch --coding-cost --serial-sdpa --compact-gdn
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-gdn-input-reuse ]; then
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-gdn-row-clones ]; then
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-gdn-row-layout ]; then
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-verifier-engine ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --ordered-cache --device-selection --request-pilot
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-verifier-selection ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 15 300 python3 /experiment-scripts/ci/sampling-kernel.py
    timeout -k 30 4200 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --ordered-cache --device-selection
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-verifier-replay ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --deferred-commit --commit-dma --captured-commit --ordered-cache --replay-inputs
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-norm-engine ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    if [ "${QWEN_TENSIX_MLP:-0}" = 1 ]; then
        if [ "${QWEN_TENSIX_MLP_PROFILE:-0}" = 1 ]; then
            timeout -k 30 1500 bash /experiment-scripts/ci/tensix-mlp-profile.sh
            exit 0
        fi
        timeout -k 30 1200 python3 -u /experiment-scripts/ci/tensix-stream-mlp-hardware.py \
            --simulator-report /experiment-scripts/ci/tensix-mlp-simulator.json \
            --simulator-exit-status /experiment-scripts/ci/tensix-mlp-simulator.exit-status \
            --output /experiment/results/tensix-mlp.json
        exit 0
    fi
    if [ "${QWEN_TINY_MLP:-0}" = 1 ]; then
        timeout -k 30 900 python3 -u /experiment-scripts/ci/tiny-mlp-hardware.py \
            --simulator-report /experiment-scripts/ci/tiny-mlp-simulator.json \
            --output /experiment/results/tiny-mlp.json
        exit 0
    fi
    if [ "${QWEN_DFLASH_VERIFIER_PROFILE:-0}" = 1 ]; then
        timeout -k 30 4800 bash /experiment-scripts/ci/dflash-request-profile.sh
        exit 0
    fi
    if [ "${QWEN_MTP_DRAFTS:-0}" != 0 ]; then
        OMP_NUM_THREADS=1 timeout -k 15 300 python3 -u /experiment-scripts/ci/mtp-hidden-row-probe.py \
            --hardware --output /experiment/results/mtp-hidden-rows.json
        OMP_NUM_THREADS=1 timeout -k 15 300 python3 -u /experiment-scripts/ci/sampling-native-rows-probe.py \
            --hardware --output /experiment/results/sampling-native-rows.json
    fi
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --ordered-cache --device-selection --request-pilot --norm-batch
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-norm-selection ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 15 300 python3 /experiment-scripts/ci/sampling-kernel.py
    timeout -k 30 4200 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --ordered-cache --device-selection --norm-batch
    exit 0
fi
if [[ "${QWEN_RUN_MODE:-baseline}" = full-norm-replay || "${QWEN_RUN_MODE:-baseline}" = target-feature-replay || "${QWEN_RUN_MODE:-baseline}" = target-feature-prefix ]]; then
    feature_options=()
    if [ "$QWEN_RUN_MODE" = target-feature-replay ]; then feature_options+=(--target-feature-replay); fi
    if [ "$QWEN_RUN_MODE" = target-feature-prefix ]; then feature_options+=(--target-feature-replay --target-feature-prefix); fi
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --deferred-commit --commit-dma --captured-commit --ordered-cache --replay-inputs --norm-batch "${feature_options[@]}"
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-attention-tree ]; then
    timeout -k 30 1920 bash /experiment-scripts/ci/sdpa-tree-build.sh
    export QWEN_SDPA_TREE_SCRATCH_ROUNDS=1
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --ordered-cache --norm-batch --grouped-attention --attention-dma --attention-parallel --attention-tree "${prefix_copy_args[@]}"
    exit 0
fi
if [[ "${QWEN_RUN_MODE:-baseline}" = full-attention-replay || "${QWEN_RUN_MODE:-baseline}" = full-attention-mask-once || "${QWEN_RUN_MODE:-baseline}" = full-attention-tree-replay ]]; then
    mask_options=()
    if [ "$QWEN_RUN_MODE" = full-attention-tree-replay ]; then
        timeout -k 30 1920 bash /experiment-scripts/ci/sdpa-tree-build.sh
        export QWEN_SDPA_TREE_SCRATCH_ROUNDS=1
        mask_options+=(--attention-mask-once --replay-group-rows 8)
    fi
    if [ "$QWEN_RUN_MODE" = full-attention-mask-once ]; then mask_options+=(--attention-mask-once); fi
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --deferred-commit --commit-dma --captured-commit --ordered-cache --replay-inputs --norm-batch --attention-replay "${mask_options[@]}"
    exit 0
fi
if [[ "${QWEN_RUN_MODE:-baseline}" = full-attention-engine || "${QWEN_RUN_MODE:-baseline}" = full-attention-engine-wide ]]; then
    engine_options=()
    if [ "$QWEN_RUN_MODE" = full-attention-engine-wide ]; then
        timeout -k 30 1920 bash /experiment-scripts/ci/sdpa-tree-build.sh
        export QWEN_SDPA_TREE_SCRATCH_ROUNDS=1
        engine_options+=(--attention-engine-wide)
    fi
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --ordered-cache --device-selection --request-pilot --norm-batch --attention-engine "${engine_options[@]}"
    exit 0
fi
if [[ "${QWEN_RUN_MODE:-baseline}" = full-norm-batch || "${QWEN_RUN_MODE:-baseline}" = target-feature-batch ]]; then
    feature_options=()
    if [ "$QWEN_RUN_MODE" = target-feature-batch ]; then feature_options+=(--target-feature-batch); fi
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --ordered-cache --norm-batch "${feature_options[@]}"
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-attention-groups ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --ordered-cache --norm-batch --grouped-attention
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-attention-dma ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --ordered-cache --norm-batch --grouped-attention --attention-dma
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-attention-parallel ]; then
    timeout -k 15 90 python3 /experiment-scripts/ci/sdpa-tree-audit.py
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --ordered-cache --norm-batch --grouped-attention --attention-dma --attention-parallel
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = verifier-profile ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    bash /experiment-scripts/ci/verifier-profile.sh
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-gdn-device-loop ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1200 python3 /experiment-scripts/ci/gdn-multitoken-conv.py --max-rows 32 --continuation --full-layer --batch-conv --dma-windows --packed-checkpoints
    timeout -k 30 4800 python3 /experiment-scripts/ci/full-prefix.py --max-rows 32 --batch --coding-cost --serial-sdpa --compact-gdn --reuse-gdn-input --skip-row-clones --hoist-row-layout --device-loop-gdn --compact-prologue --batch-conv --packed-checkpoints --ordered-cache
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = full-batch-attribution ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/full-prefix.py --batch --serial-sdpa --attribution
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-prefix ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/gdn-prefix.py
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-block ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/gdn-prefix.py --batch-output
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-active ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/gdn-prefix.py --batch-output --active-snapshot
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-inplace ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/gdn-prefix.py --batch-output --active-snapshot --direct-snapshot --working-state
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-inplace-timing ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/gdn-prefix.py --batch-output --active-snapshot --direct-snapshot --working-state --paired-timing
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-checkpoint-cost ]; then
    PYTHONPATH="/experiment-scripts/ci:$PYTHONPATH" timeout -k 30 240 python3 /experiment-optimisation/sim/gdn-prefix-copy.py \
        --hardware --output /experiment/results/gdn-prefix-copy.json
    timeout -k 30 1800 python3 /experiment-scripts/ci/gdn-prefix.py --batch-output --active-snapshot --direct-snapshot --working-state --paired-timing --checkpoint-diagnostics
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-checkpoint-dma ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/gdn-prefix.py --batch-output --active-snapshot --direct-snapshot --working-state --paired-timing --checkpoint-diagnostics --compact-checkpoint-dma
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-multitoken ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/gdn-multitoken.py
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-value-split ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1200 python3 /experiment-scripts/ci/gdn-multitoken.py --norm-gate --value-split
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-value-split-timing ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1200 python3 /experiment-scripts/ci/gdn-vsplit-timing.py --stage-timing
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-value-split-prefetch ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1200 python3 /experiment-scripts/ci/gdn-vsplit-timing.py --prefetch-inputs
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-value-split-norm-batch ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1200 python3 /experiment-scripts/ci/gdn-vsplit-timing.py --batch-norm --stage-timing
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-norm-batch-layer ]; then
    timeout -k 15 180 python3 /experiment-scripts/ci/device-readback.py
    timeout -k 30 1800 python3 /experiment-scripts/ci/gdn-multitoken-conv.py --norm-batch --max-rows 32 --continuation --full-layer --paired-timing --batch-conv --dma-windows --packed-checkpoints
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-multitoken-norm ]; then
    timeout -k 30 420 python3 /experiment-scripts/ci/gdn-multitoken.py --norm-gate
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-multitoken-conv ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/gdn-multitoken-conv.py --continuation --full-layer --paired-timing --batch-conv --dma-windows --packed-checkpoints
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = gdn-direct ]; then
    timeout -k 30 1800 python3 /experiment-scripts/ci/gdn-prefix.py --batch-output --active-snapshot --direct-snapshot
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = projection-1d ]; then
    timeout -k 30 900 python3 /experiment-scripts/ci/projection-1d.py
    timeout -k 30 900 python3 /experiment-scripts/ci/mlp-sweep.py --projection-report /experiment/results/projection-1d.json
    exit 0
fi
if [[ "${QWEN_RUN_MODE:-baseline}" = mlp-sweep || "${QWEN_RUN_MODE:-baseline}" = mlp-packing || "${QWEN_RUN_MODE:-baseline}" = mlp-fusion ]]; then
    mlp_args=()
    if [ "$QWEN_RUN_MODE" = mlp-packing ]; then mlp_args=(--packing); fi
    if [ "$QWEN_RUN_MODE" = mlp-fusion ]; then mlp_args=(--fusion); fi
    timeout -k 30 1800 python3 /experiment-scripts/ci/mlp-sweep.py "${mlp_args[@]}"
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = profile ]; then
    bash /experiment-scripts/ci/module-profile.sh
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = model-profile ]; then
    bash /experiment-scripts/ci/model-profile.sh
    exit 0
fi
extra_args=()
tt_config='{"tt":{"l1_small_size":24576,"fabric_config":"FABRIC_1D","trace_region_size":1073741824}}'
if [[ "${QWEN_RUN_MODE:-baseline}" = sampling || "${QWEN_RUN_MODE:-baseline}" = sampling-extended ]]; then
    python3 /experiment-optimisation/sim/stage-sampling.py > /experiment/results/sampling-stage.json
    tt_config='{"tt":{"l1_small_size":24576,"fabric_config":"FABRIC_1D","trace_region_size":1073741824,"sample_on_device_mode":"decode_only"}}'
    extra_args=(--limit-mm-per-prompt '{"image":0,"video":0}' --no-enable-mm-embeds)
fi
if [ "${QWEN_RUN_MODE:-baseline}" = interleave ]; then
    source /experiment-scripts/ci/interleave-args.sh
    python3 /experiment-optimisation/sim/stage-continuation.py /opt/tt-metal --apply > /experiment/results/continuation-stage.json
    python3 /experiment-optimisation/sim/stage-plugin.py /opt/vllm-tt-plugin --apply > /experiment/results/plugin-stage.json
fi
printf 'mode=%s\nratio=%s\ncontinuation=%s\ninterleave=%s\n' "${QWEN_RUN_MODE:-baseline}" \
    "${QWEN_INTERLEAVE_RATIO:-0}" "$QWEN_PREFILL_CONTINUATION" "$TT_PREFILL_DECODE_INTERLEAVE" > /experiment/results/arm.txt
python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.8-27B --revision "$revision" --served-model-name qwen3.8-27b \
    --max-model-len 65536 --max-num-seqs 8 --no-enable-prefix-caching --block-size 64 \
    --reasoning-parser qwen3 --port 8000 --host 127.0.0.1 \
    --additional-config "$tt_config" "${extra_args[@]}" \
    > /experiment/results/server.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
ready=0
for attempt in $(seq 1 180); do
    if curl -sf http://127.0.0.1:8000/v1/models > /experiment/results/models.json; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -80 /experiment/results/server.log; exit 1; fi
    if [ $((attempt % 12)) = 0 ]; then printf 'Waiting for endpoint: %s seconds\n' "$((attempt * 5))"; tail -3 /experiment/results/server.log; fi
    sleep 5
done
if [ "$ready" != 1 ]; then tail -80 /experiment/results/server.log; exit 1; fi
python3 - <<'PY' > /experiment/results/api-capabilities.json
import json
import urllib.request
with urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=30) as response:
    schema = json.load(response)
schemas = schema.get('components', {}).get('schemas', {})
print(json.dumps({name: list(value.get('properties', {})) for name, value in schemas.items()
                  if 'Completion' in name}, indent=2))
PY
printf 'Endpoint ready; starting warmed baseline matrix\n'
if [[ "${QWEN_RUN_MODE:-baseline}" = sampling || "${QWEN_RUN_MODE:-baseline}" = sampling-extended ]]; then
    timeout 3600 python3 /experiment-scripts/ci/sampling-client.py --tokenizer "$MODEL_WEIGHTS_DIR" --output /experiment/results "${sampling_args[@]}"
    exit 0
fi
if [ "${QWEN_RUN_MODE:-baseline}" = interleave ]; then
    timeout 2400 python3 /experiment-scripts/ci/interleave-client.py --tokenizer "$MODEL_WEIGHTS_DIR" --output /experiment/results
    exit 0
fi
timeout 4800 python3 /experiment-scripts/ci/baseline-client.py --tokenizer "$MODEL_WEIGHTS_DIR" \
    --output /experiment/results --context 65536 --tokens 1024 --repeats 3
