#!/usr/bin/env bash
set -euo pipefail
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test "${RUNNER_NAME:-}" = thatch-build-amd64-02-cp-temp
test -z "${TT_METAL_SIMULATOR:-}"
splitk_combined=${QWEN_SPLITK_COMBINED:-0}
mlp_64k_audit=${QWEN_64K_MLP_AUDIT:-0}
mlp_64k_timed=${QWEN_64K_MLP_TIMED:-0}
shared_64k_audit=${QWEN_64K_SHARED_QK_AUDIT:-0}
score_64k_audit=${QWEN_64K_SCORE_AUDIT:-0}
text_only_load=${QWEN_TEXT_ONLY_LOAD:-0}
[[ "$text_only_load" = 0 || "$text_only_load" = 1 ]]
if [ "$text_only_load" = 1 ]; then
    test "$score_64k_audit" = 1
    test "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1
    test "${QWEN_DSPARK_SFPU_TIMED:-0}" = 0
fi
score_64k_timed=${QWEN_64K_SCORE_TIMED:-0}
score_paired=${QWEN_SCORE_PAIRED:-0}
[[ "$score_paired" = 0 || "$score_paired" = 1 ]]
if [ "$score_paired" = 1 ]; then test "$score_64k_timed" = 1; fi
[[ "$score_64k_timed" = 0 || "$score_64k_timed" = 1 ]]
if [ "$score_64k_timed" = 1 ]; then
    test "${QWEN_64K_SHARED_QK_TIMED:-0}" = 1
    test "$score_64k_audit" = 0
fi
[[ "$score_64k_audit" = 0 || "$score_64k_audit" = 1 ]]
if [ "$score_64k_audit" = 1 ]; then test "$shared_64k_audit" = 1; fi
shared_64k_timed=${QWEN_64K_SHARED_QK_TIMED:-0}
[[ "$shared_64k_timed" = 0 || "$shared_64k_timed" = 1 ]]
if [ "$shared_64k_timed" = 1 ]; then
    test "$mlp_64k_timed" = 1
    test "$shared_64k_audit" = 0
fi
[[ "$shared_64k_audit" = 0 || "$shared_64k_audit" = 1 ]]
if [[ "$shared_64k_audit" = 1 || "$shared_64k_timed" = 1 ]]; then
    if [ "$shared_64k_audit" = 1 ]; then test "$mlp_64k_audit" = 1; fi
    shared_64k_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-64k-shared.XXXXXX")
    timeout -k 5 45 gh run download 34701425373 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34701425373 --dir "$shared_64k_evidence"
    printf '%s  %s\n' a155b893786f68ac36b4f040454892df683a0da0c3aa0b22b898697b22241ffe "$shared_64k_evidence/gdn-shared-recurrence.json" | sha256sum -c -
    test "$(cat "$shared_64k_evidence/gdn-shared-recurrence.exit-status")" = 0
    cp "$shared_64k_evidence/gdn-shared-recurrence.json" scripts/ci/gdn-shared-recurrence.json
fi
[[ "$mlp_64k_timed" = 0 || "$mlp_64k_timed" = 1 ]]
if [ "$mlp_64k_timed" = 1 ]; then
    test "$splitk_combined" = 1
    test "$mlp_64k_audit" = 0
    test "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1
fi
[[ "$mlp_64k_audit" = 0 || "$mlp_64k_audit" = 1 ]]
if [ "$mlp_64k_audit" = 1 ]; then
    test "$splitk_combined" = 1
    test "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1
    test "${QWEN_DSPARK_SFPU_TIMED:-0}" = 0
fi
[[ "$splitk_combined" = 0 || "$splitk_combined" = 1 ]]
if [ "$splitk_combined" = 1 ]; then
    [[ "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}:${QWEN_DSPARK_SFPU_TIMED:-0}" = 1:0 || "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}:${QWEN_DSPARK_SFPU_TIMED:-0}" = 0:1 ]]
    test "${QWEN_DSPARK_CENTER_TILE_FILL:-0}" = 1
    splitk_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-combined-splitk.XXXXXX")
    timeout -k 5 45 gh api repos/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/artifacts/10376709944/zip > "$splitk_evidence/hardware.zip"
    timeout -k 5 45 gh api repos/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/artifacts/10376559640/zip > "$splitk_evidence/simulator.zip"
    python3 - "$splitk_evidence" <<'PY'
from pathlib import Path
import sys
import zipfile
root = Path(sys.argv[1])
for archive, member, destination in (
    ('hardware.zip', 'dspark-splitk-hardware.json', 'dspark-splitk-hardware.json'),
    ('simulator.zip', 'dspark-splitk.json', 'dspark-splitk-simulator.json')):
    with zipfile.ZipFile(root / archive) as source:
        (Path('scripts/ci') / destination).write_bytes(source.read(member))
PY
    PYTHONPATH=scripts/ci python3 -c 'from dspark_splitk_hardware_gate import qualify; qualify("scripts/ci", "scripts/ci/dspark-splitk-hardware.json", "scripts/ci/dspark-splitk-simulator.json")'
fi
mode=${QWEN_DSPARK_MODE:-backbone}
trial_64k=${QWEN_DSPARK_64K_TRIAL:-0}
phase_probe=${QWEN_DSPARK_PHASE_PROBE:-0}
center_tile_fill=${QWEN_DSPARK_CENTER_TILE_FILL:-0}
[[ "$center_tile_fill" = 0 || "$center_tile_fill" = 1 ]]
if [ "$center_tile_fill" = 1 ]; then
    test "${QWEN_DSPARK_NORMALIZATION_DIRECT_STAGE:-0}" = 1
    [[ "${QWEN_DSPARK_SFPU_NUMERICAL:-0}" = 1 || "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1 || "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1 ]]
    center_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-center-fill.XXXXXX")
    gh run download 34840912016 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34840912016 --dir "$center_evidence"
    cp "$center_evidence/dspark-center-tile-fill.json" scripts/ci/dspark-center-tile-fill.json
    PYTHONPATH=scripts/ci python3 -c 'from dspark_center_fill_gate import qualify; qualify("scripts/ci", "scripts/ci/dspark-center-tile-fill.json")'
    if [[ "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1 || "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1 ]]; then
        test "${QWEN_TARGET_T16_64K_REQUEST:-0}" = 1
        center_hardware_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-center-hardware.XXXXXX")
        gh run download 34841454010 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
            --name qwen-hardware-inventory-34841454010 --dir "$center_hardware_evidence"
        cp "$center_hardware_evidence/dspark-center-tile-fill-hardware.json" scripts/ci/dspark-center-tile-fill-hardware.json
        PYTHONPATH=scripts/ci python3 -c 'from dspark_center_fill_request_gate import qualify; qualify("scripts/ci", "scripts/ci/dspark-center-tile-fill-hardware.json")'
    fi
fi
normalization_direct_stage=${QWEN_DSPARK_NORMALIZATION_DIRECT_STAGE:-0}
[[ "$normalization_direct_stage" = 0 || "$normalization_direct_stage" = 1 ]]
if [ "$normalization_direct_stage" = 1 ]; then
    test "${QWEN_DSPARK_DIRECT_FP32_STAGE:-0}" = 1
    [[ "${QWEN_DSPARK_SFPU_NUMERICAL:-0}" = 1 || "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1 || "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1 ]]
    normalization_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-normalization-stage.XXXXXX")
    gh run download 34838104802 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34838104802 --dir "$normalization_evidence"
    cp "$normalization_evidence/dspark-normalization-direct-stage.json" scripts/ci/dspark-normalization-direct-stage.json
    PYTHONPATH=scripts/ci python3 -c 'from dspark_normalization_stage_gate import qualify; qualify("scripts/ci", "scripts/ci/dspark-normalization-direct-stage.json")'
    if [[ "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1 || "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1 ]]; then
        test "${QWEN_TARGET_T16_64K_REQUEST:-0}" = 1
        normalization_hardware_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-normalization-hardware.XXXXXX")
        gh run download 34838689973 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
            --name qwen-hardware-inventory-34838689973 --dir "$normalization_hardware_evidence"
        cp "$normalization_hardware_evidence/dspark-normalization-direct-stage-hardware.json" scripts/ci/dspark-normalization-direct-stage-hardware.json
        PYTHONPATH=scripts/ci python3 -c 'from dspark_normalization_request_gate import qualify; qualify("scripts/ci", "scripts/ci/dspark-normalization-direct-stage-hardware.json")'
    fi
fi
direct_fp32_stage=${QWEN_DSPARK_DIRECT_FP32_STAGE:-0}
[[ "$direct_fp32_stage" = 0 || "$direct_fp32_stage" = 1 ]]
if [ "$direct_fp32_stage" = 1 ]; then
    [[ "${QWEN_DSPARK_SFPU_NUMERICAL:-0}" = 1 || "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1 || "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1 ]]
    test "${QWEN_DSPARK_MASK_BITS:-0}" = 1
    direct_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-direct-fp32.XXXXXX")
    gh run download 34834165490 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34834165490 --dir "$direct_evidence"
    cp "$direct_evidence/dspark-direct-fp32-stage.json" scripts/ci/dspark-direct-fp32-stage.json
    PYTHONPATH=scripts/ci python3 -c 'from dspark_direct_fp32_gate import qualify; qualify("scripts/ci", "scripts/ci/dspark-direct-fp32-stage.json")'
    if [[ "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1 || "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1 ]]; then
        test "${QWEN_TARGET_T16_64K_REQUEST:-0}" = 1
        direct_hardware_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-direct-fp32-hardware.XXXXXX")
        gh run download 34834720985 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
            --name qwen-hardware-inventory-34834720985 --dir "$direct_hardware_evidence"
        cp "$direct_hardware_evidence/dspark-direct-fp32-stage-hardware.json" scripts/ci/dspark-direct-fp32-stage-hardware.json
        PYTHONPATH=scripts/ci python3 -c 'from dspark_direct_fp32_request_gate import qualify; qualify("scripts/ci", "scripts/ci/dspark-direct-fp32-stage-hardware.json")'
    fi
fi
target_request=${QWEN_TARGET_T16_64K_REQUEST:-0}
[[ "$target_request" = 0 || "$target_request" = 1 ]]
if [ "$target_request" = 1 ]; then
    [[ "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1 && "${QWEN_DSPARK_SFPU_TIMED:-0}" = 0 || \
       "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 0 && "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1 ]]
    test "${QWEN_DSPARK_MASK_BITS:-0}" = 1
    test "${QWEN_DSPARK_SFPU_NUMERICAL:-0}" = 0
    target_request_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-target64-request.XXXXXX")
    gh run download 34828634864 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34828634864 --dir "$target_request_evidence"
    cp "$target_request_evidence/target-t16-attention-64k.json" scripts/ci/target-t16-attention-64k-hardware.json
    boundary_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-attention-boundary.XXXXXX")
    gh run download 34830639528 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34830639528 --dir "$boundary_evidence"
    cp "$boundary_evidence/dspark-attention-boundary.json" scripts/ci/dspark-attention-boundary.json
    PYTHONPATH=scripts/ci python3 -c 'from target_t16_64k_request import qualify; qualify("scripts/ci")'
fi
target_64k=${QWEN_TARGET_T16_64K:-0}
[[ "$target_64k" = 0 || "$target_64k" = 1 ]]
if [ "$target_64k" = 1 ]; then
    test "${QWEN_DSPARK_SFPU_NUMERICAL:-0}" = 1
    target_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-target64.XXXXXX")
    gh run download 34828115400 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34828115400 --dir "$target_evidence"
    PYTHONPATH=scripts/ci python3 -c 'import sys; from target_t16_64k_gate import qualify; qualify("scripts/ci", sys.argv[1])' \
        "$target_evidence/target-t16-attention-64k.json"
fi
score_sfpu=${QWEN_DSPARK_SCORE_SFPU:-0}
sum_sfpu=${QWEN_DSPARK_SUM_SFPU:-0}
mask_bits=${QWEN_DSPARK_MASK_BITS:-0}
[[ "$mask_bits" = 0 || "$mask_bits" = 1 ]]
if [ "$mask_bits" = 1 ]; then
    test "$sum_sfpu" = 1
    [[ "${QWEN_DSPARK_SFPU_NUMERICAL:-0}" = 1 || "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1 || "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1 ]]
    mask_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-mask-bits.XXXXXX")
    gh run download 34825080088 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34825080088 --dir "$mask_evidence"
    PYTHONPATH=scripts/ci python3 -c 'import sys; from dspark_mask_bits_gate import qualify; qualify("scripts/ci", sys.argv[1])' \
        "$mask_evidence/dspark-mask-bits.json"
    if [[ "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1 || "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1 ]]; then
        mask_hardware_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-mask-hardware.XXXXXX")
        gh run download 34825617040 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
            --name qwen-hardware-inventory-34825617040 --dir "$mask_hardware_evidence"
        PYTHONPATH=scripts/ci python3 -c 'import sys; from dspark_mask_request_gate import qualify; qualify("scripts/ci", sys.argv[1])' \
            "$mask_hardware_evidence/dspark-mask-bits-hardware.json"
    fi
fi
[[ "$sum_sfpu" = 0 || "$sum_sfpu" = 1 ]]
if [ "$sum_sfpu" = 1 ]; then
    test "$score_sfpu" = 1
    [[ "${QWEN_DSPARK_SFPU_NUMERICAL:-0}" = 1 || "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1 || "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1 ]]
    sum_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-sum-sfpu.XXXXXX")
    gh run download 34821692776 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34821692776 --dir "$sum_evidence"
    PYTHONPATH=scripts/ci python3 -c 'import sys; from dspark_sum_sfpu_gate import qualify; qualify("scripts/ci", sys.argv[1])' \
        "$sum_evidence/dspark-sum-sfpu.json"
    if [[ "${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}" = 1 || "${QWEN_DSPARK_SFPU_TIMED:-0}" = 1 ]]; then
        sum_hardware_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-sum-hardware.XXXXXX")
        gh run download 34822563217 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
            --name qwen-hardware-inventory-34822563217 --dir "$sum_hardware_evidence"
        PYTHONPATH=scripts/ci python3 -c 'import sys; from dspark_sum_request_gate import qualify; qualify("scripts/ci", sys.argv[1])' \
            "$sum_hardware_evidence/dspark-sum-sfpu-hardware.json"
    fi
fi
request_screen=${QWEN_DSPARK_SFPU_REQUEST_SCREEN:-0}
timed_requests=${QWEN_DSPARK_SFPU_TIMED:-0}
[[ "$timed_requests" = 0 || "$timed_requests" = 1 ]]
if [ "$timed_requests" = 1 ]; then test "$request_screen" = 0; fi
[[ "$request_screen" = 0 || "$request_screen" = 1 ]]
if [[ "$request_screen" = 1 || "$timed_requests" = 1 ]]; then
    test "$score_sfpu" = 1
    request_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-sfpu-request.XXXXXX")
    gh run download 34817498912 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34817498912 --dir "$request_evidence"
    PYTHONPATH=scripts/ci python3 -c 'import sys; from dspark_score_sfpu_request_gate import qualify; qualify("scripts/ci", sys.argv[1])' \
        "$request_evidence/dspark-score-sfpu-hardware.json"
fi
if [ "$timed_requests" = 1 ]; then
    screen_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-sfpu-screen.XXXXXX")
    screen_run=34819480314
    timing_gate=dspark_sfpu_timed_requests
    if [ "$sum_sfpu" = 1 ]; then
        screen_run=34823325684
        timing_gate=dspark_sum_timed_requests
        cp "$sum_hardware_evidence/dspark-sum-sfpu-hardware.json" "$request_evidence/dspark-sum-sfpu-hardware.json"
    fi
    if [ "$mask_bits" = 1 ]; then
        screen_run=34825996484
        timing_gate=dspark_mask_timed_requests
        cp "$mask_hardware_evidence/dspark-mask-bits-hardware.json" "$request_evidence/dspark-mask-bits-hardware.json"
    fi
    if [ "$target_request" = 1 ]; then
        screen_run=34831200764
        timing_gate=target_t16_64k_timed
    fi
    if [ "$direct_fp32_stage" = 1 ]; then
        screen_run=34835869483
        timing_gate=dspark_direct_fp32_timed
    fi
    if [ "$normalization_direct_stage" = 1 ]; then
        screen_run=34839119120
        timing_gate=dspark_normalization_timed
    fi
    if [ "$center_tile_fill" = 1 ]; then
        screen_run=34841889707
        timing_gate=dspark_center_fill_timed
    fi
    screen_artifact="qwen-hardware-inventory-$screen_run"
    if [ "$splitk_combined" = 1 ]; then
        screen_run=34922265472
        screen_artifact="qwen-splitk-combined-$screen_run"
        timing_gate=dspark_splitk_timed
        if [ "$mlp_64k_timed" = 1 ]; then
            screen_run=34924260072
            screen_artifact="qwen-splitk-combined-$screen_run"
            timing_gate=dspark_64k_mlp_timed
            if [ "$shared_64k_timed" = 1 ]; then
                screen_run=34925963042
                screen_artifact="qwen-splitk-combined-$screen_run"
                timing_gate=dspark_64k_shared_timed
                if [ "$score_64k_timed" = 1 ]; then
                    screen_run=34929473486
                    screen_artifact="qwen-splitk-combined-$screen_run"
                    timing_gate=dspark_64k_score_timed
                fi
            fi
        fi
    fi
    gh run download "$screen_run" --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name "$screen_artifact" --dir "$screen_evidence"
    cp "$screen_evidence/dspark-64k-request-hardware.json" "$request_evidence/dspark-sfpu-request-screen.json"
    PYTHONPATH=scripts/ci python3 -c 'import importlib, sys; importlib.import_module(sys.argv[2]).qualify("scripts/ci", sys.argv[1])' "$request_evidence" "$timing_gate"
fi
sfpu_numerical=${QWEN_DSPARK_SFPU_NUMERICAL:-0}
[[ "$sfpu_numerical" = 0 || "$sfpu_numerical" = 1 ]]
if [ "$sfpu_numerical" = 1 ]; then test "$score_sfpu" = 1; fi
[[ "$score_sfpu" = 0 || "$score_sfpu" = 1 ]]
if [ "$score_sfpu" = 1 ]; then
    test "$phase_probe" = 1
    sfpu_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-score-sfpu.XXXXXX")
    gh run download 34815143244 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34815143244 --dir "$sfpu_evidence"
    PYTHONPATH=scripts/ci python3 -c 'import sys; from dspark_score_sfpu_gate import qualify; qualify("scripts/ci", sys.argv[1])' \
        "$sfpu_evidence/dspark-score-sfpu.json"
fi
score_bitwise=${QWEN_DSPARK_SCORE_BITWISE:-0}
[[ "$score_bitwise" = 0 || "$score_bitwise" = 1 ]]
if [ "$score_bitwise" = 1 ]; then
    test "$phase_probe" = 1
    score_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-score-candidate.XXXXXX")
    gh run download 34810827485 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34810827485 --dir "$score_evidence"
    PYTHONPATH=scripts/ci python3 -c 'import sys; from dspark_score_candidate_gate import qualify; qualify("scripts/ci", sys.argv[1])' \
        "$score_evidence/dspark-score-bitwise.json"
fi
[[ "$phase_probe" = 0 || "$phase_probe" = 1 ]]
if [ "$phase_probe" = 1 ]; then test "$trial_64k" = 1; fi
[[ "$trial_64k" = 0 || "$trial_64k" = 1 ]]
if [ "$trial_64k" = 1 ]; then
    test "$mode" = request-norm-scatter
    test "${QWEN_DSPARK_CAPTURED_PUBLICATION:-0}" = 1
    for flag in QWEN_DSPARK_FUSION_T16 QWEN_DSPARK_SCORE_LAYOUT QWEN_DSPARK_BANKED_PROPOSAL QWEN_DSPARK_NATIVE_SLOT QWEN_DSPARK_MLP_DOWN QWEN_DSPARK_BIAS_CACHE QWEN_DSPARK_HISTORY_PROFILE QWEN_DSPARK_DRAFT_PROFILE QWEN_DSPARK_MLP_FOOTPRINT; do
        test "${!flag:-0}" = 0
    done
    test "${QWEN_DSPARK_CODING_TASK:-merge_intervals}" = merge_intervals
    draft_64k_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-draft-attention-64k.XXXXXX")
    gh run download 34797353681 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34797353681 --dir "$draft_64k_evidence"
    PYTHONPATH=scripts/ci python3 -c 'import sys; from dspark_attention_64k_gate import qualify; qualify("scripts/ci", sys.argv[1])' \
        "$draft_64k_evidence/dspark-ladder-hardware-65536.json"
fi
draft_profile=${QWEN_DSPARK_DRAFT_PROFILE:-0}
history_profile=${QWEN_DSPARK_HISTORY_PROFILE:-0}
[[ "$history_profile" = 0 || "$history_profile" = 1 ]]
if [ "$history_profile" = 1 ]; then
    test "$mode" = request-target-attention
    test "$draft_profile" = 0
    test "${QWEN_DSPARK_BANKED_PROPOSAL:-0}" = 0
fi
[[ "$draft_profile" = 0 || "$draft_profile" = 1 ]]
if [ "$draft_profile" = 1 ]; then test "$mode" = request-verifier-profile; fi
mlp_down=${QWEN_DSPARK_MLP_DOWN:-0}
score_layout=${QWEN_DSPARK_SCORE_LAYOUT:-0}
banked_proposal=${QWEN_DSPARK_BANKED_PROPOSAL:-0}
native_slot=${QWEN_DSPARK_NATIVE_SLOT:-0}
fusion=${QWEN_DSPARK_FUSION_T16:-0}
publication=${QWEN_DSPARK_CAPTURED_PUBLICATION:-0}
bias_cache=${QWEN_DSPARK_BIAS_CACHE:-0}
[[ "$bias_cache" = 0 || "$bias_cache" = 1 ]]
if [ "$bias_cache" = 1 ]; then
    test "$publication" = 1
    cache_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-bias-cache.XXXXXX")
    gh run download 34735206013 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34735206013 --dir "$cache_evidence"
    PYTHONPATH=scripts/ci python3 -c 'import sys; from dspark_cached_markov_gate import qualify; qualify(sys.argv[1], "scripts/ci")' "$cache_evidence"
fi
[[ "$publication" = 0 || "$publication" = 1 ]]
publication_report=''
if [ "$publication" = 1 ]; then
    if [ "$trial_64k" = 0 ]; then
        test "$fusion" = 1
        test "$mode" = request-target-attention
    fi
    test "$history_profile" = 0
    evidence=$(mktemp -d "$RUNNER_TEMP/qwen-publication-evidence.XXXXXX")
    gh run download 34677941763 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34677941763 --dir "$evidence"
    publication_report="$evidence/t32-publication.json"
    if [ "$trial_64k" = 0 ]; then
    output_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-gdn-output-l1.XXXXXX")
    gh run download 34693525557 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34693525557 --dir "$output_evidence"
    printf '%s  %s\n' 7b874af8e9f092cca0a84c134cb19da560c41aa8524bd76a2de539f3c9fae52e "$output_evidence/gdn-output-l1.json" | sha256sum -c -
    test "$(cat "$output_evidence/gdn-output-l1.exit-status")" = 0
    grid_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-gdn-output-grid.XXXXXX")
    gh run download 34694674606 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34694674606 --dir "$grid_evidence"
    printf '%s  %s\n' c8017ca9258dd32eb88f9833956bfe133b28487eb75e2487363c3ca86c9271ba "$grid_evidence/gdn-output-grid.json" | sha256sum -c -
    test "$(cat "$grid_evidence/gdn-output-grid.exit-status")" = 0
    copy_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-gdn-copy-pairs.XXXXXX")
    gh run download 34695921492 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34695921492 --dir "$copy_evidence"
    printf '%s  %s\n' 14d9fa6ae9b2e5ec0674d5b846f382c141e38ee73fa423ddb82a8ac73258bd99 "$copy_evidence/gdn-copy-pairs.json" | sha256sum -c -
    test "$(cat "$copy_evidence/gdn-copy-pairs.exit-status")" = 0
    outer_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-gdn-outer-add.XXXXXX")
    shared_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-gdn-shared-qk.XXXXXX")
    attention_8k_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-attention-8k.XXXXXX")
    draft_8k_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-draft-attention-8k.XXXXXX")
    gh run download 34728453080 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34728453080 --dir "$draft_8k_evidence"
    printf '%s  %s\n' 2baa47c721607fac512b72413024c15a510ef475954cab976ee6d59222ab61af "$draft_8k_evidence/dspark-native-8k-attention.json" | sha256sum -c -
    test "$(cat "$draft_8k_evidence/dspark-native-8k-attention.exit-status")" = 0
    gh run download 34703126782 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34703126782 --dir "$attention_8k_evidence"
    printf '%s  %s\n' 1a60ca077425c671ea9e4b30ddf0490700382d835b7aee3327d320c2ce8b83cd "$attention_8k_evidence/target-t16-attention-8k.json" | sha256sum -c -
    test "$(cat "$attention_8k_evidence/target-t16-attention-8k.exit-status")" = 0
    gh run download 34701425373 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34701425373 --dir "$shared_evidence"
    printf '%s  %s\n' a155b893786f68ac36b4f040454892df683a0da0c3aa0b22b898697b22241ffe "$shared_evidence/gdn-shared-recurrence.json" | sha256sum -c -
    test "$(cat "$shared_evidence/gdn-shared-recurrence.exit-status")" = 0
    gh run download 34699176210 --repo Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B \
        --name qwen-hardware-inventory-34699176210 --dir "$outer_evidence"
    printf '%s  %s\n' 036be9bbfa0c8de8e5f0415beb2d5d6447a5755ac894e6e0cd6ba238d4eada52 "$outer_evidence/gdn-outer-add.json" | sha256sum -c -
    test "$(cat "$outer_evidence/gdn-outer-add.exit-status")" = 0
    fi
    printf '%s  %s\n' 4bd749d6381cb7e1f5be69276d5a5011c9e6cfa7c30182b44dd09f3d1b115914 "$publication_report" | sha256sum -c -
fi
[[ "$fusion" = 0 || "$fusion" = 1 ]]
if [ "$fusion" = 1 ]; then test "$score_layout" = 1; test "$native_slot" = 0; test "$banked_proposal" = 0; fi
[[ "$native_slot" = 0 || "$native_slot" = 1 ]]
if [ "$native_slot" = 1 ]; then test "$score_layout" = 1; test "$banked_proposal" = 0; fi
[[ "$banked_proposal" = 0 || "$banked_proposal" = 1 ]]
if [ "$banked_proposal" = 1 ]; then test "$score_layout" = 1; fi
[[ "$score_layout" = 0 || "$score_layout" = 1 ]]
if [ "$score_layout" = 1 ]; then
    test "$mode" = request-target-attention
    test "$mlp_down" = 0
    test "$draft_profile" = 0
fi
mlp_footprint=${QWEN_DSPARK_MLP_FOOTPRINT:-0}
[[ "$mlp_footprint" = 0 || "$mlp_footprint" = 1 ]]
if [ "$mlp_footprint" = 1 ]; then test "$mlp_down" = 1; fi
[[ "$mlp_down" = 0 || "$mlp_down" = 1 ]]
if [ "$mlp_down" = 1 ]; then test "$mode" = request-target-attention; fi
task=${QWEN_DSPARK_CODING_TASK:-merge_intervals}
case "$task" in
    merge_intervals) ;;
    stable_unique_v1|run_length_encode_v1|rotate_right_v1) test "$mode" = request-target-attention ;;
    *) exit 64 ;;
esac
[[ "$mode" = backbone || "$mode" = target || "$mode" = request || "$mode" = request-variants || "$mode" = request-native-attention || "$mode" = request-combined || "$mode" = request-target-attention || "$mode" = request-norm-scatter || "$mode" = request-verifier-profile ]]
target_mount=()
if [ "$mode" != backbone ]; then
    target=/home/thatch/hf-cache/hub/models--Qwen--Qwen3.8-27B
    test -d "$target/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    target_mount=(--mount "type=bind,src=$target,dst=/models/hub/models--Qwen--Qwen3.8-27B,readonly")
fi
output=experiment-results
mkdir -p "$output"
result_mount=()
if [ "$phase_probe" = 1 ]; then
    chmod g+rwx "$output"
    result_mount=(--mount "type=bind,src=$PWD/$output,dst=/experiment/results" --group-add "$(stat -c %g "$output")")
fi
PYTHONPATH=scripts/ci python3 -c \
    'import json; from dspark_hardware_gate import simulator_preflight; print(json.dumps(simulator_preflight("scripts/ci"),indent=2))' \
    > "$output/dspark-simulator-preflight.json"
fixture=/home/thatch/.cache/qwen-experiments/dspark-b9a5dbdf03bc999c6c73c426b19c2d9041cea393
timeout -k 10 1200 python3 scripts/ci/dspark-hardware-fixtures.py --output "$fixture" \
    > "$output/dspark-checkpoint-manifest.json"
image=sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
if [ "$fusion" = 1 ]; then
    timeout -k 15 300 docker run --rm --network none --cap-drop ALL \
        --security-opt no-new-privileges --memory 4g --cpus 2 \
        --mount "type=bind,src=$PWD,dst=/source,readonly" --workdir /source \
        -e PYTHONPATH=/source/scripts/ci:/source/speculative-decoding/harness \
        -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 "$image" -B -m unittest \
        test_packed_weight_check test_dspark_fusion_variants test_fused_t16_scope test_fused_t16_admission \
        test_full_dspark_request test_dspark_score_layout_variants \
        2>&1 | tee "$output/fusion-host-tests.log"
fi
test_id=''
cleanup() {
    status=$?
    trap - EXIT
    if [ -n "$test_id" ]; then
        if [ "$phase_probe" = 1 ] && [ "$status" != 0 ]; then
            timeout -k 1 5 docker kill "$test_id" >/dev/null 2>&1 || true
        fi
        docker logs "$test_id" > "$output/dspark-container.log" 2>&1 || true
        if [ "$phase_probe" = 0 ]; then docker cp "$test_id:/experiment/results/." "$output/" || true; fi
        docker inspect --format '{{json .State}}' "$test_id" > "$output/dspark-container-state.json" || true
        docker rm -f "$test_id" >/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
volume=qwen-experiments-f1e9b1a64b4f
if docker volume inspect "$volume" >/dev/null 2>&1; then
    test "$(docker volume inspect --format '{{index .Labels "thatch.qwen.experiment-cache"}}' "$volume")" = true
else
    docker volume create --label thatch.qwen.experiment-cache=true "$volume" >/dev/null
fi
test_id=$(docker create --network none --hostname qwen-experiment --add-host qwen-experiment:127.0.0.1 \
    --cap-drop ALL --cap-add SYS_NICE --security-opt no-new-privileges \
    --pids-limit 4096 --memory 96g --cpus 24 --shm-size 8g \
    --device /dev/tenstorrent/0 --device /dev/tenstorrent/2 \
    --mount type=bind,src=/dev/tenstorrent,dst=/host-dev/tenstorrent,readonly \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --mount "type=bind,src=$fixture,dst=/dspark,readonly" \
    "${target_mount[@]}" \
    "${result_mount[@]}" \
    --mount "type=volume,src=$volume,dst=/experiment-cache" \
    --label thatch.qwen.baseline=true --workdir /opt/vllm-tt-plugin \
    --label "thatch.qwen.workflow-run=${GITHUB_RUN_ID:-untracked}" \
    --label "thatch.qwen.source-revision=${GITHUB_SHA:-untracked}" \
    -e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1 -e QWEN_PROJECTION_LINKS=4 -e QWEN_CCL_LAZY_BUILD=1 \
    -e "QWEN_DSPARK_MODE=$mode" \
    -e "QWEN_SPLITK_COMBINED=$splitk_combined" \
    -e "QWEN_64K_MLP_AUDIT=$mlp_64k_audit" \
    -e "QWEN_64K_MLP_TIMED=$mlp_64k_timed" \
    -e "QWEN_64K_SHARED_QK_AUDIT=$shared_64k_audit" \
    -e "QWEN_64K_SCORE_AUDIT=$score_64k_audit" \
    -e "QWEN_TEXT_ONLY_LOAD=$text_only_load" \
    -e "QWEN_64K_SHARED_QK_TIMED=$shared_64k_timed" \
    -e "QWEN_64K_SCORE_TIMED=$score_64k_timed" \
    -e "QWEN_SCORE_PAIRED=$score_paired" \
    -e "QWEN_LOAD_SAMPLE=${QWEN_LOAD_SAMPLE:-0}" \
    -e "QWEN_LAZY_WEIGHT_LOAD=${QWEN_LAZY_WEIGHT_LOAD:-0}" \
    -e "QWEN_PUBLICATION_PROFILE=${QWEN_PUBLICATION_PROFILE:-0}" \
    -e "QWEN_DSPARK_64K_TRIAL=$trial_64k" \
    -e "QWEN_DSPARK_PHASE_PROBE=$phase_probe" \
    -e "QWEN_TARGET_T16_64K=$target_64k" \
    -e "QWEN_TARGET_T16_64K_REQUEST=$target_request" \
    -e "QWEN_DSPARK_SCORE_BITWISE=$score_bitwise" \
    -e "QWEN_DSPARK_SCORE_SFPU=$score_sfpu" \
    -e "QWEN_DSPARK_SUM_SFPU=$sum_sfpu" \
    -e "QWEN_DSPARK_MASK_BITS=$mask_bits" \
    -e "QWEN_DSPARK_DIRECT_FP32_STAGE=$direct_fp32_stage" \
    -e "QWEN_DSPARK_NORMALIZATION_DIRECT_STAGE=$normalization_direct_stage" \
    -e "QWEN_DSPARK_CENTER_TILE_FILL=$center_tile_fill" \
    -e "QWEN_DSPARK_COMBINED_PHASES=${QWEN_DSPARK_COMBINED_PHASES:-0}" \
    -e "QWEN_DSPARK_COMBINED_DEVICE_PROFILE=${QWEN_DSPARK_COMBINED_DEVICE_PROFILE:-0}" \
    -e "QWEN_DSPARK_SFPU_REQUEST_SCREEN=$request_screen" \
    -e "QWEN_DSPARK_SFPU_TIMED=$timed_requests" \
    -e "QWEN_DSPARK_SFPU_NUMERICAL=$sfpu_numerical" \
    -e "QWEN_DSPARK_DRAFT_PROFILE=$draft_profile" \
    -e "QWEN_DSPARK_HISTORY_PROFILE=$history_profile" \
    -e "QWEN_DSPARK_MLP_DOWN=$mlp_down" \
    -e "QWEN_DSPARK_SCORE_LAYOUT=$score_layout" \
    -e "QWEN_DSPARK_BANKED_PROPOSAL=$banked_proposal" \
    -e "QWEN_DSPARK_NATIVE_SLOT=$native_slot" \
    -e "QWEN_DSPARK_FUSION_T16=$fusion" \
    -e "QWEN_DSPARK_CAPTURED_PUBLICATION=$publication" \
    -e "QWEN_DSPARK_BIAS_CACHE=$bias_cache" \
    -e "QWEN_DSPARK_MLP_FOOTPRINT=$mlp_footprint" \
    -e "QWEN_DSPARK_CODING_TASK=$task" \
    -e "QWEN_SOURCE_REVISION=${GITHUB_SHA:-untracked}" -e "QWEN_WORKFLOW_RUN=${GITHUB_RUN_ID:-untracked}" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TT_METAL_HOME=/opt/tt-metal \
    -e TT_CACHE_PATH=/experiment-cache/weights -e TT_METAL_CACHE=/experiment-cache/kernels -e MESH_DEVICE=P300 \
    -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=8 \
    --entrypoint /bin/bash "$image" /experiment-scripts/ci/dspark-hardware-suite.sh)
docker cp scripts "$test_id:/experiment-scripts"
if [[ "${QWEN_LAZY_WEIGHT_LOAD:-0}" = 1 && "$timed_requests" = 1 ]]; then
    lazy_evidence=$(mktemp -d "$RUNNER_TEMP/qwen-lazy-loader.XXXXXX")
    gh run download 34936162975 --repo "$GITHUB_REPOSITORY" \
        --name qwen-splitk-combined-34936162975 --dir "$lazy_evidence"
    docker cp "$lazy_evidence/dspark-64k-request-hardware.json" \
        "$test_id:/experiment-scripts/ci/qwen-lazy-weight-audit.json"
fi
if [ "$bias_cache" = 1 ]; then
    for name in dspark-cached-markov markov-cache-pipeline markov-cache-pipeline-4992 markov-cache-pipeline-3712; do
        docker cp "$cache_evidence/$name.json" "$test_id:/experiment-scripts/ci/$name.json"
    done
fi
if [ "$publication" = 1 ]; then
    docker cp "$publication_report" "$test_id:/experiment-scripts/ci/dspark-publication-simulator.json"
    if [ "$trial_64k" = 0 ]; then
    docker cp "$output_evidence/gdn-output-l1.json" "$test_id:/experiment-scripts/ci/gdn-output-l1.json"
    docker cp "$grid_evidence/gdn-output-grid.json" "$test_id:/experiment-scripts/ci/gdn-output-grid.json"
    docker cp "$copy_evidence/gdn-copy-pairs.json" "$test_id:/experiment-scripts/ci/gdn-copy-pairs.json"
    docker cp "$outer_evidence/gdn-outer-add.json" "$test_id:/experiment-scripts/ci/gdn-outer-add.json"
    docker cp "$shared_evidence/gdn-shared-recurrence.json" "$test_id:/experiment-scripts/ci/gdn-shared-recurrence.json"
    docker cp "$attention_8k_evidence/target-t16-attention-8k.json" "$test_id:/experiment-scripts/ci/target-t16-attention-8k.json"
    docker cp "$draft_8k_evidence/dspark-native-8k-attention.json" "$test_id:/experiment-scripts/ci/dspark-native-8k-attention.json"
    fi
fi
if [ "$trial_64k" = 1 ]; then
    docker cp "$draft_64k_evidence/dspark-ladder-hardware-65536.json" "$test_id:/experiment-scripts/ci/dspark-ladder-hardware-65536.json"
fi
docker cp optimisation "$test_id:/experiment-optimisation"
if [ "$score_sfpu" = 1 ]; then
    docker cp "$sfpu_evidence/dspark-score-sfpu.json" "$test_id:/experiment-scripts/ci/dspark-score-sfpu.json"
fi
if [ "$sum_sfpu" = 1 ]; then
    docker cp "$sum_evidence/dspark-sum-sfpu.json" "$test_id:/experiment-scripts/ci/dspark-sum-sfpu.json"
    if [[ "$request_screen" = 1 || "$timed_requests" = 1 ]]; then
        docker cp "$sum_hardware_evidence/dspark-sum-sfpu-hardware.json" "$test_id:/experiment-scripts/ci/dspark-sum-sfpu-hardware.json"
    fi
fi
if [ "$mask_bits" = 1 ]; then
    docker cp "$mask_evidence/dspark-mask-bits.json" "$test_id:/experiment-scripts/ci/dspark-mask-bits.json"
    if [[ "$request_screen" = 1 || "$timed_requests" = 1 ]]; then
        docker cp "$mask_hardware_evidence/dspark-mask-bits-hardware.json" "$test_id:/experiment-scripts/ci/dspark-mask-bits-hardware.json"
    fi
fi
if [[ "$request_screen" = 1 || "$timed_requests" = 1 ]]; then
    docker cp "$request_evidence/dspark-score-sfpu-hardware.json" "$test_id:/experiment-scripts/ci/dspark-score-sfpu-hardware.json"
fi
if [ "$timed_requests" = 1 ]; then
    docker cp "$request_evidence/dspark-sfpu-request-screen.json" "$test_id:/experiment-scripts/ci/dspark-sfpu-request-screen.json"
fi
if [ "$score_bitwise" = 1 ]; then
    docker cp "$score_evidence/dspark-score-bitwise.json" "$test_id:/experiment-scripts/ci/dspark-score-bitwise.json"
fi
if [[ "$mode" = request || "$mode" = request-variants || "$mode" = request-native-attention || "$mode" = request-combined || "$mode" = request-target-attention || "$mode" = request-norm-scatter || "$mode" = request-verifier-profile ]]; then
    docker cp speculative-decoding "$test_id:/speculative-decoding"
fi
docker cp optimisation/sim/sdpa-graft-registration.patch "$test_id:/tmp/ccl-graft-registration.patch"
docker start -a "$test_id" | tee "$output/dspark-console.log"
test "$(docker inspect --format '{{.State.ExitCode}}' "$test_id")" = 0
