"""Explicit hardware-only 64K combined-request entry; no folded target attention."""

import os
import json
from contextlib import nullcontext
from pathlib import Path
import sys

from dspark_64k_scope import runtime_scope


def validate_options(options):
    forbidden = ('target_attention_variants', 'combined_variants', 'mlp_down',
        'score_layout', 'native_attention_variants', 'profile_drafter',
        'profile_verifier', 'request_variants', 'fused_t16_mlp',
        'history_profile', 'banked_proposal', 'native_slot_gdn', 'mlp_equal_footprint')
    if (not options.request or not options.captured_publication
            or not options.norm_scatter_variants or options.max_new_tokens != 256
            or any(getattr(options, name) for name in forbidden)):
        raise ValueError('64K requires the isolated captured native-attention/scatter request with 256 output tokens')


def run(main):
    if (os.environ.get('QWEN_DSPARK_64K_TRIAL') != '1'
            or os.environ.get('QWEN_HARDWARE_TESTS') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('QWEN_LADDER_BACKEND') != 'hardware'
            or os.environ.get('TT_METAL_SIMULATOR')
            or '--captured-publication' not in sys.argv
            or '--norm-scatter-variants' not in sys.argv):
        raise ValueError('Allocated 64K captured native-attention/scatter comparison required')
    forbidden = ('--target-attention-variants', '--combined-variants', '--mlp-down',
        '--score-layout', '--native-attention-variants', '--profile-drafter', '--profile-verifier', '--request-variants',
        '--fused-t16-mlp', '--history-profile', '--banked-proposal', '--native-slot-gdn', '--mlp-equal-footprint')
    if any(option in sys.argv for option in forbidden):
        raise ValueError('Unqualified 64K target-attention or alternative comparison selected')
    if '--preflight' in sys.argv:
        return main()
    directory = Path(__file__).parent
    phase_probe = os.environ.get('QWEN_DSPARK_PHASE_PROBE', '0')
    if phase_probe not in ('0', '1'):
        raise ValueError('Explicit zero or one phase probe selection required')
    probe_scope = nullcontext()
    hardware_candidate_scope = nullcontext()
    score_sfpu = os.environ.get('QWEN_DSPARK_SCORE_SFPU', '0')
    if score_sfpu not in ('0', '1') or score_sfpu == '1' and phase_probe != '1':
        raise ValueError('SFPU candidate requires bounded hardware diagnostic mode')
    if score_sfpu == '1':
        from dspark_score_sfpu_hardware import hardware_scope
        hardware_candidate_scope = hardware_scope(directory)
    sum_sfpu = os.environ.get('QWEN_DSPARK_SUM_SFPU', '0')
    if sum_sfpu not in ('0', '1') or sum_sfpu == '1' and score_sfpu != '1':
        raise ValueError('Sum update requires qualified SFPU hardware mode')
    if sum_sfpu == '1':
        from dspark_sum_request_gate import qualify
        from dspark_sum_sfpu_hardware import hardware_scope
        qualify(directory, directory / 'dspark-sum-sfpu-hardware.json')
        hardware_candidate_scope = hardware_scope(directory)
    candidate_scope = nullcontext()
    score_bitwise = os.environ.get('QWEN_DSPARK_SCORE_BITWISE', '0')
    if score_bitwise not in ('0', '1') or score_bitwise == '1' and phase_probe != '1':
        raise ValueError('Score candidate requires bounded hardware diagnostic mode')
    if score_bitwise == '1':
        from dspark_score_candidate_gate import qualify
        from dspark_score_bitwise import bitwise_infinity_checks
        qualify(directory, directory / 'dspark-score-bitwise.json')
        candidate_scope = bitwise_infinity_checks()
    mask_bits = os.environ.get('QWEN_DSPARK_MASK_BITS', '0')
    if mask_bits not in ('0', '1') or mask_bits == '1' and (sum_sfpu != '1' or score_bitwise != '0'):
        raise ValueError('Mask candidate requires isolated qualified sum-update runtime')
    if mask_bits == '1':
        from dspark_mask_request_gate import qualify
        from dspark_mask_bits import mask_scope
        qualify(directory, directory / 'dspark-mask-bits-hardware.json')
        candidate_scope = mask_scope()
    request_screen = os.environ.get('QWEN_DSPARK_SFPU_REQUEST_SCREEN', '0')
    timed_requests = os.environ.get('QWEN_DSPARK_SFPU_TIMED', '0')
    target_scope = nullcontext()
    target_request = os.environ.get('QWEN_TARGET_T16_64K_REQUEST', '0')
    direct_staging = os.environ.get('QWEN_DSPARK_DIRECT_FP32_STAGE', '0')
    direct_candidate_scope = nullcontext()
    normalization_candidate_scope = nullcontext()
    normalization_staging = os.environ.get('QWEN_DSPARK_NORMALIZATION_DIRECT_STAGE', '0')
    if (normalization_staging not in ('0', '1') or normalization_staging == '1'
            and (direct_staging != '1' or request_screen != '1' or timed_requests != '0')):
        raise ValueError('Normalization staging requires its isolated combined correctness screen')
    if normalization_staging == '1':
        from dspark_normalization_direct_stage import normalization_stage_scope
        normalization_candidate_scope = normalization_stage_scope()
    if (direct_staging not in ('0', '1') or direct_staging == '1'
            and (target_request != '1' or (request_screen, timed_requests) not in (('1', '0'), ('0', '1')))):
        raise ValueError('Direct staging requires its isolated folded-T16 audit or qualified timing')
    if direct_staging == '1':
        from dspark_direct_fp32_stage import staging_scope
        direct_candidate_scope = staging_scope()
    if (target_request not in ('0', '1') or target_request == '1'
            and ((request_screen, timed_requests) not in (('1', '0'), ('0', '1')) or mask_bits != '1')):
        raise ValueError('Folded 64K verifier requires its isolated audited or qualified timed runtime')
    if target_request == '1':
        from target_t16_64k_request import request_scope
        target_scope = request_scope(directory)
    if (timed_requests not in ('0', '1') or timed_requests == '1'
            and (score_sfpu != '1' or request_screen != '0')):
        raise ValueError('Timed requests require isolated qualified SFPU candidate')
    if request_screen not in ('0', '1') or request_screen == '1' and score_sfpu != '1':
        raise ValueError('Request screen requires qualified SFPU candidate')
    if timed_requests == '1':
        from dspark_sfpu_timed_requests import timed_scope
        if sum_sfpu == '1':
            from dspark_sum_timed_requests import timed_scope
        if mask_bits == '1':
            from dspark_mask_timed_requests import timed_scope
        if target_request == '1':
            from target_t16_64k_timed import timed_scope
        if direct_staging == '1':
            from dspark_direct_fp32_timed import timed_scope
        probe_scope = timed_scope(directory)
    elif request_screen == '1':
        from dspark_sfpu_request_screen import screen_scope
        if target_request == '1':
            from target_t16_64k_screen import screen_scope
        if direct_staging == '1':
            from dspark_direct_fp32_screen import screen_scope
        if normalization_staging == '1':
            from dspark_normalization_screen import screen_scope
        probe_scope = screen_scope(directory)
    elif phase_probe == '1':
        from dspark_proposal_phase_profile import stop_after_prepared_probe

        def checkpoint(report):
            report['candidate'] = ('sfpu-score-centering' if score_sfpu == '1'
                else 'bitwise-infinity-checks' if score_bitwise == '1' else 'baseline')
            destination = Path('/experiment/results/dspark-proposal-phases.json')
            temporary = destination.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2) + '\n')
            temporary.replace(destination)
            print(json.dumps(dict(stage='proposal-phase-probe', phase=report['phase'],
                completed_replays=report['completed_replays'])), flush=True)

        probe_scope = stop_after_prepared_probe(checkpoint)
    with normalization_candidate_scope, direct_candidate_scope, hardware_candidate_scope, runtime_scope(directory, directory / 'dspark-ladder-hardware-65536.json',
            context=65536, output_tokens=256, factory_root=os.environ['TT_METAL_HOME'],
            build_path='/experiment/results/dspark-64k-hardware-build.json'):
        with candidate_scope, probe_scope, target_scope:
            return main()
