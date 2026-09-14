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
    candidate_scope = nullcontext()
    score_bitwise = os.environ.get('QWEN_DSPARK_SCORE_BITWISE', '0')
    if score_bitwise not in ('0', '1') or score_bitwise == '1' and phase_probe != '1':
        raise ValueError('Score candidate requires bounded hardware diagnostic mode')
    if score_bitwise == '1':
        from dspark_score_candidate_gate import qualify
        from dspark_score_bitwise import bitwise_infinity_checks
        qualify(directory, directory / 'dspark-score-bitwise.json')
        candidate_scope = bitwise_infinity_checks()
    if phase_probe == '1':
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
    with hardware_candidate_scope, runtime_scope(directory, directory / 'dspark-ladder-hardware-65536.json',
            context=65536, output_tokens=256, factory_root=os.environ['TT_METAL_HOME'],
            build_path='/experiment/results/dspark-64k-hardware-build.json'):
        with candidate_scope, probe_scope:
            return main()
