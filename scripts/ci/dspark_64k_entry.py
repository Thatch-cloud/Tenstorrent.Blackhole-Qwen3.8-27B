"""Explicit hardware-only 64K combined-request entry; no folded target attention."""

import os
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
    with runtime_scope(directory, directory / 'dspark-ladder-hardware-65536.json',
            context=65536, output_tokens=256, factory_root=os.environ['TT_METAL_HOME'],
            build_path='/experiment/results/dspark-64k-hardware-build.json'):
        return main()
