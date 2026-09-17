"""Complete DFlash2 request on the promoted T16 target, without DSpark-only hooks."""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from cumulative_t16_scope import scoped_cumulative_t16, validate_request
from cumulative_register_scope import scoped_register_epilogue
from cumulative_fusion_validation import validate_fusion_policy
from gdn_direct_window_gate import qualify as qualify_windows, REPORT_SHA256 as WINDOW_SHA256
from mlp_down_grid_gate import qualify as qualify_down, REPORT_SHA256 as DOWN_SHA256
from shared_qk_norm_scatter_gate import qualify as qualify_norm, REPORT_SHA256 as NORM_SHA256
from shared_qk_norm_scatter import build as scatter_build


def measure_combined_dflash(operations, model, sampler, prompt, pages, helpers, *, directory,
                            runtime_root, **options):
    from full_dflash_request import measure_dflash_request
    from fused_t16_scope import FusedT16Arm
    from gdn_shared_qk_scope import scoped_shared_qk
    import gdn_shared_qk_scope
    import gdn_shared_qk_gate
    from models.tt_transformers.tt.ccl import tt_all_reduce

    directory = Path(directory)
    if len(prompt) != 4096 or options.get('max_new_tokens') != 256:
        raise ValueError('Initial combined drafter comparison requires CTX4096 and 256 output budget')
    windows = qualify_windows(directory, directory / 'gdn-direct-window-evidence')
    down = qualify_down(directory / 'mlp-down-grid-evidence', directory, runtime_root)
    norm = qualify_norm(directory / 'shared-qk-norm-scatter.json', directory, runtime_root)
    builds = []

    def build(*arguments, **keywords):
        result = scatter_build(*arguments, **keywords)
        builds.append(len(result))
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(gdn_shared_qk_scope, 'build', build))
        stack.enter_context(patch.object(gdn_shared_qk_gate, 'qualify', lambda *args: norm))
        target = stack.enter_context(scoped_cumulative_t16(windows, None, directory,
            down_admission=down, drafter='dflash2'))
        register = stack.enter_context(scoped_register_epilogue(directory, runtime_root=runtime_root))
        shared = stack.enter_context(scoped_shared_qk(operations, norm))
        fusion = FusedT16Arm(operations, model, tt_all_reduce)
        stack.enter_context(fusion.install())
        result = measure_dflash_request(operations, model, sampler, prompt, pages, helpers,
            block_rows=16, proposal_capture=True, commit_only_gdn=True, fused_convolution=True,
            cache_history=True, target_attention_t16=True, **options)
    result['gdn_shared_qk'] = shared
    result['fused_t16_mlp'] = fusion.audit
    result['register_epilogue'] = dict(register, register_resident=True)
    result['gdn_direct_window'] = dict(direct=True, hits=target['direct']['hits'],
        report_sha256=WINDOW_SHA256, restored=target['direct']['restored'])
    result['mlp_down_grid'] = dict(wider_down=True, hits=target['down']['hits'],
        report_sha256=DOWN_SHA256, restored=target['down']['restored'])
    result['norm_reader'] = dict(policy='scatter', builds=len(builds), report_sha256=NORM_SHA256,
                                restored=True)
    validate_request(result, target, drafter='dflash2')
    validate_fusion_policy(result, 'register')
    if (not builds or any(count != 3 for count in builds)
            or len(builds) != len(shared.get('loads', []))
            or shared.get('restored') is not True or shared.get('released') is not True
            or shared.get('admission', {}).get('report_sha256') != NORM_SHA256):
        raise ValueError('Every shared-Q/K target program must use qualified scatter normalization')
    if (qualify_windows(directory, directory / 'gdn-direct-window-evidence') != windows
            or qualify_down(directory / 'mlp-down-grid-evidence', directory, runtime_root) != down
            or qualify_norm(directory / 'shared-qk-norm-scatter.json', directory, runtime_root) != norm):
        raise ValueError('Target source admission changed during DFlash2 execution')
    return result
