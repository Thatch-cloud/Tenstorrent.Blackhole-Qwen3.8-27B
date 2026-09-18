"""Complete DFlash2 request on the promoted T16 target, without DSpark-only hooks."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from cumulative_t16_scope import scoped_cumulative_t16, validate_request
from cumulative_register_scope import scoped_register_epilogue
from cumulative_fusion_validation import validate_fusion_policy
from gdn_direct_window_gate import qualify as qualify_windows, REPORT_SHA256 as WINDOW_SHA256
from mlp_down_grid_gate import qualify as qualify_down, REPORT_SHA256 as DOWN_SHA256
from shared_qk_norm_scatter_gate import qualify as qualify_norm, REPORT_SHA256 as NORM_SHA256
from shared_qk_norm_scatter import build as scatter_build


@contextmanager
def combined_runtime(operations, model, *, directory, runtime_root,
                     native_attention_evidence=None, block_stream=None, kv_publication_evidence=None):
    from fused_t16_scope import FusedT16Arm
    from gdn_shared_qk_scope import scoped_shared_qk
    import gdn_shared_qk_scope
    import gdn_shared_qk_gate
    from models.tt_transformers.tt.ccl import tt_all_reduce

    directory = Path(directory)
    windows = qualify_windows(directory, directory / 'gdn-direct-window-evidence')
    down = qualify_down(directory / 'mlp-down-grid-evidence', directory, runtime_root)
    norm = qualify_norm(directory / 'shared-qk-norm-scatter.json', directory, runtime_root)
    builds = []

    def build(*arguments, **keywords):
        result = scatter_build(*arguments, **keywords)
        builds.append(len(result))
        return result

    with ExitStack() as stack:
        publication = register = stream_audit = None
        if kv_publication_evidence is not None:
            from draft_kv_slide_scope import scoped_publication

            publication = stack.enter_context(scoped_publication(directory, kv_publication_evidence))
        if native_attention_evidence is not None:
            from dflash_t16_native_scope import scoped_native_t16

            stack.enter_context(scoped_native_t16(native_attention_evidence, directory, runtime_root))
        stack.enter_context(patch.object(gdn_shared_qk_scope, 'build', build))
        stack.enter_context(patch.object(gdn_shared_qk_gate, 'qualify', lambda *args: norm))
        target = stack.enter_context(scoped_cumulative_t16(windows, None, directory,
            down_admission=down, drafter='dflash2'))
        if block_stream is None:
            register = stack.enter_context(scoped_register_epilogue(directory, runtime_root=runtime_root))
        else:
            from mlp_block_stream_runtime import scoped_block_stream

            stream_audit = stack.enter_context(scoped_block_stream(directory, block_stream['evidence'],
                runtime_root=runtime_root, operations=operations,
                weights=[layer.feed_forward.weights.w_gate_up for layer in model.layers],
                streams=block_stream['streams'],
                **{name: block_stream[name] for name in ('pipeline_evidence', 'progressive_evidence')
                    if name in block_stream}))
        shared = stack.enter_context(scoped_shared_qk(operations, norm))
        fusion = FusedT16Arm(operations, model, tt_all_reduce)
        stack.enter_context(fusion.install())
        yield dict(publication=publication, register=register, stream_audit=stream_audit,
            target=target, shared=shared, fusion=fusion, builds=builds,
            windows=windows, down=down, norm=norm)
    if (qualify_windows(directory, directory / 'gdn-direct-window-evidence') != windows
            or qualify_down(directory / 'mlp-down-grid-evidence', directory, runtime_root) != down
            or qualify_norm(directory / 'shared-qk-norm-scatter.json', directory, runtime_root) != norm):
        raise ValueError('Target source admission changed during DFlash2 execution')


def measure_combined_dflash(operations, model, sampler, prompt, pages, helpers, *, directory,
                            runtime_root, native_attention_evidence=None, block_stream=None,
                            kv_publication_evidence=None, **options):
    from full_dflash_request import measure_dflash_request

    if 'native_proposal_attention' in options:
        raise ValueError('Select native T16 proposals through explicit simulator evidence, not a bare flag')
    if len(prompt) != 4096 or options.get('max_new_tokens') != 256:
        raise ValueError('Initial combined drafter comparison requires CTX4096 and 256 output budget')
    with combined_runtime(operations, model, directory=directory, runtime_root=runtime_root,
            native_attention_evidence=native_attention_evidence, block_stream=block_stream,
            kv_publication_evidence=kv_publication_evidence) as active:
        result = measure_dflash_request(operations, model, sampler, prompt, pages, helpers,
            block_rows=16, proposal_capture=True, commit_only_gdn=True, fused_convolution=True,
            cache_history=True, target_attention_t16=True,
            **(dict(native_proposal_attention=True) if native_attention_evidence is not None else {}), **options)
    publication, register, stream_audit = (active[key] for key in ('publication', 'register', 'stream_audit'))
    target, shared, fusion, builds = (active[key] for key in ('target', 'shared', 'fusion', 'builds'))
    if kv_publication_evidence is not None:
        result['draft_kv_slide'] = publication
    result['gdn_shared_qk'] = shared
    result['fused_t16_mlp'] = fusion.audit
    if block_stream is None:
        result['register_epilogue'] = dict(register, register_resident=True)
    else:
        from mlp_register_epilogue_gate import REPORT_SHA256 as REGISTER_SHA256

        result['block_stream'] = stream_audit
        result['fused_t16_mlp']['extra_weight_allocations'] = 64
        result['register_epilogue'] = dict(register_resident=True, report_sha256=REGISTER_SHA256,
            constructions=stream_audit['constructions'], calls=stream_audit['calls'], restored=stream_audit['restored'])
    result['gdn_direct_window'] = dict(direct=True, hits=target['direct']['hits'],
        report_sha256=WINDOW_SHA256, restored=target['direct']['restored'])
    result['mlp_down_grid'] = dict(wider_down=True, hits=target['down']['hits'],
        report_sha256=DOWN_SHA256, restored=target['down']['restored'])
    result['norm_reader'] = dict(policy='scatter', builds=len(builds), report_sha256=NORM_SHA256,
                                restored=True)
    validate_request(result, target, drafter='dflash2')
    if block_stream is None:
        validate_fusion_policy(result, 'register')
    else:
        from mlp_block_stream_request import validate_request as validate_stream_request

        validate_stream_request(result)
    if (not builds or any(count != 3 for count in builds)
            or len(builds) != len(shared.get('loads', []))
            or shared.get('restored') is not True or shared.get('released') is not True
            or shared.get('admission', {}).get('report_sha256') != NORM_SHA256):
        raise ValueError('Every shared-Q/K target program must use qualified scatter normalization')
    return result
