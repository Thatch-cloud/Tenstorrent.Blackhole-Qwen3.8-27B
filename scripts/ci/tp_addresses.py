"""Per-chip buffer addresses at any served width, and the one seam that puts them where the pair's code expects them.

gdn_multitoken_conv.addresses and release_owned are the fast path's address helpers, imported by name into some fifty
modules (the packed verifier, the cache owner, the buffer pool, every GDN state builder, the drafter). They require
exactly two chips - `if len(shards) != 2: raise ValueError('Both chips required')` - and gdn_multitoken_conv.py is one
of the 42 sources the TP2 evidence pins by sha256, so the four-card port cannot edit it (test_tp2_pins). Nor can each
importer be redirected one by one without moving every one of them onto the image's copy lists.

So the four-card process rebinds the two names once, at startup, before it attaches anything (install(), called by
serving_startup.start at QWEN_FAST_TP=4 only): the pinned module's own globals, and every module that already holds
the pinned function object, are pointed at the functions below, which count the chips from QWEN_FAST_TP. Modules that
import the names afterwards read the rebound attributes. The pair never installs: its modules keep the pinned function
objects, and these functions are call-for-call the pinned ones at two chips.

Stdlib only, except that install() imports the pinned modules it rebinds (gdn_multitoken_conv needs the fast
path's gdn_multitoken) and their twins.
"""

import sys

import tp_shapes


def addresses(operations, tensor):
    """gdn_multitoken_conv.addresses at the width this process serves at: one buffer address per chip, in chip order."""
    shards = operations.get_device_tensors(tensor)
    if len(shards) != tp_shapes.chip_count():
        raise ValueError('%s chips required' % tp_shapes.all_chips())
    return tuple(shard.buffer_address() for shard in shards)


def release_owned(operations, tensors):
    """gdn_multitoken_conv.release_owned: deallocate each distinct tensor (by its per-chip addresses) once."""
    unique = {addresses(operations, tensor): tensor for tensor in tensors}
    for tensor in unique.values():
        operations.deallocate(tensor)


# (module, attribute, twin module, twin attribute): the pinned or literal-carrying helpers the four-card process
# replaces. The first two are this module's own; the rest are the sibling twins (each written call for call after the
# pinned body, with the widths from tp_shapes).
TWINS = (
    ('gdn_multitoken_conv', 'addresses', 'tp_addresses', 'addresses'),
    ('gdn_multitoken_conv', 'release_owned', 'tp_addresses', 'release_owned'),
    ('gdn_multitoken_conv', 'validate_projected', 'gdn_multitoken_conv_tp', 'validate_projected'),
    ('gdn_multitoken_conv', 'restore_prefix', 'gdn_multitoken_conv_tp', 'restore_prefix'),
    ('gdn_multitoken_conv', 'run_projected', 'gdn_multitoken_conv_tp', 'run_projected'),
    ('gdn_multitoken', 'execute', 'gdn_multitoken_tp', 'execute'),
    ('gdn_multitoken', 'validate_geometry', 'gdn_multitoken_tp', 'validate_geometry'),
    ('attention_parallel', 'execute', 'attention_parallel_tp', 'execute'),
    ('attention_replay', 'ReplayAttentionReader', 'attention_replay_tp', 'ReplayAttentionReader'),
    ('attention_mask_replay', 'prepare', 'attention_mask_replay_tp', 'prepare'),
    ('attention_mask_replay', 'mask_position', 'attention_mask_replay_tp', 'mask_position'),
    ('attention_fold_dma', 'device_layout_dma', 'attention_fold_dma_tp', 'device_layout_dma'),
    ('attention_fold_dma', 'source_row', 'attention_fold_dma_tp', 'source_row'),
    ('ordered_cache', 'validate_shapes', 'ordered_cache_tp', 'validate_shapes'),
    ('ordered_cache', 'update', 'ordered_cache_tp', 'update'),
    ('draft_attention', 'validate_attention', 'draft_attention_tp', 'validate_attention'),
    ('draft_attention', 'draft_sdpa', 'draft_attention_tp', 'draft_sdpa'),
    ('gdn_records', 'retain_checkpoint_histories', 'gdn_records_tp', 'retain_checkpoint_histories'),
    # The packed block's split and merge as one DMA launch each (tp4/vglue, QWEN_FAST_TP4_GDN_GLUE): the twin subclasses the pinned
    # class and delegates to it with the flag off. model_batch imports the class lazily, so it gets the twin.
    ('gdn_device_loop_state', 'DeviceLoopState', 'gdn_device_loop_state_tp', 'DeviceLoopState'),
    ('gdn_commit_dma', 'validate_shapes', 'gdn_commit_dma_tp', 'validate_shapes'),
    ('gdn_commit_dma', 'prepare', 'gdn_commit_dma_tp', 'prepare'),
    ('gdn_commit_dma', 'publish', 'gdn_commit_dma_tp', 'publish'),
    # The two sources the direct-window attach hashes (gdn_direct_window_hardware_sources.BATCH_SHA256 and the
    # simulator report's file list) stay the pair's bytes; their four-card widths live in these twins.
    ('gdn_batched_conv', 'run_batched_projected', 'gdn_batched_conv_tp', 'run_batched_projected'),
    ('gdn_conv_windows', 'build_windows', 'gdn_conv_windows_tp', 'build_windows'),
    # The drafter and GDN helpers recorded evidence hashes (dspark / fused-commit / quad-draft reports, test_tp2_pins):
    # their pair bytes stay and these twins carry the four-card widths.
    ('draft_convolution', 'grouped_causal_convolution', 'draft_convolution_tp', 'grouped_causal_convolution'),
    ('draft_convolution_fused', 'fused_convolution', 'draft_convolution_fused_tp', 'fused_convolution'),
    ('draft_head_layout', 'split_projected_heads', 'draft_head_layout_tp', 'split_projected_heads'),
    ('draft_head_layout', 'concatenate_query_heads', 'draft_head_layout_tp', 'concatenate_query_heads'),
    ('draft_kv_projection', 'project_key_value', 'draft_kv_projection_tp', 'project_key_value'),
    ('draft_mlp', 'split_mlp_weights', 'draft_mlp_tp', 'split_mlp_weights'),
    ('draft_shared_head', 'candidate_chunks', 'draft_shared_head_tp', 'candidate_chunks'),
    ('draft_shared_head', 'shared_head_candidates', 'draft_shared_head_tp', 'shared_head_candidates'),
    ('draft_shared_head', 'local_head_candidates', 'draft_shared_head_tp', 'local_head_candidates'),
    ('draft_shared_head', 'merge_chunk_candidates', 'draft_shared_head_tp', 'merge_chunk_candidates'),
    ('feature_collective', 'reduce_projection', 'feature_collective_tp', 'reduce_projection'),
    ('feature_collective', 'gather_add_projection', 'feature_collective_tp', 'gather_add_projection'),
    ('pair_row_exact', 'note', 'pair_row_exact_tp', 'note'),
    ('pair_row_exact', 'validate_fold', 'pair_row_exact_tp', 'validate_fold'),
    ('pair_row_exact', 'fold_query', 'pair_row_exact_tp', 'fold_query'),
    ('pair_row_exact', 'fold_keys', 'pair_row_exact_tp', 'fold_keys'),
    ('pair_row_exact', 'unfold_output', 'pair_row_exact_tp', 'unfold_output'),
    ('attention_head_fold', 'fold_query', 'attention_head_fold_tp', 'fold_query'),
    ('attention_head_fold', 'unfold_output', 'attention_head_fold_tp', 'unfold_output'),
    ('attention_head_fold', 'causal_mask', 'attention_head_fold_tp', 'causal_mask'),
    ('attention_head_fold', 'device_layout', 'attention_head_fold_tp', 'device_layout'),
    ('gdn_conv_prefix_copy', 'validate_prefix', 'gdn_conv_prefix_copy_tp', 'validate_prefix'),
    ('gdn_conv_prefix_copy', 'copy_prefix', 'gdn_conv_prefix_copy_tp', 'copy_prefix'),
    ('gdn_working_state', 'WorkingState', 'gdn_working_state_tp', 'WorkingState'),
    ('mtp_hidden_rows', 'MTPHiddenRows', 'mtp_hidden_rows_tp', 'MTPHiddenRows'),
    # The attach's pair-only scopes: the sampler's gather links at the ring's proven count, and the direct-window and
    # K/V-publication experiments, which are off at four cards (attach_scopes_tp).
    ('sampling_link_policy', 'sampler_links', 'attach_scopes_tp', 'sampler_links'),
    ('gdn_direct_window_scope', 'scoped_direct_windows', 'attach_scopes_tp', 'scoped_direct_windows'),
    ('draft_kv_slide_scope', 'scoped_publication', 'attach_scopes_tp', 'scoped_publication'),
)

# (module, twin module): whole modules whose lazy `import name` sites must reach the four-card twin. The pinned original
# is imported first (the twin borrows its geometry-free helpers from it), then sys.modules[name] and every module global
# that IS the original module object are pointed at the twin.
MODULE_TWINS = (
    ('extent_attention_replay', 'extent_attention_replay_tp'),
    # The four-user draft pass (QWEN_FAST_QUAD_DRAFT, off in the four-card profiles that do not name it): the coordinator imports
    # quad_draft lazily, so at four cards it gets the twin; the pair keeps the pinned module and its evidence.
    ('quad_draft', 'quad_draft_tp'),
    # The fused commit (QWEN_FAST_FUSED_COMMIT, _INPLACE, _LIVE_BANKS, _AUDIT): packed_verifier, serving_packed_step, verify_prestage and
    # dflash_proposal_trace import fused_commit lazily, so at four cards they get the twin (the pair's slide scope, two-chip program,
    # four-head banks and 16-worker layout stay in the pinned module and its evidence).
    ('fused_commit', 'fused_commit_tp'),
)

# What install() changed: (namespace, name, the pinned object), so a test can put the pair's functions back.
_REBOUND = []
_ALIASED = []


def install(environ=None):
    """Rebind the pinned helpers to the chip-count-generic twins in every loaded module. -> how many names were
    rebound (0 when already installed). Refused at the pair, where nothing may change.

    A module holds a pinned function if its global of the same name IS the pinned function object (`from m import
    name`, or the pinned module's own attribute): both the name and the identity must match, so a same-named function
    of another module, and an alias under another name, are left alone."""
    if tp_shapes.chip_count(environ) == tp_shapes.PAIR:
        raise ValueError('The pinned two-chip helpers stay in place at the pair')
    import importlib

    swaps = []
    for module_name, name, twin_module, twin_name in TWINS:
        old = getattr(importlib.import_module(module_name), name)
        new = getattr(importlib.import_module(twin_module), twin_name)
        if old is not new:
            swaps.append((name, old, new))
    rebound = 0
    # The name sweep runs first, over every loaded module including the pinned originals the module twins are about to
    # replace in sys.modules: once an original is orphaned the sweep can no longer reach it, and it would keep the pair's
    # helpers (the two-chip addresses) for whoever still holds it.
    if swaps:
        for module in list(sys.modules.values()):
            namespace = getattr(module, '__dict__', None)
            if not isinstance(namespace, dict):
                continue
            for name, old, new in swaps:
                if name in namespace and namespace[name] is old:
                    namespace[name] = new
                    _REBOUND.append((namespace, name, old))
                    rebound += 1
    for original_name, twin_name in MODULE_TWINS:
        original = importlib.import_module(original_name)
        twin = importlib.import_module(twin_name)
        if original is twin or sys.modules.get(original_name) is twin:
            continue
        sys.modules[original_name] = twin
        _ALIASED.append((original_name, original))
        rebound += 1
        for module in list(sys.modules.values()):
            namespace = getattr(module, '__dict__', None)
            if isinstance(namespace, dict) and original_name in namespace and namespace[original_name] is original:
                namespace[original_name] = twin
                _REBOUND.append((namespace, original_name, original))
                rebound += 1
    # The model's tt_all_reduce, wrapped to run a wide verify block one 32-row tile at a time (tile_collective_tp): the
    # four-way ring reduction's order depends on the tile's place in the block, the sequential engine's does not.
    import tile_collective_tp

    for namespace, name, old in tile_collective_tp.install():
        _REBOUND.append((namespace, name, old))
        rebound += 1
    return rebound


def uninstall():
    """Put back every pinned object install() replaced (tests only: the four-card process never goes back)."""
    count = len(_REBOUND) + len(_ALIASED)
    while _ALIASED:
        name, original = _ALIASED.pop()
        sys.modules[name] = original
    while _REBOUND:
        namespace, key, old = _REBOUND.pop()
        namespace[key] = old
    return count
