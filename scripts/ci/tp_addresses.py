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
    ('gdn_commit_dma', 'validate_shapes', 'gdn_commit_dma_tp', 'validate_shapes'),
    ('gdn_commit_dma', 'prepare', 'gdn_commit_dma_tp', 'prepare'),
    ('gdn_commit_dma', 'publish', 'gdn_commit_dma_tp', 'publish'),
)

# (module, twin module): whole modules whose lazy `import name` sites must reach the four-card twin. The pinned original
# is imported first (the twin borrows its geometry-free helpers from it), then sys.modules[name] and every module global
# that IS the original module object are pointed at the twin.
MODULE_TWINS = (
    ('extent_attention_replay', 'extent_attention_replay_tp'),
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
            if isinstance(namespace, dict) and namespace.get(original_name) is original:
                namespace[original_name] = twin
                _REBOUND.append((namespace, original_name, original))
                rebound += 1
    if not swaps:
        return rebound
    for module in list(sys.modules.values()):
        namespace = getattr(module, '__dict__', None)
        if not isinstance(namespace, dict):
            continue
        for name, old, new in swaps:
            if namespace.get(name) is old:
                namespace[name] = new
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
