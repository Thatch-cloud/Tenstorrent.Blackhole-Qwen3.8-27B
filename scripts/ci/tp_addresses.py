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

Stdlib only, except that install() imports gdn_multitoken_conv (which needs the fast path's gdn_multitoken).
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


def install(environ=None):
    """Rebind the pinned helpers to the chip-count-generic ones in every loaded module. -> how many names were
    rebound (0 when already installed). Refused at the pair, where nothing may change."""
    if tp_shapes.chip_count(environ) == tp_shapes.PAIR:
        raise ValueError('The pinned two-chip address helpers stay in place at the pair')
    import gdn_multitoken_conv as pinned

    swaps = ((pinned.addresses, addresses), (pinned.release_owned, release_owned))
    if pinned.addresses is addresses and pinned.release_owned is release_owned:
        return 0
    rebound = 0
    for module in list(sys.modules.values()):
        namespace = getattr(module, '__dict__', None)
        if not isinstance(namespace, dict):
            continue
        for name in ('addresses', 'release_owned'):
            for old, new in swaps:
                if getattr(old, '__name__', None) == name and namespace.get(name) is old:
                    namespace[name] = new
                    rebound += 1
    return rebound
