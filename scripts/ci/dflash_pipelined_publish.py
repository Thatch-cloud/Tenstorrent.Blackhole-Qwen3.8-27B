"""QWEN_FAST_PIPELINED_PUBLISH: install dflash_device.DFlashDevice.prepare_publication's
merge_release=True onto a request's drafter for the span of one commit, without touching
dflash_request_runtime.py (unpinned, but this keeps it untouched exactly as
runtime_binary_override.py hooks serving_runtime.attach_combined_runtime from outside).

Why merge_release is safe: prepare_publication and project_features each enqueue device
work then synchronize_device(mesh) before releasing their own temporaries - a global fence
over the SAME shared mesh every concurrent request's device shares (DFlashDevice.mesh is
model.mesh_device). With a committed K/V cache, prepare_publication ALSO calls
kv_history.prepare(), which does its own synchronize_device(mesh) after everything already
enqueued in this call (project_features' work included, since it runs earlier in submission
order on the one queue) - so prepare_publication's own trailing fence is redundant in that
case. merge_release folds project_features' temporaries into prepare_publication's own
release scope (skipping project_features' own fence too) and skips the redundant trailing
one, leaving exactly ONE fence - kv_history.prepare()'s - to cover the whole call, instead
of three. Nothing here changes what is computed, read back to host, or when memory becomes
unsafe to free: every enqueued op stays in the same in-order device queue either way, and
no temporary is released before a fence that has already run over it.

Uncertified: host construction and the mocked device-call sequence only
(test_dflash_device_publish.py, test_dflash_pipelined_publish.py). Nothing here has run on
a device.
"""

import os


PUBLISH_FLAG = 'QWEN_FAST_PIPELINED_PUBLISH'


def pipelined_publish_enabled(environ=None):
    """QWEN_FAST_PIPELINED_PUBLISH=1. Read at each round rather than at import, so the
    switch a test flips is the one the round sees."""
    value = (os.environ if environ is None else environ).get(PUBLISH_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % PUBLISH_FLAG)
    return value == '1'


def install_merge_release(drafter):
    """Rebind drafter.prepare_publication (an instance attribute, shadowing the class
    method) to call the class method with merge_release=True baked in. Returns a
    restore callable that removes the override, leaving the class method looked up
    normally again - drafter.prepare_publication must not already be overridden."""
    if 'prepare_publication' in drafter.__dict__:
        raise ValueError('drafter.prepare_publication is already overridden')
    cls = type(drafter)

    def merged(features, prefix, *, position):
        return cls.prepare_publication(drafter, features, prefix, position=position, merge_release=True)

    drafter.prepare_publication = merged

    def restore():
        del drafter.prepare_publication

    return restore
