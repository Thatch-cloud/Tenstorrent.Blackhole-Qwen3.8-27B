"""QWEN_FAST_TRACED_PUBLISH: cut prepare_history's own op count in steady state,
instead of the device-trace capture its name suggests - see the module docstring
below for why true trace capture was not safe to build, and dflash_device.
prepare_publication / draft_kv_history.DraftKVHistory.prepare for the algebraic
identity this actually applies (fused_steady_state=True there).

WHY NOT A CAPTURED DEVICE TRACE. v27 (run 35584039408) measured prepare_history at
9-11 ms/user with QWEN_FAST_PIPELINED_PUBLISH's fence-merging already applied and
showing no further gain - the cost is ~75 small enqueued ops per user (project_
features ~10-15, prepare_publication ~5, DraftKVHistory.prepare 5 layers x 2 heads x
~6), each paying host dispatch, not device fences. The obvious fix - capture that
whole sequence into one per-user replay, the way dflash_proposal_trace.
PreparedDFlashProposal captures the draft proposal - does not carry over, because
prepare_history's shape depends on `prefix` (the accepted-token count this round,
1..block_rows-1), not just on a bucketed geometry:

  - project_features's every op (slice/pad/matmul/typecast/rms_norm/the final
    concat) operates on a `count`-wide (== prefix) tensor; its OUTPUT shape is
    (1, 1, prefix, 5120), literally different for every distinct prefix value.
  - DraftKVHistory.prepare's per-layer/per-head slice offsets
    (history_rows + prefix - rows) and slice widths (prefix) are prefix-dependent
    too, even once rows saturates at 2048 (see fused_steady_state's own comment).

Every existing captured trace in this codebase (PreparedDFlashProposal,
PreparedPackedDFlashProposal, the packed VERIFY block) bakes EVERY op's shape and
slice offset into the trace at capture time, varying only the DATA VALUES loaded
into fixed placeholder buffers before each replay (dflash_proposal_trace.py's own
update()/_update() pattern) - none of them vary an op's own offset or extent
per round. I found no precedent in this codebase for a captured op whose slice
offset or extent is itself read from a per-round value rather than baked in, and no
way to confirm ttnn's tracing supports that without hardware.

Capturing one trace per DISTINCT prefix value (up to block_rows - 1, so up to 15 for
T16) per history_rows bucket per user would sidestep the shape problem, but its own
placeholder buffers - a fresh ~21 MB (1, 1, 2048, 5120) bf16 history tensor, plus
five ~1 MB (1, 4, 2048, 128) K/V placeholders per layer, per distinct prefix, per
user - multiply out to hundreds of MB to low GB of extra DRAM per user; the packed-
proposal trace_region note already on record (run 35565478581: 'the fourth user's
admission ran out of memory with 716 MB largest free') makes this look impractical
without evidence a real run would fit it. I did not build it.

What IS safe and already fixed: in the steady state prepare_history actually
runs in (history_rows == 2048, permanent once reached), the concat-then-slice-then-
pad sequence in both prepare_publication and DraftKVHistory.prepare simplifies
algebraically to a plain drop-the-oldest-rows slice plus one concat, with no padding
step at all - see fused_steady_state's own comment for the proof. That is a genuine,
provable reduction in op count (roughly 75 -> 55-60 ops/user, not 75 -> a handful),
gated here the same way as every other flag in this family.
"""

import os


FLAG = 'QWEN_FAST_TRACED_PUBLISH'


def traced_publish_enabled(environ=None):
    """QWEN_FAST_TRACED_PUBLISH=1. Read at each round rather than at import, so the
    switch a test flips is the one the round sees."""
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % FLAG)
    return value == '1'


def install_publish_options(drafter, *, merge_release, fused_steady_state):
    """Rebind drafter.prepare_publication (an instance attribute, shadowing the
    class method) to call the class method with BOTH merge_release
    (dflash_pipelined_publish.QWEN_FAST_PIPELINED_PUBLISH) and fused_steady_state
    (this module's QWEN_FAST_TRACED_PUBLISH) baked in together - a single installer
    rather than two independently-stacked shims, so neither flag's effect can be
    silently dropped by the other overwriting drafter.prepare_publication after it.
    Returns a restore callable. drafter.prepare_publication must not already be
    overridden."""
    if type(merge_release) is not bool or type(fused_steady_state) is not bool:
        raise ValueError('Explicit merge_release and fused_steady_state selection required')
    if 'prepare_publication' in drafter.__dict__:
        raise ValueError('drafter.prepare_publication is already overridden')
    cls = type(drafter)

    def combined(features, prefix, *, position):
        return cls.prepare_publication(drafter, features, prefix, position=position,
            merge_release=merge_release, fused_steady_state=fused_steady_state)

    drafter.prepare_publication = combined

    def restore():
        del drafter.prepare_publication

    return restore
