"""QWEN_FAST_TRACED_PUBLISH: cut prepare_history's own op count in steady state,
instead of the device-trace capture its name suggests - see the module docstring
below for why true trace capture was not safe to build, and dflash_device.
DFlashDevice.prepare_publication's own fused_steady_state branch (the target-feature
history buffer) plus install_fused_kv_history below (the K/V cache, re-homed here -
see that function's own docstring for why) for the algebraic identity this applies.

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
    too, even once rows saturates at 2048 (see install_fused_kv_history's own
    comment).

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
step at all - see install_fused_kv_history's own comment for the proof. That is a
genuine, provable reduction in op count (roughly 75 -> 55-60 ops/user, not 75 -> a
handful), gated here the same way as every other flag in this family.

WHY THE K/V FUSION LIVES HERE AND NOT INSIDE DraftKVHistory.prepare. It did, briefly
(commit b05c8af8, image v77) - restructuring prepare()'s own body into fused/general
branches broke draft_kv_slide_adapter.build_prepare's exact-text match of prepare()'s
ORIGINAL six-op (k, v) block (run 35586910004: every arm failed at attach, flag or
not, raising out of frozen_recipe_context.replace_once). And even a byte-matching
prepare() would have been moot: that adapter's exec'd candidate REPLACES
DraftKVHistory.prepare at the CLASS level for the life of the combined-runtime scope
(draft_kv_slide_scope.scoped_publication, QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1) - code
written inside prepare()'s own body never runs there regardless. prepare() itself
(scripts/ci/draft_kv_history.py) is therefore restored byte-identical to
b05c8af8~1 and must stay that way; the fusion is installed as a transient INSTANCE-
level override on one kv_history object's own .prepare, exactly the way
install_publish_options already shadows drafter.prepare_publication - see
install_fused_kv_history below.
"""

import os


FLAG = 'QWEN_FAST_TRACED_PUBLISH'

# Logged under QWEN_FAST_PACKED_AUDIT=1 (dflash_packed_proposal_coordinator.audit_log)
# whenever install_fused_kv_history declines to install rather than guess at
# reproducing another live override's own behavior - see that function's docstring.
FUSION_DECLINED_LINE = '[PACKED-PUBLISH] fusion=declined reason={reason}'


def traced_publish_enabled(environ=None):
    """QWEN_FAST_TRACED_PUBLISH=1. Read at each round rather than at import, so the
    switch a test flips is the one the round sees."""
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % FLAG)
    return value == '1'


def _fused_kv_history_prepare(cache, original, features, prefix, *, position):
    """The steady-state fusion for one DraftKVHistory instance's own .prepare, run in
    place of the class method by the override install_fused_kv_history installs.

    Falls straight through to `original` (the true, untouched bound class method,
    captured before any override existed) whenever rows != 2048 - cache.history_rows
    is still ramping, and correctness there is entirely original's, unchanged.

    In the fused branch (rows == 2048, permanent once reached - DraftKVHistory
    requires history_rows == min(position, 2048)), mirrors dflash_device.DFlashDevice.
    prepare_publication's own fused_steady_state branch (see its comment for the same
    proof), applied per (layer, k/v head) pair instead of to the single target-feature
    history buffer: prepare()'s own general path computes, per (layer, name) -

      historical = active[name][:, :, 0:history_rows, :]
      accepted   = result[name][:, :, 0:prefix, :]
      combined   = concat([historical, accepted], dim=2)            # history_rows+prefix rows
      tail       = combined[:, :, history_rows+prefix-rows:history_rows+prefix, :]
      padded     = pad(tail, ..., 0.0)                               # rows == 2048: pad is a no-op

    tail always keeps ALL of `accepted` (its upper bound is combined's own upper
    bound) and exactly the LAST (rows - prefix) rows of `historical` (its lower bound,
    history_rows+prefix-rows, is expressed directly against active[name] below) - so
    the same padded result is exactly:

      dropped  = active[name][:, :, history_rows+prefix-rows:history_rows, :]
      combined = concat([dropped, accepted], dim=2)                  # == rows == 2048

    cutting slice + concat + slice + pad (4 ops) to slice + concat (2 ops) per
    (layer, head), with no padding step at all. history_rows+prefix-rows is 0 (a
    plain drop-nothing slice) on every round after the one where history_rows first
    reaches 2048, and only nonzero on that one transitional round.

    The validation block below duplicates DraftKVHistory.prepare's own (draft_kv_
    history.py) rather than calling into it, because that method's own lines must
    stay byte-identical (see this module's docstring) and so cannot be factored to
    share it - a drift risk against prepare()'s own guard if that guard ever changes
    without a matching update here, accepted for the same reason the fusion had to
    move out here at all.
    """
    rows = min(2048, cache.history_rows + prefix)
    if rows != 2048:
        return original(features, prefix, position=position)
    if (cache.closed or cache.pending is not None or type(position) is not int or position != cache.position
            or type(prefix) is not int or not 1 <= prefix <= 32 or position + prefix > 262144):
        raise ValueError('One accepted-prefix cache update at the current committed frontier required')
    from types import SimpleNamespace

    from draft_kv_projection import project_key_value

    operations = cache.operations
    with cache.temporaries([features]) as retain:
        inputs, tables = cache.project_inputs(features, prefix, position, retain)
        projected = cache.projection.project(inputs, tables) if cache.projection is not None else None
        for layer, (parameter, active, spare) in enumerate(zip(cache.parameters, cache.active, cache.spare, strict=True)):
            result = projected[layer] if projected is not None else project_key_value(
                operations, inputs, cache.query, tables, retain, parameters=parameter)
            for name in ('k', 'v'):
                dropped = retain(operations.slice(active[name], (0, 0, cache.history_rows + prefix - rows, 0),
                    (1, 4, cache.history_rows, 128)))
                accepted = retain(operations.slice(result[name], (0, 0, 0, 0), (1, 4, prefix, 128)))
                combined = retain(operations.concat([dropped, accepted], dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
                operations.copy(combined, spare[name])
        operations.synchronize_device(cache.mesh)
    cache.pending = SimpleNamespace(position=position, prefix=prefix, rows=rows, status='prepared')
    return cache.pending


def install_fused_kv_history(kv_history):
    """Install a transient, INSTANCE-level override on kv_history.prepare (shadowing
    draft_kv_history.DraftKVHistory.prepare the same way install_publish_options
    shadows drafter.prepare_publication - an attribute on this one object, never the
    class) that runs _fused_kv_history_prepare above in place of the class method,
    for the life of one publish call. Returns a restore callable, or None if nothing
    was installed (kv_history is None, or this declined - see below); restore()
    always removes exactly what THIS call installed, never more.

    Declines outright (returns None, logging FUSION_DECLINED_LINE under
    QWEN_FAST_PACKED_AUDIT=1) rather than installing, whenever draft_kv_slide_scope.
    scoped_publication's own candidate is CURRENTLY the live class-level
    DraftKVHistory.prepare - detected via that candidate's own `_draft_kv_slide`
    marker (draft_kv_slide_scope.py sets it on the function it installs, and checks
    the same marker itself to forbid nested overrides). That candidate already
    replaces prepare()'s entire per-(layer, head) sequence with its own prepare_slide
    transport (draft_kv_slide_adapter.py); this module has no way to confirm without
    hardware that _fused_kv_history_prepare's own slice+concat reproduces
    prepare_slide's transport bit-for-bit, so - per instruction - it leaves that
    candidate's behavior alone rather than guess at matching it.

    Raises ValueError if kv_history.prepare is already instance-overridden by
    something else (the same contract install_publish_options enforces on
    drafter.prepare_publication) - this installer is not meant to stack.
    """
    if kv_history is None:
        return None
    import draft_kv_history

    if not isinstance(kv_history, draft_kv_history.DraftKVHistory):
        return None
    if 'prepare' in vars(kv_history):
        raise ValueError('kv_history.prepare is already overridden')
    if getattr(draft_kv_history.DraftKVHistory.prepare, '_draft_kv_slide', False):
        from dflash_packed_proposal_coordinator import audit_enabled, audit_log

        if audit_enabled():
            audit_log(FUSION_DECLINED_LINE, reason='slide_candidate_live')
        return None
    original = kv_history.prepare

    def fused(features, prefix, *, position):
        return _fused_kv_history_prepare(kv_history, original, features, prefix, position=position)

    kv_history.prepare = fused

    def restore():
        del kv_history.prepare

    return restore


def install_publish_options(drafter, *, merge_release, fused_steady_state):
    """Rebind drafter.prepare_publication (an instance attribute, shadowing the
    class method) to call the class method with BOTH merge_release
    (dflash_pipelined_publish.QWEN_FAST_PIPELINED_PUBLISH) and fused_steady_state
    (this module's QWEN_FAST_TRACED_PUBLISH) baked in together - a single installer
    rather than two independently-stacked shims, so neither flag's effect can be
    silently dropped by the other overwriting drafter.prepare_publication after it.

    fused_steady_state ALSO - separately - installs a transient override on
    drafter.kv_history.prepare for the same call, via install_fused_kv_history above
    (which may itself decline - see its docstring). dflash_device.DFlashDevice.
    prepare_publication's own fused_steady_state branch (the target-feature history
    buffer, not the K/V cache) is unaffected by any of this and keeps taking the
    plain fused_steady_state argument passed through combined() below regardless of
    whether the kv_history install succeeded, declined, or was skipped.

    Returns a restore callable that undoes both installs. drafter.prepare_publication
    must not already be overridden."""
    if type(merge_release) is not bool or type(fused_steady_state) is not bool:
        raise ValueError('Explicit merge_release and fused_steady_state selection required')
    if 'prepare_publication' in drafter.__dict__:
        raise ValueError('drafter.prepare_publication is already overridden')
    cls = type(drafter)
    restore_kv_history = install_fused_kv_history(getattr(drafter, 'kv_history', None)) if fused_steady_state else None

    def combined(features, prefix, *, position):
        return cls.prepare_publication(drafter, features, prefix, position=position,
            merge_release=merge_release, fused_steady_state=fused_steady_state)

    drafter.prepare_publication = combined

    def restore():
        del drafter.prepare_publication
        if restore_kv_history is not None:
            restore_kv_history()

    return restore
