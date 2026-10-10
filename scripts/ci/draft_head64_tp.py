"""The drafter quad's LM head as ONE 64-row matmul (QWEN_FAST_DRAFT_HEAD64, default off; F-F4 of the op-fusion programme).

WHY. The device profile of the shipped stack (run 38030902670, trace of the quad) shows the tail of each quad pass running the vocabulary-shard head twice, once on each tile-aligned 32-row half of the
64-row block: two matmuls of 891 us, each reading the same 62,080-column weight shard from DRAM, then four TopK of 171 us (two chunks a half) that stay. The head is weight-bandwidth bound, so one 64-row
launch reads the weights once: an estimated -0.89 ms a quad, -1.8 ms a round.

WHAT. draft_shared_head_tp.block_head_candidates runs `linear(normalized (64 rows), lm_head_weight)` once, and takes each half's chunk candidates (slice, pad, TopK) from the 64-row logits with the
served ops, the slices cutting their rows as well as their columns. The candidate merge, the readback and the selector are the served ones.

EXACTNESS IS THE QUESTION THIS LEVER ASKS. The ops below the matmul see the same bytes iff the matmul's rows at M = 64 equal its rows at M = 32. A matmul with no program config picks one by shape;
at M = 64 the auto config may block K differently from M = 32, and a different K blocking changes where partial sums round. So the lever is bit-identical to the served head only if the two configs
agree on this shape, which the audit decides (QWEN_FAST_DRAFT_HEAD64_AUDIT, in the bucket's eager warm pass: the served two-half head runs beside it and every candidates tensor is compared byte for
byte on every chip). If the audit finds a difference the lever is TAU-ONLY: the drafter's candidates change in rare low-order cases, never a committed token (lossless greedy verification commits only the
target's argmax), and the gate is a tau A/B on the arm without the audit flag.

FLAGS (strict 0 or 1, QWEN_FAST_TP=4 only): QWEN_FAST_DRAFT_HEAD64, QWEN_FAST_DRAFT_HEAD64_AUDIT (needs the lever). Markers: '[PINDIAG] tp4 draft head64 engaged ...', '... fell back ...',
'... audit exact=True ...', '... audit mismatch ...'. A call this module cannot take returns None and the caller runs the served two-half head.

Stdlib only at import, py 3.7.
"""

import draft_permute_tp as permute

FLAG = 'QWEN_FAST_DRAFT_HEAD64'
AUDIT_FLAG = 'QWEN_FAST_DRAFT_HEAD64_AUDIT'
ENGAGED = '[PINDIAG] tp4 draft head64 engaged'
FALLBACK = '[PINDIAG] tp4 draft head64 fell back'
AUDIT = '[PINDIAG] tp4 draft head64 audit'
MISMATCH = '[PINDIAG] tp4 draft head64 audit mismatch'
RUNTIME_FILES = ('draft_head64_tp.py',)


def enabled(environ=None):
    return permute.lever_enabled(FLAG, environ)


def audit_enabled(environ=None):
    return permute.lever_audit_enabled(AUDIT_FLAG, FLAG, environ)


def hook_enabled(environ=None):
    """What the quad's head asks: the lever is on and this call is not a served reference."""
    return permute._SERVED['depth'] == 0 and enabled(environ)


def candidates(operations, model, normalized, owned, *, served, site, halves=2):
    """The two halves' candidate lists of a 64-row learned-normalized block from one matmul, or None when this call cannot take it (the caller runs the served head). `served()` is the served
    two-half head (the audit's reference, run only in the eager warm pass)."""
    import draft_shared_head_tp

    try:
        made = draft_shared_head_tp.block_head_candidates(operations, model, normalized, owned, halves=halves)
    except ValueError as failure:
        permute.note(FALLBACK, 'site=%s reason=%s' % (site, failure))
        return None
    if permute.auditing(AUDIT_FLAG, FLAG):
        with permute.served_only():
            reference = served()
        pairs = []
        for half, (mine, theirs) in enumerate(zip(made, reference)):
            for chunk, (left, right) in enumerate(zip(mine, theirs)):
                pairs += [('half%d.chunk%d.values' % (half, chunk), left['values'], right['values']),
                          ('half%d.chunk%d.indices' % (half, chunk), left['indices'], right['indices'])]
        permute.compare(operations, pairs, 'head', site, audit=AUDIT, mismatch=MISMATCH)
    permute.note(ENGAGED, 'site=%s rows=%d halves=%d chunks=%d' % (site, 32 * halves, halves, len(made[0])))
    return made
