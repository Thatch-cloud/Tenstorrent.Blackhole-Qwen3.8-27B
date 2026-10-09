"""The TP4 verify-glue levers (branch tp4/vglue): one strict flag per lever, and the markers a gated arm reads.

Every lever is a byte-identical replacement of ops the packed 64-row verify (or the GDN commit) already runs, so
each one is behind its own flag and an A/B can isolate it. All flags are strict: unset or '0' is off, '1' is on,
anything else raises. They are read at capture or attach (like verify_trace_t1/t2), never at import, and any of
them on while the process serves the pair (QWEN_FAST_TP unset or 2) raises: these levers exist for four cards only
and the pair's launches stay exactly what they were.

  QWEN_FAST_TP4_COMMIT_LANES   C1a  GDN commit: the 8-lane pipelined DMA (gdn_commit_lanes_tp.cpp).
  QWEN_FAST_TP4_SHARD_VALUES   V4a  sampler: the per-shard max value by gather at the argmax id, not a 2-core Reduce.
  QWEN_FAST_TP4_GDN_GLUE       V2   GDN split/merge as one DMA launch each (gdn_rows_dma_tp).
  QWEN_FAST_TP4_GDN_BLOCK_CONV V1   one conv-gates launch per GDN layer for all four users (needs V2).
  QWEN_FAST_TP4_ATTN_FOLD      V3a  attention query fold-in / result fold-out as one DMA launch each.
  QWEN_FAST_TP4_VGLUE_AUDIT    a correctness arm only: each engaged lever is compared with the served path in-trace.

tp4/gluefix adds two flags that are not levers of this table (gdn_pair_slice_tp has them): QWEN_FAST_GDN_PAIR_SLICE (the odd users'
half-tile slices of the packed projection share ONE row-major conversion; only when V2 is off, V2 already replaces the slices) and
QWEN_FAST_GDN_DISPATCH_DIAG (a host log of each GDN layer's launch order and enqueue gaps). Both are strict, refused at the pair, and
PAIR_SLICE is audited by QWEN_FAST_TP4_VGLUE_AUDIT like a lever.

Markers: FALLBACK and AUDIT_MISMATCH fail a gated arm; ENGAGED is the proof a lever ran.

Stdlib only, py 3.7.
"""

import os

import tp_shapes

COMMIT_LANES = 'QWEN_FAST_TP4_COMMIT_LANES'
SHARD_VALUES = 'QWEN_FAST_TP4_SHARD_VALUES'
GDN_GLUE = 'QWEN_FAST_TP4_GDN_GLUE'
GDN_BLOCK_CONV = 'QWEN_FAST_TP4_GDN_BLOCK_CONV'
ATTN_FOLD = 'QWEN_FAST_TP4_ATTN_FOLD'
AUDIT = 'QWEN_FAST_TP4_VGLUE_AUDIT'
PAIR_SLICE = 'QWEN_FAST_GDN_PAIR_SLICE'
DISPATCH_DIAG = 'QWEN_FAST_GDN_DISPATCH_DIAG'

LEVERS = (COMMIT_LANES, SHARD_VALUES, GDN_GLUE, GDN_BLOCK_CONV, ATTN_FOLD)
ALL_FLAGS = LEVERS + (PAIR_SLICE, DISPATCH_DIAG, AUDIT)

ENGAGED = '[PINDIAG] tp4 vglue engaged'
FALLBACK = '[PINDIAG] tp4 vglue fell back'
# Octo-T8 (QWEN_FAST_OCTO): the half-tile glue kernels (gdn_rows_dma_tp, gdn_block_conv_tp) move 16-row users; an 8-row user is a quarter tile no kernel here moves yet. A block of
# eight-row segments therefore takes the SERVED (pinned) path at those sites - exact, a few launches slower - and says so with this line, which is NOT the FALLBACK marker (a gated
# arm fails on FALLBACK because there it means a lever saved nothing by accident; here it is the known state of the octo block, and the timing arms measure it).
EIGHT_ROW_SERVED = '[PINDIAG] tp4 vglue octo 8-row served path'
AUDIT_MARKER = '[PINDIAG] tp4 vglue audit'
AUDIT_MISMATCH = '[PINDIAG] tp4 vglue audit mismatch'

# What each lever needs at run time, by basename. The image copy lists and the CPU allowlist must name every entry
# (test_tp4_vglue checks it), so what the CPU tests proved is what ships.
RUNTIME_FILES = ('tp4_vglue.py', 'gdn_commit_lanes_tp.cpp', 'gdn_rows_dma_tp.py', 'gdn_rows_dma_tp.cpp',
                 'gdn_device_loop_state_tp.py', 'gdn_block_conv_tp.py', 'attention_block_fold_tp.py',
                 'attention_block_fold_tp.cpp', 'extent_attention_fold_tp.py', 'extent_attention_octo_tp.py', 'gdn_pair_slice_tp.py')


def _read(name, environ):
    value = (os.environ if environ is None else environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def _check_width(environ):
    source = os.environ if environ is None else environ
    if tp_shapes.chip_count(source) != tp_shapes.PAIR:
        return
    on = [name for name in ALL_FLAGS if _read(name, source)]
    if on:
        raise ValueError('%s are TP4 levers: they need QWEN_FAST_TP=4, this process serves the pair' % ', '.join(on))


def enabled(name, environ=None):
    """Whether lever `name` (one of LEVERS) is on. Strict; raises on any lever flag at two cards, and when a lever
    needs another that is off (V1 needs V2)."""
    if name not in LEVERS:
        raise ValueError('Unknown vglue lever %r' % (name,))
    _check_width(environ)
    on = _read(name, environ)
    if on and name == GDN_BLOCK_CONV and not _read(GDN_GLUE, environ):
        raise ValueError('%s needs %s=1 (the block conv reads the split and merge DMA)' % (GDN_BLOCK_CONV, GDN_GLUE))
    return on


def pair_slice_enabled(environ=None):
    """QWEN_FAST_GDN_PAIR_SLICE (tp4/gluefix). Strict; raises at the pair."""
    _check_width(environ)
    return _read(PAIR_SLICE, environ)


def dispatch_diag_enabled(environ=None):
    """QWEN_FAST_GDN_DISPATCH_DIAG (tp4/gluefix): a host-side log only. Strict; raises at the pair."""
    _check_width(environ)
    return _read(DISPATCH_DIAG, environ)


def audit_enabled(environ=None):
    """QWEN_FAST_TP4_VGLUE_AUDIT=1 with at least one lever (or the pair slice) on."""
    _check_width(environ)
    return _read(AUDIT, environ) and (any(_read(name, environ) for name in LEVERS) or _read(PAIR_SLICE, environ))


def engaged_levers(environ=None):
    """The levers that are on, in LEVERS order."""
    return tuple(name for name in LEVERS if enabled(name, environ))


def marker(site, **counts):
    """The ENGAGED line: `[PINDIAG] tp4 vglue engaged site=<site> k=v ...` with the counts in the order given."""
    return ' '.join([ENGAGED, 'site=%s' % site] + ['%s=%s' % pair for pair in counts.items()])


def log_line(message):
    """One line into the server log: loguru where it exists, stdout otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, flush=True)
        else:
            logger.info('{}', message)
    except BaseException:
        pass


_COUNTS = {}


def note(name, count=1):
    """Count what a captured forward engaged (per GDN layer, per attention layer); packed_verifier reads it with take()
    after the capture and logs the ENGAGED line."""
    _COUNTS[name] = _COUNTS.get(name, 0) + count


def take():
    """The counts since the last take, then zero."""
    counts = dict(_COUNTS)
    _COUNTS.clear()
    return counts


# --- the in-trace audit (QWEN_FAST_TP4_VGLUE_AUDIT, gate arms only) -------------------------------------------------------
# Each engaged lever builds what the served path would have produced beside its own launch and holds both (as DRAM copies,
# outside `owned`) on the layer's result dictionary as {label, mine, served[, placement]}; the entries live under the keys
# below, on a user's result (per-user tensors) or on the block's combined result. packed_verifier compares them after the
# replay, on every chip, as int16 bit patterns (-0 and +0 differ), and ModelBatch frees them with the retained records.

AUDIT_KEYS = ('vglue_audit', 'vglue_merge_audit', 'vglue_block_audit', 'vglue_pair_audit')


def audit_entries(result):
    pieces = result.get('segment_results') or (result,)
    seen, entries = set(), []
    for holder in (*pieces, result):
        for key in AUDIT_KEYS:
            for entry in holder.get(key) or ():
                if id(entry) not in seen:
                    seen.add(id(entry))
                    entries.append(entry)
    return entries


def audit_held_of(result):
    """Every tensor the audit holds for a GDN layer result, outside `owned` (freed with the retained record)."""
    return [entry[name] for entry in audit_entries(result) for name in ('mine', 'served') if entry.get(name) is not None]


def compare_entry(operations, entry):
    """One entry on every chip: shape and int16 bits (logical rows), and the recorded placements when the entry has both.
    Returns a list of mismatch descriptions."""
    import torch

    mismatches = []
    placement = entry.get('placement')
    if placement is not None and placement[0] != placement[1]:
        mismatches.append('%s: placement %r against served %r' % (entry['label'], placement[0], placement[1]))
    for chip, (left, right) in enumerate(zip(operations.get_device_tensors(entry['mine']),
                                             operations.get_device_tensors(entry['served']))):
        a = operations.to_torch(left).contiguous().view(torch.int16)
        b = operations.to_torch(right).contiguous().view(torch.int16)
        if a.shape != b.shape:
            mismatches.append('%s chip %d: shape %r against %r' % (entry['label'], chip, tuple(a.shape), tuple(b.shape)))
        elif not torch.equal(a, b):
            mismatches.append('%s chip %d: %d of %d elements differ' % (entry['label'], chip, int((a != b).sum()), a.numel()))
    return mismatches


def expected_entries(result, environ=None):
    """The audit entries an audited GDN layer result must carry for the levers that are on, as {label prefix: count}: per
    user one 'piece user k' (V2) and, when V1 is on, eight 'block conv user k ' (conv, beta, g, z, four windows), and one
    'merged output' (V2) when the block holds two or more users. A declined or fallen-back lever leaves entries missing, so
    the audit fails it instead of passing on what is left."""
    pieces = result.get('segment_results') or (result,)
    users = len(pieces)
    want = {}
    if enabled(GDN_GLUE, environ):
        for user in range(users):
            want['piece user %d' % user] = 1
        if users >= 2:
            want['merged output'] = 1
    elif pair_slice_enabled(environ):
        # V2 off: the pair slice serves the odd users' slices (V2 on replaces them, and the slice stays idle).
        for user in range(1, users, 2):
            want['pair slice user %d' % user] = 1
    if enabled(GDN_BLOCK_CONV, environ):
        for user in range(users):
            want['block conv user %d ' % user] = BLOCK_ENTRIES_PER_USER
    return want


BLOCK_ENTRIES_PER_USER = 8


def missing_entries(result, environ=None):
    """Descriptions of the entries `expected_entries` names that `result` does not carry."""
    labels = [entry['label'] for entry in audit_entries(result)]
    found = []
    for prefix, count in expected_entries(result, environ).items():
        have = sum(1 for label in labels if label == prefix or (prefix.endswith(' ') and label.startswith(prefix)))
        if have != count:
            found.append('%s: %d entries, expected %d' % (prefix.strip(), have, count))
    return found


def audit_round(operations, records, round_number):
    """Compare the layers verify_trace_t2.audit_layers names for this round (every layer on round 1, then two per round in
    rotation). Every audited layer must carry the entries the engaged GDN levers imply (a lever that declined on a layer is
    a failure, not a pass on what is left) and something must have been compared. Logs AUDIT_MARKER
    '<n> exact=True layers=<L> entries=<k>' or AUDIT_MISMATCH and raises. With neither GDN lever on there is nothing for
    this audit to read (V4a has its own in packed_verifier) and it returns 0 without a line."""
    import verify_trace_t2

    if not (enabled(GDN_GLUE) or pair_slice_enabled()):
        return 0
    layers = verify_trace_t2.audit_layers(round_number, len(records) or verify_trace_t2.GDN_LAYERS)
    compared, mismatches = 0, []
    for layer in layers:
        result = records[layer][1]
        mismatches.extend('layer %d %s' % (layer, text) for text in missing_entries(result))
        for entry in audit_entries(result):
            compared += 1
            mismatches.extend('layer %d %s' % (layer, text) for text in compare_entry(operations, entry))
    label = verify_trace_t2.layers_label(layers)
    if mismatches or not compared:
        message = '%s round=%d layers=%s %s' % (AUDIT_MISMATCH, round_number, label,
                                               '; '.join(mismatches[:4]) or 'nothing compared')
        log_line(message)
        raise AssertionError(message)
    _AUDIT['rounds'] += 1
    log_line('%s %d exact=True layers=%s entries=%d' % (AUDIT_MARKER, _AUDIT['rounds'], label, compared))
    return compared


_AUDIT = dict(rounds=0)
