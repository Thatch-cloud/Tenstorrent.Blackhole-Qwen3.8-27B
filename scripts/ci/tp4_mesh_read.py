"""W3 MR and 2a: the verify readback as one mesh-level read, and the quad's head outputs as one joined tensor (tp4/w3-mr).

Two levers, each default off, strict 0/1, byte-identical off, TP4 only (they raise at the pair). Both move bytes and nothing else: no
operation computes a value, so the exactness argument is "the same bytes, read differently", and the audits below hold it on a card.

MR  QWEN_FAST_TP4_VERIFY_MESH_READ   PackedVerifierEngine.shard_predictions reads the sampler's ids and values with ONE mesh-level
    read each (ttnn.to_torch(tensor, mesh_composer=ConcatMeshToTensor(mesh, dim=0))) instead of one blocking read per chip, and splits
    the result back into the per-chip host tensors the loop would have read (every chip's part has the same shape, so the composer's
    dim-0 concatenation is chip 0's part, then chip 1's, ...). The downstream combine, audits and reads_ms measurement are untouched.
    Also used by the quad readback below when its flag or this one is on. The `word` arm of the MR probe (one joined uint32 tensor,
    `(column << 16) | bits`) is NOT built here: S1's kernels write separate ids and values pages, so a joined word needs a change to the
    traced pass, and which of the two reads wins is the MR4 job's verdict (tp4_mr_probe), not this module's.
    QWEN_FAST_TP4_VERIFY_MESH_READ_AUDIT=1 (needs the lever): every verify ALSO runs the per-chip loop and compares both reads bit for
    bit; `[PACKED-MESH-READ-AUDIT] round=N chips=C equal=1`, or `equal=0 ...` and an AssertionError (the round never uses unequal data).

2a  QWEN_FAST_TP4_QUAD_OUTPUTS_JOINED   the quad's pooled output set (serving_buffer_pool, group (0, 1, 2, 3) and the quad blocks' groups
    ONLY) gains one joined values tensor and one joined indices tensor, (1, 1, 64, 16 x chunks): the traced pass ends with a concat of
    the chunks' tensors into them (dflash_proposal_trace.publish_outputs, recorded in the trace), so a quad's top-16 readback is two
    reads a chip (eight at four chips) instead of 2 chunks x 2 x 4 (sixteen), and the replicated selector features are read from one
    chip instead of four (all four under the audit). `merged_candidates` takes column slices of the joined host tensors, which are the
    chunk tensors' own elements, in the loop's own chunk-major, chip-minor order.
    QWEN_FAST_TP4_QUAD_OUTPUTS_JOINED_AUDIT=1 (needs the lever): every quad readback ALSO reads the chunk tensors the old way (and all
    four chips' features) and compares them with the joined slices bit for bit; `[QUAD-JOINED-AUDIT] round=N chips=C equal=1`, or
    `equal=0 ...` and an AssertionError. QWEN_FAST_DRAFT_SINGLES_AUDIT=all and the batched-selection audit of the round (B1) read the same joined path
    through quad_draft.read_quad_outputs and hold the per-user rows against the singles' and the batched selection against the
    per-user one: the end-to-end exactness evidence.
    The join is tried ONCE, in the eager warm-up (settle), where a runtime that refuses the concat (a uint16 TILE concat at a 16-wide
    last dimension is the unverified one) costs this build the joined set only: one `fell back` line, and the quad reads its chunks
    as before. Needs a pool that holds draft output sets (QWEN_FAST_EXTENT_REPLAY=1, the pool's `draft_outputs=`).

Stdlib only at import (torch inside functions), py 3.7.
"""

import os
import sys
from types import SimpleNamespace

import tp_shapes

MESH_READ_FLAG = 'QWEN_FAST_TP4_VERIFY_MESH_READ'
MESH_READ_AUDIT_FLAG = 'QWEN_FAST_TP4_VERIFY_MESH_READ_AUDIT'
JOINED_FLAG = 'QWEN_FAST_TP4_QUAD_OUTPUTS_JOINED'
JOINED_AUDIT_FLAG = 'QWEN_FAST_TP4_QUAD_OUTPUTS_JOINED_AUDIT'
FLAGS = (MESH_READ_FLAG, MESH_READ_AUDIT_FLAG, JOINED_FLAG, JOINED_AUDIT_FLAG)
RUNTIME_FILES = ('tp4_mesh_read.py',)

MESH_READ_ENGAGED = '[PINDIAG] tp4 verify mesh read engaged'
MESH_READ_FELL_BACK = '[PINDIAG] tp4 verify mesh read fell back'
MESH_READ_AUDIT_MARKER = '[PACKED-MESH-READ-AUDIT]'
JOINED_ENGAGED = '[PINDIAG] tp4 quad outputs joined engaged'
JOINED_FELL_BACK = '[PINDIAG] tp4 quad outputs joined fell back'
JOINED_AUDIT_MARKER = '[QUAD-JOINED-AUDIT]'

# What a draft pass's head leaves per candidate chunk (draft_shared_head: top-16), the joined tensors' stride per chunk.
TOP_CANDIDATES = 16
OUTPUT_TENSOR_KEYS = ('values', 'indices')
# The joined tensors' spec key in serving_buffer_pool's draft output specs.
JOINED_KEY = 'joined'


# ---- the flags ----

def _read(name, environ):
    value = (os.environ if environ is None else environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def _require_four_cards(name, environ):
    if tp_shapes.chip_count(environ) == tp_shapes.PAIR:
        raise ValueError('%s is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % name)


def mesh_read_enabled(environ=None):
    """QWEN_FAST_TP4_VERIFY_MESH_READ. Strict (unset or 0 off, 1 on, anything else raises); raises at the pair, and when the audit flag
    is set without the lever."""
    source = os.environ if environ is None else environ
    if not _read(MESH_READ_FLAG, source):
        if _read(MESH_READ_AUDIT_FLAG, source):
            raise ValueError('%s=1 needs %s=1 (the audit compares the mesh read it replaces)' % (MESH_READ_AUDIT_FLAG, MESH_READ_FLAG))
        return False
    _require_four_cards(MESH_READ_FLAG, source)
    return True


def mesh_read_audit_enabled(environ=None):
    source = os.environ if environ is None else environ
    return mesh_read_enabled(source) and _read(MESH_READ_AUDIT_FLAG, source)


def joined_enabled(environ=None):
    """QWEN_FAST_TP4_QUAD_OUTPUTS_JOINED, strict like mesh_read_enabled."""
    source = os.environ if environ is None else environ
    if not _read(JOINED_FLAG, source):
        if _read(JOINED_AUDIT_FLAG, source):
            raise ValueError('%s=1 needs %s=1 (the audit compares the joined read it replaces)' % (JOINED_AUDIT_FLAG, JOINED_FLAG))
        return False
    _require_four_cards(JOINED_FLAG, source)
    return True


def joined_audit_enabled(environ=None):
    source = os.environ if environ is None else environ
    return joined_enabled(source) and _read(JOINED_AUDIT_FLAG, source)


def any_flag_set(environ=None):
    """Whether any of the four flags is set to something but 0 (so a caller may import this module and validate them)."""
    source = os.environ if environ is None else environ
    return any(source.get(name, '0') != '0' for name in FLAGS)


# ---- logging ----

def log_line(text):
    """One line into the server log: loguru where it exists, stderr otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, file=sys.stderr, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:  # noqa: BLE001
        pass


# ---- bytes ----

def identical(left, right):
    """Whether two host tensors hold the same dtype, shape and bytes (-0 differs from +0, a NaN payload is compared as bits)."""
    import torch

    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    left, right = left.contiguous().reshape(-1), right.contiguous().reshape(-1)
    return bool(torch.equal(left.view(torch.uint8), right.view(torch.uint8)))


# ---- the mesh-level read ----

def _outputs_refusal():
    return '%s chip-local outputs required' % tp_shapes.count_word()


def _heads_refusal():
    return '%s learned head shards required' % tp_shapes.all_chips()


def chip_parts(operations, mesh, tensor, chips, mesh_read, refusal='chip-local parts required'):
    """The per-chip host tensors of one mesh tensor, each what `operations.to_torch(part)` returns for that chip's part. `mesh_read` False is
    the loop (one blocking read a chip, after the chip-count check every caller makes); True is ONE mesh-level read, split by chip: the
    composer concatenates the chips' parts along dim 0 in device order, and every chip's part has the same shape, so chip c's part is
    rows [c * per, (c + 1) * per) of the result."""
    parts = operations.get_device_tensors(tensor)
    if len(parts) != chips:
        raise AssertionError(refusal)
    if not mesh_read:
        return [operations.to_torch(part) for part in parts]
    if mesh is None:
        raise AssertionError('a mesh-level read needs the mesh')
    host = operations.to_torch(tensor, mesh_composer=operations.ConcatMeshToTensor(mesh, dim=0))
    if host.shape[0] == 0 or host.shape[0] % chips:
        raise AssertionError('the mesh read returned %d leading rows for %d chips' % (host.shape[0], chips))
    per = host.shape[0] // chips
    return [host[chip * per:(chip + 1) * per] for chip in range(chips)]


def read_predictions(operations, mesh, ids, values, rows, chips, *, audit=False, round_number=0):
    """MR: (chip_ids, chip_values), the lists PackedVerifierEngine.shard_predictions builds with its loop of reads, from two mesh-level reads.
    Audited, the loop also runs and every chip's ids and values must match the mesh read bit for bit."""
    chip_ids = [part.reshape(-1)[:rows] for part in chip_parts(operations, mesh, ids, chips, True, _outputs_refusal())]
    chip_values = [part.reshape(-1)[:rows] for part in chip_parts(operations, mesh, values, chips, True, _outputs_refusal())]
    if audit:
        loop_ids = [part.reshape(-1)[:rows] for part in chip_parts(operations, mesh, ids, chips, False, _outputs_refusal())]
        loop_values = [part.reshape(-1)[:rows] for part in chip_parts(operations, mesh, values, chips, False, _outputs_refusal())]
        differing = ['ids.%d' % chip for chip, (mine, theirs) in enumerate(zip(chip_ids, loop_ids)) if not identical(mine, theirs)]
        differing += ['values.%d' % chip for chip, (mine, theirs) in enumerate(zip(chip_values, loop_values)) if not identical(mine, theirs)]
        if differing:
            log_line('%s round=%d chips=%d equal=0 differing=%s' % (MESH_READ_AUDIT_MARKER, round_number, chips, ','.join(differing[:8])))
            raise AssertionError('%s the mesh read differs from the per-chip reads: %s' % (MESH_READ_AUDIT_MARKER, ','.join(differing[:8])))
        log_line('%s round=%d chips=%d equal=1' % (MESH_READ_AUDIT_MARKER, round_number, chips))
    return chip_ids, chip_values


def note_mesh_read_engaged(chips):
    log_line('%s chips=%d' % (MESH_READ_ENGAGED, chips))


def note_mesh_read_fell_back(reason):
    log_line('%s reason=%s' % (MESH_READ_FELL_BACK, str(reason).replace(' ', '_')[:200]))


# ---- the joined quad outputs ----

def joined_shape(chunks, rows):
    """The joined tensors' shape: the chunks' top-16 columns side by side, in chunk order."""
    return (1, 1, rows, TOP_CANDIDATES * len(tuple(chunks)))


def joined_problem(outputs, held):
    """Why the pooled joined buffers cannot take a draft pass's chunk outputs (a short reason), or None."""
    joined = getattr(held, 'joined', None)
    if joined is None:
        return 'the pool holds no joined buffer for this group'
    chunks = list(outputs.chunks)
    if not chunks:
        return 'the pass has no candidate chunks'
    rows = tuple(chunks[0]['values'].shape)[2]
    want = joined_shape(chunks, rows)
    for key in OUTPUT_TENSOR_KEYS:
        target = joined[key]
        if tuple(target.shape) != want:
            return 'joined %s shape %s, the chunks need %s' % (key, tuple(target.shape), want)
        for chunk in chunks:
            for attribute in ('dtype', 'layout'):
                if getattr(chunk[key], attribute) != getattr(target, attribute):
                    return 'joined %s %s %s, the chunks hold %s' % (key, attribute, getattr(target, attribute), getattr(chunk[key], attribute))
            if tuple(chunk[key].shape) != (1, 1, rows, TOP_CANDIDATES):
                return 'a chunk %s is %s, not (1, 1, %d, %d)' % (key, tuple(chunk[key].shape), rows, TOP_CANDIDATES)
    return None


def publish_joined(operations, outputs, held):
    """The traced pass's last copies (dflash_proposal_trace.publish_outputs, recorded in the trace): each chunk tensor kind concatenated
    along the last dimension and copied into the pooled joined buffer. Returns (the temporaries the concat made, to be kept alive as long as
    the trace; chunk tensors themselves are never in it)."""
    temps = []
    joined = held.joined
    for key in OUTPUT_TENSOR_KEYS:
        pieces = [chunk[key] for chunk in outputs.chunks]
        if len(pieces) == 1:
            source = pieces[0]
        else:
            source = operations.concat(pieces, dim=3, memory_config=operations.DRAM_MEMORY_CONFIG)
            temps.append(source)
        operations.copy(source, joined[key])
    return temps


_STATE = dict(built=0)


def settle(operations, warm, pooled, slots, *, log=None):
    """At a quad build, after the eager warm-up and the pooled set's chunk copies (dflash_proposal_trace.pool_outputs): the set to capture
    with. With QWEN_FAST_TP4_QUAD_OUTPUTS_JOINED unset it is `pooled` untouched (and this module is not even imported by the caller). With it
    set, the join is run once eagerly over the warm-up's outputs (which compiles the concat and the copies the capture records, and is where
    a refusal is found): success returns `pooled` (its `joined` buffers make the capture's last copies and the readers' joined path) and
    logs JOINED_ENGAGED; any failure returns a plain view of the same pooled chunk tensors without `joined` and logs JOINED_FELL_BACK, so
    the quad reads its chunks as before. Never raises for a refused join."""
    log = log_line if log is None else log
    label = ','.join(str(slot) for slot in slots)
    if pooled is None:
        log('%s slots=[%s] reason=the pool holds no draft output set (QWEN_FAST_EXTENT_REPLAY=1 and draft outputs pooled are required)' % (
            JOINED_FELL_BACK, label))
        return None

    def plain():
        return SimpleNamespace(projected=pooled.projected, chunks=pooled.chunks)

    reason = joined_problem(warm, pooled)
    if reason is not None:
        log('%s slots=[%s] reason=%s' % (JOINED_FELL_BACK, label, reason))
        return plain()
    try:
        temps = publish_joined(operations, warm, pooled)
    except Exception as failure:  # noqa: BLE001 - a refused join costs this build the joined set only
        log('%s slots=[%s] reason=join refused: %s: %s' % (JOINED_FELL_BACK, label, type(failure).__name__, str(failure)[:160]))
        return plain()
    for value in temps:
        try:
            operations.deallocate(value)
        except Exception:  # noqa: BLE001
            pass
    _STATE['built'] += 1
    log('%s quads=%d slots=[%s] chunks=%d width=%d' % (JOINED_ENGAGED, _STATE['built'], label, len(pooled.chunks),
                                                      tuple(pooled.joined['values'].shape)[-1]))
    return pooled


def reset_state():
    """Tests: forget the per-process build count and the audit round counter."""
    _STATE.update(built=0, audit_rounds=0)


class Levers(object):
    """What one quad readback does: `joined` (read the joined tensors), `mesh_read` (one mesh read per tensor), `audit` (the joined audit)."""

    __slots__ = ('joined', 'mesh_read', 'audit')

    def __init__(self, joined, mesh_read, audit):
        self.joined, self.mesh_read, self.audit = joined, mesh_read, audit


def quad_levers(outputs, environ=None):
    """The readback's levers for these quad outputs, or None when neither flag is set (the caller then runs today's loop, statement for
    statement). `joined` is true only when the flag is set AND this build's outputs carry the joined tensors (a build that fell back reads
    its chunks)."""
    source = os.environ if environ is None else environ
    joined_flag, mesh_flag = joined_enabled(source), mesh_read_enabled(source)
    if not joined_flag and not mesh_flag:
        return None
    joined = joined_flag and getattr(outputs, 'joined', None) is not None
    return Levers(joined, mesh_flag, joined and joined_audit_enabled(source))


def chunk_hosts(operations, mesh, chunks, chips, rows, mesh_read):
    """The loop's host chunks: [(chunk, chip, values (rows, 16) float32, indices (rows, 16) int64)] in chunk-major, chip-minor order, from
    the per-chunk tensors (one read a chip, or one mesh read a tensor)."""
    found = []
    for chunk in chunks:
        values = chip_parts(operations, mesh, chunk['values'], chips, mesh_read, _heads_refusal())
        indices = chip_parts(operations, mesh, chunk['indices'], chips, mesh_read, _heads_refusal())
        for chip in range(chips):
            found.append((chunk, chip, values[chip].float().reshape(rows, TOP_CANDIDATES),
                          indices[chip].long().reshape(rows, TOP_CANDIDATES)))
    return found


def joined_hosts(operations, mesh, outputs, chips, rows, mesh_read):
    """The same list from the joined tensors: each chunk's columns [16 c, 16 c + 16) of every chip's joined part."""
    joined = outputs.joined
    chunks = list(outputs.chunks)
    width = TOP_CANDIDATES * len(chunks)
    values = chip_parts(operations, mesh, joined['values'], chips, mesh_read, _heads_refusal())
    indices = chip_parts(operations, mesh, joined['indices'], chips, mesh_read, _heads_refusal())
    found = []
    for number, chunk in enumerate(chunks):
        low, high = number * TOP_CANDIDATES, (number + 1) * TOP_CANDIDATES
        for chip in range(chips):
            found.append((chunk, chip, values[chip].float().reshape(rows, width)[:, low:high].contiguous(),
                          indices[chip].long().reshape(rows, width)[:, low:high].contiguous()))
    return found


def quad_chunk_hosts(operations, mesh, outputs, chips, rows, levers):
    """The quad readback's host chunks under `levers`: from the joined tensors when they are there, else from the chunk tensors (mesh-read
    when asked). Audited, both are read and compared bit for bit before the joined ones are returned."""
    if not levers.joined:
        return chunk_hosts(operations, mesh, outputs.chunks, chips, rows, levers.mesh_read)
    mine = joined_hosts(operations, mesh, outputs, chips, rows, levers.mesh_read)
    if levers.audit:
        _STATE['audit_rounds'] = _STATE.get('audit_rounds', 0) + 1
        theirs = chunk_hosts(operations, mesh, outputs.chunks, chips, rows, False)
        differing = []
        for (chunk, chip, value, index), (other_chunk, other_chip, other_value, other_index) in zip(mine, theirs):
            if (chunk['start'], chunk['stop'], chip) != (other_chunk['start'], other_chunk['stop'], other_chip) \
                    or not identical(value, other_value) or not identical(index, other_index):
                differing.append('chunk=%d-%d.chip=%d' % (chunk['start'], chunk['stop'], chip))
        if differing:
            log_line('%s round=%d chips=%d equal=0 differing=%s' % (JOINED_AUDIT_MARKER, _STATE['audit_rounds'], chips, ','.join(differing[:8])))
            raise AssertionError('%s the joined read differs from the chunk reads: %s' % (JOINED_AUDIT_MARKER, ','.join(differing[:8])))
        log_line('%s round=%d chips=%d chunks=%d equal=1' % (JOINED_AUDIT_MARKER, _STATE['audit_rounds'], chips, len(list(outputs.chunks))))
    return mine


def feature_parts(operations, mesh, projected, chips, levers):
    """The replicated selector features' host parts for the quad readback: every chip (so the caller's equality guard runs, as today), except
    under the joined lever without its audit, where chip 0's part alone is read (the features are replicated by construction; the audit
    reads all four). Returns (parts, expected count)."""
    if levers.joined and not levers.audit:
        parts = operations.get_device_tensors(projected)
        if len(parts) != chips:
            raise AssertionError('Replicated learned selector features differ')
        return [operations.to_torch(parts[0])], 1
    return chip_parts(operations, mesh, projected, chips, levers.mesh_read, 'Replicated learned selector features differ'), chips
