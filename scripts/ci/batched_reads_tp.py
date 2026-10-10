"""Op-fusion programme, host-gap package WPH, lever 3: the verify and collect read-backs as a few reads instead of dozens (QWEN_FAST_BATCHED_READS, audit twin
QWEN_FAST_BATCHED_READS_AUDIT; default off).

WHAT IT REPLACES. Both read-backs take a mesh tensor's four per-chip shards and read each with a BLOCKING ttnn.to_torch (an enqueue-read and a wait, on one chip):
  - the verify readback, packed_verifier.PackedVerifierEngine.shard_predictions: the ids and the maxima, 2 tensors x 4 chips = 8 blocking reads a block (0.85 ms
    each block of the host gap, the device idle);
  - the quad collect, quad_draft_tp.read_quad_outputs: for every candidate chunk the values and the indices, 2 x 4 chips, and the replicated selector features
    on all 4 chips: 34-40 blocking reads (4.3-4.5 ms a round, the device idle).

THE LEVER reads the same tensors as the mesh tensors they are, and cuts the result back into the same per-chip tensors. Two ways, chosen by the flag's value:
  QWEN_FAST_BATCHED_READS=1      `compose`: ONE ttnn.to_torch(tensor, mesh_composer=ConcatMeshToTensor(mesh, dim=0)) a tensor. The pinned build (tt-metal 9f9cd4fd)
                                 reads the whole mesh tensor with one blocking `from_device` (Tensor.cpu: one enqueue-read of every shard, then one wait) and
                                 composes the shards on the host along dim 0. Lever N's model patch already reads logits and GDN state this way. 8 reads become 2 and 36
                                 become 9.
  QWEN_FAST_BATCHED_READS=async  `async`: ttnn.copy_device_to_host_tensor(tensor, host, blocking=False) into host tensors allocated once (ttnn.allocate_tensor_on_host with
                                 the tensor's shape, dtype and layout, kept per tensor), ONE synchronize_device, then the same composition on the host tensors. N
                                 enqueues and one wait. Both bindings were read at the pinned commit (ttnn-nanobind/operations/core.cpp: copy_device_to_host_tensor(
                                 device_tensor, host_tensor, blocking=True, cq_id=None); tensor_ops.cpp copy_to_host: enqueue_read_tensor on the mesh command queue,
                                 `blocking` passed through). The card-M probe (optimisation/ttnn-op/batched_reads) times the served reads and both of these on one card
                                 and compares the bytes; which value the profile sets is its decision.

EXACT BY CONSTRUCTION. Reading is not computing: the bytes a read returns are the bytes in the shard's buffer, however many reads it takes. Composition along dim 0 is a
concatenation of the shards in chip order, so the chip c piece of the composed tensor IS shard c's tensor (the same shape, dtype and values: `split` cuts it back and
checks the sizes); the combine and merge code downstream (verify_trace_t1.combine_shards, draft_shared_head_tp.merge_chunk_candidates, round_host READ) receives the
per-chip tensors it received before and is not touched. A tensor whose composed size is not `chips` equal pieces is read the served way (the whole lever falls back).
The audit twin reads the served way as well, every time for the first AUDIT_FIRST reads and then every AUDIT_EVERY-th, and compares every chip's tensor bit for bit
(`exact=True`); a difference is logged (`... audit mismatch`), the SERVED bytes are returned, and the lever latches off. An exception inside the lever (a binding that
is missing, a shape it cannot cut) latches it off with a `fell back` line and the call is read the served way.

Host only: no kernel and no trace; the same device reads, issued differently.
"""

import os

FLAG = 'QWEN_FAST_BATCHED_READS'
AUDIT_FLAG = 'QWEN_FAST_BATCHED_READS_AUDIT'
ENGAGED_MARKER = '[PINDIAG] tp4 batched reads engaged'
FELL_BACK_MARKER = '[PINDIAG] tp4 batched reads fell back'
AUDIT_MARKER = '[PINDIAG] tp4 batched reads audit'
MISMATCH_MARKER = '[PINDIAG] tp4 batched reads audit mismatch'
AUDIT_FIRST = 16
AUDIT_EVERY = 16
MODES = {'0': None, '1': 'compose', 'async': 'async'}


def mode(environ=None):
    """None (off), 'compose' (=1) or 'async'. Any other value is a configuration error."""
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in MODES:
        raise ValueError('%s must be 0, 1 or async' % FLAG)
    return MODES[value]


def enabled(environ=None):
    return mode(environ) is not None


def audit_enabled(environ=None):
    value = (os.environ if environ is None else environ).get(AUDIT_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % AUDIT_FLAG)
    return value == '1' and enabled(environ)


def log_line(text):
    import verify_prestage

    verify_prestage.log_line(text)


class State:
    def __init__(self):
        self.clear()

    def clear(self):
        self.off = None
        self.engaged = set()
        self.composers = {}     # id(mesh) -> (mesh, composer)
        self.hosts = {}         # id(device tensor) -> ((shape, dtype, layout), host tensor), the async mode's preallocated landing buffers
        self.reads = 0
        self.counts = dict(tensors=0, composed=0, audited=0, mismatches=0)


STATE = State()


def reset():
    """A new attach (and the tests): everything forgotten."""
    STATE.clear()


def latch(reason):
    if STATE.off is None:
        STATE.off = str(reason).replace(' ', '_')[:160]
        log_line('%s reason=%s' % (FELL_BACK_MARKER, STATE.off))


def composer_for(operations, mesh):
    entry = STATE.composers.get(id(mesh))
    if entry is None or entry[0] is not mesh:
        entry = (mesh, operations.ConcatMeshToTensor(mesh, dim=0))
        STATE.composers[id(mesh)] = entry
    return entry[1]


def served(operations, tensors):
    """The served read: every tensor's shards, one blocking to_torch each. [[chip 0 tensor, ..., chip n tensor] per tensor]."""
    return [[operations.to_torch(part) for part in operations.get_device_tensors(tensor)] for tensor in tensors]


def split(host, chips):
    """The composed tensor cut back into `chips` equal pieces along dim 0, each its own copy (the pieces of one composed buffer must not alias, as the shards' did not).
    Raises ValueError for a tensor that is not `chips` equal pieces."""
    if host.ndim < 1 or host.shape[0] == 0 or host.shape[0] % chips:
        raise ValueError('a composed read of %s is not %d equal shards' % (tuple(host.shape), chips))
    return [piece.clone() for piece in host.chunk(chips, dim=0)]


def composed(operations, mesh, tensors, chips):
    composer = composer_for(operations, mesh)
    return [split(operations.to_torch(tensor, mesh_composer=composer), chips) for tensor in tensors]


def host_for(operations, mesh, tensor):
    """The host tensor a non-blocking copy of `tensor` lands in: allocated on first use for the tensor's shape, dtype and layout and kept per tensor (a landing buffer only: a cache
    hit by a different tensor of the same specification is as good, and nothing here holds a device tensor alive)."""
    spec = (tuple(tensor.shape), tensor.dtype, tensor.layout)
    entry = STATE.hosts.get(id(tensor))
    if entry is None or entry[0] != spec:
        entry = (spec, operations.allocate_tensor_on_host(spec[0], spec[1], spec[2], mesh))
        STATE.hosts[id(tensor)] = entry
    return entry[1]


def asynchronous(operations, mesh, tensors, chips):
    hosts = [host_for(operations, mesh, tensor) for tensor in tensors]
    for tensor, host in zip(tensors, hosts):
        operations.copy_device_to_host_tensor(tensor, host, blocking=False)
    operations.synchronize_device(mesh)
    composer = composer_for(operations, mesh)
    return [split(operations.to_torch(host, mesh_composer=composer), chips) for host in hosts]


def same_tensor_bits(left, right):
    """Whether two host tensors are the same dtype, shape and bits (prestage_diff.same_bits)."""
    import prestage_diff

    return prestage_diff.same_bits(left, right)


def read(operations, mesh, tensors, *, site):
    """Every tensor of `tensors` as its per-chip host tensors, [[chip 0, ..., chip n] per tensor], identical to `served`. Falls back to `served` (latching the lever off)
    when the batched read raises or cannot be cut, and returns the served bytes whenever the audit twin finds a difference."""
    import tp_shapes

    tensors = list(tensors)
    if STATE.off is not None:
        return served(operations, tensors)
    chips = tp_shapes.chip_count()
    try:
        how = mode()
        batched = asynchronous(operations, mesh, tensors, chips) if how == 'async' else composed(operations, mesh, tensors, chips)
    except Exception as failure:
        latch('%s:%s' % (site, type(failure).__name__))
        return served(operations, tensors)
    if site not in STATE.engaged:
        STATE.engaged.add(site)
        log_line('%s site=%s mode=%s tensors=%d chips=%d' % (ENGAGED_MARKER, site, how, len(tensors), chips))
    STATE.counts['tensors'] += len(tensors)
    STATE.counts['composed'] += 1
    STATE.reads += 1
    if audit_enabled() and (STATE.reads <= AUDIT_FIRST or STATE.reads % AUDIT_EVERY == 0):
        reference = served(operations, tensors)
        equal = len(reference) == len(batched) and all(
            len(want) == len(got) and all(same_tensor_bits(a, b) for a, b in zip(want, got)) for want, got in zip(reference, batched))
        STATE.counts['audited'] += 1
        if not equal:
            STATE.counts['mismatches'] += 1
            log_line('%s site=%s mode=%s reads=%d tensors=%d exact=False' % (MISMATCH_MARKER, site, how, STATE.reads, len(tensors)))
            latch('audit_mismatch')
            return reference
        log_line('%s site=%s mode=%s reads=%d tensors=%d chips=%d exact=True' % (AUDIT_MARKER, site, how, STATE.reads, len(tensors), chips))
    return batched


def verify_reads(operations, mesh, ids, values, block_rows):
    """The verify readback (packed_verifier.shard_predictions): the per-chip id and maximum rows, each `.reshape(-1)[:block_rows]` as the served code cut them."""
    got_ids, got_values = read(operations, mesh, [ids, values], site='verify')
    return [piece.reshape(-1)[:block_rows] for piece in got_ids], [piece.reshape(-1)[:block_rows] for piece in got_values]


def quad_reads(operations, mesh, outputs, guard):
    """The quad collect's reads (quad_draft_tp.read_quad_outputs): ([(values per chip, indices per chip) per chunk], the selector features: every chip's copy when
    `guard`, else chip 0's alone), each tensor as to_torch returned its shard."""
    chunks = list(outputs.chunks)
    tensors = [tensor for chunk in chunks for tensor in (chunk['values'], chunk['indices'])] + [outputs.projected]
    got = read(operations, mesh, tensors, site='collect')
    pairs = [(got[2 * index], got[2 * index + 1]) for index in range(len(chunks))]
    parts = got[-1]
    return pairs, (parts if guard else parts[:1])
