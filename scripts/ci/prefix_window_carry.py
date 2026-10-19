"""Exact resume P0, the fast path's half: the drafter window rows a prefix checkpoint carries (docs/tp4-exact-resume.md 5.2).

WHY. Sticky sessions resume a returning turn at Q <= floor2048(P' - 2048) because the DFlash drafter reads the target taps of the window
[P' - 2048, P') and those exist only for positions the prefill computes (dflash_prefill_window.resume_start). With the scheduler graft's
undrop (QWEN_PREFIX_STICKY_UNDROP) the next turn's hit reaches the boundary floor2048(P) the cap publishes; resuming there needs the window
rows below it. They are the taps the capturing turn's prefill computed for positions [P - 2048, floor2048(P)), by the full 2048-token chunk
programs at absolute boundaries - the same programs, on the same bytes, a cold prefill of the next prompt runs at those positions - so a
copy of them is the cold window's rows, byte for byte (no projection or kernel is re-run on them).

WHAT (QWEN_PREFIX_WINDOW_CARRY=1; default off, gate-only profiles; needs sticky sessions and the undrop):
  * after every sticky prefill (serving_lifecycle, when its capture completes), if the request stored a checkpoint at B = floor2048(P),
    export() reads rows [max(0, P - 2048), B) of each of the capture's tap outputs to host (ConcatMeshToTensor over the mesh, every chip's
    own bytes) and the registry files them on that checkpoint (PrefixRegistry.attach_window, charged to its byte budget);
  * a returning turn granted at Q above P' - 2048 (the trim checked qwen_prefix_registry.window_covers, the lifecycle checks it again)
    gets a capture built with Carried(window, P'): the capture's ledger counts the carried rows [max(0, P' - 2048), Q) and its outputs()
    put them, uploaded exactly as they were read (ShardTensorToMesh over the same mesh dimension), in front of this prefill's own rows. Every
    consumer of the window (the parked-engine rebind, the cold build) sees a complete window of the usual shape.
AUDIT (QWEN_PREFIX_WINDOW_CARRY_AUDIT=1, gate only): the uploaded rows are read back and compared with the host bytes (transport), and every
sticky prefill logs the sha256 of its whole window, so the prefix gate holds a hit's window equal to its cold twin's:
    [PINDIAG] prefix: window carry audit req=<id> P=<n> Q=<n> carried=<rows> window_sha=<64 hex> transport=<exact|mismatch|none> exact=<True|False>
MARKERS: '[PINDIAG] prefix: window carry engaged' (the scheduler graft's install), '... stored req= pos= rows= bytes= ms=' (an export),
'... carried req= Q= rows=' (a carried admission), '... skipped req= why=' (an export that could not run: the next turn resumes lower).
Stdlib at import; torch only inside the functions that move bytes, ttnn only through the `operations` the caller passes.
"""

import hashlib
import os
import time

CHUNK = 2048
FLAG = 'QWEN_PREFIX_WINDOW_CARRY'
AUDIT_FLAG = 'QWEN_PREFIX_WINDOW_CARRY_AUDIT'
MARKER = '[PINDIAG] prefix: window carry'
STORED_LINE = MARKER + ' stored req={} pos={} rows={} bytes={} ms={:.1f}'
CARRIED_LINE = MARKER + ' carried req={} Q={} P={} rows={}'
SKIPPED_LINE = MARKER + ' skipped req={} why={}'
AUDIT_LINE = MARKER + ' audit req={} P={} Q={} carried={} window_sha={} transport={} exact={}'


def _switch(environ, name):
    environ = os.environ if environ is None else environ
    raw = environ.get(name) or '0'
    if raw not in ('0', '1'):
        raise ValueError('%s=%r: only 0 or 1' % (name, raw))
    return raw == '1'


def enabled(environ=None):
    return _switch(environ, FLAG)


def audit_enabled(environ=None):
    return _switch(environ, AUDIT_FLAG)


def window_start(position):
    """The first position of the drafter window of a prompt of `position` tokens (dflash_prefill_window.prefill_window)."""
    return max(0, int(position) - CHUNK)


def host_bytes(tensor):
    """The raw bytes of a host torch tensor (bfloat16 read as int16: no numeric conversion)."""
    import torch

    flat = tensor.contiguous()
    if flat.dtype == torch.bfloat16:
        flat = flat.view(torch.int16)
    return flat.numpy().tobytes()


def digest(payload):
    sha = hashlib.sha256()
    for tensor in payload:
        sha.update(repr(tuple(tensor.shape)).encode())
        sha.update(host_bytes(tensor))
    return sha.hexdigest()


def export(operations, mesh, outputs, position, boundary):
    """(start, payload, nbytes, digest): host copies of rows [window_start(position), boundary) of every tap output of a complete capture
    (each output holds the window rows [window_start(position), position) of one tap). Raises on a shape it does not expect; the caller
    counts that as a skipped export."""
    start = window_start(position)
    rows = int(boundary) - start
    if not 0 < rows <= CHUNK or boundary > position:
        raise ValueError('no window rows below the boundary %d of a %d-token prompt' % (boundary, position))
    composer = operations.ConcatMeshToTensor(mesh, dim=0)
    payload = []
    for tap in outputs:
        shape = tuple(tap.shape)
        if len(shape) != 4 or shape[2] != min(position, CHUNK):
            raise ValueError('a complete window output has %d rows, not %r' % (min(position, CHUNK), shape))
        if rows == shape[2]:
            # A prompt that ends on the boundary carries its whole window: read the output itself (a full-range slice may be the
            # same buffer, which must not be released under the capture).
            payload.append(operations.to_torch(tap, mesh_composer=composer).contiguous())
            continue
        piece = operations.slice(tap, (0, 0, 0, 0), (shape[0], shape[1], rows, shape[3]))
        try:
            payload.append(operations.to_torch(piece, mesh_composer=composer).contiguous())
        finally:
            operations.deallocate(piece)
    nbytes = sum(tensor.numel() * tensor.element_size() for tensor in payload)
    return start, payload, nbytes, digest(payload)


class Carried(object):
    """The window rows a carried admission puts in front of its own: [window_start(prompt), q), the newest `rows` rows of the
    checkpoint's Window (which holds [window.start, q))."""

    def __init__(self, window, prompt, q):
        if window is None or not window.covers(prompt, q):
            raise ValueError('the checkpoint at %d carries no window covering [%d, %d)' % (q, window_start(prompt), q))
        self.start, self.end = window_start(prompt), int(q)
        self.rows = self.end - self.start
        self.payload = window.payload
        self.stored = int(window.end) - int(window.start)
        self.transport = 'none'
        self.uploaded = []

    def host_rows(self):
        return [tensor[:, :, self.stored - self.rows:, :].contiguous() for tensor in self.payload]

    def upload(self, operations, mesh, audit=False):
        """Device tensors of the carried rows, one per tap, laid out as the capture's own outputs (per chip, DRAM, tile); with audit, read
        back and compared with the host bytes (self.transport = 'exact' or 'mismatch')."""
        mapper = operations.ShardTensorToMesh(mesh, dim=0)
        hosts = self.host_rows()
        out = []
        try:
            for host in hosts:
                out.append(operations.from_torch(host, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT, device=mesh,
                                                 mesh_mapper=mapper, memory_config=operations.DRAM_MEMORY_CONFIG))
            if audit:
                composer = operations.ConcatMeshToTensor(mesh, dim=0)
                same = all(host_bytes(operations.to_torch(tensor, mesh_composer=composer)) == host_bytes(host)
                           for tensor, host in zip(out, hosts))
                self.transport = 'exact' if same else 'mismatch'
        except BaseException:
            for tensor in out:
                operations.deallocate(tensor)
            raise
        self.uploaded = out
        return out


def window_sha(operations, mesh, outputs):
    """sha256 of a complete window (every tap, every chip): the audit's comparison key between a hit and its cold twin."""
    composer = operations.ConcatMeshToTensor(mesh, dim=0)
    return digest([operations.to_torch(tap, mesh_composer=composer) for tap in outputs])


def after_prefill(capture, request_id, registry, operations, mesh, log, audit=False):
    """The lifecycle's call once a sticky prefill's capture is complete: export the window rows below floor2048(P) onto the checkpoint this
    request stored there (if it stored one), and, under the audit, log the window's digest. Never raises: a failed export means only that
    the next turn resumes at the window ceiling, as without the carry."""
    position = capture.position
    boundary = (position // CHUNK) * CHUNK
    carried = getattr(capture, 'carried', None)
    try:
        outputs = None
        if boundary and registry is not None and registry.stored_at(request_id, boundary):
            began = time.perf_counter()
            outputs = capture.outputs()
            start, payload, nbytes, sha = export(operations, mesh, outputs, position, boundary)
            if registry.attach_window(request_id, boundary, start, payload, nbytes, sha) is None:
                log(SKIPPED_LINE, request_id, 'checkpoint-gone')
            else:
                log(STORED_LINE, request_id, boundary, boundary - start, nbytes, (time.perf_counter() - began) * 1000.0)
        if audit:
            outputs = capture.outputs() if outputs is None else outputs
            transport = carried.transport if carried is not None else 'none'
            log(AUDIT_LINE, request_id, position, capture.start, carried.rows if carried is not None else 0,
                window_sha(operations, mesh, outputs), transport, transport != 'mismatch')
    except Exception as error:
        try:
            log(SKIPPED_LINE, request_id, '%s:%s' % (type(error).__name__, str(error)[:120].replace(' ', '_')))
        except Exception:
            pass
