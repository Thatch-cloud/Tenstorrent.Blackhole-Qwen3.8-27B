"""Prefill lever PF3-T at the model route: the masked tail pass of a prompt is a trace REPLAY (QWEN_FAST_PREFILL_TAIL_TRACE=1, GATE ONLY, default off; docs/tp4-prefill3.md).

WHAT IS PAID TODAY. Every prompt ends in ONE masked tail pass: the smallest of the buckets 128/256/512/1024/2048 that holds the last (prompt mod 2,048) tokens, run eagerly. Below the 2,048-row
bucket that pass is host-bound: ~3,155 ops dispatched at ~100 us each is a flat 320 to 410 ms before any attention (docs/tp4-prefill-2.md section 3.2: bucket 128 costs 324 ms at no
context, bucket 1024 381 ms), against 15 to 150 ms of device work. A replayed trace pays the device time only.

WHY IT WAS NOT A TRACE. A trace bakes every runtime argument and every host-built tensor. The masked tail has four per-call host values: (1) valid_len, an argument of the GDN conv op and of
the fused chunk op, and the one-hot that selects the conv state; (2) the fill width of paged_fill_cache (the real blocks of the tail); (3) the page tables and the RoPE tables; (4) the
logits row selector. (2) and (3) are persistent buffers staged with copy_host_to_device_tensor before each replay, the way the chunk trace stages its own inputs. The logits selector stays
EAGER after the replay (five small programs the stock warm already compiled). (1) is the whole difficulty and is solved WITHOUT a new kernel:

  * every GDN layer of the traced body runs with valid_len = bucket (the "full" semantics of the conv op and the chunk op, nothing is masked inside them) and the padded rows are made
    INERT through the gates: a persistent per-row vector, 0 for a real row and PAD_GATE for a pad row, is ADDED to a and to b (the in-projection's a and b slices) before the layer
    computes g = -exp(A) * softplus(a + dt_bias) and beta = sigmoid(b). For a pad row softplus(a + dt_bias) underflows to exactly 0, so g = 0 (a decay of exp(0) = 1), and
    sigmoid(b) underflows to exactly 0 (a write strength of 0): the recurrence over a pad row is the identity, so the recurrent state after `bucket` rows is the state after
    `valid_len` rows. The add leaves a real row's a and b bit for bit (x + 0 = x);
  * the conv STATE (the last K-1 = 3 real inputs, the next chunk's carry and the decode window) is selected by a persistent one-hot pair (two tiny matmuls and an add, HiFi4 with an
    fp32 accumulator, as the FIR's own one-hot matmul) instead of the op's static last-3-rows state; the op's q/k/v for the real rows are untouched (the conv is causal);
  * the K/V of the pad blocks (the fill is as wide as the bucket) go to a SINK: NP1's scratch page (idle_scratch), a KV block past vLLM's pool that no request owns. The page table the
    fill sees names the request's real blocks first and the sink for every block past the last real one.

EXACTNESS (class E0 IF the card agrees). Everything but the two claims below is the stock program on the stock bytes: the same layers, the same SDPA program (the bucket's own, flexible
chunk start), the same conv op (valid_len is a runtime argument; the program is the same), the same logits ops. The claims a card settles (gate T0/A1):
  C1  the fused chunk op with valid_len = bucket over gate-neutralised pad rows leaves the same recurrent state, byte for byte, as the op with valid_len = the real count;
  C2  the one-hot matmul pair writes the same conv state as the op's valid_len state (its -0 and denormal canonicalisation included).
QWEN_FAST_PREFILL_TAIL_TRACE_AUDIT=1 (gate arms only) proves both inside the attach on real tails: it reads the scratch before the pass, replays the trace, reads the scratch, the final
hidden state of every real row and the five drafter taps, restores the scratch, runs the STOCK pass, and requires every GDN recurrent state, conv carry and conv window, every real row of the
output and every tap row byte equal; the stock result is what the request is served. The K/V of the real rows are not compared here: the levern audit digest (QWEN_FAST_LEVERN_AUDIT) of
the control and the lever arm holds them.

THE DRAFTER TAPS. The drafter's window features are the outputs of five layers, snapshotted while the pass runs (dflash_prefill_window.PrefillWindowCapture wraps _forward_prefill_chunk_masked_tp and
the forward of each tap layer). The trace RETAINS those five layer outputs in persistent buffers; on a replay the tap layers' forward is answered from them, so the capture's own snapshot,
ownership and ledger checks run unchanged on the same bytes.

THE HOLES. A trace replay writes its intermediates into the holes its capture freed, so a buffer allocated after the capture may be overwritten (docs/tp4-traced-publish.md section 5). Every
persistent buffer of the lever (the inputs, the output, the five taps) is allocated BEFORE any capture; the captures run inside an ARENA (a ballast tensor of QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB
per chip, freed just before the first capture and allocated again after the last, required to come back at the same address): nothing else can be placed in the region the captures used. The
arena is DRAM the lever holds for the life of the process (the "prefill transient" of the admission need is a different region: the eager chunk loop still needs it).

THE FLAGS. QWEN_FAST_PREFILL_TAIL_TRACE: unset = off and nothing here is imported; '0' = off, but the tail pass prints its timing line (the CONTROL arm of the timed pair); '1' = the traced
tail. Strict: anything else raises at the attach. It needs QWEN_FAST_LEVER_N=1 and, at '1', QWEN_FAST_SOLO_PACKED=1 (the sink page). QWEN_FAST_PREFILL_TAIL_TRACE_AUDIT needs '1'.
QWEN_FAST_PREFILL_TAIL_TRACE_BUCKETS: the comma list of buckets to trace, a subset of 128,256,512,1024 (default all four; the 2,048-row bucket is device-bound and its full chunks are not tails).
QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB: the arena (default DEFAULT_ARENA_MB; an ESTIMATE, the card ladder settles it).

WHAT STAYS EAGER. A call with vision tokens, a bucket that is not traced, a page table of another width than the warm's, a tap set other than the drafter's, or after a latch.

MARKERS (the smoke table reads them): '[PINDIAG] prefill tail trace engaged ...' once after the warm, '... fell back reason=...' when something latches the lever off, '... audit n=.. exact=True' per
audited tail, '... audit mismatch' (raises). The timing line '[PF3] tail mode=<traced|eager> bucket=.. valid=.. start=.. ms=..' on every tail pass when the flag is set at all.

Stdlib only; ttnn and torch are imported on first use."""

import contextlib
import os
import sys
import time
from collections import namedtuple

FLAG = 'QWEN_FAST_PREFILL_TAIL_TRACE'
AUDIT_FLAG = 'QWEN_FAST_PREFILL_TAIL_TRACE_AUDIT'
BUCKETS_FLAG = 'QWEN_FAST_PREFILL_TAIL_TRACE_BUCKETS'
ARENA_FLAG = 'QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB'
FLAGS = (FLAG, AUDIT_FLAG, BUCKETS_FLAG, ARENA_FLAG)
LEVER_N_FLAG = 'QWEN_FAST_LEVER_N'
SOLO_FLAG = 'QWEN_FAST_SOLO_PACKED'
ENGAGED_MARKER = '[PINDIAG] prefill tail trace engaged'
FELL_BACK_MARKER = '[PINDIAG] prefill tail trace fell back'
AUDIT_MARKER = '[PINDIAG] prefill tail trace audit'
AUDIT_MISMATCH_MARKER = '[PINDIAG] prefill tail trace audit mismatch'
TIMING_MARKER = '[PF3] tail'
SKIP_MARKER = '[PF3] tail skip'
AUDIT_FIRST = 8
AUDIT_STRIDE = 24
AUDIT_PER_BUCKET = 2          # and the first two tails of every traced bucket, whatever the schedule: a bucket's program is its own

BLOCK = 64                                   # KV block size (rows)
STATE_ROWS = 3                               # K - 1 of the depthwise conv
STOCK_BUCKETS = (128, 256, 512, 1024, 2048)  # Qwen36Model._PREFILL_MASK_BUCKETS, pinned equal at install
TRACEABLE = (128, 256, 512, 1024)
DEFAULT_BUCKETS = TRACEABLE
# A pad row's added gate value: softplus(PAD_GATE + dt_bias) is exactly 0 and sigmoid(PAD_GATE) is exactly 0 in bf16 (exp underflows, and nothing overflows into a NaN).
PAD_GATE = -30000.0
# ESTIMATE: the arena must cover what the largest traced bucket's capture allocates and frees. The admission's prefill-transient figure (QWEN_FAST_DRAM_PREFILL_TRANSIENT_MB = 600) is
# the 2,048-row chunk's; the 1,024-row bucket scales it by 1024 / 2048 and the card ladder (docs/tp4-prefill3.md gate T3) shrinks it until a hole is found.
DEFAULT_ARENA_MB = 320
MAX_ARENA_MB = 2048
DRAM_BANKS = 8                               # banks of a chip's DRAM; a tensor takes the same address and 1/8 of its bytes in each
ARENA_WIDTH = 5120                           # the arena is a [1, 1, rows, 5120] bf16 tile tensor on every chip

Settings = namedtuple('Settings', 'present traced audit buckets arena_mb')
TailPlan = namedtuple('TailPlan', 'bucket valid_len chunk_start real_blocks chunk_row')


# ---------------------------------------------------------------------------------------------------------------------------------------
# The flags and the plan (pure python)
# ---------------------------------------------------------------------------------------------------------------------------------------

def _flag(name, environ, allowed):
    value = environ.get(name)
    if value is None:
        return None
    if value not in allowed:
        raise ValueError('%s must be %s, got %r' % (name, ' or '.join(repr(item) for item in allowed), value))
    return value


def parse_buckets(text):
    """The traced buckets a comma list names: a strictly ascending subset of TRACEABLE."""
    try:
        values = tuple(int(item) for item in text.split(','))
    except ValueError:
        raise ValueError('%s must be a comma list of integers, got %r' % (BUCKETS_FLAG, text)) from None
    if not values or list(values) != sorted(set(values)) or any(value not in TRACEABLE for value in values):
        raise ValueError('%s must be a strictly ascending subset of %s, got %r' % (BUCKETS_FLAG, ','.join(str(b) for b in TRACEABLE), text))
    return values


def settings(environ=None):
    """Settings(present, traced, audit, buckets, arena_mb) for this process; strict. present: the flag is set at all (the route then logs its timing line); traced: the tail is replayed."""
    environ = os.environ if environ is None else environ
    value = _flag(FLAG, environ, ('0', '1'))
    audit = _flag(AUDIT_FLAG, environ, ('0', '1'))
    if audit == '1' and value != '1':
        raise ValueError('%s=1 audits the traced tail: it needs %s=1' % (AUDIT_FLAG, FLAG))
    if value is not None and environ.get(LEVER_N_FLAG) != '1':
        raise ValueError('%s is a lever of the Lever N route: it needs %s=1' % (FLAG, LEVER_N_FLAG))
    if value == '1' and environ.get(SOLO_FLAG) != '1':
        raise ValueError('%s=1 needs %s=1: the pad blocks of a tail are written to NP1\'s KV scratch page, a block past vLLM\'s pool that no request owns' % (FLAG, SOLO_FLAG))
    buckets = DEFAULT_BUCKETS
    if BUCKETS_FLAG in environ:
        if value != '1':
            raise ValueError('%s needs %s=1' % (BUCKETS_FLAG, FLAG))
        buckets = parse_buckets(environ[BUCKETS_FLAG])
    arena = DEFAULT_ARENA_MB
    if ARENA_FLAG in environ:
        if value != '1':
            raise ValueError('%s needs %s=1' % (ARENA_FLAG, FLAG))
        text = environ[ARENA_FLAG]
        if not text.isdigit() or int(text) > MAX_ARENA_MB:
            raise ValueError('%s must be a whole number of MB between 0 and %d, got %r' % (ARENA_FLAG, MAX_ARENA_MB, text))
        arena = int(text)
    return Settings(value is not None, value == '1', audit == '1', buckets, arena)


def requested(environ=None):
    """Whether any flag of the lever is present (anything, so a malformed value still reaches the strict parser and fails at the attach)."""
    environ = os.environ if environ is None else environ
    return any(name in environ for name in FLAGS)


def bucket_for(valid_len, buckets=STOCK_BUCKETS):
    """The smallest stock bucket holding `valid_len` rows (Qwen36Model._mask_bucket_for), None when none does."""
    for bucket in buckets:
        if valid_len <= bucket:
            return bucket
    return None


def plan_tail(valid_len, chunk_start, width, page_row, sink, buckets=TRACEABLE):
    """(TailPlan, None) for a tail pass the lever may replay, (None, reason) otherwise. `page_row` is the request's page-table row (width entries). The chunk page table the fill sees names the
    request's real blocks first (those the stock pass fills) and `sink` for every block past the last real one."""
    if type(valid_len) is not int or valid_len < 1:
        return None, 'valid_len'
    bucket = bucket_for(valid_len)
    if bucket is None or bucket not in buckets:
        return None, 'bucket'
    if chunk_start < 0 or chunk_start % BLOCK:
        return None, 'unaligned'
    first = chunk_start // BLOCK
    blocks = bucket // BLOCK
    real = -(-(chunk_start + valid_len) // BLOCK) - first
    if first + blocks > width or len(page_row) != width:
        return None, 'width'
    row = [int(value) for value in page_row[first:first + real]] + [int(sink)] * (blocks - real)
    return TailPlan(bucket, valid_len, chunk_start, real, row), None


def pad_vector(valid_len, bucket):
    """The per-row gate offsets: 0.0 for a real row, PAD_GATE for a pad row (the list the persistent mask is made from)."""
    return [0.0] * valid_len + [PAD_GATE] * (bucket - valid_len)


def select_indices(valid_len):
    """[(row j of the new state, source)] for the conv state: the new state is rows valid_len - 3 .. valid_len - 1 of [carry (3 rows); the bucket's inputs]. A source is ('x', t) for input
    row t or ('c', c) for carry row c: new_state[j] = padded[valid_len + j] where padded = carry + x."""
    out = []
    for j in range(STATE_ROWS):
        index = valid_len + j
        out.append((j, ('c', index) if index < STATE_ROWS else ('x', index - STATE_ROWS)))
    return out


def audit_due(index):
    """Whether the `index`-th (0-based) traced tail is audited by the schedule: the first AUDIT_FIRST, then every AUDIT_STRIDE-th (a bucket's first AUDIT_PER_BUCKET tails are audited besides)."""
    return index < AUDIT_FIRST or (index - AUDIT_FIRST) % AUDIT_STRIDE == AUDIT_STRIDE - 1


def _ttnn():
    import ttnn

    return ttnn


def _bytes(tensor):
    import torch

    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


# ---------------------------------------------------------------------------------------------------------------------------------------
# One bucket's persistent buffers and trace
# ---------------------------------------------------------------------------------------------------------------------------------------

class _Trace(object):
    """The persistent inputs, outputs and trace id of one bucket."""

    def __init__(self, bucket):
        self.bucket = bucket
        self.blocks = bucket // BLOCK
        self.trace_id = None
        self.tok = self.cos = self.sin = self.csi = self.full_pt = self.chunk_pt = None
        self.pad_mask = self.sel_x = self.sel_c = None
        self.out = None
        self.taps = {}
        self.host_refs = []

    def buffers(self):
        named = [self.tok, self.cos, self.sin, self.csi, self.full_pt, self.chunk_pt, self.pad_mask, self.sel_x, self.sel_c, self.out]
        return [tensor for tensor in named + list(self.taps.values()) if tensor is not None]


class TailTrace(object):
    """One process's traced tails: the settings, the persistent buffers and traces per bucket, the latch, and the counters the markers and the tests read."""

    def __init__(self, config, environ=None, ops=None, addresses=None, clock=None, extent=None):
        self.settings = config
        self._extent = extent
        self.environ = os.environ if environ is None else environ
        self.ops = ops
        self._addresses = addresses
        self.clock = clock or time.perf_counter
        self.model = None
        self.original = None
        self.log = lambda message, *values: None
        self.sink = None
        self.width = None
        self.tap_ids = ()
        self.traces = {}
        self.arena = None
        self.arena_address = None
        self.arena_span = None
        self.holes = {}
        self.latched = None
        self.warmed = False
        self.engaged = 0
        self.calls = 0
        self.traced_calls = 0
        self.eager_calls = 0
        self.audited = 0
        self.bucket_calls = {}
        self.mismatches = 0
        self.skips = {}
        self.fallbacks = 0
        self.hook_calls = {'project': 0, 'conv': 0}
        self._active = None
        self._replaying = None
        self._recording = None
        self._saved = []

    @classmethod
    def from_environ(cls, environ=None, **keywords):
        environ = os.environ if environ is None else environ
        return cls(settings(environ), environ, **keywords)

    # -- plumbing ----------------------------------------------------------------------------------------------------------------------
    def _ops(self):
        if self.ops is None:
            self.ops = _ttnn()
        return self.ops

    def _addr(self, tensor):
        if self._addresses is None:
            import tp_addresses

            return tuple(tp_addresses.addresses(self._ops(), tensor))        # one address per chip at the width this process serves (the pinned pair helper needs exactly two)
        return tuple(self._addresses(tensor))

    def _span(self, tensor):
        """The bytes `tensor` takes in each DRAM bank of each chip (the lock-step allocator gives every bank the same address): the injected `extent`, else the tile-padded volume over the banks."""
        if self._extent is not None:
            return int(self._extent(tensor))
        shape = [int(dim) for dim in tensor.shape]
        if len(shape) >= 2 and 'TILE' in str(getattr(tensor, 'layout', '')).upper():
            shape[-1], shape[-2] = -(-shape[-1] // 32) * 32, -(-shape[-2] // 32) * 32
        volume = 1
        for dim in shape:
            volume *= dim
        name = str(getattr(tensor, 'dtype', 'bfloat16')).lower()
        size = 4 if ('float32' in name or 'int32' in name) else (1.0625 if 'bfloat8' in name else 2)
        return int(-(-volume * size // DRAM_BANKS))

    @contextlib.contextmanager
    def _recording_frees(self):
        """Every tensor freed through ttnn.deallocate while a capture runs, as (per-chip addresses, bytes per bank): the holes the capture leaves. A tensor freed by garbage collection is not seen here."""
        ops = self._ops()
        original = ops.deallocate
        freed = []

        def recording(tensor, *arguments, **keywords):
            try:
                freed.append((self._addr(tensor), self._span(tensor)))
            except Exception:
                self.holes['unrecorded'] = self.holes.get('unrecorded', 0) + 1
            return original(tensor, *arguments, **keywords)
        ops.deallocate = recording
        try:
            yield freed
        finally:
            ops.deallocate = original

    def _uncovered(self, freed):
        """The freed extents that do not lie inside the arena on every chip."""
        start, span = self.arena_address, self.arena_span
        bad = []
        for addresses, size in freed:
            if len(addresses) != len(start) or any(a < s or a + size > s + span for a, s in zip(addresses, start)):
                bad.append((addresses, size))
        return bad

    def _set(self, instance, name, value):
        """An instance attribute set, remembered so uninstall puts back exactly what was there."""
        had = name in vars(instance)
        self._saved.append((instance, name, had, vars(instance).get(name)))
        setattr(instance, name, value)

    def _flush(self):
        for instance, name, had, value in reversed(self._saved):
            if had:
                setattr(instance, name, value)
            else:
                vars(instance).pop(name, None)
        self._saved = []

    def _gdn_layers(self):
        return [layer.attention for layer in self.model.layers if not layer.is_full_attention]

    # -- install ------------------------------------------------------------------------------------------------------------------------
    def install(self, model, log, taps=None, sink=None):
        """Shadow the model's masked chunk forward and the tap layers' forward. Refuses (ValueError, failing the attach) a model without what the lever needs."""
        if model.num_devices <= 1:
            raise ValueError('The traced tail is the TP (num_devices>1) path')
        if self.model is not None:
            raise ValueError('The traced tail is already installed on this model')
        name = '_forward_prefill_chunk_masked_tp'
        if not callable(getattr(model, name, None)):
            raise ValueError('The model has no %s: the traced tail replaces its masked chunk forward' % name)
        if tuple(getattr(type(model), '_PREFILL_MASK_BUCKETS', STOCK_BUCKETS)) != STOCK_BUCKETS:
            raise ValueError('The model\'s masked buckets are %r, not %r: the bucket choice is part of the program the trace replays'
                             % (tuple(getattr(type(model), '_PREFILL_MASK_BUCKETS', ())), STOCK_BUCKETS))
        if self.settings.traced:
            missing = [attribute for attribute in ('_rope_tp_cos_sin_torch', '_apply_vision_merge', '_reset_gdn_state_for_new_sequence',
                                                   '_bind_gdn_prefill_scratch', '_unbind_gdn_prefill_scratch', '_ensure_gdn_prefill_scratch')
                       if not callable(getattr(model, attribute, None))]
            if self.settings.audit:
                missing += [attribute for attribute in ('_qwen_prefix_read_scratch', '_qwen_prefix_restore') if not callable(getattr(model, attribute, None))]
            if missing:
                raise ValueError('The model lacks %s: the traced tail cannot be installed' % ', '.join(missing))
            if sink is None:
                import idle_scratch

                reserved = idle_scratch.reserved()
                if not reserved:
                    raise ValueError('%s=1 needs NP1\'s KV scratch page (%s=1 and its allocation hook): none was recorded in this process' % (FLAG, SOLO_FLAG))
                sink = reserved[-1]
            if type(sink) is not int or sink < 2:
                raise ValueError('The sink is a KV block id past vLLM\'s pool (at least 2), got %r' % (sink,))
            if taps is None:
                from dflash_request_runtime import TARGET_TAPS as taps
            self.tap_ids = tuple(int(i) for i in taps)
            if any(not 0 <= i < len(model.layers) for i in self.tap_ids):
                raise ValueError('The tap layers %r are not layers of this model' % (self.tap_ids,))
        self.model, self.log, self.sink = model, log, sink
        self.original = getattr(model, name)
        self._set(model, name, self.shadow)
        if self.settings.traced:
            for index in self.tap_ids:
                self._set(model.layers[index], 'forward', self._tap_shim(index, model.layers[index].forward))
        return self

    def uninstall(self):
        """Release the traces and the buffers and take every shadow off again."""
        model = self.model
        if model is None:
            return
        ops = self._ops() if self.traces or self.arena is not None else None
        for trace in self.traces.values():
            if trace.trace_id is not None:
                try:
                    ops.release_trace(model.device, trace.trace_id)
                except Exception:
                    pass
            for tensor in trace.buffers():
                try:
                    ops.deallocate(tensor)
                except Exception:
                    pass
        if self.arena is not None:
            try:
                ops.deallocate(self.arena)
            except Exception:
                pass
        self.traces, self.arena = {}, None
        self._flush()
        self.model = None
        self.warmed = False

    def _tap_shim(self, index, forward):
        def shim(*args, **keywords):
            if self._replaying is not None:
                return self._replaying[index]
            output = forward(*args, **keywords)
            if self._recording is not None:
                self._recording[index] = self._host_rows(output, self._recording['valid'])
            return output
        return shim

    # -- the host rows of a [1, 1, rows, width] hidden (per-chip composed along dim 0) --------------------------------------------------
    def _host_rows(self, tensor, valid):
        ops = self._ops()
        composed = ops.to_torch(tensor, mesh_composer=ops.ConcatMeshToTensor(self.model.mesh_device, dim=0))
        return composed[..., :valid, :].contiguous()

    # -- the warm: buffers, programs, arena, captures ------------------------------------------------------------------------------------
    def warm(self, model, width, log=None):
        """Allocate every buffer, compile the traced body's programs, capture one trace per bucket inside the arena. Only for the traced arm; True when it ran. Runs before the packed traces
        (levern_route.warm), under the persistent B=1 GDN scratch."""
        if not self.settings.traced:
            return False
        if self.model is not model:
            raise ValueError('The traced tail was installed on another model')
        if self.warmed:
            return False
        if type(width) is not int or width <= 0 or width % 32:
            raise ValueError('The traced tail needs the runner\'s page-table width (a positive multiple of 32), got %r' % (width,))
        log = log or self.log
        ops = self._ops()
        began = self.clock()
        model._ensure_gdn_prefill_scratch()
        self.width = width
        previous = model._bind_gdn_prefill_scratch()
        try:
            self._allocate_all(model, width)
            for bucket in self.settings.buckets:
                self._compile_pass(model, self.traces[bucket])
            self._capture_all(model, log)
        finally:
            model._unbind_gdn_prefill_scratch(previous)
        ops.synchronize_device(model.device)
        self.warmed = True
        self.engaged += 1
        log('{}', '%s buckets=%s traces=%d sink=%d taps=%s arena_mb=%d arena_address=%s freed=%d uncovered=%d capture_ms=%.0f' % (
            ENGAGED_MARKER, ','.join(str(b) for b in self.settings.buckets), len(self.traces), self.sink, ','.join(str(i) for i in self.tap_ids),
            self.settings.arena_mb, '-' if self.arena_address is None else ','.join(str(a) for a in self.arena_address), self.holes.get('recorded', 0), self.holes.get('uncovered', 0),
            (self.clock() - began) * 1000.0))
        return True

    def _rep(self):
        return self._ops().ReplicateTensorToMesh(self.model.device)

    def _device_tensor(self, torch_tensor, dtype, layout):
        ops = self._ops()
        return ops.from_torch(torch_tensor, dtype=dtype, layout=layout, device=self.model.device, mesh_mapper=self._rep())

    def _host_tensor(self, torch_tensor, dtype, layout):
        return self._ops().from_torch(torch_tensor, dtype=dtype, layout=layout, device=None, mesh_mapper=self._rep())

    def _allocate_all(self, model, width):
        """Every persistent buffer of every bucket, before any capture (a buffer allocated after a capture may sit in a hole the capture freed)."""
        import torch

        ops = self._ops()
        nv = int(self._gdn_layers()[0].Nv)
        for bucket in self.settings.buckets:
            trace = self.traces[bucket] = _Trace(bucket)
            trace.nv = nv
            trace.tok = self._device_tensor(torch.zeros(1, bucket, dtype=torch.int32), ops.uint32, ops.ROW_MAJOR_LAYOUT)
            cos_t, sin_t = model._rope_tp_cos_sin_torch(0, bucket)
            trace.cos = self._device_tensor(cos_t, ops.bfloat16, ops.TILE_LAYOUT)
            trace.sin = self._device_tensor(sin_t, ops.bfloat16, ops.TILE_LAYOUT)
            trace.csi = self._device_tensor(torch.zeros(1, dtype=torch.int32), ops.int32, ops.ROW_MAJOR_LAYOUT)
            # The capture and the compile pass write only the sink: the whole table names it (nothing real is read or written before the first request).
            trace.full_pt = self._device_tensor(torch.full((1, width), self.sink, dtype=torch.int32), ops.int32, ops.ROW_MAJOR_LAYOUT)
            trace.chunk_pt = self._device_tensor(torch.full((1, trace.blocks), self.sink, dtype=torch.int32), ops.int32, ops.ROW_MAJOR_LAYOUT)
            trace.pad_mask = self._device_tensor(*self._pad_mask_host(bucket, bucket, nv))
            trace.sel_x = self._device_tensor(*self._sel_x_host(bucket, bucket))
            trace.sel_c = self._device_tensor(*self._sel_c_host(bucket))

    def _pad_mask_host(self, valid_len, bucket, nv):
        import torch

        ops = self._ops()
        column = torch.tensor(pad_vector(valid_len, bucket), dtype=torch.float32).reshape(1, bucket, 1)
        return column.expand(1, bucket, nv).contiguous().to(torch.bfloat16), ops.bfloat16, ops.TILE_LAYOUT

    def _sel_x_host(self, valid_len, bucket):
        import torch

        ops = self._ops()
        sel = torch.zeros(1, STATE_ROWS, bucket, dtype=torch.float32)
        for j, (kind, index) in select_indices(valid_len):
            if kind == 'x':
                sel[0, j, index] = 1.0
        return sel.to(torch.bfloat16), ops.bfloat16, ops.TILE_LAYOUT

    def _sel_c_host(self, valid_len):
        import torch

        ops = self._ops()
        sel = torch.zeros(1, STATE_ROWS, STATE_ROWS, dtype=torch.float32)
        for j, (kind, index) in select_indices(valid_len):
            if kind == 'c':
                sel[0, j, index] = 1.0
        return sel.to(torch.bfloat16), ops.bfloat16, ops.TILE_LAYOUT

    # -- the traced body -----------------------------------------------------------------------------------------------------------------
    def _body(self, model, trace):
        """The masked chunk forward at `trace.bucket` rows over the persistent buffers, valid_len = bucket in every GDN layer; the final hidden state and the tap outputs are copied into the
        persistent output buffers (so every intermediate is freed inside the body and nothing but the buffers outlives a capture). Returns nothing."""
        ops = self._ops()
        bucket = trace.bucket
        x = model.embd(trace.tok)
        x = ops.reshape(x, (1, 1, bucket, x.shape[-1]))
        x = ops.to_memory_config(x, ops.DRAM_MEMORY_CONFIG)
        x = model._apply_vision_merge(x, length=bucket)
        for index, layer in enumerate(model.layers):
            if layer.is_full_attention:
                x_new = layer.forward(x, cos=trace.cos, sin=trace.sin, mode='prefill', page_table=trace.full_pt, chunk_page_table=trace.chunk_pt,
                                      chunk_start_idx_tensor=trace.csi)
            else:
                x_new = layer.forward(x, mode='prefill', chunk_size=model.args.gdn_chunk_size, valid_len=bucket)
            ops.deallocate(x)
            x = x_new
            if index in self.tap_ids:
                self._into(trace, index, x)
        self._into(trace, None, x)
        ops.deallocate(x)

    def _into(self, trace, index, tensor):
        """Copy `tensor` into the persistent buffer for tap `index` (None: the output), allocating the buffer on first use (the compile pass), never inside a capture."""
        ops = self._ops()
        holder = trace.taps if index is not None else None
        current = holder.get(index) if holder is not None else trace.out
        if current is None:
            import torch

            shape = tuple(int(dim) for dim in tensor.shape)
            current = self._device_tensor(torch.zeros(shape, dtype=torch.bfloat16), tensor.dtype, tensor.layout)
            if holder is not None:
                holder[index] = current
            else:
                trace.out = current
        ops.copy(tensor, current)

    def _compile_pass(self, model, trace):
        """One eager run of the body (the programs compile here, outside any capture), then the scratch is zeroed again."""
        before = dict(self.hook_calls)
        with self._hooks(model, trace):
            model._reset_gdn_state_for_new_sequence()
            self._body(model, trace)
            self._ops().synchronize_device(model.device)
        model._reset_gdn_state_for_new_sequence()
        gdn = len(self._gdn_layers())
        seen = dict((name, self.hook_calls[name] - before[name]) for name in before)
        if seen['project'] != gdn or seen['conv'] != gdn:
            raise ValueError('The traced tail needs the GDN conv op engaged on every GDN layer (QWEN_FAST_GDN_PREFILL_CONV=1) and the gate hook on every projection: the compile pass at %d rows saw %d conv '
                             'and %d projection calls for %d GDN layers' % (trace.bucket, seen['conv'], seen['project'], gdn))

    def _check_conv_audit(self):
        """The graft conv op audits its first QWEN_FAST_GDN_PREFILL_CONV_AUDIT calls against the FIR, whose one-hot is a host write: inside a capture that is a TT_FATAL, so the audit must have run out
        (the eager warm before this one calls the op far more often than the knob counts) before the first capture."""
        module = sys.modules[type(self._gdn_layers()[0]).__module__]
        state = getattr(module, '_QWEN_PREFILL_CONV', None)
        limit = int(self.environ.get('QWEN_FAST_GDN_PREFILL_CONV_AUDIT', '0') or 0)
        if isinstance(state, dict) and int(state.get('audited', limit)) < limit:
            raise ValueError('The graft conv op has audited %d of its %d calls (QWEN_FAST_GDN_PREFILL_CONV_AUDIT): the next call would run the FIR inside a trace capture; the eager warm must come first'
                             % (int(state['audited']), limit))

    def _arena_tensor(self, rows):
        ops = self._ops()
        return ops.zeros((1, 1, rows, ARENA_WIDTH), device=self.model.device, dtype=ops.bfloat16, layout=ops.TILE_LAYOUT, memory_config=ops.DRAM_MEMORY_CONFIG)

    def _capture_all(self, model, log):
        """The arena protocol around one capture per bucket (the module docstring, THE HOLES)."""
        ops = self._ops()
        self._check_conv_audit()
        arena_rows = (self.settings.arena_mb * 1024 * 1024) // (ARENA_WIDTH * 2) // 32 * 32
        arena = None
        if arena_rows:
            arena = self._arena_tensor(arena_rows)
            self.arena_address = self._addr(arena)
            self.arena_span = self._span(arena)
            ops.deallocate(arena)
        freed_all = []
        for bucket in self.settings.buckets:
            trace = self.traces[bucket]
            with self._hooks(model, trace):
                model._reset_gdn_state_for_new_sequence()
                with self._recording_frees() as freed:
                    trace.trace_id = ops.begin_trace_capture(model.device, cq_id=0)
                    try:
                        self._body(model, trace)
                    finally:
                        ops.end_trace_capture(model.device, trace.trace_id, cq_id=0)
            freed_all += freed
            model._reset_gdn_state_for_new_sequence()
            ops.synchronize_device(model.device)
        if arena_rows:
            arena = self._arena_tensor(arena_rows)
            if self._addr(arena) != self.arena_address:
                ops.deallocate(arena)
                raise ValueError('The arena came back at %s, not at %s: the regions the captures used are not covered, so a later allocation may sit in a hole a replay writes'
                                 % (self._addr(arena), self.arena_address))
            self.arena = arena
            bad = self._uncovered(freed_all)
            self.holes.update(recorded=len(freed_all), uncovered=len(bad))
            if bad:
                raise ValueError('%d of %d tensors the captures freed lie outside the arena (first: addresses %s, %d bytes a bank, arena %s + %d): the allocator had free gaps below it that '
                                 'a capture used, and a later allocation may sit in one of them' % (len(bad), len(freed_all), bad[0][0], bad[0][1], self.arena_address, self.arena_span))
        elif freed_all:
            self.holes.update(recorded=len(freed_all), uncovered=len(freed_all))

    # -- the hooks of the traced body (installed only while a pass of the body runs) --------------------------------------------------------
    @contextlib.contextmanager
    def _hooks(self, model, trace):
        """The gate and conv-state hooks, on while ONE pass of the body runs (compile or capture), off again after."""
        layers = self._gdn_layers()
        module = sys.modules[type(layers[0]).__module__]
        if not callable(getattr(module, '_qwen_prefill_conv', None)):
            raise ValueError('%s has no _qwen_prefill_conv: the conv state fix has nothing to wrap (the graft conv op is not in this model tree)' % type(layers[0]).__module__)
        self._active = trace
        try:
            for dn in layers:
                self._set(dn, '_project_qkvzab', self._project_hook(dn._project_qkvzab, trace))
            self._set(module, '_qwen_prefill_conv', self._conv_hook(module._qwen_prefill_conv, trace))
            yield self
        finally:
            self._active = None
            self._flush_hooks()

    def _flush_hooks(self):
        """Take the pass hooks off (the install-time shadows stay)."""
        keep = [item for item in self._saved if item[1] in ('_forward_prefill_chunk_masked_tp', 'forward')]
        hooks = [item for item in self._saved if item not in keep]
        for instance, name, had, value in reversed(hooks):
            if had:
                setattr(instance, name, value)
            else:
                vars(instance).pop(name, None)
        self._saved = keep

    def _project_hook(self, original, trace):
        tail = self

        def project(x, S, out_mc=None):
            qkv, z, a, b = original(x, S, out_mc=out_mc)
            if S != trace.bucket:
                return qkv, z, a, b
            ops = tail._ops()
            tail.hook_calls['project'] += 1
            a2 = ops.add(a, trace.pad_mask, memory_config=a.memory_config())
            ops.deallocate(a)
            b2 = ops.add(b, trace.pad_mask, memory_config=b.memory_config())
            ops.deallocate(b)
            return qkv, z, a2, b2
        return project

    def _conv_hook(self, original, trace):
        tail = self

        def conv(layer, qkv, cstate, valid_len, T):
            q, k, v, state = original(layer, qkv, cstate, valid_len, T)
            if T != trace.bucket or cstate is None:
                return q, k, v, state
            ops = tail._ops()
            tail.hook_calls['conv'] += 1
            config = ops.init_device_compute_kernel_config(layer.mesh.arch(), math_fidelity=ops.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
            from_x = ops.matmul(trace.sel_x, qkv, memory_config=ops.DRAM_MEMORY_CONFIG, compute_kernel_config=config)
            from_c = ops.matmul(trace.sel_c, cstate, memory_config=ops.DRAM_MEMORY_CONFIG, compute_kernel_config=config)
            fixed = ops.add(from_x, from_c, memory_config=ops.DRAM_MEMORY_CONFIG)
            ops.deallocate(from_x)
            ops.deallocate(from_c)
            ops.deallocate(state)
            return q, k, v, fixed
        return conv

    # -- the shadow of the masked chunk forward -----------------------------------------------------------------------------------------------
    def decide(self, valid_len, chunk_start, page_table, bucket, flex_sdpa, vision_tokens):
        """(TailPlan, None) when this call is replayed, (None, reason) when it runs the stock pass."""
        if not self.settings.traced:
            return None, 'control'
        if self.latched is not None:
            return None, 'latched'
        if not self.warmed:
            return None, 'cold'
        if bucket not in self.traces or valid_len >= 2048:
            return None, 'bucket'
        if vision_tokens is not None:
            return None, 'vision'
        if not flex_sdpa:
            return None, 'flex'
        capture = getattr(self.model, '_qwen_target_feature_capture', None)
        if capture is not None and tuple(capture.layer_ids) != self.tap_ids:
            return None, 'taps'
        try:
            row = page_table[0].tolist()
        except Exception:
            return None, 'page_table'
        plan, reason = plan_tail(valid_len, chunk_start, self.width, row, self.sink, self.settings.buckets)
        if plan is not None and plan.bucket != bucket:
            return None, 'bucket'
        return plan, reason

    def shadow(self, token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=True, vision_tokens=None):
        """Replacement for Qwen36Model._forward_prefill_chunk_masked_tp: a replay when the call is a tail the lever traces, the stock pass otherwise."""
        began = self.clock()
        plan, reason = self.decide(valid_len, chunk_start, page_table, bucket, flex_sdpa, vision_tokens)
        timed = self.settings.present and bucket in TRACEABLE
        if plan is None:
            if reason not in ('control', 'bucket') or (reason == 'bucket' and timed):
                self._skip(reason)
            hidden = self.original(token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=flex_sdpa, vision_tokens=vision_tokens)
            if timed:
                self._ops().synchronize_device(self.model.device)
                self.eager_calls += 1
                self.log('{}', '%s mode=eager bucket=%d valid=%d start=%d ms=%.1f' % (TIMING_MARKER, bucket, valid_len, chunk_start, (self.clock() - began) * 1000.0))
            return hidden
        index = self.traced_calls
        self.traced_calls += 1
        self.calls += 1
        nth = self.bucket_calls[bucket] = self.bucket_calls.get(bucket, 0) + 1
        if self.settings.audit and (audit_due(index) or nth <= AUDIT_PER_BUCKET):
            return self._audited(plan, token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa, vision_tokens, began)
        try:
            staged = self._stage(plan, token_buf, page_table)
        except Exception as error:
            self._latch('staging: %s: %s' % (type(error).__name__, error))
            return self.shadow(token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=flex_sdpa, vision_tokens=vision_tokens)
        trace = self.traces[bucket]
        trace.host_refs = staged
        staged_at = self.clock()
        self._ops().execute_trace(self.model.device, trace.trace_id, cq_id=0, blocking=False)
        self._answer_taps(trace)
        self._ops().synchronize_device(self.model.device)
        done = self.clock()
        self.log('{}', '%s mode=traced bucket=%d valid=%d start=%d ms=%.1f stage_ms=%.1f replay_ms=%.1f' % (
            TIMING_MARKER, bucket, valid_len, chunk_start, (done - began) * 1000.0, (staged_at - began) * 1000.0, (done - staged_at) * 1000.0))
        return trace.out

    def _skip(self, reason):
        first = reason not in self.skips
        self.skips[reason] = self.skips.get(reason, 0) + 1
        if first and self.settings.traced:
            self.log('{}', '%s reason=%s' % (SKIP_MARKER, reason))

    def _latch(self, reason):
        self.latched = reason
        self.fallbacks += 1
        self.log('{}', '%s reason=%s' % (FELL_BACK_MARKER, reason))

    def _stage(self, plan, token_buf, page_table):
        """Host tensors for every per-call buffer, copied to the device (the persistent buffers' addresses never change). Returns the host tensors, which must outlive the copies."""
        import torch

        ops = self._ops()
        model = self.model
        trace = self.traces[plan.bucket]
        cos_t, sin_t = model._rope_tp_cos_sin_torch(plan.chunk_start, plan.bucket)
        nv = trace.nv
        host = [
            (self._host_tensor(token_buf.to(torch.int32), ops.uint32, ops.ROW_MAJOR_LAYOUT), trace.tok),
            (self._host_tensor(cos_t, ops.bfloat16, ops.TILE_LAYOUT), trace.cos),
            (self._host_tensor(sin_t, ops.bfloat16, ops.TILE_LAYOUT), trace.sin),
            (self._host_tensor(torch.tensor([plan.chunk_start], dtype=torch.int32), ops.int32, ops.ROW_MAJOR_LAYOUT), trace.csi),
            (self._host_tensor(page_table.to(torch.int32), ops.int32, ops.ROW_MAJOR_LAYOUT), trace.full_pt),
            (self._host_tensor(torch.tensor([plan.chunk_row], dtype=torch.int32), ops.int32, ops.ROW_MAJOR_LAYOUT), trace.chunk_pt),
            (self._host_tensor(*self._pad_mask_host(plan.valid_len, plan.bucket, nv)), trace.pad_mask),
            (self._host_tensor(*self._sel_x_host(plan.valid_len, plan.bucket)), trace.sel_x),
            (self._host_tensor(*self._sel_c_host(plan.valid_len)), trace.sel_c),
        ]
        # The vision buffers were staged by the caller (prefill_masked_bucket calls _set_vision_merge before the forward), so a text-only tail has cleared the mask the traced where reads.
        for source, target in host:
            ops.copy_host_to_device_tensor(source, target)
        return [source for source, _target in host]

    def _answer_taps(self, trace):
        """When a drafter feature capture is open, call each tap layer's forward once, in layer order: its shim answers from the retained output, and the capture takes its snapshot."""
        if getattr(self.model, '_qwen_target_feature_capture', None) is None:
            return
        self._replaying = trace.taps
        try:
            for index in self.tap_ids:
                self.model.layers[index].forward(None)
        finally:
            self._replaying = None

    # -- the audit -------------------------------------------------------------------------------------------------------------------------
    def _snapshot(self):
        ops = self._ops()
        comp = ops.ConcatMeshToTensor(self.model.mesh_device, dim=0)
        return [(ops.to_torch(dn.rec_state, mesh_composer=comp), ops.to_torch(dn.conv_carry, mesh_composer=comp),
                 [ops.to_torch(conv, mesh_composer=comp) for conv in dn.conv_states]) for dn in self._gdn_layers()]

    def _audited(self, plan, token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa, vision_tokens, began):
        model, ops = self.model, self._ops()
        trace = self.traces[bucket]
        before = model._qwen_prefix_read_scratch()
        trace.host_refs = self._stage(plan, token_buf, page_table)
        ops.execute_trace(model.device, trace.trace_id, cq_id=0, blocking=False)
        ops.synchronize_device(model.device)
        traced_state = self._snapshot()
        traced_hidden = self._host_rows(trace.out, valid_len)
        traced_taps = dict((index, self._host_rows(tap, valid_len)) for index, tap in trace.taps.items())
        model._qwen_prefix_restore(before[0], before[1])
        self._recording = {'valid': valid_len}
        try:
            hidden = self.original(token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=flex_sdpa, vision_tokens=vision_tokens)
        finally:
            recorded, self._recording = self._recording, None
        ops.synchronize_device(model.device)
        stock_state = self._snapshot()
        stock_hidden = self._host_rows(hidden, valid_len)
        problems = []
        for li, (a, b) in enumerate(zip(traced_state, stock_state)):
            if _bytes(a[0]) != _bytes(b[0]):
                problems.append('layer %d recurrent state' % li)
            if _bytes(a[1]) != _bytes(b[1]):
                problems.append('layer %d conv carry' % li)
            for tap, (x, y) in enumerate(zip(a[2], b[2])):
                if _bytes(x) != _bytes(y):
                    problems.append('layer %d conv window %d' % (li, tap))
        if _bytes(traced_hidden) != _bytes(stock_hidden):
            problems.append('the output rows')
        for index in self.tap_ids:
            if index in recorded and _bytes(traced_taps[index]) != _bytes(recorded[index]):
                problems.append('tap layer %d rows' % index)
        self.audited += 1
        if problems:
            self.mismatches += 1
            self.log('{}', '%s n=%d exact=False bucket=%d valid=%d problems=%d first=%s' % (AUDIT_MISMATCH_MARKER, self.audited, bucket, valid_len, len(problems), problems[0]))
            raise AssertionError('%s: %s' % (AUDIT_MISMATCH_MARKER, '; '.join(problems[:3])))
        self.log('{}', '%s n=%d exact=True bucket=%d valid=%d layers=%d taps=%d' % (AUDIT_MARKER, self.audited, bucket, valid_len, len(traced_state), len(traced_taps)))
        self.log('{}', '%s mode=audited bucket=%d valid=%d start=%d ms=%.1f' % (TIMING_MARKER, bucket, valid_len, chunk_start, (self.clock() - began) * 1000.0))
        return hidden
