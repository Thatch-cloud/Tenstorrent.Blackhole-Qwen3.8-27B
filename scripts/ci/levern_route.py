"""Lever N at TP4: the model route that continues its own suspended prefill scratch (QWEN_FAST_LEVER_N=1), and the digest instrument
(QWEN_FAST_LEVERN_AUDIT=1, which also stands alone as the NON-interleaved control).

WHY A ROUTE. The image's model (qwen36 model.py after the G1 stage, which every C2 build applies) has the resumable chunk loops and
prefill_traced_chunked(start=...), but the entry a request reaches today, Qwen36Model.prefill_paged_slots, ignores the runner's start_pos,
and the G1 route (_qwen_prefix_prefill_slots) accepts start_pos > 0 only with a prefix grant. A prefill that arrives in several engine steps
needs a route that, at start > 0, continues the B=1 GDN scratch the previous step left (fp32 recurrent state and the conv carry, in place
between steps) and reads the earlier chunks' K/V from the paged cache. This module is that route, installed at the attach as instance
attributes (nothing in the plugin or the model tree is edited, so no sha256 pin moves):

    model.prefill_paged_slots_range(token_ids_list, page_table, empty_slots, starts, ends, valid_lens=None, step=...)
        the entry name Lever N M1 chose; dflash_prefill_window.PrefillWindowCapture already wraps it (BATCHED_PREFILL_ENTRIES) and
        its ledger already demands starts == cursor on every call after the first, so a foreign or replayed call is refused there too
    wrapper.prefill_forward(...)
        the vLLM wrapper's entry. It sends a step the lifecycle announced to the route when the step is part of a split prompt, and
        every other step (a prompt that runs whole, the warm requests, every step when nothing was announced) to the original, untouched.

THE ANNOUNCEMENT. The plugin runner hands the model start_pos and prompt_lens (the chunk's end) but no request id and not the prompt's
full length, and the id is the owner token's. The lifecycle knows all of them before it calls the worker (a new request's NewRequestData
and the capture's position), so it sets model._qwen_levern_step = Step(req_id, start, end, total) for the one execute_model call and
withdraws it in a finally. The wrapper pops it; the route checks it against what the runner passed (start_pos, prompt_lens, one row) before
any device work.

EXACTNESS (design 3.4). A step ends on a multiple of 2,048 unless it is the final one, so the route runs prefill_traced_chunked(tokens[:end],
actual_len=end, start=start): the model's own chunk loop, from chunk start/2048, with the GDN reset only at start 0. The same programs run on
the same bytes in the same order as the whole prompt's loop, whose chunks are already separated by a device synchronize that keeps the
scratch and reads K/V from the pages. The route adds one rule the whole path never needed, the OWNER TOKEN: the scratch is single-occupancy and a
wrong continuation corrupts without an error (the v73 to v80 class), so the route continues only when owner == (request id, start) and
raises before any device work otherwise. A start-0 step takes a new owner (the scheduler admits one prefill at a time); the lifecycle
clears the owner when the request is finished or aborted mid-prefill (release).

WHAT IT SKIPS. A step that is not the last runs no host snapshot of the scratch (154 MB across four chips) and no decode-slot write, and says so
(wrote_slot=0): the decoders' working row 0 is not touched, so the lifecycle does not displace the resident engine for it. The final step snapshots and
writes the slot exactly as prefill_paged_slots does (the same _write_gdn_slot), so the engine the lifecycle then builds adopts the same state.

THE AUDIT. With QWEN_FAST_LEVERN_AUDIT=1 the end of every prefill (the whole-prompt path in the control arm, the final step in the interleaved
arm) logs levern_policy.DIGEST_LINE: sha256 digests (32 hex) of the prompt's GDN slot (every GDN layer's recurrent state, then its conv
states, read from the B=1 scratch in the same order in both arms), of the last-position logits the entry returns, and of the prompt's KV pages
[0, P) (one cache tensor at a time, so host memory stays one tensor). levern_compare.py requires the two arms' digests for the same prompt equal.

THE WARM. warm(...) runs the route once, before any trace is captured, over a 6,208-token prompt in three steps (an aligned first step, an
aligned middle step, a final step with a tail), so every program and buffer the route touches exists before the packed traces and the route's own
program growth is read where it cannot hang the engine (#48536); levern_policy.ROUTE_WARM_LINE.

THE MERGED ROUTE (docs/lever-n-prefix-merged-route.md; QWEN_FAST_LEVER_N=1 beside QWEN_PREFIX_REUSE=1 and QWEN_FAST_STICKY_SESSIONS=1). start_pos > 0
then has ONE meaning: before the step's first chunk the B=1 scratch must hold the GDN state after exactly `start` tokens of this request's prompt,
and every step names where that state comes from (SOURCES):
    COLD         start 0; the chunk loop's own reset.
    CHECKPOINT   a prefix hit's first step: the registry's committed grant at Q == start, checked by the staged model's own row check
                 (_qwen_prefix_row: the grant's request, Q, the checkpoint's token ids and GDN spec) and restored (_qwen_prefix_restore).
    SCRATCH      the request's own suspended scratch: owner == (request, start).
    PARKED       the request's own state taken to the host while a short prefill ran (admission v2) and restored (_qwen_prefix_restore).
The lifecycle announces the source in Step.source; the route resolves it again from its own evidence (the registry's grant, the owner token, the
park table) and refuses before any device work when the two disagree or when no source holds. The route only calls the model methods the G1
stage already puts into every C2 image (prefill_traced_chunked(start, capture_at, on_capture), _qwen_prefix_restore, _qwen_prefix_read_scratch,
_qwen_prefix_capture) and the module functions it reaches through sys.modules[type(model).__module__] (_qwen_prefix_row, _qwen_prefix_registry,
_qwen_prefix_declare_mid_loop, _qwen_prefix_dram, _qwen_prefix_restored): no sha256 pin moves.

CHECKPOINTS FROM A SPLIT PREFILL. Each step passes capture_at = the request's planned positions inside (start, end] (the grant's plan on its first
step, the registry's in-flight plan on every later one, qwen_prefix_registry.Inflight), so a split 254k prompt publishes the same checkpoints as a
whole one. A step that is not the last takes no host snapshot of the scratch and writes no slot (wrote_slot=0), exactly as stage 1; after its unbind
it checks that every GDN layer is bound to the batched decode buffers it had before (IDENTITY_ATTR), which the fixture-epoch scope needs.

THE ROW MARKER. The final step of a prompt logs one '[PREFIX] row=' line in the G1 route's format (Q = the first step's start, L = the prompt, the
plan, captured aggregated over every step, the restore time of the first step, the program counts over the request), so prefix_markers,
prefix_judge and c2_prefix_gate read the merged route unchanged.
"""

import hashlib
import inspect
import os
import sys
import time
import types
from collections import namedtuple

import levern_policy

ENTRY = 'prefill_paged_slots_range'
STEP_ATTR = '_qwen_levern_step'
OWNER_ATTR = '_qwen_levern_owner'
# Read by dflash_prefill_window.PrefillWindowCapture.wrap_slots after every batched prefill call (the literal is pinned equal there).
WROTE_ATTR = '_qwen_levern_wrote_slot'
HANDLE_ATTR = '_qwen_levern_handle'
WARM_REQUEST = '__levern_warm__'
WARM_PROMPT = 3 * levern_policy.CHUNK + 64
# What the route needs from the model: prefill_traced_chunked takes start, and the chunk loops are resumable (the G1 stage's edits).
REQUIRED = (('prefill_traced_chunked', ('start',)), ('_prefill_chunked_eager_tp', ('chunk_from', 'chunk_to', 'do_reset')),
            ('_prefill_traced_chunked_tp', ('chunk_from', 'chunk_to', 'do_reset')))
NEEDED_METHODS = ('_bind_gdn_prefill_scratch', '_unbind_gdn_prefill_scratch', '_write_gdn_slot')
# The merged route (the module docstring). SOURCES is the capability the contract reads at boot (serving_c2_contract.levern_problems): an image
# whose route does not carry all four cannot boot a profile that turns Lever N and prefix reuse on together, and it is not a flag.
SOURCES = ('COLD', 'CHECKPOINT', 'SCRATCH', 'PARKED')
# Set on the model while the merged route is installed (dflash_prefill_window.LEVERN_MERGED_ATTR, pinned equal by test_levern_prefix_route).
MERGED_ATTR = '_qwen_levern_merged'
# True after a step whose unbind found every GDN layer bound to the batched decode buffers it had before the step (the fixture-epoch scope).
IDENTITY_ATTR = '_qwen_levern_identity'
# What the merged route needs beyond stage 1: the G1 stage's restore, scratch read and capture methods, and its module functions.
MERGED_METHODS = ('_qwen_prefix_restore', '_qwen_prefix_read_scratch', '_qwen_prefix_capture')
MERGED_FUNCTIONS = ('_qwen_prefix_row', '_qwen_prefix_registry', '_qwen_prefix_declare_mid_loop', '_qwen_prefix_dram',
                    '_qwen_prefix_restored')
PREFIX_ENV = 'QWEN_PREFIX_REUSE'
STICKY_ENV = 'QWEN_FAST_STICKY_SESSIONS'
# The warm's merged prompt: [0, 2048) capturing at 2048, [2048, 4096), a CHECKPOINT restore of the 2048 state and [2048, 4096) again, the final
# step [4096, 6208), and under host parking a park and an unpark around that final step.
MERGED_WARM_STEPS = 4

Step = namedtuple('Step', 'req_id start end total source park_owner', defaults=(None, False))
Park = namedtuple('Park', 'start rec carry nbytes at')


def final(step):
    """Whether `step` ends its prompt."""
    return step.end == step.total


def whole(step):
    """Whether `step` is the whole prompt (the stock path runs it)."""
    return step.start == 0 and step.end == step.total


def merged_requested(environ=None):
    """Whether the environment asks for the merged route: Lever N beside prefix reuse (strictly '1'). Prefix reuse without sticky sessions is
    refused here too (the contract refuses the profile; this is the attach's own check)."""
    environ = os.environ if environ is None else environ
    if environ.get(levern_policy.FLAG) != '1' or environ.get(PREFIX_ENV) != '1':
        return False
    if environ.get(STICKY_ENV) != '1':
        raise ValueError('%s=1 and %s=1 need %s=1: the merged route is the sticky-session route' % (levern_policy.FLAG, PREFIX_ENV, STICKY_ENV))
    return True


def mode(environ=None):
    """(route, audit): QWEN_FAST_LEVER_N and QWEN_FAST_LEVERN_AUDIT, strictly."""
    environ = os.environ if environ is None else environ
    return levern_policy.enabled(environ), levern_policy.audit_enabled(environ)


def announce(model, step):
    """The lifecycle's announcement of the step the next execute_model call carries."""
    if not isinstance(step, Step):
        raise TypeError('a levern Step is required, got %r' % (step,))
    setattr(model, STEP_ATTR, step)


def withdraw(model):
    """Take the announcement back (a step that failed before the wrapper consumed it must not leak into the next)."""
    vars(model).pop(STEP_ATTR, None)


def _take(model):
    return vars(model).pop(STEP_ATTR, None)


def owner(model):
    return vars(model).get(OWNER_ATTR)


def release(model, request_id):
    """The lifecycle's call when a request is finished or aborted mid-prefill: its suspended scratch, and its parked state, are nobody's any more."""
    released = False
    held = vars(model).get(OWNER_ATTR)
    if held is not None and held[0] == request_id:
        del vars(model)[OWNER_ATTR]
        released = True
    handle = vars(model).get(HANDLE_ATTR)
    if handle is not None:
        if handle.parks.pop(request_id, None) is not None:
            released = True
        handle.rows.pop(request_id, None)
    if released:
        _publish(model)
    return released


def _publish(model):
    """Write the owner and the parks into the shared state the scheduler reads (levern_policy.state_holder)."""
    holder = levern_policy.state_holder()
    handle = vars(model).get(HANDLE_ATTR)
    holder.owner = vars(model).get(OWNER_ATTR)
    holder.parks = {} if handle is None else {request_id: park.start for request_id, park in handle.parks.items()}
    holder.route = handle is not None and handle.route


def _set_owner(model, value):
    vars(model)[OWNER_ATTR] = value
    _publish(model)


def _clear_owner(model):
    vars(model).pop(OWNER_ATTR, None)
    _publish(model)


def _ttnn():
    import ttnn

    return ttnn


def _log(message, *values):
    try:
        from loguru import logger
    except ImportError:
        print(message.format(*values), flush=True)
        return
    logger.info(message, *values)


def program_count(model):
    try:
        return int(model.mesh_device.num_program_cache_entries())
    except Exception:
        return None


def _bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(_uint8()).numpy().tobytes()


def _uint8():
    import torch

    return torch.uint8


def tokens_sha(tokens, total):
    """sha256 (32 hex) of the first `total` token ids of row 0 as int32: the prompt's identity in a digest line."""
    import torch

    return hashlib.sha256(_bytes(tokens[0:1, :total].to(torch.int32))).hexdigest()[:32]


def window_programs():
    """The drafter-window snapshots' program count so far (0 when the counter is not installed or the module is absent)."""
    try:
        import dflash_prefill_window

        return int(dflash_prefill_window.window_programs())
    except Exception:
        return 0


class _SlotWrites(object):
    """Counts the real _write_gdn_slot calls the model makes while it is entered (an instance attribute shadowing the method)."""

    def __init__(self, model):
        self.model, self.count = model, 0

    def __enter__(self):
        model = self.model
        self.had = '_write_gdn_slot' in vars(model)
        self.saved = vars(model).get('_write_gdn_slot')
        method = model._write_gdn_slot

        def counted(*args, **kwargs):
            self.count += 1
            return method(*args, **kwargs)
        vars(model)['_write_gdn_slot'] = counted
        return self

    def __exit__(self, *exc):
        if self.had:
            vars(self.model)['_write_gdn_slot'] = self.saved
        else:
            vars(self.model).pop('_write_gdn_slot', None)
        return False


def digests(rec_snap, conv_snap, logits):
    """(slot_sha, logits_sha): 32 hex digits each, over every GDN layer's recurrent state then its conv states, and over the logits."""
    gdn = hashlib.sha256()
    for rec in rec_snap:
        gdn.update(_bytes(rec))
    for convs in conv_snap:
        for conv in convs:
            gdn.update(_bytes(conv))
    return gdn.hexdigest()[:32], hashlib.sha256(_bytes(logits)).hexdigest()[:32]


def kv_digest(model, page_row, length):
    """sha256 (32 hex) over the unpacked K/V values of positions [0, length) of one request, in cache order, read one cache tensor at a time."""
    import torch

    ttnn = _ttnn()
    heads = ttnn.ConcatMeshToTensor(model.mesh_device, dim=1)
    blocks = torch.as_tensor(page_row, dtype=torch.long).reshape(-1)
    digest = hashlib.sha256()
    for k_cache, v_cache in model._paged_kv_caches:
        for cache in (k_cache, v_cache):
            host = ttnn.to_torch(cache, mesh_composer=heads)
            size = int(host.shape[2])
            needed = -(-length // size)
            sel = host.index_select(0, blocks[:needed])
            del host
            seq = sel.permute(1, 0, 2, 3).reshape(sel.shape[1], needed * size, sel.shape[3])[:, :length]
            digest.update(_bytes(seq))
            del sel, seq
    return digest.hexdigest()[:32]


def scratch_snapshot(model):
    """Host copies of the B=1 prefill scratch's GDN state, as prefill_paged_slots snapshots it for the slot write: per GDN layer the recurrent
    state, and per layer the list of conv states. The scratch is bound for the read and unbound after (the batched decode buffers are
    never touched)."""
    ttnn = _ttnn()
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
    previous = model._bind_gdn_prefill_scratch()
    try:
        return _read_scratch(ttnn, comp, layers)
    finally:
        model._unbind_gdn_prefill_scratch(previous)


def _read_scratch(ttnn, comp, layers):
    return ([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in layers],
            [[ttnn.to_torch(conv, mesh_composer=comp) for conv in dn.conv_states] for dn in layers])


def audit_prefill(model, step, logits, page_row, rec_snap=None, conv_snap=None, log=_log, tokens=None):
    """Log the digests of a finished prefill (levern_policy.DIGEST_LINE); `rec_snap`/`conv_snap` are the scratch's host copies when the
    caller already holds them (the final step), else the scratch is read here (the whole-prompt path)."""
    if rec_snap is None:
        rec_snap, conv_snap = scratch_snapshot(model)
    slot_sha, logits_sha = digests(rec_snap, conv_snap, logits)
    kv = kv_digest(model, page_row, step.total) if step.total <= levern_policy.KV_DIGEST_MAX_PROMPT else levern_policy.KV_SKIPPED
    log(levern_policy.DIGEST_LINE, step.req_id, step.total, tokens_sha(tokens, step.total) if tokens is not None else levern_policy.KV_SKIPPED,
        slot_sha, logits_sha, kv)


class Handle(object):
    """What install() returns: the instance attributes it set, so the attach's scope can take them off again, and the merged route's state (the
    parked scratches on the host, the per-request row aggregates for the final marker, and the counters a gate reads)."""

    def __init__(self, wrapper, model, original, route, audit, merged=False, merged_cfg=None):
        self.wrapper, self.model, self.original, self.route, self.audit = wrapper, model, original, route, audit
        self.merged, self.merged_cfg = merged, merged_cfg
        self.fault_fired = False
        self.owner_refusals = 0
        self.parks = {}
        self.rows = {}
        self.parked_total = 0
        self.unparked_total = 0
        self.source_refusals = 0
        self.park_failures = 0

    def uninstall(self):
        if self.wrapper.__dict__.get('prefill_forward') is not None:
            del self.wrapper.__dict__['prefill_forward']
        vars(self.model).pop(ENTRY, None)
        for name in (STEP_ATTR, OWNER_ATTR, WROTE_ATTR, HANDLE_ATTR, MERGED_ATTR, IDENTITY_ATTR):
            vars(self.model).pop(name, None)
        self.parks.clear()
        self.rows.clear()
        levern_policy.reset_state()


def requirement_problems(model, merged=False):
    """What the model lacks for the route, one string each: the G1 stage's resumable loops and the scratch helpers; with `merged`, also its
    restore, scratch read and capture methods, its module functions, and the absence of a chunk-input page-table buffer and chunk trace (the
    merged route passes the runner's page table at its own width and never fits it to a buffer, which would be a compile after the traces)."""
    problems = []
    for name, parameters in REQUIRED:
        method = getattr(model, name, None)
        if not callable(method):
            problems.append('no %s' % name)
            continue
        try:
            accepted = inspect.signature(method).parameters
        except (TypeError, ValueError):
            problems.append('%s has no readable signature' % name)
            continue
        missing = [parameter for parameter in parameters if parameter not in accepted]
        if missing:
            problems.append('%s lacks %s (the G1 stage\'s resumable loops are not in this model tree)' % (name, ', '.join(missing)))
    problems += ['no %s' % name for name in NEEDED_METHODS if not callable(getattr(model, name, None))]
    if merged:
        try:
            accepted = inspect.signature(model.prefill_traced_chunked).parameters
            problems += ['prefill_traced_chunked lacks %s (the G1 stage\'s capture hook)' % name
                         for name in ('capture_at', 'on_capture') if name not in accepted]
        except (AttributeError, TypeError, ValueError):
            pass
        problems += ['no %s (the G1 model graft)' % name for name in MERGED_METHODS if not callable(getattr(model, name, None))]
        module = sys.modules.get(type(model).__module__)
        problems += ['no %s in %s (the G1 model graft)' % (name, type(model).__module__)
                     for name in MERGED_FUNCTIONS if not callable(getattr(module, name, None))]
        if getattr(model, '_chunk_full_page_table_buf', None) is not None:
            problems.append('a chunk-input page-table buffer exists: the merged route keeps the runner\'s page-table width and a buffer of any '
                            'width would make a hit compile after the traces (#48536)')
        if getattr(model, '_chunked_trace_id', None) is not None:
            problems.append('a chunk trace is captured: the merged route runs the eager chunk loop (trace_mode decode_only)')
    return problems


def install(runner, model, *, environ=None, log=_log):
    """Install the route and the digest instrument under QWEN_FAST_LEVER_N=1 / QWEN_FAST_LEVERN_AUDIT=1; None when neither is set.

    Refuses (ValueError, failing the attach) a model without what the route needs, a second install, and a wrapper without prefill_forward.
    Beside QWEN_PREFIX_REUSE=1 and QWEN_FAST_STICKY_SESSIONS=1 it installs the MERGED route (the module docstring)."""
    environ = os.environ if environ is None else environ
    route_on, audit_on = mode(environ)
    if not route_on and not audit_on:
        return None
    if vars(model).get(HANDLE_ATTR) is not None:
        raise ValueError('Lever N is already installed on this model')
    wrapper = runner.model
    original = getattr(wrapper, 'prefill_forward', None)
    if not callable(original):
        raise ValueError('The vLLM model wrapper has no prefill_forward: Lever N cannot route a chunked step')
    merged = route_on and merged_requested(environ)
    if route_on:
        problems = requirement_problems(model, merged=merged)
        if problems:
            raise ValueError('Lever N cannot be installed on this model tree: %s' % '; '.join(problems))
    handle = Handle(wrapper, model, original, route_on, audit_on, merged=merged,
                    merged_cfg=levern_policy.merged_config(environ) if merged else None)
    fault = levern_policy.fault(environ) if route_on else None

    def prefill_forward(*args, **kwargs):
        step = _take(model)
        if merged:
            if step is None:
                starts = kwargs.get('start_pos')
                if starts is not None and any(int(value) > 0 for value in starts):
                    raise AssertionError('Lever N: a prefill step at start_pos=%r reached the model without a lifecycle '
                                         'announcement: nothing says whose state it would continue' % ([int(v) for v in starts],))
                if vars(model).get(OWNER_ATTR) is not None or handle.parks:
                    raise AssertionError('Lever N: an unannounced prefill reached the model while a suspended prefill holds the scratch (owner=%r '
                                         'parks=%r): it would reset that state' % (vars(model).get(OWNER_ATTR), sorted(handle.parks)))
                return original(*args, **kwargs)
            return run_chunk(model, step, _named(args, kwargs), handle, fault, audit_on, log)
        if step is None or whole(step):
            starts = kwargs.get('start_pos')
            if step is None and starts is not None and any(int(value) > 0 for value in starts):
                raise AssertionError('Lever N: a prefill step at start_pos=%r reached the model without a lifecycle '
                                     'announcement: nothing says whose scratch it would continue' % ([int(v) for v in starts],))
            # A prompt that runs whole (and any request the lifecycle did not announce: the warm requests) is the stock path, and it
            # resets the B=1 scratch at start 0: whatever suspended prompt owned it is gone, and its continuation must be refused.
            _clear_owner(model)
        if step is None:
            return original(*args, **kwargs)
        if whole(step):
            result = original(*args, **kwargs)
            if audit_on:
                audit_prefill(model, step, result[0], _page_row(_named(args, kwargs)), log=log, tokens=_named(args, kwargs).get('tokens'))
            return result
        if not route_on:
            raise AssertionError('Lever N: a step of a split prompt reached a process without %s=1: %r' % (levern_policy.FLAG, step))
        return run_chunk(model, step, _named(args, kwargs), handle, fault, audit_on, log)

    if route_on:
        vars(model)[ENTRY] = types.MethodType(
            lambda self, *a, **k: route(self, *a, handle=handle, fault=fault, audit=audit_on, log=log, **k), model)
    vars(wrapper)['prefill_forward'] = prefill_forward
    vars(model)[HANDLE_ATTR] = handle
    if merged:
        vars(model)[MERGED_ATTR] = True
    levern_policy.reset_state()
    _publish(model)
    log(levern_policy.ROUTE_INSTALLED_LINE, 'route=%d audit=%d' % (route_on, audit_on) + (' merged=%s' % ','.join(SOURCES) if merged else ''))
    return handle


PREFILL_FORWARD_ARGS = ('tokens', 'page_table', 'kv_cache', 'prompt_lens')


def _named(args, kwargs):
    """The step's arguments by name: the plugin runner calls prefill_forward with keywords, a test or another caller may pass the first four
    positionally, as the vLLM wrapper's own signature allows."""
    if len(args) > len(PREFILL_FORWARD_ARGS):
        raise AssertionError('Lever N: prefill_forward takes %d positional arguments, got %d' % (len(PREFILL_FORWARD_ARGS), len(args)))
    named = dict(zip(PREFILL_FORWARD_ARGS, args))
    duplicated = sorted(set(named) & set(kwargs))
    if duplicated:
        raise AssertionError('Lever N: prefill_forward got %s both positionally and by keyword' % ', '.join(duplicated))
    named.update(kwargs)
    return named


def _page_row(kwargs):
    table = kwargs.get('page_table')
    if table is None:
        raise AssertionError('Lever N: the prefill step carried no page_table')
    if hasattr(table, 'reshape') and not isinstance(table, (list, tuple)):
        return table[0] if table.dim() > 1 else table
    return table[0]


def run_chunk(model, step, kwargs, handle, fault, audit, log):
    """One step of a split prompt, through the wrapper: what _prefill_forward_tp_batched does for the stock entry, calling the route."""
    import torch

    for name in ('tokens', 'page_table', 'prompt_lens', 'empty_slots'):
        if kwargs.get(name) is None:
            raise AssertionError('Lever N: the chunked prefill step carries no %s' % name)
    tokens, table, lengths, slots = kwargs['tokens'], kwargs['page_table'], kwargs['prompt_lens'], kwargs['empty_slots']
    starts = kwargs.get('start_pos')
    if tokens.shape[0] != 1 or len(lengths) != 1 or len(slots) != 1:
        raise AssertionError('Lever N: a split prompt runs one request per step, got %d rows' % tokens.shape[0])
    if starts is None or int(starts[0]) != step.start or int(lengths[0]) != step.end:
        raise AssertionError('Lever N: the runner\'s step (start_pos=%r, prompt_lens=%r) is not the lifecycle\'s %r'
                             % (None if starts is None else [int(value) for value in starts], [int(value) for value in lengths], step))
    token_ids = tokens[0:1, :step.end].to(torch.int32)
    page = table if isinstance(table, torch.Tensor) else _ttnn().to_torch(table)
    host_logits = model.prefill_paged_slots_range([token_ids], page, [int(slots[0])], [step.start], [step.end],
                                                  valid_lens=[step.end], step=step)
    logits = torch.cat([value.reshape(1, 1, -1) for value in host_logits], dim=0)
    return logits, torch.zeros(1, dtype=torch.long)


def _gdn_layers(model):
    return [layer.attention for layer in model.layers if not layer.is_full_attention]


_MISSING = object()
_BOUND = ('B', 'rec_state', 'conv_states', 'conv_carry', '_zero_conv0', '_stable_state')


def _binding(layers):
    """The batched decode buffers every GDN layer is bound to, by identity: what a step must leave exactly as it found them (hazard H3). Every
    attribute _bind_gdn_prefill_scratch rebinds (B, rec_state, conv_states, conv_carry, _zero_conv0, _stable_state) and each ELEMENT of conv_states."""
    found = []
    for dn in layers:
        values = tuple(getattr(dn, name, _MISSING) for name in _BOUND)
        convs = getattr(dn, 'conv_states', None)
        try:
            elements = tuple(convs) if convs is not None else ()
        except TypeError:
            elements = ()
        found.append((dn, values, elements))
    return found


def _binding_holds(before):
    for dn, values, elements in before:
        for name, was in zip(_BOUND, values):
            now = getattr(dn, name, _MISSING)
            if now is not was and not (type(was) in (int, bool) and type(now) is type(was) and now == was):
                return False
        convs = getattr(dn, 'conv_states', None)
        try:
            now_elements = tuple(convs) if convs is not None else ()
        except TypeError:
            now_elements = ()
        if len(now_elements) != len(elements) or any(a is not b for a, b in zip(now_elements, elements)):
            return False
    return True


def route(model, token_ids_list, page_table, empty_slots, starts, ends, valid_lens=None, *, step=None, handle=None, fault=None,
          audit=False, log=_log):
    """prefill_paged_slots_range (the module docstring): one step of one prompt into the persistent B=1 scratch."""
    import torch

    ttnn = _ttnn()
    if step is None:
        raise AssertionError('Lever N: the route runs only a step the lifecycle announced')
    if model.num_devices <= 1:
        raise AssertionError('Lever N: the route is the TP (num_devices>1) path')
    if (len(token_ids_list) != 1 or len(empty_slots) != 1 or len(starts) != 1 or len(ends) != 1
            or (valid_lens is not None and len(valid_lens) != 1)):
        raise AssertionError('Lever N: one request, one slot, one start and one end per step')
    start, end = int(starts[0]), int(ends[0])
    tokens = token_ids_list[0]
    if tokens.shape[0] != 1 or int(valid_lens[0] if valid_lens is not None else tokens.shape[1]) != end or tokens.shape[1] < end:
        raise AssertionError('Lever N: token_ids must be [1, >= end=%d] with valid_lens == end' % end)
    if (start, end) != (step.start, step.end):
        raise AssertionError('Lever N: the call [%d, %d) is not the announced step %r' % (start, end, step))
    if not 0 <= start < end <= step.total or (end < step.total and end % levern_policy.CHUNK) or (start and start % levern_policy.CHUNK):
        raise AssertionError('Lever N: a step must start and (unless final) end on a %d-token boundary inside the prompt: %r'
                             % (levern_policy.CHUNK, step))
    if handle is not None and handle.merged:
        return _merged_route(model, tokens, page_table, int(empty_slots[0]), step, handle, fault, audit, log)
    held = vars(model).get(OWNER_ATTR)
    if start and fault == 'foreign' and handle is not None and not handle.fault_fired:
        handle.fault_fired = True
        held = ('__foreign__', start)
        _set_owner(model, held)
    if start and held != (step.req_id, start):
        if handle is not None:
            handle.owner_refusals += 1
        raise AssertionError('Lever N: request %r continues at %d but the prefill scratch is owned by %r: a chunk was skipped, '
                             'replayed or belongs to another prompt (refused before any device work)' % (step.req_id, start, held))
    vars(model).pop(WROTE_ATTR, None)
    page = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    layers = _gdn_layers(model)
    began, programs_before, window_before = time.perf_counter(), program_count(model), window_programs()
    writes = _SlotWrites(model)
    # A start-0 step takes a new owner at once, a continuation keeps its own; an error below leaves nobody owning a half-advanced scratch.
    _set_owner(model, (step.req_id, start))
    previous = model._bind_gdn_prefill_scratch()
    rec_snap = conv_snap = None
    with writes:
        try:
            lg = model.prefill_traced_chunked(tokens[:, :end], page[0:1], actual_len=end, start=start)
            host = (ttnn.to_torch(lg, mesh_composer=comp).reshape(-1, model.args.vocab_size)[:1].float().view(1, 1, -1))
            ttnn.deallocate(lg)
            if final(step):
                rec_snap, conv_snap = _read_scratch(ttnn, comp, layers)
        except BaseException:
            _clear_owner(model)
            raise
        finally:
            model._unbind_gdn_prefill_scratch(previous)
        # The window is read before the slot write so a program that write compiled is not forgiven as the window's.
        window = window_programs() - window_before
        if final(step):
            model._write_gdn_slot(int(empty_slots[0]), rec_snap, conv_snap)
            _clear_owner(model)
        else:
            _set_owner(model, (step.req_id, end))
    vars(model)[WROTE_ATTR] = writes.count > 0
    log(levern_policy.ROUTE_LINE, step.req_id, start, end, step.total, int(final(step)), writes.count,
        (time.perf_counter() - began) * 1000.0, programs_before, program_count(model), window)
    if audit and final(step):
        audit_prefill(model, step, host, page[0], rec_snap, conv_snap, log=log, tokens=tokens)
    return [host]


# ---------------------------------------------------------------------------------------------------------------------------------------
# The merged route
# ---------------------------------------------------------------------------------------------------------------------------------------

Resolved = namedtuple('Resolved', 'source restore plan grant')


def host_memory_ok(nbytes, environ=None):
    """Whether the host has `nbytes` (and the same again as headroom) available: what a park needs before the scheduler lets a short prefill take
    the scratch. Linux's MemAvailable; anywhere it cannot be read the answer is yes."""
    try:
        with open('/proc/meminfo', encoding='ascii') as handle:
            for line in handle:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024 >= 2 * int(nbytes)
    except (OSError, ValueError, IndexError):
        pass
    return True


def _plan_positions(registry, request_id, start, end, chunk_size):
    """The planned positions this step may capture: inside (start, end], on a chunk boundary the step's loop runs past or ends on."""
    if registry is None:
        return []
    last = end // chunk_size * chunk_size
    return [pos for pos in registry.planned(request_id) if start < pos <= last and pos % chunk_size == 0]


def _resolve(model, handle, step, tokens, registry, module, chunk_size):
    """Where the state this step starts from comes from (the module docstring), checked against the lifecycle's announcement. Every refusal is
    an AssertionError before any device work, and counted in handle.source_refusals."""
    request_id, start, end = step.req_id, step.start, step.end
    grant = registry.grant_for(request_id) if registry is not None else None
    held = vars(model).get(OWNER_ATTR)
    parked = handle.parks.get(request_id)
    where = 'Lever N: request %r step [%d, %d) of %d' % (request_id, start, end, step.total)
    refusal = None
    source = restore = None
    plan = []
    if start == 0:
        source = 'COLD'
        if grant is not None and int(grant.q) != 0:
            refusal = 'a committed grant at Q=%d beside a cold start' % int(grant.q)
    elif grant is not None and int(grant.q) == start:
        source = 'CHECKPOINT'
    elif held == (request_id, start):
        source = 'SCRATCH'
    elif parked is not None and parked.start == start:
        source = 'PARKED'
    else:
        refusal = ('no source holds the state after %d tokens: grant=%r owner=%r parked=%r'
                   % (start, None if grant is None else int(grant.q), held, None if parked is None else parked.start))
    if refusal is None and step.source is not None and step.source != source:
        refusal = 'the lifecycle announced %s and the route\'s own evidence says %s' % (step.source, source)
    if refusal is not None:
        handle.source_refusals += 1
        if source == 'SCRATCH' or source is None:
            handle.owner_refusals += 1
        raise AssertionError('%s: %s (refused before any device work)' % (where, refusal))
    if registry is not None:
        module._qwen_prefix_declare_mid_loop()
    if source in ('COLD', 'CHECKPOINT') and registry is not None and grant is not None:
        spec = getattr(model, '_qwen_prefix_state_spec', None)
        row = module._qwen_prefix_row(0, request_id, start, end, tokens[:, :end], registry, chunk_size, spec)
        plan = list(row.plan)
        if source == 'CHECKPOINT':
            restore = (row.rec, row.carry)
    elif source == 'CHECKPOINT':
        raise AssertionError('%s: a checkpoint resume with no registry' % where)
    else:
        plan = _plan_positions(registry, request_id, start, end, chunk_size)
    if source == 'PARKED':
        restore = (parked.rec, parked.carry)
    return Resolved(source, restore, plan, grant)


class ParkFailed(Exception):
    """The scratch of a suspended prefill could not be read to the host: nothing was parked, the scratch is untouched."""


def _quarantine_unparkable(model, handle, held, failure, log):
    """End the suspended prefill `held` = (request id, position) that could not be parked, per request; False when no quarantine consumer is installed."""
    request_id = held[0]
    try:
        import serving_request_quarantine

        serving_request_quarantine.register(request_id, 'its suspended state could not be parked to the host: %s' % failure)
    except (ImportError, ValueError):
        return False
    handle.parks.pop(request_id, None)
    handle.rows.pop(request_id, None)
    _clear_owner(model)
    handle.park_failures += 1
    log(levern_policy.QUARANTINE_LINE, request_id, 'park failed: %s' % failure)
    return True


def _park(model, handle, held, log):
    """Take the scratch's suspended state to the host (the G1 capture path) and give the scratch up: the owner `held` = (request id, position)
    continues later through source PARKED. The slot cap is the contract's; a park the host cannot hold is never started."""
    request_id, at = held
    slots = handle.merged_cfg.park_slots
    if len(handle.parks) >= slots:
        raise AssertionError('Lever N: no park slot for %r (%d of %d in use): refused before any device work' % (request_id, len(handle.parks), slots))
    began = time.perf_counter()
    previous = model._bind_gdn_prefill_scratch()
    try:
        rec, carry, nbytes = model._qwen_prefix_read_scratch()
    except Exception as failure:
        raise ParkFailed('the host read of the suspended scratch failed (%s: %s)' % (type(failure).__name__, str(failure)[:200])) from failure
    finally:
        model._unbind_gdn_prefill_scratch(previous)
    handle.parks[request_id] = Park(at, rec, carry, nbytes, time.monotonic())
    handle.parked_total += 1
    _clear_owner(model)
    log(levern_policy.PARK_LINE, 'out', request_id, at, (time.perf_counter() - began) * 1000.0, nbytes, len(handle.parks))


def _merged_route(model, tokens, page_table, slot, step, handle, fault, audit, log):
    """One step of the merged route (the module docstring)."""
    import torch

    ttnn = _ttnn()
    start, end, request_id = step.start, step.end, step.req_id
    module = sys.modules[type(model).__module__]
    registry = module._qwen_prefix_registry()
    chunk_size = getattr(model, '_chunked_chunk_size', None) or levern_policy.CHUNK
    page = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
    resolved = _resolve(model, handle, step, tokens, registry, module, chunk_size)
    source = resolved.source
    held = vars(model).get(OWNER_ATTR)
    if held is not None and held[0] != request_id:
        if not step.park_owner or handle.merged_cfg.park != 'host':
            handle.source_refusals += 1
            raise AssertionError('Lever N: request %r step [%d, %d) takes the scratch while %r is suspended on it and the step does not park it '
                                 '(refused before any device work)' % (request_id, start, end, held))
        if len(handle.parks) >= handle.merged_cfg.park_slots:
            handle.source_refusals += 1
            raise AssertionError('Lever N: no park slot for %r (%d of %d in use): refused before any device work'
                                 % (held[0], len(handle.parks), handle.merged_cfg.park_slots))
    elif step.park_owner and held is None:
        handle.source_refusals += 1
        raise AssertionError('Lever N: request %r step [%d, %d) was announced as parking an owner and the scratch has none' % (request_id, start, end))
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    layers = _gdn_layers(model)
    vars(model).pop(WROTE_ATTR, None)
    vars(model).pop(IDENTITY_ATTR, None)
    began, programs_before, window_before = time.perf_counter(), program_count(model), window_programs()
    before = _binding(layers)
    if held is not None and held[0] != request_id:
        try:
            _park(model, handle, held, log)
        except ParkFailed as failure:
            # The long prefill's blocks were all written by earlier steps and the short's step is already allocated, so the short must run: end the
            # LONG one request-shaped (the D2 consumer finishes it at this step's update_from_output; the lifecycle and the route release it on the
            # finished id) instead of failing the engine. Without a consumer the failure stays fatal.
            if not _quarantine_unparkable(model, handle, held, failure, log):
                raise
    final_step = final(step)
    parked_bytes = handle.parks[request_id].nbytes if source == 'PARKED' else 0
    captured = []
    writes = _SlotWrites(model)
    restored_ms = None
    # An error below leaves nobody owning a half-advanced scratch, and a parked state this step began to consume is gone with the request.
    _set_owner(model, (request_id, start))
    previous = model._bind_gdn_prefill_scratch()
    rec_snap = conv_snap = None
    with writes:
        try:
            if resolved.restore is not None:
                restored_began = time.perf_counter()
                model._qwen_prefix_restore(*resolved.restore)
                restored_ms = (time.perf_counter() - restored_began) * 1000.0
                if source == 'CHECKPOINT':
                    module._qwen_prefix_restored(registry, restored_ms)
            resume = dict(start=start)
            if resolved.plan:
                resume['capture_at'] = frozenset(resolved.plan)
                resume['on_capture'] = lambda pos: model._qwen_prefix_capture(registry, request_id, pos, captured)
            lg = model.prefill_traced_chunked(tokens[:, :end], page[0:1], actual_len=end, **resume)
            host = (ttnn.to_torch(lg, mesh_composer=comp).reshape(-1, model.args.vocab_size)[:1].float().view(1, 1, -1))
            ttnn.deallocate(lg)
            if final_step:
                rec_snap, conv_snap = _read_scratch(ttnn, comp, layers)
        except BaseException:
            _clear_owner(model)
            if handle.parks.pop(request_id, None) is not None:
                _publish(model)
            handle.rows.pop(request_id, None)
            raise
        finally:
            model._unbind_gdn_prefill_scratch(previous)
        identity = _binding_holds(before)
        if not identity:
            _clear_owner(model)
            raise AssertionError('Lever N: after step [%d, %d) of %r a GDN layer is not bound to the batched decode buffers it had before the step'
                                 % (start, end, request_id))
        window = window_programs() - window_before
        if source == 'PARKED':
            handle.parks.pop(request_id, None)
            handle.unparked_total += 1
            log(levern_policy.PARK_LINE, 'in', request_id, start, restored_ms or 0.0, parked_bytes, len(handle.parks))
        if final_step:
            model._write_gdn_slot(slot, rec_snap, conv_snap)
            _clear_owner(model)
        else:
            _set_owner(model, (request_id, end))
    vars(model)[WROTE_ATTR] = writes.count > 0
    vars(model)[IDENTITY_ATTR] = identity
    elapsed = (time.perf_counter() - began) * 1000.0
    programs_after = program_count(model)
    log(levern_policy.ROUTE_LINE_MERGED, request_id, start, end, step.total, int(final_step), writes.count, elapsed, programs_before,
        programs_after, window, source, ','.join(captured) if captured else '-')
    if registry is not None:
        dn = layers[0].rec_state if layers else None
        module._qwen_prefix_dram('registry', dn)
        if any(':stored:' in item for item in captured):
            module._qwen_prefix_dram('first capture', dn)
    _aggregate(handle, step, resolved, registry, module, captured, restored_ms, elapsed, programs_before, programs_after, final_step,
               host, rec_snap, conv_snap, page, tokens, model, log)
    if audit and final_step:
        audit_prefill(model, step, host, page[0], rec_snap, conv_snap, log=log, tokens=tokens)
    return [host]


def _aggregate(handle, step, resolved, registry, module, captured, restored_ms, elapsed, before, after, final_step, host, rec_snap, conv_snap, page,
               tokens, model, log):
    """Collect one request's steps into the single '[PREFIX] row=' line the G1 route logs for a whole prompt, and emit it on the final step."""
    request_id = step.req_id
    row = handle.rows.get(request_id)
    if row is None:
        plan = module._qwen_prefix_plan(resolved.grant) if resolved.grant is not None else []
        row = handle.rows[request_id] = dict(q=step.start, plan=list(plan), captured=[], restored_ms=restored_ms, ms=0.0, before=before,
                                             granted=resolved.grant is not None)
    row['captured'] += captured
    row['ms'] += elapsed
    row['after'] = after
    if not final_step:
        return
    del handle.rows[request_id]
    shas = ''
    if os.environ.get('QWEN_PREFIX_AUDIT') == '1' or os.environ.get('QWEN_PREFIX_DIGESTS') == '1':
        slot_sha, logits_sha = module._qwen_prefix_digests(rec_snap, conv_snap, host)
        shas = ' slot_sha=%s logits_sha=%s' % (slot_sha, logits_sha)
    if os.environ.get('QWEN_PREFIX_AUDIT') == '1':
        import types as _types

        model._qwen_prefix_audit(_types.SimpleNamespace(actual=step.total, req_id=request_id, start=row['q']), page[0:1], rec_snap, conv_snap, host)
    restored = '-' if row['restored_ms'] is None else '%.1f' % row['restored_ms']
    log('{}', '[PREFIX] row=0 req=%s path=eager registry=%s grant=%s Q=%d L=%d plan=[%s] restored_ms=%s captured=[%s] dropped=[] ms=%.1f '
        'programs=%s->%s programs_across_restore=None route=merged%s' % (
            request_id, 'present' if registry is not None else 'absent', 'committed' if row['granted'] else 'none', row['q'], step.total,
            ', '.join(str(pos) for pos in row['plan']), restored, ','.join(row['captured']), row['ms'], row['before'], row['after'], shas))


def warm(runner, model, *, environ=None, width=None, slot=0, log=_log):
    """Run the route once over WARM_PROMPT tokens in the plan's three steps, before any trace is captured; True when it ran.

    The page table is the width every prefill has (runner.max_num_blocks_per_req), the prompt zeros, the slot 0: the same warm requests
    serving_runtime.prefill_warm_before_traces runs through the stock entry, so the route's programs and buffers (and the
    tripwire's count of them) exist before the packed traces. A model without the route installed is left alone.

    The merged route's warm (docs/lever-n-prefix-merged-route.md 3.3) also runs a capture at 2,048, a CHECKPOINT restore of that state and a
    second step from it, and under host parking a park and an unpark around the final step, at the same width, so every program the four
    sources touch exists before the packed traces; the rule B - A - W == 0 is unchanged."""
    import torch

    environ = os.environ if environ is None else environ
    if not levern_policy.enabled(environ) or vars(model).get(ENTRY) is None:
        return False
    width = getattr(runner, 'max_num_blocks_per_req', None) if width is None else width
    if type(width) is not int or width <= 0:
        raise ValueError('The Lever N route warm needs the plugin runner\'s max_num_blocks_per_req, not %r' % (width,))
    page = torch.arange(width, dtype=torch.int32).reshape(1, width)
    handle = vars(model).get(HANDLE_ATTR)
    began, programs_before = time.perf_counter(), program_count(model)
    if handle is not None and handle.merged:
        steps = _warm_merged(model, handle, page, slot, log)
    else:
        steps = levern_policy.plan(WARM_PROMPT)
        try:
            for start, end in steps:
                step = Step(WARM_REQUEST, start, end, WARM_PROMPT)
                model.prefill_paged_slots_range([torch.zeros((1, end), dtype=torch.int32)], page, [slot], [start], [end],
                                                valid_lens=[end], step=step)
        finally:
            release(model, WARM_REQUEST)
            vars(model).pop(WROTE_ATTR, None)
    try:
        _ttnn().synchronize_device(model.mesh_device)
    except Exception:
        pass
    log(levern_policy.ROUTE_WARM_LINE, len(steps), programs_before, program_count(model), (time.perf_counter() - began) * 1000.0)
    return True


def _warm_merged(model, handle, page, slot, log):
    """The merged warm: the four sources on the warm prompt, driven through the model directly (no registry exists before the scheduler)."""
    import torch

    ttnn = _ttnn()
    if getattr(model, '_qwen_prefix_restore_mode', None) is None:
        raise ValueError('The merged Lever N warm needs the prefix-reuse model graft to have chosen its GDN restore path first '
                         '(serving_startup.prefix_warm)')
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    layers = _gdn_layers(model)
    total = WARM_PROMPT
    sink = []

    def run(start, end, restore=None, capture=False, final_step=False):
        tokens = torch.zeros((1, end), dtype=torch.int32)
        previous = model._bind_gdn_prefill_scratch()
        try:
            if restore is not None:
                model._qwen_prefix_restore(*restore)
            resume = dict(start=start)
            if capture:
                resume['capture_at'] = frozenset((end,))
                resume['on_capture'] = lambda pos: sink.append(model._qwen_prefix_read_scratch()[:2])
            lg = model.prefill_traced_chunked(tokens, page[0:1], actual_len=end, **resume)
            ttnn.to_torch(lg, mesh_composer=comp)
            ttnn.deallocate(lg)
            snapshot = _read_scratch(ttnn, comp, layers) if final_step else None
        finally:
            model._unbind_gdn_prefill_scratch(previous)
        if final_step:
            model._write_gdn_slot(slot, *snapshot)

    steps = [(0, 2048), (2048, 4096), (2048, 4096), (4096, total)]
    try:
        run(0, 2048, capture=True)
        run(2048, 4096)
        if not sink:
            raise ValueError('The merged Lever N warm took no capture at 2048: the G1 stage\'s capture hook did not run')
        run(2048, 4096, restore=sink[0])
        if handle.merged_cfg.park == 'host':
            began = time.perf_counter()
            previous = model._bind_gdn_prefill_scratch()
            try:
                rec, carry, nbytes = model._qwen_prefix_read_scratch()
            finally:
                model._unbind_gdn_prefill_scratch(previous)
            log(levern_policy.PARK_LINE, 'out', WARM_REQUEST, 4096, (time.perf_counter() - began) * 1000.0, nbytes, 1)
            previous = model._bind_gdn_prefill_scratch()
            try:
                model._qwen_prefix_restore(rec, carry)
            finally:
                model._unbind_gdn_prefill_scratch(previous)
            run(4096, total, final_step=True)
        else:
            run(4096, total, final_step=True)
    finally:
        release(model, WARM_REQUEST)
        vars(model).pop(WROTE_ATTR, None)
    return steps


# ---------------------------------------------------------------------------------------------------------------------------------------
# The epoch scope (design section 8)
# ---------------------------------------------------------------------------------------------------------------------------------------

EXTERNAL_WRITER = 'levern-route'
# The device-write census (test_levern_prefix_census): every call the route makes on the model or ttnn that can write a device buffer, by name. A new
# write site the audited list does not name fails the census, because the disjoint-writer class (verify_prestage.register_external_writer) is sound
# only while this list is complete: the scratch's own tensors (rec_state, conv_states, conv_carry, the zero sources) written in place by the chunk
# loops and by restore, the paged KV pool written through the request's own page table, the decode slot written by _write_gdn_slot on the final step
# (a global bump), and transients allocated and freed inside the step.
WRITE_SITES = ('prefill_traced_chunked', '_qwen_prefix_restore', '_write_gdn_slot', '_qwen_prefix_capture', '_qwen_prefix_read_scratch',
               '_bind_gdn_prefill_scratch', '_unbind_gdn_prefill_scratch', 'deallocate', 'to_torch', 'synchronize_device')


def persistent_writes(model, addresses):
    """The (chip, address) pairs of the persistent device buffers a route step writes between decode rounds: the B=1 prefill scratch of every GDN
    layer (rec_state, conv_states, conv_carry and the zero source). `addresses(tensor)` is the chip-local addresses of one tensor
    (gdn_multitoken_conv.addresses bound to the attach's operations)."""
    ensure = getattr(model, '_ensure_gdn_prefill_scratch', None)
    if callable(ensure):
        ensure()
    pairs = set()
    for entry in getattr(model, '_gdn_prefill_scratch', None) or ():
        dn, rec, convs, carry, zero0 = entry
        for tensor in [rec, carry, zero0] + list(convs):
            for chip, address in enumerate(addresses(tensor)):
                pairs.add((chip, address))
    return pairs


def engage_epoch_scope(model, addresses, *, environ=None, log=_log):
    """QWEN_FAST_LEVERN_EPOCH_SCOPE=route on the merged route: register the route's persistent writes with verify_prestage as an external writer, so an
    intermediate step that wrote no decode slot moves no fixture epoch (serving_lifecycle._epoch_after). Called once at attach, after the route warm.
    Returns 'engaged', 'overlap' (a block's staging destinations share an address: the scope stays global), or None when it is not asked for."""
    environ = os.environ if environ is None else environ
    if not merged_requested(environ) or levern_policy.merged_config(environ).epoch_scope != 'route':
        return None
    from verify_prestage import register_external_writer

    outcome = register_external_writer(EXTERNAL_WRITER, persistent_writes(model, addresses))
    log('{}', '[PINDIAG] lever N epoch scope: route writer %s' % outcome)
    return outcome
