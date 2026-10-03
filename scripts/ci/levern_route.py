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
"""

import hashlib
import inspect
import os
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

Step = namedtuple('Step', 'req_id start end total')


def final(step):
    """Whether `step` ends its prompt."""
    return step.end == step.total


def whole(step):
    """Whether `step` is the whole prompt (the stock path runs it)."""
    return step.start == 0 and step.end == step.total


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
    """The lifecycle's call when a request is finished or aborted mid-prefill: its suspended scratch is nobody's any more."""
    held = vars(model).get(OWNER_ATTR)
    if held is not None and held[0] == request_id:
        del vars(model)[OWNER_ATTR]
        return True
    return False


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


def audit_prefill(model, step, logits, page_row, rec_snap=None, conv_snap=None, log=_log):
    """Log the digests of a finished prefill (levern_policy.DIGEST_LINE); `rec_snap`/`conv_snap` are the scratch's host copies when the
    caller already holds them (the final step), else the scratch is read here (the whole-prompt path)."""
    if rec_snap is None:
        rec_snap, conv_snap = scratch_snapshot(model)
    slot_sha, logits_sha = digests(rec_snap, conv_snap, logits)
    log(levern_policy.DIGEST_LINE, step.req_id, step.total, slot_sha, logits_sha, kv_digest(model, page_row, step.total))


class Handle(object):
    """What install() returns: the instance attributes it set, so the attach's scope can take them off again."""

    def __init__(self, wrapper, model, original, route, audit):
        self.wrapper, self.model, self.original, self.route, self.audit = wrapper, model, original, route, audit
        self.fault_fired = False
        self.owner_refusals = 0

    def uninstall(self):
        if self.wrapper.__dict__.get('prefill_forward') is not None:
            del self.wrapper.__dict__['prefill_forward']
        vars(self.model).pop(ENTRY, None)
        for name in (STEP_ATTR, OWNER_ATTR, WROTE_ATTR, HANDLE_ATTR):
            vars(self.model).pop(name, None)


def requirement_problems(model):
    """What the model lacks for the route, one string each: the G1 stage's resumable loops and the scratch helpers."""
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
    return problems


def install(runner, model, *, environ=None, log=_log):
    """Install the route and the digest instrument under QWEN_FAST_LEVER_N=1 / QWEN_FAST_LEVERN_AUDIT=1; None when neither is set.

    Refuses (ValueError, failing the attach) a model without what the route needs, a second install, and a wrapper without prefill_forward."""
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
    if route_on:
        problems = requirement_problems(model)
        if problems:
            raise ValueError('Lever N cannot be installed on this model tree: %s' % '; '.join(problems))
    handle = Handle(wrapper, model, original, route_on, audit_on)
    fault = levern_policy.fault(environ) if route_on else None

    def prefill_forward(*args, **kwargs):
        step = _take(model)
        if step is None or whole(step):
            starts = kwargs.get('start_pos')
            if step is None and starts is not None and any(int(value) > 0 for value in starts):
                raise AssertionError('Lever N: a prefill step at start_pos=%r reached the model without a lifecycle '
                                     'announcement: nothing says whose scratch it would continue' % ([int(v) for v in starts],))
            # A prompt that runs whole (and any request the lifecycle did not announce: the warm requests) is the stock path, and it
            # resets the B=1 scratch at start 0: whatever suspended prompt owned it is gone, and its continuation must be refused.
            vars(model).pop(OWNER_ATTR, None)
        if step is None:
            return original(*args, **kwargs)
        if whole(step):
            result = original(*args, **kwargs)
            if audit_on:
                audit_prefill(model, step, result[0], _page_row(_named(args, kwargs)), log=log)
            return result
        if not route_on:
            raise AssertionError('Lever N: a step of a split prompt reached a process without %s=1: %r' % (levern_policy.FLAG, step))
        return run_chunk(model, step, _named(args, kwargs), handle, fault, audit_on, log)

    if route_on:
        vars(model)[ENTRY] = types.MethodType(
            lambda self, *a, **k: route(self, *a, handle=handle, fault=fault, audit=audit_on, log=log, **k), model)
    vars(wrapper)['prefill_forward'] = prefill_forward
    vars(model)[HANDLE_ATTR] = handle
    log(levern_policy.ROUTE_INSTALLED_LINE, 'route=%d audit=%d' % (route_on, audit_on))
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
    held = vars(model).get(OWNER_ATTR)
    if start and fault == 'foreign' and handle is not None and not handle.fault_fired:
        handle.fault_fired = True
        held = ('__foreign__', start)
        vars(model)[OWNER_ATTR] = held
    if start and held != (step.req_id, start):
        if handle is not None:
            handle.owner_refusals += 1
        raise AssertionError('Lever N: request %r continues at %d but the prefill scratch is owned by %r: a chunk was skipped, '
                             'replayed or belongs to another prompt (refused before any device work)' % (step.req_id, start, held))
    vars(model).pop(WROTE_ATTR, None)
    page = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
    began, programs_before = time.perf_counter(), program_count(model)
    # A start-0 step takes a new owner at once, a continuation keeps its own; an error below leaves nobody owning a half-advanced scratch.
    vars(model)[OWNER_ATTR] = (step.req_id, start)
    previous = model._bind_gdn_prefill_scratch()
    rec_snap = conv_snap = None
    try:
        lg = model.prefill_traced_chunked(tokens[:, :end], page[0:1], actual_len=end, start=start)
        host = (ttnn.to_torch(lg, mesh_composer=comp).reshape(-1, model.args.vocab_size)[:1].float().view(1, 1, -1))
        ttnn.deallocate(lg)
        if final(step):
            rec_snap, conv_snap = _read_scratch(ttnn, comp, layers)
    except BaseException:
        vars(model).pop(OWNER_ATTR, None)
        raise
    finally:
        model._unbind_gdn_prefill_scratch(previous)
    if final(step):
        model._write_gdn_slot(int(empty_slots[0]), rec_snap, conv_snap)
        vars(model).pop(OWNER_ATTR, None)
        vars(model)[WROTE_ATTR] = True
    else:
        vars(model)[OWNER_ATTR] = (step.req_id, end)
        vars(model)[WROTE_ATTR] = False
    log(levern_policy.ROUTE_LINE, step.req_id, start, end, step.total, int(final(step)), int(final(step)),
        (time.perf_counter() - began) * 1000.0, programs_before, program_count(model))
    if audit and final(step):
        audit_prefill(model, step, host, page[0], rec_snap, conv_snap, log=log)
    return [host]


def warm(runner, model, *, environ=None, width=None, slot=0, log=_log):
    """Run the route once over WARM_PROMPT tokens in the plan's three steps, before any trace is captured; True when it ran.

    The page table is the width every prefill has (runner.max_num_blocks_per_req), the prompt zeros, the slot 0: the same warm requests
    serving_runtime.prefill_warm_before_traces runs through the stock entry, so the route's programs and buffers (and the
    tripwire's count of them) exist before the packed traces. A model without the route installed is left alone."""
    import torch

    environ = os.environ if environ is None else environ
    if not levern_policy.enabled(environ) or vars(model).get(ENTRY) is None:
        return False
    width = getattr(runner, 'max_num_blocks_per_req', None) if width is None else width
    if type(width) is not int or width <= 0:
        raise ValueError('The Lever N route warm needs the plugin runner\'s max_num_blocks_per_req, not %r' % (width,))
    page = torch.arange(width, dtype=torch.int32).reshape(1, width)
    steps = levern_policy.plan(WARM_PROMPT)
    began, programs_before = time.perf_counter(), program_count(model)
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
