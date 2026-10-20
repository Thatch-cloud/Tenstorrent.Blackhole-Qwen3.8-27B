"""Prefill lever PF3-K at the model: the K-split of the chunked prefill SDPA for the tail passes (QWEN_FAST_SDPA_PF_KSPLIT=1, default off, a NUMERICS change, gate-only until the owner's waiver).

WHAT. A tail pass of 1,024 or 512 rows keeps 48 or 24 of the grid's 130 cores busy under the oneq call (one q chunk per core, QWEN_FAST_SDPA_PF_ONEQ=1) and a q chunk's K stream is serial on its
core. The K-split factory edit (optimisation/ttnn-op/sdpa_prefill_ksplit, graft K64j-OQ-KS) adds flag 0x10 to the program word: every q chunk is streamed by S = min(4, cores / q chunks)
slices on S consecutive cores and the slices' partial results are folded by the pinned ring merge, in slice order (the output differs from the oneq call in the last bits: docs/tp4-prefill3-ksplit.md,
the numerics waiver). 2,048-row chunk passes (96 of 130 cores busy) have one slice and stay the oneq call, bitwise.

HOW. This module changes ONE thing: the program word of the calls that are ALREADY oneq calls (the oneq lever's wrapper of the graft's _qwen_pf_program_config returned a word with the chain and
oneq bits) gets the 0x10 bit, for the rows the lever takes (512 and 1,024: the chain-qualified rows with a second slice) on a grid with room for the slices. It stacks on the oneq wrapper
(tp_addresses binds it after that one), calls it, and rebuilds the config with the extra bit. Nothing else moves: the graft is not edited, and with the flag off nothing here is imported.

FLAGS (strict 0/1; anything else raises at attach):
  QWEN_FAST_SDPA_PF_KSPLIT=1        the lever; needs QWEN_FAST_SDPA_PF_ONEQ=1 (and so the chain, a production flag set and four-card serving) and a _ttnncpp.so with the K-split edits (the K64j-OQ-KS
                                    graft: checked on the first engaged call, a clear RuntimeError otherwise)
  QWEN_FAST_SDPA_PF_KSPLIT_AUDIT=1  needs the lever. The first 8 engaged calls and then every 24th also run the oneq program (the same word without 0x10) on the same inputs and compare the two outputs by
                                    the waiver's statistic (ksplit_numerics.error_stats: the error in bf16 ulps of the row's scale) against the bound (ksplit_numerics.BOUND, PROVISIONAL until card M measures
                                    it); a call outside the bound logs '[PINDIAG] sdpa prefill ksplit audit mismatch' and raises. NOT a bitwise audit. Gate arms only: it doubles those calls.
THE CONTRACT: a TRAFFIC profile carrying the flag is refused (serving_c2_contract.ksplit_problems) unless the owner's traffic waiver names it; a gate-only profile boots it.

MARKERS (the smoke rule ksplit_smoke.py reads them): '[PINDIAG] sdpa prefill ksplit engaged ...' (once per rows), '... skipped rows=.. reason=..' (once per rows and reason: a call the lever does not take
by design, e.g. every 2,048-row chunk pass; NEVER a failure), '... fell back reason=..' (an unexpected miss: the call was not a oneq call), '... audit n=.. rows=.. context=.. max_ulps=.. rms_ulps=.. within_bound=True'.

KILL SWITCH. Unset or QWEN_FAST_SDPA_PF_KSPLIT=0: the oneq wrapper stays bound and every call is the oneq call, bitwise the s5 path.

Stdlib only; ttnn, torch and loguru are imported on first use.
"""

import functools
import importlib
import os
import sys

FLAG = 'QWEN_FAST_SDPA_PF_KSPLIT'
AUDIT_FLAG = 'QWEN_FAST_SDPA_PF_KSPLIT_AUDIT'
ONEQ_FLAG = 'QWEN_FAST_SDPA_PF_ONEQ'
GRAFT_MODULE = 'models.demos.blackhole.qwen36.tt.attention.tp'
GRAFT_FUNCTION = '_qwen_pf_program_config'
PF_TAG = 0x5EFA0000
PF_TAG_MASK = 0xFFFF0000
CHAIN_BIT = 0x1
ONEQ_BIT = 0x8
KSPLIT_BIT = 0x10
ROWS = (512, 1024)                              # the chain-qualified rows (the graft's _QWEN_PF_ROWS) that have a second slice at TP4 on the 13x10 grid
Q_CHUNK = 128
SLICE_CAP = 4
AUDIT_FIRST = 8
AUDIT_STRIDE = 24
AUDIT_LOG_FIRST = 8
AUDIT_LOG_STRIDE = 8
BINARY_MARKER = b'[QWEN-SDPA-PF] ksplit needs at least 2 slices'
BASE = 'sdpa prefill ksplit'
ENGAGED_MARKER = '[PINDIAG] %s engaged' % BASE
SKIPPED_MARKER = '[PINDIAG] %s skipped' % BASE
FELL_BACK_MARKER = '[PINDIAG] %s fell back' % BASE
AUDIT_MARKER = '[PINDIAG] %s audit' % BASE
AUDIT_MISMATCH_MARKER = '[PINDIAG] %s audit mismatch' % BASE

# Process state: the function this module wrapped (captured by install), the reasons and rows already logged, the audit's counters and last pair.
_STATE = {'original': None, 'audit': False, 'binary_ok': None, 'logged': set(), 'engaged': 0, 'skipped': 0, 'fallbacks': 0, 'audit_calls': 0, 'audited': 0, 'pair': None, 'op': None}


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def settings(environ=None):
    """(on, audit) for this process; strict. Raises ValueError at attach for anything the lever cannot honour."""
    source = os.environ if environ is None else environ
    on = _flag(FLAG, source)
    audit = _flag(AUDIT_FLAG, source)
    if audit and not on:
        raise ValueError('%s needs %s=1' % (AUDIT_FLAG, FLAG))
    if not on:
        return False, False
    if not _flag(ONEQ_FLAG, source):
        raise ValueError('%s=1 splits the oneq call\'s K stream: it needs %s=1' % (FLAG, ONEQ_FLAG))
    import sdpa_pf_oneq_tp

    sdpa_pf_oneq_tp.settings(source)          # the oneq lever's own prerequisites (the chain, a production flag set, four-card serving): its error text is the better one
    return True, audit


def enabled(environ=None):
    """The gate for a flagged twin row: whether the lever is on (strict; raises on a malformed or unsupported setting)."""
    return settings(environ)[0]


def split_k_slices(rows, nqh, cores, q_chunk=Q_CHUNK, cap=SLICE_CAP):
    """S, the factory's slice count (apply_factory_pk.py PK0): min(cap, cores // q chunks), 1 when the pass does not divide into q chunks (ksplit_planner.split_k_slices is the twin)."""
    if rows <= 0 or rows % q_chunk:
        return 1
    return max(1, min(cap, cores // (nqh * (rows // q_chunk))))


def eligibility(nqh, nkh, rows, grid, q_chunk=Q_CHUNK):
    """None when the lever splits the call, else the reason it does not: what the oneq call needs first, then the rows the chain qualifies, then a second slice."""
    import sdpa_pf_oneq_tp

    reason = sdpa_pf_oneq_tp.eligibility(nqh, nkh, rows, grid, q_chunk)
    if reason is not None:
        return reason
    if rows not in ROWS:
        return 'rows %d: no second slice (%d q chunks of %d on %d cores) or not chain-qualified for the split' % (
            rows, nqh * (rows // q_chunk), q_chunk, int(grid[0]) * int(grid[1]))
    cores = int(grid[0]) * int(grid[1])
    if split_k_slices(rows, nqh, cores, q_chunk) < 2:
        return '%d q chunks on %d cores have one slice' % (nqh * (rows // q_chunk), cores)
    return None


def topology(nqh, nkh, rows, grid, q_chunk=Q_CHUNK):
    """(slices, q_chunks, busy cores, chains) of an engaged call, as the factory logs them (ksplit=1 slices=.. q_chunks=.. busy=.. chains=..)."""
    chunks = rows // q_chunk
    total = nqh * chunks
    slices = split_k_slices(rows, nqh, int(grid[0]) * int(grid[1]), q_chunk)
    return slices, total, total * slices, chunks * nkh * slices


def _grid_of(layer):
    grid = layer.mesh.compute_with_storage_grid_size()
    if hasattr(grid, 'x'):
        return int(grid.x), int(grid.y)
    return int(grid[0]), int(grid[1])


def _log(text):
    try:
        from loguru import logger
    except ImportError:
        print(text, flush=True)
        return
    logger.info(text)


def _note(kind, key, text):
    """Log once per (kind, key)."""
    if (kind, key) not in _STATE['logged']:
        _STATE['logged'].add((kind, key))
        _log(text)


def binary_has_ksplit(maps='/proc/self/maps'):
    """Whether the one _ttnncpp.so this process mapped carries the K-split edits (a K64j-OQ binary TT_FATALs on flag 0x10 as an unknown flag)."""
    import mmap

    with open(maps) as handle:
        paths = sorted({line.split()[-1] for line in handle if line.rstrip().endswith('_ttnncpp.so')})
    if len(paths) != 1:
        raise RuntimeError('[QWEN-SDPA-PF] expected exactly one mapped _ttnncpp.so, found %r' % (paths,))
    with open(paths[0], 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        return view.find(BINARY_MARKER) >= 0


def _require_binary(check=None):
    if _STATE['binary_ok'] is None:
        _STATE['binary_ok'] = bool((binary_has_ksplit if check is None else check)())
    if not _STATE['binary_ok']:
        raise RuntimeError('%s=1 but the loaded _ttnncpp.so lacks the K-split edits of the prefill factory (mount the K64j-OQ-KS graft: '
                           'optimisation/ttnn-op/sdpa_prefill_ksplit/build_k64j_oq_ks.sh)' % FLAG)


def _word_of(config):
    return int(getattr(config, 'max_cores_per_head_batch'))


def program_config(layer, served, qk_chunk, flexible, S, original=None, binary_check=None):
    """The oneq wrapper's (or the graft's) config with the K-split bit when the call is a oneq call the factory splits.

    `original` is the function install captured (the oneq wrapper bound just before this one, or the graft's own function). A call the graft keeps on the served path (served is returned) is served;
    a call that is not a oneq call (the oneq lever did not take it) is returned unchanged with the fell-back line once per (rows, reason); a oneq call the split does not take BY DESIGN (a
    2,048-row pass, a grid with no room) is returned unchanged with the skipped line once per (rows, reason)."""
    original = original or _STATE['original']
    if original is None:
        raise RuntimeError('%s: the wrapped function was not captured (install() has not run)' % GRAFT_FUNCTION)
    config = original(layer, served, qk_chunk, flexible, S)
    if config is served:
        return served
    word = _word_of(config)
    if (word & PF_TAG_MASK) != PF_TAG or (word & (CHAIN_BIT | ONEQ_BIT)) != (CHAIN_BIT | ONEQ_BIT):
        _STATE['fallbacks'] += 1
        _note('fell back', (S, word & 0xFFFF), '%s rows=%d reason=the call is not a oneq call (word %#x)' % (FELL_BACK_MARKER, S, word))
        return config
    nqh, nkh, grid = int(layer.NH), int(layer.NKV), _grid_of(layer)
    reason = eligibility(nqh, nkh, S, grid, qk_chunk)
    if reason is not None:
        _STATE['skipped'] += 1
        _note('skipped', (S, reason), '%s rows=%d reason=%s' % (SKIPPED_MARKER, S, reason))
        return config
    _require_binary(binary_check)
    import ttnn

    chosen = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=layer.mesh.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=qk_chunk,
        k_chunk_size=qk_chunk,
        max_cores_per_head_batch=word | KSPLIT_BIT,
    )
    _STATE['engaged'] += 1
    slices, q_chunks, busy, chains = topology(nqh, nkh, S, grid, qk_chunk)
    _note('engaged', S, '%s flags=%#x rows=%d heads=%d/%d q_chunks=%d slices=%d cores=%d busy=%d chains=%d'
          % (ENGAGED_MARKER, (word | KSPLIT_BIT) & 0xFFFF, S, nqh, nkh, q_chunks, slices, grid[0] * grid[1], busy, chains))
    if _STATE['audit']:
        _STATE['pair'] = (chosen, config, layer.mesh)
        _install_audit(ttnn)
    return chosen


def _install_audit(ttnn):
    """Wrap ttnn.transformer.chunked_scaled_dot_product_attention once: a call carrying the last K-split config also runs its oneq reference and the two are judged by the waiver's statistic."""
    if _STATE['op'] is not None:
        return
    original = ttnn.transformer.chunked_scaled_dot_product_attention
    _STATE['op'] = original

    @functools.wraps(original)
    def audited(*args, **kwargs):
        pair = _STATE['pair']
        out = original(*args, **kwargs)
        if pair is None or kwargs.get('program_config') is not pair[0]:
            return out
        _STATE['audit_calls'] += 1
        index = _STATE['audit_calls'] - 1
        if index >= AUDIT_FIRST and (index - AUDIT_FIRST) % AUDIT_STRIDE:
            return out
        reference = original(*args, **dict(kwargs, program_config=pair[1]))
        rows = _rows_of(args, kwargs)
        context = _context_of(ttnn, pair[2], kwargs, rows)
        stats, problems = _compare(ttnn, pair[2], out, reference, context)
        ttnn.deallocate(reference)
        _STATE['audited'] += 1
        if problems:
            _log('%s n=%d call=%d rows=%d context=%s within_bound=False %s' % (AUDIT_MISMATCH_MARKER, _STATE['audited'], index, rows, context, '; '.join(problems)))
            raise RuntimeError('%s: the K-split SDPA output is outside the waiver bound against the oneq call (call %d: %s)' % (AUDIT_FLAG, index, '; '.join(problems)))
        if _STATE['audited'] <= AUDIT_LOG_FIRST or _STATE['audited'] % AUDIT_LOG_STRIDE == 0:
            _log('%s n=%d call=%d rows=%d context=%s max_ulps=%.3f rms_ulps=%.4f frac_gt2=%.5f within_bound=True' % (
                AUDIT_MARKER, _STATE['audited'], index, rows, context, stats['max_ulps'], stats['rms_ulps'], stats['frac_gt2']))
        return out

    ttnn.transformer.chunked_scaled_dot_product_attention = audited


def _rows_of(args, kwargs):
    """The query's row count of a call (the third dimension of input_tensor_q, given by keyword or first); 0 when it cannot be read (the line still logs)."""
    query = kwargs.get('input_tensor_q', args[0] if args else None)
    try:
        return int(query.shape[2])
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0


def _context_of(ttnn, mesh, kwargs, rows):
    """The K length the call attends over (its chunk start plus its rows), for the context-scaled bound; None when it cannot be read (the bound then takes its lenient end, and the line says so).
    The chunk start is the integer keyword of the legacy path or the first element of the flexible path's device tensor (read back: this is the audit, a gate arm that already reads outputs)."""
    start = kwargs.get('chunk_start_idx')
    try:
        if start is None and kwargs.get('chunk_start_idx_tensor') is not None:
            import torch

            composer = ttnn.ConcatMeshToTensor(mesh, dim=0) if hasattr(ttnn, 'ConcatMeshToTensor') else None
            tensor = kwargs['chunk_start_idx_tensor']
            host = ttnn.to_torch(tensor, mesh_composer=composer) if composer is not None else ttnn.to_torch(tensor)
            start = int(host.reshape(-1)[0].item()) if isinstance(host, torch.Tensor) else None
        return int(start) + int(rows) if start is not None else None
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        return None


def _compare(ttnn, mesh, mine, reference, context=None):
    """(stats, problems): the two outputs read back from every chip and judged by ksplit_numerics (the error in bf16 ulps of the row's scale against the bound at `context`)."""
    import torch

    import ksplit_numerics

    composer = ttnn.ConcatMeshToTensor(mesh, dim=0) if hasattr(ttnn, 'ConcatMeshToTensor') else None
    a = ttnn.to_torch(mine, mesh_composer=composer) if composer is not None else ttnn.to_torch(mine)
    b = ttnn.to_torch(reference, mesh_composer=composer) if composer is not None else ttnn.to_torch(reference)
    if tuple(a.shape) != tuple(b.shape):
        return {'max_ulps': float('inf'), 'rms_ulps': float('inf'), 'frac_gt2': 1.0}, ['shapes %s vs %s' % (tuple(a.shape), tuple(b.shape))]
    stats = ksplit_numerics.error_stats(a.to(torch.bfloat16), b.to(torch.bfloat16), torch)
    return stats, ksplit_numerics.within_bound(stats, ksplit_numerics.bound_for(context))


def install(environ=None):
    """Bind the wrapper in place of the function currently bound as the graft's _qwen_pf_program_config (the oneq wrapper, bound just before) in every loaded module that holds it.
    -> [(namespace, name, old)], empty (and nothing imported) when the lever is off. Called by tp_addresses.install at four cards."""
    on, audit = settings(environ)
    if not on:
        return []
    module = importlib.import_module(GRAFT_MODULE)
    old = getattr(module, GRAFT_FUNCTION, None)
    if old is None:
        raise RuntimeError('%s=1 but %s has no %s (a graft without the prefill chain opt-in)' % (FLAG, GRAFT_MODULE, GRAFT_FUNCTION))
    if getattr(old, '_ksplit_wrapper', False):
        return []
    if not getattr(old, '_oneq_wrapper', False):
        raise RuntimeError('%s=1 but the function bound as %s is not the oneq lever\'s wrapper (tp_addresses binds this lever after the oneq one)' % (FLAG, GRAFT_FUNCTION))
    _STATE['original'] = old
    _STATE['audit'] = audit

    @functools.wraps(old)
    def wrapper(layer, served, qk_chunk, flexible, S):
        return program_config(layer, served, qk_chunk, flexible, S)

    wrapper._ksplit_wrapper = True
    rebound = []
    for loaded in list(sys.modules.values()):
        namespace = getattr(loaded, '__dict__', None)
        if isinstance(namespace, dict) and namespace.get(GRAFT_FUNCTION) is old:
            namespace[GRAFT_FUNCTION] = wrapper
            rebound.append((namespace, GRAFT_FUNCTION, old))
    return rebound


def reset_state():
    """Tests only: forget the process state."""
    _STATE.update(original=None, audit=False, binary_ok=None, logged=set(), engaged=0, skipped=0, fallbacks=0, audit_calls=0, audited=0, pair=None, op=None)
