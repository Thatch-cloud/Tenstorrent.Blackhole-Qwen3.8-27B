"""Prefill lever S1 at the model: ONE q chunk per core in the chunked prefill SDPA (QWEN_FAST_SDPA_PF_ONEQ=1, default off).

WHAT. The graft's forward_prefill_paged asks the K64g/K64j chain for the served prefill SDPA with SDPAProgramConfig.max_cores_per_head_batch =
0x5EFA0000 | flags (QWEN_FAST_SDPA_PF=1, flags 0x3) on the calls card M qualified (flexible, bf8, q/k chunk 128, 512/1024/2048 rows). At four
cards that call has 6 local Q heads x 16 q chunks of 128 rows = 96 q chunks, paired causally (chunk i with 15-i) over 48 cores. The oneq
factory edit (optimisation/ttnn-op/sdpa_prefill_oneq, graft K64j-OQ) adds flag 0x8: no pairing, one q chunk per core, 96 busy cores, 16 chains
of 6 - the same arithmetic per q chunk in the same k-chunk order, so the bytes are the chain's and the stock path's (card-M sweep), and the
SDPA time per layer halves at long context (0.2115 -> ~0.106 ms per 1k tokens of context per layer; est. -7.9 s at a 128k solo prefill).

HOW. This module changes ONE thing: the program word of the calls the graft already sends to the chain gets the 0x8 bit, when the call is
one the factory accepts at one chunk per core (Q chunks <= cores, even chunk count, whole GQA groups). It does so by replacing the graft's
module-level _qwen_pf_program_config (attention/tp.py) with a wrapper that calls the graft's own function and, on the config it returns
(the chain's), builds the same config with the extra bit. The graft file is not edited; the wrapper is bound ONLY when the flag is on
(install(), called by tp_addresses.install at four cards), so with the flag off nothing here is even imported by the served path.

FLAGS (strict 0/1; anything else raises at attach):
  QWEN_FAST_SDPA_PF_ONEQ=1        the lever; needs QWEN_FAST_SDPA_PF=1 (the chain it extends), a production QWEN_FAST_SDPA_PF_FLAGS
                                  (unset = 0x3), four-card serving, and a _ttnncpp.so with the oneq edits (the K64j-OQ graft: checked on the
                                  first engaged call, a clear RuntimeError otherwise)
  QWEN_FAST_SDPA_PF_ONEQ_AUDIT=1  needs the lever. The first 8 engaged calls and then every 24th also run the chain's own program (flags
                                  without 0x8) on the same inputs and compare the two outputs bit for bit on every chip; a difference logs
                                  '[PINDIAG] sdpa prefill oneq audit mismatch' and raises. Gate arms only: it doubles those calls.

MARKERS (the smoke tables read them): '[PINDIAG] sdpa prefill oneq engaged ...' (once per rows value), '... fell back reason=...' (once per
rows and reason: a call the factory would refuse, served by the plain chain instead - never a failure of the request), '... audit n=.. exact=True'.

KILL SWITCH. Unset or QWEN_FAST_SDPA_PF_ONEQ=0: the graft's function stays bound and every call is the chain's. A call the lever cannot take
(too few cores, an odd chunk count) is served by the plain chain with the fell-back line.

Stdlib only; ttnn, torch and loguru are imported on first use.
"""

import functools
import importlib
import os
import sys

FLAG = 'QWEN_FAST_SDPA_PF_ONEQ'
AUDIT_FLAG = 'QWEN_FAST_SDPA_PF_ONEQ_AUDIT'
PF_FLAG = 'QWEN_FAST_SDPA_PF'
PF_FLAGS_FLAG = 'QWEN_FAST_SDPA_PF_FLAGS'
GRAFT_MODULE = 'models.demos.blackhole.qwen36.tt.attention.tp'
GRAFT_FUNCTION = '_qwen_pf_program_config'
PF_TAG = 0x5EFA0000
PF_TAG_MASK = 0xFFFF0000
ONEQ_BIT = 0x8
PRODUCTION_FLAGS = (0x1, 0x3, 0x5, 0x7)         # the graft's _QWEN_PF_FLAGS
DEFAULT_FLAGS = 0x3
ROWS = (512, 1024, 2048)                        # the graft's _QWEN_PF_ROWS: the card-M-qualified S
Q_CHUNK = 128
AUDIT_FIRST = 8
AUDIT_STRIDE = 24
AUDIT_LOG_FIRST = 8
AUDIT_LOG_STRIDE = 8
BINARY_MARKER = b'[QWEN-SDPA-PF] oneq needs one q chunk per core'
ENGAGED_MARKER = '[PINDIAG] sdpa prefill oneq engaged'
FELL_BACK_MARKER = '[PINDIAG] sdpa prefill oneq fell back'
AUDIT_MARKER = '[PINDIAG] sdpa prefill oneq audit'
AUDIT_MISMATCH_MARKER = '[PINDIAG] sdpa prefill oneq audit mismatch'

# Process state: the graft's own function (captured by install), the reasons and rows already logged, the audit's counters and last pair.
_STATE = {'original': None, 'audit': False, 'binary_ok': None, 'logged': set(), 'engaged': 0, 'fallbacks': 0, 'audit_calls': 0, 'audited': 0,
          'pair': None, 'op': None}


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
    if source.get(PF_FLAG) != '1':
        raise ValueError('%s=1 extends the prefill chain: it needs %s=1' % (FLAG, PF_FLAG))
    text = (source.get(PF_FLAGS_FLAG) or '').strip()
    if text:
        try:
            flags = int(text, 0)
        except ValueError:
            flags = None
        if flags not in PRODUCTION_FLAGS:
            raise ValueError('%s must be a production chain flag set (%s) with %s, got %r'
                             % (PF_FLAGS_FLAG, ', '.join('%#x' % value for value in PRODUCTION_FLAGS), FLAG, text))
    import tp_shapes

    if tp_shapes.chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s is a four-card lever (one q chunk per core needs the 6 local Q heads of TP4): this process serves the pair' % FLAG)
    return True, audit


def enabled(environ=None):
    """The gate for a flagged twin row: whether the lever is on (strict; raises on a malformed or unsupported setting)."""
    return settings(environ)[0]


def eligibility(nqh, nkh, rows, grid, q_chunk=Q_CHUNK):
    """None when the factory takes the call at one q chunk per core, else the reason it would refuse (the factory's own predicates)."""
    if rows % q_chunk:
        return 'rows %d is not a whole number of %d-row q chunks' % (rows, q_chunk)
    chunks = rows // q_chunk
    if chunks % 2:
        return 'an odd q chunk count (%d)' % chunks
    if nkh < 1 or nqh % nkh:
        return '%d Q heads do not split over %d KV heads' % (nqh, nkh)
    cores = int(grid[0]) * int(grid[1])
    if nqh * chunks > cores:
        return '%d q chunks (%d heads x %d) exceed the %d cores' % (nqh * chunks, nqh, chunks, cores)
    return None


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


def binary_has_oneq(maps='/proc/self/maps'):
    """Whether the one _ttnncpp.so this process mapped carries the oneq edits (a K64j binary ignores flag 0x8's meaning and TT_FATALs on it)."""
    import mmap

    with open(maps) as handle:
        paths = sorted({line.split()[-1] for line in handle if line.rstrip().endswith('_ttnncpp.so')})
    if len(paths) != 1:
        raise RuntimeError('[QWEN-SDPA-PF] expected exactly one mapped _ttnncpp.so, found %r' % (paths,))
    with open(paths[0], 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        return view.find(BINARY_MARKER) >= 0


def _require_binary(check=None):
    if _STATE['binary_ok'] is None:
        _STATE['binary_ok'] = bool((binary_has_oneq if check is None else check)())
    if not _STATE['binary_ok']:
        raise RuntimeError('%s=1 but the loaded _ttnncpp.so lacks the oneq edits of the prefill factory (mount the K64j-OQ graft: '
                           'optimisation/ttnn-op/sdpa_prefill_oneq/build_k64j_oq.sh)' % FLAG)


def program_config(layer, served, qk_chunk, flexible, S, original=None, binary_check=None):
    """The graft's _qwen_pf_program_config with the oneq bit on the chain's config when the factory takes the call at one chunk per core.

    `original` is the graft's own function (install captured it). A call the graft keeps on the served path (served is returned) is served;
    a chain call the lever cannot take is the graft's config unchanged, with the fell-back line once per (rows, reason)."""
    original = original or _STATE['original']
    if original is None:
        raise RuntimeError('%s: the graft function was not captured (install() has not run)' % GRAFT_FUNCTION)
    config = original(layer, served, qk_chunk, flexible, S)
    if config is served:
        return served
    nqh, nkh, grid = int(layer.NH), int(layer.NKV), _grid_of(layer)
    reason = eligibility(nqh, nkh, S, grid, qk_chunk)
    if reason is not None:
        _STATE['fallbacks'] += 1
        _note('fell back', (S, reason), '%s rows=%d reason=%s' % (FELL_BACK_MARKER, S, reason))
        return config
    _require_binary(binary_check)
    import ttnn

    word = int(layer._sdpa_pf_word) | ONEQ_BIT
    chosen = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=layer.mesh.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=qk_chunk,
        k_chunk_size=qk_chunk,
        max_cores_per_head_batch=word,
    )
    _STATE['engaged'] += 1
    chunks = S // qk_chunk
    _note('engaged', S, '%s flags=%#x rows=%d heads=%d/%d q_chunks=%d cores=%d chunks_per_core=1 chains=%d'
          % (ENGAGED_MARKER, word & 0xFFFF, S, nqh, nkh, nqh * chunks, grid[0] * grid[1], chunks * nkh))
    if _STATE['audit']:
        reference = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=layer.mesh.compute_with_storage_grid_size(),
            exp_approx_mode=False,
            q_chunk_size=qk_chunk,
            k_chunk_size=qk_chunk,
            max_cores_per_head_batch=word & ~ONEQ_BIT,
        )
        _STATE['pair'] = (chosen, reference, layer.mesh)
        _install_audit(ttnn)
    return chosen


def _install_audit(ttnn):
    """Wrap ttnn.transformer.chunked_scaled_dot_product_attention once: a call carrying the last oneq config also runs its chain reference."""
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
        exact, detail = _compare(ttnn, pair[2], out, reference)
        ttnn.deallocate(reference)
        _STATE['audited'] += 1
        if not exact:
            _log('%s n=%d call=%d exact=False %s' % (AUDIT_MISMATCH_MARKER, _STATE['audited'], index, detail))
            raise RuntimeError('%s: the oneq SDPA output differs from the chain program\'s (call %d: %s)' % (AUDIT_FLAG, index, detail))
        if _STATE['audited'] <= AUDIT_LOG_FIRST or _STATE['audited'] % AUDIT_LOG_STRIDE == 0:
            _log('%s n=%d call=%d exact=True' % (AUDIT_MARKER, _STATE['audited'], index))
        return out

    ttnn.transformer.chunked_scaled_dot_product_attention = audited


def _compare(ttnn, mesh, mine, reference):
    """(exact, detail): the two outputs read back from every chip and compared as int16 bit patterns."""
    import torch

    composer = ttnn.ConcatMeshToTensor(mesh, dim=0) if hasattr(ttnn, 'ConcatMeshToTensor') else None
    a = ttnn.to_torch(mine, mesh_composer=composer) if composer is not None else ttnn.to_torch(mine)
    b = ttnn.to_torch(reference, mesh_composer=composer) if composer is not None else ttnn.to_torch(reference)
    if tuple(a.shape) != tuple(b.shape):
        return False, 'shapes %s vs %s' % (tuple(a.shape), tuple(b.shape))
    a16 = a.to(torch.bfloat16).contiguous().view(torch.int16)
    b16 = b.to(torch.bfloat16).contiguous().view(torch.int16)
    if torch.equal(a16, b16):
        return True, ''
    differing = int((a16 != b16).sum())
    return False, '%d of %d elements differ' % (differing, a16.numel())


def install(environ=None):
    """Bind the wrapper in place of the graft's _qwen_pf_program_config in every loaded module that holds it. -> [(namespace, name, old)], empty
    (and nothing imported) when the lever is off. Called by tp_addresses.install at four cards."""
    on, audit = settings(environ)
    if not on:
        return []
    module = importlib.import_module(GRAFT_MODULE)
    old = getattr(module, GRAFT_FUNCTION, None)
    if old is None:
        raise RuntimeError('%s=1 but %s has no %s (a graft without the prefill chain opt-in)' % (FLAG, GRAFT_MODULE, GRAFT_FUNCTION))
    if getattr(old, '_oneq_wrapper', False):
        return []
    _STATE['original'] = old
    _STATE['audit'] = audit

    @functools.wraps(old)
    def wrapper(layer, served, qk_chunk, flexible, S):
        return program_config(layer, served, qk_chunk, flexible, S)

    wrapper._oneq_wrapper = True
    rebound = []
    for loaded in list(sys.modules.values()):
        namespace = getattr(loaded, '__dict__', None)
        if isinstance(namespace, dict) and namespace.get(GRAFT_FUNCTION) is old:
            namespace[GRAFT_FUNCTION] = wrapper
            rebound.append((namespace, GRAFT_FUNCTION, old))
    return rebound


def reset_state():
    """Tests only: forget the process state."""
    _STATE.update(original=None, audit=False, binary_ok=None, logged=set(), engaged=0, fallbacks=0, audit_calls=0, audited=0, pair=None, op=None)
