"""NP1, the scratch page behind the lone-user padded round (QWEN_FAST_SOLO_PACKED=1, GATE ONLY, default off; docs/tp4-np1-fastlane.md).

THE PROBLEM. A lone live user in a 4-user M3 block leaves THREE idle segments. Each idle segment's K/V is written through an all-zero page table, so into physical page 0 (vLLM's
null block), and page 0 has two 32-row tile rows; the K/V writers (the chained writer, the page-parallel writer) are exact only while no two segments write one (page, tile row)
(verify_trace_t2.kv_conflict). So a block holds two idle segments (packed_verifier.PackedVerifierEngine.MAX_IDLE_SEGMENTS) and the block refuses padded_min_users below users - 2
at construction: one live user cannot pad the block, and runs the 4-row per-request engine (83.9 ms a pass, 3.7 tokens a round) instead of the 64-row packed pass (52.8 ms, 6.8 a user).

THE ANSWER, and the only physical one: a third (page, tile row) target that nothing else owns. vLLM owns every block id below its pool size (the scheduler hands out any of them, and its
prefix cache keeps free blocks' content), so the target has to be a block it does not know. The TT worker allocates the cache tensors from the shape vLLM gives it
(vllm_tt_plugin model_runner._allocate_kv_caches -> Qwen36ForCausalLM.allocate_kv_cache(shape, dtype, layers)), so this module makes that shape ONE BLOCK LONGER on the worker:

  - `install(module)` (a serving_c2_contract.PostImportHook on the model entry, armed at boot by QWEN_FAST_SOLO_PACKED=1 and by nothing else) wraps Qwen36ForCausalLM.allocate_kv_cache: the
    caches get PAGES_WANTED blocks more than vLLM's pool holds and the pair (first scratch block id = vLLM's block count, count) is recorded in this process. The scheduler never sees
    them: block ids at or past its count are never handed to a request. The new block costs 557,056 bytes per chip (16 attention layers x K and V x 64 x 256 bfloat8_b).
  - `claim(physical_pages)` (serving_cache_owner.ServingCacheOwner, at the attach) turns the record into the block ids and refuses a cache that is not exactly vLLM's pool plus them. The owner
    then reports the POOL's page count as physical_pages, so a request whose block table names a scratch id is refused by the bridge factory's own bound check, as any id outside the pool.
  - PackedVerifierEngine(idle_scratch_pages=...) sends idle segments 2 and 3 to tile rows 0 and 1 of the scratch page (idle_inputs), and refuses a live table holding it (padded_refusal),
    as it refuses page 0.

Nothing here changes a kernel, a trace, a program or a captured address: the idle table is a host value staged the way the zero table is. The cache tensors are zero-filled (ttnn.zeros), so the
block starts as finite as page 0 does, and only idle segments ever write it (finite K/V of token 1 at positions 0..47).

Off (QWEN_FAST_SOLO_PACKED unset or '0'), `wanted()` is 0, `install` wraps nothing, nothing imports this module on the serving path, and every path is what it was. Stdlib only; py 3.7 syntax.
"""

import os

SOLO_PACKED_FLAG = 'QWEN_FAST_SOLO_PACKED'
# serving_c2_contract.MODEL_ENTRY and the class and method the plugin's runner calls (vllm_tt_plugin model_runner.py _allocate_kv_caches, the legacy uniform API); pinned by test_idle_scratch.
MODEL_ENTRY = 'models.demos.blackhole.qwen36.tt.qwen36_vllm'
MODEL_CLASS = 'Qwen36ForCausalLM'
ALLOCATOR = 'allocate_kv_cache'
# One scratch page is two idle targets (its two 32-row tile rows): page 0's two plus these two serve the three idle segments of a lone user in a 4-user block. A page more is
# two targets more (the octo block's 5 live seats would need two pages), and is a decision for the day something asks for it.
PAGES_WANTED = 1
MARKER = '[NP1]'
INSTALLED_LINE = MARKER + ' kv scratch: {}.{} allocates {} block(s) beyond the pool vLLM sized (pool {} blocks, scratch ids {}..{})'
CLAIMED_LINE = MARKER + ' kv scratch claimed: pool pages {}, scratch pages {}'
WRAPPED_ATTRIBUTE = '__np1_kv_scratch__'

# (first scratch block id, count) as the allocation recorded them in THIS process; None before any allocation.
_RECORD = [None]


def wanted(environ=None):
    """The scratch pages this process's KV allocation must add: 0 unless QWEN_FAST_SOLO_PACKED=1, else PAGES_WANTED. Strict: unset or '0' is off, '1' is on, anything else is a
    configuration error (serving_octo.solo_packed_requested's rule)."""
    value = (os.environ if environ is None else environ).get(SOLO_PACKED_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (SOLO_PACKED_FLAG, value))
    return PAGES_WANTED if value == '1' else 0


def record(first, count):
    """Note the allocation: `count` blocks from block id `first` (= vLLM's pool size) are scratch."""
    if type(first) is not int or type(count) is not int or first < 2 or count < 1:
        raise ValueError('A scratch allocation is a count of 1 or more blocks from vLLM\'s pool size (at least 2), got first=%r count=%r' % (first, count))
    _RECORD[0] = (first, count)


def forget():
    """Drop the record (a test's, or a process re-initialising its cache)."""
    _RECORD[0] = None


def reserved():
    """The scratch block ids the allocation recorded, () when none."""
    if _RECORD[0] is None:
        return ()
    first, count = _RECORD[0]
    return tuple(range(first, first + count))


def claim(physical_pages, environ=None):
    """The scratch block ids of a cache of `physical_pages` blocks, or ValueError. Called by the cache owner at the attach with QWEN_FAST_SOLO_PACKED=1: the cache must be vLLM's pool and
    exactly the pages `wanted()` asks for, recorded by this process's allocation. A cache that was not extended (the hook did not run, or ran in another process) fails the attach here,
    before any block is built, instead of a lone-user round writing into a block a request owns."""
    count = wanted(environ)
    if not count:
        return ()
    pages = reserved()
    if len(pages) != count:
        raise ValueError('%s=1 needs %d KV scratch page(s) beyond vLLM\'s pool and the allocation recorded %s: the model\'s allocate_kv_cache was not extended in this process (%s.%s.%s, armed at '
                         'boot by serving_c2_contract)' % (SOLO_PACKED_FLAG, count, list(pages) or 'none', MODEL_ENTRY, MODEL_CLASS, ALLOCATOR))
    if type(physical_pages) is not int or pages[-1] + 1 != physical_pages:
        raise ValueError('The KV cache holds %r blocks and the scratch pages %s are not its last: the cache is not vLLM\'s pool of %d blocks plus the scratch pages' % (
            physical_pages, list(pages), pages[0]))
    return pages


def extended_shape(shape, count):
    """`shape` (the (blocks, kv_heads, block_size, head_size) vLLM sized the cache at) with `count` blocks more."""
    shape = tuple(shape)
    if len(shape) != 4 or type(shape[0]) is not int or shape[0] < 2:
        raise ValueError('A paged KV cache shape is (blocks, kv_heads, block_size, head_size), got %r' % (shape,))
    return (shape[0] + count,) + shape[1:]


def install(module, environ=None, log=None):
    """The PostImportHook callback on the model entry: wrap MODEL_CLASS.ALLOCATOR so the caches it allocates hold `wanted()` blocks more than the shape it is given, and record them.
    True when the method is wrapped (now or already); False when the flag is off (nothing is touched). A model entry without the class or the method raises: the profile asked for a
    scratch page this image cannot give, and a boot that serves without it would be the flag mounted and never executed."""
    count = wanted(environ)
    if not count:
        return False
    cls = getattr(module, MODEL_CLASS, None)
    original = getattr(cls, ALLOCATOR, None)
    if cls is None or not callable(original):
        raise ValueError('%s=1 needs %s.%s.%s (the legacy uniform KV allocation), which this model entry does not define' % (SOLO_PACKED_FLAG, MODEL_ENTRY, MODEL_CLASS, ALLOCATOR))
    if getattr(original, WRAPPED_ATTRIBUTE, False):
        return True
    emit = log if log is not None else _emit

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers, *arguments, **keywords):
        shape = extended_shape(kv_cache_shape, count)
        record(shape[0] - count, count)
        emit(INSTALLED_LINE.format(MODEL_CLASS, ALLOCATOR, count, shape[0] - count, shape[0] - count, shape[0] - 1))
        return original(self, shape, dtype, num_layers, *arguments, **keywords)

    setattr(allocate_kv_cache, WRAPPED_ATTRIBUTE, True)
    allocate_kv_cache.__wrapped__ = original
    allocate_kv_cache.__name__ = getattr(original, '__name__', ALLOCATOR)
    allocate_kv_cache.__doc__ = getattr(original, '__doc__', None)
    setattr(cls, ALLOCATOR, allocate_kv_cache)
    return True


def _emit(text):
    """One line into the server log (loguru where it exists, stdout otherwise). Never raises: a failed report must not fail an allocation."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:
        pass
