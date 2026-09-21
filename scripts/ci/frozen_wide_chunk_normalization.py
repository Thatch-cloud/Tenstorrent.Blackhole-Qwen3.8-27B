"""65536-only geometry and factory patch for the frozen-recipe numerical
probe: dedicated-scratch-CB FP32 reciprocal (frozen_wide_chunk_scratch.py,
ported from dspark_ladder_normalization.py's hardware-validated candidate,
run 34797353681) plus 1024-key online-softmax chunking
(dspark_ladder_geometry.py:14), recomputed under the frozen recipe's own
context+256 capacity basis instead of the ladder module's output_tokens=1024
basis:

    storage_keys = (65536 + 256) + 64 = 65856      (unchanged - same formula
                                                      as frozen_context_geometry.py)
    key_chunk    = 1024                             (dspark_ladder_geometry.py:14)
    padded_keys  = ceil(65856 / 1024) * 1024 = 66560
    Skt          = padded_keys // 32 = 2080         (NOT the currently-generated
                                                      2064 [256-key chunk, the
                                                      unfixed/failing geometry],
                                                      and NOT the ladder's 2112
                                                      [a different, larger
                                                      capacity basis])
    Sk_chunk_t   = key_chunk // 32 = 32
    iterations   = padded_keys // key_chunk = 65    (was 258 at the standard
                                                      256-key chunk)

adapt(sources, context) is a pure no-op for every context other than 65536:
it never imports or edits frozen_context_geometry.py (so factory_selector()'s
output - and therefore every OTHER context's generated
dspark_fp32_intermediates.py text, staged files, build-cache key and built
binary - is provably unaffected by this module), dspark_ladder_normalization.py
(the already-hardware-qualified ladder candidate), or
dflash_combined_sim_runtime.py (T16-pinned; see
dflash_t16_native_attention_gate.py:14-17). All edits below target only the
65536-context staged copies of dspark-native-8k-attention-probe.py,
dspark_attention_chunk_trial.py, dspark_stats_pack.py,
dspark_fp32_intermediates.py and frozen_sim_build_cache.py, and add
frozen_wide_chunk_scratch.py itself to the 65536-context staged tree.

UNVALIDATED: this is new source-generation, not yet built, simulated or run
on hardware. The construction here has been checked for structural
correctness (unique anchors against the actual staged file text, and a
standalone round-trip simulation of the factory_scope()/validate_manifest()
reconstruction logic in frozen_wide_chunk_scratch.py) but the generated C++
has not been compiled. Required next step, in order:

  1. context-build-65536-v1 (build-only: QWEN_FROZEN_BUILD_ONLY=1) - confirms
     the widened factory condition and the scratch-CB substitutions compose
     and compile cleanly against the pinned SOURCE_SHA256 factory, and that a
     fresh (non-stale) build-cache entry is produced.
  2. experiment/frozen-64k-reciprocal-replay-v3 (or a fresh numerical-only
     tag) - the actual numerical requalification this port targets.
  3. target-replay and diagnostics lanes, then the combined-runtime evidence
     chain, only after (2) passes. See the CB-budget note below: this port
     does not touch the target-replay probe's kernel at all (different
     program factory - sdpa_decode, not sdpa - so it can neither fix nor
     worsen that probe's separate CB overflow).
"""

CONTEXT = 65536

CAPACITY = CONTEXT + 256                                  # 65792 - unchanged formula
STORAGE_KEYS = CAPACITY + 64                               # 65856 - unchanged formula
KEY_CHUNK = 1024                                            # dspark_ladder_geometry.py:14
PADDED_KEYS = -(-STORAGE_KEYS // KEY_CHUNK) * KEY_CHUNK     # 66560
SKT = PADDED_KEYS // 32                                     # 2080
SK_CHUNK_T = KEY_CHUNK // 32                                # 32
ITERATIONS = PADDED_KEYS // KEY_CHUNK                        # 65
ADDED_MASKED_POISON_ROWS = PADDED_KEYS - STORAGE_KEYS        # 704

# Sanity-check the derivation at import time, so a future edit that breaks
# the arithmetic fails loudly instead of silently staging inconsistent
# geometry.
assert CAPACITY == 65792
assert STORAGE_KEYS == 65856
assert PADDED_KEYS == 66560
assert SKT == 2080
assert SK_CHUNK_T == 32
assert ITERATIONS == 65
assert ADDED_MASKED_POISON_ROWS == 704
assert STORAGE_KEYS % KEY_CHUNK != 0 and PADDED_KEYS % KEY_CHUNK == 0, (
    'padded_keys must be an exact multiple of key_chunk for chunking to tile evenly')


def _once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Wide-chunk normalization anchor missing or ambiguous (expected exactly 1): '
            + repr(before[:96]))
    return source.replace(before, after, 1)


def _patch_probe(source):
    """dspark-native-8k-attention-probe.py, post adapt_probe_sources(): make
    the two bare report-dict literals (key_chunk_size, native_padded_keys,
    added_masked_poison_rows) reflect the wide-chunk geometry instead of the
    standard 256-key one."""
    before = ("kernel_audit=kernel_audit, key_chunk_size=256, native_padded_keys=SHAPE['padded_keys'],\n"
        "        added_masked_poison_rows=192,")
    after = (f"kernel_audit=kernel_audit, key_chunk_size={KEY_CHUNK}, native_padded_keys={PADDED_KEYS},\n"
        f"        added_masked_poison_rows={ADDED_MASKED_POISON_ROWS},")
    source = _once(source, before, after)
    # Route both validate_manifest call sites through frozen_wide_chunk_scratch's
    # wrapper (factory_scope()-aware) instead of dspark_fp32_build's plain one,
    # so the scratch-CB substitutions are correctly seen and reversed by the
    # probe's own runtime reconciliation call (a separate process from the
    # build phase; see frozen_wide_chunk_scratch.validate_manifest's docstring).
    source = _once(source,
        "from dspark_fp32_build import validate_manifest\n        build = validate_manifest(",
        "from frozen_wide_chunk_scratch import validate_manifest\n        build = validate_manifest(")
    source = _once(source,
        "from dspark_fp32_build import validate_manifest\n        report['factory_build'] = validate_manifest(",
        "from frozen_wide_chunk_scratch import validate_manifest\n        report['factory_build'] = validate_manifest(")
    return source


def _patch_chunk_trial(source):
    """dspark_attention_chunk_trial.py, post adapt_probe_sources(): 1024-key
    chunk width and the wide-chunk padded-key count. Everything downstream
    (key/value/mask padding amounts, k_chunk_size passed into
    SDPAProgramConfig) already derives from these two module-level names, so
    no further edit is needed in this file."""
    source = _once(source, 'KEY_CHUNK = 256\nfrom frozen_context_geometry',
        f'KEY_CHUNK = {KEY_CHUNK}\nfrom frozen_context_geometry')
    source = _once(source, "PADDED_KEYS = SHAPE['padded_keys']", f'PADDED_KEYS = {PADDED_KEYS}')
    return source


def _patch_stats_pack(source):
    """dspark_stats_pack.py, post adapt_probe_sources(): the build-time
    static_assert must check the SAME (Skt, Sk_chunk_t) pair the op call
    actually instantiates, or the 65536 build fails to compile outright."""
    before = "get_compile_time_arg_val(3) == {selected_geometry()['padded_keys'] // 32} && get_compile_time_arg_val(8) == 8,"
    after = f'get_compile_time_arg_val(3) == {SKT} && get_compile_time_arg_val(8) == {SK_CHUNK_T},'
    source = _once(source, before, after)
    source = _once(source, '"8K diagnostic must use 8704 keys and 256-key chunks");',
        f'"8K diagnostic must use {PADDED_KEYS} keys and {KEY_CHUNK}-key chunks");')
    return source


def _patch_fp32_intermediates(source):
    """dspark_fp32_intermediates.py, post adapt_probe_sources(): OR in a
    (Skt, Sk_chunk_t) branch for the 65536 rung alongside the untouched
    factory_selector()-driven clause every other context still uses (with
    Sk_chunk_t == 8, unchanged). factory_selector() itself is not called
    differently and its own output is not altered."""
    before = '        {factory_selector()} && Sq_chunk_t == 1 && Sk_chunk_t == 8 &&'
    after = ('        (({factory_selector()} && Sk_chunk_t == 8) ||\n'
        f'        (Skt == {SKT} && Sk_chunk_t == {SK_CHUNK_T})) && Sq_chunk_t == 1 &&')
    return _once(source, before, after)


def _patch_build_cache(source):
    """frozen_sim_build_cache.py (copied verbatim from the working tree for
    every context today): wrap both the build-time and the (separate,
    same-process) post-build validate_manifest call in
    frozen_wide_chunk_scratch.factory_scope(), and add the new module to the
    content-addressed cache key's tracked builders so a change to it alone
    still invalidates stale cache entries."""
    source = _once(source, 'import dspark_fp32_build as baseline\n',
        'import dspark_fp32_build as baseline\nimport frozen_wide_chunk_scratch\n')
    source = _once(source,
        "                ('dspark_fp32_build.py', 'dspark_fp32_intermediates.py',\n"
        "                 'frozen_sim_build_cache.py', 'frozen_binary_cache.py', 'dspark_hardware_gate.py')},",
        "                ('dspark_fp32_build.py', 'dspark_fp32_intermediates.py',\n"
        "                 'frozen_sim_build_cache.py', 'frozen_binary_cache.py', 'dspark_hardware_gate.py',\n"
        "                 'frozen_wide_chunk_scratch.py')},")
    source = _once(source,
        "    with patch.object(baseline.subprocess, 'run', run):\n        baseline.main()\n",
        "    with frozen_wide_chunk_scratch.factory_scope(), patch.object(baseline.subprocess, 'run', run):\n"
        "        baseline.main()\n")
    source = _once(source,
        "    baseline.validate_manifest(root, '/experiment/results/dspark-fp32-build.json')\n",
        "    with frozen_wide_chunk_scratch.factory_scope():\n"
        "        baseline.validate_manifest(root, '/experiment/results/dspark-fp32-build.json')\n")
    return source


def adapt_wide_chunk_normalization(sources, context):
    """Entry point called from frozen_recipe_context.main(). No-op unless
    context == 65536: every other context's sources dict is returned
    unchanged (same keys, same values, same objects), so no staged file,
    generated patch or downstream build-cache key for those contexts can
    differ from before this module existed."""
    if context != CONTEXT:
        return dict(sources)
    result = dict(sources)
    result['dspark-native-8k-attention-probe.py'] = _patch_probe(result['dspark-native-8k-attention-probe.py'])
    result['dspark_attention_chunk_trial.py'] = _patch_chunk_trial(result['dspark_attention_chunk_trial.py'])
    result['dspark_stats_pack.py'] = _patch_stats_pack(result['dspark_stats_pack.py'])
    result['dspark_fp32_intermediates.py'] = _patch_fp32_intermediates(result['dspark_fp32_intermediates.py'])
    result['frozen_sim_build_cache.py'] = _patch_build_cache(result['frozen_sim_build_cache.py'])
    from pathlib import Path
    result['frozen_wide_chunk_scratch.py'] = Path(__file__).with_name('frozen_wide_chunk_scratch.py').read_text()
    for name, source in result.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    return result
