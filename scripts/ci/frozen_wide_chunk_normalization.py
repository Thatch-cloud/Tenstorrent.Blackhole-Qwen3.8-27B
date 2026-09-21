"""65536-only geometry and factory patch for the frozen-recipe numerical
probe: dedicated-scratch-CB FP32 reciprocal (frozen_wide_chunk_scratch.py,
ported from dspark_ladder_normalization.py's hardware-validated candidate,
run 34797353681) plus 1024-key online-softmax chunking
(dspark_ladder_geometry.py:14), recomputed under the frozen recipe's own
context+256 capacity basis instead of the ladder module's output_tokens=1024
basis. The padded-key count - and therefore Skt - is selectable, since two
candidates are now live:

    storage_keys = (65536 + 256) + 64 = 65856   (unchanged - same formula as
                                                   frozen_context_geometry.py;
                                                   never selectable)
    key_chunk    = 1024                          (dspark_ladder_geometry.py:14;
                                                   fixed regardless of Skt choice)

    Skt == 2080: padded_keys = ceil(65856/1024)*1024 = 66560, 65 iterations,
        704 masked/poisoned padding keys. This port's own first derivation,
        under the frozen recipe's context+256 capacity basis. NOT the value
        that has ever passed on hardware.
    Skt == 2112: padded_keys = 67584, 66 iterations, 1728 masked/poisoned
        padding keys. The ladder's own Skt (dspark_ladder_geometry.py,
        output_tokens=1024 capacity basis - a different, larger capacity than
        this probe's 15-proposal fixture uses, which is why padded_keys
        differs from the ladder's own 67584-under-its-own-storage_keys even
        though the numeric Skt matches by construction: choosing this branch
        keeps THIS probe's storage_keys/positions/capacity at their existing
        65856/[65536,65777]/65792 values and only changes how many keys the
        kernel pads out to and masks - a kernel-internal padding change, not
        a change to what's logically being tested). This IS a value that has
        passed on hardware (run 34797353681) - but not with only this port's
        fix; see the module docstring on the sum_update/score_center gap.

Selectable via QWEN_FROZEN_65536_SKT (read once per adapt call, at staging
time - frozen_recipe_context.py runs on the CI host, not inside the
simulator container, so this is a plain os.environ lookup, not a runtime
knob like frozen_wide_chunk_replay.py's). Accepted values '2080' and '2112'
(as strings); default '2112' as of the v4 lane, since that's the value with
an actual zero-failure hardware result behind it (docs/numerics-65536-
attention.md section 5). Anything else refuses loudly. Exactly one Skt
branch is added to the generated dspark_fp32_intermediates.py REPLACEMENT
text per staging - whichever this resolves to - never both.

KNOWN GAP, not closed by this knob: run 34797353681's own probe
(dspark-ladder-attention-probe.py) applies scalar_sum_update()
(dspark_ladder_sum_update.py) and scalar_score_center(key_tiles=2112)
(dspark_ladder_score_center.py) ALONGSIDE scratch_normalization() - three
separate precision fixes, not one. Both of the other two are themselves
gated on `get_compile_time_arg_val(3) == 2112` in their own C++ text, exactly
like the scratch-CB fix this module ports - but neither is staged or applied
by this module at all. Selecting Skt==2112 here makes the *scratch-CB*
fix's condition match that hardware-validated value; it does NOT bring the
sum-update or score-centering fixes along, since those live in modules this
port never touches. See docs/numerics-65536-attention.md section 6 for what
they each do and why their absence is a plausible, unverified explanation
for a residual that this Skt swap alone might not close.

adapt(sources, context) is a pure no-op for every context other than 65536:
it never imports or edits frozen_context_geometry.py (so factory_selector()'s
output - and therefore every OTHER context's generated
dspark_fp32_intermediates.py text, staged files, build-cache key and built
binary - is provably unaffected by this module), dspark_ladder_normalization.py
/ dspark_ladder_sum_update.py / dspark_ladder_score_center.py (the already-
hardware-qualified ladder candidates), or dflash_combined_sim_runtime.py
(T16-pinned; see dflash_t16_native_attention_gate.py:14-17). All edits below
target only the 65536-context staged copies of
dspark-native-8k-attention-probe.py, dspark_attention_chunk_trial.py,
dspark_stats_pack.py, dspark_fp32_intermediates.py and
frozen_sim_build_cache.py, and add frozen_wide_chunk_scratch.py itself
(with its own SKT constant patched to match the resolved value) to the
65536-context staged tree.

UNVALIDATED at Skt==2080 (this port's own first derivation): not run on
hardware or shown correct beyond the simulator numerical probe, which it
still fails (9 residual elements, run 35593521063,
experiment/frozen-64k-reciprocal-replay-v3). Skt==2112 alone (this fix, no
sum-update/score-center) is equally unvalidated - it has not itself been
built or run; only the *combination* of all three ladder fixes together at
that Skt has a passing hardware result.
"""

import os
from pathlib import Path


CONTEXT = 65536

CAPACITY = CONTEXT + 256                # 65792 - unchanged formula, never selectable
STORAGE_KEYS = CAPACITY + 64             # 65856 - unchanged formula, never selectable
KEY_CHUNK = 1024                         # dspark_ladder_geometry.py:14 - fixed regardless of Skt choice
SK_CHUNK_T = KEY_CHUNK // 32             # 32

KNOB = 'QWEN_FROZEN_65536_SKT'
ACCEPTED_SKT = (2080, 2112)
DEFAULT_SKT = 2112


def geometry_for_skt(skt):
    if skt not in ACCEPTED_SKT:
        raise ValueError(f"Explicit {' or '.join(map(str, ACCEPTED_SKT))} Skt required, got {skt!r}")
    padded_keys = skt * 32
    if padded_keys % KEY_CHUNK != 0 or padded_keys <= STORAGE_KEYS:
        raise ValueError('padded_keys must be an exact multiple of key_chunk, and cover storage_keys, '
            'for chunking to tile evenly and the fixture to fit')
    return dict(skt=skt, padded_keys=padded_keys, key_chunk=KEY_CHUNK, sk_chunk_t=SK_CHUNK_T,
        iterations=padded_keys // KEY_CHUNK, poison_rows=padded_keys - STORAGE_KEYS)


# Sanity-check both accepted values at import time, so a future edit that
# breaks the arithmetic (or the accepted-value list) fails loudly instead of
# silently staging inconsistent geometry.
assert geometry_for_skt(2080) == dict(skt=2080, padded_keys=66560, key_chunk=1024, sk_chunk_t=32,
    iterations=65, poison_rows=704)
assert geometry_for_skt(2112) == dict(skt=2112, padded_keys=67584, key_chunk=1024, sk_chunk_t=32,
    iterations=66, poison_rows=1728)


def resolve_skt():
    value = os.environ.get(KNOB, str(DEFAULT_SKT))
    if value not in tuple(map(str, ACCEPTED_SKT)):
        raise ValueError(f"Explicit {'/'.join(map(str, ACCEPTED_SKT))} {KNOB} required, got {value!r}")
    return int(value)


def manifest_geometry_override(context):
    """Called from frozen_recipe_context.main() to keep frozen-geometry.json's
    recorded padded_keys/key_chunk honest for a 65536 staging, instead of
    silently reporting the standard-chunk formula the staged files no longer
    use (the cosmetic gap flagged in docs/numerics-65536-attention.md section
    1). None for every other context - an explicit no-op, not an empty dict,
    so the caller can tell 'nothing to override' from 'override with
    nothing'."""
    if context != CONTEXT:
        return None
    geometry = geometry_for_skt(resolve_skt())
    return dict(padded_keys=geometry['padded_keys'], key_chunk=geometry['key_chunk'])


def _once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Wide-chunk normalization anchor missing or ambiguous (expected exactly 1): '
            + repr(before[:96]))
    return source.replace(before, after, 1)


def _patch_probe(source, geometry):
    """dspark-native-8k-attention-probe.py, post adapt_probe_sources(): make
    the two bare report-dict literals (key_chunk_size, native_padded_keys,
    added_masked_poison_rows) reflect the wide-chunk geometry instead of the
    standard 256-key one."""
    before = ("kernel_audit=kernel_audit, key_chunk_size=256, native_padded_keys=SHAPE['padded_keys'],\n"
        "        added_masked_poison_rows=192,")
    after = (f"kernel_audit=kernel_audit, key_chunk_size={geometry['key_chunk']}, "
        f"native_padded_keys={geometry['padded_keys']},\n"
        f"        added_masked_poison_rows={geometry['poison_rows']},")
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
    # Wire the three kernel-level fixes into main()'s context-manager chain,
    # in the ladder's own order (dspark-ladder-attention-probe.py:77-85:
    # scalar_reciprocal, scalar_sum_update, scalar_score_center,
    # scratch_normalization), inserted immediately before the untouched,
    # pre-existing scoped_stats_pack(). Deliberately targets the bare
    # `scoped_stats_pack(),` call site, not `with scalar_reciprocal(),
    # scoped_stats_pack(),` - context==65536 staging runs unconditionally,
    # for every flag combination frozen_recipe_context.py accepts, and not
    # every 65536 lane passes --scalar-reciprocal (the real
    # experiment/frozen-64k-target-replay-* lanes do not - only the
    # reciprocal-eager/replay/diagnostics ones do,
    # .github/workflows/qwen-frozen-32k-numerical.yml's sequential, non-elif
    # if-blocks). Targeting the wider anchor would raise "anchor missing" for
    # every target-replay-only staging, breaking lanes this port does not
    # even need to touch dspark-native-8k-attention-probe.py's main() for
    # (target-replay runs target-t16-attention-8k-probe.py instead; this
    # file just needs to stage without crashing). The bare anchor works
    # identically either way: when scalar_reciprocal() precedes it (the
    # common case for lanes that exercise this port's fixes at all), the
    # three new ones land between it and scoped_stats_pack(), matching the
    # ladder's order exactly; when it does not, they still compose cleanly,
    # just without scalar_reciprocal() preceding them (a combination that
    # never actually executes this probe as the container's main entry
    # point, since QWEN_SIM_CASE routes elsewhere for those lanes).
    source = _once(source, '    from dspark_stats_pack import scoped_stats_pack\n',
        '    from dspark_stats_pack import scoped_stats_pack\n'
        '    from frozen_wide_chunk_sum_update import sum_update_scope\n'
        '    from frozen_wide_chunk_score_center import score_center_scope\n'
        '    from frozen_wide_chunk_scratch import kernel_scope\n')
    source = _once(source, 'scoped_stats_pack(),',
        'sum_update_scope(), score_center_scope(), kernel_scope(), scoped_stats_pack(),')
    return source


def _patch_chunk_trial(source, geometry):
    """dspark_attention_chunk_trial.py, post adapt_probe_sources(): 1024-key
    chunk width and the wide-chunk padded-key count. Everything downstream
    (key/value/mask padding amounts, k_chunk_size passed into
    SDPAProgramConfig) already derives from these two module-level names, so
    no further edit is needed in this file."""
    source = _once(source, 'KEY_CHUNK = 256\nfrom frozen_context_geometry',
        f"KEY_CHUNK = {geometry['key_chunk']}\nfrom frozen_context_geometry")
    source = _once(source, "PADDED_KEYS = SHAPE['padded_keys']", f"PADDED_KEYS = {geometry['padded_keys']}")
    return source


def _patch_stats_pack(source, geometry):
    """dspark_stats_pack.py, post adapt_probe_sources(): the build-time
    static_assert must check the SAME (Skt, Sk_chunk_t) pair the op call
    actually instantiates, or the 65536 build fails to compile outright."""
    before = "get_compile_time_arg_val(3) == {selected_geometry()['padded_keys'] // 32} && get_compile_time_arg_val(8) == 8,"
    after = f"get_compile_time_arg_val(3) == {geometry['skt']} && get_compile_time_arg_val(8) == {geometry['sk_chunk_t']},"
    source = _once(source, before, after)
    source = _once(source, '"8K diagnostic must use 8704 keys and 256-key chunks");',
        f"\"8K diagnostic must use {geometry['padded_keys']} keys and {geometry['key_chunk']}-key chunks\");")
    return source


def _patch_fp32_intermediates(source, geometry):
    """dspark_fp32_intermediates.py, post adapt_probe_sources(): OR in a
    (Skt, Sk_chunk_t) branch for the 65536 rung alongside the untouched
    factory_selector()-driven clause every other context still uses (with
    Sk_chunk_t == 8, unchanged). factory_selector() itself is not called
    differently and its own output is not altered. Exactly one such branch is
    added - whichever geometry['skt'] resolved to - never both at once."""
    before = '        {factory_selector()} && Sq_chunk_t == 1 && Sk_chunk_t == 8 &&'
    after = ('        (({factory_selector()} && Sk_chunk_t == 8) ||\n'
        f"        (Skt == {geometry['skt']} && Sk_chunk_t == {geometry['sk_chunk_t']})) && Sq_chunk_t == 1 &&")
    return _once(source, before, after)


KERNEL_FIX_MODULES = ('frozen_wide_chunk_scratch.py', 'frozen_wide_chunk_sum_update.py',
    'frozen_wide_chunk_score_center.py')


def _patch_skt_constant(source, geometry):
    """Each of KERNEL_FIX_MODULES' own SKT constant, so the kernel-level
    substitutions it applies are gated on the SAME Skt value this staging
    resolved to, not a stale default."""
    return _once(source, 'SKT = 2080', f"SKT = {geometry['skt']}")


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


def _patch_target_probe(source):
    """target-t16-attention-8k-probe.py, post frozen_target_replay.
    adapt_target_probe() (only present in `sources` when --target-replay was
    passed): redirect its compact-scratch branch's validate_manifest import
    the same way _patch_probe does for dspark-native-8k-attention-probe.py.

    Without this, QWEN_FROZEN_TARGET_SCRATCH=1 (the frozen-64k-target-replay-
    scratch-* lanes) calls the plain, unwrapped dspark_fp32_build.
    validate_manifest (frozen_target_replay.py:77,81 - injected by
    adapt_target_probe(), not by this module), which cannot reconstruct a
    scratch-CB-patched sdpa_program_factory.cpp: its reconstruction check
    (restore the condition-text swap, then re-transform and compare) leaves
    the scratch-CB layer in place, so the intermediate "original" never
    matches SOURCE_SHA256, and dspark_fp32_intermediates.transform()'s own
    guard raises 'Exact pinned SDPA factory required' (run 35598585759) -
    exactly the class of bug frozen_wide_chunk_scratch.factory_scope() was
    built to fix for the numerical probe; this port simply never redirected
    the target probe's own, separate import of the same function.

    Confirmed safe to fix this way rather than needing a per-context
    expected-factory-hash mechanism: sdpa_tree_scratch.py's own pins
    (HASHES / PATCHED_FACTORY_SHA256, sdpa_tree_scratch.py:9-13,20) are
    entirely about ttnn/.../sdpa_decode/device/sdpa_decode_program_factory.cpp
    - the DECODE kernel family attention_replay.py's ReplayAttentionReader
    actually computes through (paged_scaled_dot_product_attention_decode) -
    a completely different file from sdpa_program_factory.cpp, which this
    port's wide-chunk patch touches. The compact-scratch target probe's own
    numerics and scratch CB layout (chunk sizes, CB ids) are therefore not
    coupled to sdpa_program_factory.cpp at all; its validate_manifest call
    here is a build-provenance audit of a binary the target probe doesn't
    otherwise depend on, not a source of correctness assumptions about it."""
    return _once(source, "        from dspark_fp32_build import validate_manifest\n",
        "        from frozen_wide_chunk_scratch import validate_manifest\n")


def adapt_wide_chunk_normalization(sources, context):
    """Entry point called from frozen_recipe_context.main(). No-op unless
    context == 65536: every other context's sources dict is returned
    unchanged (same keys, same values, same objects), so no staged file,
    generated patch or downstream build-cache key for those contexts can
    differ from before this module existed - regardless of the
    QWEN_FROZEN_65536_SKT knob's value, which is only ever consulted below
    this check."""
    if context != CONTEXT:
        return dict(sources)
    geometry = geometry_for_skt(resolve_skt())
    result = dict(sources)
    result['dspark-native-8k-attention-probe.py'] = _patch_probe(
        result['dspark-native-8k-attention-probe.py'], geometry)
    result['dspark_attention_chunk_trial.py'] = _patch_chunk_trial(
        result['dspark_attention_chunk_trial.py'], geometry)
    result['dspark_stats_pack.py'] = _patch_stats_pack(result['dspark_stats_pack.py'], geometry)
    result['dspark_fp32_intermediates.py'] = _patch_fp32_intermediates(
        result['dspark_fp32_intermediates.py'], geometry)
    result['frozen_sim_build_cache.py'] = _patch_build_cache(result['frozen_sim_build_cache.py'])
    # Only present when --target-replay was passed (frozen_target_replay.
    # adapt_target_probe(), applied earlier in frozen_recipe_context.main()'s
    # pipeline, is what stages this file at all).
    if 'target-t16-attention-8k-probe.py' in result:
        result['target-t16-attention-8k-probe.py'] = _patch_target_probe(
            result['target-t16-attention-8k-probe.py'])
    for name in KERNEL_FIX_MODULES:
        module_source = Path(__file__).with_name(name).read_text()
        result[name] = _patch_skt_constant(module_source, geometry)
    for name, source in result.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    return result
