"""65536-only diagnostic lever for the target-replay CB overflow (run
35585801822, program.cpp:1868, 1,600,448 B against a 1,572,864 B L1 budget):
make attention_replay.py's hardcoded k_chunk_size=256 (ReplayAttentionReader.__init__,
attention_replay.py:53-54) a runtime knob, QWEN_FROZEN_65536_REPLAY_K_CHUNK
(default '128', accepted '64'/'128'/'256' - 256 is the unchanged control, for
an A/B against the current failing default).

Checked the other two files this lever might also need to touch, per the
task: attention_parallel.py never references k_chunk_size or any chunk-width
literal - it passes `config` through opaquely to
paged_scaled_dot_product_attention_decode (attention_parallel.py:17-19).
pooled_attention_replay.py's three occurrences of the literal 256
(MAX_FAMILY_CAPACITY's step, family_start's position-offset formula, and
PACKED_QUERY_WIDTH, the model's head_dim) are all unrelated to k_chunk_size -
confirmed by reading the file in full, not by pattern-matching the digits.
So attention_replay.py:54 is the only site that needs to change.

CAVEAT, restated from the port report (docs/numerics-65536-attention.md and
the port thread): the bounding in docs/sdpa-batch64-plan.md's "CB arithmetic
detail" section argues the dominant Skt-scaled overflow term is more likely
the dense attn_mask buffer (attention_replay.py:50-51, shaped
(batches, 1, rows*12, capacity) - full capacity width, not chunk width) than
anything k_chunk_size-scoped. This lever is a diagnostic as much as a
candidate fix: if QWEN_FROZEN_65536_REPLAY_K_CHUNK=64 does not close the
27,584 B gap, that is itself the answer (points at the mask buffer / a
QWEN_SDPA_TREE_SCRATCH_ROUNDS-class compact-scratch lever instead), not a
failure of this change.

PINNING HAZARD, worth flagging even though it does not block this lane:
pooled_attention_replay.py:3-9 documents that attention_replay.py's bytes are
frozen-recipe evidence, hashed by admission checks that have historically
refused a run outright when this file changed ("Combined runtime component
source differs: attention_replay.py", run 35495227738). The only gate this
change is verified against here - frozen_target_replay.validate_target_report,
via target_t16_attention_8k_gate.SOURCES, which does include
'attention_replay.py' - is a same-run self-consistency check (report['sources']
must equal report['sources_after']), not a comparison against a pinned
external hash, so it is unaffected by staging a different 65536-only variant.
But any FUTURE attempt to feed 65536 target-replay evidence into a gate that
DOES pin attention_replay.py's hash (the combined-runtime/serving-image
admission pooled_attention_replay.py's docstring describes) would need the
same wrap-don't-modify treatment that module already establishes as the
house pattern for this exact problem, not a direct edit.

adapt_replay_k_chunk(sources, context, checkout) is a pure no-op for every
context other than 65536, and touches no file this module doesn't explicitly
list - attention_replay.py is not otherwise staged by frozen_recipe_context.py
at all today (it stays at its pristine 8c102b20 checkout content, untouched,
for every context), so leaving it out of `sources` for context != 65536
reproduces that exact behaviour rather than approximating it.
"""

from pathlib import Path


CONTEXT = 65536
KNOB = 'QWEN_FROZEN_65536_REPLAY_K_CHUNK'
DEFAULT = '128'
ACCEPTED = ('64', '128', '256')


def _once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Replay k-chunk anchor missing or ambiguous (expected exactly 1): ' + repr(before[:96]))
    return source.replace(before, after, 1)


def _patch_attention_replay(source):
    source = _once(source,
        '        grid = mesh.compute_with_storage_grid_size()\n',
        '        grid = mesh.compute_with_storage_grid_size()\n'
        f"        _qwen_replay_k_chunk = os.environ.get('{KNOB}', '{DEFAULT}')\n"
        f'        if _qwen_replay_k_chunk not in {ACCEPTED!r}:\n'
        f"            raise ValueError('Explicit {'/'.join(ACCEPTED)} replay K-chunk width required')\n"
        '        _qwen_replay_k_chunk = int(_qwen_replay_k_chunk)\n')
    source = _once(source, 'k_chunk_size=256)', 'k_chunk_size=_qwen_replay_k_chunk)')
    return source


def adapt_replay_k_chunk(sources, context, checkout):
    """checkout: the --checkout directory (already verified by
    frozen_recipe_context.main() to be a clean checkout at exactly REVISION,
    before this or any other adapter runs). attention_replay.py is read
    directly from it rather than via `git show`, since that upfront
    verification already guarantees its on-disk content is the pinned
    historical bytes for any file no earlier adapter has touched."""
    if context != CONTEXT:
        return dict(sources)
    if 'attention_replay.py' in sources:
        raise ValueError('attention_replay.py unexpectedly already staged before this adapter runs')
    result = dict(sources)
    path = Path(checkout) / 'scripts/ci' / 'attention_replay.py'
    if not path.is_file():
        raise ValueError('attention_replay.py missing from checkout (a real checkout at REVISION always '
            'has it): ' + str(path))
    original = path.read_text()
    result['attention_replay.py'] = _patch_attention_replay(original)
    compile(result['attention_replay.py'], 'attention_replay.py', 'exec')
    return result
