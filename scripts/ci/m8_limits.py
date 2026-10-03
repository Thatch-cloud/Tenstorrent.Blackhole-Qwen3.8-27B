"""The census of every 64-row / 4-user limit between the packed verify block and one 128-row block of eight users (tp4/m8-phase1).

block128-design.md section 2 listed about a dozen; block128-review.md found seven more; building the twins found the rest (rows 24, 30-33
below). Each row says WHERE the limit is (a file and a regular expression that must still match it: `locate` turns it into file:line, and a
row whose code moved fails test_tp4_m8_limits until it is re-read), whether the file is PINNED (its bytes are hashed by recorded evidence, so
it is never edited: the fix is a rebind at runtime), and what phase 1 did:

  accepts   the code already takes 128 rows / 8 users; the test calls it at that size.
  rebound   tp4_m8.install() replaces the limit with a twin that logs an engaged marker (or, for a table, widens it); the test calls the
            code before and after: refused before (the flag-off bytes), accepted after, and the marker present.
  phase2    still refuses at 128 / 8; the test asserts the refusal (so phase 2 cannot land silently), and `action` is the phase 2 work.

`python m8_limits.py` prints the table (markdown) with the line numbers read from the tree. Stdlib only, py 3.7; nothing here is served.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GRAFT = 'docker/qwen-c2-graft/graft'

ACCEPTS, REBOUND, PHASE2 = 'accepts', 'rebound', 'phase2'

# Pin kinds. 'json' rows name the recorded pin file(s) that list the file; 'doc' rows are pinned by a document or a qualified-kernel rule
# the code cannot check; None is not pinned.
EVIDENCE = 'frozen evidence (dspark-*.json, frozen-ladder-corpus.json)'
PACKED_ANY = 'packed_any_evidence_tp4.json'
PAIR_K5 = 'pair module of the qualified K5 launch (docs/tp4-recurrence-split.md: do not edit)'
IMAGE = 'the image\'s own model source (tp_common / model.py are sha256-pinned on the serving path), not in this repo'


def row(ident, where, file, needle, status, action, pinned=None, count=1, source='design', pin_via=None):
    """`pin_via` names the pinned scripts/ci file that makes this one untouchable when it is not itself listed (row 20: the evidence-pinned
    reader calls the function that carries the limit)."""
    return dict(id=ident, where=where, file=file, needle=needle, status=status, action=action, pinned=pinned, count=count, source=source,
                pin_via=pin_via)


LIMITS = (
    # --- shape, admission, scheduling -------------------------------------------------------------------------------------
    row(1, 'packed shape: legal block widths', 'scripts/ci/packed_shapes.py', r'^BLOCK_ROWS = ', REBOUND,
        'table widened to 128 (serving_buffer_pool aliases the same tuple: row 2); the M8 shape is tp4_m8.m8_shape'),
    row(2, 'buffer pool: legal widths (alias of packed_shapes.BLOCK_ROWS) and its top width', 'scripts/ci/serving_buffer_pool.py',
        r'^from packed_shapes import BLOCK_ROWS as PACKED_BLOCK_WIDTHS', REBOUND,
        'alias and PACKED_BLOCK_ROWS rebound; the pool still allocates only its replay shapes (row 3)'),
    row(3, 'buffer pool: replay shapes (2, 16) and (4, 16), extent storage and placement per block', 'scripts/ci/serving_buffer_pool.py',
        r'^PACKED_REPLAY_SHAPES = ', PHASE2, 'add (8, 16): extent storage for 8 users (8x the bundle tables and cur_pos words), the lowest free slot, one block built whole'),
    row(4, 'serving shape for a scheduler request count', 'scripts/ci/packed_shapes.py', r'^SERVING_SHAPES = ', PHASE2,
        'a count of 8 takes M8 only under QWEN_FAST_M8_BLOCK, refused beside QWEN_FAST_M3_BLOCKS=2; wired with the runtime (row 5)'),
    row(5, 'runtime: M3 blocks logic, two-phase build, place_blocks, the eight-seat count', 'scripts/ci/serving_runtime.py',
        r'^M3_BLOCKS_USERS = 8', PHASE2, 'one block built whole (no two-phase), carries_in_place kept, request-width warm once before the capture'),
    row(6, 'admission: the extent path is the M3 shape', 'scripts/ci/packed_any_admission.py', r'^def check_environment', PHASE2,
        'admit the M8 shape (and re-run CB2b only if an evidence-pinned file changes: it does not, see row 12)'),
    row(7, 'packed step: per-block rounds and the padded rule', 'scripts/ci/serving_packed_step.py', r'^def pads\(block, count\):', PHASE2,
        'one block of 8: packed_device_step; eligibility follows the padded rule of row 8'),
    row(8, 'idle-segment cap: page 0 holds two idle segments, so one 8-user block serves 6-8 live users', 'scripts/ci/packed_verifier.py',
        r'^    MAX_IDLE_SEGMENTS = 2', PHASE2, 'sink pages (S1) or shared idle tile rows (S4), with the v188 idle-exactness gate repeated at 128 rows'),
    row(9, 'packed rows total (literal tuple inside the function)', 'scripts/ci/target_packed_pages.py', r'if total not in \(1, 2, 4, 8, 16, 32, 64\):', REBOUND,
        'twin tp4_m8.validate_users (the body with 128 added; the test holds the two sources equal modulo that tuple)', source='phase 1'),
    row(10, 'packed participants fill one legal width (literal tuple inside the function)', 'scripts/ci/verifier_pack.py',
        r'block_rows not in \(1, 2, 4, 8, 16, 32, 64\)', REBOUND, 'twin tp4_m8.build_pack (same drift guard)', source='phase 1'),
    row(11, 'sequential capture rows: the trim is keyed on block_rows == 64', 'scripts/ci/packed_shapes.py', r'if shape\.block_rows == M3_BLOCK_ROWS else',
        REBOUND, 'twin answers 4 for the 128-row block, so no per-request engine captures T16 beside it (the v52 OOM pattern)', source='review 7'),
    row(12, 'verifier engine: verify widths and block widths', 'scripts/ci/verifier_engine.py', r'^VERIFY_WIDTHS = ', REBOUND,
        'VERIFY_WIDTHS and BLOCK_WIDTHS widened at runtime, and verifier_engine_tp (which imports VERIFY_WIDTHS by name) follows (the file is in the serving bundle inventory and the frozen evidence)', pinned=EVIDENCE,
        source='phase 1'),
    row(13, 'packed policy: block widths and the native GDN slots', 'scripts/ci/serving_fast_policy.py', r'^PACKED_BLOCK_WIDTHS = ', REBOUND,
        'PACKED_BLOCK_WIDTHS widened (packed_geometry(8, 128) then gives 16 rows and 15 proposals a user)', source='phase 1'),
    row(14, 'drafter proposal block widths (32 or 64)', 'scripts/ci/dflash_packed_proposal.py', r'^BLOCK_WIDTHS = \(32, 64\)', PHASE2,
        'not hit while the quads propose in 64-row passes over rows 0-63 and 64-127 of one tap set; C1 must show it unreached', source='phase 1'),
    # --- the model forward at 128 rows ------------------------------------------------------------------------------------
    row(15, 'model_batch: BLOCK_WIDTHS and validate_checkpoint', 'scripts/ci/model_batch.py', r'^BLOCK_WIDTHS = ', REBOUND,
        'table widened at runtime (the file is held byte for byte by test_quad_draft\'s inventory and the frozen evidence)', pinned=EVIDENCE),
    row(16, 'model_batch: native_m3 detection by the _64 attribute (the engines, binders and the native_attn guard)', 'scripts/ci/model_batch.py',
        r"native_m3 = hasattr\(getattr\(model, 'args', None\), 'attn_wo_decode_1d_progcfg_64'\)", PHASE2,
        'native_m8 detection by the _128 attributes through a tp_addresses-style twin; the binders log engaged counts', pinned=EVIDENCE, count=2),
    row(17, 'gdn_prefix: ROW_WIDTHS, called by the pinned gdn_device_loop_state.decode', 'scripts/ci/gdn_prefix.py', r'^ROW_WIDTHS = ', REBOUND,
        'table widened at runtime (attach refused in the warm forward without it)', pinned=EVIDENCE, source='review 1'),
    row(18, 'gdn_device_loop_state.project_qkvzab_by_tile: cap 64, silently four 32-row calls above it', 'scripts/ci/gdn_device_loop_state.py',
        r'^    cap = 64 if native_m3 else TILE', REBOUND,
        'twin: one native call when the graft has the _128 configs, else the tiled path AND a fell-back line (+8.4 to 9.1 ms a verify, no marker today)',
        pinned=EVIDENCE, source='review 2'),
    row(19, 'gdn_records: BLOCK_ROWS (RetainedGDNBlock)', 'scripts/ci/gdn_records.py', r'^BLOCK_ROWS = ', REBOUND, 'table widened at runtime',
        pinned=EVIDENCE, source='review 3'),
    row(20, 'extent reader: validate_segments limit 64 (bound as a default where it is defined)', 'scripts/ci/pooled_attention_replay.py',
        r'^def validate_segments\(segments, block_rows_limit=PACKED_BLOCK_ROWS\)', REBOUND,
        'twin passes 128 and marks eight segments; the reader (extent_attention_replay_tp) is evidence-pinned and untouched', pinned=PACKED_ANY,
        source='review 4', pin_via='scripts/ci/extent_attention_replay_tp.py'),
    row(21, 'force_argmax: SAMPLE_WIDTHS (sample_tiles is generic per 32-row tile)', 'scripts/ci/force_argmax.py', r'^SAMPLE_WIDTHS = ', REBOUND,
        'table widened at runtime (four in-trace sampler calls at 128 rows)', pinned=EVIDENCE),
    row(22, 'graft matmul gates: rows <= 2 * TILE select the _64 configs, above it the prefill arms (mlp.py)', GRAFT + '/mlp.py',
        r'<= 2 \* ttnn\.TILE_SIZE', PHASE2, 'lever_n_m3native_patch M = 128 section: gates <= 4 * TILE, select _64 or _128 by rows, new graft.sha256', count=2),
    row(23, 'graft matmul gates (gdn/tp.py)', GRAFT + '/gdn/tp.py', r'<= 2 \* tpc\.TILE_SIZE', PHASE2, 'same section', count=3),
    row(24, 'graft matmul gates (attention/tp.py)', GRAFT + '/attention/tp.py', r'<= 2 \* (ttnn|tpc)\.TILE_SIZE', PHASE2, 'same section', count=3),
    row(25, 'graft model_config: the seven _64 configs (the _128 siblings do not exist)', GRAFT + '/model_config.py', r'^        M64 = 64', PHASE2,
        'seven *_progcfg_128 (per_core_M 4) from the SAME builder, asserting in0_block_w, fuse_batch, mcast_in0 equal the _64 ones: m8_matmul_plan holds '
        'the arithmetic and B1 is the byte gate'),
    row(26, 'LM head: ttnn.linear auto program config (K blocking may change with M)', 'image: model.py ttnn.linear(x, lm_head_weight)', None, PHASE2,
        'B1 byte-compares M = 128 against 64-row halves and 32-row tiles; pin the 64-row choice if it differs', pinned=IMAGE),
    row(27, 'norms at 128 rows (two_tile_norm: block_h = rows / 32)', 'scripts/ci/two_tile_norm.py', r'^def validate_two_tile_rows', ACCEPTS,
        'none: generic (block_h 4); rms_norm sharded vs per tile is a card byte check (B5)'),
    row(28, 'AttnPrep (K64 graft) is qualified at batch 64 only (512 instances over 110 cores at 128 is an unqualified kind mix)',
        'scripts/ci/two_tile_decode.py', r'^def prep_by_tile\(', PHASE2, 'run the qualified batch-64 op on each 64-row half; q and gate join by one Concat each'),
    row(29, 'NLPConcatHeadsDecode: one input core per user (128 users need 128 cores of 110)', 'scripts/ci/two_tile_decode.py',
        r'^class TwoTileConcatHeads', PHASE2, 'two 64-user calls'),
    row(30, 'K/V write: LAUNCH_ROWS (64, 32) and the rows != 64 refusal', 'scripts/ci/packed_ordered_cache.py', r'^LAUNCH_ROWS = \(64, 32\)', PHASE2,
        'halves mode: two chained64 launches per cache over rows 0-63 and 64-127 (disjoint tile rows, per-half metadata); B4 on card M'),
    row(31, 'K/V write launch rows flag grammar', 'scripts/ci/verify_trace_t2.py', r'^KV_ROWS = \(64, 32\)', PHASE2, 'a halves spelling beside 64 and 32'),
    row(32, 'tile-split all-reduce at 128 rows (four 32-row tiles)', 'scripts/ci/tile_collective_tp.py', r'^def tile_spans', ACCEPTS,
        'none: generic; 512 reduce-scatters a round either way'),
    # --- GDN at eight users -----------------------------------------------------------------------------------------------
    row(33, 'K5-A users: the pinned pair module', 'scripts/ci/gdn_user_batch.py', r'^MAX_USERS = 4', REBOUND,
        'the pair file stays 4; the TP4 sibling carries the bound (row 34)', pinned=PAIR_K5),
    row(34, 'K5-A users: the TP4 sibling inherits the pair\'s bound', 'scripts/ci/gdn_user_batch_tp.py', r'^MAX_USERS = pair\.MAX_USERS', REBOUND,
        'MAX_USERS = 8 at runtime: 8 x 12 = 96 distinct points on 11 x 10; a new program instance, so the P0 compare at 8 users is a card gate (B2)',
        source='design'),
    row(35, 'K5-A / V1 users bound in the conv path (reads gdn_seq_block.batch, the TP4 sibling at four cards)', 'scripts/ci/gdn_user_batch_conv.py',
        r'gdn_seq_block\.batch\.MAX_USERS', ACCEPTS, 'none after row 34', count=2),
    row(36, 'T2 packed windows bound by the PAIR module\'s MAX_USERS (and so V1 block conv falls back)', 'scripts/ci/gdn_conv_windows_packed.py',
        r'^    if not 1 <= len\(users\) <= gdn_user_batch\.MAX_USERS', REBOUND,
        'twin asks the original about each group of four users and answers the first reason', source='review 5'),
    row(37, 'rows DMA: 256 runtime words a core, 8 a task: at most 31 tasks a core, 3,410 on 110 cores', 'scripts/ci/gdn_rows_dma_tp.py',
        r'^MAX_ARGUMENT_WORDS = 256', REBOUND, 'launch twin splits an oversize list into equal launches (the 3,600-task unstack becomes 1,800 + 1,800)',
        source='review 6'),
    row(38, 'unstack destination layout hard-codes four users (conv 0-3, beta 4-7, g 8-11, z 12-15, windows 16 + 4u): at eight, conv 4-7 lands on beta',
        'scripts/ci/gdn_block_conv_tp.py', r'add\(rows_dma\.unstack_users\(users, 1, 0, 1\), 1, 4\)', REBOUND,
        'twin derives every base from the user count (the original\'s list at 1-4 users); in neither the design nor the review', source='phase 1'),
    row(39, 'GDN split / canon / merge planners at 8 users (1,032 / 516 / 192 tasks)', 'scripts/ci/gdn_rows_dma_tp.py', r'^def split_pieces', ACCEPTS,
        'none: within one launch\'s capacity; V2 / V1 at 8 users are card-checked (B6)'),
    row(40, 'attention fold planners at 8 bundles', 'scripts/ci/attention_block_fold_tp.py', r'^def chunks_of', ACCEPTS, 'none: within capacity'),
    # --- tooling ----------------------------------------------------------------------------------------------------------
    row(41, 'ops profile plan: users', 'scripts/ci/ops_profile_plan.py', r'^def plan_users', ACCEPTS,
        'none: an eight-seat profile plans eight users (tp4/262k8); C6 profiles the M8 recipe on one block'),
    row(42, 'K5 card harness users (gdn_tp4_card_test USERS = 4; the TP2 gdn_seq_block_device_test --users 1-4)', 'scripts/ci/gdn_tp4_card_test.py',
        r'^USERS = 4', PHASE2, 'extend the UB and K5 sections to --users 8 (B2)', source='review'),
)


def read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read().replace(chr(13) + chr(10), chr(10))


def pin_kind(entry):
    """'json' when the entry's pin is a recorded manifest the code can check, 'doc' when a document or the image holds it, None."""
    if not entry['pinned']:
        return None
    return 'doc' if entry['pinned'] in (PAIR_K5, IMAGE) else 'json'


def locate(entry, root=ROOT):
    """The (first line number, how many lines match) of the entry's needle in its file, or (None, 0) for an image-side row."""
    if entry['needle'] is None:
        return None, 0
    lines = read(os.path.join(root, entry['file'])).split(chr(10))
    found = [number + 1 for number, line in enumerate(lines) if re.search(entry['needle'], line)]
    return (found[0] if found else None), len(found)


def pin_files(root=ROOT):
    """{pin file: text} of the recorded pin manifests the census checks 'pinned' against."""
    out = {}
    base = os.path.join(root, 'scripts', 'ci')
    for folder, _dirs, files in os.walk(base):
        for name in files:
            if name.endswith('.json') and (name.startswith('dspark-') or name in ('tp2_pinned_sources.json', 'frozen-ladder-corpus.json')
                                           or name.startswith('packed_any_evidence')):
                out[name] = read(os.path.join(folder, name))
    return out


def listed_in_pins(file, pins):
    """The names of the pin files that list this scripts/ci file by name."""
    name = os.path.basename(file)
    pattern = re.compile(r'["/]' + re.escape(name) + r'"\s*[:,]')
    return sorted(key for key, text in pins.items() if pattern.search(text))


def markdown(root=ROOT):
    out = ['| # | limit | file:line | pinned | phase-1 action |', '|---|---|---|---|---|']
    for entry in LIMITS:
        line, _count = locate(entry, root)
        where = entry['file'] + (':%d' % line if line else '')
        pinned = 'yes: ' + entry['pinned'] if entry['pinned'] else 'no'
        action = '%s: %s' % (entry['status'], entry['action'])
        out.append('| %d | %s | `%s` | %s | %s |' % (entry['id'], entry['where'], where, pinned, action))
    return chr(10).join(out)


if __name__ == '__main__':
    sys.stdout.write(markdown() + chr(10))
    counts = {}
    for entry in LIMITS:
        counts[entry['status']] = counts.get(entry['status'], 0) + 1
    sys.stdout.write(json.dumps(counts, sort_keys=True) + chr(10))
