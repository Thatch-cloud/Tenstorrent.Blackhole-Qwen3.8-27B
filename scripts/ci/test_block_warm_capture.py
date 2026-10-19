"""Every block the combined candidate builds, warm forward then capture, behind the capture rules (tp4/ship-4; two card runs died here: 38117538847, 38124550905).

The engine builds each block's WARM fixture (placeholders on one tile row, K/V written as ONE chain over every row), runs its eager forward, builds the CAPTURED fixture, and only after the
last block's attach captures every block in turn. A capture may launch only programs some eager forward has run (the device's "Cannot load new binaries during trace capture"; the
program cache is the device's, shared by all blocks). A lever that decides a block's path from the warm fixture's collapsed spans, or from anything else the two fixtures do not share, breaks
that: the K/V page writer once did, for the octo block, and the layout that fixed it was first read from a pack the engine does not build (a dict; build_pack returns a list), so it was always None
in production while the CPU tests, which handed the writer a dict, passed.

THIS TEST drives the engine's own `PackedVerifierEngine.build_fixture` with the real `build_pack`, `participant` and `model_batch.validate_pack`, the real chained-writer options and writer
selection (`model_batch.chained_writer_options`, `cache_writer`) and the real K/V page writer, under the exact env of the candidate and of its audited twin, for every block of the s4 shape
(two 4 x 16 blocks and the 8 x 8 octo block) in every order, behind `capture_rules`. Only the model's layers are not built: a ModelBatch stand-in builds the K/V writers the way
`ModelBatch.__init__` does and its forward calls them.

THE LEVER SWEEP (the code that can differ between the warm fixture and the captured one is exactly what branches on what the two fixtures do not share: the writer's `spans`, built from
`kv_single_chain`, and the placeholders' values; test_the_sweep_names_the_levers_that_read_what_the_fixtures_do_not_share reads the source for them):

  reads the writer's spans/layout     kv_page_writer_tp4.attach (the octo decline; fixed), kv_page_model_tp4 (host model of the page writer: no device program), packed_cache_writer /
                                      packed_ordered_cache (the writers themselves: the warm chain is the same kernels and compile arguments by design)
  reads the placeholders' values      verify_trace_t2.kv_conflict (refuses a round, staging only), the extent readers' tables (staged per round, outside the capture)
  branches on `warm`                  packed_verifier.sample_shards (the sampler's eager prewarm: an extra eager launch, never a missing one)
  decides from the block's shape      octo_attn_bundle, octo_glue8, gdn_block_conv8_tp, extent_attention_octo_tp, sdpa_multi_tp / w2_switch (excludes the octo block): all read the
                                      reader's or the pack's segments, the same object in both fixtures
"""

import contextlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import capture_rules  # noqa: E402
from capture_rules import CaptureViolation  # noqa: E402
import kv_page_writer_tp4 as kvpw  # noqa: E402
import model_batch  # noqa: E402
import packed_verifier  # noqa: E402
from test_kv_page_writer_tp4 import FakeTTNN, KERNELS, WIDTH, block, mesh, tiles  # noqa: E402

PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
NS = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-'
# the combined candidate of tp4/ship-4, its audited twin and its no-grid control; the octo-3 candidate and the octo twins of the fused stack (the profiles this branch has)
CANDIDATES = tuple(name for name in (NS + 'traffic-s4', NS + 's4-audit', NS + 's4-nogrid13', NS + 'traffic-o3', NS + 'all-octo3', NS + 'all-octo3-audit') if name in PROFILES)
M3 = dict(users=4, rows_per_user=16)
OCTO = dict(users=8, rows_per_user=8)
LAYERS = 2
GDN_LAYERS = 48


class CapturingTTNN(FakeTTNN):
    def __init__(self, chips=1):
        FakeTTNN.__init__(self, chips)
        self.next_trace = 0

    def begin_trace_capture(self, device, cq_id=0):
        self.next_trace += 1
        return self.next_trace

    def end_trace_capture(self, device, trace, cq_id=0):
        return None

    def synchronize_device(self, device):
        return None


class WriterBatch(object):
    """ModelBatch as far as the K/V writers go: the pack validated by the real validate_pack, the chained writer options and the writer selection of ModelBatch.__init__ (model_batch.py:651-682),
    one writer per full-attention layer, and a forward that writes every layer's K and V through them (two_tile_decode / _decode_from_prep calls the writer once per cache)."""

    environment = None          # set by the scenario: (operations, device, tensors, call)

    def __init__(self, model, tokens, position, pages, helpers, checkpoint, rows, *, pack=None, kv_single_chain=False, **options):
        operations, device, tensors, self.call = type(self).environment
        self.pack = model_batch.validate_pack(pack)
        self.operations, self.tensors = operations, tensors
        chained = model_batch.chained_writer_options(self.pack, tiles(operations.target, WIDTH), tensors['positions'], tensors['pages'], single_chain=kv_single_chain)
        self.kv_single_chain = kv_single_chain
        self.writers = [model_batch.cache_writer(operations, device, KERNELS, ordered_cache=True, cache_tiles=tiles(operations.target, WIDTH), serial=lambda: None,
                                                 **({'chained': chained} if chained is not None else {}))
                        for unused in range(LAYERS)]

    def forward(self):
        for writer in self.writers:
            for cache, packed in zip(self.tensors['caches'], self.tensors['packed']):
                writer(cache, packed, **self.call)


def engine_of(shape, patched_batch):
    """A PackedVerifierEngine with exactly what build_fixture reads, and nothing else."""
    engine = object.__new__(packed_verifier.PackedVerifierEngine)
    users, rows = shape['users'], shape['rows_per_user']
    engine.model, engine.helpers = object(), object()
    engine.block_rows = users * rows
    engine.rows_per_user = rows
    engine.capture_position = 4096
    engine.checkpoints = [tuple(object() for unused in range(GDN_LAYERS)) for unused in range(users)]
    engine.carries = [tuple(object() for unused in range(GDN_LAYERS)) for unused in range(users)]
    engine.extent, engine.extent_storage, engine.replay_tables = False, None, None
    engine.mask_once, engine.replay_group_rows = False, 4
    engine.kv_chains_cut = True
    engine.placeholders = [SimpleNamespace(pages=torch.zeros((1, 8), dtype=torch.int32), position=4096) for unused in range(users)]
    return engine


def lifecycle(profile, order, audit_replays=0):
    """The engine's order of work for the blocks in `order` (a tuple of shapes): per block its warm fixture and eager forward, then its captured fixture; then every capture in turn."""
    environ = {key: str(value) for key, value in PROFILES[profile]['env'].items()}
    environ.setdefault('QWEN_FAST_VERIFY_T2', '1')              # (the served container's own setting; the card log says kv_chains=32)
    lines = []
    with contextlib.ExitStack() as stack:
        for patch in (mock.patch.dict(os.environ, environ), mock.patch.object(kvpw, 'log_line', lines.append), mock.patch.object(kvpw, 'evidence_problems', return_value=[]),
                      mock.patch.object(kvpw, '_NOTED', set()), mock.patch.object(kvpw, '_REGISTRY', []), mock.patch.object(kvpw, '_LAYERS', [0]),
                      mock.patch.object(packed_verifier, 'ModelBatch', WriterBatch)):
            stack.enter_context(patch)
        raw = CapturingTTNN(chips=1)
        operations = capture_rules.guard(raw)
        operations.target = raw
        device = mesh(chips=1, grid=(13, 10))
        tensors = block(raw, source='l1')
        WriterBatch.environment = (operations, device, tensors, dict(update_idxs_tensor=SimpleNamespace(shape=(64,)), page_table=SimpleNamespace(shape=(64, WIDTH))))
        built = []
        for shape in order:
            engine = engine_of(shape, WriterBatch)
            warm = engine.build_fixture(engine.placeholders, warm=True)
            warm.forward()
            captured = engine.build_fixture(engine.placeholders)
            built.append((shape, warm, captured))
        escapes = []
        for shape, warm, captured in built:
            trace = operations.begin_trace_capture(device)
            try:
                captured.forward()
            except CaptureViolation as error:
                escapes.append((shape, error))
            finally:
                operations.end_trace_capture(device, trace)
    return SimpleNamespace(rules=operations.rules, built=built, escapes=escapes, lines=lines)


class EveryBlockTests(unittest.TestCase):
    ORDERS = sorted(set(itertools.permutations((4, 4, 8))))            # by users: the two 4 x 16 blocks and the 8 x 8 octo block, every order

    def test_every_block_of_the_s4_shape_captures_only_programs_the_warm_forwards_ran_in_every_order(self):
        for profile in CANDIDATES:
            for users in self.ORDERS:
                order = tuple(OCTO if count == 8 else M3 for count in users)
                with self.subTest(profile=profile[len(NS):], order=list(users)):
                    found = lifecycle(profile, order)
                    self.assertEqual(found.rules.problems(), [])
                    self.assertEqual(found.escapes, [])
                    self.assertTrue(found.rules.captured_launches)
                    for shape, warm, captured in found.built:
                        octo = shape['users'] == 8
                        with self.subTest(shape=shape):
                            for writer in warm.writers + captured.writers:
                                self.assertIs(writer.page is None, octo, 'the octo block keeps the chained writer in BOTH forwards; the 4 x 16 blocks take the page writer in both')

    def test_the_layout_the_engine_hands_the_writer_is_the_packs_segments(self):
        """The first fix read `pack['segments']`, which build_pack's list does not have: the layout was None in production. The layout is the running sum of the users' rows."""
        seen = []
        original = kvpw.attach

        def spy(*args, **kwargs):
            seen.append((tuple(kwargs['spans']), kwargs.get('layout_spans')))
            return original(*args, **kwargs)

        with mock.patch.object(kvpw, 'attach', spy):
            lifecycle(CANDIDATES[0], (M3, OCTO))
        warm_m3, captured_m3, warm_octo, captured_octo = seen[0], seen[LAYERS], seen[2 * LAYERS], seen[3 * LAYERS]
        self.assertEqual(warm_m3, (((0, 64),), ((0, 16), (16, 32), (32, 48), (48, 64))))
        self.assertEqual(captured_m3, (((0, 16), (16, 32), (32, 48), (48, 64)), ((0, 16), (16, 32), (32, 48), (48, 64))))
        self.assertEqual(warm_octo, (((0, 64),), kvpw.OCTO_SPANS))
        self.assertEqual(captured_octo, (kvpw.OCTO_SPANS, kvpw.OCTO_SPANS))

    def test_a_layout_that_is_none_in_the_warm_fixture_is_the_card_failure(self):
        """The negative control: the engine without its layout context (what card run 38124550905 ran) puts the octo block's warm forward on the page writer and its capture on the chained
        writer: the capture launches programs no eager forward ran."""
        import verify_trace_t2

        class Mute(object):
            def __init__(self, spans):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch.object(verify_trace_t2, 'block_layout', Mute):
            found = lifecycle(CANDIDATES[0], (M3, M3, OCTO))
        self.assertTrue(found.escapes)
        self.assertEqual([shape['users'] for shape, error in found.escapes], [8])
        self.assertIn('Cannot load new binaries during trace capture', str(found.escapes[0][1]))


class SweepTests(unittest.TestCase):
    def test_the_sweep_names_the_levers_that_read_what_the_fixtures_do_not_share(self):
        """The warm fixture differs from the captured one in the writer's spans (kv_single_chain) and in the placeholders' values, and the engine's eager sampler prewarm. Source readers of the
        first: only kv_page_writer_tp4 (and its host model) and the writers themselves branch on a writer's spans."""
        readers = []
        for path in sorted(HERE.glob('*.py')):
            if path.name.startswith('test_') or 'probe' in path.name or 'sweep' in path.name:
                continue
            text = path.read_text(encoding='utf-8')
            if re.search(r'\bspans\b', text) and ('kv_single_chain' in text or 'layout_spans' in text or 'launch_rows' in text):
                readers.append(path.name)
        self.assertEqual(readers, ['kv_page_writer_tp4.py', 'model_batch.py', 'packed_ordered_cache.py', 'verify_trace_t2.py'])
        single_chain = [path.name for path in sorted(HERE.glob('*.py')) if not path.name.startswith('test_') and 'kv_single_chain' in path.read_text(encoding='utf-8')]
        self.assertEqual(single_chain, ['model_batch.py', 'packed_verifier.py', 'verify_trace_t2.py'])
        # the levers that run inside the block's forward (the model's layers call them in the warm forward and in the capture alike) hold no warm-only branch: only the engine's own sampler prewarm does
        forward_levers = ('octo_attn_bundle.py', 'octo_glue8.py', 'gdn_block_conv8_tp.py', 'extent_attention_octo_tp.py', 'extent_attention_fold_tp.py', 'sdpa_multi_tp.py', 'sdpa_long_tp.py',
                          'w2_switch.py', 'tile_collective_tp.py', 'tp4_vglue.py', 'gdn_conv_gates_spread.py', 'ccl_options_tp.py', 'tp4_mlp_gateup.py', 'tp4_mlp_fused.py', 'tp4_shard_argmax.py',
                          'tp4_sampdraft.py', 'distributed_norm_gather_tp.py', 'draft_fusion_tp.py', 'grid13_tp.py', 'gdn_seq_block.py', 'gdn_user_batch.py', 'gdn_user_batch_conv.py',
                          'attention_block_fold_tp.py', 'octo_draft_tp.py')
        for name in forward_levers:
            with self.subTest(module=name):
                self.assertTrue((HERE / name).is_file(), name)
                self.assertIsNone(re.search(r'\bif warm\b|\bif not warm\b|\(warm and |warm=warm|kv_single_chain', (HERE / name).read_text(encoding='utf-8')), name)
        self.assertIn('warm and self.sampler_prewarm', (HERE / 'packed_verifier.py').read_text(encoding='utf-8'))

if __name__ == '__main__':
    unittest.main()
