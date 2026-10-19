"""NP1, the lone-user padded round (QWEN_FAST_SOLO_PACKED=1, tp4/np1-fastlane, docs/tp4-np1-fastlane.md): the host side, none of it needing a card.

What is held here, and what is not.

  - idle_scratch (the scratch KV page): the flag's strict reading; the allocation wrapper (it lengthens the shape the plugin's runner gives the model's allocate_kv_cache by one block and
    records where the scratch block starts), its idempotence, its refusals, its chain beside the prefix profiles' own PostImportHook on the model entry; the source pins it leans on (the
    plugin's runner takes the legacy uniform allocation, and the model defines it); the cache owner's claim and the pool bound it keeps for requests.
  - the REAL PackedVerifierEngine (test_packed_extent_block's fake two-chip device, the real extent readers): with a scratch page the 4-user block holds FOUR idle segments, builds with
    padded_min_users=1, and a lone user's round stages page 0's two tile rows and the scratch page's first for the three idle segments, with every (page, tile row) written by one segment
    only (the T2 guard, verify_trace_t2.kv_conflict); a live table holding the scratch page, or page 0, is refused as page 0 always was.
  - the routing at one, two, three, four, five, six and eight live seats over two M3 blocks and the octo block, flag on and off: what each seat count drafts at and where it runs.
  - the exactness argument as far as a CPU can show it: the flag changes nothing when at most two segments are idle (the staged inputs are byte for byte the flag-off block's), a live
    segment's staged inputs do not depend on how many segments are idle or where they write, and an emulated paged cache written by the staged rows shows the idle writes landing on
    page 0 and the scratch page only, never on a page a live row reads or writes.
  - flag off: the parent commit's idle_inputs, padded_refusal and describe are what they are today, for every live pattern.

What it cannot show (the card questions, docs/tp4-np1-fastlane.md section 6): that the cache tensors accept a block id past vLLM's pool (they are allocated that long), that the K/V
writers and the extent reader handle the scratch table, and that a lone user's text equals its per-request text byte for byte.
"""

import ast
from itertools import combinations
import os
from pathlib import Path
import random
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import idle_scratch
import packed_verifier
from packed_verifier import PackedVerifierEngine, m3_shape, page_zero_index
from serving_buffer_pool import PackedExtentStorage
import serving_c2_contract as contract
import serving_octo as octo
import serving_packed_step
from serving_packed_step import PackedStep, proposal_groups
import test_octo_block as octo_block_tests
import test_packed_extent_block as ext
import test_packed_verifier as base
import test_padded_block as padded_tests
import verify_trace_t2

HERE = Path(__file__).resolve().parent
FLAG = idle_scratch.SOLO_PACKED_FLAG
ON = {FLAG: '1'}
POOL = 16500                       # vLLM's block count in the ship-4 profile: the scratch block is the next id
SCRATCH = POOL
# The commit tp4/np1-fastlane was cut from (origin/tp4/swift): every flag-off comparison is against its modules.
PARENT = '27af6902'


class Lines:
    def __init__(self):
        self.lines = []

    def __call__(self, template, *values):
        self.lines.append(template.format(*values))


def fake_model_module():
    """A model entry whose Qwen36ForCausalLM has the legacy allocation, and the calls it received."""
    calls = []

    class Qwen36ForCausalLM:
        def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers, extra=None):
            """the original"""
            calls.append((tuple(kv_cache_shape), dtype, num_layers, extra))
            return ['caches']

    return types.SimpleNamespace(Qwen36ForCausalLM=Qwen36ForCausalLM), calls


class FlagAndRecordTests(unittest.TestCase):
    def setUp(self):
        idle_scratch.forget()
        self.addCleanup(idle_scratch.forget)

    def test_the_flag_is_strict_and_off_by_default(self):
        for environ in ({}, {FLAG: '0'}):
            self.assertEqual(idle_scratch.wanted(environ), 0)
        self.assertEqual(idle_scratch.wanted(ON), idle_scratch.PAGES_WANTED)
        self.assertEqual(idle_scratch.PAGES_WANTED, 1, 'one page: two more idle targets, the three idle segments of a lone user in a 4-user block')
        for value in ('', '2', 'yes', 'true', ' 1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'QWEN_FAST_SOLO_PACKED must be 0 or 1'):
                idle_scratch.wanted({FLAG: value})

    def test_the_names_are_the_ones_the_rest_of_the_stack_uses(self):
        self.assertEqual(idle_scratch.SOLO_PACKED_FLAG, octo.SOLO_PACKED_FLAG)
        self.assertEqual(idle_scratch.MODEL_ENTRY, contract.MODEL_ENTRY)
        self.assertIn(FLAG, contract.OCTO_GATE_ONLY, 'a traffic profile can never carry the flag')
        self.assertIn(FLAG, contract.OCTO_NAMES, 'an inherited value is removed by apply_environment')

    def test_the_record_claim_and_refusals(self):
        self.assertEqual(idle_scratch.reserved(), ())
        self.assertEqual(idle_scratch.claim(POOL + 1, {}), (), 'flag off: nothing is claimed, whatever was recorded')
        with self.assertRaisesRegex(ValueError, 'recorded none'):
            idle_scratch.claim(POOL + 1, ON)
        idle_scratch.record(POOL, 1)
        self.assertEqual(idle_scratch.reserved(), (SCRATCH,))
        self.assertEqual(idle_scratch.claim(POOL + 1, ON), (SCRATCH,))
        self.assertEqual(idle_scratch.claim(POOL + 1, {}), ())
        for physical in (POOL, POOL + 2, None, '16501'):
            with self.subTest(physical=physical), self.assertRaisesRegex(ValueError, 'not vLLM\'s pool'):
                idle_scratch.claim(physical, ON)
        for arguments in ((1, 1), (POOL, 0), ('a', 1), (POOL, '1'), (None, None)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                idle_scratch.record(*arguments)

    def test_the_extended_shape_adds_blocks_and_nothing_else(self):
        self.assertEqual(idle_scratch.extended_shape((POOL, 1, 64, 256), 1), (POOL + 1, 1, 64, 256))
        self.assertEqual(idle_scratch.extended_shape(torch.Size((POOL, 1, 64, 256)), 2), (POOL + 2, 1, 64, 256))
        for bad in ((POOL, 1, 64), (1, 1, 64, 256), ('a', 1, 64, 256), ()):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                idle_scratch.extended_shape(bad, 1)


class SourceTests(unittest.TestCase):
    def test_idle_scratch_is_stdlib_only_python_37_source_that_nothing_on_the_serving_path_imports_at_module_level(self):
        source = (HERE / 'idle_scratch.py').read_text(encoding='utf-8')
        tree = ast.parse(source, feature_version=(3, 7))
        top = {alias.name.split('.')[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
        top |= {node.module.split('.')[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
        self.assertEqual(top, {'os'})
        for name in ('serving_runtime', 'serving_cache_owner', 'serving_octo', 'serving_c2_contract', 'packed_verifier', 'serving_packed_step'):
            module = ast.parse((HERE / (name + '.py')).read_text(encoding='utf-8'))
            imported = {alias.name for node in module.body if isinstance(node, ast.Import) for alias in node.names}
            imported |= {node.module for node in module.body if isinstance(node, ast.ImportFrom)}
            self.assertNotIn('idle_scratch', imported, '%s imports it lazily, with the flag set only' % name)

    def test_the_flag_off_paths_import_nothing_new(self):
        for name, anchor in (('serving_cache_owner', "if os.environ.get('QWEN_FAST_SOLO_PACKED', '0') != '0':"), ('serving_runtime', "if os.environ.get('QWEN_FAST_SOLO_PACKED', '0') != '0':"),
                             ('serving_c2_contract', "if environ.get('QWEN_FAST_SOLO_PACKED', '0') != '0':")):
            self.assertIn(anchor, (HERE / (name + '.py')).read_text(encoding='utf-8'), name)


class AllocationHookTests(unittest.TestCase):
    def setUp(self):
        idle_scratch.forget()
        self.addCleanup(idle_scratch.forget)

    def test_the_wrapper_lengthens_the_shape_by_one_block_records_it_and_returns_the_originals_answer(self):
        module, calls = fake_model_module()
        original = module.Qwen36ForCausalLM.allocate_kv_cache
        lines = []
        self.assertTrue(idle_scratch.install(module, ON, log=lines.append))
        wrapped = module.Qwen36ForCausalLM.allocate_kv_cache
        self.assertIsNot(wrapped, original)
        self.assertIs(wrapped.__wrapped__, original)
        self.assertEqual((wrapped.__name__, wrapped.__doc__), ('allocate_kv_cache', 'the original'))
        model = module.Qwen36ForCausalLM()
        self.assertEqual(model.allocate_kv_cache((POOL, 1, 64, 256), 'bf8', 16), ['caches'])
        self.assertEqual(calls, [((POOL + 1, 1, 64, 256), 'bf8', 16, None)], 'the plugin\'s shape, one block longer')
        self.assertEqual(model.allocate_kv_cache((POOL, 1, 64, 256), 'bf8', 16, extra='x'), ['caches'])
        self.assertEqual(calls[-1][3], 'x', 'extra arguments pass through')
        self.assertEqual(idle_scratch.reserved(), (SCRATCH,))
        self.assertEqual(len(lines), 2)
        self.assertIn('[NP1] kv scratch: Qwen36ForCausalLM.allocate_kv_cache allocates 1 block(s) beyond the pool vLLM sized (pool 16500 blocks, scratch ids 16500..16500)', lines[0])

    def test_it_is_idempotent_and_off_without_the_flag(self):
        module, calls = fake_model_module()
        original = module.Qwen36ForCausalLM.allocate_kv_cache
        self.assertFalse(idle_scratch.install(module, {}))
        self.assertFalse(idle_scratch.install(module, {FLAG: '0'}))
        self.assertIs(module.Qwen36ForCausalLM.allocate_kv_cache, original, 'flag off: nothing is wrapped')
        self.assertTrue(idle_scratch.install(module, ON, log=lambda text: None))
        wrapped = module.Qwen36ForCausalLM.allocate_kv_cache
        self.assertTrue(idle_scratch.install(module, ON, log=lambda text: None))
        self.assertIs(module.Qwen36ForCausalLM.allocate_kv_cache, wrapped, 'a second install wraps nothing')
        module.Qwen36ForCausalLM().allocate_kv_cache((POOL, 1, 64, 256), 'bf8', 16)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], POOL + 1, 'one block more, not two')

    def test_a_model_entry_without_the_allocation_is_refused_and_a_bad_shape_fails_the_allocation(self):
        for module in (types.SimpleNamespace(), types.SimpleNamespace(Qwen36ForCausalLM=object())):
            with self.subTest(module=module), self.assertRaisesRegex(ValueError, 'needs models.demos.blackhole.qwen36.tt.qwen36_vllm.Qwen36ForCausalLM.allocate_kv_cache'):
                idle_scratch.install(module, ON)
        module, calls = fake_model_module()
        idle_scratch.install(module, ON, log=lambda text: None)
        with self.assertRaises(ValueError):
            module.Qwen36ForCausalLM().allocate_kv_cache((POOL, 1, 64), 'bf8', 16)
        self.assertEqual((calls, idle_scratch.reserved()), ([], ()), 'nothing allocated, nothing recorded')

    def test_the_hook_chains_beside_another_post_import_hook_on_the_same_module(self):
        """The prefix profiles already hook the model entry (log_model_tree): a second PostImportHook on the same name must still fire, in the order they were inserted."""
        name = 'np1_probe_model_entry'
        ran = []
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, name + '.py').write_text('class Qwen36ForCausalLM:\n    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):\n        return tuple(kv_cache_shape)\n', encoding='utf-8')
            before = list(sys.meta_path)
            sys.path.insert(0, directory)
            self.addCleanup(lambda: (sys.path.remove(directory), sys.modules.pop(name, None), sys.meta_path.__setitem__(slice(None), before)))
            sys.meta_path.insert(0, contract.PostImportHook(name, lambda module: ran.append('first')))
            sys.meta_path.insert(0, contract.PostImportHook(name, lambda module: (ran.append('second'), idle_scratch.install(module, ON, log=lambda text: None))))
            import importlib

            module = importlib.import_module(name)
            self.assertEqual(sorted(ran), ['first', 'second'])
            self.assertEqual(module.Qwen36ForCausalLM().allocate_kv_cache((POOL, 1, 64, 256), 'bf8', 16), (POOL + 1, 1, 64, 256))

    def test_the_boot_arms_the_hook_with_the_flag_and_only_with_it(self):
        before = list(sys.meta_path)
        self.addCleanup(lambda: sys.meta_path.__setitem__(slice(None), before))
        self.assertEqual(contract.arm_idle_scratch('p', {}), 0)
        self.assertEqual(contract.arm_idle_scratch('p', {FLAG: '0'}), 0)
        self.assertEqual(sys.meta_path, before)
        self.assertEqual(contract.arm_idle_scratch('p', ON), 1)
        added = [finder for finder in sys.meta_path if finder not in before]
        self.assertEqual(len(added), 1)
        self.assertIsInstance(added[0], contract.PostImportHook)
        self.assertEqual((added[0].name, added[0].callback), (idle_scratch.MODEL_ENTRY, idle_scratch.install))

    def test_the_plugin_runner_takes_the_legacy_allocation_and_the_model_defines_it(self):
        """Source pins: what the wrapper leans on. The runner allocates through the model's legacy allocate_kv_cache(shape, dtype, layers) when the model has no per-layer one."""
        runner = (HERE / 'fixtures' / 'vllm_tt_plugin_bf77cd63' / 'model_runner.py').read_text(encoding='utf-8')
        self.assertIn('if hasattr(self.model, "allocate_kv_cache_per_layer"):', runner)
        self.assertIn('return self.model.allocate_kv_cache(shape, dtype, len(per_layer_specs))', runner)
        self.assertIn('shape = self._kv_cache_shape(spec, kv_cache_config.num_blocks)', runner)
        tree = ast.parse((HERE / 'fixtures' / 'qwen36_vllm.py').read_text(encoding='utf-8'))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == idle_scratch.MODEL_CLASS]
        self.assertEqual(len(classes), 1)
        methods = {node.name: node for node in classes[0].body if isinstance(node, ast.FunctionDef)}
        self.assertIn(idle_scratch.ALLOCATOR, methods)
        self.assertNotIn('allocate_kv_cache_per_layer', methods, 'a per-layer allocation would take the other branch of the runner')
        self.assertEqual([argument.arg for argument in methods[idle_scratch.ALLOCATOR].args.args], ['self', 'kv_cache_shape', 'dtype', 'num_layers'])


class CacheOwnerTests(unittest.TestCase):
    def fixture(self, blocks):
        caches = [[SimpleNamespace(shape=(blocks, 1, 64, 256), dtype='bf8', addresses=(index * 2 + side + 1, index * 2 + side + 100))
                   for side in range(2)] for index in range(16)]
        layers = [SimpleNamespace(is_full_attention=True, attention=SimpleNamespace(paged_k=pair[0], paged_v=pair[1], use_paged=True)) for pair in caches]
        model = SimpleNamespace(num_devices=4, args=SimpleNamespace(max_batch_size=8), _paged_kv_caches=caches, layers=layers)
        runner = SimpleNamespace(model=SimpleNamespace(model=[model]), kv_caches=caches)
        operations = SimpleNamespace(bfloat8_b='bf8', get_device_tensors=lambda tensor: [SimpleNamespace(buffer_address=lambda value=value: value) for value in tensor.addresses])
        return operations, runner, model

    def setUp(self):
        import tp_shapes

        idle_scratch.forget()
        self.addCleanup(idle_scratch.forget)
        for name, value in (('chip_count', 4), ('active', SimpleNamespace(attn_kv_heads=1))):
            patcher = patch.object(tp_shapes, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def owner(self, blocks):
        from serving_cache_owner import ServingCacheOwner

        return ServingCacheOwner(*self.fixture(blocks))

    def test_flag_off_the_owner_is_what_it_was(self):
        owner = self.owner(POOL)
        self.assertEqual((owner.physical_pages, owner.scratch_pages), (POOL, ()))
        with patch.dict(os.environ, {FLAG: '0'}):
            self.assertEqual(self.owner(POOL).scratch_pages, ())

    def test_with_the_flag_the_cache_is_the_pool_plus_the_scratch_page_and_requests_see_the_pool(self):
        idle_scratch.record(POOL, 1)
        with patch.dict(os.environ, ON):
            owner = self.owner(POOL + 1)
        self.assertEqual((owner.physical_pages, owner.scratch_pages), (POOL, (SCRATCH,)), 'the bound a request\'s block ids are checked against stays vLLM\'s pool')
        owner.validate()

    def test_a_cache_that_was_not_extended_fails_the_attach_by_name(self):
        with patch.dict(os.environ, ON), self.assertRaisesRegex(ValueError, 'the model\'s allocate_kv_cache was not extended in this process'):
            self.owner(POOL)
        idle_scratch.record(POOL, 1)
        with patch.dict(os.environ, ON), self.assertRaisesRegex(ValueError, 'not vLLM\'s pool of 16500 blocks plus the scratch pages'):
            self.owner(POOL)
        with patch.dict(os.environ, {FLAG: 'x'}), self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
            self.owner(POOL)


class AdmissionTests(unittest.TestCase):
    M3_TWO = (True, 'users=8 FOUR_AS_TWO=0 PACKED_STEP=1 M3_BLOCKS=2')

    def setUp(self):
        idle_scratch.forget()
        self.addCleanup(idle_scratch.forget)

    def environment(self, **changes):
        from test_octo import solo_packed_environment

        return solo_packed_environment(**changes)

    def test_without_the_scratch_page_a_lone_user_is_refused_for_the_idle_segments_alone_as_before(self):
        log = Lines()
        with self.assertRaises(ValueError) as caught:
            octo.solo_packed_admission(self.M3_TWO, self.environment(), log=log)
        self.assertEqual(str(caught.exception).count('; '), 0, 'one reason')
        self.assertIn('a lone user leaves 3 idle segments of the 4-user block and a block holds 2', str(caught.exception))
        self.assertIn('padded_min_users below 2 at construction', str(caught.exception))
        self.assertIn('KV scratch page', str(caught.exception))

    def test_with_it_the_block_holds_four_idle_segments_and_the_lone_user_is_admitted(self):
        idle_scratch.record(POOL, 1)
        self.assertEqual(octo.scratch_pages(), (SCRATCH,))
        self.assertEqual((octo.idle_capacity(), octo.idle_capacity(1), octo.idle_capacity(2)), (2, 4, 6))
        log = Lines()
        record = octo.solo_packed_admission(self.M3_TWO, self.environment(), log=log)
        self.assertEqual(record, dict(min_users=1, scratch_pages=(SCRATCH,)))
        self.assertEqual(log.lines, ['[SOLO-PACKED] admitted min_users=1 scratch_pages=16500 (gate only)'])

    def test_the_rest_of_the_admission_is_unchanged(self):
        idle_scratch.record(POOL, 1)
        with self.assertRaisesRegex(ValueError, 'gate run of a gate-only'):
            octo.solo_packed_admission(self.M3_TWO, self.environment(**{octo.GATE_ENV: None}), log=Lines())
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_PADDED_BLOCK=\\(unset\\), not 1'):
            octo.solo_packed_admission(self.M3_TWO, self.environment(QWEN_FAST_PADDED_BLOCK=None), log=Lines())
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_SOLO_LANE=1'):
            octo.solo_packed_admission(self.M3_TWO, self.environment(QWEN_FAST_SOLO_LANE='1'), log=Lines())
        self.assertIsNone(octo.solo_packed_admission(self.M3_TWO, {}))
        self.assertEqual(octo.idle_capacity(), 2, 'the octo block\'s own rule reads the class constant: its six live seats are unchanged')


class FreshStorage:
    """A pool that lends the extent storage of a shape once per block (the ship-4 pool lends one per M3 block: packed_replicas), so a test can hold several blocks of one shape."""

    def fresh_storage(self, users, rows):
        def integers(shape):
            return self.ttnn.allocate(shape, 'int32', 'row_major', torch.zeros(shape, dtype=torch.int32))

        bundles = 2 if rows == 16 else 1
        return PackedExtentStorage(users, rows, [[integers((bundles, ext.WIDTH))] for user in range(users)], [[integers((bundles,))] for user in range(users)])


class LoneUserBlockTests(FreshStorage, ext.ExtentFixture):
    """The real 4-user M3 block (extent readers, fake device), with and without the scratch page."""

    def setUp(self):
        super().setUp()
        self.pool.packed_extent = self.fresh_storage

    def lone_block(self, **options):
        return self.build(padded_min_users=1, idle_scratch_pages=(SCRATCH,), **options)

    def staged(self, block):
        fixture = block.fixture
        return (fixture.tokens.value.clone(), fixture.positions.value.clone(), fixture.pages.value.clone())

    def test_the_block_holds_four_idle_segments_with_a_scratch_page_and_two_without(self):
        block = self.lone_block()
        self.assertEqual((block.padded_min_users, block.idle_scratch_pages, block.idle_capacity, block.MAX_IDLE_SEGMENTS), (1, (SCRATCH,), 4, 2))
        self.assertEqual([block.pads(count) for count in range(0, 6)], [False, True, True, True, False, False])
        self.assertTrue(any('max_idle=4 carries_in_place' in line and 'scratch_pages=16500' in line for line in self.lines if line.startswith(packed_verifier.PADDED_ADMITTED_MARKER)))
        plain = self.build(padded_min_users=2)
        self.assertEqual((plain.idle_scratch_pages, plain.idle_capacity), ((), 2))
        self.assertFalse(plain.pads(1))
        with self.assertRaisesRegex(ValueError, r'padded_min_users must be an integer in \[2, 4\) for a 4-user block'):
            self.build(padded_min_users=1)
        self.assertNotIn('scratch', ' '.join(line for line in self.lines if line.startswith(packed_verifier.PADDED_ADMITTED_MARKER) and 'min_users=2' in line))

    def test_the_scratch_pages_are_checked_at_construction(self):
        for pages in ((0,), (-1,), (SCRATCH, SCRATCH), ('7',), (1.0,), (True,)):
            with self.subTest(pages=pages), self.assertRaisesRegex(ValueError, 'idle_scratch_pages are distinct physical pages, each 1 or more'):
                self.build(padded_min_users=2, idle_scratch_pages=pages)
        two = self.build(padded_min_users=1, idle_scratch_pages=(SCRATCH, SCRATCH + 1))
        self.assertEqual(two.idle_capacity, 6)

    def test_a_lone_users_idle_segments_go_to_page_zero_row_zero_page_zero_row_one_and_the_scratch_page_row_zero(self):
        block = self.lone_block()
        for live in range(4):
            idle = block.idle_inputs((live,))
            segments = [segment for segment in range(4) if segment != live]
            self.assertEqual(sorted(idle), segments)
            starts = [idle[segment][1] for segment in segments]
            self.assertEqual(starts, [0, 32, 0], 'tile rows 0, 1 of page 0, then tile row 0 of the scratch page')
            tables = [idle[segment][2] for segment in segments]
            self.assertFalse(bool(tables[0].any() or tables[1].any()), 'page 0 for the first two')
            self.assertTrue(bool((tables[2] == SCRATCH).all()) and tuple(tables[2].shape) == (1, ext.WIDTH) and tables[2].dtype == torch.int32)
            self.assertEqual({idle[segment][0] for segment in segments}, {(1,) * 16})
            cells = [verify_trace_t2.kv_tile_rows(range(start, start + 16), table[0]) for start, table in zip(starts, tables)]
            self.assertEqual(cells, [{(0, 0)}, {(0, 1)}, {(SCRATCH, 0)}], 'three distinct (page, tile row) cells')
        self.assertEqual(len(block.idle_inputs(())), 4, 'no live segment at all: four idle segments, the block\'s whole capacity (verify refuses it through pads(), a block needs one live user)')
        self.assertEqual(block.idle_inputs((0, 1, 2, 3)), {})

    def test_a_lone_user_round_stages_the_user_and_three_idle_segments_and_commits_like_a_padded_round(self):
        block = self.lone_block()
        copies = len(self.ttnn.host_copies)
        predictions, metrics = block.verify(self.round((None, None, 70000, None)))
        self.assertGreater(len(self.ttnn.host_copies), copies)
        self.assertEqual((metrics['idle'], metrics['live'], metrics['segments']), ([0, 1, 3], 1, (2,)))
        self.assertEqual(len(predictions), 1)
        positions, pages = block.fixture.positions.value, block.fixture.pages.value
        self.assertEqual(positions[0:16].tolist(), list(range(0, 16)), 'page 0, tile row 0')
        self.assertEqual(positions[16:32].tolist(), list(range(32, 48)), 'page 0, tile row 1')
        self.assertEqual(positions[32:48].tolist(), list(range(70000, 70016)))
        self.assertEqual(positions[48:64].tolist(), list(range(0, 16)), 'the scratch page, tile row 0')
        self.assertFalse(bool(pages[0:32].any()))
        self.assertTrue(bool((pages[32:48] == 9).all()) and bool((pages[48:64] == SCRATCH).all()))
        self.assertIsNone(verify_trace_t2.kv_conflict(verify_trace_t2.block_users(positions, pages, 16, 4)))
        readers = self.readers(block)
        for segment, start in ((0, 0), (1, 32), (3, 0)):
            self.assertEqual(readers[segment].positions.value.tolist(), [start] + [0] * 7)
            self.assertEqual(readers[segment].cur_pos[0].value.tolist(), [255, 255], 'family E = 256')
        self.assertFalse(bool(readers[0].metadata[0][1].value.any() or readers[1].metadata[0][1].value.any()))
        self.assertTrue(bool((readers[3].metadata[0][1].value == SCRATCH).all()), 'the extent reader reads the scratch page\'s pages, not page 0\'s')
        self.assertTrue(any('packed padded round live=1 ' in line and 'idle=0,1,3' in line for line in self.lines))
        self.assertIn('[PINDIAG] packed extent round round=1 live=1 families=[2:70144] idle=[0,1,3] capped=[]', self.lines)
        self.assertEqual(block.pending_segments, {2})
        self.finish(block, prefix=5)
        self.assertEqual((block.phase, block.rounds, block.padded_rounds), ('idle', 1, 1))

    def test_the_lone_user_can_be_in_any_segment_and_the_round_repeats(self):
        block = self.lone_block()
        for round_number, segment in enumerate((3, 0, 1, 2, 2), 1):
            positions = [None] * 4
            positions[segment] = 5000 + 100 * segment
            block.verify(self.round(tuple(positions)))
            self.assertEqual(block.idle_segments, frozenset(set(range(4)) - {segment}))
            self.finish(block, prefix=1 + segment)
            self.assertEqual((block.phase, block.rounds), ('idle', round_number))

    def test_two_and_three_live_users_stage_byte_for_byte_what_the_flag_off_block_stages(self):
        """At most two idle segments: every idle segment sits on page 0, so the scratch page is never named and the staged inputs equal the flag-off block's."""
        reference = self.build(padded_min_users=2)
        scratch = self.lone_block()
        patterns = ((70000, 90000, None, None), (None, 70000, 90000, None), (70000, None, 90000, 20000), (None, 5000, 6000, 7000), (1500, 20000, 60000, None))
        for positions in patterns:
            staged = []
            for block in (reference, scratch):
                block.verify(self.round(positions))
                staged.append(self.staged(block) + (tuple(tuple(reader.positions.value.tolist()) for reader in self.readers(block)),
                                                    tuple(tuple(reader.metadata[0][1].value.flatten().tolist()) for reader in self.readers(block))))
                self.finish(block)
            with self.subTest(positions=positions):
                for left, right in zip(*staged):
                    if isinstance(left, torch.Tensor):
                        self.assertTrue(torch.equal(left, right))
                    else:
                        self.assertEqual(left, right)
            self.assertFalse(bool((staged[1][2] == SCRATCH).any()), 'the scratch page is not staged')

    def test_a_live_segments_staged_rows_do_not_depend_on_how_many_segments_are_idle(self):
        block = self.lone_block()
        user = 70000
        rows = []
        for others in ((None, None, None), (None, 90000, None), (20000, 90000, 30000)):
            block.verify(self.round((user,) + others))
            tokens, positions, pages = self.staged(block)
            rows.append((tokens[0:16].tolist(), positions[0:16].tolist(), pages[0:16].tolist(),
                         self.readers(block)[0].positions.value.tolist(), self.readers(block)[0].cur_pos[0].value.tolist()))
            self.finish(block)
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[1], rows[2])

    def test_a_live_table_that_holds_the_scratch_page_or_page_zero_is_refused_before_anything_is_staged(self):
        block = self.lone_block()
        table = torch.full((1, ext.WIDTH), 9, dtype=torch.int32)
        table[0, 1094] = SCRATCH                        # 70000 // 64 = 1093: the used range [0, 1095) holds index 1094
        copies, executed = len(self.ttnn.host_copies), len(self.ttnn.executed)
        with self.assertRaisesRegex(ValueError, 'page0 scratch page 16500 in a live table: segment 2 position 70000 page_index 1094'):
            block.verify(self.round((None, None, 70000, None), pages={2: table}))
        self.assertEqual((block.phase, len(self.ttnn.host_copies), len(self.ttnn.executed)), ('idle', copies, executed))
        self.assertTrue(any(line.startswith(packed_verifier.PADDED_PAGE0_MARKER) and 'scratch page' in line for line in self.lines), 'logged under the page-0 marker')
        zero = torch.full((1, ext.WIDTH), 9, dtype=torch.int32)
        zero[0, 1094] = 0
        with self.assertRaisesRegex(ValueError, 'page0 in a live table: segment 2 position 70000 page_index 1094'):
            block.verify(self.round((None, None, 70000, None), pages={2: zero}))
        # a scratch id OUTSIDE the range the round reads and writes is not a hit (the table pads with real ids, never the scratch page, so this is the rule's edge, not a case)
        outside = torch.full((1, ext.WIDTH), 9, dtype=torch.int32)
        outside[0, 1200] = SCRATCH
        block.verify(self.round((None, None, 70000, None), pages={2: outside}))
        self.finish(block)

    def test_without_scratch_pages_the_scratch_id_is_just_another_page(self):
        plain = self.build(padded_min_users=2)
        table = torch.full((1, ext.WIDTH), 9, dtype=torch.int32)
        table[0, 1094] = SCRATCH
        plain.verify(self.round((None, 70000, None, 90000), pages={1: table}))
        self.finish(plain)

    def test_the_t2_guard_of_the_step_reads_the_idle_fills_and_passes_a_lone_round_and_still_refuses_a_shared_tile_row(self):
        block = self.lone_block()
        user = self.round((None, None, 70000, None))[0]['request']
        self.assertIsNone(serving_packed_step.kv_guard([(user.engine, 70000)], block))
        # the same round with the live user's table pointing its first page at the scratch page: the idle segment 3 and the user would share (scratch, tile row 0)
        table = torch.full((1, ext.WIDTH), 9, dtype=torch.int32)
        table[0, 0] = SCRATCH
        user.engine.pages = table
        shared = serving_packed_step.kv_guard([(user.engine, 0)], block)
        self.assertIsNotNone(shared)
        self.assertIn('verify t2 kv tile rows shared', shared)
        # a block that does not pad one user (no scratch page): the guard does not fill the idle segments, it leaves the round to the block's own count rule
        plain = self.build(padded_min_users=2)
        self.assertIsNone(serving_packed_step.kv_guard([(user.engine, 70000)], plain))

    def test_padded_refusal_names_the_scratch_page_and_page_zero_apart_and_leaves_the_flag_off_reasons_alone(self):
        block = self.lone_block()
        plain = self.build(padded_min_users=2)
        table = torch.full((1, ext.WIDTH), 9, dtype=torch.int32)
        users = [None, None, (tuple(range(16)), 70000, table), None]
        self.assertIsNone(block.padded_refusal(users))
        users[2] = (tuple(range(16)), 70000, table.clone())
        users[2][2][0, 5] = SCRATCH
        self.assertEqual(block.padded_refusal(users), 'page0 scratch page 16500 in a live table: segment 2 position 70000 page_index 5')
        self.assertIsNone(plain.padded_refusal([None, None, (tuple(range(16)), 70000, table), (tuple(range(16)), 90000, table)]), 'two live: the plain block accepts')
        self.assertEqual(plain.padded_refusal([None, None, users[2], (tuple(range(16)), 90000, table)]), None, 'without scratch pages the scratch id is not forbidden')
        zero = table.clone()
        zero[0, 5] = 0
        for candidate in (block, plain):
            live = [None, None, (tuple(range(16)), 70000, zero), (tuple(range(16)), 90000, table)] if candidate is plain else [None, None, (tuple(range(16)), 70000, zero), None]
            self.assertEqual(candidate.padded_refusal(live), 'page0 in a live table: segment 2 position 70000 page_index 5')
        self.assertEqual(block.padded_refusal([None] * 4), 'padded round of 0 live users outside [1, 4)')
        self.assertIsNone(page_zero_index(table, 70000, 16, extent=True, also=(SCRATCH,)))
        self.assertEqual(page_zero_index(users[2][2], 70000, 16, extent=True, also=(SCRATCH,)), 5)
        self.assertIsNone(page_zero_index(users[2][2], 70000, 16, extent=True))

    def test_the_description_names_the_scratch_pages_only_when_there_are_some(self):
        self.assertEqual(self.lone_block().describe()['padded'], dict(min_users=1, max_idle=4, carries_in_place=False, rounds=0, scratch_pages=[SCRATCH]))
        self.assertEqual(self.build(padded_min_users=2).describe()['padded'], dict(min_users=2, max_idle=2, carries_in_place=False, rounds=0))


class EmulatedCacheTests(FreshStorage, ext.ExtentFixture):
    """The exactness argument in miniature: the K/V rows a staged round writes, applied to an emulated paged cache, never touch a page a live row reads or writes except its own rows."""

    def setUp(self):
        super().setUp()
        self.pool.packed_extent = self.fresh_storage

    def written(self, block):
        """{(page, row in page): segment} for every row the staged round writes; two segments writing one cell is a failure of the test, not of the block."""
        positions, pages = block.fixture.positions.value, block.fixture.pages.value
        cells = {}
        for row in range(positions.numel()):
            segment = row // block.rows_per_user
            cell = (int(pages[row][int(positions[row]) // 64]), int(positions[row]) % 64)
            self.assertNotIn(cell, cells, 'one cell, two writers')
            cells[cell] = segment
        return cells

    def test_idle_writes_land_on_page_zero_and_the_scratch_page_and_nowhere_a_live_row_lives(self):
        rng = random.Random(20261019)
        block = self.build(padded_min_users=1, idle_scratch_pages=(SCRATCH,))
        for trial in range(40):
            live = sorted(rng.sample(range(4), rng.choice((1, 1, 2, 3))))
            positions = [None] * 4
            tables = {}
            for segment in live:
                positions[segment] = rng.choice((130, 1500, 4000, 20000, 70000, 131296)) + rng.randrange(0, 16)
                base_page = 100 * (segment + 1) + trial
                table = torch.arange(base_page, base_page + ext.WIDTH, dtype=torch.int32).reshape(1, -1) % 16000 + 1
                tables[segment] = table
            block.verify(self.round(tuple(positions), pages=tables))
            cells = self.written(block)
            live_pages = set()
            for segment in live:
                start = positions[segment]
                live_pages |= {int(tables[segment][0][(start + row) // 64]) for row in range(16)}
                live_pages |= {int(tables[segment][0][index]) for index in range(0, (start // 256 + 1) * 256 // 64)}      # the extent reader's pages
            self.assertFalse(live_pages & {0, SCRATCH}, 'the live tables never name page 0 or the scratch page')
            idle = [segment for segment in range(4) if segment not in live]
            for cell, segment in cells.items():
                if segment in idle:
                    self.assertIn(cell[0], (0, SCRATCH), 'an idle segment writes page 0 or the scratch page only')
                else:
                    self.assertNotIn(cell[0], (0, SCRATCH))
                    self.assertIn(cell[0], live_pages)
            # an emulated cache: live pages hold their own sentinel, page 0 and the scratch page hold zeros; after the round only the live rows differ from before and only idle cells moved off zero
            cache = {}
            for page in live_pages:
                for row in range(64):
                    cache[(page, row)] = ('live', page, row)
            before = dict(cache)
            for cell, segment in cells.items():
                cache[cell] = ('segment', segment)
            changed = {cell for cell in cache if cache[cell] != before.get(cell)}
            self.assertEqual({cell for cell in changed if cell[0] in live_pages}, {cell for cell, segment in cells.items() if segment in live},
                             'on a live page only the live segments\' own rows changed')
            self.assertEqual(len({cell for cell, segment in cells.items() if segment in idle}), 16 * len(idle), 'every idle row had a cell of its own')
            self.finish(block, prefix=rng.randrange(1, 17))

    def test_every_live_pattern_of_one_to_three_has_disjoint_tile_rows_with_a_scratch_page_and_the_fourth_idle_segment_has_none(self):
        block = self.build(padded_min_users=1, idle_scratch_pages=(SCRATCH,))
        for count in (1, 2, 3):
            for live in combinations(range(4), count):
                users = [None] * 4
                for segment in live:
                    users[segment] = (tuple(range(16)), 9000 + segment, torch.full((1, ext.WIDTH), 50 + segment, dtype=torch.int32))
                fills = block.padded_users(users, live)
                guarded = [(range(start, start + 16), table[0]) for tokens, start, table in fills]
                self.assertIsNone(verify_trace_t2.kv_conflict(guarded), live)
        plain = self.build(padded_min_users=2)
        users = [(tuple(range(16)), 9000, torch.full((1, ext.WIDTH), 50, dtype=torch.int32)), None, None, None]
        with self.assertRaisesRegex(ValueError, 'At most 2 idle segments: page 0 holds two 32-row tile rows, so idle segments'):
            plain.padded_users(users, (0,))


class FlagOffIdentityTests(FreshStorage, ext.ExtentFixture):
    """The parent commit's block against today's, for the padded paths this change touched."""

    def setUp(self):
        super().setUp()
        self.pool.packed_extent = self.fresh_storage

    def parent(self):
        module = padded_tests.parent_module('packed_verifier.py', PARENT)
        if module is None:
            self.skipTest('no git history for %s' % PARENT)
        return module

    def test_idle_inputs_padded_refusal_and_describe_are_the_parents_for_every_flag_off_pattern(self):
        parent = self.parent()
        block = self.build(padded_min_users=2)
        old_class = parent.PackedVerifierEngine
        for count in (2, 3, 4):
            for live in combinations(range(4), count):
                new = block.idle_inputs(live)
                old = old_class.idle_inputs(block, live)
                self.assertEqual(sorted(new), sorted(old), live)
                for segment in new:
                    self.assertEqual(new[segment][:2], old[segment][:2])
                    self.assertTrue(torch.equal(new[segment][2], old[segment][2]) and new[segment][2].dtype == old[segment][2].dtype)
        for live in ((0,), ()):
            for module in (block.idle_inputs, lambda live, block=block: old_class.idle_inputs(block, live)):
                with self.assertRaisesRegex(ValueError, r'^At most 2 idle segments: page 0 holds two 32-row tile rows, so idle segments \[\d(, \d)*\] would share one$'):
                    module(live)
        table = torch.full((1, ext.WIDTH), 9, dtype=torch.int32)
        zero = table.clone()
        zero[0, 5] = 0
        for users in ([None, None, (tuple(range(16)), 70000, table), (tuple(range(16)), 90000, table)], [None, None, (tuple(range(16)), 70000, zero), (tuple(range(16)), 90000, table)],
                      [None, (tuple(range(16)), 70000, table), (tuple(range(16)), 70001, table), (tuple(range(16)), 90000, table)], [None] * 4):
            self.assertEqual(block.padded_refusal(users), old_class.padded_refusal(block, users))
        self.assertEqual(block.describe()['padded'], dict(min_users=2, max_idle=2, carries_in_place=False, rounds=0))
        self.assertEqual(page_zero_index(zero, 70000, 16), parent.page_zero_index(zero, 70000, 16))
        self.assertEqual(page_zero_index(zero, 70000, 16, extent=True), parent.page_zero_index(zero, 70000, 16, extent=True))

    def test_the_admission_marker_line_is_the_parents_without_scratch_pages(self):
        block = self.build(padded_min_users=3)
        admitted = [line for line in self.lines if line.startswith(packed_verifier.PADDED_ADMITTED_MARKER)]
        self.assertEqual(admitted, ['%s min_users=3 users=4 max_idle=2 carries_in_place=0' % packed_verifier.PADDED_ADMITTED_MARKER])
        self.assertNotIn('scratch', admitted[0])
        self.assertTrue(block.pads(3) and not block.pads(2))


def live_request(fixture, slot, position, page_value):
    """A request bound to pool slot `slot`'s carry with the session fields the routing policy reads."""
    request = base.request('R%d' % slot, fixture.pool.slots[slot], position, page_value, page_width=ext.WIDTH)
    request.session.finished, request.session.max_new_tokens, request.session.emitted, request.session.position = False, 4096, [], position
    return request


class RoutingTests(FreshStorage, octo_block_tests.OctoFixture):
    """Two M3 blocks over slots 0..3 and 4..7 and the octo block over 0..7, as the ship-4 attach builds them: which block each seat count drafts for, and at how many rows."""

    def setUp(self):
        super().setUp()
        octo_storage = self.storages[(8, 8)]
        self.pool.packed_extent = lambda users, rows: octo_storage if (users, rows) == (8, 8) else self.fresh_storage(users, rows)
        self.m3_a = self.block(m3_shape(ext.WIDTH), pool_slots=(0, 1, 2, 3), padded_min_users=1, idle_scratch_pages=(SCRATCH,))
        self.m3_b = self.block(m3_shape(ext.WIDTH), pool_slots=(4, 5, 6, 7), padded_min_users=1, idle_scratch_pages=(SCRATCH,))
        self.octo = self.octo_block(padded_min_users=6)
        self.plain_a = self.block(m3_shape(ext.WIDTH), pool_slots=(0, 1, 2, 3), padded_min_users=2)
        self.plain_b = self.block(m3_shape(ext.WIDTH), pool_slots=(4, 5, 6, 7), padded_min_users=2)

    def requests(self, slots):
        return [live_request(self, slot, 5000 + 300 * slot, 20 + slot) for slot in slots]

    def groups(self, slots, flag_on=True, octo_min_live=6):
        a, b = (self.m3_a, self.m3_b) if flag_on else (self.plain_a, self.plain_b)
        state = octo.OctoState('live', octo_min_live, clock=lambda: 0.0)
        step = PackedStep((a, b), per_block_widths=True, octo=self.octo, octo_state=state)
        requests = self.requests(slots)
        names = {id(a): 'A', id(b): 'B', id(self.octo): 'octo', id(None): None}
        return [(names[id(block)], rows, sorted(request.session.request_id for request in members)) for block, rows, members in step.proposal_groups(requests)]

    def test_flag_on_the_lone_user_and_the_split_blocks_draft_at_sixteen_rows_on_their_block(self):
        self.assertEqual(self.groups([0]), [('A', 16, ['R0'])])
        self.assertEqual(self.groups([2]), [('A', 16, ['R2'])])
        self.assertEqual(self.groups([6]), [('B', 16, ['R6'])])
        self.assertEqual(self.groups([0, 1]), [('A', 16, ['R0', 'R1'])])
        self.assertEqual(self.groups([0, 1, 2]), [('A', 16, ['R0', 'R1', 'R2'])])
        self.assertEqual(self.groups([0, 1, 2, 3]), [('A', 16, ['R0', 'R1', 'R2', 'R3'])])
        # a split: one user in each block (a block emptied by departures), five seats (a full block and a lone user in the other)
        self.assertEqual(self.groups([1, 5]), [('A', 16, ['R1']), ('B', 16, ['R5'])])
        self.assertEqual(self.groups([0, 1, 2, 3, 4]), [('A', 16, ['R0', 'R1', 'R2', 'R3']), ('B', 16, ['R4'])])
        # six or more live seats: the octo block, eight rows, every seat in one group
        self.assertEqual(self.groups([0, 1, 2, 3, 4, 5]), [('octo', 8, ['R0', 'R1', 'R2', 'R3', 'R4', 'R5'])])
        self.assertEqual(self.groups(list(range(8))), [('octo', 8, ['R%d' % slot for slot in range(8)])])

    def test_flag_off_the_lone_user_keeps_its_own_engines_width_and_nothing_else_changes(self):
        self.assertEqual(self.groups([0], flag_on=False), [('A', None, ['R0'])])
        self.assertEqual(self.groups([6], flag_on=False), [('B', None, ['R6'])])
        self.assertEqual(self.groups([1, 5], flag_on=False), [('A', None, ['R1']), ('B', None, ['R5'])])
        self.assertEqual(self.groups([0, 1, 2, 3, 4], flag_on=False), [('A', 16, ['R0', 'R1', 'R2', 'R3']), ('B', None, ['R4'])])
        for slots in ([0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5], list(range(8))):
            self.assertEqual(self.groups(slots, flag_on=False), self.groups(slots, flag_on=True), slots)

    def test_the_flag_changes_exactly_the_blocks_with_one_live_seat(self):
        for slots in ([0], [1], [5], [0, 4], [3, 7], [0, 1, 2, 3, 4], [0, 1, 2, 3, 7], [4, 5, 6, 7, 0]):
            on, off = self.groups(slots, flag_on=True), self.groups(slots, flag_on=False)
            self.assertEqual([item[0] for item in on], [item[0] for item in off], slots)
            for turned, plain in zip(on, off):
                if len(turned[2]) == 1:
                    self.assertEqual((turned[1], plain[1]), (16, None), slots)
                else:
                    self.assertEqual(turned, plain, slots)

    def test_a_lone_user_below_the_budget_or_outside_the_extent_range_still_drafts_at_its_own_width(self):
        a, b = self.m3_a, self.m3_b
        near_end = self.requests([0])[0]
        near_end.session.emitted = list(range(4096))             # no token left under the budget cap's one-token rule
        self.assertIsNone(serving_packed_step.block_rows_for(a, [near_end]))
        low = live_request(self, 0, 100, 20)
        self.assertIsNone(serving_packed_step.block_rows_for(a, [low]), 'below 128: the extent path does not admit it')
        table_zero = live_request(self, 0, 5000, 20)
        table_zero.engine.pages[0, 78] = 0
        self.assertIsNone(serving_packed_step.block_rows_for(a, [table_zero]), 'page 0 inside the used range: the round is drafted narrow')
        table_scratch = live_request(self, 0, 5000, 20)
        table_scratch.engine.pages[0, 78] = SCRATCH
        self.assertIsNone(serving_packed_step.block_rows_for(a, [table_scratch]), 'so is the scratch page')
        self.assertEqual(serving_packed_step.block_rows_for(a, self.requests([0])), 16)

    def test_a_lone_user_round_runs_on_its_block_through_the_step_and_the_octo_state_counts_it_as_an_m3_round(self):
        state = octo.OctoState('live', 6, clock=lambda: 0.0, log=lambda text: None)
        step = PackedStep((self.m3_a, self.m3_b), per_block_widths=True, octo=self.octo, octo_state=state)
        request = self.requests([5])[0]
        groups = step.proposal_groups([request])
        self.assertEqual([(rows, len(members)) for block, rows, members in groups], [(16, 1)])
        self.assertIs(groups[0][0], self.m3_b)
        self.assertEqual((state.plan_eligible, state.planned), (False, 'm3'), 'one seat is not an octo round')
        entry = base.entry(request, range(10, 26))
        self.assertIsNone(serving_packed_step.ineligible([entry], self.m3_b))
        self.assertIsNotNone(serving_packed_step.ineligible([entry], self.plain_b), 'the flag-off block refuses the same round')
        self.assertFalse(serving_packed_step.octo_serves(self.octo, [entry]), 'a 16-row ticket is not an octo ticket')


if __name__ == '__main__':
    unittest.main()
