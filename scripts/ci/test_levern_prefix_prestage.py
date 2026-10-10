"""Lever N with prefix reuse: the fixture epoch (verify_prestage's disjoint external-writer class), what registers it, and the census of the route's device
writes that keeps it sound (docs/lever-n-prefix-merged-route.md section 8).

What is held:
  class      an external writer registered with its persistent addresses engages the disjoint class only beside the per-block epochs; a step in the class
             moves no epoch (counted), another writer still moves the global one, a block whose staging destinations share an address with the writer takes
             the class off for the process (at registration or at the block's first pre-stage), and the audit flag still forces the full stage;
  route      levern_route.persistent_writes reads the B=1 scratch of every GDN layer, engage_epoch_scope registers it only under the route scope on the merged
             route, and an overlap with a block's destinations is reported;
  hook       the worker hook's pass-through charges a prefill nothing while the class is engaged and as before otherwise;
  census     every call the route makes on the model or ttnn is on the audited list (levern_route.WRITE_SITES): a new write site fails here, because the
             disjoint class is sound only while the list is complete; the stage-1 route's own writes are the same set."""

import ast
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import levern_policy
import levern_route as route
import qwen_prefix_model_fixture as F
import verify_prestage as vp
from test_tp4_hostgap import Clean, stub_block

HERE = Path(__file__).resolve().parent
ROUTE_SOURCE = HERE / 'levern_route.py'


class Fresh(Clean):
    def setUp(self):
        super(Fresh, self).setUp()
        vp.reset_external_writers()
        self.addCleanup(vp.reset_external_writers)


def addresses_of(table):
    return lambda destination: table[destination]


class DisjointClassTests(Fresh):
    def test_registered_writer_engages_the_class_only_with_the_per_block_epochs(self):
        self.assertFalse(vp.disjoint_engaged())
        self.assertEqual(vp.register_external_writer('levern-route', [(0, 100), (1, 100)]), 'engaged')
        self.assertFalse(vp.disjoint_engaged(), 'one global epoch: every writer bumps')
        vp._MODE.update(first=True, blocks=True)
        self.assertTrue(vp.disjoint_engaged())
        self.assertEqual(len(self.log_lines(vp.EXTERNAL_WRITER_ENGAGED_MARKER)), 1)

    def log_lines(self, marker):
        return [line for line in self.log if line.startswith(marker)]

    def test_a_step_in_the_class_moves_no_epoch_and_is_counted_and_any_other_writer_still_does(self):
        vp.register_external_writer('levern-route', [(0, 100)])
        vp._MODE.update(first=True, blocks=True)
        first, second = stub_block('A'), stub_block('B')
        before, local = vp.epoch(), (vp.local_epoch(first.fixture), vp.local_epoch(second.fixture))
        vp.note_disjoint('prefill')
        vp.note_disjoint('prefill-chunk')
        self.assertEqual((vp.epoch(), vp.local_epoch(first.fixture), vp.local_epoch(second.fixture)), (before,) + local)
        self.assertEqual(vp.disjoint_count(), 2)
        vp.bump('admission')
        self.assertEqual(vp.epoch(), before + 1)

    def test_a_snapshot_survives_a_disjoint_step_and_dies_with_a_global_one(self):
        vp.register_external_writer('levern-route', [(0, 100)])
        vp._MODE.update(first=True, blocks=True)
        block = stub_block('A')
        block.prestaged.snapshot = vp.Snapshot(vp.epoch(), [], [], 0, 0.0, 1, local=vp.local_epoch(block.fixture))
        vp.note_disjoint('prefill')
        self.assertIsNone(block.prestaged.usable()[1])
        vp.bump('prefill')
        self.assertEqual(block.prestaged.usable(), (None, 'epoch:prefill'))

    def test_a_blocks_destinations_that_share_an_address_with_the_writer_take_the_class_off(self):
        vp._MODE.update(first=True, blocks=True)
        vp.register_external_writer('levern-route', [(0, 100), (1, 100)])
        block = stub_block('A')
        table = {'x': (100, 7), 'y': (8, 9)}
        self.assertIsNone(vp.register_destinations(block, ['x'], addresses_of(table)))
        self.assertFalse(vp.disjoint_engaged())
        self.assertEqual(len(self.log_lines(vp.EXTERNAL_WRITER_REFUSED_MARKER)), 1)
        self.assertIn('reason=overlap', self.log_lines(vp.EXTERNAL_WRITER_REFUSED_MARKER)[0])
        self.assertTrue(vp._MODE['blocks'], 'the per-block epochs stand: only the writer class went')

    def test_disjoint_destinations_leave_the_class_engaged(self):
        vp._MODE.update(first=True, blocks=True)
        vp.register_external_writer('levern-route', [(0, 100), (1, 100)])
        first, second = stub_block('A'), stub_block('B')
        vp.register_destinations(first, ['x'], addresses_of({'x': (1, 2)}))
        vp.register_destinations(second, ['y'], addresses_of({'y': (3, 4)}))
        self.assertTrue(vp.disjoint_engaged())

    def test_the_same_address_on_another_chip_is_not_an_overlap(self):
        vp._MODE.update(first=True, blocks=True)
        vp.register_external_writer('levern-route', [(0, 100)])
        vp.register_destinations(stub_block('A'), ['x'], addresses_of({'x': (5, 100)}))
        self.assertTrue(vp.disjoint_engaged())

    def test_registering_after_a_block_checks_against_it_too(self):
        vp._MODE.update(first=True, blocks=True)
        vp.register_destinations(stub_block('A'), ['x'], addresses_of({'x': (100, 7)}))
        self.assertEqual(vp.register_external_writer('levern-route', [(0, 100), (1, 100)]), 'overlap')
        self.assertFalse(vp.disjoint_engaged())

    def test_a_writer_with_no_addresses_is_refused(self):
        self.assertEqual(vp.register_external_writer('levern-route', []), 'overlap')
        self.assertFalse(vp.disjoint_engaged())

    def test_reset_forgets_the_writer(self):
        vp._MODE.update(first=True, blocks=True)
        vp.register_external_writer('levern-route', [(0, 100)])
        vp.note_disjoint('x')
        vp.reset_external_writers()
        self.assertEqual((vp.disjoint_engaged(), vp.disjoint_count()), (False, 0))

    def test_engaging_the_two_block_mode_does_not_forget_a_writer_the_route_registered_before_it(self):
        """The route registers at attach, before the packed blocks are built: serving_packed_step's engage_two_block must not undo it."""
        vp.register_external_writer('levern-route', [(0, 100)])
        vp.engage_two_block([stub_block('A'), stub_block('B')], dict(os.environ, QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE='1',
                                                                    QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS='1'))
        self.assertEqual(vp.two_block_mode(), 'blocks')
        self.assertTrue(vp.disjoint_engaged())

    def test_the_audit_flag_still_forces_the_full_stage_whatever_the_class_says(self):
        self.assertTrue(hasattr(vp, 'full_audit_enabled'))
        with mock.patch.dict(os.environ, {'QWEN_FAST_PRESTAGE': '1', 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT': '1'}):
            self.assertTrue(vp.full_audit_enabled())
        vp._MODE.update(first=True, blocks=True)
        vp.register_external_writer('levern-route', [(0, 100)])
        with mock.patch.dict(os.environ, {'QWEN_FAST_PRESTAGE': '1', 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT': '1'}):
            self.assertTrue(vp.full_audit_enabled() and vp.disjoint_engaged(), 'the audited arm runs with the class engaged: its full stage is the proof')


class RouteRegistrationTests(Fresh):
    def model(self):
        fake = F.FakeTTNN()
        engine = F.Toy(F.load_source(F.staged_sources()[0], 'qwen36_model_for_prestage', F.model_stubs(fake, F.FakeLogger())), fake)
        return engine.model

    def test_the_persistent_writes_are_the_scratch_of_every_gdn_layer(self):
        model = self.model()
        model._ensure_gdn_prefill_scratch()
        seen = []

        def addresses(tensor):
            seen.append(id(tensor))
            return (id(tensor) % 100000, id(tensor) % 100000 + 1)

        pairs = route.persistent_writes(model, addresses)
        layers = [layer for layer in model.layers if not layer.is_full_attention]
        per_layer = 3 + model.layers[0].attention.K
        self.assertEqual(len(seen), len(layers) * per_layer)
        self.assertEqual(len(set(seen)), len(seen), 'every tensor of the scratch is read once')
        self.assertEqual(len(pairs), len(layers) * per_layer * 2)

    def test_the_scope_registers_only_under_the_route_scope_on_the_merged_route(self):
        model = self.model()
        merged = {'QWEN_FAST_LEVER_N': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_STICKY_SESSIONS': '1'}
        logs = []
        log = lambda message, *values: logs.append(message.format(*values) if values else message)    # noqa: E731
        addresses = lambda tensor: (id(tensor) % 1000000, 0)                                           # noqa: E731
        self.assertIsNone(route.engage_epoch_scope(model, addresses, environ=merged, log=log), 'global scope: nothing registered')
        self.assertIsNone(route.engage_epoch_scope(model, addresses, environ={'QWEN_FAST_LEVER_N': '1',
                                                                              'QWEN_FAST_LEVERN_EPOCH_SCOPE': 'route'}, log=log))
        vp._MODE.update(first=True, blocks=True)
        outcome = route.engage_epoch_scope(model, addresses, environ=dict(merged, QWEN_FAST_LEVERN_EPOCH_SCOPE='route'), log=log)
        self.assertEqual(outcome, 'engaged')
        self.assertTrue(vp.disjoint_engaged())
        self.assertTrue(any('route writer engaged' in line for line in logs), logs)

    def test_an_overlap_with_a_registered_block_is_reported_and_the_scope_stays_global(self):
        model = self.model()
        vp._MODE.update(first=True, blocks=True)
        model._ensure_gdn_prefill_scratch()
        one_tensor = model._gdn_prefill_scratch[0][1]
        vp.register_destinations(stub_block('A'), ['x'], addresses_of({'x': (123, 456)}))
        outcome = route.engage_epoch_scope(model, lambda tensor: (123, 456) if tensor is one_tensor else (id(tensor), 0),
                                           environ={'QWEN_FAST_LEVER_N': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_STICKY_SESSIONS': '1',
                                                    'QWEN_FAST_LEVERN_EPOCH_SCOPE': 'route'}, log=lambda *a: None)
        self.assertEqual(outcome, 'overlap')
        self.assertFalse(vp.disjoint_engaged())


class WriteCensusTests(unittest.TestCase):
    """The route's device writes, found in its source rather than assumed: the audited list is levern_route.WRITE_SITES (and the final-step slot write
    and the chunk loop are the only writes that reach the batched decode state or the pool)."""

    MODEL_NAMES = ('model', 'self')

    @classmethod
    def calls(cls):
        tree = ast.parse(ROUTE_SOURCE.read_text(encoding='utf-8'))
        found = {}
        for function in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                owner = node.func.value
                name = owner.id if isinstance(owner, ast.Name) else None
                if name in ('model', 'ttnn') or (name is None and isinstance(owner, ast.Attribute) and owner.attr == 'model'):
                    found.setdefault((name, node.func.attr), set()).add(function.name)
        return found

    def test_every_call_on_the_model_or_ttnn_is_on_the_audited_list(self):
        # reads: the host reads of the scratch and the logits, the audit's program-free KV read (_qwen_prefix_audit reads each cache to the host), the
        # entry the lifecycle's capture wraps, and the allocation of the scratch the scope registers. The audit digest's region read (QWEN_FAST_LEVERN_KV_READ,
        # kv_digest_region) is device -> host only: allocate_tensor_on_host makes a HOST tensor, qwen_read_blocks copies named blocks of a cache into it (the cache is
        # never written, nothing is compiled), Shape builds its shape.
        reads = {'to_torch', 'ConcatMeshToTensor', 'num_program_cache_entries', 'prefill_paged_slots_range', '_ensure_gdn_prefill_scratch',
                 'synchronize_device', 'ShardTensorToMesh', '_qwen_prefix_audit', 'allocate_tensor_on_host', 'qwen_read_blocks', 'Shape'}
        unknown = sorted((owner, attribute) for (owner, attribute) in self.calls() if attribute not in route.WRITE_SITES and attribute not in reads)
        self.assertEqual(unknown, [], 'a new call on the model or ttnn in levern_route: audit it as a persistent write or a read, add it to WRITE_SITES '
                                      'or to this test\'s reads (the disjoint-writer class is sound only while the list is complete)')

    def test_the_decode_slot_is_written_only_by_the_final_step(self):
        sites = self.calls()
        self.assertEqual(sorted(sites[('model', '_write_gdn_slot')]), ['_merged_route', '_warm_merged', 'route', 'run'])
        source = ROUTE_SOURCE.read_text(encoding='utf-8')
        for function in ('route', '_merged_route'):
            tree = ast.parse(source)
            node = next(item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == function)
            writes = [call for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                      and call.func.attr == '_write_gdn_slot']
            self.assertEqual(len(writes), 1, function)
            text = ast.get_source_segment(source, node)
            index = text.index('_write_gdn_slot')
            self.assertIn('if final', text[max(0, index - 400):index], '%s writes the slot only under the final-step condition' % function)

    def test_the_scratch_is_the_only_persistent_buffer_the_scope_registers(self):
        source = ROUTE_SOURCE.read_text(encoding='utf-8')
        start = source.index('def persistent_writes')
        end = source.index('def engage_epoch_scope')
        body = source[start:end]
        for name in ('_gdn_prefill_scratch', 'rec', 'carry', 'zero0', 'convs'):
            self.assertIn(name, body)
        self.assertNotIn('_paged_kv_caches', body, 'the KV pool is data the verify reads through its page tables, not a staged input')

    def test_the_write_sites_list_names_only_methods_the_stage_puts_in_every_image(self):
        for name in route.WRITE_SITES:
            self.assertTrue(name.startswith('_') or name in ('prefill_traced_chunked', 'deallocate', 'to_torch', 'synchronize_device'), name)


if __name__ == '__main__':
    unittest.main()
