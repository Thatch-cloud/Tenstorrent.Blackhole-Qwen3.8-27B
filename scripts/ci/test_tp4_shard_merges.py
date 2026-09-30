"""S2T-04: the fast path's per-chip readbacks and merges at QWEN_FAST_TP=4 (four shards of 62,080 vocabulary columns),
and the pair's unchanged.

Every site below carried a literal two: the shard argmax fold, the packed verifier's readback, the drafter's
top-16 chunks over the borrowed LM head and their merge, the replicated-output checks, the audit's chip compare.
The pair's results are held by the modules' own suites (test_verify_trace_t1, test_draft_shared_head, ...); this
module drives each at four shards, with the environment switch a four-card process is launched with."""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import dflash_device
import dflash_packed_proposal
import draft_shared_head
import force_argmax
import gdn_multitoken_conv as PINNED
import packed_verifier
import serving_sequential_step
import tp_addresses
import tp_shapes


class Four(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4'})
        patcher.start()
        self.addCleanup(patcher.stop)
        # the readbacks below call gdn_multitoken_conv.addresses through their own imports, as a four-card worker
        # does after serving_startup installs the seam; undone so no other test sees the four-chip helpers
        holders = [module for module in list(sys.modules.values()) if getattr(module, '__dict__', None) is not None
                   and (module.__dict__.get('addresses') is PINNED.addresses
                        or module.__dict__.get('release_owned') is PINNED.release_owned)]
        original = (PINNED.addresses, PINNED.release_owned)
        tp_addresses.install(os.environ)

        def restore():
            for module in holders:
                for name, function in zip(('addresses', 'release_owned'), original):
                    if module.__dict__.get(name) in (tp_addresses.addresses, tp_addresses.release_owned):
                        module.__dict__[name] = function

        self.addCleanup(restore)


class ChipCountTests(unittest.TestCase):
    def test_words_follow_the_width(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual((tp_shapes.chip_count(), tp_shapes.count_word(), tp_shapes.all_chips()), (2, 'Two', 'Both'))
            self.assertEqual(tp_shapes.vocab_shard(), 124160)
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            self.assertEqual((tp_shapes.chip_count(), tp_shapes.count_word(), tp_shapes.all_chips()),
                             (4, 'Four', 'All 4'))
            self.assertEqual(tp_shapes.vocab_shard(), 62080)


class SharedHeadTests(Four):
    def values(self, rows=8):
        values = -torch.arange(248320, dtype=torch.float32).repeat(rows, 1)
        # a winner at each end of every chip's shard and of its 32,768-column chunks
        boundaries = [0, 32767, 32768, 62079, 62080, 94847, 94848, 124159,
                      124160, 156927, 156928, 186239, 186240, 219007, 219008, 248319]
        values[:, boundaries] = torch.arange(1, 17, dtype=torch.float32)
        return values

    def chunks(self, values):
        found = []
        for chip in range(4):
            for start, stop in draft_shared_head.candidate_chunks():
                scores, indices = values[:, chip * 62080 + start:chip * 62080 + stop].topk(16, dim=-1)
                found.append(dict(chip=chip, start=start, stop=stop, values=scores, indices=indices))
        return found

    def test_a_shard_is_two_chunks_the_second_short(self):
        self.assertEqual(draft_shared_head.candidate_chunks(), ((0, 32768), (32768, 62080)))
        with patch.dict(os.environ, {'QWEN_FAST_TP': '2'}):
            self.assertEqual(len(draft_shared_head.candidate_chunks()), 4)
            self.assertEqual(draft_shared_head.candidate_chunks()[-1], (98304, 124160))

    def test_the_global_top16_survives_the_four_shard_merge_in_any_arrival_order(self):
        values = self.values()
        chunks = self.chunks(values)
        expected_scores, expected_tokens = values.topk(16, dim=-1)
        for order in (chunks, list(reversed(chunks))):
            tokens, scores = draft_shared_head.merge_chunk_candidates(order)
            self.assertTrue(torch.equal(tokens, expected_tokens[None, 1:8]))
            self.assertTrue(torch.equal(scores, expected_scores[None, 1:8]))

    def test_a_missing_or_repeated_chunk_or_a_pair_of_chips_is_refused(self):
        chunks = self.chunks(self.values())
        for selected in (chunks[:-1], chunks + [chunks[0]], [chunk for chunk in chunks if chunk['chip'] < 2]):
            with self.assertRaises(ValueError):
                draft_shared_head.merge_chunk_candidates(selected)

    def test_an_index_past_the_short_chunk_is_out_of_range(self):
        chunks = self.chunks(self.values())
        last = [chunk for chunk in chunks if chunk['stop'] == 62080][0]
        last['indices'][0, 0] = 29312
        with self.assertRaisesRegex(ValueError, 'in-range'):
            draft_shared_head.merge_chunk_candidates(chunks)

    def test_the_local_head_takes_only_the_four_card_shard_and_pads_the_short_chunk(self):
        operations = SimpleNamespace(slice=Mock(side_effect=lambda logits, start, stop, step: ('slice', start[3], stop[3])),
                                     pad=Mock(side_effect=lambda chunk, padding, value: ('pad', chunk)),
                                     topk=Mock(return_value=('values', 'indices')))
        logits = SimpleNamespace(shape=(1, 1, 8, 62080))
        owned = []
        found = draft_shared_head.local_head_candidates(operations, logits, owned)
        self.assertEqual([(chunk['start'], chunk['stop']) for chunk in found], [(0, 32768), (32768, 62080)])
        operations.pad.assert_called_once()
        with self.assertRaises(ValueError):
            draft_shared_head.local_head_candidates(operations, SimpleNamespace(shape=(1, 1, 8, 124160)), [])

    def test_the_head_needs_the_four_card_mesh(self):
        model = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True)
        normalized = SimpleNamespace(shape=(1, 1, 8, 5120), dtype='bf16', layout='tile',
                                     memory_config=lambda: 'dram')
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram')
        with self.assertRaisesRegex(ValueError, 'TP4'):
            draft_shared_head.shared_head_candidates(operations, model, normalized, [])


class ReplicatedOutputTests(Four):
    def device(self, feature_parts, chunk_parts=4):
        chunk = dict(start=0, stop=32768, values='v', indices='i')
        outputs = SimpleNamespace(chunks=[chunk], projected='projected')

        def tensors(tensor):
            return {'v': [torch.zeros(32, 16)] * chunk_parts, 'i': [torch.zeros(32, 16, dtype=torch.int64)] * chunk_parts,
                    'projected': feature_parts}[tensor]

        operations = SimpleNamespace(get_device_tensors=tensors, to_torch=lambda value: value)
        return SimpleNamespace(operations=operations), outputs

    def read(self, function, feature_parts, chunk_parts=4):
        device, outputs = self.device(feature_parts, chunk_parts)
        with patch('dflash_packed_proposal.merged_candidates', return_value=('candidates', 'unary')) as merged, \
                patch('dflash_packed_proposal.split_selection', return_value='split'), \
                patch('dflash_packed_proposal.select_packed', return_value='selected'):
            result = function(device, outputs, None, [], [], 32) if function is dflash_packed_proposal.select_device_outputs \
                else function(device, outputs, [], 32)
        return result, merged

    def test_four_equal_selector_copies_pass_and_every_chip_contributes_chunks(self):
        same = [torch.ones(1, 32, 256) for _ in range(4)]
        for function in (dflash_packed_proposal.read_device_outputs,):
            device, outputs = self.device(same)
            with patch('dflash_packed_proposal.merged_candidates', return_value=('candidates', 'unary')) as merged, \
                    patch('dflash_packed_proposal.split_selection', return_value='split'):
                self.assertEqual(function(device, outputs, [], 32), 'split')
            self.assertEqual(sorted(chunk['chip'] for chunk in merged.call_args.args[2]), [0, 1, 2, 3])

    def test_a_late_diverging_copy_is_caught(self):
        for last in range(1, 4):
            parts = [torch.ones(1, 32, 256) for _ in range(4)]
            parts[last] = parts[last] + 1
            device, outputs = self.device(parts)
            with patch('dflash_packed_proposal.merged_candidates', return_value=('c', 'u')):
                with self.assertRaisesRegex(AssertionError, 'Replicated learned selector features differ'):
                    dflash_packed_proposal.read_device_outputs(device, outputs, [], 32)

    def test_two_readbacks_at_four_cards_are_refused(self):
        device, outputs = self.device([torch.ones(1, 32, 256)] * 2, chunk_parts=2)
        with self.assertRaisesRegex(AssertionError, 'All 4 learned head shards required'):
            dflash_packed_proposal.read_device_outputs(device, outputs, [], 32)


class PackedReadbackTests(Four):
    def engine(self, id_parts, value_parts, audit=False):
        tensors = {'ids': id_parts, 'values': value_parts}
        operations = SimpleNamespace(get_device_tensors=lambda name: tensors[name], to_torch=lambda part: part)
        return SimpleNamespace(operations=operations, output=(None, 'ids', 'values', 'reference'), block_rows=3,
                               shard_audit=audit)

    def test_four_shards_are_combined_in_chip_order(self):
        ids = [torch.tensor([1, 1, 1, 0]), torch.tensor([2, 2, 2, 0]), torch.tensor([3, 3, 3, 0]),
               torch.tensor([4, 4, 4, 0])]
        values = [torch.tensor([1.0, 1.0, 5.0, 0.0]), torch.tensor([2.0, 1.0, 1.0, 0.0]),
                  torch.tensor([1.0, 1.0, 1.0, 0.0]), torch.tensor([1.0, 3.0, 1.0, 0.0])]
        found = packed_verifier.PackedVerifierEngine.shard_predictions(self.engine(ids, values))
        self.assertEqual(found, [2 + 62080, 4 + 3 * 62080, 1])

    def test_two_shards_are_refused_at_four_cards(self):
        parts = [torch.tensor([1, 1, 1, 0])] * 2
        with self.assertRaisesRegex(AssertionError, 'Four chip-local outputs required'):
            packed_verifier.PackedVerifierEngine.shard_predictions(self.engine(parts, parts))


class NativeRowTests(Four):
    def test_the_native_row_guard_reads_the_four_card_shard(self):
        sampler = SimpleNamespace(
            tt_sampling=SimpleNamespace(force_argmax_sampling=True, max_batch_size=32, vocab_size=248320,
                                        padded_vocab_size=248320),
            seed_manager=SimpleNamespace(has_active_request_seed=lambda: False), _penalties_active=False)
        wrong = SimpleNamespace(shape=(1, 1, 8, 124160))
        with self.assertRaisesRegex(ValueError, 'TP4'):
            force_argmax.sample_rows(sampler, wrong, 8, SimpleNamespace(), native_rows=True)


class AuditTests(Four):
    def audit(self, shards):
        operations = SimpleNamespace(get_device_tensors=lambda value: shards, to_torch=lambda shard: shard)
        seen = []
        audit = dflash_device.ProposalAudit.__new__(dflash_device.ProposalAudit)
        audit.operations, audit.stages, audit.first_divergent = operations, [], None
        audit.line = lambda *args: seen.append(args)
        return audit, seen

    def test_a_divergence_on_any_chip_is_the_stages_result(self):
        base = torch.ones(2, 32, dtype=torch.bfloat16)
        for chip in range(1, 4):
            shards = [base.clone() for _ in range(4)]
            shards[chip] = shards[chip] + 1
            audit, seen = self.audit(shards)
            audit.observe('stage', 'value')
            self.assertEqual(audit.stages[0][1]['differing'], 64)
            self.assertEqual(audit.first_divergent, 'stage')
            self.assertIn('chip%d' % chip if chip != 1 else 'chip1', seen[0][-1].__str__() + str(audit.stages[0][1]))

    def test_four_equal_copies_report_no_difference(self):
        base = torch.ones(2, 32, dtype=torch.bfloat16)
        audit, seen = self.audit([base.clone() for _ in range(4)])
        audit.observe('stage', 'value')
        self.assertEqual(audit.stages[0][1]['differing'], 0)
        self.assertIsNone(audit.first_divergent)

    def test_two_copies_are_refused_at_four_cards(self):
        base = torch.ones(2, 32, dtype=torch.bfloat16)
        audit, _ = self.audit([base, base])
        with self.assertRaisesRegex(AssertionError, 'All 4 chips required'):
            audit.observe('stage', 'value')


class CheckShardsTests(Four):
    """serving_sequential_step.check_shards over four-chip tensors: the replicated buffers must agree on every chip."""

    def setUp(self):
        super().setUp()
        import sys
        from test_serving_sequential_step_shards import FakeLogger

        self.logger = FakeLogger()
        for patcher in (patch.dict(sys.modules, {'loguru': SimpleNamespace(logger=self.logger)}),
                        patch.object(serving_sequential_step, 'SHARD_CHECK', '1')):
            patcher.start()
            self.addCleanup(patcher.stop)
        for table in (serving_sequential_step.RECORDED, serving_sequential_step.REPORTED,
                      serving_sequential_step.SNAPSHOTS):
            table.clear()
            self.addCleanup(table.clear)

    @staticmethod
    def widen(drafter):
        """Two more chips for every tensor the pair-shaped fixture built: copies of chip 1 one page up each."""
        from test_serving_sequential_step_shards import FakeShard

        for tensor in drafter.named.values():
            for extra in (2, 3):
                base = tensor.shards[1]
                tensor.shards.append(FakeShard(base.data.clone(), base.address + 0x100 * (extra - 1)))
        return drafter

    def entries(self, damaged=None):
        from test_serving_sequential_step_shards import FakeOperations, device, entry

        operations, stepped = FakeOperations(), []
        first, second = (self.widen(device(operations, kv_layers=1)), self.widen(device(operations, kv_layers=1)))
        if damaged is not None:
            first.named['history'].shards[damaged].data[0, :3] += 2
        return [entry('A', first, stepped), entry('B', second, stepped)]

    def test_four_equal_copies_of_every_replicated_buffer_pass(self):
        outputs = serving_sequential_step.sequential_packed_step(self.entries(), cancelled=lambda: False)
        self.assertEqual([output.request_id for output in outputs], ['A', 'B'])
        self.assertTrue(any('shards equal after step' in line for line in self.logger.lines))
        addresses = [line for line in self.logger.lines if line.startswith('[PINDIAG] address B ')]
        self.assertTrue(addresses)
        self.assertTrue(all(len(line.split()) == 4 + 4 for line in addresses), addresses[:1])

    def test_a_copy_that_differs_only_on_the_last_chip_is_caught(self):
        for chip in (1, 3):
            for table in (serving_sequential_step.RECORDED, serving_sequential_step.REPORTED,
                          serving_sequential_step.SNAPSHOTS):
                table.clear()
            with self.assertRaisesRegex(AssertionError, 'Replicated draft'):
                serving_sequential_step.sequential_packed_step(self.entries(damaged=chip), cancelled=lambda: False)


if __name__ == '__main__':
    unittest.main()
