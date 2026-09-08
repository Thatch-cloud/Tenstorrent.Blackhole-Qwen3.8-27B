from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from mtp_cache_only import cache_digest, update_cache
from mtp_device_step import MTPDeviceStep


class MTPCacheOnlyTests(unittest.TestCase):
    def fixture(self):
        row = SimpleNamespace(shape=(1, 1, 1, 5120))
        positions = SimpleNamespace(shape=(1,))
        attention = SimpleNamespace(use_paged=True, _fused_qkv=True, _qkv_raw_decode=Mock(return_value='projection'),
            _kv_shard_cfg=Mock(return_value='shard'), tw=dict(q_norm='qnorm', k_norm='knorm'),
            NH=12, NKV=2, HD=256, rope_dim=64, paged_k='key-cache', paged_v='value-cache')
        mtp = SimpleNamespace(attention=attention, eps=1e-6, w_fc='fc', _nw={
            'mtp.pre_fc_norm_embedding.weight': 'embedding-norm', 'mtp.pre_fc_norm_hidden.weight': 'hidden-norm',
            'mtp.layers.0.input_layernorm.weight': 'input-norm'}, forward=Mock(), feed_forward=Mock())
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram', rms_norm=Mock(side_effect=['embedding', 'hidden', 'attention-input']),
            concat=Mock(return_value='joined'), matmul=Mock(return_value='fused'),
            transformer=SimpleNamespace(attn_decode_prep=Mock(return_value=('query', 'gate', 'keys', 'values'))),
            experimental=SimpleNamespace(paged_update_cache=Mock()))
        return operations, mtp, row, positions

    def test_native_prefix_publishes_both_caches_without_sdpa_or_mlp(self):
        operations, mtp, row, positions = self.fixture()
        owned = []
        update_cache(operations, mtp, row, row, positions, 'cos', 'sin', 'pages', owned)
        operations.concat.assert_called_once_with(['embedding', 'hidden'], dim=-1)
        operations.matmul.assert_called_once_with('joined', 'fc', memory_config='dram')
        mtp.attention._qkv_raw_decode.assert_called_once_with('attention-input')
        operations.transformer.attn_decode_prep.assert_called_once_with('projection', 'cos', 'sin', 'qnorm', 'knorm',
            12, 2, 256, 64, 'shard', batch=1, memory_config='dram')
        calls = operations.experimental.paged_update_cache.call_args_list
        self.assertEqual([call.args for call in calls], [('key-cache', 'keys'), ('value-cache', 'values')])
        self.assertTrue(all(call.kwargs == dict(update_idxs_tensor=positions, page_table='pages') for call in calls))
        self.assertEqual(owned, ['embedding', 'hidden', 'joined', 'fused', 'attention-input', 'projection', 'query', 'gate', 'keys', 'values'])
        mtp.forward.assert_not_called()
        mtp.feed_forward.assert_not_called()

    def test_invalid_cache_geometry_fails_before_operations(self):
        for key in ('use_paged', '_fused_qkv'):
            operations, mtp, row, positions = self.fixture()
            setattr(mtp.attention, key, False)
            with self.assertRaises(ValueError):
                update_cache(operations, mtp, row, row, positions, 'cos', 'sin', 'pages', [])
            operations.rms_norm.assert_not_called()

    def test_kv_only_mode_applies_only_to_head_free_steps(self):
        operations, mtp, row, positions = self.fixture()
        step = MTPDeviceStep.__new__(MTPDeviceStep)
        step.operations, step.mtp, step.owned, step.kv_only_repair = operations, mtp, [], True
        step.inputs = dict(embedding=row, hidden=row, positions=positions, cosine='cos', sine='sin')
        step.pages = 'pages'
        with patch('mtp_cache_only.update_cache') as update:
            self.assertEqual(step.execute(False), (row, None))
            update.assert_called_once()
        mtp.forward.assert_not_called()

        step.shortlist, step.sampler, step.native_sampling_rows = None, 'sampler', True
        step.model = SimpleNamespace(lm_head_weight='head')
        operations.linear = Mock(return_value='logits')
        mtp.forward.return_value = 'output-hidden'
        with patch('mtp_cache_only.update_cache') as update, patch('mtp_device_step.sample_rows', return_value='ids'):
            self.assertEqual(step.execute(True), ('output-hidden', 'ids'))
            update.assert_not_called()
        mtp.forward.assert_called_once()

    def test_valid_cache_digest_covers_both_chips_and_excludes_future_positions(self):
        tensor = torch.arange(2 * 2 * 4 * 32).reshape(2, 2, 4, 32).to(torch.bfloat16)
        parts = [tensor.clone(), tensor.clone()]
        mtp = SimpleNamespace(attention=SimpleNamespace(paged_k=tensor, paged_v=tensor))
        operations = SimpleNamespace(get_device_tensors=lambda value: parts, to_torch=lambda value: value)
        control = cache_digest(operations, mtp, 5)
        self.assertEqual(control['valid_rows'], 5)
        self.assertEqual(len(control['keys']), 2)
        parts[1][1, :, 1:] = -1
        self.assertEqual(cache_digest(operations, mtp, 5), control)
        parts[1][1, :, 0] = -1
        self.assertNotEqual(cache_digest(operations, mtp, 5)['keys'][1], control['keys'][1])
        for invalid in (False, 0, 9):
            with self.assertRaises(ValueError):
                cache_digest(operations, mtp, invalid)

    def test_invalid_step_option_fails_before_device_allocation(self):
        for enabled, native in ((1, True), (True, False)):
            with self.assertRaises(ValueError):
                MTPDeviceStep(None, None, None, None, None, None, kv_only_repair=enabled, native_sampling_rows=native)
