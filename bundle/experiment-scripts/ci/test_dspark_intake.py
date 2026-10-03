import copy
import json
import struct
import unittest
from unittest.mock import MagicMock, patch

import dspark_intake as intake


def configuration():
    nested = dict(attention_mode='gqa', confidence_head_alpha=1.0, confidence_head_with_markov=True,
        enable_confidence_head=True, markov_head_type='vanilla', markov_rank=256,
        mask_token_id=248070, projector_type='dspark', target_layer_ids=intake.TAPS.copy())
    rope = dict(beta_fast=32.0, beta_slow=1.0, factor=32.0, original_max_position_embeddings=8192,
        rope_theta=10000000, rope_type='yarn')
    return dict(architectures=['DSparkDraftModel'], block_size=7, training_block_size=16,
        hidden_size=5120, intermediate_size=17408, num_hidden_layers=5, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, num_target_layers=64, vocab_size=248320,
        draft_vocab_size=248320, target_layer_ids=intake.TAPS.copy(), dtype='bfloat16', attention_bias=False,
        layer_types=['full_attention'] * 5, sliding_window=None, use_sliding_window=False,
        markov_head_type='vanilla', markov_rank=256, projector_type='dspark',
        enable_confidence_head=True, confidence_head_with_markov=True,
        mask_token_id=248070, eos_token_id=248044, rms_norm_eps=1e-6,
        dflash_config=copy.deepcopy(nested), dspark_config=nested, rope_parameters=rope,
        rope_scaling={key: value for key, value in rope.items() if key != 'rope_theta'})


def tensor_header():
    header = {'__metadata__': {'format': 'pt'}}
    cursor = 0
    for name, shape in sorted(intake.tensor_shapes().items()):
        size = 2 * intake.math.prod(shape)
        header[name] = dict(dtype='BF16', shape=shape, data_offsets=[cursor, cursor + size])
        cursor += size
    return header


class DSparkIntakeTests(unittest.TestCase):
    def test_serving_and_training_width_are_distinct(self):
        result = intake.validate_config(configuration())
        self.assertEqual((result['serving_proposals'], result['target_verify_rows'],
            result['training_future_positions']), (7, 8, 16))
        self.assertFalse(result['wider_serving_qualified'])

    def test_reject_dflash2_and_unreviewed_width_or_precision(self):
        for key, value in (('block_size', 8), ('block_size', 15), ('training_block_size', 8),
                ('dtype', 'float32'), ('sliding_window', 2048), ('use_sliding_window', True),
                ('num_hidden_layers', 4), ('enable_confidence_head', 1),
                ('target_layer_ids', [1, 31, 60]), ('markov_rank', 0), ('vocab_size', 151936)):
            with self.subTest(key=key, value=value):
                config = configuration()
                config[key] = value
                with self.assertRaises(ValueError):
                    intake.validate_config(config)

    def test_reject_default_rotary_or_conflicting_nested_config(self):
        for path, value in ((('rope_parameters', 'rope_type'), 'default'),
                (('rope_scaling', 'factor'), 1.0), (('dspark_config', 'markov_rank'), 128),
                (('dflash_config', 'target_layer_ids'), [0, 1, 2, 3, 4])):
            config = configuration()
            config[path[0]][path[1]] = value
            with self.assertRaises(ValueError):
                intake.validate_config(config)

    def test_exact_inventory_without_payload_claim(self):
        result = intake.validate_header(tensor_header(), intake.HEADER_BYTES, intake.CHECKPOINT_BYTES)
        self.assertEqual(result['tensors'], 62)
        self.assertEqual(result['parameters'], 1857358337)
        self.assertFalse(result['weight_payload_fetched'])
        self.assertFalse(result['weight_payload_hash_verified'])

    def test_reject_missing_extra_or_wrong_geometry(self):
        for mode in ('missing', 'extra', 'dtype', 'shape', 'bool'):
            with self.subTest(mode=mode):
                header = tensor_header()
                if mode == 'missing':
                    del header['norm.weight']
                elif mode == 'extra':
                    header['candidate_selector.hidden_projection.weight'] = header['fc.weight']
                elif mode == 'dtype':
                    header['norm.weight']['dtype'] = 'F16'
                elif mode == 'shape':
                    header['norm.weight']['shape'] = [2560, 2]
                else:
                    header['confidence_head.proj.bias']['shape'] = [True]
                with self.assertRaises(ValueError):
                    intake.validate_header(header, intake.HEADER_BYTES, intake.CHECKPOINT_BYTES)

    def test_reject_overlap_gap_wrong_size_and_boolean_offsets(self):
        for offsets in ([2, 4], [-2, 0], [False, 2], [0, 4]):
            header = tensor_header()
            header['confidence_head.proj.bias']['data_offsets'] = offsets
            with self.assertRaises(ValueError):
                intake.validate_header(header, intake.HEADER_BYTES, intake.CHECKPOINT_BYTES)
        for header_size, total in ((intake.HEADER_BYTES + 1, intake.CHECKPOINT_BYTES),
                (intake.HEADER_BYTES, intake.CHECKPOINT_BYTES + 1)):
            with self.assertRaises(ValueError):
                intake.validate_header(tensor_header(), header_size, total)

    def test_source_hash_and_size_both_required(self):
        data = b'not checkpoint code'
        digest = intake.hashlib.sha256(data).hexdigest()
        with patch.dict(intake.FILES, {'fixture': (len(data), digest)}):
            self.assertEqual(intake.verified_file('fixture', data), data)
            for bad in (data + b'x', b'x' * len(data)):
                with self.assertRaises(ValueError):
                    intake.verified_file('fixture', bad)

    def test_fetch_is_bounded_metadata_only_and_never_qualifies_hardware(self):
        data = json.dumps(tensor_header()).encode()
        digest = intake.hashlib.sha256(data).hexdigest()
        responses = []
        for name in intake.FILES:
            response = MagicMock()
            response.__enter__.return_value = response
            response.status = 200
            response.read.return_value = json.dumps(configuration()).encode() if name == 'config.json' else b'unexecuted'
            responses.append(response)
        with patch.object(intake.urllib.request, 'urlopen', side_effect=responses) as opened, \
                patch.object(intake, 'verified_file', side_effect=lambda name, value: value), \
                patch.object(intake, 'HEADER_SHA256', digest), \
                patch.object(intake, 'read_range', side_effect=[
                    (struct.pack('<Q', intake.HEADER_BYTES), intake.CHECKPOINT_BYTES),
                    (data, intake.CHECKPOINT_BYTES)]) as ranged:
            result, files = intake.fetch_metadata()
        self.assertEqual(ranged.call_args_list[0].args[1:], (0, 8))
        self.assertEqual(ranged.call_args_list[1].args[1:], (8, intake.HEADER_BYTES))
        self.assertEqual(opened.call_count, 3)
        for response, (size, _) in zip(responses, intake.FILES.values()):
            response.read.assert_called_once_with(size + 1)
        self.assertEqual(result['fetched_bytes'], 8 + sum(map(len, files.values())))
        self.assertFalse(result['remote_code_executed'])
        for gate in ('eligible_for_simulator', 'eligible_for_hardware', 'eligible_for_serving'):
            self.assertFalse(result[gate])

    def test_unknown_header_size_fails_before_second_range(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = b'unexecuted'
        with patch.object(intake.urllib.request, 'urlopen', return_value=response), \
                patch.object(intake, 'verified_file', side_effect=lambda name, value: value), \
                patch.object(intake, 'read_range', return_value=(
                    struct.pack('<Q', 1 << 40), intake.CHECKPOINT_BYTES)) as ranged:
            with self.assertRaises(ValueError):
                intake.fetch_metadata()
        self.assertEqual(ranged.call_count, 1)


if __name__ == '__main__':
    unittest.main()
