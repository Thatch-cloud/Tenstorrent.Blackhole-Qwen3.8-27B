from contextlib import nullcontext
from types import SimpleNamespace
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import dspark_64k_scope as candidate
import dspark_history
import dspark_stable_history


class ScopeTests(unittest.TestCase):
    def test_shared_header_defaults_before_use_without_overriding_draft(self):
        import native_draft_sdpa

        root = Path(os.environ.get('TT_NATIVE_TEST_ROOT', '/opt/ttsim/tt-metal'))
        directory = root / native_draft_sdpa.KERNEL_DIRECTORY
        original = {name: (directory / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        with patch.object(candidate, 'admitted_request', return_value=nullcontext({})):
            with candidate.runtime_scope('.', '/report', context=65536, output_tokens=256,
                    factory_root=str(root), build_path='/build'):
                sources = native_draft_sdpa.patched_sources(original)
                common = sources['compute_common.hpp'].decode()
                first_use = common.index('QWEN_DRAFT_EXP_APPROX')
                self.assertEqual(common[first_use - len('#ifndef '):first_use], '#ifndef ')
                self.assertLess(common.index('#define QWEN_DRAFT_EXP_APPROX true'),
                    common.index('void recip_block_inplace'))
                draft = sources['sdpa.cpp'].decode()
                self.assertLess(draft.index('#define QWEN_DRAFT_EXP_APPROX'),
                    draft.index('#include "compute_common.hpp"'))

    def test_target_allocation_includes_decode_headroom(self):
        with patch.object(candidate, 'current_admission', return_value={
                'context': 65536, 'capacity': 66560, 'output_tokens': 256}):
            allocation = candidate.target_allocation()
        self.assertEqual(allocation, dict(max_seq_len=66560, page_count=1040, cache_blocks=1048, block_size=64))
        self.assertLess((65536 + 255) // 64, allocation['page_count'])
        self.assertLess((66560 - 1) // 64, allocation['page_count'])
        with self.assertRaises(ValueError):
            candidate.target_allocation()

    def test_publication_bounds_and_scope_lifetime(self):
        extended = candidate.stable_history_class(dspark_stable_history.StableHistoryKV)
        cache = object.__new__(extended)
        cache.closed, cache.pending, cache.position, cache.capacity = False, None, 66545, 66560
        with patch.object(candidate, 'require_scope', return_value={}):
            cache.check_prefix(15, 66545)
            for prefix, position in ((16, 66545), (15, 66544), (0, 66545)):
                with self.assertRaises(ValueError):
                    cache.check_prefix(prefix, position)
        with self.assertRaises(ValueError):
            cache.check_prefix(15, 66545)

    def test_no_allocation_without_admission(self):
        extended = candidate.stable_history_class(dspark_stable_history.StableHistoryKV)
        with self.assertRaises(ValueError):
            extended(None, None, None, None, None, None, None, position=65536, capacity=66560)

    def test_allocation_failure_releases_old_and_new_buffers(self):
        previous = tuple((object(), object()) for layer in range(5))
        padded = []
        released = []

        def pad(value, padding, fill):
            self.assertEqual(padding[2], (0, 1024))
            output = object()
            padded.append(output)
            return output

        operations = SimpleNamespace(pad=pad, clone=Mock(side_effect=RuntimeError('allocation failed')),
            DRAM_MEMORY_CONFIG='dram', synchronize_device=Mock(), deallocate=released.append)
        extended = candidate.stable_history_class(dspark_stable_history.StableHistoryKV)
        with patch.object(candidate, 'require_scope', return_value={}), \
                patch.object(dspark_history, 'project_chunks', return_value=previous), \
                patch.object(dspark_history, 'addresses', side_effect=lambda operations, value: (id(value),)):
            with self.assertRaisesRegex(RuntimeError, 'allocation failed'):
                extended(operations, None, None, None, None, None, None, position=65536, capacity=66560)
        self.assertEqual(set(map(id, released)), set(map(id, dspark_history.leaves(previous) + tuple(padded))))
        self.assertEqual(len(released), 20)

    def test_runtime_bindings_restore_after_error(self):
        import dspark_full_attention
        import dspark_native_cached_layer
        import dspark_prefill
        import native_draft_sdpa

        original = (dspark_full_attention.MAX_CONTEXT, dspark_stable_history.StableHistoryKV,
            dspark_prefill.FullHistoryCapture, dspark_native_cached_layer.attend, native_draft_sdpa.replacements)
        with patch.object(candidate, 'admitted_request', return_value=nullcontext({})):
            with self.assertRaisesRegex(RuntimeError, 'abort'):
                with candidate.runtime_scope('.', '/report', context=65536, output_tokens=256,
                        factory_root='/native', build_path='/build'):
                    self.assertEqual(dspark_full_attention.MAX_CONTEXT, 66560)
                    self.assertIsNot(dspark_stable_history.StableHistoryKV, original[1])
                    self.assertIn('QWEN_DRAFT_EXP_APPROX ||', native_draft_sdpa.replacements()['sdpa.cpp'][0][1])
                    raise RuntimeError('abort')
        self.assertEqual(original, (dspark_full_attention.MAX_CONTEXT, dspark_stable_history.StableHistoryKV,
            dspark_prefill.FullHistoryCapture, dspark_native_cached_layer.attend, native_draft_sdpa.replacements))
