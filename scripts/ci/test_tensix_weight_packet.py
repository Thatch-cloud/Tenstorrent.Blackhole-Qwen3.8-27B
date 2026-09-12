from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tensix_weight_packet import packet_runtime


class WeightPacketTests(unittest.TestCase):
    def test_candidate_changes_only_read_dispatch_and_static_guards(self):
        directory = Path(__file__).parent
        original = (directory / 'tensix_weight_stream_reader.cpp').read_text()
        expected = original.replace('    constexpr uint32_t block_rows',
            '    static_assert(tile_bytes == 576 || tile_bytes == 1088);\n'
            '    static_assert(tile_bytes <= NOC_MAX_BURST_SIZE);\n    constexpr uint32_t block_rows')
        expected = expected.replace('noc_async_read_tile(page, source, destination);',
            'noc_async_read<tile_bytes>(source.get_noc_addr(page), destination, tile_bytes);')
        self.assertEqual((directory / 'tensix_weight_packet_reader.cpp').read_text(), expected)

    def test_overlay_changes_only_the_exact_reader_and_preserves_all_arguments(self):
        operations = SimpleNamespace(KernelDescriptor=Mock(return_value='descriptor'), marker=object())
        engagements = []
        runtime = packet_runtime(operations, engagements)
        self.assertIs(runtime.marker, operations.marker)
        for arguments in ([576, 8, 4, 272, 99], [1088, 8, 2, 160, 98]):
            kwargs = dict(kernel_source=str(Path(__file__).with_name('tensix_weight_stream_reader.cpp')),
                compile_time_args=arguments, runtime_args=object(), core_ranges=object(), config=object())
            self.assertEqual(runtime.KernelDescriptor(**kwargs), 'descriptor')
            actual = operations.KernelDescriptor.call_args.kwargs
            self.assertEqual(Path(actual['kernel_source']).name, 'tensix_weight_packet_reader.cpp')
            for name in ('compile_time_args', 'runtime_args', 'core_ranges', 'config'):
                self.assertIs(actual[name], kwargs[name])
        self.assertEqual(len(engagements), 2)
        kwargs = dict(kernel_source=str(Path(__file__).with_name('tensix_weight_stream_writer.cpp')))
        runtime.KernelDescriptor(**kwargs)
        self.assertEqual(operations.KernelDescriptor.call_args.kwargs, kwargs)
        self.assertEqual(len(engagements), 2)

    def test_other_transfer_sizes_and_geometries_reject(self):
        operations = SimpleNamespace(KernelDescriptor=Mock())
        runtime = packet_runtime(operations, [])
        for arguments in ([2048, 8, 4, 272], [576, 4, 4, 272], [576, 8, 8, 544], []):
            with self.assertRaises(ValueError):
                runtime.KernelDescriptor(kernel_source=str(Path(__file__).with_name('tensix_weight_stream_reader.cpp')),
                    compile_time_args=arguments)
        operations.KernelDescriptor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
