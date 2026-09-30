"""tp4_fabric_probe: the arithmetic, the verdict and the report file, on the CPU (ttnn and the models tree are
imported only inside run(), which the hardware job exercises)."""

import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tp4_fabric_probe as probe  # noqa: E402
import tp4_mesh  # noqa: E402

K4 = {(0, 1): 2, (0, 2): 2, (0, 3): 2, (1, 2): 2, (1, 3): 2, (2, 3): 2}


def good_report(**changes):
    order = [0, 1, 3, 2]
    report = dict(opened=True, order=order, ring=tp4_mesh.check_ring(order, K4), num_links=2,
                  ops=[dict(name='all_gather/decode/ring/2link', exact=True, median_ms=0.05, gbps=1.0)])
    report.update(changes)
    return report


class ArithmeticTests(unittest.TestCase):
    def test_bus_bytes_is_the_ring_algorithm_count(self):
        # a decode residual, 32 x 5,120 bf16 = 327,680 B; each of four devices moves 3/4 of it
        self.assertEqual(probe.bus_bytes('all_gather', 32, 5120, 4), 327680 * 3 // 4)
        self.assertEqual(probe.bus_bytes('reduce_scatter', 2048, 5120, 4), 2048 * 5120 * 2 * 3 // 4)
        self.assertEqual(probe.bus_bytes('all_gather', 32, 5120, 2), 327680 // 2)

    def test_bandwidth_is_decimal_gb_per_second_and_none_for_no_time(self):
        seconds = probe.bus_bytes('all_gather', 2048, 5120, 4) / 10e9
        self.assertAlmostEqual(probe.bandwidth('all_gather', 2048, 5120, 4, seconds), 10.0, places=2)
        self.assertIsNone(probe.bandwidth('all_gather', 2048, 5120, 4, 0))

    def test_op_names_cover_every_combination_once(self):
        names = [probe.op_name(op, size, topology, links) for size, _, _ in probe.SIZES for topology in probe.TOPOLOGIES
                 for links in probe.LINKS for op in probe.OPS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 2 * 2 * 2 * 2)
        self.assertIn('reduce_scatter/prefill/linear/1link', names)

    def test_the_sizes_are_the_models_and_shard_evenly_over_four_chips(self):
        self.assertEqual(dict((name, (rows, width)) for name, rows, width in probe.SIZES),
                         dict(decode=(32, 5120), prefill=(2048, 5120)))
        for _, rows, width in probe.SIZES:
            self.assertEqual(width % tp4_mesh.DEVICES, 0)
            self.assertEqual((width // tp4_mesh.DEVICES) % 32, 0, 'a shard of whole tiles')
            self.assertEqual(rows % 32, 0)

    def test_known_signatures(self):
        self.assertIn('49701', probe.known_signature('RuntimeError: No core coordinate found at (1, 2)'))
        self.assertIn('reset all four', probe.known_signature('Timed out while waiting for active ethernet core 3'))
        self.assertIn('fewer links', probe.known_signature('TT_FATAL: Expected 2 eth links between chips 0 and 1'))
        self.assertIsNone(probe.known_signature('nothing known'))
        self.assertIsNone(probe.known_signature(None))


class VerdictTests(unittest.TestCase):
    def test_a_good_report_passes(self):
        self.assertEqual(probe.verdict(good_report()), (True, []))

    def test_a_mesh_that_did_not_open_fails_with_the_error(self):
        passed, reasons = probe.verdict(dict(opened=False, error='RuntimeError: boom'))
        self.assertFalse(passed)
        self.assertIn('boom', reasons[0])

    def test_a_broken_degraded_or_unknown_ring_is_not_a_pass(self):
        broken, degraded = dict(K4), dict(K4)
        broken[(1, 3)] = 0
        degraded[(1, 3)] = 1
        for links, word in ((broken, 'BROKEN'), (degraded, 'DEGRADED'), ({}, 'UNKNOWN')):
            order = [0, 1, 3, 2]
            passed, reasons = probe.verdict(good_report(ring=tp4_mesh.check_ring(order, links)))
            self.assertFalse(passed, word)
            self.assertTrue(any(word in reason for reason in reasons), (word, reasons))

    def test_one_link_from_tt_ccl_or_an_inexact_op_fails(self):
        passed, reasons = probe.verdict(good_report(num_links=1))
        self.assertFalse(passed)
        self.assertIn('not 2', reasons[0])
        bad = good_report(ops=[dict(name='a', exact=True), dict(name='b', exact=False), dict(name='c')])
        passed, reasons = probe.verdict(bad)
        self.assertFalse(passed)
        self.assertIn('b, c', reasons[0])

    def test_no_collective_at_all_fails(self):
        passed, reasons = probe.verdict(good_report(ops=[]))
        self.assertFalse(passed)
        self.assertIn('no collective ran', reasons)


class DescriptorRefusalTests(unittest.TestCase):
    RING = '/c2/scripts/ci/' + tp4_mesh.DESCRIPTOR_NAME

    def test_only_the_ring_descriptor_without_problems_is_admitted(self):
        self.assertIsNone(probe.descriptor_refusal(self.RING, []))

    def test_the_pairs_descriptor_or_none_is_refused(self):
        for path in (None, '', '/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor'
                                                                                     '.textproto'):
            self.assertIn('not the ring descriptor', probe.descriptor_refusal(path, []))

    def test_an_unreadable_or_faulty_ring_descriptor_is_refused(self):
        self.assertIn('not readable', probe.descriptor_refusal(self.RING, None))
        self.assertIn('4 channels', probe.descriptor_refusal(self.RING, ['4 channels, not the 2']))

    def test_the_verdict_fails_on_a_descriptor_problem_alone(self):
        passed, reasons = probe.verdict(good_report(descriptor_problems=['4 channels, not the 2']))
        self.assertFalse(passed)
        self.assertIn('descriptor', reasons[0])
        passed, reasons = probe.verdict(good_report(descriptor_refusal='wrong path'))
        self.assertFalse(passed)

    def test_run_refuses_before_importing_a_device_library(self):
        # a stub environment with no ttnn: the refusal returns first, so the import inside run() is the only
        # thing that could fail - it must come after the refusal in the source
        with open(probe.__file__, encoding='utf-8') as handle:
            body = handle.read().split('def run(')[1].split('def time_op')[0]
        self.assertLess(body.index('descriptor_refusal(descriptor, problems)'), body.index('open_mesh_device'))
        old = os.environ.get('TT_MESH_GRAPH_DESC_PATH')
        os.environ['TT_MESH_GRAPH_DESC_PATH'] = '/x/p150_x2_mesh_graph_descriptor.textproto'
        fake = {}
        for name in ('torch', 'ttnn', 'models', 'models.common', 'models.common.modules',
                     'models.common.modules.tt_ccl', 'models.tt_transformers', 'models.tt_transformers.tt',
                     'models.tt_transformers.tt.ccl'):
            fake[name] = sys.modules.get(name)
            module = type(sys)(name)
            module.get_num_links = module.TT_CCL = module.tt_all_gather = module.tt_all_reduce = None
            module.cluster = None
            sys.modules[name] = module
        try:
            options = probe.build_parser().parse_args(['--output', 'x'])
            report = probe.run(options, log=lambda *a: None)
        finally:
            for name, old_module in fake.items():
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module
            if old is None:
                os.environ.pop('TT_MESH_GRAPH_DESC_PATH', None)
            else:
                os.environ['TT_MESH_GRAPH_DESC_PATH'] = old
        self.assertFalse(report['opened'])
        self.assertIn('refused to open', report['error'])
        self.assertFalse(probe.verdict(report)[0])


class MainTests(unittest.TestCase):
    def run_main(self, report):
        lines = []
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'fabric-probe.json')
            status = probe.main(['--fabric', 'FABRIC_1D', '--output', output], runner=lambda options, log=print: report,
                                log=lines.append)
            with open(output) as handle:
                written = json.load(handle)
        return status, written, lines

    def test_a_pass_exits_zero_writes_the_report_and_logs_one_verdict_line(self):
        status, written, lines = self.run_main(good_report())
        self.assertEqual(status, 0)
        self.assertTrue(written['passed'])
        self.assertEqual(written['reasons'], [])
        self.assertIn('passed=True', lines[-1])
        self.assertIn('num_links=2', lines[-1])

    def test_a_failure_exits_one_and_names_a_known_failure(self):
        report = dict(opened=False, error='x', known_failure='tenstorrent/tt-metal#49701 (a signature)')
        status, written, lines = self.run_main(report)
        self.assertEqual(status, 1)
        self.assertFalse(written['passed'])
        self.assertTrue(any('known failure' in line for line in lines))

    def test_the_parser_takes_only_the_two_fabric_configs(self):
        parser = probe.build_parser()
        self.assertEqual(parser.parse_args(['--output', 'x']).fabric, 'FABRIC_1D')
        with self.assertRaises(SystemExit), open(os.devnull, 'w') as sink:
            old, sys.stderr = sys.stderr, sink
            try:
                parser.parse_args(['--output', 'x', '--fabric', 'FABRIC_2D'])
            finally:
                sys.stderr = old

    def test_the_module_imports_no_device_library_at_import(self):
        with open(probe.__file__, encoding='utf-8') as handle:
            head = handle.read().split('def run(')[0]
        for banned in ('import ttnn', 'import torch', 'from models'):
            self.assertNotIn(banned, head)


if __name__ == '__main__':
    unittest.main()
