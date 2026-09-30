"""llk_kernels: the hotspot registry, its pins and anchors against this repo's own copies, where a copy may be
mounted, and the generated (K5-A) route against the real gdn_seq_block generator (a synthetic native prefix,
as test_gdn_seq_block uses)."""
import ast
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import llk_kernels as kernels  # noqa: E402
import llk_zones as zones  # noqa: E402


def read(path):
    with open(os.path.join(ROOT, path), 'rb') as handle:
        return zones.decode(handle.read())


def k5a_build():
    """The level-0 variant-A K5-A build from the real generator over the synthetic native prefix."""
    import test_gdn_seq_block
    return test_gdn_seq_block.build()


class RegistryTests(unittest.TestCase):
    def test_keys_names_routes(self):
        keys = [entry['key'] for entry in kernels.KERNELS]
        self.assertEqual(len(keys), len(set(keys)))
        for entry in kernels.KERNELS:
            with self.subTest(key=entry['key']):
                zones.zone_name(entry['key'])
                for stage, _, _, multiplicity in entry.get('stages') or ():
                    zones.zone_name(entry['key'], stage)
                    self.assertGreater(multiplicity, 0)
                self.assertIn(entry['route'], ('generated', 'file', 'discover'))
                self.assertIn(entry['phase'], ('decode', 'prefill', 'both'))
                if entry['route'] == 'file':
                    self.assertEqual(kernels.check_destination(entry['path']), '/opt/tt-metal/' + entry['path'])
                if entry['route'] == 'generated':
                    self.assertIn(entry['part'], ('compute', 'reader', 'writer'))
        self.assertEqual(sorted(kernels.GENERATED_PARTS), ['compute', 'reader', 'writer'])

    def test_repo_copies_match_their_pins(self):
        """A pinned entry's repo copy is the bytes its anchors were written against (K64j's recorded shas)."""
        for entry in kernels.KERNELS:
            if entry.get('sha256') and entry.get('repo'):
                with self.subTest(key=entry['key']):
                    self.assertEqual(zones.sha256(read(entry['repo'])), entry['sha256'])
        build_script = read('optimisation/ttnn-op/k64j/build_k64j.sh')
        self.assertIn('K64J_COMPUTE_QWEN=%s' % kernels.BY_KEY['SDPA_DEC']['sha256'], build_script)
        self.assertIn('K64J_READER_QWEN=%s' % kernels.BY_KEY['SDPA_DEC_RD']['sha256'], build_script)

    def test_required_kernels(self):
        """Every phase requires a kernel, so no arm passes having measured nothing: K5-A and the K64j SDPA decode in
        decode, the SDPA prefill compute in prefill (the only prefill kernel whose path the repo pins)."""
        self.assertEqual(sorted(entry['key'] for entry in kernels.KERNELS if entry.get('required')),
                         ['K5A', 'SDPA_DEC', 'SDPA_PF'])
        for phase in kernels.PHASES:
            with self.subTest(phase=phase):
                self.assertTrue([entry for entry in kernels.KERNELS
                                 if entry.get('required') and kernels.in_phase(entry, phase)])

    def test_the_stock_matmul_is_prefill_only(self):
        """In decode the stock matmul runs on most cores every round; its markers would crowd the profiler buffer
        the K5-A stage zones need (llk_profile_plan.marker_budget)."""
        self.assertEqual(kernels.BY_KEY['MM']['phase'], 'prefill')
        self.assertNotIn(kernels.BY_KEY['MM']['path'], kernels.file_requests('decode')[0])
        self.assertIn(kernels.BY_KEY['MM']['path'], kernels.file_requests('prefill')[0])


class DestinationTests(unittest.TestCase):
    def test_only_op_kernel_sources_may_be_overlaid(self):
        good = 'ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp'
        self.assertEqual(kernels.check_destination(good), '/opt/tt-metal/' + good)
        for bad in ('/opt/tt-metal/' + good, 'ttnn/cpp/ttnn/operations/../../x/kernels/a.cpp',
                    'models/demos/blackhole/qwen36/tt/tp_common.py', 'build_Release/lib/_ttnncpp.so',
                    'ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp',
                    'ttnn/cpp/ttnn/operations/x/kernels/tp_common.py', 'tt_metal/tools/profiler/kernel_profiler.hpp',
                    'ttnn/cpp/ttnn/operations/x/kernels/./a.cpp'):
            with self.subTest(path=bad):
                with self.assertRaises(kernels.KernelError):
                    kernels.check_destination(bad)

    def test_the_image_workdir_is_why_copies_are_file_mounts(self):
        """tt-metal resolves a relative kernel path against the working directory before TT_METAL_KERNEL_PATH
        (kernel.cpp resolve_path); the serving images' WORKDIR is /opt/tt-metal, so an overlay directory would
        never be read. If the WORKDIR moves, revisit llk_kernels' delivery."""
        for dockerfile in ('docker/qwen-fast-serving.Dockerfile', 'docker/tenstorrent-serving.Dockerfile'):
            self.assertIn('WORKDIR /opt/tt-metal', read(dockerfile), dockerfile)


class StageTests(unittest.TestCase):
    def test_sdpa_decode_stages_on_the_pinned_copy(self):
        entry = kernels.BY_KEY['SDPA_DEC']
        text = read(entry['repo'])
        result, record = kernels.instrument_entry(entry, text, 'stages')
        self.assertTrue(record['pin_match'])
        self.assertEqual(record['stages_applied'], ['KLOOP', 'TREE', 'FINAL'])
        self.assertEqual([zone['name'] for zone in record['zones']],
                         ['QWEN_LLK_SDPA_DEC', 'QWEN_LLK_SDPA_DEC_KLOOP', 'QWEN_LLK_SDPA_DEC_TREE', 'QWEN_LLK_SDPA_DEC_FINAL'])
        self.assertLessEqual(record['markers']['per_risc_per_program'], zones.MARKER_BUDGET - zones.MARKER_RESERVE)
        self.assertEqual(zones.remove(result), text)

    def test_a_pin_mismatch_gets_envelope_and_sums_only(self):
        entry = kernels.BY_KEY['SDPA_DEC']
        text = read(entry['repo']).replace('// Compile time arguments', '// Compile time args', 1)
        result, record = kernels.instrument_entry(entry, text, 'stages')
        self.assertFalse(record['pin_match'])
        self.assertEqual(record['stages_applied'], [])
        self.assertIn('image bytes differ from the pin', record['note'])
        self.assertEqual([zone['kind'] for zone in record['zones']], ['envelope'])
        self.assertTrue(record['sync']['enabled'])

    def test_k5a_stages_on_the_generated_build(self):
        build = k5a_build()
        texts, records = kernels.instrument_generated(dict(build), 'stages')
        by_key = dict((record['key'], record) for record in records)
        self.assertEqual(sorted(by_key), ['K5A', 'K5A_RD', 'K5A_WR'])
        compute = by_key['K5A']
        self.assertEqual(compute['stages_applied'], ['PRO', 'T1', 'T23', 'T45', 'T6', 'T7', 'EPI'])
        self.assertEqual(compute['markers']['per_risc_per_program'], 168)
        self.assertGreater(compute['sync']['wait_in'], 0)
        self.assertGreater(compute['sync']['wait_out'], 0)
        for part in ('compute', 'reader', 'writer'):
            self.assertEqual(zones.remove(texts[part]), build[part])
            self.assertNotEqual(texts[part], build[part])
        # The served helper's wait (in the native prefix) is timed too.
        self.assertIn('{ DeviceZoneScopedSumN1("QWEN_LLK_WAIT_IN"); CircularBuffer(cb).wait_front(n); }', texts['compute'])
        # The token loop's declarations stay outside every region (the lint would refuse otherwise).
        self.assertIn('        const bool more = t + 1 < SB_T;\n\n        { DeviceZoneScopedN("QWEN_LLK_K5A_T1");',
                      texts['compute'])

    def test_a_refused_part_keeps_its_served_text(self):
        build = k5a_build()
        broken = tuple(stage if stage[0] != 'T6' else ('T6', '        // T6: no such anchor', stage[2], stage[3])
                       for stage in kernels.K5A_STAGES)
        lines = []
        with mock.patch.dict(kernels.BY_KEY['K5A'], stages=broken):
            texts, records = kernels.instrument_generated(dict(build), 'stages', log=lines.append)
        self.assertEqual(texts['compute'], build['compute'])
        refused = [record for record in records if 'refused' in record]
        self.assertEqual([record['kernel'] for record in refused], ['K5A'])
        self.assertIn('exactly once', refused[0]['refused'])
        self.assertTrue(lines and lines[0].startswith('[LLK] K5A not instrumented'))

    def test_tag_level_is_the_envelope_only(self):
        build = k5a_build()
        texts, records = kernels.instrument_generated(dict(build), 'tag')
        for record in records:
            self.assertEqual([zone['kind'] for zone in record['zones']], ['envelope'])
            self.assertEqual(record['markers']['per_risc_per_program'], 2)
        self.assertNotIn('DeviceZoneScopedSum', texts['compute'])


class PlanFilesTests(unittest.TestCase):
    def image(self):
        files = {}
        for entry in kernels.KERNELS:
            if entry['route'] == 'file' and entry.get('repo'):
                files[entry['path']] = read(entry['repo'])
        return files

    def test_decode_files(self):
        paths, patterns = kernels.file_requests('decode')
        self.assertIn(kernels.BY_KEY['SDPA_DEC']['path'], paths)
        self.assertNotIn(kernels.BY_KEY['SDPA_PF']['path'], paths)
        self.assertEqual(patterns, list(kernels.BY_KEY['CONV_GATES']['patterns']))
        found = 'ttnn/cpp/ttnn/operations/transformer/gdn_decode_conv_gates/device/kernels/compute/conv_gates.cpp'
        files = self.image()
        files[found] = 'void kernel_main() {\n    cb_wait_front(0, 1);\n}\n'
        planned = kernels.plan_files('decode', files, {patterns[0]: [found]}, 'stages')
        by_key = dict((record.get('key') or record['kernel'], (path, text, record)) for path, text, record in planned)
        self.assertIsNotNone(by_key['SDPA_DEC'][1])
        self.assertIsNotNone(by_key['CONV_GATES_CONV_GATES'][1])
        self.assertNotIn('MM', by_key)                     # prefill only
        self.assertIn('not in the image', by_key['SDPA_COMMON'][2]['refused'])
        self.assertEqual(by_key['ATTN_PREP'][2]['pin_match'], True)

    def test_discover_with_no_match_is_reported(self):
        planned = kernels.plan_files('prefill', {}, {}, 'stages')
        refused = dict((record['kernel'], record['refused']) for _, _, record in planned if record.get('refused'))
        self.assertIn('no file in the image matches', refused['AGMM'])

    def test_header_is_sums_only_and_absent_at_tag(self):
        header = kernels.BY_KEY['SDPA_COMMON']
        text = '#pragma once\n#include "api/compute/cb_api.h"\nALWI void f(uint32_t cb) {\n    cb_wait_front(cb, 1);\n}\n'
        planned = kernels.plan_files('prefill', {header['path']: text}, {}, 'stages')
        record = [record for path, _, record in planned if path == header['path']][0]
        self.assertIsNone(record['envelope'])
        self.assertEqual(record['sync']['wait_in'], 1)
        planned = kernels.plan_files('prefill', {header['path']: text}, {}, 'tag')
        record = [record for path, _, record in planned if path == header['path']][0]
        self.assertIn('nothing to do at level tag', record['refused'])

    def test_zone_index(self):
        planned = kernels.plan_files('decode', self.image(), {}, 'stages')
        index = kernels.zone_index([record for _, _, record in planned])
        self.assertEqual(index['QWEN_LLK_SDPA_DEC_KLOOP']['kernel'], 'SDPA_DEC')
        self.assertEqual(index['QWEN_LLK_SDPA_DEC_KLOOP']['stage'], 'KLOOP')
        self.assertEqual(index['QWEN_LLK_SDPA_DEC']['kind'], 'envelope')


class SyntaxTests(unittest.TestCase):
    def test_module_parses_as_python_37(self):
        with open(os.path.join(HERE, 'llk_kernels.py'), encoding='utf-8') as handle:
            ast.parse(handle.read(), 'llk_kernels.py', feature_version=(3, 7))


if __name__ == '__main__':
    unittest.main()
