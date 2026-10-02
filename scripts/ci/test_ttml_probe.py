"""ttml_probe against a fake ttml (torch on CPU): stage order and gating, thresholds, the op references, the decision rule, the report's
privacy, the mesh descriptor validator and the backports manifest."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import tau_lab_report as rep  # noqa: E402
import ttml_probe as probe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SMALL = dict(sdpa_square_mask_6500=48, sdpa_square_mask_8600=64, per_row_rope=32, grouped_conv=3, chunked_ce_248320=8, topk16=4)


class FakeOps(object):
    """What a healthy ttml would do, with knobs to break one thing at a time."""
    def __init__(self, **broken):
        self.broken = broken
        self.calls = []

    def import_ok(self):
        self.calls.append('import')
        return not self.broken.get('import')

    def devices(self):
        return 0 if self.broken.get('import') else 4

    def tiny_train(self, steps):
        self.calls.append('train')
        if self.broken.get('train'):
            return [2.0] * steps
        return [2.0 * (0.9 ** step) for step in range(steps)]

    def allreduce_gbps(self):
        self.calls.append('allreduce')
        return self.broken.get('gbps', 85.0)

    def run_op(self, name, inputs):
        self.calls.append(name)
        if self.broken.get('raise') == name:
            raise RuntimeError('SENTINELTEXT')
        import dflash2_torch as d2
        if name.startswith('sdpa'):
            out = torch.nn.functional.scaled_dot_product_attention(inputs['q'], inputs['k'], inputs['v'], attn_mask=inputs['mask'])
        elif name == 'per_row_rope':
            cos, sin = d2.rotary(inputs['positions'], inputs['x'].shape[-1], inputs['theta'], torch.float32)
            out = d2.apply_rope(inputs['x'], cos, sin)
        elif name == 'grouped_conv':
            conv = d2.GroupedDynamicCausalConv(inputs['hidden'].shape[-1], 2, inputs['group'])
            with torch.no_grad():
                conv.base_kernel.copy_(inputs['base_kernel'])
                conv.kernel_projection.weight.copy_(inputs['projection'])
                out = conv.prepare(inputs['hidden'], inputs['block'])[0]
        elif name == 'chunked_ce_248320':
            out = torch.nn.functional.cross_entropy(inputs['hidden'] @ inputs['weight'].T, inputs['targets'], reduction='none')
        else:
            out = torch.topk(inputs['logits'], inputs['k'], dim=-1).values
        if self.broken.get('wrong') == name:
            out = out * 1.1
        return out

    def rect_attention_gib(self):
        return self.broken.get('rect', 4.5)

    def adam_view_error(self):
        return self.broken.get('adam', 1e-3)

    def layer_tflops(self):
        return self.broken.get('tflops', 30.0)

    def ddp_memory_gib(self):
        return self.broken.get('memory', 22.0)


class StageTests(unittest.TestCase):
    def go(self, **broken):
        return probe.run(FakeOps(**broken), sizes=SMALL)

    def test_a_healthy_ttml_is_a_go(self):
        report = self.go()
        self.assertEqual(report['decision'], 'GO')
        self.assertEqual([report['stages'][name]['status'] for name in probe.STAGES], ['PASS'] * 7)
        for name in probe.OPS:
            self.assertTrue(report['stages']['P3']['ops'][name]['passed'], name)

    def test_the_rectangular_path_memory_is_informational(self):
        report = self.go(rect=9.0)
        self.assertEqual(report['decision'], 'GO')
        self.assertFalse(report['stages']['P3']['ops']['composite_rect_memory']['fits'])

    def test_each_gating_failure_is_a_fallback(self):
        for knob in (dict(train=True), dict(gbps=40.0), dict(wrong='per_row_rope'), dict(wrong='grouped_conv'), dict(wrong='topk16'),
                     dict(wrong='sdpa_square_mask_8600'), dict(adam=0.5), dict(tflops=9.0)):
            report = self.go(**knob)
            self.assertEqual(report['decision'], 'FALLBACK', knob)

    def test_p6_is_reported_but_not_gating(self):
        report = self.go(memory=40.0)
        self.assertEqual(report['stages']['P6']['status'], 'FAIL')
        self.assertEqual(report['decision'], 'GO')

    def test_a_failed_import_stops_everything_after_it(self):
        ops = FakeOps(**{'import': True})
        report = probe.run(ops, sizes=SMALL)
        self.assertEqual(report['stages']['P0']['status'], 'FAIL')
        self.assertTrue(all(report['stages'][name]['status'] == 'NOT_RUN' for name in probe.STAGES[1:]))
        self.assertEqual(report['decision'], 'NOT_ESTABLISHED')
        self.assertNotIn('train', ops.calls)

    def test_a_loss_that_does_not_fall_stops_the_later_stages_unless_asked_to_keep_going(self):
        report = self.go(train=True)
        self.assertEqual(report['stages']['P1']['status'], 'FAIL')
        self.assertEqual(report['stages']['P2']['status'], 'NOT_RUN')
        keep = probe.run(FakeOps(train=True), sizes=SMALL, stop_after_p1_failure=False)
        self.assertEqual(keep['stages']['P2']['status'], 'PASS')
        self.assertEqual(keep['decision'], 'FALLBACK')

    def test_a_probe_that_raises_is_a_failed_probe_and_the_rest_still_run(self):
        report = self.go(raise_='x', **{'raise': 'per_row_rope'})
        ops = report['stages']['P3']['ops']
        self.assertTrue(ops['per_row_rope']['raised'])
        self.assertFalse(ops['per_row_rope']['passed'])
        self.assertTrue(ops['topk16']['passed'])
        self.assertEqual(report['stages']['P4']['status'], 'PASS')
        self.assertNotIn('SENTINELTEXT', json.dumps(report))

    def test_thresholds_are_the_designs(self):
        limits = probe.THRESHOLDS
        self.assertEqual((limits['p5_tflops_min'], limits['p2_gbps_min']), (15.0, 60.0))
        self.assertLessEqual(limits['p2_gbps_min'], 84.0)
        self.assertEqual(probe.GATING, ('P0', 'P1', 'P2', 'P3', 'P4'))

    def test_the_ops_are_the_designs_list(self):
        self.assertEqual(set(probe.OPS), set(['sdpa_square_mask_6500', 'sdpa_square_mask_8600', 'per_row_rope', 'grouped_conv',
                                              'chunked_ce_248320', 'topk16']))
        self.assertEqual(probe.SDPA_SIZES, dict(sdpa_square_mask_6500=6500, sdpa_square_mask_8600=8600))


class ReferenceTests(unittest.TestCase):
    def test_references_are_deterministic_and_have_the_real_vocabulary(self):
        a, expected_a = probe.reference_inputs('chunked_ce_248320', sizes=SMALL)
        b, expected_b = probe.reference_inputs('chunked_ce_248320', sizes=SMALL)
        self.assertTrue(torch.equal(expected_a, expected_b))
        self.assertEqual(a['weight'].shape[0], 248320)
        logits, top = probe.reference_inputs('topk16', sizes=SMALL)
        self.assertEqual(top.shape[-1], 16)

    def test_the_grouped_conv_reference_is_block_isolated(self):
        inputs, expected = probe.reference_inputs('grouped_conv', sizes=SMALL)
        self.assertEqual(expected.shape, inputs['hidden'].shape)
        import dflash2_torch as d2
        conv = d2.GroupedDynamicCausalConv(64, 2, 16)
        with torch.no_grad():
            conv.base_kernel.copy_(inputs['base_kernel'])
            conv.kernel_projection.weight.copy_(inputs['projection'])
            whole = conv.prepare(inputs['hidden'])[0]
        self.assertFalse(torch.allclose(whole, expected))            # without block isolation the answer is different

    def test_the_rope_reference_uses_arbitrary_positions(self):
        inputs, _ = probe.reference_inputs('per_row_rope', sizes=SMALL)
        steps = inputs['positions'][0][1:] - inputs['positions'][0][:-1]
        self.assertTrue(bool((steps != 1).any()))

    def test_relative_error_and_non_finite_outputs(self):
        self.assertEqual(probe.relative_error(torch.ones(3), torch.ones(3)), 0.0)
        self.assertEqual(probe.relative_error(torch.full((3,), float('nan')), torch.ones(3)), float('inf'))
        self.assertEqual(probe.relative_error(torch.ones(2), torch.ones(3)), float('inf'))

    def test_an_unknown_probe_is_refused(self):
        with self.assertRaises(probe.ProbeError):
            probe.reference_inputs('nope')


class ReportTests(unittest.TestCase):
    def test_the_report_is_public(self):
        report = probe.public_report(probe.run(FakeOps(), sizes=SMALL))
        text = json.dumps(report)
        self.assertNotIn('SENTINELTEXT', text)
        rep.assert_public(report, words=probe.WORDS)

    def test_a_stray_string_is_refused(self):
        report = probe.run(FakeOps(), sizes=SMALL)
        report['stages']['P0']['note'] = 'a host name'
        with self.assertRaises(rep.PrivacyError):
            probe.public_report(report)

    def test_cli_writes_the_report_and_prints_statuses(self):
        root = tempfile.mkdtemp()
        try:
            out = os.path.join(root, 'probe.json')
            lines = []
            original = probe.run
            probe.run = lambda ops, **kwargs: original(ops, sizes=SMALL, **kwargs)
            try:
                code = probe.main(['--out', out], say=lines.append, make_ops=FakeOps)
            finally:
                probe.run = original
            self.assertEqual(code, 0)
            self.assertEqual(lines[-1], 'decision GO')
            with open(out) as handle:
                self.assertEqual(json.load(handle)['decision'], 'GO')
        finally:
            shutil.rmtree(root)

    def test_cli_without_an_adapter_refuses_by_type(self):
        lines = []
        self.assertEqual(probe.main(['--out', os.devnull], say=lines.append), 2)
        self.assertEqual(lines, ['refused: NotImplementedError'])


class MeshTests(unittest.TestCase):
    def descriptor(self):
        with open(os.path.join(HERE, 'qwen_p150x4_ring_mesh_graph_descriptor.textproto'), encoding='utf-8') as handle:
            return handle.read()

    def test_the_repositorys_four_card_descriptor_is_accepted(self):
        self.assertEqual(probe.validate_mesh(self.descriptor()), [])

    def test_wrong_dimensions_channels_and_a_pair_are_refused(self):
        text = self.descriptor()
        self.assertTrue(probe.validate_mesh(text.replace('count: 2', 'count: 4')))
        self.assertTrue(probe.validate_mesh(text.replace('dims: [ 2, 2 ]', 'dims: [ 1, 4 ]')))
        import tp4_mesh
        self.assertTrue(probe.validate_mesh(tp4_mesh.pair_descriptor_text()))                 # a (1, 2) pair: two devices
        self.assertTrue(probe.validate_mesh('mesh_descriptors { }'))                             # unreadable


class BackportTests(unittest.TestCase):
    def manifest(self):
        with open(os.path.join(ROOT, 'patches', 'tt-train', 'backports.json'), encoding='utf-8') as handle:
            return json.load(handle)

    def test_the_committed_manifest_lists_the_six_fixes_and_the_guard_pr(self):
        manifest = self.manifest()
        prs = [entry['pr'] for entry in manifest['backports']]
        self.assertEqual(prs, ['57825', '57829', '57933', '58002', '57863', '57748'])
        self.assertEqual([entry['pr'] for entry in manifest['guards']], ['57911'])
        self.assertEqual(probe.check_backports(manifest, os.path.join(ROOT, 'patches', 'tt-train')), [])

    def test_a_ready_entry_needs_its_patch_file(self):
        root = tempfile.mkdtemp()
        try:
            manifest = dict(backports=[dict(pr='1', purpose='x', status='ready'), dict(pr='2', purpose='y', status='not_fetched')])
            self.assertEqual(probe.check_backports(manifest, root), ['PR 1 is ready but its patch file is missing'])
            with open(os.path.join(root, '1.patch'), 'w'):
                pass
            self.assertEqual(probe.check_backports(manifest, root), [])
        finally:
            shutil.rmtree(root)

    def test_refusals(self):
        self.assertTrue(probe.check_backports({}, '.'))
        self.assertTrue(probe.check_backports(dict(backports=[dict(pr='1')]), '.'))
        self.assertTrue(probe.check_backports(dict(backports=[dict(pr='1', purpose='a', status='weird')]), '.'))
        self.assertTrue(probe.check_backports(dict(backports=[dict(pr='1', purpose='a', status='ready')] * 2), '.'))


class ImageTests(unittest.TestCase):
    DOCKERFILE = os.path.join(ROOT, 'docker', 'tt-train', 'Dockerfile')

    def read(self):
        with open(self.DOCKERFILE, encoding='utf-8') as handle:
            return handle.read()

    def test_the_probes_imports_are_all_copied_into_the_image(self):
        import ast
        text = self.read().replace(chr(92) + chr(10), ' ')
        copied = set()
        for line in text.splitlines():
            if line.startswith('COPY ') and 'scripts/ci' in line:
                copied.update(os.path.basename(part)[:-3] for part in line.split()[1:-1] if part.endswith('.py'))
        pending, seen = ['ttml_probe'], set()
        while pending:
            module = pending.pop()
            if module in seen:
                continue
            seen.add(module)
            with open(os.path.join(HERE, module + '.py'), encoding='utf-8') as handle:
                tree = ast.parse(handle.read())
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                for name in names:
                    head = name.split('.')[0]
                    if os.path.exists(os.path.join(HERE, head + '.py')):
                        self.assertIn(head, copied, '%s imports %s, which the image does not copy' % (module, head))
                        pending.append(head)

    def test_the_base_is_an_argument_and_the_build_fails_on_a_missing_patch(self):
        text = self.read()
        self.assertIn('ARG BASE', text)
        self.assertIn('--build-tt-train', text)
        self.assertIn('is ready but its patch is missing', text)
        self.assertNotIn('|| true', text)


if __name__ == '__main__':
    unittest.main()
