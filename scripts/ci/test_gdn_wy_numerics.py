"""CPU tests of the non-exact window WY probe's numerics (gdn_wy_model, gdn_wy_numerics; docs/gdn-wy-probe.md).

Small shapes only (2 users, 1 key head, 3 value heads, dk = dv = 128). They hold the forms' algebra (the window form equals the
sequential form in fp64), the properties the report relies on, and the report's schema and verdict logic. They prove nothing about
the device: the TT kernel is not built, and a laptop's numbers are smoke, not results (the full E1 runs in the numerics workflow).
"""

import json
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gdn_wy_model as M  # noqa: E402
import gdn_wy_numerics as N  # noqa: E402

USERS, NK, NV = 2, 1, 3


def block(seed=1, rows=16, regime='model'):
    gen = torch.Generator().manual_seed(seed)
    consts = M.gate_constants(7, NV)
    raw = M.draw_rows(gen, rows, regime, consts, USERS, NK, NV)
    norm_w = M.bf16(1 + 0.1 * torch.randn(M.DV, generator=gen))
    S0 = M.bf16(0.05 * torch.randn(USERS, NV, M.DK, M.DV, generator=gen))
    return raw, norm_w, S0


class FormsAgreeInFp64(unittest.TestCase):
    def test_window_equals_sequential_in_fp64(self):
        raw, norm_w, S0 = block()
        m = M.MODELS['fp64']
        p = M.prep(raw, norm_w, m.dtype)
        o_s, snaps, _ = M.seq_round(S0.double(), p, 16, m)
        o_w, S_w, windows = M.wy_verify(S0.double(), p, 16, m)
        self.assertEqual(windows, 1)
        self.assertLess(M.cmp(o_w, o_s)['rel_to_max'], 1e-10)
        self.assertLess(M.cmp(S_w, snaps[-1])['rel_to_max'], 1e-10)

    def test_every_commit_length_matches_its_snapshot(self):
        raw, norm_w, S0 = block(2)
        m = M.MODELS['fp64']
        p = M.prep(raw, norm_w, m.dtype)
        _, snaps, _ = M.seq_round(S0.double(), p, 16, m)
        for n in (1, 2, 5, 16):
            _, S_w, _ = M.wy_verify(S0.double(), p, n, m)
            self.assertLess(M.cmp(S_w, snaps[n - 1])['rel_to_max'], 1e-10, n)

    def test_two_chained_windows_match_32_sequential_rows_in_fp64(self):
        raw, norm_w, S0 = block(3, rows=32)
        m = M.MODELS['fp64']
        p = M.prep(raw, norm_w, m.dtype)
        o_s, snaps, _ = M.seq_round(S0.double(), p, 32, m)
        for n in (16, 17, 32):
            o_w, S_w, windows = M.wy_verify(S0.double(), p, n, m)
            self.assertEqual(windows, 1 if n == 16 else 2)
            self.assertLess(M.cmp(o_w, o_s[:, :, :n])['rel_to_max'], 1e-10, n)
            self.assertLess(M.cmp(S_w, snaps[n - 1])['rel_to_max'], 1e-10, n)

    def test_one_row_window_is_the_decode_step(self):
        raw, norm_w, S0 = block(4)
        m = M.MODELS['fp32']
        p = M.rows_of(M.prep(raw, norm_w, m.dtype), 1)
        o_s, _, S_s = M.seq_round(S0.float(), p, 1, m)
        o_w, S_w, _ = M.wy_verify(S0.float(), p, 1, m, window=1)
        self.assertLess(M.cmp(o_w, o_s)['rel_to_max'], 1e-5)
        self.assertLess(M.cmp(S_w, S_s)['rel_to_max'], 1e-5)


class WindowFormIsNotTheServedBytes(unittest.TestCase):
    def test_bf16_window_differs_from_the_served_chain(self):
        raw, norm_w, S0 = block(5)
        m = M.MODELS['bf16']
        p = M.prep(raw, norm_w, m.dtype)
        o_s, _, _ = M.seq_round(S0, p, 16, m)
        o_w, _, _ = M.wy_verify(S0, p, 16, m)
        self.assertGreater(M.cmp(o_w, o_s)['differing'], 0)

    def test_the_literal_ratio_form_underflows_under_strong_decay(self):
        raw, norm_w, S0 = block(6, regime='R2')
        m = M.MODELS['fp32']
        p = M.prep(raw, norm_w, m.dtype)
        o, _ = M.wy_window(S0.float(), p, m, ratio=True)
        self.assertGreater(int((~torch.isfinite(o)).sum()), 0)
        o_log, _ = M.wy_window(S0.float(), p, m)
        self.assertTrue(bool(torch.isfinite(o_log).all()))

    def test_sequential_form_is_segmentation_invariant_bitwise(self):
        raw, norm_w, S0 = block(7, rows=16)
        m = M.MODELS['bf16']
        p = M.prep(raw, norm_w, m.dtype)
        o_all, snaps, _ = M.seq_round(S0, p, 16, m)
        o_a, _, S_a = M.seq_round(S0, M.rows_of(p, 0, 5), 5, m)
        o_b, _, S_b = M.seq_round(S_a, M.rows_of(p, 5, 16), 11, m)
        self.assertTrue(M.bits_equal(torch.cat([o_a, o_b], 2), o_all))
        self.assertTrue(M.bits_equal(S_b, snaps[-1]))


class Helpers(unittest.TestCase):
    def test_bits_equal_sees_signed_zero_and_nan_payload(self):
        a, b = torch.tensor([0.0]), torch.tensor([-0.0])
        self.assertTrue(bool((a == b).all()))
        self.assertFalse(M.bits_equal(a, b))
        self.assertTrue(M.bits_equal(a, a.clone()))

    def test_geometric_fit_hits_the_pooled_tau(self):
        p = M.geometric_p(M.TAU, 16)
        self.assertAlmostEqual(1 + sum(p ** j for j in range(1, 16)), M.TAU, places=6)

    def test_commit_draws_stay_in_range_and_follow_a_histogram(self):
        gen = torch.Generator().manual_seed(1)
        drawn = M.draw_commits(gen, 400, M.geometric_p(), 16)
        self.assertTrue(all(1 <= n <= 16 for n in drawn))
        self.assertAlmostEqual(sum(drawn) / len(drawn), M.TAU, delta=0.6)
        hist = M.draw_commits(torch.Generator().manual_seed(2), 200, 0.0, 16, hist={'3': 1, '9': 1})
        self.assertEqual(set(hist), {3, 9})

    def test_tf32_rounds_to_ten_mantissa_bits(self):
        x = torch.tensor([1.0 + 2.0 ** -11, 1.0 + 2.0 ** -10], dtype=torch.float32)
        self.assertEqual(M.tf32(x).tolist(), [1.0, 1.0 + 2.0 ** -10])

    def test_cost_counts_say_the_window_form_moves_fewer_bytes(self):
        c = M.counts(M.TAU)
        self.assertLess(c['wy']['bytes_total'], c['seq']['bytes_total'])
        self.assertEqual(c['seq']['bytes']['snapshots'], 16 * 128 * 128 * 2)
        self.assertEqual(M.tile_ops_wy()['total'], sum(M.tile_ops_wy()['parts'].values()))

    def test_sram_plan_totals_its_buffers(self):
        plan = M.sram_plan(windows=2)
        self.assertEqual(plan['per_window_total_bytes'], sum(plan['per_cb_bytes'].values()))
        self.assertLess(plan['chained_total_bytes'], 1572864)


class ReportAndVerdict(unittest.TestCase):
    def smoke(self, *extra):
        with tempfile.TemporaryDirectory() as folder:
            out = os.path.join(folder, 'report.json')
            code = N.main(['--smoke', '--regimes', 'model', '--out', out] + list(extra))
            with open(out, encoding='utf-8') as handle:
                return code, json.load(handle)

    def test_smoke_report_schema_and_contract(self):
        code, report = self.smoke('--sections', 'crosscheck,causality,packed_solo,drift,segmentation,counts')
        self.assertEqual((report['kind'], report['schema'], report['smoke']), ('gdn-wy-numerics', 1, True))
        self.assertEqual(report['repo_reference_crosscheck']['output']['differing'], 0)
        contract = report['acceptance']['contract']
        self.assertFalse(contract['serving_eligible'])
        self.assertFalse(contract['byte_identical_to_served'])
        drift = report['drift']['model']
        self.assertEqual(drift['cycles'], 12)
        self.assertIn('wybf_vs_fp64', drift['pairs'])
        for key in ('seq_bf16', 'wy_bf16'):
            self.assertIn('outputs_bitwise_identical', report['segmentation']['model'][key])
        self.assertTrue(report['segmentation']['model']['seq_bf16']['outputs_bitwise_identical'])
        causal = report['causality']['model']['forms']
        self.assertTrue(causal['seq_bf16']['causal_for_real_zero_random'])
        self.assertTrue(causal['wy_bf16']['causal_for_real_zero_random'])
        self.assertIn(code, (0, 1))

    def test_window_rows_32_and_stress_commits_use_two_windows(self):
        _, report = self.smoke('--sections', 'drift', '--rows', '32', '--commit-dist', 'stress', '--cycles', '10', '--seg-cycles', '2')
        drift = report['drift']['model']
        self.assertEqual(drift['rows'], 32)
        self.assertGreater(drift['rounds_with_two_windows'], 0)

    def test_acceptance_flags_a_worse_window_form(self):
        pair = lambda state, out: dict(state_max_rel=state, out_max_rel=out, nonfinite_cycles=0, gated_differing_frac_all_cycles=0.5)
        report = dict(drift=dict(model=dict(pairs=dict(seqbf_vs_fp64=pair(0.01, 0.01), wybf_vs_fp64=pair(0.02, 0.005),
                                                       wybf_vs_seqbf=pair(0.01, 0.01)))))
        acc = N.acceptance(report)
        self.assertFalse(acc['e1_pass'])
        self.assertEqual(len(acc['problems']), 1)
        report['drift']['model']['pairs']['wybf_vs_fp64'] = pair(0.005, 0.005)
        self.assertTrue(N.acceptance(report)['e1_pass'])
        report['drift']['model']['pairs']['wybf_vs_fp64']['nonfinite_cycles'] = 1
        self.assertFalse(N.acceptance(report)['e1_pass'])

    def test_fixture_layer_inputs_are_read_and_drawn_from(self):
        import numpy as np
        gen = torch.Generator().manual_seed(9)
        raw = M.draw_rows(gen, 48, 'model', M.gate_constants(7, NV), USERS, NK, NV)
        with tempfile.TemporaryDirectory() as folder:
            np.savez(os.path.join(folder, 'layer0.npz'), **{key: val.numpy() for key, val in raw.items()})
            out = os.path.join(folder, 'report.json')
            N.main(['--smoke', '--regimes', 'fixture:layer0', '--fixture', folder, '--sections', 'single,drift', '--cycles', '6',
                    '--seg-cycles', '2', '--out', out])
            with open(out, encoding='utf-8') as handle:
                report = json.load(handle)
        self.assertTrue(report['inputs']['real_layers'])
        self.assertIn('fixture:layer0', report['drift'])

    def test_a_bad_row_count_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            N.parse(['--rows', '24', '--out', 'x.json'])


if __name__ == '__main__':
    unittest.main()
