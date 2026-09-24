"""CPU checks for P1, the pair drafter's row-1 probe (probe_pair_row_card_b.py, run_card_b.sh); no device, no ttnn.

  - the pure helpers:
    - the operands: fixed draws, and regimes that change only what they name (user b's bits are the same in the
      normal and partner100 regimes);
    - the pair's host assembly is key_value_plan's;
    - the masks: the single mask is pair_row_exact's, the packed mask's blocks are it, the swapped, leading and
      per-head masks keep one visible key per pad row;
    - the bitwise comparison;
    - the verdict table on synthetic reports, one per outcome, and the verdict line;
  - the contract: the pinned SDPA sources are the T16 admission's ORIGINAL, and the recorded ones are among its
    NATIVE_SOURCES;
  - a full dry run of the device flow against a torch stand-in for ttnn:
    - exact arithmetic gives NOT-SDPA;
    - a residue on rows whose leading chunks are all masked (what M1 names) gives M1, with the fold exact;
    - a lower-face residue gives SDPA-FACE, with the fold still exact and the unshifted fold not;
    - a faulty six-piece concat gives DATA-MOVEMENT;
    - a lower-face head gives head=DIFFERS;
    - the row-local sweep and the timing rows are complete;
  - the runner (needs bash):
    - the dry run launches on card B by board id with the arm's graft mounts, the checkout's served modules
      beside the harness and a fresh kernel cache;
    - the watcher pass;
    - KOPGRAFT64=none;
    - the pre-launch graft checks refuse a missing part, a manifest that does not verify and a wrong binary.

    py -3.11 -B -m unittest discover -s optimisation/ttnn-op/pair_row_probe -p 'test_*.py'
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CI = ROOT / 'scripts' / 'ci'
for path in (str(HERE), str(CI)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

import probe_pair_row_card_b as probe  # noqa: E402

RUNNER = HERE / 'run_card_b.sh'
CARD_B = 'blackhole-F36F768B9A5CAFA0'
CARD_M = 'blackhole-CEF5729692C19E6D'
NL = chr(10)


def bits(tensor):
    return tensor.contiguous().to(torch.bfloat16).view(torch.int16)


# ---------------------------------------------------------------------------------------------
# A torch stand-in for the ttnn surface the harness and the served modules use.
# ---------------------------------------------------------------------------------------------

class FakeTensor:
    """A device tensor. `base` is the buffer it lives in: a reshape is a view that shares its source's buffer, as in
    ttnn, so deallocating the view frees the source (the card-B watcher pass lost its K/V that way)."""

    buffers = itertools.count(1)

    def __init__(self, value, dtype, layout='tile', base=None):
        self.value, self.dtype, self.layout = value, dtype, layout
        self.shape = tuple(value.shape)
        # a buffer number, never reused (id() is, once an object is collected)
        self.base = next(FakeTensor.buffers) if base is None else base

    def memory_config(self):
        return 'dram'

    def buffer_address(self):
        return self.base


class FakeTtnn:
    """ttnn on torch. The SDPA is a GQA reference computed per (batch, head) in float64; matmul and linear are
    computed per 16-row block, so a row's result never depends on where its block sits. `mode` injects the
    effects the verdicts separate:
      exact  nothing
      m1     rows whose first visible key lies past the first 32-key chunk come out scaled by 1 + 2^-5 (what M1 names)
      face   the SDPA's rows 16-31 come out scaled (a lower-face asymmetry)
      dm     a six-piece concat (the pair's K/V assembly) flips one element (a data-movement fault)
      head   linear's rows 16-31 come out scaled (a lower-face head)"""

    bfloat16, bfloat8_b, float32, uint32 = 'bf16', 'bf8', 'fp32', 'u32'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'tile', 'row', 'dram'
    MathFidelity = SimpleNamespace(HiFi4='hifi4')

    def __init__(self, mode='exact'):
        self.mode, self.calls, self.freed, self.captures = mode, [], 0, 0
        self.freed_buffers = set()
        self.transformer = SimpleNamespace(scaled_dot_product_attention=self.sdpa)
        self.experimental = SimpleNamespace(rotary_embedding_hf=self.rotary)
        self.device = None

    # device and trace
    def open_device(self, **options):
        self.device = SimpleNamespace(options=options, enable_program_cache=lambda: None)
        return self.device

    def close_device(self, device):
        self.calls.append('close_device')

    def synchronize_device(self, device):
        pass

    def begin_trace_capture(self, device, cq_id=0):
        self.captures += 1
        return SimpleNamespace(kind='trace')

    def end_trace_capture(self, device, trace, cq_id=0):
        pass

    def execute_trace(self, device, trace, cq_id=0, blocking=True):
        pass

    def release_trace(self, device, trace):
        pass

    # configuration objects
    def WormholeComputeKernelConfig(self, **options):
        return ('kernel', tuple(sorted(options.items())))

    def SDPAProgramConfig(self, **options):
        return ('program', tuple(sorted(options.items())))

    def MatmulMultiCoreReuseMultiCast1DProgramConfig(self, **options):
        return ('matmul', tuple(sorted(options.items())))

    # data
    def cast(self, value, dtype):
        if dtype == self.float32:
            return value.float()
        return value.to(torch.bfloat16) if value.is_floating_point() else value

    def from_torch(self, value, dtype=None, layout=None, device=None, memory_config=None):
        return FakeTensor(self.cast(value.clone(), dtype or self.bfloat16), dtype or self.bfloat16, layout)

    def live(self, *tensors):
        for tensor in tensors:
            if tensor.base in self.freed_buffers:
                raise RuntimeError('TT_THROW: Tensor is not allocated')

    def to_torch(self, tensor):
        self.live(tensor)
        return tensor.value.clone()

    def deallocate(self, tensor):
        self.freed += 1
        self.freed_buffers.add(tensor.base)

    def slice(self, tensor, start, end, steps=None):
        self.live(tensor)
        index = tuple(slice(low, high) for low, high in zip(start, end))
        return FakeTensor(tensor.value[index].clone(), tensor.dtype)

    def concat(self, parts, dim, memory_config=None):
        self.live(*parts)
        value = torch.cat([part.value for part in parts], dim=dim)
        if self.mode == 'dm' and len(parts) == 6:
            value = value.clone()
            value.view(-1)[12345] = value.view(-1)[12345] + 1
        return FakeTensor(value, parts[0].dtype)

    def reshape(self, tensor, shape):
        # ttnn's rule: a tile-layout reshape is a view (the same buffer) when the last dim is kept and the
        # second-last dims are equal or both tile-aligned; otherwise it is a copy.
        self.live(tensor)
        old, new = tuple(tensor.shape), tuple(shape)
        view = (tensor.layout == 'tile' and old[-1] == new[-1]
                and (old[-2] == new[-2] or (old[-2] % 32 == 0 and new[-2] % 32 == 0)))
        return FakeTensor(tensor.value.reshape(shape), tensor.dtype, tensor.layout, base=tensor.base if view else None)

    def pad(self, tensor, padding, value):
        pads = []
        for low, high in reversed(padding):
            pads.extend([low, high])
        return FakeTensor(torch.nn.functional.pad(tensor.value.float(), pads, value=value).to(tensor.value.dtype),
                          tensor.dtype)

    def typecast(self, tensor, dtype):
        return FakeTensor(self.cast(tensor.value, dtype), dtype)

    # compute
    def blockwise(self, left, right):
        rows = left.shape[-2]
        parts = [left[..., start:start + 16, :].float() @ right.float() for start in range(0, rows, 16)]
        return torch.cat(parts, dim=-2)

    def matmul(self, left, right, dtype=None, compute_kernel_config=None, program_config=None, memory_config=None):
        return FakeTensor(self.blockwise(left.value, right.value), self.float32)

    def linear(self, left, right):
        out = self.blockwise(left.value, right.value)
        if self.mode == 'head' and out.shape[-2] == 32:
            out = out.clone()
            out[..., 16:, :] = out[..., 16:, :] * (1 + 2 ** -5)
        return FakeTensor(out.bfloat16(), self.bfloat16)

    def topk(self, tensor, k, dim=-1, largest=True, sorted=True):
        values, indices = torch.topk(tensor.value.float(), k, dim=dim, largest=largest, sorted=sorted)
        return FakeTensor(values.bfloat16(), self.bfloat16), FakeTensor(indices.to(torch.int16), self.uint32)

    def rms_norm(self, tensor, epsilon, weight, compute_kernel_config=None, memory_config=None):
        value = tensor.value.float()
        scale = torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + epsilon)
        out = value * scale * weight.value.float().reshape(-1)
        return FakeTensor(self.cast(out, tensor.dtype), tensor.dtype)

    def rotary(self, value, cosine, sine, is_decode_mode=False, compute_kernel_config=None, memory_config=None):
        x = value.value.float()
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return FakeTensor(x * cosine.value.float() + rotated * sine.value.float(), self.float32)

    def silu(self, tensor, memory_config=None):
        return FakeTensor(torch.nn.functional.silu(tensor.value.float()), self.float32)

    def multiply(self, left, right, dtype=None):
        return FakeTensor(left.value.float() * right.value.float(), self.float32)

    def add(self, left, right, dtype=None):
        return FakeTensor(left.value.float() + right.value.float(), self.float32)

    def sdpa(self, query, key, value, *, attn_mask, is_causal, scale, program_config, compute_kernel_config,
             memory_config):
        self.live(query, key, value, attn_mask)
        q, k, v, mask = query.value, key.value, value.value, attn_mask.value
        batches, heads, rows = q.shape[0], q.shape[1], q.shape[2]
        kv_heads = k.shape[1]
        out = torch.empty((batches, heads, rows, v.shape[-1]))
        for batch in range(batches):
            for head in range(heads):
                kv = head // (heads // kv_heads)
                head_mask = mask[batch if mask.shape[0] > 1 else 0, head if mask.shape[1] > 1 else 0].float()
                # float64, so a longer key axis of masked keys changes no bf16 bit (a float32 softmax's reduction
                # order moves with the row length); the modes below are the only effects
                scores = q[batch, head].double() @ k[batch, kv].double().T * scale + head_mask.double()
                result = (torch.softmax(scores, dim=-1) @ v[batch, kv].double()).float()
                if self.mode == 'm1':
                    visible = torch.isfinite(head_mask)
                    first = visible.float().argmax(dim=-1)
                    result[first >= 32] = result[first >= 32] * (1 + 2 ** -5)
                elif self.mode == 'face':
                    result[16:] = result[16:] * (1 + 2 ** -5)
                out[batch, head] = result
        return FakeTensor(out.bfloat16(), self.bfloat16)


def dry_run(mode='exact', *extra, hidden=None):
    """The whole harness against FakeTtnn; returns (exit code, report, stdout lines)."""
    fake = FakeTtnn(mode)
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / 'report.json'
        argv = ['--out', str(out), '--skip-binary-check', '--tt-metal-home', '', '--seeds', '0',
                '--regimes', 'normal,partner100', '--leading', '1,65', '--iters', '1', '--rounds', '1',
                '--replays', '1', '--trace-layers', '1', '--trace-warmup', '0', *extra]
        lines = []
        with mock.patch('builtins.print', side_effect=lambda *args, **kwargs: lines.append(' '.join(map(str, args)))), \
                mock.patch.object(probe, 'HIDDEN', hidden or probe.HIDDEN):
            code = probe.main(argv, ttnn=fake, torch=torch)
        report = json.loads(out.read_text())
    return code, report, lines, fake


# ---------------------------------------------------------------------------------------------
# The pure helpers.
# ---------------------------------------------------------------------------------------------

class HelperTests(unittest.TestCase):
    def test_the_pins_are_the_t16_admissions(self):
        from dflash_t16_native_attention_gate import NATIVE_SOURCES, ORIGINAL

        self.assertEqual(probe.PINNED_SOURCES, ORIGINAL)
        self.assertTrue(set(probe.RECORDED_SOURCES) <= set(NATIVE_SOURCES))

    def test_the_regimes_change_only_what_they_name(self):
        normal = probe.build_fixture(torch, 3, 'normal')
        again = probe.build_fixture(torch, 3, 'normal')
        loud = probe.build_fixture(torch, 3, 'partner100')
        for part in ('cache', 'live', 'pad'):
            for name in 'kv':
                self.assertTrue(torch.equal(bits(normal['a'][part][name]), bits(again['a'][part][name])))
                self.assertTrue(torch.equal(bits(normal['b'][part][name]), bits(loud['b'][part][name])), 'b unchanged')
                torch.testing.assert_close(loud['a'][part][name].float(), (normal['a'][part][name].float() * 100),
                                           rtol=1e-2, atol=1e-2)
        self.assertTrue(torch.equal(bits(normal['b']['query']), bits(loud['b']['query'])))
        negative = probe.build_fixture(torch, 3, 'negative')
        for user in 'ab':
            self.assertTrue(bool((negative[user]['query'] >= 0).all()))
            self.assertTrue(bool((negative[user]['cache']['k'][:, :, :32] <= 0).all()))
            scores = negative[user]['query'].float() @ negative[user]['cache']['k'][0, :, :32].float().repeat_interleave(
                4, dim=0).transpose(-1, -2)
            self.assertTrue(bool((scores <= 0).all()), 'every first-chunk score is below zero')
        peaked = probe.build_fixture(torch, 3, 'peaked')
        torch.testing.assert_close(peaked['a']['query'].float(), normal['a']['query'].float() * 4, rtol=1e-2, atol=1e-2)
        with self.assertRaises(ValueError):
            probe.build_fixture(torch, 0, 'other')

    def test_the_pair_assembly_is_key_value_plans(self):
        fixture = probe.build_fixture(torch, 0, 'normal')
        pair = probe.pair_operands(torch, fixture, ('a', 'b'))
        a, b = fixture['a'], fixture['b']
        for name in 'kv':
            expected = torch.cat([a['cache'][name], a['live'][name], a['live'][name],
                                  b['cache'][name], b['live'][name], a['live'][name]], dim=2)
            self.assertTrue(torch.equal(bits(pair['expected'][name]), bits(expected)))
        self.assertEqual(tuple(pair['query'].shape), (1, 16, 32, 128))
        swapped = probe.pair_operands(torch, fixture, ('b', 'a'))
        self.assertTrue(torch.equal(bits(swapped['query'][:, :, :16]), bits(b['query'])))

    def test_the_masks(self):
        from pair_row_exact import fold_mask

        single = probe.single_mask()
        self.assertTrue(torch.equal(bits(single), bits(fold_mask((2048, 2048), 16))))
        packed = probe.packed_mask()
        self.assertTrue(torch.equal(bits(packed[:, :, :16, :2080]), bits(single[:, :, :16])))
        self.assertTrue(torch.equal(bits(packed[:, :, 16:, 2080:]), bits(single[:, :, :16])))
        swapped = probe.swapped_mask(torch)
        self.assertTrue(torch.equal(bits(swapped[:, :, 16:]), bits(single[:, :, :16])))
        for chunks in (1, 65):
            mask = probe.leading_mask(torch, chunks)
            self.assertEqual(tuple(mask.shape), (1, 1, 32, 2080 + 32 * chunks))
            self.assertTrue(bool(torch.isneginf(mask[..., :32 * chunks]).all()))
            self.assertTrue(bool(((mask[0, 0, 16:] == 0).sum(-1) == 1).all()), 'one visible key per pad row')
        noshift = probe.noshift_mask(torch)
        self.assertEqual(tuple(noshift.shape), (1, 32, 32, 2080))
        self.assertTrue(torch.equal(bits(noshift[:, 3:4]), bits(single)))
        self.assertTrue(torch.equal(bits(noshift[:, 12:13]), bits(swapped)))

    def test_leading_keys_fill_65_chunks_from_the_partner(self):
        fixture = probe.build_fixture(torch, 0, 'normal')
        for content in probe.CONTENTS:
            query, keys, values = probe.leading_keys(torch, fixture, 'b', 65, content)
            self.assertEqual(tuple(keys.shape), (1, 4, 4160, 128))
            self.assertTrue(torch.equal(bits(keys[:, :, 2080:]), bits(probe.single_operands(torch, fixture, 'b')[1])))
        self.assertTrue(torch.equal(bits(probe.leading_keys(torch, fixture, 'b', 2, 'partner')[1][:, :, :64]),
                                    bits(fixture['a']['cache']['k'][:, :, :64])))

    def test_compare_is_bitwise(self):
        value = torch.randn(2, 16, 128).bfloat16()
        self.assertTrue(probe.compare(torch, value, value.clone())['equal'])
        changed = value.clone()
        changed[0, 0, 0] = -changed[0, 0, 0] if changed[0, 0, 0] != 0 else 1.0
        result = probe.compare(torch, changed, value)
        self.assertEqual((result['equal'], result['differing']), (False, 1))
        zero = torch.zeros(4).bfloat16()
        self.assertFalse(probe.compare(torch, zero, -zero)['equal'], 'signed zero is a different bit pattern')
        self.assertFalse(probe.compare(torch, torch.zeros(3), torch.zeros(4))['equal'])


def synthetic(**overrides):
    """A report whose cases say M1, unless `overrides` rewrites a group."""
    equal, differs = dict(equal=True, differing=0), dict(equal=False, differing=7)
    cases = []
    for seed in (0,):
        for regime in ('normal', 'partner100'):
            base = dict(seed=seed, regime=regime)
            cases.append(dict(case='C1', rows=[equal, differs], digests=['x', 'y'], **base))
            cases.append(dict(case='C2', rows=[equal, differs], **base))
            cases.append(dict(case='C3', rows=[equal], **base))
            cases.append(dict(case='C5', assembly=equal, **base))
            for chunks in (1, 65):
                for content in probe.CONTENTS:
                    cases.append(dict(case='C4', rows=[differs], chunks=chunks, content=content, digest='d%d' % chunks,
                                      **base))
            for variant in probe.FIXES:
                cases.append(dict(case='C6', variant=variant, rows=[equal, equal], **base))
    cases.append(dict(case='P', seed=0, regime='partner100', equal=True, rows=[equal]))
    for group, rows in overrides.items():
        for case in cases:
            if case['case'] == group:
                if group == 'C5':
                    case['assembly'] = rows
                elif group == 'C4':
                    case.update(rows)
                else:
                    case['rows'] = list(rows)
    timing = [dict(basis='eager', scope='layer', variant='served', median_us=400.0),
              dict(basis='eager', scope='layer', variant='r1g', median_us=360.0)]
    return dict(cases=cases, timing=timing, leading=[1, 65])


class DecideTests(unittest.TestCase):
    equal, differs = dict(equal=True, differing=0), dict(equal=False, differing=7)

    def test_the_m1_report(self):
        verdict = probe.decide(synthetic())
        self.assertEqual((verdict['mechanism'], verdict['fix']), ('M1', 'R1G-EXACT'))
        self.assertAlmostEqual(verdict['eager_change'], -0.1)
        line = probe.verdict_line(verdict)
        self.assertTrue(line.startswith('PAIR_ROW_PROBE mechanism=M1 fix=R1G-EXACT '))
        self.assertIn('c1=row0:2/2,row1:0/2', line)
        self.assertIn('eager=served:400.0us,r1g:360.0us(-10.0%)', line)

    def test_every_other_outcome(self):
        cases = (
            (dict(C1=[self.equal, self.equal]), 'NOT-SDPA'),
            (dict(C3=[self.differs]), 'SDPA-FACE'),
            (dict(C5=self.differs), 'DATA-MOVEMENT'),
            (dict(C1=[self.differs, self.differs]), 'UNEXPECTED'),
            (dict(C4=dict(rows=[self.equal])), 'UNEXPECTED'),
            (dict(C2=[self.equal, self.equal]), 'UNEXPECTED'),
            (dict(P=[self.differs]), 'M1'),
        )
        for overrides, mechanism in cases:
            with self.subTest(overrides=overrides):
                report = synthetic(**{name: value for name, value in overrides.items() if name != 'P'})
                if 'P' in overrides:
                    for case in report['cases']:
                        if case['case'] == 'P':
                            case['equal'] = False
                    mechanism = 'UNEXPECTED'
                self.assertEqual(probe.decide(report)['mechanism'], mechanism)

    def test_content_dependence_is_not_m1(self):
        report = synthetic()
        for case in report['cases']:
            if case['case'] == 'C4' and case['content'] == 'partner100':
                case['digest'] = 'other'
        verdict = probe.decide(report)
        self.assertFalse(verdict['c4_content_free'])
        self.assertEqual(verdict['mechanism'], 'UNEXPECTED')

    def test_the_fix_verdicts(self):
        report = synthetic()
        for case in report['cases']:
            if case['case'] == 'C6' and case['variant'] == 'r1g':
                case['rows'] = [self.equal, self.differs]
        self.assertEqual(probe.decide(report)['fix'], 'R1G-DIFFERS')
        report['cases'] = [case for case in report['cases'] if case['case'] != 'C6']
        self.assertEqual(probe.decide(report)['fix'], 'NO-DECISION')
        self.assertEqual(probe.decide(dict(cases=[]))['mechanism'], 'NO-DECISION')

    def test_errored_cases_are_counted_and_not_judged(self):
        report = synthetic()
        report['cases'].append(dict(case='C6', variant='r1g', error='RuntimeError: x', seed=0, regime='normal'))
        verdict = probe.decide(report)
        self.assertEqual((verdict['errors'], verdict['fix']), (1, 'R1G-EXACT'))
        self.assertIn('case_errors=1', probe.verdict_line(verdict))


# ---------------------------------------------------------------------------------------------
# The device flow against the torch stand-in.
# ---------------------------------------------------------------------------------------------

class FlowTests(unittest.TestCase):
    NO_EXTRAS = ('--head-dtypes', '', '--no-faces')

    def test_exact_arithmetic_is_not_the_sdpa(self):
        code, report, lines, fake = dry_run('exact', *self.NO_EXTRAS)
        self.assertEqual(code, 0, report.get('error'))
        verdict = report['verdict']
        self.assertEqual((verdict['mechanism'], verdict['fix']), ('NOT-SDPA', 'R1G-EXACT'))
        self.assertEqual(verdict['c6'], {variant: '4/4' for variant in probe.FIXES})
        self.assertEqual(verdict['c5'], '2/2')
        self.assertTrue(any(line.startswith('PAIR_ROW_PROBE mechanism=NOT-SDPA') for line in lines))
        self.assertTrue(any(line.startswith('PAIR_ROW_PROBE_DONE passed=True') for line in lines))
        self.assertEqual(sorted({(row['basis'], row['scope']) for row in report['timing']}),
                         [('eager', 'attention'), ('eager', 'layer'), ('trace', 'attention'), ('trace', 'layer')])
        self.assertEqual(sorted({row['variant'] for row in report['timing']}), sorted(probe.VARIANTS))
        self.assertEqual(fake.captures, 10)
        self.assertIn('close_device', fake.calls)

    def test_a_residue_after_masked_leading_chunks_is_m1_and_the_fold_is_exact(self):
        code, report, lines, _ = dry_run('m1', *self.NO_EXTRAS)
        self.assertEqual(code, 0, report.get('error'))
        verdict = report['verdict']
        self.assertEqual((verdict['mechanism'], verdict['fix']), ('M1', 'R1G-EXACT'))
        self.assertEqual(verdict['c1'], dict(row0='2/2', row1='0/2', row1_differing=verdict['c1']['row1_differing']))
        self.assertEqual(verdict['c4'], {'1': '0/2', '65': '0/2'})
        self.assertTrue(verdict['c4_content_free'])
        self.assertEqual(verdict['partner'], '1/1')
        self.assertEqual(verdict['c6'], {variant: '4/4' for variant in probe.FIXES})

    def test_a_lower_face_sdpa_is_sdpa_face_and_only_the_unshifted_fold_differs(self):
        code, report, _, _ = dry_run('face', *self.NO_EXTRAS)
        self.assertEqual(code, 0, report.get('error'))
        verdict = report['verdict']
        self.assertEqual((verdict['mechanism'], verdict['fix']), ('SDPA-FACE', 'R1G-EXACT'))
        self.assertEqual(verdict['c3'], '0/4')
        self.assertEqual(verdict['c6'], {'r1g': '4/4', 'r1g-noshift': '2/4', 'b2': '4/4', 'two-calls': '4/4'})

    def test_a_faulty_assembly_is_data_movement(self):
        code, report, _, _ = dry_run('dm', *self.NO_EXTRAS)
        self.assertEqual(report['verdict']['mechanism'], 'DATA-MOVEMENT')
        self.assertEqual(report['verdict']['c5'], '0/2')

    def test_the_head_is_compared_alone_row_0_and_row_1(self):
        code, report, _, _ = dry_run('exact', '--no-faces', '--head-dtypes', 'bf16', '--no-timing', hidden=64)
        self.assertEqual(code, 0, report.get('error'))
        heads = [case for case in report['cases'] if case['case'] == 'C7']
        self.assertEqual(len(heads), 1)
        self.assertTrue(heads[0]['equal'], heads[0])
        self.assertEqual(report['verdict']['head'], 'EQUAL')
        code, report, _, _ = dry_run('head', '--no-faces', '--head-dtypes', 'bf16,bf8', '--no-timing', hidden=64)
        heads = [case for case in report['cases'] if case['case'] == 'C7']
        self.assertEqual([case['layouts']['row1']['logits_equal'] for case in heads], [False, False])
        self.assertEqual([case['layouts']['row0']['logits_equal'] for case in heads], [True, True])
        self.assertEqual(report['verdict']['head'], 'DIFFERS(bf16,bf8)')

    def test_the_row_local_sweep_covers_every_op(self):
        code, report, _, _ = dry_run('exact', '--head-dtypes', '', '--no-timing', '--regimes', 'normal',
                                     '--leading', '')
        self.assertEqual(code, 0, report.get('error'))
        faces = [case for case in report['cases'] if case['case'] == 'C8']
        self.assertEqual(len(faces), 14)
        self.assertEqual([case['op'] for case in faces if not case['equal']], [], faces)
        self.assertEqual(report['verdict']['rowlocal'], 'EQUAL')

    def test_a_wrong_binary_and_unpinned_sources_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / '_ttnncpp.so'
            binary.write_bytes(b'not the served binary')
            report = dict(failures=[])
            args = probe.parse_args(['--out', str(Path(directory) / 'r.json'), '--expect-binary-sha256', '0' * 64])
            with mock.patch.object(probe, 'loaded_binary', return_value=str(binary)), mock.patch('builtins.print'):
                self.assertFalse(probe.check_binary(args, report))
            self.assertIn('not the expected', report['failures'][0])
            report = dict(failures=[])
            self.assertFalse(probe.check_sources(directory, report))
            self.assertEqual(len(report['failures']), 4)
            self.assertTrue(all('missing' in failure for failure in report['failures']))

    def test_the_arguments(self):
        args = probe.parse_args(['--out', 'x.json'])
        self.assertEqual((args.seeds, args.regimes, args.leading), ([0, 1, 2], list(probe.REGIMES), list(probe.LEADING)))
        self.assertEqual((args.head_dtypes, args.variants), (['bf16', 'bf8'], list(probe.VARIANTS)))
        for argv in (['--regimes', 'other'], ['--leading', '66'], ['--variants', 'b2'], ['--head-dtypes', 'fp8'],
                     ['--expect-binary-sha256', 'abc'], ['--iters', '0']):
            with self.subTest(argv=argv), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                probe.parse_args(['--out', 'x.json', *argv])


# ---------------------------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------------------------

def find_bash():
    candidates = []
    if os.name == 'nt':
        for root in (os.environ.get('ProgramW6432'), os.environ.get('ProgramFiles'), 'C:/Program Files'):
            if root:
                candidates.append(Path(root) / 'Git' / 'bin' / 'bash.exe')
    found = shutil.which('bash')
    if found and not (os.name == 'nt' and ('system32' in found.lower() or 'windowsapps' in found.lower())):
        candidates.append(Path(found))
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)


BASH = find_bash()


@unittest.skipUnless(BASH, 'bash not found')
class RunnerTests(unittest.TestCase):
    def run_runner(self, directory, **env):
        base = {name: value for name, value in os.environ.items()
                if name not in ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'KOPGRAFT64', 'IMAGE', 'RESULTS', 'CARD_B_ARGS',
                                'EXPECT_TTNNCPP_SHA256', 'WATCHER', 'WATCHDOG_S', 'PAIR_ROW_DRY_RUN')}
        base.update(HOME=Path(directory).as_posix(), PAIR_ROW_DRY_RUN='1', MSYS_NO_PATHCONV='1')
        base.update(env)
        return subprocess.run([BASH, RUNNER.as_posix()], capture_output=True, text=True, timeout=120, env=base)

    def argv(self, result):
        lines = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')]
        self.assertEqual(len(lines), 1, result.stdout + result.stderr)
        return shlex.split(lines[0][len('### argv: '):])

    def graft(self, directory, *, verify=True):
        graft = Path(directory) / 'graft'
        for part in ('attn_prep', 'nlp_concat_heads_decode', 'sdpa_decode', 'sdpa'):
            (graft / part).mkdir(parents=True)
        (graft / '_ttnn.so').write_bytes(b'ttnn')
        (graft / '_ttnncpp.so').write_bytes(b'ttnncpp')
        manifest = ''.join('%s  %s%s' % (hashlib.sha256((graft / name).read_bytes()).hexdigest(), name, NL)
                           for name in ('_ttnn.so', '_ttnncpp.so'))
        (graft / 'MANIFEST.sha256').write_bytes((manifest if verify else manifest.replace('0', '1', 1)).encode())
        return graft

    def test_the_dry_run_launches_on_card_b_with_the_served_graft_and_the_checkouts_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_runner(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = self.argv(result)
        self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_B)
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-pairrow-card-b')
        mounts = [argv[index + 1] for index, word in enumerate(argv) if word == '--mount']
        for name in ('probe_pair_row_card_b.py', 'pair_row_exact.py', 'draft_attention.py', 'dflash_batched_mask.py',
                     'draft_shared_head.py'):
            self.assertEqual(len([mount for mount in mounts if mount.endswith('dst=/bench/%s,readonly' % name)]), 1, name)
        ops = '/opt/tt-metal/ttnn/cpp/ttnn/operations/'
        for target in ('/opt/tt-metal/ttnn/ttnn/_ttnn.so', '/opt/tt-metal/build_Release/ttnn/_ttnncpp.so',
                       '/opt/tt-metal/build_Release/lib/_ttnncpp.so', ops + 'transformer/attn_prep',
                       ops + 'experimental/transformer/nlp_concat_heads_decode', ops + 'transformer/sdpa_decode',
                       ops + 'transformer/sdpa'):
            self.assertEqual(len([mount for mount in mounts if ',dst=%s,' % target in mount]), 1, target)
        self.assertTrue(all('opgraft-K64i' in mount for mount in mounts if '/opt/tt-metal/' in mount))
        self.assertIn('TT_METAL_CACHE=/kcache', argv)
        self.assertIn('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', argv)
        self.assertIn('sha256:0fd9ad1f14a4e5d3d4464be55465cb6bb8d52e1b533c430d1c2f7f219df2b0e8', argv)
        self.assertEqual(argv[argv.index('probe') + 1:argv.index('probe') + 3], ['--out', argv[argv.index('--out') + 1]])
        self.assertIn('--watchdog', argv)
        self.assertIn('dry run: ', result.stdout)

    def test_the_watcher_pass_and_no_graft(self):
        with tempfile.TemporaryDirectory() as directory:
            watcher = self.argv(self.run_runner(directory, WATCHER='1'))
            bare = self.argv(self.run_runner(directory, KOPGRAFT64='none'))
        self.assertIn('TT_METAL_WATCHER=5', watcher)
        self.assertEqual(watcher[watcher.index('--head-dtypes') + 1], '')
        self.assertIn('--no-timing', watcher)
        self.assertEqual(watcher[watcher.index('--leading') + 1], '1,65')
        self.assertEqual([word for word in bare if word.startswith('type=bind') and ',dst=/opt/tt-metal/' in word], [])
        self.assertTrue([word for word in watcher if word.startswith('type=bind') and ',dst=/opt/tt-metal/' in word])

    def test_the_graft_is_checked_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            graft = self.graft(directory)
            ok = self.run_runner(directory, KOPGRAFT64=graft.as_posix())
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn('manifest verified', ok.stdout)
            expected = hashlib.sha256(b'ttnncpp').hexdigest()
            self.assertEqual(self.run_runner(directory, KOPGRAFT64=graft.as_posix(),
                                             EXPECT_TTNNCPP_SHA256=expected).returncode, 0)
            wrong = self.run_runner(directory, KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256='0' * 64)
            self.assertEqual(wrong.returncode, 1)
            self.assertIn('not 0000000000000000', wrong.stderr)
            (graft / 'sdpa').rmdir()
            missing = self.run_runner(directory, KOPGRAFT64=graft.as_posix())
            self.assertEqual(missing.returncode, 1)
            self.assertIn('sdpa missing', missing.stderr)
        with tempfile.TemporaryDirectory() as directory:
            bad = self.run_runner(directory, KOPGRAFT64=self.graft(directory, verify=False).as_posix())
            self.assertEqual(bad.returncode, 1)
            self.assertIn('does not verify', bad.stderr)

    def test_the_runner_parses_with_lf_endings(self):
        result = subprocess.run([BASH, '-n', RUNNER.as_posix()], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(b'\r', RUNNER.read_bytes())
        self.assertNotIn(b'\r', (HERE / 'probe_pair_row_card_b.py').read_bytes())


class RegistrationTests(unittest.TestCase):
    def test_the_qual_card_lists_and_the_cpu_suite_carry_it(self):
        text = (CI / 'test_qual_card.py').read_text(encoding='utf-8')
        self.assertGreaterEqual(text.count("OPS / 'pair_row_probe' / 'run_card_b.sh'"), 4)
        self.assertIn("'PAIR_ROW_DRY_RUN'", text)
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertIn("python -B -m unittest discover -s optimisation/ttnn-op/pair_row_probe -p 'test_*.py'", workflow)


if __name__ == '__main__':
    unittest.main()
