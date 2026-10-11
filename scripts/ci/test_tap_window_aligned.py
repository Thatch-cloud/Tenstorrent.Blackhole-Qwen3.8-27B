"""CPU tests of the tile-aligned drafter window (tap_window_aligned, QWEN_FAST_TAP_WINDOW_ALIGNED) on dflash_prefill_window's capture.

The fake ttnn records every slice, clone and concat with the rows it sees, so the tests can hold the claims the module makes:
the flag off issues exactly today's operations; the flag on hands the drafter the same bytes for every prompt length, every
operation before the final slice is tile-aligned, and the final slice sees one fixed input height at one of 31 offsets."""

import os
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import dflash_prefill_window
import tap_window_aligned as twa
from dflash_prefill_window import PrefillWindowCapture, validate_prefill_chunks

WIDTH = 2560
TAPS = (1, 3)


class Recorder:
    """A fake ttnn: torch tensors, every data-movement op logged as (op, rows in, begin, end) or (op, (rows of each input),)."""

    def __init__(self, broken_slice=None):
        self.log, self.freed = [], []
        self.bfloat16, self.DRAM_MEMORY_CONFIG = torch.bfloat16, 'dram'
        self.broken_slice = broken_slice

    def slice(self, value, start, end):
        self.log.append(('slice', value.shape[2], start[2], end[2]))
        out = value[..., start[2]:end[2], :]
        if self.broken_slice is not None and self.broken_slice(value, start, end):
            out = out.clone()
            out[..., 0, 0] += 1
        return out

    def clone(self, value, **kwargs):
        self.log.append(('clone', value.shape[2]))
        return value.clone()

    def concat(self, values, dim, **kwargs):
        self.log.append(('concat', tuple(value.shape[2] for value in values)))
        return torch.cat(values, dim=dim)

    def deallocate(self, value):
        self.freed.append(value)

    def get_device_tensors(self, value):
        return [value, value]

    def to_torch(self, value):
        return value


def model():
    layers = [SimpleNamespace(forward=lambda value: value) for _ in range(4)]
    result = SimpleNamespace(layers=layers)

    def chunk(token_buf, valid_len, chunk_start, page_table, bucket, **kwargs):
        for layer in result.layers:
            token_buf = layer.forward(token_buf)
        return token_buf
    result._forward_prefill_chunk_masked_tp = chunk
    return result


def host_rows(start, count):
    """The prompt's feature rows [start, start + count): distinct per row and per column, so a shifted or swapped row cannot compare
    equal (built per block, never for the whole prompt)."""
    rows = torch.arange(start, start + count, dtype=torch.float32).reshape(1, 1, -1, 1)
    cols = torch.arange(WIDTH, dtype=torch.float32).reshape(1, 1, 1, -1)
    return ((rows * 7 + cols) % 251 - 125).bfloat16()


def run_capture(position, chunk_rows=2048, environ=None, operations=None, start=0, checks=None):
    """A prefill of `position` tokens in chunks of `chunk_rows` (the last shorter), each chunk output padded to its 32-row bucket with
    -500 rows (the pad tokens' features, which must never reach the window). Returns (capture, outputs, operations, host)."""
    operations = operations or Recorder()
    environ = {} if environ is None else environ
    window = dflash_prefill_window.prefill_window(position)
    host = host_rows(window['start'], window['rows'])
    target = model()
    with patch.dict(os.environ, environ, clear=False), \
            patch('dflash_prefill_window.addresses',
                  side_effect=lambda operations, tensor: (tensor.data_ptr(),) * 2), \
            patch('dflash_prefill_window._log') as log:
        for name in (twa.FLAG, twa.AUDIT_FLAG):
            if name not in environ:
                os.environ.pop(name, None)
        capture = PrefillWindowCapture(operations, target, position, TAPS, checks=checks)
        with capture.capture():
            for chunk_start in range(start, position, chunk_rows):
                valid = min(chunk_rows, position - chunk_start)
                bucket = -(-valid // 32) * 32
                if chunk_start + valid <= window['start']:
                    # No row of the window: nothing is snapshotted from it.
                    target._forward_prefill_chunk_masked_tp(torch.empty((1, 1, bucket, WIDTH), dtype=torch.bfloat16), valid,
                                                            chunk_start, None, bucket)
                    continue
                value = torch.full((1, 1, bucket, WIDTH), -500, dtype=torch.bfloat16)
                value[..., :valid, :] = host_rows(chunk_start, valid)
                target._forward_prefill_chunk_masked_tp(value, valid, chunk_start, None, bucket)
                value.zero_()
        validate_prefill_chunks(position, capture.chunks)
        outputs = capture.outputs()
        capture.log_calls = list(log.call_args_list)
    return capture, outputs, operations, host


def window_of(host, position):
    """`host` holds exactly the window rows (run_capture)."""
    return host


def same(left, right):
    return left.shape == right.shape and torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16))


def logged(capture, prefix):
    return [call.args[0].format(*call.args[1:]) for call in capture.log_calls if str(call.args[0]).startswith(prefix)]


POSITIONS = (170, 2047, 2048, 2049, 4093, 4096, 4097, 6144, 6150, 33834, 35377, 65504, 2080, 4128)


class FlagTests(unittest.TestCase):
    def test_flag_names_are_pinned(self):
        self.assertEqual(dflash_prefill_window.ALIGNED_FLAG, twa.FLAG)
        self.assertEqual(twa.FLAG, 'QWEN_FAST_TAP_WINDOW_ALIGNED')
        self.assertEqual(twa.AUDIT_FLAG, 'QWEN_FAST_TAP_WINDOW_ALIGNED_AUDIT')

    def test_strict_values(self):
        self.assertFalse(twa.enabled({}))
        self.assertFalse(twa.enabled({twa.FLAG: '0'}))
        self.assertTrue(twa.enabled({twa.FLAG: '1'}))
        for bad in ('2', 'true', ' 1', ''):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                twa.enabled({twa.FLAG: bad})
        self.assertTrue(twa.audit_enabled({twa.FLAG: '1', twa.AUDIT_FLAG: '1'}))
        with self.assertRaises(ValueError):
            twa.audit_enabled({twa.AUDIT_FLAG: '1'})
        with self.assertRaises(ValueError):
            twa.audit_enabled({twa.FLAG: '1', twa.AUDIT_FLAG: 'yes'})

    def test_capture_refuses_a_bad_flag_value(self):
        with self.assertRaises(ValueError):
            run_capture(4093, environ={twa.FLAG: 'on'})


class OffPathTests(unittest.TestCase):
    def test_off_issues_exactly_todays_operations(self):
        """Flag unset (and '0'): the snapshots slice the exact pieces and the join is today's concat of them, nothing else."""
        for environ in ({}, {twa.FLAG: '0'}):
            for position in POSITIONS:
                with self.subTest(position=position, environ=environ):
                    capture, outputs, operations, host = run_capture(position, environ=environ)
                    self.assertFalse(capture.aligned)
                    window = dflash_prefill_window.prefill_window(position)
                    expected = []
                    pieces = []
                    for chunk_start in range(0, position, 2048):
                        valid = min(2048, position - chunk_start)
                        start, end = max(window['start'], chunk_start), min(position, chunk_start + valid)
                        if end > start:
                            pieces.append(end - start)
                            for _ in TAPS:
                                expected += [('slice', -(-valid // 32) * 32, start - chunk_start, end - chunk_start),
                                             ('clone', end - start)]
                    if len(pieces) > 1:
                        expected += [('concat', tuple(pieces))] * len(TAPS)
                    self.assertEqual(operations.log, expected)
                    for output in outputs:
                        self.assertTrue(same(output, window_of(host, position)))
                    self.assertEqual(logged(capture, twa.MARKER), [])

    def test_off_never_imports_the_module_path(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(twa.FLAG, None)
            with patch('tap_window_aligned.wanted', side_effect=AssertionError('imported under the flag off')), \
                    patch('tap_window_aligned.aligned_bounds', side_effect=AssertionError('imported under the flag off')):
                run_capture(35377)


class AlignedPathTests(unittest.TestCase):
    ON = {twa.FLAG: '1'}

    def test_same_bytes_for_every_prompt_length(self):
        positions = list(POSITIONS) + random.Random(7).sample(range(2049, 65504), 24)
        for position in positions:
            with self.subTest(position=position):
                capture, outputs, operations, host = run_capture(position, environ=self.ON)
                self.assertTrue(capture.aligned)
                self.assertEqual(len(outputs), len(TAPS))
                for output in outputs:
                    self.assertEqual(output.shape[2], min(position, 2048))
                    self.assertTrue(same(output, window_of(host, position)))
                    self.assertFalse(bool((output == -500).any()), 'a pad row reached the window')

    def test_engages_only_for_an_unaligned_window_across_chunks(self):
        for position in POSITIONS:
            with self.subTest(position=position):
                capture, _, _, _ = run_capture(position, environ=self.ON)
                window = dflash_prefill_window.prefill_window(position)
                across = position > 2048 and position % 2048 != 0
                unaligned = window['start'] % 32 != 0 or window['end'] % 32 != 0
                engaged = logged(capture, twa.ENGAGED_MARKER)
                self.assertEqual(bool(engaged), across and unaligned, engaged)
                self.assertEqual(logged(capture, twa.FELL_BACK_MARKER), [])
                if engaged:
                    self.assertIn('lead=%d union=2080 rows=2048' % (window['start'] % 32), engaged[0])

    def test_shapes_are_bounded(self):
        """Every slice and concat before the final cut is tile-aligned; the only unaligned operation is the final slice, of a 2080-row
        input at one of 31 offsets: the set of programs a new prompt length can need is fixed."""
        unaligned = set()
        aligned_shapes = set()
        for position in range(2049, 2049 + 4096, 29):
            _, _, operations, _ = run_capture(position, environ=self.ON)
            for entry in operations.log:
                if entry[0] == 'slice':
                    _, rows, begin, end = entry
                    if begin % 32 or end % 32:
                        unaligned.add(entry)
                    else:
                        aligned_shapes.add(entry)
                elif entry[0] == 'concat':
                    window = dflash_prefill_window.prefill_window(position)
                    if window['start'] % 32:
                        self.assertTrue(all(rows % 32 == 0 for rows in entry[1]), entry)
                        aligned_shapes.add(entry)
        self.assertTrue(unaligned)
        for _, rows, begin, end in unaligned:
            self.assertEqual(rows, 2080)
            self.assertTrue(1 <= begin <= 31)
            self.assertEqual(end - begin, 2048)
        self.assertLessEqual(len(unaligned), 31)
        # Tile-aligned slices of a 2048-row bucket (64 begins, 64 ends) and their joins: bounded whatever the prompt.
        self.assertLessEqual(len(aligned_shapes), 64 * 4)

    def test_todays_path_compiles_a_new_shape_per_split(self):
        """The control's claim, on the same fake: today's pieces are unaligned and their heights follow the prompt length."""
        heights = set()
        for position in range(2049, 2049 + 2048, 17):
            _, _, operations, _ = run_capture(position)
            heights.update(entry[1] for entry in operations.log if entry[0] == 'concat')
        self.assertGreater(len(heights), 100)

    def test_short_chunks_fall_back_to_the_exact_join(self):
        """Chunks of 1000 rows start off tile boundaries: the aligned pieces are not contiguous, so every piece is cut exact and
        joined as today, and the window is still exact."""
        for position in (3000, 4093, 5999):
            with self.subTest(position=position):
                capture, outputs, operations, host = run_capture(position, chunk_rows=1000, environ=self.ON)
                for output in outputs:
                    self.assertTrue(same(output, window_of(host, position)))
                fell = logged(capture, twa.FELL_BACK_MARKER)
                engaged = logged(capture, twa.ENGAGED_MARKER)
                self.assertEqual(len(fell) + len(engaged), 1 if (fell or engaged) else 0)

    def test_checks_disable_the_aligned_snapshot(self):
        checks = []
        capture, outputs, operations, host = run_capture(35377, environ=self.ON, checks=checks)
        self.assertFalse(capture.aligned)
        self.assertTrue(checks)
        self.assertEqual(logged(capture, twa.MARKER), [])
        for output in outputs:
            self.assertTrue(same(output, window_of(host, 35377)))

    def test_close_releases_every_window_and_snapshot_exactly_once(self):
        for position in (4093, 35377, 2047, 6144, 3000):
            with self.subTest(position=position):
                capture, outputs, operations, _ = run_capture(position, environ=self.ON,
                                                              **({'chunk_rows': 1000} if position == 3000 else {}))
                children = [value for child in capture.children for value in child.records.values()]
                before = len(operations.freed)
                capture.close()
                freed = operations.freed[before:]
                ids = [id(value) for value in freed]
                self.assertEqual(len(ids), len(set(ids)), 'a tensor was released twice')
                for value in list(outputs) + children:
                    self.assertTrue(any(value is other for other in freed), 'a window or snapshot outlived its capture')


class AuditTests(unittest.TestCase):
    ON = {twa.FLAG: '1', twa.AUDIT_FLAG: '1'}

    def test_audit_passes_and_serves_the_aligned_window(self):
        capture, outputs, operations, host = run_capture(35377, environ=self.ON)
        audits = logged(capture, twa.AUDIT_MARKER)
        self.assertEqual(len(audits), len(TAPS))
        self.assertTrue(all('exact=True' in line for line in audits), audits)
        self.assertEqual(logged(capture, twa.FELL_BACK_MARKER), [])
        for output in outputs:
            self.assertTrue(same(output, window_of(host, 35377)))

    def test_a_broken_aligned_slice_is_caught_and_the_served_window_used(self):
        broken = Recorder(broken_slice=lambda value, start, end: value.shape[2] == 2080)
        capture, outputs, operations, host = run_capture(35377, environ=self.ON, operations=broken)
        audits = logged(capture, twa.AUDIT_MARKER)
        self.assertTrue(audits and all('exact=False' in line for line in audits), audits)
        self.assertEqual(len(logged(capture, twa.FELL_BACK_MARKER)), 1)
        self.assertIn('reason=audit_mismatch', logged(capture, twa.FELL_BACK_MARKER)[0])
        for output in outputs:
            self.assertTrue(same(output, window_of(host, 35377)))

    def test_audit_without_flag_is_refused(self):
        with self.assertRaises(ValueError):
            run_capture(35377, environ={twa.FLAG: '1', twa.AUDIT_FLAG: '2'})


class HelperTests(unittest.TestCase):
    def test_aligned_bounds(self):
        window = dflash_prefill_window.prefill_window(33834)   # [31786, 33834)
        self.assertEqual(twa.aligned_bounds(window, 30720, 2048, 2048), (1056, 2048))
        self.assertEqual(twa.aligned_bounds(window, 32768, 1066, 1088), (0, 1088))
        self.assertIsNone(twa.aligned_bounds(window, 32768, 1066, 1066))      # the bucket rows are not there
        self.assertIsNone(twa.aligned_bounds(window, 32760, 1074, 1088))      # off a tile boundary
        self.assertIsNone(twa.aligned_bounds(window, 28672, 2048, 2048))      # no row of the window

    def test_plan_needs_contiguous_aligned_parts(self):
        window = dflash_prefill_window.prefill_window(33834)
        cover = twa.covers(window, [(30720, 2048), (32768, 1066)], [992, 1088])
        self.assertEqual(cover, [dict(begin=31776, end=32768, start=31786, stop=32768, aligned=True),
                                 dict(begin=32768, end=33856, start=32768, stop=33834, aligned=True)])
        self.assertTrue(twa.expanded(cover))
        self.assertEqual(twa.plan(window, cover), dict(lead=10, union=2080))
        exact = twa.covers(window, [(30720, 2048), (32768, 1066)], [982, 1066])
        self.assertIsNone(twa.plan(window, exact))
        with self.assertRaises(ValueError):
            twa.covers(window, [(30720, 2048), (32768, 1066)], [1000, 1066])


class ReaderTests(unittest.TestCase):
    """rebind_split.py, the pack's READ, on a synthetic log in the container's line format."""

    @staticmethod
    def line(seconds, text):
        whole = int(seconds)
        return '2026-10-10T20:45:%02d.%06dZ (EngineCore pid=67) 2026-10-10 20:45:%02d.000 | INFO | x:y:1 - %s' % (
            whole, round((seconds - whole) * 1e6), whole, text)

    def lines(self, req, t0, b_ms, programs, window):
        short = req[-12:]
        t = t0
        out = [self.line(t, '[PINDIAG] lever N route req=%s start=32768 end=35377 prompt=35377 final=1 wrote_slot=1 ms=900.0 '
                            'programs=1->1 window=%d source=SCRATCH captured=-' % (req, window))]
        t += 0.12
        began = t
        t += 0.075
        out.append(self.line(t, '[PINDIAG] prefill slot 0, nothing to adopt for request %s' % req))
        t += b_ms / 1000.0
        out.append(self.line(t, '[MEMLEDGER] before op=rebind point=req=%s slot=0 chip0 largest_free=1MB' % short))
        t += 0.13
        out.append(self.line(t, '[PINDIAG] parked rebind req=%s slot=0 gen=2 P=35377 budget=128 ms=128.0 single_rebuilt=0' % req))
        t += 0.068
        out.append(self.line(t, '[MEMLEDGER] after op=rebind point=req=%s slot=0 chip0 cost=0.0MB' % short))
        t += 0.001
        out.append(self.line(t, '[PINDIAG] sticky engine built req=%s ms=%.1f frontier=32768 prompt=35377 kind=rebind' % (req, (t - began) * 1000)))
        out.append(self.line(t + 0.066, '[PINDIAG] four-card engine programs=10->%d req=%s' % (10 + programs, req)))
        return out

    def test_split_and_modes(self):
        import io
        import rebind_split

        text = self.lines('chatcmpl-a-hit-00000001', 1.0, 650.0, 3, 7) + self.lines('chatcmpl-b-cold-0000002', 10.0, 70.0, 0, 0)
        result = rebind_split.read(text)
        self.assertEqual(len(result['rebinds']), 2)
        slow, fast = result['rebinds']
        self.assertTrue(slow['slow'])
        self.assertFalse(fast['slow'])
        self.assertAlmostEqual(slow['segments'][1], 650.0, delta=1.0)
        self.assertAlmostEqual(sum(slow['segments']), slow['ms'], delta=1.0)
        self.assertEqual((slow['programs'], fast['programs']), (3, 0))
        self.assertEqual(rebind_split.route_pairs(result['routes']), [(0.0, 7)])
        out = io.StringIO()
        rebind_split.report(result, out)
        self.assertIn('programs +3 SLOW', out.getvalue())
        self.assertIn('programs +0 fast', out.getvalue())


if __name__ == '__main__':
    unittest.main()
