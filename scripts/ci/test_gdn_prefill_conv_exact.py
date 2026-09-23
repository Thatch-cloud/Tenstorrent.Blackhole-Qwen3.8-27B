"""CPU tests for gdn_prefill_conv_exact (GDN prefill conv lever #2): the planners, the copy tables,
the descriptor cache and its per-call arg rewrites, and the kernels' agreement with the Python
tables. No device: the hardware half is optimisation/ttnn-op/gdn_prefill_conv.

The bit emulator here is NOT a hardware oracle. It runs the FIR's arithmetic (RNE after the tap-0
product, fp32 MAD then RNE per addcmul) once the direct way (concat, slices) and once through the
planner's own unit split, ring and face-row copy tables, so a wrong index anywhere shows up as a
byte difference. SiLU and the SFPU's denormal / tie behaviour are settled on card M only.
"""

import re
import tempfile
import unittest
from pathlib import Path

import torch

import gdn_prefill_conv_exact as pcx

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------------------------
# Tile byte layout (the lane() addressing of draft_convolution_fused_io.cpp).
# ---------------------------------------------------------------------------------------------

def lane(row, column):
    return (row // 16) * 512 + (column // 16) * 256 + (row % 16) * 16 + column % 16


LANES = torch.tensor([[lane(r, c) for c in range(32)] for r in range(32)])


def to_tile(block):
    """32x32 int16 matrix -> 1024 int16 in face order."""
    flat = torch.zeros(1024, dtype=torch.int16)
    flat[LANES.reshape(-1)] = block.reshape(-1)
    return flat


def from_tile(flat):
    return flat[LANES.reshape(-1)].reshape(32, 32)


def apply_copies(copies, sources, destination):
    """Byte offsets / lengths are multiples of 32 bytes = 16 int16 lanes."""
    for source, offset, length, target in copies:
        assert offset % 32 == 0 and length % 32 == 0 and target % 32 == 0
        destination[target // 2:(target + length) // 2] = sources[source][offset // 2:(offset + length) // 2]
    return destination


def bf16_bits(values):
    return values.to(torch.bfloat16).view(torch.int16)


def rne_bf16(values):
    """float32 -> bf16 with round-to-nearest-even (torch's own conversion is RNE)."""
    return values.to(torch.float32).to(torch.bfloat16)


class PlanTests(unittest.TestCase):
    def test_served_geometry(self):
        layout = pcx.plan(2048, 5120, 110)
        self.assertEqual((layout['Ht'], layout['Ct'], layout['R'], layout['strips'], layout['units']), (64, 160, 32, 2, 320))
        counts = [count for _, count in layout['ranges']]
        self.assertEqual(counts.count(3), 100)
        self.assertEqual(counts.count(2), 10)
        self.assertEqual(max(counts) * layout['R'], 96)

    def test_every_tile_is_covered_exactly_once(self):
        for T, C, cores in ((2048, 5120, 110), (32, 5120, 110), (64, 5120, 110), (128, 5120, 80),
                            (4096, 5120, 110), (2048, 5120, 1), (96, 160, 7), (32, 32, 110)):
            with self.subTest(T=T, C=C, cores=cores):
                layout = pcx.plan(T, C, cores)
                seen = {}
                for start, count in layout['ranges']:
                    for unit in range(start, start + count):
                        ct, rows = pcx.unit_tiles(unit, layout['strips'], layout['R'])
                        for ht in rows:
                            key = (ht, ct)
                            self.assertNotIn(key, seen)
                            seen[key] = unit
                self.assertEqual(len(seen), (T // 32) * (C // 32))
                self.assertEqual(sum(count for _, count in layout['ranges']), layout['units'])
                starts = [start for start, _ in layout['ranges']]
                self.assertEqual(starts, sorted(starts))

    def test_r_minimises_the_busiest_cores_tile_and_halo_reads(self):
        for T in (32, 64, 256, 2048, 4096):
            layout = pcx.plan(T, 5120, 110)
            best = min(pcx.ceil_div(160 * (T // 32 // r), 110) * (r + 1) for r in pcx.divisors(T // 32))
            self.assertEqual(max(count for _, count in layout['ranges']) * (layout['R'] + 1), best)

    def test_refuses_ragged_geometry(self):
        for T, C in ((33, 5120), (2048, 5000), (0, 5120)):
            with self.assertRaises(pcx.Unsupported):
                pcx.plan(T, C, 110)


class ShiftCopyTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(7)
        self.x = torch.randint(-30000, 30000, (96, 32), dtype=torch.int16, generator=generator)
        self.carry = torch.randint(-30000, 30000, (3, 32), dtype=torch.int16, generator=generator)

    def padded(self, carry):
        return torch.cat([carry, self.x], dim=0)   # x_padded row i+3 = x[i]

    def shifted(self, s, ht, halo):
        cur = to_tile(self.x[32 * ht:32 * ht + 32])
        if halo == 'x':
            prev = to_tile(self.x[32 * ht - 32:32 * ht])
        else:
            block = torch.zeros(32, 32, dtype=torch.int16)
            if halo == 'carry':
                block[:3] = self.carry
            prev = to_tile(block)
        out = torch.full((1024,), 0x5555, dtype=torch.int16)
        apply_copies(pcx.shift_copies(s, halo == 'x'), dict(cur=cur, prev=prev), out)
        return from_tile(out)

    def test_eight_copies_per_shift_all_32_byte_aligned(self):
        for s in (1, 2, 3):
            for prev_is_x in (True, False):
                copies = pcx.shift_copies(s, prev_is_x)
                self.assertEqual(len(copies), 8)
                self.assertTrue(all(value % 32 == 0 for copy in copies for value in copy[1:]))
                self.assertEqual(sum(copy[2] for copy in copies), 2048)

    def test_matches_a_naive_row_shift_of_the_concatenation(self):
        for s in (1, 2, 3):
            for ht in (1, 2):
                with self.subTest(s=s, ht=ht):
                    expected = self.padded(self.carry)[32 * ht + 3 - s:32 * ht + 35 - s]
                    self.assertTrue(torch.equal(self.shifted(s, ht, 'x'), expected))

    def test_the_carry_halo_and_the_zero_halo(self):
        for s in (1, 2, 3):
            with self.subTest(s=s):
                self.assertTrue(torch.equal(self.shifted(s, 0, 'carry'), self.padded(self.carry)[3 - s:35 - s]))
                zeros = torch.zeros(3, 32, dtype=torch.int16)
                self.assertTrue(torch.equal(self.shifted(s, 0, 'zero'), self.padded(zeros)[3 - s:35 - s]))

    def test_refuses_other_shifts(self):
        for s in (0, 4):
            with self.assertRaises(ValueError):
                pcx.shift_copies(s, True)


class StateRowTests(unittest.TestCase):
    VALID_LENS = (1, 2, 3, 31, 32, 33, 34, 264, 520, 1288, 2047, 2048)

    def test_state_rows_select_x_padded_vl_plus_j(self):
        T = 2048
        x = torch.arange(T)
        carry = torch.tensor([-3, -2, -1])
        padded = torch.cat([carry, x])
        for vl in self.VALID_LENS:
            with self.subTest(valid_len=vl):
                ht_v = pcx.state_tile_row(vl)
                self.assertEqual(ht_v, (vl - 1) // 32)
                got = []
                for where, row in pcx.state_rows(vl):
                    self.assertTrue(0 <= row < 32)
                    if where == 'cur':
                        got.append(int(x[32 * ht_v + row]))
                    elif where == 'prev':
                        self.assertGreater(ht_v, 0)
                        got.append(int(x[32 * (ht_v - 1) + row]))
                    else:
                        self.assertLess(row, 3)
                        got.append(int(carry[row]))
                self.assertEqual(got, [int(padded[vl + j]) for j in range(3)])

    def test_refuses_zero(self):
        with self.assertRaises(ValueError):
            pcx.state_rows(0)


class RouteTests(unittest.TestCase):
    def test_q_k_v_pages(self):
        q_t, c_t = 32, 160
        self.assertEqual(pcx.output_page(0, 0, q_t, c_t), ('q', 0))
        self.assertEqual(pcx.output_page(31, 5, q_t, c_t), ('q', 5 * 32 + 31))
        self.assertEqual(pcx.output_page(32, 5, q_t, c_t), ('k', 5 * 32))
        self.assertEqual(pcx.output_page(64, 5, q_t, c_t), ('v', 5 * 96))
        self.assertEqual(pcx.output_page(159, 63, q_t, c_t), ('v', 63 * 96 + 95))
        pages = {}
        for ct in range(c_t):
            for ht in range(4):
                which, page = pcx.output_page(ct, ht, q_t, c_t)
                pages.setdefault(which, set()).add(page)
        self.assertEqual({k: len(v) for k, v in pages.items()}, {'q': 128, 'k': 128, 'v': 384})
        self.assertEqual(pages['v'], set(range(384)))


# ---------------------------------------------------------------------------------------------
# Bit emulator: the FIR direct, and the same arithmetic through the planner's tables.
# ---------------------------------------------------------------------------------------------

def fir_chain(x_taps, taps):
    """x_taps: four [T, C] bf16 (x shifted by 3, 2, 1, 0); taps [4, C] bf16 -> pre-SiLU bf16."""
    x0 = x_taps[0].float()
    w0 = taps[0].float()
    acc = rne_bf16(x0 * w0)
    acc = torch.where((x0 == 0) | (w0 == 0), torch.zeros_like(acc), acc)
    for k in (1, 2, 3):
        acc = rne_bf16((x_taps[k].double() * taps[k].double() + acc.double()).float())
    return acc


def silu_bf16(pre):
    return rne_bf16(torch.nn.functional.silu(pre.float()))


def canonical(state_bits, denorm=False):
    out = state_bits.clone()
    out[out == torch.tensor(-32768, dtype=torch.int16)] = 0
    if denorm:
        out[(out.int() & 0x7F80) == 0] = 0
    return out


def flush_x(bits, mode):
    """The reader's flush_half over int16 bf16 bits: a denormal (exponent 0, mantissa != 0) -> +0
    (mode 1) or a zero of its own sign (mode 2); mode 0 is the identity."""
    if not mode:
        return bits
    word = bits.int() & 0xFFFF
    denormal = ((word & 0x7F80) == 0) & ((word & 0x7F) != 0)
    flushed = (word & 0x8000) if mode == 2 else torch.zeros_like(word)
    out = torch.where(denormal, flushed, word)
    return torch.where(out >= 0x8000, out - 0x10000, out).to(torch.int16)


def direct(x, carry, taps, valid_len, flush=0):
    """The served composition: concat, three slices, the chain, SiLU, the q/k/v split, the state.
    flush models a round trip of x_padded that flushes denormals (PCX_FLUSH_X_DENORM)."""
    T, C = x.shape
    padded = torch.cat([carry if carry is not None else torch.zeros(3, C, dtype=torch.bfloat16), x], dim=0)
    padded = flush_x(padded.view(torch.int16), flush).view(torch.bfloat16)
    pre = fir_chain([padded[k:k + T] for k in range(4)], taps)
    conv = silu_bf16(pre)
    vl = T if valid_len is None else valid_len
    state = padded[vl:vl + 3].view(torch.int16)
    if valid_len is not None:
        state = canonical(state)
    return conv.view(torch.int16), state


def through_the_planner(x, carry, taps, valid_len, cores=110, flush=0):
    """The reader's unit / ring / face-row copies and state build, tile by tile. flush applies the
    reader's PCX_FLUSH_X_DENORM to each x / carry tile as it lands (before any copy)."""
    T, C = x.shape
    layout = pcx.plan(T, C, cores)
    vl = T if valid_len is None else valid_len
    ht_v = pcx.state_tile_row(vl)
    bits = x.view(torch.int16)
    tiles = {(ht, ct): to_tile(bits[32 * ht:32 * ht + 32, 32 * ct:32 * ct + 32])
             for ht in range(layout['Ht']) for ct in range(layout['Ct'])}
    zero = torch.zeros(1024, dtype=torch.int16)
    out = torch.full((T, C), 0x7FC1, dtype=torch.int16)
    state = torch.full((3, C), 0x7FC1, dtype=torch.int16)
    written = set()
    for start, count in layout['ranges']:
        for unit in range(start, start + count):
            ct, rows = pcx.unit_tiles(unit, layout['strips'], layout['R'])
            if carry is not None:
                block = torch.zeros(32, 32, dtype=torch.int16)
                block[:3] = carry.view(torch.int16)[:, 32 * ct:32 * ct + 32]
                carry_tile = flush_x(to_tile(block), flush)
            else:
                carry_tile = zero
            for ht in rows:
                cur = flush_x(tiles[(ht, ct)], flush)
                prev, prev_is_x = (flush_x(tiles[(ht - 1, ct)], flush), True) if ht else (carry_tile, False)
                shifted = []
                for s in (3, 2, 1):
                    tile = torch.full((1024,), 0x5555, dtype=torch.int16)
                    apply_copies(pcx.shift_copies(s, prev_is_x), dict(cur=cur, prev=prev), tile)
                    shifted.append(from_tile(tile).view(torch.bfloat16))
                shifted.append(from_tile(cur).view(torch.bfloat16))
                w = taps[:, 32 * ct:32 * ct + 32].unsqueeze(1).expand(4, 32, 32)
                pre = fir_chain(shifted, w)
                tile_out = silu_bf16(pre).view(torch.int16)
                assert (ht, ct) not in written
                written.add((ht, ct))
                out[32 * ht:32 * ht + 32, 32 * ct:32 * ct + 32] = tile_out
                if ht == ht_v:
                    for j, (where, row) in enumerate(pcx.state_rows(vl)):
                        source = dict(cur=cur, prev=prev, carry=carry_tile)[where]
                        state[j, 32 * ct:32 * ct + 32] = from_tile(source)[row]
    assert len(written) == layout['Ht'] * layout['Ct']
    if valid_len is not None:
        state = canonical(state)
    return out, state


class EmulatorTests(unittest.TestCase):
    def cases(self):
        generator = torch.Generator().manual_seed(11)
        T, C = 96, 320
        x = torch.randn(T, C, generator=generator).to(torch.bfloat16)
        x[5, 7] = -0.0
        x[40, 3] = 0.0
        carry = torch.randn(3, C, generator=generator).to(torch.bfloat16)
        carry[1, 9] = -0.0
        taps = torch.randn(4, C, generator=generator).to(torch.bfloat16)
        taps[0, 11] = 0.0
        return x, carry, taps

    def test_the_planner_path_is_byte_identical_to_the_direct_composition(self):
        x, carry, taps = self.cases()
        for use_carry in (True, False):
            for vl in (None, 1, 2, 3, 31, 32, 33, 34, 64, 95, 96):
                with self.subTest(carry=use_carry, valid_len=vl):
                    c = carry if use_carry else None
                    conv, state = direct(x, c, taps, vl)
                    got_conv, got_state = through_the_planner(x, c, taps, vl, cores=7)
                    self.assertTrue(torch.equal(conv, got_conv))
                    self.assertTrue(torch.equal(state, got_state))

    def test_the_minus_zero_carry_row_is_canonical_only_on_the_one_hot_path(self):
        x, carry, taps = self.cases()
        _, sliced = through_the_planner(x[:32], carry, taps, None, cores=3)
        _, one_hot = through_the_planner(x[:32], carry, taps, 32, cores=3)
        self.assertTrue(torch.equal(sliced, x[29:32].view(torch.int16)))
        # vl=2: rows x_padded[2..4] = carry[2], x[0], x[1]; carry row 1 (-0) is only selected at vl=1.
        _, early = through_the_planner(x[:32], carry, taps, 1, cores=3)
        self.assertEqual(int(early[0, 9]), 0)
        self.assertEqual(int(carry.view(torch.int16)[1, 9]), -32768)

    def test_the_x_flush_through_the_planner_is_the_flushed_composition(self):
        """PCX_FLUSH_X_DENORM: flushing each loaded tile once equals flushing x_padded up front, on
        both state paths, and the flush is visible (so the card-M selection can tell it apart)."""
        x, carry, taps = self.cases()
        tiny = torch.tensor([3e-39, -3e-39, 1e-40], dtype=torch.bfloat16)
        x[0, 0], x[33, 5], x[95, 319] = tiny[0], tiny[1], tiny[2]
        x[64, 1:5] = tiny[1]                    # one row of four denormals feeds every tap of a column
        carry[2, 4], carry[0, 6] = tiny[1], tiny[0]
        bits = x.view(torch.int16).int() & 0xFFFF
        self.assertGreater(int((((bits & 0x7F80) == 0) & ((bits & 0x7F) != 0)).sum()), 3)
        for mode in (0, 1, 2):
            for vl in (None, 1, 2, 34, 96):
                with self.subTest(mode=mode, valid_len=vl):
                    conv, state = direct(x, carry, taps, vl, flush=mode)
                    got_conv, got_state = through_the_planner(x, carry, taps, vl, cores=7, flush=mode)
                    self.assertTrue(torch.equal(conv, got_conv))
                    self.assertTrue(torch.equal(state, got_state))
        _, raw = direct(x, carry, taps, None)
        _, plus = direct(x, carry, taps, None, flush=1)
        _, signed = direct(x, carry, taps, None, flush=2)
        self.assertFalse(torch.equal(raw, plus))            # x[95, 319] is in the static-slice state
        self.assertEqual(int(plus[2, 319]), 0)
        _, early = direct(x, carry, taps, 1, flush=2)       # rows carry[1], carry[2], x[0]
        self.assertEqual(int(early[1, 4]), 0)               # the one-hot path's canon: -0 -> +0
        self.assertEqual(int(signed[2, 319]), 0)            # +1e-40 -> +0
        self.assertEqual(int(flush_x(torch.tensor([-32767], dtype=torch.int16), 2)[0]), -32768)  # 0x8001 -> -0
        # exact +-0 and the smallest normal (0x0080) are never touched
        self.assertEqual(flush_x(torch.tensor([-32768, 0, 0x80], dtype=torch.int16), 1).tolist(), [-32768, 0, 0x80])

    def test_a_wrong_shift_is_caught(self):
        """The emulator's own negative control: swapping two shift tables changes bytes."""
        x, carry, taps = self.cases()
        original = pcx.shift_copies

        def swapped(s, prev_is_x):
            return original({1: 2, 2: 1, 3: 3}[s], prev_is_x)

        conv, _ = direct(x, carry, taps, None)
        pcx.shift_copies = swapped
        try:
            got, _ = through_the_planner(x, carry, taps, None, cores=7)
        finally:
            pcx.shift_copies = original
        self.assertFalse(torch.equal(conv, got))


# ---------------------------------------------------------------------------------------------
# Kernels vs the Python tables.
# ---------------------------------------------------------------------------------------------

def kernel(name):
    return (HERE / name).read_text(encoding='utf-8')


class KernelAgreementTests(unittest.TestCase):
    def test_runtime_files_sit_next_to_the_module(self):
        self.assertEqual(pcx.RUNTIME_FILES[0], 'gdn_prefill_conv_exact.py')
        for name in pcx.RUNTIME_FILES:
            with self.subTest(file=name):
                self.assertTrue((HERE / name).is_file())
                self.assertNotIn(chr(13), (HERE / name).read_text(encoding='utf-8'), 'CRLF in %s' % name)

    def test_cb_indices_match(self):
        expected = dict(cb_x3=pcx.CB_X3, cb_x2=pcx.CB_X2, cb_x1=pcx.CB_X1, cb_x0=pcx.CB_X0, cb_taps=pcx.CB_TAPS,
                        cb_bcast=pcx.CB_BCAST, cb_out=pcx.CB_OUT, cb_ring=pcx.CB_RING, cb_state=pcx.CB_STATE,
                        cb_acc=pcx.CB_ACC, cb_halo=pcx.CB_HALO)
        for name in pcx.KERNEL_FILES:
            found = dict((key, int(value)) for key, value in
                         re.findall(r'constexpr uint32_t (cb_\w+) = (\d+);', kernel(name)))
            self.assertTrue(found, name)
            for key, value in found.items():
                with self.subTest(kernel=name, cb=key):
                    self.assertEqual(value, expected[key])
            self.assertIn('namespace pcx', kernel(name))

    def test_common_arg_indices_match_the_layout(self):
        for name in (pcx.READER, pcx.WRITER):
            text = kernel(name)
            for tensor, index in re.findall(r'const auto (\w+) = TensorAccessor\(\w+_args, get_common_arg_val<uint32_t>\((\d+)\)', text):
                with self.subTest(kernel=name, tensor=tensor):
                    self.assertEqual(pcx.COMMON_ARGS.index(tensor), int(index))
            for scalar, index in re.findall(r'const uint32_t (valid_len|canon) = get_common_arg_val<uint32_t>\((\d+)\)', text):
                self.assertEqual(pcx.COMMON_ARGS.index(scalar), int(index))
            accessors = re.findall(r'constexpr auto (\w+)_args = TensorAccessorArgs<', text)
            self.assertEqual(accessors, list(pcx.ACCESSORS[:len(accessors)]))
            self.assertIn('TensorAccessorArgs<%d>()' % len(pcx.COMPILE_ARGS), text)

    def test_compile_arg_indices_match(self):
        names = {index: name for index, name in enumerate(pcx.COMPILE_ARGS)}
        mapping = dict(Ct='Ct', R='R', strips='strips', q_tiles='q_tiles')
        for name in pcx.KERNEL_FILES:
            for variable, index in re.findall(r'constexpr (?:uint32_t|bool) (\w+) = get_compile_time_arg_val\((\d+)\)', kernel(name)):
                with self.subTest(kernel=name, arg=variable):
                    self.assertEqual(names[int(index)], mapping.get(variable, variable))
        self.assertIn('get_compile_time_arg_val(5) != 0', kernel(pcx.READER))

    def test_compute_calls_the_served_llk_sequence(self):
        text = kernel(pcx.COMPUTE)
        body = text[text.index('void kernel_main'):]
        order = ['mul_binary_tile(0, 1, 0)', 'addcmul_tile<PCX_ADDCMUL_FORMAT>(0, 1, 2, 0, one_f32)', 'silu_tile(0)']
        positions = [body.index(call) for call in order]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual([body.count(call) for call in order], [1, 1, 1])
        self.assertIn('constexpr uint32_t one_f32 = 0x%08Xu;' % pcx.ONE_F32_BITS, text)
        self.assertIn('#define PCX_ADDCMUL_FORMAT DataFormat::Float16_b', text)
        self.assertIn('unary_bcast<BroadcastType::ROW>(cb_taps, j, 0)', text)
        self.assertNotIn('fp32_dest_acc_en', body)

    def test_the_reader_shift_matches_shift_copies(self):
        """shift_tile in the reader is the literal of shift_copies: same four copies per half."""
        text = kernel(pcx.READER)
        body = text[text.index('FORCE_INLINE void shift_tile'):text.index('FORCE_INLINE void zero_tile')]
        self.assertIn('prev_is_x ? lower + row * (16 - s) : upper + row * (state_rows - s)', body)
        calls = re.findall(r'lcopy\(([^;]*)\);', body)
        self.assertEqual(calls, ['cur + upper, dst + upper + row * s, (16 - s) * row',
                                 'prev + halo, dst + upper, row * s',
                                 'cur + lower, dst + lower + row * s, (16 - s) * row',
                                 'cur + upper + row * (16 - s), dst + lower, row * s'])

    def test_the_reader_flushes_every_loaded_x_and_carry_tile_under_the_define(self):
        text = kernel(pcx.READER)
        self.assertIn('#ifdef PCX_FLUSH_X_DENORM', text)
        self.assertIn('if ((value & 0x7F80u) == 0 && (value & 0x7Fu) != 0) {', text)
        self.assertIn('#if PCX_FLUSH_X_DENORM == 2', text)
        self.assertIn('#define PCX_FLUSH(address) ((void)0)', text)
        body = text[text.index('void kernel_main'):]
        self.assertEqual(body.count('PCX_FLUSH(carry_tile);'), 2)       # compute path and shift-only path
        self.assertIn('PCX_FLUSH(cur);', body)
        self.assertIn('PCX_FLUSH(halo);', body)
        # cur is flushed after its read barrier and before any copy out of it, and outside `if (canon)`.
        loop = body[body.index('for (uint32_t ht = ht0; ht < ht0 + R; ++ht)'):]
        self.assertLess(loop.index('noc_async_read_barrier();'), loop.index('PCX_FLUSH(cur);'))
        self.assertLess(loop.index('PCX_FLUSH(cur);'), loop.index('lcopy(cur, get_write_ptr(cb_x0), page);'))
        self.assertLess(loop.index('PCX_FLUSH(halo);'), loop.index('shift_tile(3,'))
        self.assertLess(body.index('PCX_FLUSH(cur);'), body.index('if (canon)'))

    def test_source_sha_tracks_every_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in pcx.KERNEL_FILES:
                (root / name).write_bytes(b'kernel ' + name.encode())
            first = pcx.source_sha(root)
            self.assertEqual(len(first), 8)
            for name in pcx.KERNEL_FILES:
                (root / name).write_bytes(b'changed ' + name.encode())
                self.assertNotEqual(pcx.source_sha(root), first, name)
                first = pcx.source_sha(root)


# ---------------------------------------------------------------------------------------------
# Descriptor building against a fake ttnn.
# ---------------------------------------------------------------------------------------------

class FakeConfig:
    def __init__(self, kind):
        self.kind = kind

    def __eq__(self, other):
        return isinstance(other, FakeConfig) and other.kind == self.kind

    def __hash__(self):
        return hash(self.kind)


class FakeShard:
    def __init__(self, address, kind):
        self.address, self.kind = address, kind

    def buffer_address(self):
        return self.address


class FakeTensor:
    def __init__(self, ops, shape, kind='dram', dtype='bf16', layout='tile', addresses=None):
        self.shape, self.dtype, self.layout = tuple(shape), dtype, layout
        self._config = FakeConfig(kind)
        self.shards = [FakeShard(address, kind) for address in (addresses or ops.fresh())]
        self.deallocated = False

    def memory_config(self):
        return self._config


class Grid:
    x, y = 11, 10


class FakeMesh:
    def __init__(self, shape=(1, 2)):
        self.shape = shape

    def compute_with_storage_grid_size(self):
        return Grid()


class Record:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeRuntimeArgs:
    def __init__(self):
        self.cells = {}

    def __getitem__(self, x):
        runtime = self

        class Column:
            def __setitem__(self, y, values):
                runtime.cells[(x, y)] = list(values)
        return Column()


class FakeKernel(Record):
    def __init__(self, **kwargs):
        kwargs.setdefault('common_runtime_args', [])
        super().__init__(**kwargs)
        self.history = []

    def __setattr__(self, key, value):
        if key == 'common_runtime_args' and hasattr(self, 'history'):
            self.history.append(list(value))
        object.__setattr__(self, key, value)


class FakeProgram(Record):
    def __init__(self, kernels, cbs):
        super().__init__(kernels=kernels, cbs=cbs)
        # A ProgramDescriptor copies its kernels; snapshot what the call would hand the device.
        self.snapshot = [dict(source=Path(k.kernel_source).name, common=list(k.common_runtime_args),
                              compile=list(k.compile_time_args), defines=list(k.defines), runtime=k.runtime_args.cells,
                              config=k.config) for k in kernels]


class FakeMeshProgram(dict):
    pass


class FakeOps:
    bfloat16, TILE_LAYOUT = 'bf16', 'tile'
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = FakeConfig('dram'), FakeConfig('l1')

    class MathFidelity:
        HiFi4 = 'HiFi4'

    class DataMovementProcessor:
        RISCV_0, RISCV_1 = 'RISCV_0', 'RISCV_1'

    class NOC:
        RISCV_0_default, RISCV_1_default = 'NOC0', 'NOC1'

    def __init__(self):
        self.next = 0x10000
        self.calls = []
        self.deallocated = []

    def fresh(self, chips=2):
        self.next += 0x100000
        return [self.next + chip for chip in range(chips)]

    # tensors
    def empty(self, shape, dtype, layout, device, memory_config):
        tensor = FakeTensor(self, shape, kind=memory_config.kind, addresses=self.fresh(len(pcx.mesh_coordinates(device.shape))))
        self.calls.append(('empty', tuple(shape)))
        return tensor

    def deallocate(self, tensor):
        self.deallocated.append(tensor)

    def get_device_tensors(self, tensor):
        return tensor.shards

    def TensorAccessorArgs(self, shard):
        ops = self

        class Args:
            def get_compile_time_args(self):
                return [7 if shard.kind == 'l1' else 3, 0]
        return Args()

    # descriptors
    def CoreCoord(self, x, y):
        return (x, y)

    def CoreRange(self, a, b):
        return (a, b)

    def CoreRangeSet(self, ranges):
        return tuple(ranges)

    def Tile(self, dims):
        return tuple(dims)

    def TileDescriptor(self, tile):
        return tile

    def CBFormatDescriptor(self, **kwargs):
        return Record(**kwargs)

    def CBDescriptor(self, **kwargs):
        return Record(**kwargs)

    def RuntimeArgs(self):
        return FakeRuntimeArgs()

    def KernelDescriptor(self, **kwargs):
        return FakeKernel(**kwargs)

    def DataMovementConfigDescriptor(self, **kwargs):
        return Record(kind='dm', **kwargs)

    def ComputeConfigDescriptor(self, **kwargs):
        return Record(kind='compute', **kwargs)

    def ProgramDescriptor(self, kernels, cbs):
        return FakeProgram(kernels, cbs)

    def MeshProgramDescriptor(self):
        return FakeMeshProgram()

    def MeshCoordinate(self, row, col):
        return (row, col)

    def MeshCoordinateRange(self, a, b):
        return (a, b)

    def generic_op(self, tensors, program):
        self.calls.append(('generic_op', list(tensors), {key: value.snapshot for key, value in program.items()}))
        return tensors[-1]


def inputs(ops, T=2048, C=5120, carry=True, qkv_kind='l1', mesh_shape=(1, 2)):
    chips = len(pcx.mesh_coordinates(mesh_shape))
    qkv = FakeTensor(ops, (1, T, C), kind=qkv_kind, addresses=ops.fresh(chips))
    state = FakeTensor(ops, (1, 3, C), addresses=ops.fresh(chips)) if carry else None
    taps = [FakeTensor(ops, (1, 1, C), addresses=ops.fresh(chips)) for _ in range(4)]
    return qkv, state, taps


class DescriptorTests(unittest.TestCase):
    def setUp(self):
        pcx.clear_cache()
        self.ops = FakeOps()
        self.mesh = FakeMesh()

    def tearDown(self):
        pcx.clear_cache()

    def call(self, qkv, carry, taps, **kwargs):
        kwargs.setdefault('key_dim_tp', 1024)
        return pcx.gdn_prefill_conv_exact(self.mesh, qkv, carry, taps, operations=self.ops, **kwargs)

    def last_program(self):
        return [call for call in self.ops.calls if call[0] == 'generic_op'][-1]

    def test_outputs_are_four_fresh_dram_tensors_of_the_served_shapes(self):
        qkv, carry, taps = inputs(self.ops)
        q, k, v, state = self.call(qkv, carry, taps, valid_len=1288)
        self.assertEqual([t.shape for t in (q, k, v, state)], [(1, 2048, 1024), (1, 2048, 1024), (1, 2048, 3072), (1, 3, 5120)])
        self.assertTrue(all(t.memory_config() == self.ops.DRAM_MEMORY_CONFIG for t in (q, k, v, state)))
        self.assertIsNot(state, carry)

    def test_the_program_per_chip(self):
        qkv, carry, taps = inputs(self.ops)
        q, k, v, state = self.call(qkv, carry, taps, valid_len=1288)
        _, tensors, programs = self.last_program()
        self.assertEqual(tensors, [qkv, carry, *taps, q, k, v, state])
        self.assertEqual(sorted(programs), [((0, 0), (0, 0)), ((0, 1), (0, 1))])
        for chip, key in enumerate(sorted(programs)):
            kernels = programs[key]
            self.assertEqual([entry['source'] for entry in kernels], [pcx.READER, pcx.WRITER, pcx.COMPUTE])
            addresses = [t.shards[chip].address for t in (qkv, carry, *taps, q, k, v, state)]
            for entry in kernels:
                self.assertEqual(entry['common'], addresses + [1288, 1])
            head = [64, 160, 32, 2, 32, 1]
            accessor = [7, 0] + [3, 0] * 9
            self.assertEqual(kernels[0]['compile'], head + accessor)
            self.assertEqual(kernels[1]['compile'], head + accessor)
            self.assertEqual(kernels[2]['compile'], head)
            self.assertEqual(kernels[0]['config'].processor, 'RISCV_1')
            self.assertEqual(kernels[1]['config'].processor, 'RISCV_0')
            config = kernels[2]['config']
            self.assertEqual((config.math_fidelity, config.fp32_dest_acc_en, config.math_approx_mode), ('HiFi4', False, False))
            defines = dict(kernels[0]['defines'])
            self.assertEqual(sorted(defines), ['CANON_DENORM', 'PCX_SRC_SHA'])   # CANON_DENORM_DEFAULT
            self.assertEqual(defines['PCX_SRC_SHA'], '0x' + pcx.source_sha())
            runtime = kernels[0]['runtime']
            self.assertEqual(len(runtime), 110)
            self.assertEqual(runtime[(0, 0)], [0, 3])
            self.assertEqual(runtime[(10, 9)], [318, 2])
            self.assertEqual(runtime[(0, 1)], [33, 3])

    def test_cbs(self):
        qkv, carry, taps = inputs(self.ops)
        self.call(qkv, carry, taps, valid_len=2048)
        prepared = list(pcx._DESCRIPTORS.values())[0]
        sizes = {cb.format_descriptors[0].buffer_index: cb.total_size for cb in prepared.cbs}
        self.assertEqual(sizes, {index: pages * 2048 for index, pages in pcx.CB_PAGES.items()})
        self.assertEqual(sum(sizes.values()), 52 * 1024)
        pcx.clear_cache()
        self.call(qkv, carry, taps, valid_len=2048, mirror_pack=True)
        prepared = list(pcx._DESCRIPTORS.values())[0]
        self.assertEqual(sum(cb.total_size for cb in prepared.cbs), 56 * 1024)

    def test_cache_hit_rewrites_only_addresses_and_valid_len(self):
        qkv, carry, taps = inputs(self.ops)
        self.call(qkv, carry, taps, valid_len=2048)
        first = self.last_program()[2]
        self.assertEqual(pcx.cache_size(), 1)
        qkv2, carry2, taps2 = inputs(self.ops)
        outs = self.call(qkv2, carry2, taps2, valid_len=1)
        second = self.last_program()[2]
        self.assertEqual(pcx.cache_size(), 1)
        for key in first:
            for a, b in zip(first[key], second[key]):
                self.assertEqual({k: v for k, v in a.items() if k != 'common'}, {k: v for k, v in b.items() if k != 'common'})
                self.assertNotEqual(a['common'], b['common'])
                self.assertEqual(b['common'][10:], [1, 1])
                self.assertEqual(len(a['common']), len(b['common']))
        chip1 = second[((0, 1), (0, 1))][0]['common']
        self.assertEqual(chip1[:10], [t.shards[1].address for t in (qkv2, carry2, *taps2, *outs)])

    def test_valid_len_none_is_t_with_the_static_canon_state(self):
        """valid_len None is T; its state is canonicalised (canon 1) under STATIC_CANON_DEFAULT,
        which card M pcx-20260923T063828 set, and raw (canon 0) with static_canon=False. canon is
        a runtime arg, so the switch reuses the cached program."""
        self.assertTrue(pcx.STATIC_CANON_DEFAULT)
        self.assertTrue(pcx.CANON_DENORM_DEFAULT)
        qkv, carry, taps = inputs(self.ops)
        self.call(qkv, carry, taps)
        common = self.last_program()[2][((0, 0), (0, 0))][0]['common']
        self.assertEqual(common[10:], [2048, 1])
        self.call(qkv, carry, taps, static_canon=False)
        common = self.last_program()[2][((0, 0), (0, 0))][0]['common']
        self.assertEqual(common[10:], [2048, 0])
        self.call(qkv, carry, taps, valid_len=9, static_canon=False)    # inert on the one-hot path
        common = self.last_program()[2][((0, 0), (0, 0))][0]['common']
        self.assertEqual(common[10:], [9, 1])
        self.assertEqual(pcx.cache_size(), 1)
        self.assertEqual(pcx.common_args(list(range(10)), None, 64), list(range(10)) + [64, 0])
        self.assertEqual(pcx.common_args(list(range(10)), None, 64, True), list(range(10)) + [64, 1])

    def test_new_keys(self):
        qkv, carry, taps = inputs(self.ops)
        self.call(qkv, carry, taps, valid_len=5)
        qkv64, carry64, taps64 = inputs(self.ops, T=64)
        self.call(qkv64, carry64, taps64, valid_len=5)
        self.assertEqual(pcx.cache_size(), 2)
        self.call(qkv, None, taps, valid_len=5)                      # has_carry 0
        self.assertEqual(pcx.cache_size(), 3)
        dram_qkv, _, _ = inputs(self.ops, qkv_kind='dram')
        self.call(dram_qkv, carry, taps, valid_len=5)                # accessor args differ
        self.assertEqual(pcx.cache_size(), 4)
        self.call(qkv, carry, taps, valid_len=5, negative='tapswap')
        self.assertEqual(pcx.cache_size(), 5)

    def test_the_x_flush_define_and_its_own_cache_entry(self):
        qkv, carry, taps = inputs(self.ops)
        self.call(qkv, carry, taps, valid_len=5)
        self.assertEqual(pcx.FLUSH_X_DENORM_DEFAULT, 0)
        for kernel_entry in self.last_program()[2][((0, 0), (0, 0))]:
            self.assertNotIn('PCX_FLUSH_X_DENORM', dict(kernel_entry['defines']))
        for mode, size in ((1, 2), (2, 3)):
            for valid_len in (5, None):
                self.call(qkv, carry, taps, valid_len=valid_len, flush_x_denorm=mode)
                reader = self.last_program()[2][((0, 0), (0, 0))][0]
                self.assertEqual(dict(reader['defines'])['PCX_FLUSH_X_DENORM'], str(mode))
            self.assertEqual(pcx.cache_size(), size)
        self.assertEqual(pcx.kernel_defines('ab', flush_x_denorm=0), (('PCX_SRC_SHA', '0xab'),))
        with self.assertRaises(ValueError):
            pcx.kernel_defines('ab', flush_x_denorm=3)
        with self.assertRaises(ValueError):
            self.call(qkv, carry, taps, valid_len=5, flush_x_denorm=True + 2)

    def test_no_carry_passes_qkv_in_the_carry_slot_and_has_carry_zero(self):
        qkv, _, taps = inputs(self.ops)
        self.call(qkv, None, taps, valid_len=7)
        _, tensors, programs = self.last_program()
        self.assertEqual(tensors[:2], [qkv, taps[0]])
        reader = programs[((0, 0), (0, 0))][0]
        self.assertEqual(reader['compile'][5], 0)
        self.assertEqual(reader['common'][1], qkv.shards[0].address)

    def test_negative_controls(self):
        qkv, carry, taps = inputs(self.ops)
        self.call(qkv, carry, taps, valid_len=9, negative='fp32')
        compute = self.last_program()[2][((0, 0), (0, 0))][2]
        self.assertTrue(compute['config'].fp32_dest_acc_en)
        self.assertIn(('NEG_FP32', '1'), compute['defines'])
        self.call(qkv, carry, taps, valid_len=9, negative='stale')
        reader = self.last_program()[2][((0, 0), (0, 0))][0]
        self.assertEqual(reader['compile'][5], 0)
        self.assertIn(('NEG_STALE', '1'), reader['defines'])
        with self.assertRaises(ValueError):
            self.call(qkv, carry, taps, valid_len=9, negative='bogus')

    def test_shift_only_microbench_has_no_compute_kernel_and_one_output(self):
        qkv, carry, taps = inputs(self.ops)
        out = self.call(qkv, carry, taps, shift_only=True)
        self.assertEqual(out.shape, (1, 2048, 5120))
        _, tensors, programs = self.last_program()
        self.assertEqual(tensors, [qkv, carry, *taps, out])
        kernels = programs[((0, 0), (0, 0))]
        self.assertEqual([entry['source'] for entry in kernels], [pcx.READER, pcx.WRITER])
        self.assertIn(('PCX_MB_SHIFT_ONLY', '1'), kernels[0]['defines'])

    def test_single_chip_mesh(self):
        self.mesh = FakeMesh((1, 1))
        qkv, carry, taps = inputs(self.ops, mesh_shape=(1, 1))
        self.call(qkv, carry, taps, valid_len=3)
        self.assertEqual(sorted(self.last_program()[2]), [((0, 0), (0, 0))])

    def test_unsupported_inputs_refuse_before_any_allocation(self):
        qkv, carry, taps = inputs(self.ops)
        bad = [
            dict(qkv=FakeTensor(self.ops, (2, 2048, 5120))),
            dict(qkv=FakeTensor(self.ops, (1, 2040, 5120))),
            dict(qkv=FakeTensor(self.ops, (1, 2048, 5120), dtype='f32')),
            dict(qkv=FakeTensor(self.ops, (1, 2048, 5120), kind='sharded')),
            dict(carry=FakeTensor(self.ops, (1, 4, 5120))),
            dict(taps=taps[:3]),
            dict(taps=[FakeTensor(self.ops, (1, 1, 2560))] * 4),
            dict(valid_len=0),
            dict(valid_len=2049),
            dict(key_dim_tp=2560),
            dict(key_dim_tp=1000),
            dict(mesh=FakeMesh((2, 2))),
        ]
        for case in bad:
            with self.subTest(case=sorted(case)):
                args = dict(qkv=qkv, carry=carry, taps=taps, valid_len=5, key_dim_tp=1024, mesh=self.mesh)
                args.update(case)
                before = len(self.ops.calls)
                with self.assertRaises(pcx.Unsupported):
                    pcx.gdn_prefill_conv_exact(args.pop('mesh'), args.pop('qkv'), args.pop('carry'), args.pop('taps'),
                                               operations=self.ops, **args)
                self.assertEqual(len(self.ops.calls), before)
        self.assertIsNone(pcx.unsupported(self.ops, self.mesh, qkv, carry, taps, 5, 1024))
        self.assertEqual(pcx.unsupported(self.ops, self.mesh, qkv, carry, taps, 5, 1024, flat=False), 'q/k/v not flat')
        self.assertIn('kernel size', pcx.unsupported(self.ops, self.mesh, qkv, carry, taps, 5, 1024, kernel_size=3))

    def test_an_aliased_output_is_refused_and_every_output_released(self):
        qkv, carry, taps = inputs(self.ops)
        original = self.ops.empty

        def aliasing(shape, **kwargs):
            tensor = original(shape, **kwargs)
            if tuple(shape) == (1, 3, 5120):
                tensor.shards[0].address = qkv.shards[0].address
            return tensor
        self.ops.empty = aliasing
        with self.assertRaisesRegex(ValueError, 'alias'):
            self.call(qkv, carry, taps, valid_len=5)
        self.assertEqual(len(self.ops.deallocated), 4)
        self.assertFalse(any(call[0] == 'generic_op' for call in self.ops.calls))


class AuditTests(unittest.TestCase):
    def test_the_audit_calls_the_fir_the_way_tp_py_does_and_compares_bytes(self):
        ops = FakeOps()
        mesh = FakeMesh()
        qkv, carry, taps = inputs(ops, T=64, C=320)
        calls = []
        values = {}

        def fir(x, weight, bias, kernel_size, device, **kwargs):
            calls.append(((x, weight, bias, kernel_size, device), kwargs))
            conv = FakeTensor(ops, (1, 64, 320))
            state = FakeTensor(ops, (1, 3, 320))
            values[id(state)] = torch.ones(1, 3, 320, dtype=torch.bfloat16)
            return conv, state

        def slice_(tensor, start, end):
            out = FakeTensor(ops, tuple(e - s for s, e in zip(start, end)))
            values[id(out)] = torch.full(out.shape, 2.0, dtype=torch.bfloat16)
            return out
        ops.slice = slice_
        shard_owner = {}

        def get_device_tensors(tensor):
            for shard in tensor.shards:
                shard_owner[id(shard)] = tensor
            return tensor.shards
        ops.get_device_tensors = get_device_tensors
        ops.to_torch = lambda shard: values[id(shard_owner[id(shard)])]
        outs = []
        for shape, fill in (((1, 64, 64), 2.0), ((1, 64, 64), 2.0), ((1, 64, 192), 2.0), ((1, 3, 320), 1.0)):
            tensor = FakeTensor(ops, shape)
            values[id(tensor)] = torch.full(shape, fill, dtype=torch.bfloat16)
            outs.append(tensor)
        report = pcx.audit_against_fir(ops, fir, mesh, qkv, carry, taps, 1288 // 32, 64, outs)
        self.assertTrue(report['exact'], report)
        (args, kwargs), = calls
        self.assertEqual(args, (qkv, None, None, 4, mesh))
        self.assertEqual(kwargs, dict(memory_config=ops.L1_MEMORY_CONFIG, conv_state=carry, weight_taps=taps,
                                      bias_dev=None, valid_len=40))
        self.assertEqual(len(ops.deallocated), 5)
        values[id(outs[3])] = torch.full((1, 3, 320), -0.0, dtype=torch.bfloat16)
        values[id(outs[3])][0, 0, 0] = 1.0
        report = pcx.audit_against_fir(ops, fir, mesh, qkv, carry, taps, 40, 64, outs)
        self.assertFalse(report['exact'])
        self.assertEqual(sorted(report['mismatches']), ['new_state/chip0', 'new_state/chip1'])


if __name__ == '__main__':
    unittest.main()
