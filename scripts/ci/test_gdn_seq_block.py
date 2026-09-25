"""K5-A (gdn_seq_block) host tests: the G-CPU list of the K5 plan, section 5.

What is held here, all on the host:
  - the flags: QWEN_FAST_GDN_SEQ_BLOCK parsing and its QWEN_FAST_GDN_USER_BATCH requirement, the
    level bitmask, the audit layer list;
  - the CB plan: totals, index range, the one-producer / one-consumer owner table, ONES == 6, and
    that the three .cpp sources name every CB at the plan's index;
  - the compile and runtime args, held against gdn_user_batch's served fused values;
  - the native prefix: the hash check before slicing, the anchors, the generated build header;
  - QUALIFIED: empty, and an unqualified triple refused everywhere but the probe's builder argument;
  - descriptor parity with gdn_user_batch over the same recording fake;
  - the model_batch marker and the gate regex; the audit launch kept out of the T1 gate's count;
  - the rig runner's pins (card B only, the hang hint); the probe's compare helpers and its P0,
    traced and P1 verdicts, as pure functions;
  - QUALIFIED: the card-B level-0 triple, and (where the evidence export is checked out) that the
    pinned sources generate exactly it;
  - the wiring: gdn_user_batch_conv's dispatch (K5-A only with the flag and every width 16), the
    model_batch counter, level and marker, the audit (scope, launches, holds, the after-replay
    compare, packed_verifier's call), the arm's refusals and passthroughs, the gate's checks,
    both image copy lists and the CPU allowlist; flag off, gdn_user_batch_conv and model_batch
    are the parent's (e7bfd324), call for call, and packed_verifier's served rounds are its
    parent's byte for byte (test_padded_probe's fixture).

gdn_seq_block.execute checks the served runtime pin (gdn_multitoken.validate_handoff_runtime) as
gdn_user_batch.execute does; it reads tt-metal files, so this module patches it out for every test
(setUpModule) and HandoffPinTests holds the call itself.
"""

from contextlib import nullcontext
import hashlib
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gdn_multitoken as native
import gdn_seq_block as seq
import gdn_user_batch as batch
from test_gdn_user_batch import FakeTTNN, FakeTensor, KERNELS, mesh, user_inputs
import test_padded_probe
from test_qual_card import BASH, CARD_B, CARD_M, q, run as run_bash
import verify_trace_t1


HERE = Path(__file__).resolve().parent
# The pinned native kernels, where a checkout has the evidence export (never in CI).
EVIDENCE = [Path(os.environ['GDN_NATIVE_ROOT'])] if os.environ.get('GDN_NATIVE_ROOT') else []
EVIDENCE += [parent / 'runner-evidence.local' / '34695341501' / 'gdn-source' for parent in HERE.parents]
NATIVE_ROOT = next((path for path in EVIDENCE if (path / native.KERNEL_ROOT / seq.NATIVE_COMPUTE).exists()), None)

# A synthetic native compute with the served prefix's shape: every anchor once, the helper namespace,
# then the entry point.
SYNTHETIC_PREFIX = '#include <cstdint>\nnamespace {\n' + '\n'.join(
    anchor if anchor.endswith(';') or anchor.endswith('}') else anchor + '\n}' for anchor in seq.NATIVE_ANCHORS
) + '\n}  // namespace\n\n'
SYNTHETIC_NATIVE = SYNTHETIC_PREFIX + 'void kernel_main() {\n    // served body\n}\n'


class SyntheticRoot:
    """A temporary tt-metal root whose native compute is SYNTHETIC_NATIVE, pinned by a patched hash."""

    def __init__(self, source=SYNTHETIC_NATIVE):
        self.source = source

    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        path = root / native.KERNEL_ROOT / seq.NATIVE_COMPUTE
        path.parent.mkdir(parents=True)
        path.write_bytes(self.source.encode())
        digest = hashlib.sha256(self.source.encode()).hexdigest()
        self.patch = patch.dict(native.HASHES, {seq.NATIVE_COMPUTE: digest})
        self.patch.start()
        return root

    def __exit__(self, *unused):
        self.patch.stop()
        self.directory.cleanup()


_HANDOFF = patch('gdn_multitoken.validate_handoff_runtime')


def setUpModule():
    _HANDOFF.start()


def tearDownModule():
    _HANDOFF.stop()


def build(variant='A', diag=None):
    with SyntheticRoot() as root:
        return seq.load_kernels(root, 0, variant=variant, diag=diag, unqualified=True)


def cb_constants(source):
    """SB_<NAME> = <index> for every constexpr the source declares."""
    found = {}
    for line in source.splitlines():
        if line.startswith('constexpr uint32_t SB_'):
            for name, value in re.findall(r'SB_([A-Z0-9_]+) = (\d+)', line):
                found[name] = int(value)
    return found


class FlagTests(unittest.TestCase):
    def test_the_flag_is_off_by_default_and_rejects_anything_but_zero_or_one(self):
        self.assertFalse(seq.enabled({}))
        self.assertFalse(seq.enabled({seq.FLAG: '0'}))
        self.assertTrue(seq.enabled({seq.FLAG: '1', batch.FLAG: '1'}))
        for value in ('true', 'yes', '', '2', 'on', ' 1'):
            with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                seq.enabled({seq.FLAG: value, batch.FLAG: '1'})

    def test_the_flag_requires_the_user_batch_flag(self):
        for environ in ({seq.FLAG: '1'}, {seq.FLAG: '1', batch.FLAG: '0'}):
            with self.assertRaisesRegex(ValueError, 'requires QWEN_FAST_GDN_USER_BATCH=1'):
                seq.enabled(environ)
        # Off never looks at the batch flag, and a malformed batch flag is still its own error.
        self.assertFalse(seq.enabled({seq.FLAG: '0', batch.FLAG: '0'}))
        with self.assertRaisesRegex(ValueError, batch.FLAG):
            seq.enabled({seq.FLAG: '1', batch.FLAG: 'yes'})

    def test_the_level_is_a_decimal_bitmask_of_the_increments(self):
        self.assertEqual(seq.level({}), 0)
        self.assertEqual(seq.level({seq.LEVEL_FLAG: '0'}), 0)
        self.assertEqual(seq.level({seq.LEVEL_FLAG: '5'}), 5)
        self.assertEqual(seq.level({seq.LEVEL_FLAG: '15'}), 15)
        for value in ('16', '', '-1', '01', '0x1', ' 1', '1.0', 'one'):
            with self.assertRaises(ValueError):
                seq.level({seq.LEVEL_FLAG: value})

    def test_the_audit_flag_lists_distinct_gdn_layers(self):
        self.assertEqual(seq.audit_layers({}), ())
        self.assertEqual(seq.audit_layers({seq.AUDIT_FLAG: ''}), ())
        self.assertEqual(seq.audit_layers({seq.AUDIT_FLAG: '0,23,47'}), (0, 23, 47))
        self.assertEqual(seq.audit_layers({seq.AUDIT_FLAG: '5'}), (5,))
        for value in ('48', '0,0', '01', '-1', ' 1', '1,', ',1', '1;2', 'all'):
            with self.assertRaisesRegex(ValueError, 'distinct GDN layers'):
                seq.audit_layers({seq.AUDIT_FLAG: value})

    def test_nothing_reads_the_probe_only_builds_from_the_environment(self):
        text = (HERE / 'gdn_seq_block.py').read_text()
        for name in ('unqualified', 'variant', 'diag'):
            self.assertNotIn("environ.get('%s" % name, text)
        self.assertEqual(sorted(re.findall(r"'(QWEN_FAST_[A-Z0-9_]+)'", text)),
                         sorted([seq.FLAG, seq.LEVEL_FLAG, seq.AUDIT_FLAG]))


class PlanTests(unittest.TestCase):
    def test_the_totals_are_the_plans_and_within_todays_budget(self):
        io, fp32 = seq.cb_plan()
        # The plan's 78 bf16 pages plus SOUT2 (14); its 109 fp32 pages less FAC_K (1).
        self.assertEqual(sum(io.values()), 92)
        self.assertEqual(sum(io.values()) * 2048, 188416)
        self.assertEqual(sum(fp32.values()), 108)
        self.assertEqual(sum(fp32.values()) * 4096, 442368)
        self.assertEqual(seq.cb_bytes(), 630784)
        self.assertEqual(seq.SERVED_CB_BYTES, 630784)
        self.assertEqual(seq.SERVED_CB_BYTES - seq.cb_bytes(), 0)

    def test_the_snapshot_leaves_in_two_halves_one_per_noc(self):
        # SOUT (compute -> writer, NoC 0) carries K rows 0-1, SOUT2 (compute -> reader, NoC 1) rows
        # 2-3: 8 pages a token each, pushed and popped two pages (a column) at a time.
        self.assertEqual(seq.CB_PLAN[7], ('SOUT', 16, 'bf16', 'compute', 'writer'))
        self.assertEqual(seq.CB_PLAN[21], ('SOUT2', 14, 'bf16', 'compute', 'reader'))
        self.assertNotIn('FAC_K', [entry[0] for entry in seq.CB_PLAN.values()])
        for index in (7, 21):
            self.assertEqual(seq.CB_PLAN[index][1] % 2, 0)
            self.assertGreaterEqual(seq.CB_PLAN[index][1], 8)
        compute = (HERE / seq.SOURCES['compute']).read_text()
        self.assertIn('pack_tile(0, i < SB_KT_HALF ? out : out2, i % SB_KT_HALF);', compute)
        self.assertEqual(compute.count('l2_norm_rows(SB_X, SB_FAC_Q, SB_QKN'), 2)
        # Mirror of the kernels' page maps: every states page of a token written exactly once, by
        # the RISC whose ring holds that tile, from the CB page the compute packed it to; each half
        # walks the 8 banks (page mod 8) once from a per-core start.
        Kt = Vt = 4
        rows, half = Kt // 2, Kt // 2 * Vt
        for h in range(24):
            written = []
            for role, start, offset in (('writer', h % half, 0), ('reader', (h + half // 2) % half, half)):
                banks = []
                for k in range(half):
                    q = (start + k) % half
                    slot = (q % Vt) * rows + q // Vt           # CB page 2j + i (i within the half)
                    i, j = q // Vt + offset // Vt, q % Vt        # the tile (i, j)
                    self.assertEqual(slot, 2 * j + (i % rows))
                    written.append(offset + q)
                    self.assertEqual(offset + q, 4 * i + j)     # the served page 4i + j
                    banks.append((offset + q) % 8)
                self.assertEqual(sorted(banks), list(range(8)), (h, role))
            self.assertEqual(sorted(written), list(range(16)), h)
        for role, name in (('writer', 'sout'), ('reader', 'sout2')):
            source = (HERE / seq.SOURCES[role]).read_text()
            self.assertIn('const uint32_t slot = (q % Vt) * rows + q / Vt;', source)
            self.assertIn('%s.pop_front(rows);' % name, source)
            self.assertIn('if (addr >= %s_limit) {' % name, source)

    def test_the_bisection_build_adds_only_its_outer_ring(self):
        io, fp32 = seq.cb_plan('A0')
        self.assertEqual(set(io) | set(fp32), set(range(32)))
        self.assertEqual(fp32[31], 16)
        self.assertEqual(seq.plan('A0')[31][0], 'OUTER')
        self.assertEqual(seq.cb_bytes('A0') - seq.cb_bytes('A'), 65536)
        self.assertEqual(seq.cb_plan('N'), seq.cb_plan('A'))
        with self.assertRaises(ValueError):
            seq.cb_plan('B')

    def test_indices_are_unique_and_within_the_documented_range(self):
        for variant in seq.VARIANTS:
            io, fp32 = seq.cb_plan(variant)
            self.assertFalse(set(io) & set(fp32))
            self.assertTrue(all(0 <= index <= 31 for index in list(io) + list(fp32)))
        # 0-30: the plan's table (thirty indices; its text says 29) plus RT, the round-trip ring.
        self.assertEqual(sorted(seq.CB_PLAN), list(range(31)))

    def test_every_cb_has_one_producer_and_one_consumer(self):
        for variant in seq.VARIANTS:
            for index, (name, pages, dtype, producer, consumer) in seq.plan(variant).items():
                self.assertIn(dtype, seq.PAGE_BYTES)
                self.assertIn(consumer, seq.RISCS, name)
                self.assertIn(producer, seq.RISCS, name)
        names = [entry[0] for entry in seq.plan('A0').values()]
        self.assertEqual(len(names), len(set(names)))
        self.assertFalse(hasattr(seq, 'SHARED_RINGS'))
        # The plan's 8-page W ring is split: norm_w in, the epilogue's round trips in RT. Same bytes.
        self.assertEqual(seq.CB_PLAN[8], ('W', 4, 'bf16', 'reader', 'compute'))
        self.assertEqual(seq.CB_PLAN[30], ('RT', 4, 'bf16', 'compute', 'compute'))

    def test_the_compute_never_produces_w_and_the_round_trips_use_rt(self):
        compute = (HERE / seq.SOURCES['compute']).read_text()
        body = compute[compute.index('void kernel_main() {'):]
        uses = [line.strip() for line in body.splitlines() if re.search(r'\bSB_W\b', line)]
        self.assertEqual(uses, ['WAIT(SB_W, SB_VT);', 'copy_tiles(SB_W, SB_WF, SB_VT);', 'POP(SB_W, SB_VT);'])
        # Both bf16 round trips, as served [540-552], through RT; nothing else touches RT.
        trips = [line.strip() for line in body.splitlines() if re.search(r'\bSB_RT\b', line)]
        self.assertEqual(trips, ['copy_tiles(SB_H, SB_RT, SB_VT);  // -> bf16', 'WAIT(SB_RT, SB_VT);',
                                 'copy_tiles(SB_RT, SB_H, SB_VT);  // -> fp32 again, bf16-valued',
                                 'POP(SB_RT, SB_VT);', 'copy_tiles(SB_DL, SB_RT, SB_VT);  // -> bf16',
                                 'WAIT(SB_RT, SB_VT);', 'copy_tiles(SB_RT, SB_DL, SB_VT);  // -> fp32 again, bf16-valued',
                                 'POP(SB_RT, SB_VT);'])
        for role in ('reader', 'writer'):
            self.assertNotIn('RT', cb_constants((HERE / seq.SOURCES[role]).read_text()))
        # norm_w is read in the epilogue only: the reader sends it after the last token.
        self.assertGreater(body.index('WAIT(SB_W, SB_VT);'), body.index('// ---- epilogue'))

    def test_the_reader_reads_once_and_never_zeroes_the_staging_rings(self):
        reader = (HERE / seq.SOURCES['reader']).read_text()
        body = reader[reader.index('void kernel_main() {'):]
        # Every DRAM read behind one barrier, ONES built while they fly, the inputs pushed after it.
        self.assertEqual(body.count('noc.async_read_barrier();'), 1)
        barrier = body.index('noc.async_read_barrier();')
        self.assertLess(body.index('cb.push_back(1);'), barrier)  # ONES
        self.assertEqual(body.count('noc.async_read('), 2)       # read_pages and the column-major S0
        for push in ('in_gb.push_back(2);', 'in_v.push_back(Vt);', 'in_z.push_back(Vt);', 'in_qk.push_back(2 * Kt);',
                     's0.push_back(Kt * Vt);'):
            self.assertGreater(body.index(push), barrier, push)
        # TOKA/TOKB are never zeroed (only what is staged is read); W is finished after the chain's staging.
        self.assertNotIn('zero(', body)
        self.assertGreater(body.index('w.push_back(Vt);'), body.index('qkn.pop_front(2 * Kt);'))
        self.assertEqual(body.count('w.push_back('), 1)
        self.assertEqual(body.count('reserve_back('), 9)  # 6 inputs, ONES, TOKA, TOKB

    def test_ones_stays_where_the_served_helpers_read_it(self):
        self.assertEqual(seq.ONES, 6)
        self.assertEqual(seq.CB_PLAN[6], ('ONES', 1, 'fp32', 'reader', 'compute'))
        self.assertIn('constexpr uint32_t cb_state = 5, cb_ones = 6;', seq.NATIVE_ANCHORS)
        self.assertIn('static_assert(SB_ONES == cb_ones', (HERE / seq.SOURCES['compute']).read_text())

    def test_the_sources_name_every_cb_at_the_plans_index(self):
        by_name = {entry[0]: index for index, entry in seq.plan('A0').items()}
        declared = {role: cb_constants((HERE / seq.SOURCES[role]).read_text()) for role in seq.ROLES}
        self.assertEqual({name: declared['compute'][name] for name in by_name}, by_name)
        for role, expected in (('reader', ('IN_QK', 'IN_V', 'IN_Z', 'IN_GB', 'S0', 'ONES', 'W', 'QKN', 'VF',
                                           'BF', 'GEXP', 'TOKA', 'TOKB', 'SOUT2')),
                               ('writer', ('SOUT', 'OUT', 'OT', 'O'))):
            cbs = {name: value for name, value in declared[role].items() if name in by_name}
            self.assertEqual(sorted(cbs), sorted(expected), role)
            self.assertEqual(cbs, {name: by_name[name] for name in expected}, role)
        # Each DM kernel touches exactly the CBs whose other end is on its RISC.
        for role in ('reader', 'writer'):
            touched = {index for index, entry in seq.CB_PLAN.items() if role in (entry[3], entry[4])}
            self.assertEqual({by_name[name] for name in declared[role] if name in by_name}, touched, role)

    def test_the_staging_rings_are_one_block_each(self):
        # TOKA/TOKB: ring == block, one token's operands each (the reader restages them per token).
        for role in ('reader', 'compute'):
            source = (HERE / seq.SOURCES[role]).read_text()
            self.assertEqual(int(re.search(r'TOKA_PAGES = (\d+)', source).group(1)), seq.CB_PLAN[22][1])
            self.assertEqual(int(re.search(r'TOKB_PAGES = (\d+)', source).group(1)), seq.CB_PLAN[23][1])
            self.assertEqual(re.findall(r'TOKA_[A-Z]+ = (\d+)', source), ['0', '1', '2', '6', '10'])
            self.assertEqual(re.findall(r'TOKB_[A-Z]+ = (\d+)', source), ['0', '4', '8'])
        # W, RT, O, OT and OUT are one Vt block; the state rings hold one token (16 tiles).
        for index in (8, 9, 28, 29, 30):
            self.assertEqual(seq.CB_PLAN[index][1], 4)
        for index in (4, 5, 7, 24, 25):
            self.assertEqual(seq.CB_PLAN[index][1], 16)


class ArgsTests(unittest.TestCase):
    def test_reader_args_are_the_served_fused_geometry(self):
        kernels = build()
        served = batch.compile_args('reader', 16)
        args = seq.compile_args('reader', kernels)
        # Kt, Vt, H, RF, Ct, QOT, KOT, VOT, WTZ, ZOT from the served list's positions.
        self.assertEqual(args[:10], [served[0], served[1], served[5], served[11], served[7], served[8], served[9],
                                     served[10], served[14], served[15]])
        self.assertEqual(args[:10], [4, 4, 24, 3, 160, 0, 32, 64, 96, 0])
        self.assertEqual(args[10], kernels.tag('reader'))
        self.assertIn('TensorAccessorArgs<%d>()' % len(args), kernels['reader'])

    def test_writer_and_compute_args(self):
        kernels = build()
        args = seq.compile_args('writer', kernels)
        self.assertEqual(args, [4, 4, 24, kernels.tag('writer')])
        self.assertIn('TensorAccessorArgs<%d>()' % len(args), kernels['writer'])
        served = batch.compile_args('compute', 16)
        args = seq.compile_args('compute', kernels)
        self.assertEqual(args[:4], [served[3], served[4], served[6], served[7]])
        self.assertEqual(args[4:], [0, kernels.tag('compute')])
        self.assertEqual(args[0], batch.bits(1e-6))
        self.assertEqual(args[3], batch.bits(128 ** 0.5))
        with self.assertRaises(ValueError):
            seq.compile_args('unknown', kernels)

    def test_src_tag_is_the_first_32_bits_of_each_generated_sources_sha256(self):
        kernels = build()
        for role in seq.ROLES:
            self.assertEqual(kernels.tag(role), int(hashlib.sha256(kernels[role].encode()).hexdigest()[:8], 16))
        tags = {variant: build(variant).tag('compute') for variant in seq.VARIANTS}
        self.assertEqual(len(set(tags.values())), len(tags))
        self.assertNotEqual(build(diag='nosnap').tag('writer'), kernels.tag('writer'))

    def test_runtime_args_place_every_address_where_the_kernels_read_it(self):
        addresses = list(range(10, 90, 10))  # qkv, beta, gate, initial, output, states, z, norm_w
        self.assertEqual(seq.runtime_args('reader', 5, addresses), [5, 10, 20, 30, 40, 70, 80, 60])
        self.assertEqual(seq.runtime_args('writer', 5, addresses), [5, 50, 60])
        self.assertEqual(seq.runtime_args('compute', 5, addresses), [16])
        self.assertEqual(seq.ACCESSORS, dict(reader=(0, 1, 2, 3, 6, 7, 5), writer=(4, 5), compute=()))
        for role in seq.ROLES:
            with self.assertRaises(ValueError):
                seq.runtime_args(role, 24, addresses)
            with self.assertRaises(ValueError):
                seq.runtime_args(role, 0, addresses[:7])
        with self.assertRaises(ValueError):
            seq.runtime_args('unknown', 0, addresses)

    def test_the_kernels_read_their_args_where_the_host_puts_them(self):
        names = ('qkv', 'beta', 'gate', 'initial', 'output', 'states', 'z', 'norm_w')
        aliases = dict(qkv='qkv', beta='beta', g='gate', s0='initial', z='z', w='norm_w', out='output', s='states')
        addresses = list(range(100, 900, 100))
        kernels = build()
        for role in ('reader', 'writer'):
            source = (HERE / seq.SOURCES[role]).read_text()
            # get_arg_val<uint32_t>(N) in index order, each N where runtime_args puts that value.
            read = re.findall(r'const uint32_t (\w+) = get_arg_val<uint32_t>\((\d+)\);', source)
            self.assertEqual([int(index) for unused, index in read], list(range(len(read))), role)
            args = seq.runtime_args(role, 5, addresses)
            self.assertEqual(len(args), len(read), role)
            for name, index in read:
                expected = 5 if name == 'h' else addresses[names.index(aliases[name[:-len('_addr')]])]
                self.assertEqual(args[int(index)], expected, (role, name))
            # The accessors in declaration order are the ACCESSORS tensors, the first right after the
            # compile args, and each is built on its own tensor's address.
            declared = re.findall(r'constexpr auto (\w+)_a = TensorAccessorArgs<', source)
            self.assertEqual(tuple(names.index(aliases[name]) for name in declared), seq.ACCESSORS[role], role)
            first = int(re.search(r'TensorAccessorArgs<(\d+)>\(\)', source).group(1))
            self.assertEqual(first, len(seq.compile_args(role, kernels)), role)
            built = re.findall(r'TensorAccessor\((\w+)_a, (\w+)_addr, \w+\)', source)
            self.assertEqual(sorted(built), sorted((name, name) for name in declared), role)
        # The compute reads no runtime arg (it is still given the served [rows]).
        self.assertNotIn('get_arg_val', (HERE / seq.SOURCES['compute']).read_text())
        self.assertEqual(seq.ACCESSORS['compute'], ())

    def test_only_sixteen_row_segments_are_taken(self):
        good = [[tuple(value.shape) for value in user_inputs(index)] for index in range(4)]
        self.assertEqual(seq.validate_users(good), [16] * 4)
        for rows in (8, 32, 2):
            ragged = [list(user) for user in good]
            ragged[1] = [tuple(value.shape) for value in user_inputs(1, rows=rows)]
            with self.assertRaisesRegex(ValueError, '16-row segments only'):
                seq.validate_users(ragged)
        with self.assertRaises(ValueError):
            seq.validate_users(good + [good[0]])


class SourceTests(unittest.TestCase):
    def test_the_prefix_is_the_native_source_up_to_its_entry_point(self):
        with SyntheticRoot() as root:
            kernels = seq.generate(root)
        self.assertTrue(kernels['compute'].startswith(SYNTHETIC_PREFIX))
        body = kernels['compute'][len(SYNTHETIC_PREFIX):]
        self.assertNotIn('served body', kernels['compute'])
        self.assertTrue(body.startswith((HERE / seq.SOURCES['compute']).read_text().split(seq.BUILD_ANCHOR)[0]))
        self.assertEqual(kernels['compute'].count('void kernel_main() {'), 1)
        for role in seq.ROLES:
            self.assertNotIn('@@GDN_SEQ_BLOCK_BUILD@@', kernels[role])
            self.assertEqual(kernels[role].count('#define GDN_SEQ_BLOCK_VARIANT 0\n'), 1)
            self.assertIn('#define GDN_SEQ_BLOCK_LEVEL 0\n', kernels[role])
            self.assertIn('#define GDN_SEQ_BLOCK_DIAG 0\n', kernels[role])

    def test_the_hash_is_checked_before_anything_is_sliced(self):
        with SyntheticRoot() as root:
            path = root / native.KERNEL_ROOT / seq.NATIVE_COMPUTE
            path.write_bytes(SYNTHETIC_NATIVE.replace('served body', 'served  body').encode())
            with self.assertRaisesRegex(ValueError, 'Native compute hash changed'):
                seq.generate(root)
            with self.assertRaisesRegex(ValueError, 'Native compute hash changed'):
                seq.load_kernels(root, unqualified=True)

    def test_a_moved_anchor_or_entry_point_is_refused(self):
        first = seq.NATIVE_ANCHORS[4]
        cases = (
            (SYNTHETIC_NATIVE.replace(first, first.replace('void ew_off', 'void ew_offset')), 'anchor changed'),
            (SYNTHETIC_NATIVE.replace(seq.NATIVE_ANCHORS[0], seq.NATIVE_ANCHORS[0] + '\n' + seq.NATIVE_ANCHORS[0]),
             'anchor changed'),
            (SYNTHETIC_NATIVE + 'void kernel_main() {\n}\n', 'entry point'),
            (SYNTHETIC_NATIVE.replace('}  // namespace\n\n', '}\n\n'), 'helper namespace'),
        )
        for source, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                seq.prefix(source)
            with SyntheticRoot(source) as root, self.assertRaisesRegex(ValueError, message):
                seq.generate(root)

    def test_builds_differ_only_in_their_header_and_refuse_what_is_not_implemented(self):
        with SyntheticRoot() as root:
            base = seq.generate(root)
            for variant in seq.VARIANTS:
                for diag in seq.DIAGNOSTICS:
                    other = seq.generate(root, variant=variant, diag=diag)
                    header = seq.build_header(0, variant, diag)
                    for role in seq.ROLES:
                        self.assertIn(header, other[role])
                        self.assertEqual(other[role].replace(header, seq.build_header(0, 'A', None)), base[role])
            for level in (1, 2, 15, -1, '0', True):
                with self.assertRaisesRegex(ValueError, 'not implemented'):
                    seq.generate(root, level)
            with self.assertRaisesRegex(ValueError, 'variant'):
                seq.generate(root, variant='B1')
            with self.assertRaisesRegex(ValueError, 'diagnostic'):
                seq.generate(root, diag='profile')

    def test_the_variant_macros_select_the_probe_only_code(self):
        compute = (HERE / seq.SOURCES['compute']).read_text()
        self.assertEqual(compute.count('#if GDN_SEQ_BLOCK_VARIANT == 2'), 3)
        self.assertEqual(seq.VARIANTS.index('N'), 2)
        self.assertIn('static_assert(LEVEL == SB_LEVEL', compute)
        self.assertIn('if constexpr (SB_DIAG != SB_DIAG_NOSNAP)', (HERE / seq.SOURCES['writer']).read_text())

    def test_a_crlf_source_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            for role, name in seq.SOURCES.items():
                text = (HERE / name).read_text()
                Path(directory, name).write_bytes((text.replace('\n', '\r\n') if role == 'writer' else text).encode())
            with SyntheticRoot() as root, self.assertRaisesRegex(ValueError, 'LF-only'):
                seq.generate(root, sources=directory)

    def test_the_new_sources_are_lf_and_never_edit_a_pinned_file(self):
        for name in list(seq.SOURCES.values()) + ['gdn_seq_block.py', 'test_gdn_seq_block.py',
                                                 'gdn_seq_block_device_test.py', 'gdn-seq-block-rig.sh']:
            self.assertNotIn(b'\r', (HERE / name).read_bytes(), name)
        self.assertEqual(native.HASHES[seq.NATIVE_COMPUTE],
                         'b59314e0acaea06b574feffe91a256d3022e819fb5257154e2e72eeb7978b928')

    @unittest.skipUnless(NATIVE_ROOT, 'the pinned native kernels are not in this checkout')
    def test_the_pinned_native_source_slices_and_the_control_is_the_served_build(self):
        prefix = seq.native_prefix(NATIVE_ROOT)
        self.assertEqual(prefix.count('\n'), 364)
        kernels = seq.load_kernels(NATIVE_ROOT, unqualified=True)
        self.assertTrue(kernels['compute'].startswith(prefix))
        import gdn_seq_block_device_test as probe
        control = batch.load_kernels(NATIVE_ROOT)
        self.assertEqual({role: hashlib.sha256(text.encode()).hexdigest() for role, text in control.items()},
                         probe.SERVED_SHA256)


class QualifiedTests(unittest.TestCase):
    def test_qualified_holds_level_0_only(self):
        self.assertEqual(sorted(seq.QUALIFIED), [0])
        self.assertEqual(sorted(seq.QUALIFIED[0]), sorted(seq.ROLES))

    def test_an_unqualified_triple_is_refused_without_the_probes_builder_argument(self):
        with SyntheticRoot() as root:
            with self.assertRaisesRegex(ValueError, 'not qualified'):
                seq.load_kernels(root)
            kernels = seq.load_kernels(root, unqualified=True)
            self.assertFalse(kernels.qualified)
            for value in (1, 'True', None):
                with self.assertRaisesRegex(ValueError, 'explicit bool'):
                    seq.load_kernels(root, unqualified=value)

    def test_a_committed_triple_qualifies_exactly_that_level_and_build(self):
        with SyntheticRoot() as root:
            triple = seq.sha256(seq.generate(root))
            with patch.dict(seq.QUALIFIED, {0: triple}):
                kernels = seq.load_kernels(root)
                self.assertTrue(kernels.qualified)
                for variant, diag in (('A0', None), ('N', None), ('A', 'nosnap'), ('A', 'passthrough')):
                    with self.assertRaisesRegex(ValueError, 'not qualified'):
                        seq.load_kernels(root, variant=variant, diag=diag)
            with patch.dict(seq.QUALIFIED, {0: dict(triple, compute='0' * 64)}):
                with self.assertRaisesRegex(ValueError, 'not qualified'):
                    seq.load_kernels(root)
            # A probe-only build's own triple never qualifies it.
            with patch.dict(seq.QUALIFIED, {0: seq.sha256(seq.generate(root, variant='N'))}):
                with self.assertRaisesRegex(ValueError, 'not qualified'):
                    seq.load_kernels(root, variant='N')

    def test_serving_refuses_the_unqualified_level_before_any_allocation(self):
        fake = FakeTTNN()
        with SyntheticRoot() as root, patch.dict(seq._SERVED, clear=True):
            with self.assertRaisesRegex(ValueError, 'not qualified'):
                seq.served_kernels(root, {})
            with patch.dict(os.environ, {'TT_METAL_HOME': str(root)}):
                with self.assertRaisesRegex(ValueError, 'not qualified'):
                    seq.execute(mesh(), [user_inputs(index) for index in range(4)], fake)
        self.assertEqual(fake.allocated, [])
        self.assertEqual(fake.launches, [])

    def test_plain_source_dicts_are_refused(self):
        fake = FakeTTNN()
        with self.assertRaisesRegex(ValueError, 'must come from gdn_seq_block.load_kernels'):
            seq.execute(mesh(), [user_inputs(0)], fake, kernels=dict(KERNELS))
        self.assertEqual(fake.allocated, [])

    def test_the_served_argument_order_is_refused(self):
        # gdn_user_batch.execute(mesh, users, kernels, operations, ...) swapped in unchanged would put
        # the build where operations goes: refused, never mistaken for a ttnn module.
        for wrong in (build(), dict(KERNELS)):
            with self.assertRaisesRegex(ValueError, 'operations third'):
                seq.execute(mesh(), [user_inputs(0)], wrong)
            with self.assertRaises(TypeError):
                seq.execute(mesh(), [user_inputs(0)], wrong, FakeTTNN())


class AddressFreeFake(FakeTTNN):
    """Interleaved accessors carry no address on device, so the coalesced build merges users."""

    @staticmethod
    def TensorAccessorArgs(value):
        return SimpleNamespace(get_compile_time_args=lambda: [7])


class ParityTests(unittest.TestCase):
    """K5-A's program against the served batched program, over the same recording fake."""

    def run_both(self, users=4, fake_type=FakeTTNN, coalesce=False):
        verify_trace_t1.take()
        cut = verify_trace_t1.cut
        with patch('verify_trace_t1.cut', lambda name: coalesce if name == 'coalesce' else cut(name)):
            served = fake_type()
            groups = [user_inputs(index) for index in range(users)]
            with patch('gdn_multitoken.validate_handoff_runtime'):
                served_out = batch.execute(mesh(), groups, KERNELS, served, output_memory=served.L1_MEMORY_CONFIG)
            served_notes = verify_trace_t1.take()
            candidate = fake_type()
            kernels = build()
            with patch('gdn_multitoken.validate_handoff_runtime') as handoff:
                produced = seq.execute(mesh(), groups, candidate, output_memory=candidate.L1_MEMORY_CONFIG,
                                       kernels=kernels)
            self.assertEqual(handoff.call_count, 1)  # the served runtime pin, checked as served
            candidate_notes = verify_trace_t1.take()
        return served, served_out, served_notes, candidate, produced, candidate_notes, kernels

    def check(self, users, fake_type, coalesce, notes):
        served, served_out, served_notes, candidate, produced, candidate_notes, kernels = self.run_both(
            users, fake_type, coalesce)
        self.assertEqual(served_notes, candidate_notes)
        self.assertEqual(candidate_notes, notes)
        self.assertEqual([(value.shape, value._memory, value.dtype, value.layout) for value in served.allocated],
                         [(value.shape, value._memory, value.dtype, value.layout) for value in candidate.allocated])
        self.assertEqual(len(produced), users)
        self.assertEqual(served.launches[-1][0], candidate.launches[-1][0])  # the generic_op tensor list
        control, program = served.launches[-1][1], candidate.launches[-1][1]
        self.assertEqual(sorted(control), sorted(program))
        for key in control:
            left, right = control[key].kernels, program[key].kernels
            self.assertEqual(len(left), len(right))
            for one, two in zip(left, right):
                self.assertEqual(one.core_ranges, two.core_ranges)
                self.assertEqual(one.config, two.config)
                self.assertEqual(one.source_type, two.source_type)
                self.assertEqual([(x, y, args[0]) for x, y, args in one.runtime_args.flattened()],
                                 [(x, y, args[0]) for x, y, args in two.runtime_args.flattened()])
                role = [role for role in seq.ROLES if two.kernel_source == kernels[role]]
                self.assertEqual(len(role), 1)
                self.assertEqual(two.compile_time_args[:len(seq.compile_args(role[0], kernels))],
                                 seq.compile_args(role[0], kernels))
            unions = {cb[2] for cb in control[key].cbs} | {cb[2] for cb in program[key].cbs}
            self.assertEqual(len(unions), 1)  # every CB of both programs on the one union of cores
            plan = {cb[3][0][1]: (cb[1], cb[3][0][2], cb[3][0][3]) for cb in program[key].cbs}
            expected = {index: (entry[1] * seq.PAGE_BYTES[entry[2]], 'bf16' if entry[2] == 'bf16' else 'fp32',
                                seq.PAGE_BYTES[entry[2]]) for index, entry in seq.CB_PLAN.items()}
            self.assertEqual(plan, expected)

    def test_four_users_per_core_placement_is_the_served_launchs(self):
        self.check(4, FakeTTNN, False, {})
        self.check(1, FakeTTNN, False, {})

    def test_the_coalesced_build_merges_and_falls_back_exactly_as_served(self):
        self.check(4, FakeTTNN, True, {'coalesce_fallback': 1})
        self.check(4, AddressFreeFake, True, {'coalesced': 1})
        served, unused, unused_too, candidate, unused_three, unused_four, kernels = self.run_both(
            4, AddressFreeFake, True)
        chip = candidate.launches[-1][1][((0, 0), (0, 0))]
        self.assertEqual([descriptor.kernel_source for descriptor in chip.kernels],
                         [kernels['reader'], kernels['writer'], kernels['compute']])
        heads = sorted(args[0] for unused, unused_too, args in chip.kernels[0].runtime_args.flattened())
        self.assertEqual(heads, sorted(list(range(24)) * 4))

    def test_the_outputs_are_the_served_shapes_and_placement(self):
        fake = FakeTTNN()
        produced = seq.execute(mesh(), [user_inputs(index) for index in range(4)], fake, kernels=build())
        self.assertEqual([(output.shape, states.shape) for output, states in produced],
                         [((1, 16, 3072), (16, 24, 128, 128))] * 4)
        self.assertEqual({output._memory for output, unused in produced}, {'l1'})
        self.assertEqual({states._memory for unused, states in produced}, {'dram'})
        fake = FakeTTNN()
        produced = seq.execute(mesh(), [user_inputs(0)], fake, output_memory='dram', kernels=build())
        self.assertEqual(produced[0][0]._memory, 'dram')
        with self.assertRaisesRegex(ValueError, 'interleaved DRAM or L1'):
            seq.execute(mesh(), [user_inputs(0)], FakeTTNN(), output_memory='sharded', kernels=build())

    def test_every_users_reader_points_at_that_users_own_buffers(self):
        fake = FakeTTNN()
        seq.execute(mesh(), [user_inputs(index) for index in range(4)], fake, kernels=build())
        chip = fake.launches[-1][1][((0, 0), (0, 0))]
        for user in range(4):
            qkv, beta, gate, initial, z, norm_w = user_inputs(user)
            states = {args[2] for unused, unused_too, args in chip.kernels[3 * user + 1].runtime_args.flattened()}
            self.assertEqual(len(states), 1)  # the writer's states buffer: this user's own
            for unused, unused_too, args in chip.kernels[3 * user].runtime_args.flattened():
                self.assertEqual(args[1:], (qkv.address, beta.address, gate.address, initial.address,
                                            z.address, norm_w.address) + tuple(states))

    def test_the_bisection_build_carries_its_outer_ring(self):
        fake = FakeTTNN()
        seq.execute(mesh(), [user_inputs(0)], fake, kernels=build('A0'))
        indices = {cb[3][0][1] for cb in fake.launches[-1][1][((0, 0), (0, 0))].cbs}
        self.assertEqual(indices, set(range(32)))

    def test_misuse_is_refused_before_any_launch_and_failures_free_everything(self):
        fake = FakeTTNN()
        groups = [list(user_inputs(index)) for index in range(2)]
        groups[1][3] = groups[0][3]
        with self.assertRaisesRegex(ValueError, 'must not share'):
            seq.execute(mesh(), groups, fake, kernels=build())
        self.assertEqual(fake.launches, [])
        self.assertEqual(len(fake.freed), len(fake.allocated))
        fake = FakeTTNN()
        with self.assertRaisesRegex(ValueError, '16-row segments only'):
            seq.execute(mesh(), [user_inputs(0, rows=8)], fake, kernels=build())
        self.assertEqual(fake.allocated, [])
        fake = FakeTTNN()
        with self.assertRaisesRegex(ValueError, 'worker cores required'):
            seq.execute(mesh(8, 8), [user_inputs(index) for index in range(4)], fake, kernels=build())
        self.assertEqual(fake.allocated, [])
        fake = FakeTTNN()
        groups = [list(user_inputs(0))]
        groups[0][0] = FakeTensor('bad', groups[0][0].shape, 777, memory='l1')
        with self.assertRaisesRegex(ValueError, 'interleaved DRAM BF16 TILE'):
            seq.execute(mesh(), groups, fake, kernels=build())
        fake = FakeTTNN()
        fake.generic_op = lambda tensors, program: (_ for _ in ()).throw(RuntimeError('device'))
        with self.assertRaises(RuntimeError):
            seq.execute(mesh(), [user_inputs(index) for index in range(4)], fake, kernels=build())
        self.assertEqual(len(fake.freed), 8)

    def test_a_one_chip_mesh_builds_one_chip_program_for_the_single_card_rig(self):
        fake = FakeTTNN()
        single = SimpleNamespace(shape=(1, 1), compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        fake.get_device_tensors = lambda value: [FakeTensor(value.name + ':0', value.shape, value.address)]
        seq.execute(single, [user_inputs(index) for index in range(4)], fake, kernels=build())
        self.assertEqual(len(fake.launches[-1][1]), 1)
        self.assertEqual(len(fake.launches[-1][1][((0, 0), (0, 0))].kernels), 12)


class HandoffPinTests(unittest.TestCase):
    def test_the_served_runtime_pin_is_checked_before_any_allocation(self):
        fake = FakeTTNN()
        with patch('gdn_multitoken.validate_handoff_runtime') as pin, \
                patch.dict(os.environ, {'TT_METAL_HOME': '/rig/tt-metal'}):
            seq.execute(mesh(), [user_inputs(0)], fake, kernels=build())
        pin.assert_called_once_with(Path('/rig/tt-metal'))
        fake = FakeTTNN()
        with patch('gdn_multitoken.validate_handoff_runtime',
                   side_effect=ValueError('CB handoff runtime changed: llk_io_pack.h')):
            with self.assertRaisesRegex(ValueError, 'handoff runtime changed'):
                seq.execute(mesh(), [user_inputs(0)], fake, kernels=build())
        self.assertEqual(fake.allocated, [])
        self.assertEqual(fake.launches, [])


class AuditLaunchTests(unittest.TestCase):
    """QWEN_FAST_GDN_SEQ_BLOCK_AUDIT runs the served launch beside K5-A's on an audited layer; the T1
    gate counts one 'coalesced' per GDN layer per capture (lever_n_m3native_gate.py:1190-1194)."""

    def coalescing(self):
        return patch('verify_trace_t1.cut', lambda name: name == 'coalesce')

    def test_the_audit_launch_is_the_served_launch_and_is_not_counted(self):
        verify_trace_t1.take()
        groups = [user_inputs(index) for index in range(4)]
        with self.coalescing():
            fake = AddressFreeFake()
            seq.execute(mesh(), groups, fake, output_memory=fake.L1_MEMORY_CONFIG, kernels=build())
            audited = seq.served_audit_launch(mesh(), groups, KERNELS, fake, output_memory=fake.L1_MEMORY_CONFIG)
            self.assertEqual(verify_trace_t1.take(), {'coalesced': 1})  # the layer's own launch only
            self.assertEqual(len(audited), 4)
            served = AddressFreeFake()
            batch.execute(mesh(), groups, KERNELS, served, output_memory=served.L1_MEMORY_CONFIG)
            self.assertEqual(verify_trace_t1.take(), {'coalesced': 1})  # counted when called directly
        # The audit launch is the served program (its own outputs; the same kernels and inputs).
        self.assertEqual(len(fake.launches), 2)
        inputs = [value for group in groups for value in group]
        self.assertEqual([value for value in fake.launches[-1][0] if any(value is kept for kept in inputs)],
                         [value for value in served.launches[-1][0] if any(value is kept for kept in inputs)])
        chip = fake.launches[-1][1][((0, 0), (0, 0))]
        self.assertEqual([kernel.kernel_source for kernel in chip.kernels],
                         [kernel.kernel_source for kernel in served.launches[-1][1][((0, 0), (0, 0))].kernels])

    def test_counts_survive_an_audit_launch_that_raises(self):
        verify_trace_t1.take()
        verify_trace_t1.note('coalesced', 3)
        fake = AddressFreeFake()
        fake.generic_op = lambda tensors, program: (_ for _ in ()).throw(RuntimeError('device'))
        with self.coalescing(), self.assertRaises(RuntimeError):
            seq.served_audit_launch(mesh(), [user_inputs(index) for index in range(4)], KERNELS, fake)
        self.assertEqual(verify_trace_t1.take(), {'coalesced': 3})


class MarkerTests(unittest.TestCase):
    def test_the_marker_text_and_the_gate_regex(self):
        line = seq.marker(48, 48, 0)
        self.assertEqual(line, '[PINDIAG] gdn seq_block calls this captured forward: 48 of 48 GDN layers level=0')
        match = seq.GATE_PATTERN.search('2026-09-25 | INFO | ' + line)
        self.assertEqual(match.groups(), ('48', '48', '0'))
        self.assertEqual(seq.GATE_PATTERN.pattern,
                         r'gdn seq_block calls this captured forward: ([1-9][0-9]*) of ([0-9]+) GDN layers '
                         r'level=([0-9]+)')
        self.assertIsNone(seq.GATE_PATTERN.search(seq.marker(0, 48, 0)))
        self.assertEqual(seq.GATE_PATTERN.search(seq.marker(12, 48, 5)).groups(), ('12', '48', '5'))

    def test_the_audit_line(self):
        line = seq.audit_line(23, 3, 0)
        self.assertEqual(line, '[GDN-SEQ-BLOCK-AUDIT] layer=23 user=3 mismatches=0')
        self.assertEqual(seq.AUDIT_PATTERN.search(line).groups(), ('23', '3', '0'))


class AuditTests(unittest.TestCase):
    def test_the_host_audit(self):
        with SyntheticRoot() as root:
            report = seq.audit(root)
            self.assertEqual(report['generated_sha256'], seq.sha256(seq.generate(root)))
        self.assertFalse(report['qualified'])
        self.assertEqual(report['cb_bytes_per_worker'], 630784)
        self.assertEqual(report['served_cb_bytes_per_worker'], 630784)
        self.assertEqual(report['cb_indices'], list(range(31)))
        self.assertEqual(report['cb_owners'][30]['name'], 'RT')
        self.assertEqual(report['workers'], 96)
        self.assertEqual(report['dram_page_reads_per_worker'], 38)
        self.assertEqual(report['cb_owners'][6]['name'], 'ONES')


class RigTests(unittest.TestCase):
    TEXT = (HERE / 'gdn-seq-block-rig.sh').read_text()

    def test_the_rig_pins_p5_and_mounts_only_the_named_files_at_bench(self):
        self.assertIn('P5=sha256:0fd9ad1f14a4e5d3d4464be55465cb6bb8d52e1b533c430d1c2f7f219df2b0e8', self.TEXT)
        listed = re.search(r'for name in (.*?); do', self.TEXT, re.S).group(1).replace('\\\n', ' ').split()
        self.assertEqual(sorted(listed), sorted(['gdn_seq_block.py', 'gdn_seq_block_compute.cpp',
                                                 'gdn_seq_block_reader.cpp', 'gdn_seq_block_writer.cpp',
                                                 'gdn_seq_block_device_test.py', 'gdn_user_batch.py',
                                                 'verify_trace_t1.py']))
        self.assertIn('dst=/bench/$name,readonly', self.TEXT)
        self.assertNotIn('dst=/experiment-scripts', self.TEXT)
        self.assertNotIn('src=$here,', self.TEXT)  # no directory mount of the checkout

    def test_the_rig_selects_the_card_by_board_id_and_never_resets(self):
        for needed in ('. "$here/qual_card.sh"', 'qual_card_select', 'qual_card_resolve', 'qual_refuse_holders',
                       'qual_card_recheck', '--network none', '-e QWEN_FAST_VERIFY_T1=1',
                       '-e "TT_METAL_CACHE=/tmp/$name-kernel-cache"'):
            self.assertIn(needed, self.TEXT)
        # qual_reset_hint only prints recovery lines (qual_card.sh: "it never resets anything itself").
        code = [line.replace('qual_reset_hint', '') for line in self.TEXT.splitlines()
                if not line.lstrip().startswith('#')]
        for banned in ('tt-smi', 'reset', '/dev/tenstorrent', 'blackhole-'):
            self.assertFalse([line for line in code if banned in line], banned)
        # test_qual_card's order for the scripts/ci runners: holders, then recheck, then the launch.
        launch = re.search(r'^timeout [^\n]*docker run', self.TEXT, flags=re.M).start()
        holders = re.search(r'^qual_refuse_holders$', self.TEXT, flags=re.M).start()
        recheck = re.search(r'^qual_card_recheck\b', self.TEXT, flags=re.M).start()
        self.assertLess(holders, recheck)
        self.assertLess(recheck, launch)
        self.assertNotIn('docker run', self.TEXT[recheck:launch])
        self.assertIn('device=$QUAL_NODE', self.TEXT)

    def test_the_rig_forces_card_b_whatever_the_environment_says(self):
        self.assertLess(self.TEXT.index('ALLOW_SERVING_CARD=0'), self.TEXT.index('\nqual_card_select'))
        refuse = self.TEXT.index('[ "$QUAL_CARD" != "$QUAL_CARD_B" ] || [ "$QUAL_SERVING" != 0 ]')
        self.assertLess(self.TEXT.index('\nqual_card_resolve'), refuse)
        self.assertLess(refuse, self.TEXT.index('\nqual_refuse_holders'))

    @unittest.skipUnless(BASH, 'bash not found')
    def test_the_card_b_pin_refuses_every_other_board(self):
        # The rig's own selection lines, run against the real qual_card.sh, with qual_card_resolve
        # stubbed after sourcing (it would readlink the rig's /dev/tenstorrent).
        start = self.TEXT.index('. "$here/qual_card.sh"')
        block = self.TEXT[start:self.TEXT.index('\nqual_refuse_holders', start)].splitlines()
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'select.sh'

            def select(resolve='QUAL_NODE=/dev/null', **env):
                script.write_bytes(('set -euo pipefail\nhere=%s\n%s\nqual_card_resolve() { %s; }\n%s\n'
                                    'echo "SELECTED $QUAL_CARD $QUAL_SERVING"\n'
                                    % (q(HERE), block[0], resolve, '\n'.join(block[1:]))).encode())
                return run_bash([script.as_posix()], **env)

            chosen = select()
            self.assertEqual(chosen.returncode, 0, chosen.stderr)
            self.assertIn('SELECTED %s 0' % CARD_B, chosen.stdout)
            self.assertIn('SELECTED %s 0' % CARD_B, select(QUAL_CARD=CARD_B).stdout)
            # A serving card is refused even with the caller's override left set.
            refused = select(QUAL_CARD=CARD_M, ALLOW_SERVING_CARD='1')
            self.assertEqual(refused.returncode, 1)
            self.assertIn('refusing', refused.stderr)
            self.assertNotIn('SELECTED', refused.stdout)
            # A board this harness does not name, or card B's id resolving to a serving node.
            other = select(QUAL_CARD='blackhole-0000000000000000')
            self.assertEqual(other.returncode, 2)
            self.assertIn('runs on card B', other.stderr)
            aliased = select(resolve='QUAL_NODE=/dev/null; QUAL_SERVING=1')
            self.assertEqual(aliased.returncode, 2)
            self.assertNotIn('SELECTED', aliased.stdout)

    def test_a_hang_prints_the_recovery_lines_and_the_status_is_kept(self):
        self.assertIn('\nstatus=0\ntimeout -k 30 2700 docker run', self.TEXT)
        self.assertIn('> "$outdir/gdn-seq-block-console.log" 2>&1 || status=$?\n', self.TEXT)
        self.assertNotIn('console.log" 2>&1 || true', self.TEXT)
        tail = self.TEXT[self.TEXT.index('case "$status" in'):]
        self.assertIn('124|137)', tail)
        self.assertEqual(tail.count('qual_reset_hint >&2'), 2)  # a timeout, and any exit but 0 or 1
        self.assertTrue(self.TEXT.endswith('exit "$status"\n'))

    def test_the_probe_names_the_served_control_and_prints_one_summary_line(self):
        text = (HERE / 'gdn_seq_block_device_test.py').read_text()
        self.assertIn("'9512188a", text)
        self.assertIn("'6c56547a", text)
        self.assertIn("'2d62a883", text)
        self.assertIn("summary = dict(kind='gdn-seq-block-probe'", text)
        self.assertIn('print(json.dumps(summary), flush=True)', text)
        self.assertIn('{"kind": "gdn-seq-block-probe"', self.TEXT)


class ProbeHelperTests(unittest.TestCase):
    """The card-B harness's pure helpers: the compare it reports P0 with, and its inputs."""

    @classmethod
    def setUpClass(cls):
        import torch
        import gdn_seq_block_device_test as probe
        cls.torch, cls.probe = torch, probe

    def test_an_exact_compare_and_a_located_mismatch(self):
        torch, probe = self.torch, self.probe
        states = torch.randn(16, 24, 128, 128).bfloat16()
        self.assertEqual(probe.compare(torch, states, states.clone(), probe.locate_states)['exact'], True)
        other = states.clone()
        other[3, 5, 40, 70] = (states[3, 5, 40, 70].view(torch.int16) + 1).view(torch.bfloat16)  # one ulp
        other[9, 1, 0, 0] = float('nan')
        result = probe.compare(torch, states, other, probe.locate_states)
        self.assertFalse(result['exact'])
        self.assertEqual(result['differing'], 2)
        # The first mismatch in token-major order, as the writer lays it out: page 4i + j of (h + 24t).
        self.assertEqual({key: result['first'][key] for key in ('token', 'head', 'tile', 'element')},
                         dict(token=3, head=5, tile=4 * 1 + 2, element=[8, 6]))
        self.assertEqual(result['ulp_histogram']['1'], 1)
        self.assertEqual(result['nan_or_inf_mismatches'], 1)

    def test_output_locations_name_the_head_tile_and_padding(self):
        probe = self.probe
        self.assertEqual(probe.locate_output(5 * 3072 + 300), dict(token=5, padding_row=False, head=2, tile=1,
                                                                   element=[5, 12]))
        self.assertEqual(probe.locate_output(20 * 3072)['padding_row'], True)

    def test_the_reference_holds_the_state_when_nothing_is_written(self):
        torch, probe = self.torch, self.probe
        norm_w, users, poison = probe.regime_inputs(torch, 'R1', 17, 1)
        values = users[0]
        output, states, decay = probe.reference(torch, values['qkv'], torch.zeros(1, 16, 24).bfloat16(),
                                                torch.zeros(1, 16, 24).bfloat16(), values['initial'],
                                                values['z'], norm_w)
        self.assertEqual(tuple(output.shape), (1, 16, 3072))
        self.assertTrue(torch.equal(states[15], values['initial'][0].double()))
        self.assertTrue(torch.equal(decay, torch.ones(16, 24, dtype=torch.float64)))

    def test_the_regimes(self):
        torch, probe = self.torch, self.probe
        for regime in ('R1', 'R2', 'R3', 'R3b', 'R4'):
            norm_w, users, poison = probe.regime_inputs(torch, regime, 17, 4)
            self.assertEqual(len(users), 4)
            self.assertEqual({name: tuple(value.shape) for name, value in users[0].items()},
                             dict(qkv=(1, 16, 5120), beta=(1, 16, 24), gate=(1, 16, 24), initial=(1, 24, 128, 128),
                                  z=(1, 16, 3072)))
            self.assertEqual(any(value is not None for value in poison), regime == 'R3b')
        norm_w, users, poison = probe.regime_inputs(torch, 'R3', 17, 1)
        initial = users[0]['initial'].float()
        self.assertTrue(bool((initial == 2.0 ** -126).any()) and bool((initial == -2.0 ** -133).any())
                        and bool((initial == 2.0 ** -130).any()))
        self.assertTrue(bool((users[0]['initial'].view(torch.int16) == -32768).any()))  # -0.0
        self.assertEqual(float(users[0]['qkv'][0, 3, 1024:2048].abs().max()), 0.0)
        self.assertEqual(set(users[0]['gate'].float().unique().tolist()), {0.0, -88.0})
        self.assertEqual(set(users[0]['beta'].float().unique().tolist()), {0.0, 1 - 2.0 ** -8})
        poison = probe.regime_inputs(torch, 'R3b', 17, 4)[2]
        self.assertTrue(math.isnan(poison[0]) and math.isnan(poison[1]))
        self.assertEqual(poison[2:], [math.inf, -math.inf])

    def test_the_exact_byte_path_tiles_as_the_kernels_index_and_round_trips(self):
        torch, probe = self.torch, self.probe
        value = torch.randn(1, 16, 96).bfloat16()
        value[0, 0, 0], value[0, 1, 1], value[0, 2, 2] = -0.0, float('nan'), 2.0 ** -130
        image = probe.pad_image(torch, value, float('inf'))
        self.assertEqual(tuple(image.shape), (1, 32, 96))
        self.assertTrue(torch.equal(image[:, :16, :].contiguous().view(torch.int16), value.view(torch.int16)))
        self.assertTrue(bool(torch.isinf(image[:, 16:, :]).all()))
        words = probe.tile_image(torch, image)
        self.assertEqual(tuple(words.shape), (3, 512))
        halves = words.view(torch.int16).reshape(3, 1024)
        # Element (r, c) of tile t sits at face (r//16)*2 + c//16, offset (r%16)*16 + c%16.
        for row, column in ((0, 0), (1, 1), (5, 17), (20, 3), (31, 95)):
            tile, local = column // 32, column % 32
            offset = ((row // 16) * 2 + local // 16) * 256 + (row % 16) * 16 + local % 16
            self.assertEqual(int(halves[tile, offset]), int(image.view(torch.int16)[0, row, column]))
        back = probe.untile_image(torch, words, (1, 32, 96))
        self.assertTrue(torch.equal(back.view(torch.int16), image.view(torch.int16)))
        states = torch.randn(2, 3, 64, 128).bfloat16()
        self.assertTrue(torch.equal(probe.untile_image(torch, probe.tile_image(torch, states), states.shape)
                                    .view(torch.int16), states.view(torch.int16)))
        # Page order of the states tensor is (token, head, tile row, tile column): the writer's.
        pages = probe.tile_image(torch, torch.arange(2 * 3 * 64 * 128, dtype=torch.float32)
                                 .reshape(2, 3, 64, 128).bfloat16())
        self.assertEqual(tuple(pages.shape), (2 * 3 * 2 * 4, 512))
        self.assertEqual(probe.padded_shape((1, 16, 24)), (1, 32, 32))
        self.assertEqual(probe.padded_shape((1, 1, 128)), (1, 32, 128))
        unsigned = torch.tensor([0xFFFFFFFF, 0x80000000, 5], dtype=torch.int64)
        self.assertEqual(probe.words_of(torch, unsigned).tolist(), [-1, -2 ** 31, 5])

    def test_the_p1_verdict_thresholds(self):
        verdict = self.probe.p1_verdict
        self.assertEqual(verdict(297.0, 525.8)['label'], 'proceed')
        self.assertEqual(verdict(388.0, 525.8)['label'], 'image-build')
        self.assertEqual(verdict(388.1, 525.8)['label'], 'kill')
        self.assertEqual(verdict(None, 525.8)['label'], 'not-run')
        self.assertEqual(verdict(300.0, None)['label'], 'not-run')
        result = verdict(300.0, 525.8)
        self.assertAlmostEqual(result['a_minus_c_us'], -225.8)
        self.assertAlmostEqual(result['a_over_c'], 300.0 / 525.8)
        self.assertTrue(result['calibrated'])
        # Calibrated C (511 us, -2.8%), A under 388 us but only 123 us under C: the delta line fails.
        self.assertEqual(verdict(388.0, 511.0)['label'], 'kill')
        self.assertEqual(verdict(300.0, 540.0)['label'], 'image-build')  # A - C = -240 but A > 297
        # C outside 525.8 us +-3%: no line applies, and never image-build or proceed.
        for control in (400.0, 509.0, 542.0, 700.0):
            for candidate in (100.0, 297.0, 388.0, 900.0):
                self.assertEqual(verdict(candidate, control)['label'], 'uncalibrated', (candidate, control))
        self.assertEqual(verdict(200.0, 510.1)['label'], 'proceed')  # -2.98%: still calibrated

    def test_the_timing_order_is_serpentine(self):
        self.assertEqual(self.probe.serpentine(['C', 'A', 'A0'], 3),
                         [['C', 'A', 'A0'], ['A0', 'A', 'C'], ['C', 'A', 'A0']])
        stats = self.probe.summarize_timing([4.0, 1.0, 3.0, 2.0])
        self.assertEqual((stats['median_us'], stats['min_us'], stats['max_us'], stats['replays']), (2.5, 1.0, 4.0, 4))


class ProbeVerdictTests(unittest.TestCase):
    """The probe's P0 bookkeeping: a shape mismatch, a missing N, partial coverage or a surviving
    sentinel page must never read as a pass."""

    @classmethod
    def setUpClass(cls):
        import torch
        import gdn_seq_block_device_test as probe
        cls.torch, cls.probe = torch, probe

    EXACT = dict(exact=True, differing=0, differing_bytes=0, of=8)

    def tallies(self, arms=('A', 'A0', 'N'), cases=3):
        probe, tallies = self.probe, {}
        for arm in arms:
            tallies[arm] = probe.new_tally()
            for case in range(cases):
                part = self.EXACT if arm != 'N' else dict(exact=False, differing=5, differing_bytes=9, of=8)
                probe.record(tallies[arm], 'case%d' % case, dict(user0_output=part, user0_states=self.EXACT))
        return tallies

    def full(self):
        probe = self.probe
        return probe.plan_coverage(list(probe.ARMS), list(probe.REGIMES), list(probe.PLAN_R1_SEEDS), 4, 128, 128)

    def verdict(self, tallies, coverage=None, errors=(), inputs_unchanged=True, unwritten=()):
        return self.probe.p0_verdict(tallies, errors=list(errors), inputs_unchanged=inputs_unchanged,
                                     unwritten=list(unwritten), coverage=coverage or self.full())

    def test_the_plans_defaults_are_the_whole_plan_and_pass(self):
        probe = self.probe
        arguments = probe.parse(['--out', 'x.json'])
        coverage = probe.plan_coverage(arguments.arms, arguments.regimes, arguments.seeds, arguments.users,
                                       arguments.r4_launches, arguments.r4_launches)
        self.assertEqual(coverage, dict(complete=True, missing=[]))
        result = self.verdict(self.tallies(), coverage)
        self.assertEqual((result['result'], result['passed'], result['may_commit_qualified']), ('pass', True, True))
        self.assertEqual(result['exact'], dict(A=True, A0=True))
        self.assertTrue(result['n_detects'])

    def test_a_shape_mismatch_is_unmeasured_and_fails(self):
        torch, probe = self.torch, self.probe
        part = probe.compare(torch, torch.zeros(2, 3).bfloat16(), torch.zeros(3, 2).bfloat16(), probe.locate_states)
        self.assertEqual((part['exact'], part['differing_bytes']), (False, None))
        tallies = self.tallies()
        self.assertFalse(probe.record(tallies['A'], 'bad', dict(user0_output=self.EXACT, user0_states=part)))
        self.assertEqual(tallies['A']['differing_bytes'], 0)  # adds no bytes - and still fails
        self.assertEqual(tallies['A']['unmeasured'], 1)
        self.assertEqual(tallies['A']['first_failure']['case'], 'bad')
        result = self.verdict(tallies)
        self.assertEqual(result['result'], 'fail')
        self.assertEqual(result['exact']['A'], False)
        # A dtype mismatch, and a case with no parts at all, likewise.
        other = probe.compare(torch, torch.zeros(4).bfloat16(), torch.zeros(4), probe.locate_states)
        self.assertIsNone(other['differing_bytes'])
        empty = probe.new_tally()
        self.assertFalse(probe.record(empty, 'nothing', {}))
        self.assertFalse(probe.arm_exact(empty))

    def test_n_must_run_and_must_see_a_difference(self):
        probe = self.probe
        result = self.verdict(self.tallies(('A', 'A0')),
                              probe.plan_coverage(['C', 'A', 'A0'], list(probe.REGIMES), list(probe.PLAN_R1_SEEDS),
                                                  4, 128, 128))
        self.assertEqual(result['result'], 'partial-pass')
        self.assertIsNone(result['n_detects'])
        self.assertIn('arm N', result['coverage']['missing'])
        blind = self.tallies(('A', 'A0'))
        blind['N'] = probe.new_tally()
        probe.record(blind['N'], 'case0', dict(user0_output=self.EXACT))
        self.assertEqual(self.verdict(blind)['result'], 'fail')
        self.assertIs(self.verdict(blind)['n_detects'], False)
        # And an N whose only 'difference' was unmeasured detects nothing.
        unmeasured = self.tallies(('A', 'A0'))
        unmeasured['N'] = probe.new_tally()
        probe.record(unmeasured['N'], 'case0', dict(user0_output=dict(exact=False, differing=None, differing_bytes=None)))
        self.assertIs(self.verdict(unmeasured)['n_detects'], False)

    def test_partial_coverage_is_labelled_and_never_licenses_a_qualified_triple(self):
        probe = self.probe
        for arms, regimes, seeds, users, launches, completed, missing in (
                (probe.ARMS, ['R1'], [17], 4, 128, 0, 'regime R2'),
                (probe.ARMS, probe.REGIMES, [17, 23], 4, 128, 128, 'R1 seed 29'),
                (probe.ARMS, probe.REGIMES, probe.PLAN_R1_SEEDS, 1, 128, 128, 'users 1 (plan: 4)'),
                (probe.ARMS, probe.REGIMES, probe.PLAN_R1_SEEDS, 4, 3, 3, 'R4 3 of 3 launches completed (plan: 128)'),
                (probe.ARMS, probe.REGIMES, probe.PLAN_R1_SEEDS, 4, 128, 40, 'R4 40 of 128 launches completed (plan: 128)')):
            coverage = probe.plan_coverage(list(arms), list(regimes), list(seeds), users, launches, completed)
            self.assertFalse(coverage['complete'])
            self.assertIn(missing, coverage['missing'])
            result = self.verdict(self.tallies(), coverage)
            self.assertEqual((result['result'], result['passed'], result['may_commit_qualified']),
                             ('partial-pass', False, False))
        # A miss is a fail whatever the coverage; no exact arm at all is never a pass.
        bad = self.tallies()
        probe.record(bad['A0'], 'r4', dict(user0_states=dict(exact=False, differing=1, differing_bytes=1, of=8)))
        self.assertEqual(self.verdict(bad, coverage)['result'], 'fail')
        self.assertEqual(self.verdict(self.tallies(('N',)))['result'], 'fail')

    def test_a_surviving_sentinel_page_moved_input_or_error_fails(self):
        unwritten = [dict(arm='A', case='R1/seed17', user=0, chip=0, tensor='states', pages=16)]
        self.assertEqual(self.verdict(self.tallies(), unwritten=unwritten)['result'], 'fail')
        self.assertEqual(self.verdict(self.tallies(), inputs_unchanged=False)['result'], 'fail')
        self.assertEqual(self.verdict(self.tallies(), errors=['R4'])['result'], 'error')

    def test_the_traced_status_and_the_overall_result(self):
        probe = self.probe
        self.assertEqual(probe.traced_status(None), 'not-run')
        self.assertEqual(probe.traced_status(dict(error='RuntimeError()', exact={})), 'error')
        self.assertEqual(probe.traced_status(dict(error=None, exact=dict(A=True, A0=True), unwritten=[])), 'exact')
        self.assertEqual(probe.traced_status(dict(error=None, exact=dict(A=True, A0=False), unwritten=[])), 'differs')
        self.assertEqual(probe.traced_status(dict(error=None, exact={}, unwritten=[])), 'differs')
        self.assertEqual(probe.traced_status(dict(error=None, exact=dict(A=True), unwritten=[dict(pages=1)])), 'differs')
        differs = dict(error=None, exact=dict(A=False), unwritten=[])
        self.assertEqual(probe.overall('pass', differs), 'fail')
        self.assertEqual(probe.overall('partial-pass', differs), 'fail')
        self.assertEqual(probe.overall('pass', dict(error='x', exact={})), 'error')
        self.assertEqual(probe.overall('pass', None), 'pass')
        self.assertEqual(probe.overall('fail', dict(error=None, exact=dict(A=True), unwritten=[])), 'fail')

    def test_unwritten_pages_are_whole_sentinel_pages(self):
        torch, probe = self.torch, self.probe
        states = torch.randn(2, 3, 64, 128).bfloat16()
        self.assertEqual(probe.unwritten_pages(torch, states), 0)
        words = probe.tile_image(torch, states)
        words[5] = probe.SENTINEL_WORD
        words[7, :511] = probe.SENTINEL_WORD  # one word short of a whole page: written
        image = probe.untile_image(torch, words, states.shape)
        self.assertEqual(probe.unwritten_pages(torch, image), 1)
        self.assertEqual(probe.SENTINEL_WORD, 0x7FC17FC1)
        self.assertTrue(math.isnan(float(torch.tensor([probe.SENTINEL_HALF], dtype=torch.int16).view(torch.bfloat16))))
        self.assertEqual(probe.MAX_PAGES, probe.page_count((16, 24, 128, 128)))
        self.assertEqual(probe.page_count((1, 16, 3072)), 96)

    def test_every_launch_allocation_is_filled_before_the_launch_and_freed_on_failure(self):
        probe = self.probe
        fake, filled = FakeTTNN(), []
        fake.generic_op = lambda tensors, program: fake.launches.append((tensors, program, list(filled)))
        operations = probe.SentinelOperations(fake, filled.append)
        produced = seq.execute(mesh(), [user_inputs(index) for index in range(4)], operations, kernels=build())
        self.assertEqual(operations.filled, 8)
        self.assertEqual(filled, fake.allocated)
        self.assertEqual(fake.launches[-1][2], fake.allocated)  # every fill happened before the launch
        self.assertEqual([value for pair in produced for value in pair], fake.allocated)
        self.assertIs(operations.L1_MEMORY_CONFIG, fake.L1_MEMORY_CONFIG)
        # The served launch takes it in the same slot.
        served, marks = FakeTTNN(), []
        batch.execute(mesh(), [user_inputs(0)], KERNELS, probe.SentinelOperations(served, marks.append))
        self.assertEqual(marks, served.allocated)
        # A fill that fails frees that tensor and every one before it.
        broken, calls = FakeTTNN(), []

        def fill(value):
            calls.append(value)
            if len(calls) == 3:
                raise RuntimeError('fill')

        with self.assertRaisesRegex(RuntimeError, 'fill'):
            seq.execute(mesh(), [user_inputs(index) for index in range(4)], probe.SentinelOperations(broken, fill),
                        kernels=build())
        self.assertEqual(sorted(broken.freed), sorted(value.name for value in broken.allocated))
        self.assertEqual(broken.launches, [])


# ---------------------------------------------------------------------------------------------
# The wiring (K5 plan 3.1, 3.9, 5 G-CPU): gdn_user_batch_conv's dispatch, the model_batch counter
# and marker, the audit, the arm, the gate, the copy lists and the CPU allowlist.
# ---------------------------------------------------------------------------------------------

ROOT = HERE.parent.parent
# The wiring's parent: the K5-A files committed, nothing wired. Flag off, every module this change
# touches must be that commit's, call for call.
PARENT = 'e7bfd324'
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'
ARM = HERE / 'lever_n_m3native_run_arm.sh'
TRIPLE = dict(reader='784deb4e398cfaa5a7c2b415c1a8be71925ccfd61fe18d801b11fb138dbc45c0',
              writer='215ef7da5d01d5c5fa720e8f6218481f27491c075151140376b457d760e1870d',
              compute='a9782f1434673d05fb7270b810836101ef38b51becfbd5c13d67880a937deffe')


def clean_environment(**flags):
    """No QWEN_FAST_* flag but the ones given."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environment.update(flags)
    return patch.dict(os.environ, environment, clear=True)


def parent_module(relative, name):
    from test_dflash_proposal_trace import pinned_module

    module = pinned_module(relative, name, commit=PARENT)
    if module is None:
        raise unittest.SkipTest('no git history for %s' % PARENT)
    return module


def plain(value, operations=None):
    """A result or call argument as something two runs compare: fakes by name, containers by element,
    `operations` (a fixture's own fake) by role, a DeviceLoopState by its type."""
    if operations is not None and value is operations:
        return 'operations'
    if type(value).__name__ == 'DeviceLoopState':
        return 'DeviceLoopState'
    if isinstance(value, dict):
        return {key: plain(item, operations) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item, operations) for item in value]
    if isinstance(value, tuple):
        return tuple(plain(item, operations) for item in value)
    return getattr(value, 'name', value)


def calls_of(mock, operations=None):
    """A mock's calls as (args, kwargs) pairs through plain()."""
    return [(plain(tuple(call.args), operations), plain(dict(call.kwargs), operations)) for call in mock.call_args_list]


class QualifiedLevelTests(unittest.TestCase):
    def test_qualified_holds_the_card_b_level_0_variant_a_triple_only(self):
        self.assertEqual(seq.QUALIFIED, {0: TRIPLE})

    @unittest.skipUnless(NATIVE_ROOT, 'the pinned native kernels are not in this checkout')
    def test_the_pinned_sources_generate_exactly_the_qualified_triple(self):
        self.assertEqual(seq.sha256(seq.generate(NATIVE_ROOT)), TRIPLE)
        kernels = seq.load_kernels(NATIVE_ROOT)
        self.assertTrue(kernels.qualified)
        self.assertEqual((kernels.level, kernels.variant, kernels.diag), (0, 'A', None))
        for variant, diag in (('A0', None), ('N', None), ('A', 'nosnap'), ('A', 'passthrough')):
            with self.subTest(variant=variant, diag=diag), self.assertRaisesRegex(ValueError, 'not qualified'):
                seq.load_kernels(NATIVE_ROOT, variant=variant, diag=diag)
        with patch.dict(seq._SERVED, clear=True):
            self.assertIs(seq.served_kernels(NATIVE_ROOT, {}), seq.served_kernels(NATIVE_ROOT, {}))
            with self.assertRaisesRegex(ValueError, 'not implemented'):
                seq.served_kernels(NATIVE_ROOT, {seq.LEVEL_FLAG: '1'})


class DispatchTests(unittest.TestCase):
    """gdn_user_batch_conv.run_user_batched_projected: K5-A only with the flag and only when every
    packed user is a 16-row segment; flag off, the parent's calls exactly."""

    def launch(self, module=None, widths=(16, 16, 16, 16), seq_execute=None, audit=None):
        import gdn_user_batch_conv
        from test_gdn_user_batch_conv import fake_operations, user_group

        module = module or gdn_user_batch_conv
        calls = []
        operations = fake_operations(calls)
        groups = [user_group(index, operations, rows) for index, rows in enumerate(widths)]
        taps = [operations.make('tap%d' % tap, (1, 1, 5120)) for tap in range(4)]
        extras = [operations.make('dt', (1, 1, 24)), operations.make('nega', (1, 1, 24)),
                  operations.make('norm_w', (1, 1, 128))]

        def windows(mesh, projected, history):
            calls.append(('windows', projected.name, tuple(value.name for value in history)))
            return [operations.make('window:%s.%d' % (projected.name, slot), (1, projected.shape[1], 5120))
                    for slot in range(4)]

        def launched(tag):
            def run(mesh, inputs, *args, **kwargs):
                calls.append((tag, plain(inputs), plain(args, operations), plain(kwargs, operations)))
                return [(operations.make('%s-out%d' % (tag, index), (1, user[0].shape[1], 3072)),
                         operations.make('%s-prefix%d' % (tag, index), (user[0].shape[1], 24, 128, 128)))
                        for index, user in enumerate(inputs)]
            return run

        build = SimpleNamespace(level=0, name='k5a-build')
        with patch('gdn_conv_windows.build_windows', side_effect=windows), \
                patch('gdn_user_batch.execute', side_effect=launched('served')) as served, \
                patch('gdn_seq_block.execute', side_effect=seq_execute or launched('k5a')) as candidate, \
                patch('gdn_seq_block.served_kernels', return_value=build), \
                patch('gdn_seq_block.audit_launches', side_effect=audit) as audited:
            self.mocks = (served, candidate, audited)
            results = module.run_user_batched_projected('mesh', groups, taps, *extras, 'kernels', operations)
        return calls, operations, results, served, candidate, audited

    def test_flag_off_the_dispatch_is_the_parents_call_for_call(self):
        parent = parent_module('gdn_user_batch_conv.py', 'gdn_user_batch_conv_k5a_parent')
        for environ in ({}, {seq.FLAG: '0'}, {seq.FLAG: '0', batch.FLAG: '1', seq.AUDIT_FLAG: '0,23'}):
            for widths in ((16, 16, 16, 16), (16, 8)):
                with self.subTest(environ=environ, widths=widths), clean_environment(**environ):
                    mine = self.launch(widths=widths)
                    theirs = self.launch(parent, widths=widths)
                    self.assertEqual(mine[0], theirs[0])                       # every call, in order
                    self.assertEqual(plain(mine[2]), plain(theirs[2]))         # every result, key for key
                    self.assertEqual(mine[3].call_count, 1)
                    mine[4].assert_not_called()
                    mine[5].assert_not_called()
                    for result in mine[2]:
                        self.assertNotIn('seq_block', result)
                        self.assertNotIn('seq_block_audit', result)

    def test_flag_on_every_16_row_block_takes_k5a_with_the_served_calls_inputs(self):
        with clean_environment():
            served_calls = self.launch()[0]
        with clean_environment(**{seq.FLAG: '1', batch.FLAG: '1'}):
            calls, operations, results, served, candidate, audited = self.launch()
        served.assert_not_called()
        self.assertEqual(candidate.call_count, 1)
        k5a = [entry for entry in calls if entry[0] == 'k5a']
        served_entry = [entry for entry in served_calls if entry[0] == 'served']
        self.assertEqual(k5a[0][1], served_entry[0][1], 'the same per-user input tuples')
        self.assertEqual(k5a[0][2], ('operations',))
        self.assertEqual(k5a[0][3], dict(output_memory=None, kernels='k5a-build'))
        # Everything before the launch is the served path's, call for call.
        self.assertEqual(calls[:calls.index(k5a[0])], served_calls[:served_calls.index(served_entry[0])])
        for index, result in enumerate(results):
            self.assertIs(result['seq_block'], True)
            self.assertEqual(result['seq_block_level'], 0)
            self.assertEqual(result['output'].name, 'k5a-out%d' % index)
            self.assertIn(result['states'], result['owned'])
            self.assertTrue(result['user_batched'] and not result['norm_batch'])
            self.assertNotIn('seq_block_audit', result)
        audited.assert_not_called()
        self.assertEqual((seq.RESULT_KEY, seq.LEVEL_KEY, seq.AUDIT_KEY),
                         ('seq_block', 'seq_block_level', 'seq_block_audit'))

    def test_any_other_width_keeps_the_served_launch(self):
        for widths in ((16, 8), (8, 8, 8, 8), (16, 16, 16, 32), (32,)):
            with self.subTest(widths=widths), clean_environment(**{seq.FLAG: '1', batch.FLAG: '1'}):
                calls, operations, results, served, candidate, audited = self.launch(widths=widths)
                self.assertEqual(served.call_count, 1)
                candidate.assert_not_called()
                self.assertTrue(all('seq_block' not in result for result in results))

    def test_the_flag_without_the_user_batch_or_a_bad_value_raises_before_any_launch(self):
        for environ, message in (({seq.FLAG: '1'}, 'requires QWEN_FAST_GDN_USER_BATCH=1'),
                                 ({seq.FLAG: 'yes', batch.FLAG: '1'}, 'must be 0 or 1')):
            released = []
            with self.subTest(environ=environ), clean_environment(**environ), \
                    patch('gdn_user_batch_conv.release_owned', side_effect=lambda ops, values: released.extend(values)):
                with self.assertRaisesRegex(ValueError, message):
                    self.launch()
                for mock in self.mocks:
                    mock.assert_not_called()
                # every user's windows and conv outputs, built before the dispatch, are released
                names = {value.name for value in released}
                self.assertEqual(len([name for name in names if name.startswith('window:')]), 16)
                self.assertEqual(len([name for name in names if name.startswith('conv:')]), 4)

    def test_an_audited_layer_holds_the_audit_launches_outside_owned(self):
        held = []

        def audit(mesh, inputs, served, layer, outputs, operations):
            held.append((plain(inputs), served, layer, [output.name for output in outputs]))
            return [dict(layer=layer, output=operations.make('audit-out%d' % index, (1, 16, 3072)),
                         served_output=operations.make('served-out%d' % index, (1, 16, 3072)),
                         served_states=operations.make('served-prefix%d' % index, (16, 24, 128, 128)))
                    for index in range(len(inputs))]

        environ = {seq.FLAG: '1', batch.FLAG: '1', seq.AUDIT_FLAG: '0,23,47'}
        with clean_environment(**environ), seq.audit_scope(23):
            calls, operations, results, served, candidate, audited = self.launch(audit=audit)
        self.assertEqual(len(held), 1)
        k5a = [entry for entry in calls if entry[0] == 'k5a'][0]
        self.assertEqual(held[0], (k5a[1], 'kernels', 23, ['k5a-out%d' % index for index in range(4)]),
                         "the same inputs, the served build, the model's own K5-A outputs")
        for index, result in enumerate(results):
            hold = result['seq_block_audit']
            self.assertEqual(hold['layer'], 23)
            self.assertEqual((hold['output'].name, hold['served_output'].name, hold['served_states'].name),
                             ('audit-out%d' % index, 'served-out%d' % index, 'served-prefix%d' % index))
            owned = {value.name for value in result['owned']}
            self.assertFalse(owned & {'audit-out%d' % index, 'served-out%d' % index, 'served-prefix%d' % index})
        self.assertEqual(seq.audit_held_of(dict(results[0], segment_results=tuple(results))),
                         [value for result in results for value in (result['seq_block_audit']['output'],
                                                                     result['seq_block_audit']['served_output'],
                                                                     result['seq_block_audit']['served_states'])])
        # A layer the list does not name, no scope at all, or the audit without the flag: no audit.
        for scope, flags in ((5, environ), (None, environ), (23, {seq.AUDIT_FLAG: '23'})):
            with self.subTest(scope=scope, flags=flags), clean_environment(**flags), \
                    (seq.audit_scope(scope) if scope is not None else nullcontext()):
                calls, operations, results, served, candidate, audited = self.launch()
                audited.assert_not_called()
                self.assertTrue(all('seq_block_audit' not in result for result in results))

    def test_a_failed_audit_releases_the_layers_outputs(self):
        released = []

        def explode(*args, **kwargs):
            raise RuntimeError('audit launch')

        with clean_environment(**{seq.FLAG: '1', batch.FLAG: '1', seq.AUDIT_FLAG: '3'}), seq.audit_scope(3), \
                patch('gdn_user_batch_conv.release_owned', side_effect=lambda ops, values: released.extend(values)):
            with self.assertRaisesRegex(RuntimeError, 'audit launch'):
                self.launch(audit=explode)
        names = {value.name for value in released}
        for index in range(4):
            self.assertIn('k5a-out%d' % index, names)
            self.assertIn('k5a-prefix%d' % index, names)


class AuditScopeTests(unittest.TestCase):
    def test_the_scope_names_the_layer_and_restores_the_previous_one(self):
        environ = {seq.AUDIT_FLAG: '0,23,47'}
        self.assertIsNone(seq.audit_layer(environ))
        with seq.audit_scope(23):
            self.assertEqual(seq.audit_layer(environ), 23)
            with seq.audit_scope(5):
                self.assertIsNone(seq.audit_layer(environ), 'not listed')
            self.assertEqual(seq.audit_layer(environ), 23)
            with self.assertRaisesRegex(RuntimeError, 'decode'):
                with seq.audit_scope(47):
                    raise RuntimeError('decode')
            self.assertEqual(seq.audit_layer(environ), 23)
        self.assertIsNone(seq.audit_layer(environ))
        for layer in (48, -1, '3', None, 2.0):
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, 'GDN layer'):
                with seq.audit_scope(layer):
                    pass

    def test_the_audit_is_active_only_with_the_flag(self):
        self.assertEqual(seq.audit_active({}), ())
        self.assertEqual(seq.audit_active({seq.AUDIT_FLAG: '0,23,47'}), ())
        self.assertEqual(seq.audit_active({seq.FLAG: '1', batch.FLAG: '1', seq.AUDIT_FLAG: '0,23,47'}), (0, 23, 47))
        self.assertEqual(seq.audit_active({seq.FLAG: '1', batch.FLAG: '1'}), ())
        with self.assertRaisesRegex(ValueError, 'distinct GDN layers'):
            seq.audit_active({seq.FLAG: '1', batch.FLAG: '1', seq.AUDIT_FLAG: '0,0'})
        with self.assertRaisesRegex(ValueError, 'requires'):
            seq.audit_active({seq.FLAG: '1', seq.AUDIT_FLAG: '0'})


class CloningFake(AddressFreeFake):
    """AddressFreeFake with ttnn.clone: a new tensor in `memory_config`, recorded in `clones`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clones = []

    def clone(self, value, memory_config=None):
        copy = self.empty(value.shape, dtype=value.dtype, layout=value.layout, memory_config=memory_config)
        self.clones.append((value, copy))
        return copy


class AuditLaunchPairTests(unittest.TestCase):
    """audit_launches: a DRAM copy of each of the model's own K5-A outputs, then the served launch on
    the same inputs with its outputs in DRAM, not counted by the T1 gate. Nothing launches K5-A again."""

    def launch_layer(self, fake, groups):
        return seq.execute(mesh(), groups, fake, output_memory=fake.L1_MEMORY_CONFIG, kernels=build())

    def test_the_models_own_outputs_are_copied_and_the_served_launch_runs_uncounted(self):
        verify_trace_t1.take()
        groups = [user_inputs(index) for index in range(4)]
        with patch('verify_trace_t1.cut', lambda name: name == 'coalesce'):
            fake = CloningFake()
            produced = self.launch_layer(fake, groups)
            held = seq.audit_launches(mesh(), groups, KERNELS, 23, [output for output, states in produced], fake)
            self.assertEqual(verify_trace_t1.take(), {'coalesced': 1}, 'the layer launch only')
        self.assertEqual(len(fake.launches), 2, 'the layer launch and the served launch; no second K5-A launch')
        inputs = [value.name for group in groups for value in group]
        self.assertEqual([name for name in fake.launches[1][0] if name in inputs],
                         [name for name in fake.launches[0][0] if name in inputs])
        self.assertEqual({kernel.kernel_source for kernel in fake.launches[1][1][((0, 0), (0, 0))].kernels},
                         set(KERNELS.values()))
        self.assertEqual([source for source, copy in fake.clones], [output for output, states in produced])
        self.assertEqual({copy._memory for source, copy in fake.clones}, {'dram'})
        self.assertEqual({output._memory for output, states in produced}, {'l1'})
        self.assertEqual(fake.freed, [])
        self.assertEqual(len(held), 4)
        served = fake.allocated[12:]
        self.assertEqual({value._memory for value in served}, {'dram'})
        for index, hold in enumerate(held):
            self.assertEqual(hold['layer'], 23)
            self.assertIs(hold['output'], fake.clones[index][1])
            self.assertIsNot(hold['output'], produced[index][0])
            self.assertIs(hold['served_output'], served[2 * index])
            self.assertIs(hold['served_states'], served[2 * index + 1])
            self.assertEqual(hold['served_states'].shape, (16, 24, 128, 128))

    def test_a_failure_frees_everything_it_allocated_and_nothing_of_the_models(self):
        groups = [user_inputs(index) for index in range(4)]
        for failing in ('clone', 'launch', 'alias'):
            with self.subTest(failing=failing):
                fake = CloningFake()
                produced = self.launch_layer(fake, groups)
                before = len(fake.allocated)
                if failing == 'launch':
                    fake.generic_op = lambda tensors, program: (_ for _ in ()).throw(RuntimeError('device'))
                else:
                    original = fake.clone

                    def clone(value, memory_config=None, original=original, failing=failing):
                        if len(fake.clones) == 2:
                            if failing == 'alias':
                                return value
                            raise RuntimeError('device')
                        return original(value, memory_config=memory_config)

                    fake.clone = clone
                with self.assertRaisesRegex((RuntimeError, AssertionError), 'device|new tensor'):
                    seq.audit_launches(mesh(), groups, KERNELS, 0, [output for output, states in produced], fake)
                self.assertEqual(sorted(fake.freed), sorted(value.name for value in fake.allocated[before:]))
                self.assertGreater(len(fake.freed), 0)
        with self.assertRaisesRegex(ValueError, 'One K5-A output per audited user'):
            seq.audit_launches(mesh(), groups, KERNELS, 0, [], CloningFake())


class ComparisonFake:
    """to_torch and get_device_tensors over per-chip torch payloads (`chips`)."""

    @staticmethod
    def get_device_tensors(value):
        return list(value.chips)

    @staticmethod
    def to_torch(value):
        return value


class AuditRoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def tensor(self, shape, seed):
        generator = self.torch.Generator().manual_seed(seed)
        return SimpleNamespace(chips=[self.torch.randn(shape, generator=generator).bfloat16() for chip in range(2)])

    def copy(self, value):
        return SimpleNamespace(chips=[chip.clone() for chip in value.chips])

    def record(self, layer, users=4, held=True, flip=None):
        pieces = []
        for user in range(users):
            states = self.tensor((16, 24, 4, 8), 100 * layer + user)
            output = self.tensor((1, 16, 32), 1000 + 100 * layer + user)
            hold = dict(layer=layer, output=output, served_output=self.copy(output), served_states=self.copy(states))
            if flip and flip[0] == user:
                target = hold['served_states'] if flip[1] == 'states' else hold['served_output']
                bits = target.chips[1].view(self.torch.int16)
                bits.view(-1)[7] ^= 1
            pieces.append(dict(states=states, seq_block=True, seq_block_level=0,
                               **({'seq_block_audit': hold} if held else {})))
        result = dict(pieces[0], segment_results=tuple(pieces))
        return ('state', result, 'checkpoint')

    def run_round(self, records, layers=(0, 2), round_number=1):
        lines = []
        with patch('gdn_seq_block.log_line', side_effect=lines.append):
            try:
                return seq.audit_round(ComparisonFake(), records, layers, round_number), lines
            except AssertionError as error:
                return error, lines

    def test_an_exact_round_logs_one_zero_line_per_layer_and_user(self):
        records = [self.record(layer) for layer in range(3)]
        compared, lines = self.run_round(records)
        self.assertEqual(compared, 8)
        self.assertEqual(lines, [seq.audit_line(layer, user, 0, round=1, output=0, states=0)
                                 for layer in (0, 2) for user in range(4)])
        self.assertEqual([seq.AUDIT_PATTERN.search(line).groups() for line in lines][:2],
                         [('0', '0', '0'), ('0', '1', '0')])

    def test_a_differing_element_is_counted_logged_and_raised_after_the_whole_round(self):
        for where in ('states', 'output'):
            with self.subTest(where=where):
                records = [self.record(0), self.record(1), self.record(2, flip=(2, where))]
                error, lines = self.run_round(records)
                self.assertIsInstance(error, AssertionError)
                self.assertIn('1 differing elements', str(error))
                self.assertEqual(len(lines), 8, 'every line of the round is logged first')
                self.assertEqual(lines[6], seq.audit_line(2, 2, 1, round=1, output=int(where == 'output'),
                                                          states=int(where == 'states')))
                import lever_n_m3native_gate as gate
                report = gate.gdn_seq_block_report({seq.FLAG: '1', batch.FLAG: '1', seq.AUDIT_FLAG: '0,2'}, 4,
                                                   '\n'.join(lines + [seq.marker(48, 48, 0)]))
                self.assertEqual(len(report['problems']), 1)
                self.assertIn('mismatches', report['problems'][0])

    def test_an_audit_that_holds_nothing_or_names_another_layer_is_not_a_pass(self):
        for records, message in (([self.record(0, held=False)], 'holds no K5-A audit launch'),
                                 ([self.record(5)], 'holds no K5-A audit launch'),
                                 ([self.record(0)] * 2, 'retains 2 GDN layers')):
            with self.subTest(message=message):
                error, lines = self.run_round(records, layers=(0,) if message != 'retains 2 GDN layers' else (7,))
                self.assertIsInstance(error, AssertionError)
                self.assertIn(message, str(error))
                self.assertTrue(lines and message in lines[-1])

    def test_a_shape_or_chip_difference_counts(self):
        torch = self.torch
        left = SimpleNamespace(chips=[torch.zeros(2, 3).bfloat16()] * 2)
        self.assertEqual(seq.differing(ComparisonFake(), left, SimpleNamespace(chips=[torch.zeros(3, 2).bfloat16()] * 2)), 12)
        self.assertEqual(seq.differing(ComparisonFake(), left, SimpleNamespace(chips=[torch.zeros(2, 3).bfloat16()])), 1)
        minus = SimpleNamespace(chips=[torch.full((2, 3), -0.0).bfloat16()] * 2)
        self.assertEqual(seq.differing(ComparisonFake(), left, minus), 12, 'bit patterns, not values')
        none = SimpleNamespace(chips=[])
        self.assertEqual(seq.differing(ComparisonFake(), none, none), 1, 'no chip compared is not a match')
        empty = SimpleNamespace(chips=[torch.zeros(0, 3).bfloat16()] * 2)
        self.assertEqual(seq.differing(ComparisonFake(), empty, empty), 2, 'an empty readback is not a match')

    def test_a_side_compared_with_itself_or_with_the_freed_l1_output_is_not_a_pass(self):
        for alias in ('output', 'states', 'freed'):
            with self.subTest(alias=alias):
                record = self.record(0)
                piece = record[1]['segment_results'][1]
                hold = piece['seq_block_audit']
                if alias == 'output':
                    hold['served_output'] = hold['output']
                elif alias == 'states':
                    hold['served_states'] = piece['states']
                else:
                    piece['output'] = hold['output']
                error, lines = self.run_round([record], layers=(0,))
                self.assertIsInstance(error, AssertionError)
                self.assertIn('GDN layer 0 user 1 compares a tensor with itself', str(error))
                self.assertFalse(any(seq.AUDIT_PATTERN.search(line) for line in lines), 'no line reads as a pass')


class ModelBatchWiringTests(unittest.TestCase):
    """model_batch: the counter and the level come from the result dict, the marker is one line per
    captured forward, the audit scope names the GDN layer, and the audit holds are released; flag
    off, gdn_forward and run are the parent's, call for call."""

    def fixture(self, module, retained=True, audit=()):
        import torch
        from model_batch import validate_pack

        packs = [dict(start=100 * (index + 1), rows=16, pages=torch.full((1, 4), index + 1, dtype=torch.int32),
                      prefix=0, checkpoints=['ck%d.%d' % (index, layer) for layer in range(48)],
                      slots=[['slot%d.%d.%d' % (index, layer, part) for part in range(5)] for layer in range(48)])
                 for index in range(2)]
        fixture = module.ModelBatch.__new__(module.ModelBatch)
        fixture.rows = 32
        fixture.pack = validate_pack(packs)
        fixture.device_loop_gdn = fixture.compact_prologue = fixture.batch_conv = fixture.packed_checkpoints = True
        fixture.commit_only_gdn = True
        fixture.norm_batch = fixture.prefix_zero_reuse = fixture.defer_conv_publication = False
        fixture.verify_t2_audit = False
        fixture.operations = SimpleNamespace(reshape=lambda value, shape: SimpleNamespace(shape=shape),
            get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda: id(value))] * 2)
        fixture.working_states, fixture.gdn_calls, fixture.norm_batch_calls, fixture.user_batched_calls = [], 0, 0, 0
        if module.__name__ == 'model_batch':
            fixture.seq_block_calls, fixture.seq_block_levels, fixture.seq_block_audit = 0, [], audit
        fixture.retained = SimpleNamespace(append=Mock()) if retained else None
        return fixture

    def forward(self, module, decoded, retained=True, audit=(), slot=3, environ=None):
        fixture = self.fixture(module, retained, audit)
        layer = SimpleNamespace(B=8, _stable_state=True, rec_state='rec', conv_states=['c0', 'c1', 'c2', 'c3'])
        helper = SimpleNamespace(direct=True, gdn=layer, live=['rec', 'c0', 'c1', 'c2', 'c3'],
                                 allocate=Mock(side_effect=lambda: ['part%d' % part for part in range(5)]))
        scopes = []

        def decode(*args, **kwargs):
            scopes.append(seq._AUDIT_LAYER[0])
            return decoded

        with clean_environment(**(environ or {})), \
                patch.dict(sys.modules, {'models.tt_transformers.tt.ccl': SimpleNamespace(tt_all_reduce='reduce')}), \
                patch('gdn_multitoken.load_kernels', return_value='kernels'), \
                patch('gdn_device_loop_state.DeviceLoopState.decode', side_effect=decode) as decode_mock, \
                patch('gdn_multitoken_conv.finish_output') as finish, \
                patch('gdn_multitoken_conv.release_owned') as release, \
                patch('gdn_records.retain_checkpoint_histories') as retain:
            output = fixture.gdn_forward(layer, helper, 'ck%d' % slot, slot)(SimpleNamespace(shape=(1, 1, 32, 5120)))
        ops = fixture.operations
        calls = dict(output=output, decode=calls_of(decode_mock, ops), finish=calls_of(finish, ops),
                     release=calls_of(release, ops), retain=calls_of(retain, ops),
                     append=calls_of(fixture.retained.append, ops) if retained else None,
                     counts=(fixture.gdn_calls, fixture.norm_batch_calls, fixture.user_batched_calls))
        return fixture, calls, scopes

    @staticmethod
    def decoded(**extra):
        return dict(commit_only_gdn=True, owned=[], layer_output='reduced', user_batched=True, norm_batch=False,
                    **extra)

    def test_flag_off_gdn_forward_is_the_parents_call_for_call(self):
        import model_batch

        parent = parent_module('model_batch.py', 'model_batch_k5a_parent')
        for retained in (True, False):
            with self.subTest(retained=retained):
                mine = self.forward(model_batch, self.decoded(), retained)
                theirs = self.forward(parent, self.decoded(), retained)
                self.assertEqual(mine[1], theirs[1])
                self.assertEqual(mine[2], [None])
                self.assertEqual((mine[0].seq_block_calls, mine[0].seq_block_levels), (0, []))

    def run_forward(self, module, environ, seq_calls, levels=None, user_batched=48):
        fixture = module.ModelBatch.__new__(module.ModelBatch)
        fixture.retained = fixture.pack = None
        fixture.rows = 32
        fixture.gdn_calls = fixture.norm_batch_calls = fixture.user_batched_calls = 0
        if module.__name__ == 'model_batch':
            fixture.seq_block_calls, fixture.seq_block_levels = 0, []
        fixture.norm_batch = True
        fixture.attention_mask_once = fixture.skip_row_clones = False
        fixture.compact_gdn = fixture.device_loop_gdn = True
        fixture.writers, fixture.readers, fixture.bindings = [], [], []
        fixture.tokens, fixture.cos, fixture.sin, fixture.positions, fixture.pages = range(5)
        fixture.working_states = [SimpleNamespace(calls=0, checkpoint_calls=0, skipped_clones=0) for layer in range(48)]

        def forward(*args, **kwargs):
            fixture.gdn_calls += 48
            for state in fixture.working_states:
                state.calls += 1
                state.checkpoint_calls += 1
            fixture.user_batched_calls += user_batched
            fixture.norm_batch_calls += 48 - user_batched
            if seq_calls:
                fixture.seq_block_calls += seq_calls
                fixture.seq_block_levels.extend(levels or [0] * seq_calls)
            return 'logits'

        fixture.model = SimpleNamespace(_forward_decode=Mock(side_effect=forward))
        with clean_environment(**environ), patch('dflash_device.pindiag') as marker:
            result = fixture.run()
        return result, [call.args for call in marker.call_args_list]

    def test_flag_off_run_logs_what_the_parent_logs(self):
        import model_batch

        parent = parent_module('model_batch.py', 'model_batch_k5a_parent_run')
        for environ in ({}, {batch.FLAG: '1'}, {seq.FLAG: '0', batch.FLAG: '1'}):
            for user_batched in (0, 48):
                with self.subTest(environ=environ, user_batched=user_batched):
                    self.assertEqual(self.run_forward(model_batch, environ, 0, user_batched=user_batched),
                                     self.run_forward(parent, environ, 0, user_batched=user_batched))

    def test_the_marker_counts_the_forwards_k5a_layers_at_their_level(self):
        import lever_n_m3native_gate as gate
        import model_batch

        on = {seq.FLAG: '1', batch.FLAG: '1'}
        result, lines = self.run_forward(model_batch, on, 48)
        self.assertEqual(result, 'logits')
        self.assertIn((seq.MARKER_TEMPLATE, 48, 48, 0), lines)
        text = seq.MARKER_TEMPLATE.format(48, 48, 0)
        self.assertEqual(text, '[PINDIAG] gdn seq_block calls this captured forward: 48 of 48 GDN layers level=0')
        self.assertEqual(gate.GDN_SEQ_BLOCK.search(text).groups(), ('48', '48', '0'))
        # With the flag and nothing engaged: 0, at the level the flag asks for.
        self.assertIn((seq.MARKER_TEMPLATE, 0, 48, 0), self.run_forward(model_batch, on, 0)[1])
        self.assertIn((seq.MARKER_TEMPLATE, 0, 48, 5), self.run_forward(model_batch, dict(on, **{seq.LEVEL_FLAG: '5'}), 0)[1])
        # An engaged layer is reported even without the flag; two levels in one forward are refused.
        self.assertIn((seq.MARKER_TEMPLATE, 12, 48, 0), self.run_forward(model_batch, {}, 12)[1])
        with self.assertRaisesRegex(AssertionError, 'more than one level'):
            self.run_forward(model_batch, on, 2, levels=[0, 1])
        self.assertEqual([line for line in self.run_forward(model_batch, {}, 0)[1] if line[0] == seq.MARKER_TEMPLATE], [])

    def test_the_count_and_level_come_from_the_result_dict(self):
        import model_batch

        fixture, calls, scopes = self.forward(model_batch, self.decoded(seq_block=True, seq_block_level=0))
        self.assertEqual((fixture.seq_block_calls, fixture.seq_block_levels), (1, [0]))
        fixture, calls, scopes = self.forward(model_batch, self.decoded())
        self.assertEqual((fixture.seq_block_calls, fixture.seq_block_levels), (0, []))

    def test_the_audit_scope_names_the_gdn_layer_to_the_launch_inside(self):
        import model_batch

        self.forward(model_batch, self.decoded(), audit=(0, 23), slot=23)
        fixture, calls, scopes = self.forward(model_batch, self.decoded(), audit=(0, 23), slot=23)
        self.assertEqual(scopes, [23])
        self.assertIsNone(seq._AUDIT_LAYER[0], 'restored after the decode')
        fixture, calls, scopes = self.forward(model_batch, self.decoded(), audit=(), slot=23)
        self.assertEqual(scopes, [None])

    def test_the_audit_holds_are_released_with_the_layer_or_the_retained_block(self):
        import model_batch

        holds = [dict(layer=3, output=SimpleNamespace(name='a%d' % user), served_output=SimpleNamespace(name='s%d' % user),
                      served_states=SimpleNamespace(name='p%d' % user)) for user in range(2)]
        pieces = tuple(dict(seq_block=True, seq_block_level=0, seq_block_audit=hold) for hold in holds)
        decoded = self.decoded(segment_results=pieces, seq_block=True, seq_block_level=0)
        fixture, calls, scopes = self.forward(model_batch, decoded, retained=False, audit=(3,))
        self.assertEqual([args for args, kwargs in calls['release']],
                         [('operations', []), ('operations', ['a0', 's0', 'p0', 'a1', 's1', 'p1'])])
        fixture, calls, scopes = self.forward(model_batch, decoded, retained=False, audit=())
        self.assertEqual([args for args, kwargs in calls['release']], [('operations', [])],
                         'only with the audit read at construction')
        # Retained: close() releases them, and only with the audit read at construction.
        fixture = self.fixture(model_batch, retained=True, audit=(3,))
        fixture.retained = SimpleNamespace(records=[('state', decoded, 'carry')], close=Mock())
        fixture.working_states, fixture.grouped_readers, fixture.buffers = [], [], []
        with patch('gdn_multitoken_conv.release_owned') as release:
            fixture.close()
        self.assertEqual(plain(release.call_args_list[0][0][1]), ['a0', 's0', 'p0', 'a1', 's1', 'p1'])
        fixture.retained.close.assert_called_once_with()
        fixture = self.fixture(model_batch, retained=True, audit=())
        fixture.retained = SimpleNamespace(records=[('state', decoded, 'carry')], close=Mock())
        fixture.working_states, fixture.grouped_readers, fixture.buffers = [], [], []
        with patch('gdn_multitoken_conv.release_owned') as release:
            fixture.close()
        release.assert_not_called()


class PackedVerifierFlagTests(test_padded_probe.ProbeFixture):
    """packed_verifier over test_packed_verifier's four-user block with test_padded_probe's fake
    device model (every replay writes what is staged): flag off, three served rounds are the
    parent's (e7bfd324) byte for byte; flag on with an audit, every replay is followed by exactly
    one audit_round over that fixture's retained records, after the replay has run."""

    def test_flag_off_three_rounds_are_the_parents_byte_for_byte(self):
        import packed_verifier

        parent = parent_module('packed_verifier.py', 'packed_verifier_k5a_parent')
        run_rounds = test_padded_probe.FlagOffTests.run_rounds
        for environ in ({}, {seq.FLAG: '0'}, {seq.FLAG: '0', batch.FLAG: '1', seq.AUDIT_FLAG: '0,23'}):
            with self.subTest(environ=environ), patch.dict(os.environ, environ), \
                    patch('gdn_seq_block.audit_round') as audit:
                before = run_rounds(self, parent)
                today = run_rounds(self, packed_verifier)
                audit.assert_not_called()
                self.assertGreater(len(before['copies']), 400)
                self.assertEqual(today, before)

    def test_flag_on_every_replay_is_followed_by_one_audit_round(self):
        os.environ.update({seq.FLAG: '1', batch.FLAG: '1', seq.AUDIT_FLAG: '0,23'})
        calls, blocks = [], []

        def audit_round(operations, records, layers, round_number):
            calls.append((operations is self.ttnn, records is blocks[0].fixture.retained.records, layers,
                          round_number, self.model_hook.replays))
            return 8

        with patch('gdn_seq_block.audit_round', side_effect=audit_round):
            blocks.append(self.probed_block(probe=False))
            self.assertEqual(blocks[0].seq_block_audit, (0, 23))
            self.serve(blocks[0], 3)
        self.assertEqual(calls, [(True, True, (0, 23), number, number) for number in (1, 2, 3)])
        with patch('gdn_seq_block.audit_round', side_effect=AssertionError('[GDN-SEQ-BLOCK-AUDIT] round=4')):
            with self.assertRaisesRegex(AssertionError, 'round=4'):
                blocks[0].verify(self.four())

    def test_every_replay_compares_the_audited_layers_after_the_windows_audit(self):
        source = (HERE / 'packed_verifier.py').read_text(encoding='utf-8')
        self.assertIn('import gdn_seq_block\n', source)
        self.assertIn('        self.seq_block_audit = gdn_seq_block.audit_active()\n', source)
        windows = source.index('verify_trace_t2.audit_round(self.operations, self.fixture.retained.records, '
                               'self.rounds + 1)')
        call = source.index('gdn_seq_block.audit_round(self.operations, self.fixture.retained.records, '
                            'self.seq_block_audit,')
        self.assertLess(windows, call)
        self.assertLess(call, source.index('predictions = [host[slice(*segment_rows(self.shape, segment))]'))
        self.assertIn("if getattr(self, 'seq_block_audit', ()):", source[windows:call])


class GateWiringTests(unittest.TestCase):
    ON = {seq.FLAG: '1', batch.FLAG: '1'}
    BATCHED = '[PINDIAG] gdn user_batched calls this captured forward: 48 of 48 GDN layers'

    def report(self, environ, lines, users=4):
        import lever_n_m3native_gate as gate
        return gate.flag_marker_report(environ, users, '\n'.join(['2026-09-25 | INFO | ' + line for line in lines]))

    def test_the_gate_reads_the_modules_names_and_patterns(self):
        import lever_n_m3native_gate as gate
        self.assertEqual((gate.GDN_SEQ_BLOCK_FLAG, gate.GDN_SEQ_BLOCK_LEVEL_FLAG, gate.GDN_SEQ_BLOCK_AUDIT_FLAG),
                         (seq.FLAG, seq.LEVEL_FLAG, seq.AUDIT_FLAG))
        self.assertEqual(gate.GDN_SEQ_BLOCK.pattern, seq.GATE_PATTERN.pattern)
        self.assertEqual(gate.GDN_SEQ_BLOCK_AUDIT_LINE.pattern, seq.AUDIT_PATTERN.pattern)
        self.assertEqual(gate.GDN_SEQ_BLOCK_LEVEL_BITS, seq.LEVEL_BITS)
        self.assertEqual(gate.select_diagnostic(['x ' + seq.audit_line(0, 1, 0, round=3), 'chatter']),
                         ['x ' + seq.audit_line(0, 1, 0, round=3)])

    def test_no_flag_asks_nothing(self):
        report = self.report({}, [])
        self.assertEqual(report['missing'], [])
        self.assertNotIn('gdn_seq_block', report)
        self.assertNotIn(seq.FLAG, report['found'])
        self.assertEqual(self.report({seq.FLAG: '0'}, [])['missing'], [])

    def test_four_users_need_every_layer_at_the_requested_level(self):
        self.assertEqual(self.report(self.ON, [self.BATCHED, seq.marker(48, 48, 0)])['missing'], [])
        report = self.report(self.ON, [self.BATCHED, seq.marker(0, 48, 0), seq.marker(48, 48, 0)])
        self.assertEqual(report['gdn_seq_block']['captures'], [(48, 48, 0)], 'a 0-count line is not a capture')
        self.assertTrue(report['found'][seq.FLAG]['n of n GDN layers at the level'])
        for lines in ([self.BATCHED], [self.BATCHED, seq.marker(47, 48, 0)], [self.BATCHED, seq.marker(48, 48, 1)]):
            with self.subTest(lines=lines):
                missing = self.report(self.ON, lines)['missing']
                self.assertTrue(any(entry.startswith(seq.FLAG + ': a captured forward running K5-A in all 48')
                                    for entry in missing), missing)
        level = dict(self.ON, **{seq.LEVEL_FLAG: '5'})
        self.assertEqual(self.report(level, [self.BATCHED, seq.marker(48, 48, 5)])['missing'], [])
        self.assertEqual(self.report(level, [self.BATCHED, seq.marker(48, 48, 5)])['gdn_seq_block']['level'], 5)
        mixed = self.report(self.ON, [self.BATCHED, seq.marker(48, 48, 0), seq.marker(48, 48, 3)])['missing']
        self.assertEqual(mixed, [seq.FLAG + ': captures at level 3, not the 0 requested'])
        # A capture that ran K5-A in only some layers fails beside a complete one, at any user count.
        for users in (4, 2):
            with self.subTest(users=users):
                partial = self.report(self.ON, [self.BATCHED, seq.marker(48, 48, 0), seq.marker(47, 48, 0)],
                                      users=users)['missing']
                self.assertEqual(partial, [seq.FLAG + ': captures running K5-A in only some GDN layers: 47 of 48 level=0'])
        self.assertEqual(len(self.report(self.ON, [self.BATCHED, seq.marker(48, 47, 0)])['missing']), 2)
        # One or two users need no every-layer capture (the 64-row block is the four-user one).
        self.assertEqual(self.report(self.ON, [], users=1)['missing'], [])

    def test_configuration_errors_fail_the_arm(self):
        cases = (({seq.FLAG: '1'}, 'needs QWEN_FAST_GDN_USER_BATCH=1'),
                 ({seq.FLAG: 'yes', batch.FLAG: '1'}, '0 or 1'),
                 (dict(self.ON, **{seq.LEVEL_FLAG: '16'}), 'a decimal bitmask below 16'),
                 (dict(self.ON, **{seq.LEVEL_FLAG: '01'}), 'a decimal bitmask below 16'),
                 (dict(self.ON, **{seq.AUDIT_FLAG: '0,0'}), 'distinct GDN layers 0..47'),
                 (dict(self.ON, **{seq.AUDIT_FLAG: '48'}), 'distinct GDN layers 0..47'),
                 ({seq.AUDIT_FLAG: '0'}, 'without QWEN_FAST_GDN_SEQ_BLOCK=1 do nothing'),
                 ({seq.LEVEL_FLAG: '0'}, 'without QWEN_FAST_GDN_SEQ_BLOCK=1 do nothing'))
        for environ, message in cases:
            with self.subTest(environ=environ):
                missing = self.report(environ, [self.BATCHED, seq.marker(48, 48, 0)], users=1)['missing']
                self.assertTrue(any(message in entry for entry in missing), missing)

    def test_every_audit_line_must_be_zero_and_every_listed_layer_audited_for_every_user(self):
        environ = dict(self.ON, **{seq.AUDIT_FLAG: '0,23,47'})
        lines = [self.BATCHED, seq.marker(48, 48, 0)] + [
            seq.audit_line(layer, user, 0, round=round_number, output=0, states=0)
            for round_number in (1, 2) for layer in (0, 23, 47) for user in range(4)]
        report = self.report(environ, lines)
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['gdn_seq_block']['audit_lines'], 24)
        bad = list(lines)
        bad[9] = seq.audit_line(23, 3, 17, round=1, output=0, states=17)
        missing = self.report(environ, bad)['missing']
        self.assertEqual(len(missing), 1)
        self.assertIn('1 audit line(s) with mismatches', missing[0])
        self.assertIn('layer=23 user=3 mismatches=17', missing[0])
        # a mismatch fails at any user count
        self.assertEqual(len(self.report(dict(environ), bad, users=1)['missing']), 1)
        absent = [line for line in lines if 'layer=47 user=2 ' not in line]
        self.assertEqual(self.report(environ, absent)['missing'],
                         [seq.AUDIT_FLAG + ': no audit line for layer/user 47/2'])
        self.assertEqual(self.report(environ, lines[:2])['missing'],
                         [seq.AUDIT_FLAG + ': no audit line for layer/user ' + ','.join(
                             '%d/%d' % (layer, user) for layer in (0, 23, 47) for user in range(4))])


class ArmWiringTests(unittest.TestCase):
    START = '# K5-A (gdn_seq_block.py; default off).'
    END = '  echo "K5-A seq block 1 level $seq_block_level audit ${seq_block_audit:-none}"' + chr(10) + 'fi' + chr(10)
    LINES = ('${M3NATIVE_GDN_SEQ_BLOCK:+-e QWEN_FAST_GDN_SEQ_BLOCK=1}',
             '${M3NATIVE_GDN_SEQ_BLOCK_LEVEL:+-e QWEN_FAST_GDN_SEQ_BLOCK_LEVEL=$M3NATIVE_GDN_SEQ_BLOCK_LEVEL}',
             '${M3NATIVE_GDN_SEQ_BLOCK_AUDIT:+-e QWEN_FAST_GDN_SEQ_BLOCK_AUDIT=$M3NATIVE_GDN_SEQ_BLOCK_AUDIT}')

    def text(self):
        return ARM.read_text(encoding='utf-8')

    def bash(self, script, **environ):
        import shutil
        import subprocess

        found = shutil.which('bash')
        if found is None:
            self.skipTest('no bash')
        try:
            return subprocess.run([found, '-c', script], capture_output=True, text=True, timeout=60,
                                  env=dict(PATH=os.environ.get('PATH', ''), **environ))
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)

    def validate(self, **environ):
        text = self.text()
        start = text.index(self.START)
        end = text.index(self.END, start) + len(self.END)
        script = 'set -euo pipefail' + chr(10) + 'users="${USERS_UNDER_TEST}"' + chr(10) + text[start:end] + 'echo VALID' + chr(10)
        return self.bash(script, USERS_UNDER_TEST=environ.pop('users', '4'), **environ)

    def test_the_arm_refuses_what_the_image_or_the_gate_would(self):
        on = dict(M3NATIVE_GDN_SEQ_BLOCK='1', M3NATIVE_GDN_USER_BATCH='1')
        for environ in ({}, on, dict(on, M3NATIVE_GDN_SEQ_BLOCK_LEVEL='0'), dict(on, M3NATIVE_GDN_SEQ_BLOCK_LEVEL='15'),
                        dict(on, M3NATIVE_GDN_SEQ_BLOCK_AUDIT='0,23,47'), dict(on, M3NATIVE_GDN_SEQ_BLOCK_AUDIT='5'),
                        dict(on, users='2'), dict(M3NATIVE_GDN_USER_BATCH='1'),
                        dict(on, M3NATIVE_GDN_USER_BATCH_MIN_USERS='3'),
                        dict(on, M3NATIVE_GDN_SEQ_BLOCK_AUDIT='0', M3NATIVE_GDN_USER_BATCH_MIN_USERS='1')):
            with self.subTest(accepted=environ):
                result = self.validate(**dict(environ))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('VALID', result.stdout)
        for environ, message in (
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK='yes'), 'must be 1 or unset'),
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK='0'), 'must be 1 or unset'),
                (dict(M3NATIVE_GDN_SEQ_BLOCK='1'), 'needs M3NATIVE_GDN_USER_BATCH=1'),
                (dict(M3NATIVE_GDN_SEQ_BLOCK='1', M3NATIVE_GDN_USER_BATCH='yes'), 'needs M3NATIVE_GDN_USER_BATCH=1'),
                (dict(M3NATIVE_GDN_SEQ_BLOCK_LEVEL='0'), 'without M3NATIVE_GDN_SEQ_BLOCK=1 do nothing'),
                (dict(M3NATIVE_GDN_USER_BATCH='1', M3NATIVE_GDN_SEQ_BLOCK_AUDIT='0'), 'without M3NATIVE_GDN_SEQ_BLOCK=1'),
                (dict(on, users='1'), 'serves the packed block only'),
                (dict(on, M3NATIVE_SEQUENTIAL_USERS='4'), 'serves the packed block only'),
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK_LEVEL='16'), 'decimal bitmask 0..15'),
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK_LEVEL='01'), 'decimal bitmask 0..15'),
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK_LEVEL='x'), 'decimal bitmask 0..15'),
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK_AUDIT='48'), 'distinct GDN layers 0..47'),
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK_AUDIT='0,0'), 'distinct GDN layers 0..47'),
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK_AUDIT='23,5,23'), 'distinct GDN layers 0..47'),
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK_AUDIT='01'), 'distinct GDN layers 0..47'),
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK_AUDIT='0,,1'), 'distinct GDN layers 0..47'),
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK_AUDIT='all'), 'distinct GDN layers 0..47'),
                (dict(on, M3NATIVE_GDN_SEQ_BLOCK_AUDIT='0', M3NATIVE_GDN_USER_BATCH_MIN_USERS='2'),
                 'unset M3NATIVE_GDN_USER_BATCH_MIN_USERS')):
            with self.subTest(refused=environ):
                result = self.validate(**dict(environ))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn('VALID', result.stdout)

    def test_the_three_switches_cross_right_after_the_user_batch_before_the_entrypoint(self):
        text = self.text()
        lines = text.split(chr(10))
        start = next(number for number, line in enumerate(lines) if self.LINES[0] in line)
        self.assertEqual(lines[start - 1].strip(), '${M3NATIVE_GDN_USER_BATCH:+-e QWEN_FAST_GDN_USER_BATCH=1} ' + chr(92))
        for offset, expected in enumerate(self.LINES):
            with self.subTest(line=expected):
                self.assertEqual(text.count(expected), 1)
                self.assertEqual(lines[start + offset].strip(), expected + ' ' + chr(92), 'nothing else on the line')
                self.assertLess(text.index(expected), text.index('--entrypoint python3'))
        self.assertLess(text.index(self.START), text.index('docker run --rm --name "$name"'))
        self.assertNotIn(b'\r', ARM.read_bytes())

    def test_unset_nothing_crosses_and_set_each_crosses_as_given(self):
        script = 'printf "%s|" ' + ' '.join(self.LINES) + chr(10)
        self.assertEqual(self.bash(script).stdout.strip('|'), '')
        both = self.bash(script, M3NATIVE_GDN_SEQ_BLOCK='1', M3NATIVE_GDN_SEQ_BLOCK_LEVEL='0',
                         M3NATIVE_GDN_SEQ_BLOCK_AUDIT='0,23,47')
        self.assertEqual(both.stdout, '-e|QWEN_FAST_GDN_SEQ_BLOCK=1|-e|QWEN_FAST_GDN_SEQ_BLOCK_LEVEL=0|'
                                      '-e|QWEN_FAST_GDN_SEQ_BLOCK_AUDIT=0,23,47|')


class ShippingTests(unittest.TestCase):
    FILES = ('gdn_seq_block.py', 'gdn_seq_block_compute.cpp', 'gdn_seq_block_reader.cpp', 'gdn_seq_block_writer.cpp')

    def test_the_module_and_its_three_kernels_reach_the_image_through_both_copy_lists(self):
        from test_serving_image_copy_closure import WORKFLOW, context_modules, dockerfile_modules, dockerfile_text

        docker = dockerfile_text()
        for name in self.FILES + ('gdn_user_batch_conv.py', 'model_batch.py', 'packed_verifier.py',
                                  'gdn_user_batch.py', 'verify_trace_t1.py'):
            with self.subTest(module=name):
                self.assertIn(name, dockerfile_modules(docker))
                self.assertIn(name, context_modules())
        # The kernels travel beside gdn_state_copy.cpp, in both lists.
        copy = [line for line in docker.splitlines() if line.startswith('COPY ') and 'gdn_state_copy.cpp' in line]
        loop = [line for line in WORKFLOW.read_text(encoding='utf-8').splitlines()
                if 'for name in' in line and 'gdn_state_copy.cpp' in line]
        for lines in (copy, loop):
            self.assertEqual(len(lines), 1)
            for name in self.FILES[1:]:
                self.assertIn(name, lines[0])
        self.assertEqual(seq.SOURCES, dict(reader='gdn_seq_block_reader.cpp', writer='gdn_seq_block_writer.cpp',
                                           compute='gdn_seq_block_compute.cpp'))

    def test_the_cpu_suite_runs_this_file(self):
        self.assertRegex(CPU_WORKFLOW.read_text(encoding='utf-8'), r'python -B -m unittest [^\n]*\btest_gdn_seq_block\b')

    def test_every_touched_file_is_lf(self):
        for relative in ('scripts/ci/gdn_seq_block.py', 'scripts/ci/gdn_user_batch_conv.py', 'scripts/ci/model_batch.py',
                         'scripts/ci/packed_verifier.py', 'scripts/ci/lever_n_m3native_gate.py',
                         'scripts/ci/lever_n_m3native_run_arm.sh', 'scripts/ci/gdn-seq-block-rig.sh',
                         'scripts/ci/test_gdn_seq_block.py', 'docker/qwen-fast-serving.Dockerfile',
                         '.github/workflows/qwen-fast-serving-image.yml', '.github/workflows/qwen-integration-cpu.yml',
                         'scripts/ci/test_lever_n_m3native_gate.py', 'scripts/ci/test_m3native_arm_env.py',
                         'scripts/ci/test_qual_card.py'):
            with self.subTest(file=relative):
                self.assertNotIn(b'\r', (ROOT / relative).read_bytes())

    def test_the_kernel_sources_are_the_bytes_card_b_qualified(self):
        """QUALIFIED[0] is the hash of what these generate (with the pinned native prefix, which CI
        does not have); the sources themselves are pinned here so CI sees a change too. Probe run
        r2 (the split-NoC writer, W4) shipped exactly these (ship2)."""
        probed = dict(compute='6373114a211bb57ece4ce3517b7c35bd91b50aedd11b6864d69ae47b8772aa45',
                      reader='2423c67cbbeab03249d61fb260faa23af6af6a412e091a2b20b67252ef20945e',
                      writer='1016497072c14fedcb240476c34c645275733fc981843c4de4744ceff142ef13')
        self.assertEqual({role: hashlib.sha256((HERE / seq.SOURCES[role]).read_bytes()).hexdigest() for role in seq.ROLES},
                         probed)

    def test_the_pinned_sources_are_untouched_since_the_parent(self):
        import subprocess

        pinned = ['scripts/ci/gdn_multitoken.py', 'scripts/ci/gdn_multitoken_conv.py', 'scripts/ci/gdn_device_loop_state.py',
                  'scripts/ci/gdn_user_batch.py', 'scripts/ci/gdn_records.py', 'scripts/ci/gdn_commit_dma.py',
                  'scripts/ci/gdn_commit_dma.cpp']
        try:
            result = subprocess.run(['git', 'diff', '--name-only', PARENT, '--', *pinned], capture_output=True,
                                    cwd=str(ROOT), timeout=60, text=True)
        except (OSError, subprocess.SubprocessError):
            self.skipTest('no git')
        if result.returncode != 0:
            self.skipTest('no git history for %s' % PARENT)
        self.assertEqual(result.stdout.strip(), '')


if __name__ == '__main__':
    unittest.main()
