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
    traced and P1 verdicts, as pure functions.
The wiring (gdn_user_batch_conv's dispatch, the model_batch counter, the flag-off call-for-call
proof) comes with the model wiring, not here.

gdn_seq_block.execute checks the served runtime pin (gdn_multitoken.validate_handoff_runtime) as
gdn_user_batch.execute does; it reads tt-metal files, so this module patches it out for every test
(setUpModule) and HandoffPinTests holds the call itself.
"""

import hashlib
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_multitoken as native
import gdn_seq_block as seq
import gdn_user_batch as batch
from test_gdn_user_batch import FakeTTNN, FakeTensor, KERNELS, mesh, user_inputs
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
    def test_the_totals_are_the_plans_and_under_todays_plan(self):
        io, fp32 = seq.cb_plan()
        self.assertEqual(sum(io.values()), 78)
        self.assertEqual(sum(io.values()) * 2048, 159744)
        self.assertEqual(sum(fp32.values()), 109)
        self.assertEqual(sum(fp32.values()) * 4096, 446464)
        self.assertEqual(seq.cb_bytes(), 606208)
        self.assertEqual(seq.SERVED_CB_BYTES, 630784)
        self.assertEqual(seq.SERVED_CB_BYTES - seq.cb_bytes(), 24576)

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
                                           'BF', 'GEXP', 'TOKA', 'TOKB')),
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
        self.assertEqual(seq.runtime_args('reader', 5, addresses), [5, 10, 20, 30, 40, 70, 80])
        self.assertEqual(seq.runtime_args('writer', 5, addresses), [5, 50, 60])
        self.assertEqual(seq.runtime_args('compute', 5, addresses), [16])
        self.assertEqual(seq.ACCESSORS, dict(reader=(0, 1, 2, 3, 6, 7), writer=(4, 5), compute=()))
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
    def test_qualified_starts_empty(self):
        self.assertEqual(seq.QUALIFIED, {})

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
            for unused, unused_too, args in chip.kernels[3 * user].runtime_args.flattened():
                self.assertEqual(args[1:], (qkv.address, beta.address, gate.address, initial.address,
                                            z.address, norm_w.address))

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
        self.assertEqual(report['cb_bytes_per_worker'], 606208)
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


if __name__ == '__main__':
    unittest.main()
