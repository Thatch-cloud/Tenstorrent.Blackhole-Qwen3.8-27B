"""The four-card fused commit (fused_commit_tp.py): the round-fence plan's H1b fused commit at (1, 4) with two KV heads a chip,
installed through tp_addresses.MODULE_TWINS at QWEN_FAST_TP=4 only. fused_commit.py (the pair's) is not touched.

What is held here, on the CPU (nothing runs on a card; the one-card 2-head proof of the slide in place and the audited in-model smoke are
the hardware half, scripts/ci/references/tp4-fcommit-jobs):

  - the seam: the twin replaces fused_commit in sys.modules at four cards and only there, the pair process never loads it, the pinned
    module and the slide files it rests on are the bytes they were, and every name the twin does not define is the pinned module's
    own object (one set of flags, markers and log lines for the gate and the smoke);
  - the width: banks (1, 2, 2048, 128), deltas (1, 2, 32, 128), 8 workers a bank, ten banks (80 workers) a program at four cards,
    the pair's (1, 4, ...), 16 and five at the pair, and no import-time width;
  - the slide program: at the pair the twin's is field for field the pinned one (in place and out of place); at four cards it is
    draft_kv_slide_tp.prepare's per-chip program for one bank, and four chips of 80 workers for a user's ten banks (40 for five);
    the pair's shapes, two-shard tensors, aliasing, a small grid and any mesh but (1, 4) are refused;
  - EXACTNESS OF THE SLIDE IN PLACE at two KV heads: the M-F0 harness's kernel mirrors (both qualified kernels, pinned to the sources
    by that harness's own tests) give, for every history in the ramp, at the boundaries and in the steady window, and every prefix
    1..16, the same bytes in place as out of place, as the eager six-op chain (slice, concat, slice, pad) written here, and as the
    per-row rule of draft_kv_slide_tp's test; no bank page is read after its worker wrote it for any of the eight workers; the
    backwards walk (the negative control) is caught;
  - A REQUEST'S LIFE, per segment 0..3: ramp commits through today's path (out of place and swap), the one `parity` refusal after an odd
    number of swaps, steady fused in-place commits at every prefix, and a sequential step between packed rounds (which flips parity
    and refuses once), against the pure eager sequence: after EVERY commit the four chips' active banks, the position and the history
    rows are equal to the eager path's, and a fused commit never swaps;
  - the guards (every refusal and its reason, the four-card scope: the flag and the four-card cache class), the build's refusals
    (mesh, grid, kernel, scope, weights), the capture (four T_proj traces and 64 slide traces, warmed first, the timing replays), the
    out-of-place branch through the four-card transport, and the audit at four chips (80 items, a bank mismatch on chip 3 repaired
    from the spare, a delta mismatch is the E2 claim failing);
  - E2 at four-card shapes: T_proj's op sequence at count 16 is today's publication at prefix 16 call for call through the four-card
    twins (all-gather of the fp32 chip partials, chip-order adds, two KV heads), every op in it is row-local, and a numeric model of
    those ops shows rows 0..prefix-1 equal for every prefix with a hundredfold partner row (and a row-mixing control leaks);
  - shipping: all three image lists carry the twin, the closure test classifies it served, and the CPU suite runs this file.

    py -3.11 -B -m unittest test_fused_commit_tp4      (from scripts/ci)
"""

from contextlib import ExitStack
from collections import defaultdict
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from dflash_device import DFlashDevice  # noqa: E402
import draft_kv_history  # noqa: E402
import draft_kv_history_tp  # noqa: E402
import draft_kv_slide  # noqa: E402
import draft_kv_slide_tp  # noqa: E402
import fused_commit as pinned  # noqa: E402
import fused_commit_tp as twin  # noqa: E402
from packed_shapes import m3_shape  # noqa: E402
from packed_verifier import PackedFeatureTaps  # noqa: E402
import test_fused_commit as base  # noqa: E402
import test_draft_kv_slide_tp as slide_tests  # noqa: E402
import test_packed_verifier as tpv  # noqa: E402
import tp_addresses  # noqa: E402
import tp_shapes  # noqa: E402
from tp_test_support import four_cards, pair  # noqa: E402

HARNESS = ROOT / 'optimisation' / 'ttnn-op' / 'draft_slide_inplace' / 'draft_slide_inplace_card_b.py'
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'
CHIPS, KV_HEADS = 4, 2
KV_SHAPE, DELTA_SHAPE = (1, KV_HEADS, 2048, 128), (1, KV_HEADS, 32, 128)
HEADS = ('k', 'v')
PAGE_WIDTH = tpv.PAGE_WIDTH
# The pair's fused commit, its quad and the slide files it rests on at the commit the four-card twin branched from (tp4/stack d179ccd9):
# the twin exists so that these bytes never move.
PINNED_SHA256 = {
    'fused_commit.py': '2a434fb71af4d710c710a636c4ec10636b7c439264fa21f416a9f983a1f5eab8',
    'draft_kv_slide.py': '8ea58ae42260cb03f37e224d18572c4efe467ff556912a64ebcb1d15aed9c7fc',
    'draft_kv_slide.cpp': 'bc45d47257c844aff4bf17f478b536a48763578e544597884b7d1620083b6ba1',
    'draft_kv_slide_direct.cpp': '1679bbd779add56b4bd445a6b4c51bd3e49c39a8a520dfaddfc3bd9f36667d47',
}
TWIN_OWN = {'kv_shape', 'delta_shape', 'workers_per_bank', 'banks_per_program', 'kernel_path', 'slide_scope_live', 'cache_class',
            'slide_program', '_project_features', '_project_key_value', '_transport', '_addresses', 'FusedCommit', 'build'}


def clean_environment(**flags):
    """No QWEN_FAST_* flag but the ones given."""
    return base.clean_environment(**flags)


def load_harness():
    spec = importlib.util.spec_from_file_location('draft_slide_inplace_card_b_for_fcommit', str(HARNESS))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------
# The seam and the width.
# ---------------------------------------------------------------------------------------------

class SeamTests(unittest.TestCase):
    def test_the_seam_names_the_twin_and_installs_it_at_four_cards_only(self):
        self.assertIn(('fused_commit', 'fused_commit_tp'), tp_addresses.MODULE_TWINS)
        self.assertIs(sys.modules['fused_commit'], pinned)
        with four_cards():
            self.assertIs(sys.modules['fused_commit'], twin)
            import fused_commit as reached

            self.assertIs(reached, twin, 'packed_verifier, serving_packed_step and the pair trace import it lazily: the twin')
        self.assertIs(sys.modules['fused_commit'], pinned)
        with pair(), self.assertRaises(ValueError):
            tp_addresses.install()
        self.assertIs(sys.modules['fused_commit'], pinned)

    def test_the_lazy_importers_reach_the_twin_at_four_cards(self):
        """Every module that names fused_commit imports it inside a function (or through the flag's check), so the swap of
        sys.modules is what reaches them; none of them holds the module object at import time."""
        for name in ('packed_verifier.py', 'serving_packed_step.py', 'dflash_proposal_trace.py', 'verify_prestage.py',
                     'serving_runtime.py', 'serving_worker_hook.py'):
            text = (HERE / name).read_text(encoding='utf-8')
            for line in text.splitlines():
                if 'import fused_commit' in line or 'from fused_commit import' in line:
                    self.assertTrue(line.startswith((' ', '\t')), (name, line, 'a module-level import would hold the pinned module'))

    def test_the_pair_process_never_loads_the_twin(self):
        code = ('import sys; sys.path.insert(0, %r); import tp_addresses, fused_commit, packed_verifier, serving_packed_step; '
                "assert 'fused_commit_tp' not in sys.modules, 'the pair loaded the four-card fused commit'" % str(HERE))
        environment = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_')}
        subprocess.run([sys.executable, '-B', '-c', code], env=environment, check=True, timeout=120)

    def test_the_pinned_fused_commit_and_the_slide_files_are_the_bytes_they_were(self):
        for name, digest in PINNED_SHA256.items():
            data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest, '%s moved: the twin exists so it never does' % name)

    def test_the_twin_redefines_only_the_width_scope_and_seam_names_and_shares_the_rest(self):
        own = {name for name, value in vars(twin).items() if not name.startswith('__')
               and getattr(value, '__module__', None) == 'fused_commit_tp'}
        self.assertEqual(own, TWIN_OWN)
        for name in dir(pinned):
            if name.startswith('__') or name in TWIN_OWN | {'KV_SHAPE', 'DELTA_SHAPE', 'WORKERS', 'BANKS_PER_PROGRAM'}:
                continue
            if name in ('FusedCommit', 'build'):
                continue
            with self.subTest(name=name):
                self.assertIs(getattr(twin, name), getattr(pinned, name), 'the pinned module\'s own object')
        # the shared state is one object, so the pair-trace's one-line marker is once per process whichever module logs it
        self.assertIs(twin._LIVE_NOTED, pinned._LIVE_NOTED)
        self.assertIs(twin.EXPECTED_REFUSALS, pinned.EXPECTED_REFUSALS)
        self.assertIs(twin.install_fused_commit, pinned.install_fused_commit)
        self.assertIs(twin.live_bank_history, pinned.live_bank_history)
        with self.assertRaises(AttributeError):
            twin.no_such_name

    def test_the_pairs_literal_widths_are_refused_by_name_not_read_from_the_pinned_module(self):
        for name in ('KV_SHAPE', 'DELTA_SHAPE', 'WORKERS', 'BANKS_PER_PROGRAM'):
            with self.subTest(name=name), self.assertRaisesRegex(AttributeError, 'literal width'):
                getattr(twin, name)
            self.assertTrue(hasattr(pinned, name))

    def test_the_twin_class_overrides_every_method_that_reads_a_width_a_scope_or_a_seam(self):
        overridden = {name for name, value in vars(twin.FusedCommit).items() if callable(value) and not name.startswith('__')}
        self.assertEqual(overridden, {'retainer', 'project', 'capture', 'measure', 'engaged_line', 'refusal', 'prepare',
                                      'eager_reference'})
        inherited = {'allocate', 'allocated', 'owner', 'run_slides', 'trace_count', 'stage_tables', 'stage_window', 'note',
                     'commit_kv', 'discard_kv', 'audit_round', 'describe', 'release_buffers', 'close'}
        for name in inherited:
            self.assertIs(getattr(twin.FusedCommit, name), getattr(pinned.FusedCommit, name), name)

    def test_the_twin_has_no_import_time_width(self):
        """Read at each call: the same module object serves a pair-width test and a four-card one."""
        with pair():
            self.assertEqual((twin.kv_shape(), twin.delta_shape(), twin.workers_per_bank(), twin.banks_per_program()),
                             ((1, 4, 2048, 128), (1, 4, 32, 128), 16, 5))
        with four_cards():
            self.assertEqual((twin.kv_shape(), twin.delta_shape(), twin.workers_per_bank(), twin.banks_per_program()),
                             (KV_SHAPE, DELTA_SHAPE, 8, 10))
        with pair():
            self.assertEqual(twin.workers_per_bank(), 16)


class WidthTests(unittest.TestCase):
    def test_a_program_is_eighty_workers_at_both_widths_and_the_layout_flag_names_a_fallback(self):
        self.assertEqual(twin.PROGRAM_WORKERS, 80)
        with pair():
            self.assertEqual(twin.workers_per_bank() * twin.banks_per_program(), 80)
        with four_cards():
            self.assertEqual(twin.workers_per_bank() * twin.banks_per_program(), 80)
            self.assertEqual(twin.banks_per_program({twin.BANKS_FLAG: '5'}), 5, '2 x 40 workers')
            self.assertEqual(twin.banks_per_program({twin.BANKS_FLAG: '1'}), 1)
            for text in ('3', '20', '0', 'all'):
                with self.assertRaises(ValueError):
                    twin.banks_per_program({twin.BANKS_FLAG: text})
        with pair():
            with self.assertRaises(ValueError):
                twin.banks_per_program({twin.BANKS_FLAG: '10'})

    def test_the_kernel_is_the_slide_transports_and_qualified(self):
        self.assertEqual(twin.kernel_path(), draft_kv_slide_tp.KERNEL)
        self.assertEqual(twin.kernel_path(), HERE / 'draft_kv_slide.cpp')
        self.assertIn(twin.kernel_kind(twin.kernel_path()), ('scalar', 'direct'))
        self.assertEqual(twin.kernel_kind(HERE / 'draft_kv_slide_direct.cpp'), 'direct')
        self.assertIsNone(twin.kernel_kind(HERE / 'fused_commit_tp.py'))

    def test_the_scope_is_the_four_card_slide_flag(self):
        with four_cards():
            with patch.dict(os.environ):
                os.environ.pop('QWEN_FAST_TP_KV_SLIDE', None)
                self.assertFalse(twin.slide_scope_live())
            for value in ('0', '', 'true', '2'):
                with patch.dict(os.environ, {'QWEN_FAST_TP_KV_SLIDE': value}):
                    self.assertFalse(twin.slide_scope_live(), value)
            with patch.dict(os.environ, {'QWEN_FAST_TP_KV_SLIDE': '1'}):
                self.assertTrue(twin.slide_scope_live())
        self.assertIs(twin.cache_class(), draft_kv_history_tp.DraftKVHistory)
        self.assertNotIsInstance(object.__new__(draft_kv_history.DraftKVHistory), twin.cache_class())


# ---------------------------------------------------------------------------------------------
# The slide program.
# ---------------------------------------------------------------------------------------------

class ChipTTNN(base.ProgramTTNN):
    """base.ProgramTTNN with a chip count: device tensors of `chips` shards, generic_op recorded."""

    def __init__(self, chips):
        super().__init__()
        self.chips = chips

    def tensor(self, shape, *, tile=(32, 32), dtype='bf16'):
        shards = []
        for chip in range(self.chips):
            address = next(self.addresses)
            shards.append(SimpleNamespace(shape=tuple(shape), dtype=dtype, layout='tile', memory_config=lambda: 'dram',
                                          tile=SimpleNamespace(tile_shape=tile, transpose_of_faces=False,
                                                               transpose_within_face=False),
                                          buffer_address=lambda address=address: address, chip=chip))
        return SimpleNamespace(shape=tuple(shape), shards=shards)


def grid_mesh(width, x=11, y=10):
    return SimpleNamespace(shape=[1, width], compute_with_storage_grid_size=lambda: SimpleNamespace(x=x, y=y))


class SlideProgramTests(unittest.TestCase):
    def use(self, chips):
        self.ttnn = ChipTTNN(chips)
        patcher = patch.dict('sys.modules', {'ttnn': self.ttnn})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_at_the_pair_the_twins_program_is_the_pinned_one_field_for_field(self):
        self.use(2)
        with pair():
            grid = grid_mesh(2)
            kernel = twin.kernel_path()
            banks = [(self.ttnn.tensor(pinned.KV_SHAPE), self.ttnn.tensor(pinned.DELTA_SHAPE)) for bank in range(5)]
            for prefix in (1, 9, 16):
                with self.subTest(mode='in place', prefix=prefix):
                    self.assertEqual(twin.slide_program(self.ttnn, grid, kernel, banks, history_rows=2048, prefix=prefix),
                                     pinned.slide_program(self.ttnn, grid, kernel, banks, history_rows=2048, prefix=prefix))
            active, delta, spare = (self.ttnn.tensor(shape) for shape in (pinned.KV_SHAPE, pinned.DELTA_SHAPE, pinned.KV_SHAPE))
            for history, prefix in ((2048, 7), (2047, 2), (31, 2)):
                with self.subTest(mode='out of place', history=history, prefix=prefix):
                    self.assertEqual(
                        twin.slide_program(self.ttnn, grid, kernel, [(active, delta, spare)], history_rows=history,
                                           prefix=prefix, in_place=False),
                        pinned.slide_program(self.ttnn, grid, kernel, [(active, delta, spare)], history_rows=history,
                                             prefix=prefix, in_place=False))

    def test_one_bank_out_of_place_is_the_four_card_transports_program_field_for_field(self):
        self.use(CHIPS)
        with four_cards():
            grid = grid_mesh(CHIPS)
            active, delta, spare = (self.ttnn.tensor(shape) for shape in (KV_SHAPE, DELTA_SHAPE, KV_SHAPE))
            for history, prefix in ((2048, 16), (2048, 1), (2047, 2), (100, 7)):
                with self.subTest(history=history, prefix=prefix):
                    draft_kv_slide_tp.prepare(grid, active, delta, spare, history_rows=history, prefix=prefix)()
                    tensors, served = self.ttnn.generic.pop()
                    ours = twin.slide_program(self.ttnn, grid, twin.kernel_path(), [(active, delta, spare)],
                                              history_rows=history, prefix=prefix, in_place=False)
                    self.assertEqual(tensors, [active, delta, spare])
                    self.assertEqual(len(ours), CHIPS)
                    self.assertEqual(ours, served)

    def test_a_users_ten_banks_are_four_chips_of_one_eighty_worker_program(self):
        self.use(CHIPS)
        with four_cards():
            grid = grid_mesh(CHIPS)
            banks = [(self.ttnn.tensor(KV_SHAPE), self.ttnn.tensor(DELTA_SHAPE)) for bank in range(10)]
            program = twin.slide_program(self.ttnn, grid, twin.kernel_path(), banks, history_rows=2048, prefix=9)
            self.assertEqual(len(program), CHIPS)
            for chip in range(CHIPS):
                (kernel,) = program[('coordinate-range', ('coordinate', 0, chip), ('coordinate', 0, chip))].kernels
                self.assertEqual(kernel.kernel_source, str(twin.kernel_path()))
                cores = [(x, y) for x, row in kernel.runtime_args.items() for y in row]
                self.assertEqual(len(cores), 80)
                for index, (bank, delta) in enumerate(banks):
                    for worker in range(8):
                        core = index * 8 + worker
                        arguments = kernel.runtime_args[core % 11][core // 11]
                        address = bank.shards[chip].buffer_address()
                        self.assertEqual(arguments, [address, delta.shards[chip].buffer_address(), address, 2048, 9, 9, 2048,
                                                     worker])
            half = twin.slide_program(self.ttnn, grid, twin.kernel_path(), banks[:5], history_rows=2048, prefix=9)
            (kernel,) = half[('coordinate-range', ('coordinate', 0, 3), ('coordinate', 0, 3))].kernels
            self.assertEqual(sum(len(row) for row in kernel.runtime_args.values()), 40)
        self.assertEqual(twin.io_list(banks[:2]), [banks[0][0], banks[0][1], banks[0][0], banks[1][0], banks[1][1], banks[1][0]])
        self.assertIs(twin.io_list, pinned.io_list)

    def test_the_builder_refuses_what_the_transport_would_and_what_the_width_cannot_hold(self):
        self.use(CHIPS)
        with four_cards():
            grid = grid_mesh(CHIPS)
            bank = self.ttnn.tensor(KV_SHAPE)
            pair_bank = self.ttnn.tensor(pinned.KV_SHAPE)
            two_shard = ChipTTNN(2)
            cases = {
                'aliased delta': ([(bank, bank)], grid),
                'short delta': ([(bank, self.ttnn.tensor((1, KV_HEADS, 16, 128)))], grid),
                'the pairs four-head bank': ([(pair_bank, self.ttnn.tensor(pinned.DELTA_SHAPE))], grid),
                'transposed tile': ([(bank, self.ttnn.tensor(DELTA_SHAPE, tile=(16, 32)))], grid),
                'two shards': ([(two_shard.tensor(KV_SHAPE), two_shard.tensor(DELTA_SHAPE))], grid),
                'small grid': ([(self.ttnn.tensor(KV_SHAPE), self.ttnn.tensor(DELTA_SHAPE)) for index in range(10)],
                               grid_mesh(CHIPS, 4, 4)),
                'the pairs mesh': ([(bank, self.ttnn.tensor(DELTA_SHAPE))], grid_mesh(2)),
                'a square mesh': ([(bank, self.ttnn.tensor(DELTA_SHAPE))],
                                  SimpleNamespace(shape=[2, 2], compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))),
            }
            for name, (banks, mesh) in cases.items():
                with self.subTest(name=name), self.assertRaises(ValueError):
                    twin.slide_program(self.ttnn, mesh, twin.kernel_path(), banks, history_rows=2048, prefix=4)
            with self.assertRaises(ValueError):
                twin.slide_program(self.ttnn, grid, twin.kernel_path(), [], history_rows=2048, prefix=4)
            with self.assertRaises(ValueError):
                twin.slide_program(self.ttnn, grid, twin.kernel_path(), [(bank, self.ttnn.tensor(DELTA_SHAPE), bank)],
                                   history_rows=2048, prefix=4)   # in place takes (bank, delta)
            with self.assertRaises(ValueError):
                twin.slide_program(self.ttnn, grid, twin.kernel_path(), [(bank, self.ttnn.tensor(DELTA_SHAPE))],
                                   history_rows=2048, prefix=4, in_place=False)   # out of place takes three


# ---------------------------------------------------------------------------------------------
# Exactness of the slide in place at two KV heads.
# ---------------------------------------------------------------------------------------------

RAMP_HISTORIES = (1, 17, 31, 32, 33, 100, 1000, 2000, 2016, 2031, 2032, 2033, 2040, 2047, 2048)


def eager_chain(active, delta, history_rows, prefix):
    """DraftKVHistory_tp.prepare's flag-off chain, on int16 bit tensors: slice the history, slice the accepted rows, concat, slice the
    last `rows`, pad with zeros to 2048."""
    rows = min(2048, history_rows + prefix)
    historical = active[:, :, :history_rows]
    accepted = delta[:, :, :prefix]
    combined = torch.cat([historical, accepted], dim=2)
    tail = combined[:, :, history_rows + prefix - rows:history_rows + prefix]
    return torch.nn.functional.pad(tail, (0, 0, 0, 2048 - rows))


class InPlaceExactnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_harness()
        cls.card.configure(KV_HEADS)

    def mirror(self, kind, active, delta, history, prefix, in_place, order='ascending'):
        card = self.card
        active = active.clone()
        out = active if in_place else torch.full(card.KV_SHAPE, 0x1234, dtype=torch.int16)   # a poisoned spare
        card.mirror_slide(torch, kind, active, delta.clone(), out, history_rows=history, prefix=prefix, order=order)
        return out

    def test_the_shapes_are_the_twins(self):
        self.assertEqual((self.card.KV_SHAPE, self.card.DELTA_SHAPE, self.card.WORKERS), (KV_SHAPE, DELTA_SHAPE, 8))
        with four_cards():
            self.assertEqual((twin.kv_shape(), twin.delta_shape(), twin.workers_per_bank()),
                             (self.card.KV_SHAPE, self.card.DELTA_SHAPE, self.card.WORKERS))

    def test_in_place_is_out_of_place_is_the_eager_chain_is_the_oracle_for_every_history_and_prefix(self):
        card = self.card
        # A valid active window: the rows past history_rows are zero, as every published bank holds them. The delta's rows past the
        # prefix and the spare are random (poison): a slide that reads or keeps them shows.
        for history in RAMP_HISTORIES:
            active = card.random_bits(torch, card.KV_SHAPE, 1000 + history)
            active[:, :, history:] = 0
            delta = card.random_bits(torch, card.DELTA_SHAPE, 2000 + history)
            for kind in ('scalar', 'direct'):
                for prefix in range(1, 17):
                    with self.subTest(kind=kind, history=history, prefix=prefix):
                        expected = card.oracle(torch, active, delta, history, prefix)
                        eager = eager_chain(active, delta, history, prefix)
                        modelled = slide_tests.kernel_model(history, prefix, active, delta)
                        oop = self.mirror(kind, active, delta, history, prefix, False)
                        inplace = self.mirror(kind, active, delta, history, prefix, True)
                        for label, value in (('oracle', expected), ('eager', eager), ('rule', modelled), ('oop', oop)):
                            self.assertTrue(torch.equal(inplace, value), label)

    def test_a_ramp_history_with_poison_past_it_is_rewritten_to_zero(self):
        """R3c's premise: at history under 2048 the rows past history + prefix of the bank hold whatever was there; the slide's output
        zeros them, in place as out of place."""
        card = self.card
        for kind in ('scalar', 'direct'):
            for history, prefix in ((17, 16), (1000, 16), (2031, 16)):
                active = card.random_bits(torch, card.KV_SHAPE, 77 + history)
                delta = card.random_bits(torch, card.DELTA_SHAPE, 78 + history)
                inplace = self.mirror(kind, active, delta, history, prefix, True)
                self.assertTrue(torch.equal(inplace[:, :, history + prefix:], torch.zeros_like(inplace[:, :, history + prefix:])))
                self.assertTrue(torch.equal(inplace[:, :, :history], active[:, :, :history]))
                self.assertTrue(torch.equal(inplace[:, :, history:history + prefix], delta[:, :, :prefix]))

    def test_no_bank_page_is_read_after_its_worker_wrote_it_for_any_of_the_eight_workers(self):
        card = self.card
        for kind in ('scalar', 'direct'):
            for history in RAMP_HISTORIES + (2044, 2046):
                for prefix in range(1, 33):
                    for worker in range(8):
                        self.assertEqual(card.in_place_hazards(kind, history, prefix, worker), [], (kind, history, prefix, worker))

    def test_the_negative_control_fires_at_two_heads(self):
        card = self.card
        for kind in ('scalar', 'direct'):
            for prefix in (1, 16):
                self.assertTrue(card.in_place_hazards(kind, 2048, prefix, 7, order='descending'), (kind, prefix))
            active = card.random_bits(torch, card.KV_SHAPE, 5)
            delta = card.random_bits(torch, card.DELTA_SHAPE, 6)
            expected = card.oracle(torch, active, delta, 2048, 16)
            self.assertTrue(torch.equal(self.mirror(kind, active, delta, 2048, 16, False, 'descending'), expected))
            self.assertFalse(torch.equal(self.mirror(kind, active, delta, 2048, 16, True, 'descending'), expected))

    def test_the_kernels_are_pinned_and_head_count_generic(self):
        card = self.card
        self.assertEqual(hashlib.sha256((HERE / 'draft_kv_slide.cpp').read_bytes()).hexdigest(), card.SCALAR_SHA256)
        self.assertEqual(hashlib.sha256((HERE / 'draft_kv_slide_direct.cpp').read_bytes()).hexdigest(), card.DIRECT_SHA256)
        for name in ('draft_kv_slide.cpp', 'draft_kv_slide_direct.cpp'):
            self.assertIn('const uint32_t head = worker / 4;', (HERE / name).read_text())


# ---------------------------------------------------------------------------------------------
# The fused commit over a fake four-card, four-user block.
# ---------------------------------------------------------------------------------------------

class Ops4(base.FusedOps):
    """base.FusedOps with four shards a device tensor, and traces that run their registered effects."""

    def __init__(self):
        super().__init__()
        self.effects = {}

    def allocate(self, shape, dtype='bf16', layout='tile', value=None, mapper=None):
        tensor = SimpleNamespace(shape=tuple(shape), dtype=dtype, layout=layout, value=value, mapper=mapper)
        tensor.shards = [tpv.FakeShard(tensor, next(tpv._addresses)) for chip in range(CHIPS)]
        return tensor

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        super().execute_trace(mesh, trace, cq_id=cq_id, blocking=blocking)
        effect = self.effects.get(trace)
        if effect is not None:
            effect()


def pooled_slot(ops, index):
    kv = [{side: {name: ops.allocate(KV_SHAPE, value=torch.zeros(KV_SHAPE, dtype=torch.bfloat16)) for name in HEADS}
           for side in ('active', 'spare')} for layer in range(5)]
    return SimpleNamespace(index=index, lent=False, kv=kv, query=ops.allocate((1, 1, 32, 2048)),
                           verifier=SimpleNamespace(carry=tpv.snapshot_set(ops)))


class TwinFixture(unittest.TestCase):
    """A twin FusedCommit over a fake four-user block at QWEN_FAST_TP=4 with the four-card slide on."""

    INPLACE = True
    AUDIT = False

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(clean_environment(QWEN_FAST_TP_KV_SLIDE='1'))
        stack.enter_context(four_cards())
        self.ops = Ops4()
        self.mesh = grid_mesh(CHIPS)
        self.slots = [pooled_slot(self.ops, index) for index in range(4)]
        self.weights = base.shared_weights(self.ops)
        self.collectives = SimpleNamespace(name='shared-ccl')
        self.taps = tuple(self.ops.allocate((1, 1, 64, 1280)) for tap in range(5))
        self.block = SimpleNamespace(rows_per_user=16, users=4, shape=m3_shape(PAGE_WIDTH), segment_slots=self.slots,
                                     taps=self.taps, rounds=3, segment_of=lambda engine: engine.segment)
        self.calls = []
        for target, value in (('slide_program', Mock(side_effect=self.program)),
                              ('_project_features', Mock(side_effect=self.project_features)),
                              ('_project_key_value', Mock(side_effect=self.project_key_value))):
            stack.enter_context(patch.object(twin, target, value))
        self.lines, loguru = base.logged()
        stack.enter_context(loguru)
        self.traces = iter(range(1, 10 ** 6))

    def program(self, ttnn, grid, source, banks, *, history_rows, prefix, in_place=True):
        return SimpleNamespace(prefix=prefix, banks=tuple(bank for bank, delta in banks), history_rows=history_rows,
                               source=source, in_place=in_place)

    def project_features(self, owner, features, count_, row_offset, retain):
        self.calls.append(('project_features', owner, tuple(features), count_, row_offset))
        return retain(self.ops.allocate((1, 1, count_, 5120)))

    def project_key_value(self, operations, inputs, query, tables, retain, *, parameters):
        self.calls.append(('project_key_value', inputs.shape, query, tuple(tables), parameters['name']))
        zeros = lambda: torch.zeros(DELTA_SHAPE, dtype=torch.bfloat16)
        return dict(q=retain(self.ops.allocate((1, 8, 32, 128))), k=retain(self.ops.allocate(DELTA_SHAPE, value=zeros())),
                    v=retain(self.ops.allocate(DELTA_SHAPE, value=zeros())))

    def capture(self, operations, grid, operation):
        self.ops.log.append(('capture', getattr(operation, '__qualname__', '?')))
        operation()
        return 'trace%d' % next(self.traces), None

    def build(self, **options):
        values = dict(operations=self.ops, mesh=self.mesh, pool=None, shared_weights=self.weights,
                      collectives=self.collectives, inplace=self.INPLACE, audit=self.AUDIT)
        values.update(options)
        return twin.FusedCommit(self.block, **values)

    def built(self):
        fused = self.build()
        fused.capture(self.capture)
        self.ops.log.clear()
        self.calls.clear()
        return fused


class ConstructionTests(TwinFixture):
    def test_every_segment_allocates_two_tables_and_ten_two_head_deltas_and_nothing_else(self):
        fused = self.build()
        self.assertEqual(len(fused.allocated()), 4 * 12)
        self.assertEqual(self.ops.device_uploads, fused.allocated())
        for storage in fused.segments:
            self.assertEqual([tuple(table.shape) for table in storage.tables], [(1, 1, 32, 128)] * 2)
            self.assertEqual({tuple(storage.deltas[layer][name].shape) for layer in range(5) for name in HEADS}, {DELTA_SHAPE})
        self.assertEqual([storage.row_offset for storage in fused.segments], [0, 16, 32, 48])
        self.assertEqual((fused.chips, fused.per_program), (4, 10))

    def test_host_checks_refuse_before_anything_is_allocated(self):
        cases = dict(collectives=dict(collectives=None), weights=dict(shared_weights=SimpleNamespace(layers=())))
        for name, options in cases.items():
            with self.subTest(name=name), self.assertRaises(twin.Refused):
                self.build(**options)
        self.slots[2].kv = self.slots[2].kv[:4]
        with self.assertRaises(twin.Refused):
            self.build()
        self.slots[2].kv = pooled_slot(self.ops, 2).kv
        for name, mesh in (('the pairs mesh', grid_mesh(2)), ('a square mesh', SimpleNamespace(
                shape=[2, 2], compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))),
                           ('a small grid', grid_mesh(CHIPS, 4, 4))):
            with self.subTest(name=name), self.assertRaises(twin.Refused):
                self.build(mesh=mesh)
        with patch.object(twin, 'kernel_path', return_value=HERE / 'fused_commit_tp.py'), self.assertRaises(twin.Refused):
            self.build()
        self.assertEqual(self.ops.device_uploads, [])

    def test_the_slide_flag_is_the_scope_and_the_out_of_place_mode_does_not_need_the_grid(self):
        with clean_environment(), four_cards():
            with self.assertRaisesRegex(twin.Refused, 'QWEN_FAST_TP_KV_SLIDE'):
                self.build()
        self.build(inplace=False, mesh=grid_mesh(CHIPS, 2, 2)).release_buffers()

    def test_build_logs_a_refusal_and_serves_today_and_names_the_flag_gate(self):
        lines = []
        with clean_environment(QWEN_FAST_FUSED_COMMIT='1', QWEN_FAST_TP_KV_SLIDE='1'), four_cards():
            self.assertIsNone(twin.build(self.block, operations=self.ops, mesh=self.mesh, pool=None,
                                         shared_weights=self.weights, collectives=None, diagnostic=lines.append))
        self.assertTrue(lines[0].startswith(pinned.REFUSED_MARKER + ' users=4 reason=no_collectives'))
        with clean_environment(QWEN_FAST_TP_KV_SLIDE='1'), four_cards():
            self.assertIsNone(twin.build(self.block, operations=self.ops, mesh=self.mesh, pool=None,
                                         shared_weights=self.weights, collectives=self.collectives, diagnostic=lines.append))
        with clean_environment(QWEN_FAST_FUSED_COMMIT='1', QWEN_FAST_FUSED_COMMIT_INPLACE='1', QWEN_FAST_FUSED_COMMIT_AUDIT='1',
                               QWEN_FAST_TP_KV_SLIDE='1'), four_cards():
            built = twin.build(self.block, operations=self.ops, mesh=self.mesh, pool=None, shared_weights=self.weights,
                               collectives=self.collectives, diagnostic=lines.append)
        self.assertIsInstance(built, twin.FusedCommit)
        self.assertEqual((built.inplace, built.audit), (True, True))
        built.release_buffers()


class CaptureTests(TwinFixture):
    def test_t_proj_is_the_feature_projection_the_pad_and_five_kv_projections_into_the_deltas(self):
        fused = self.build()
        fused.capture(self.capture)
        owner_calls = [call for call in self.calls if call[0] == 'project_features']
        self.assertEqual([call[3:] for call in owner_calls],
                         [(16, 0), (16, 0), (16, 16), (16, 16), (16, 32), (16, 32), (16, 48), (16, 48)])
        owner = owner_calls[4][1]
        self.assertIs(owner.projection, self.weights.projection)
        self.assertIs(owner.collectives, self.collectives)
        kv_calls = [call for call in self.calls if call[0] == 'project_key_value']
        self.assertEqual(len(kv_calls), 4 * 2 * 5)
        self.assertEqual({call[1] for call in kv_calls}, {(1, 1, 32, 5120)})
        copies = [entry for entry in self.ops.log if entry[0] == 'copy']
        self.assertEqual(len(copies), 4 * 2 * 10)

    def test_in_place_sixty_four_slide_traces_follow_four_projection_traces_each_warmed_first(self):
        fused = self.build()
        fused.capture(self.capture)
        self.assertEqual(fused.trace_count(), 4 + 64)
        for storage in fused.segments:
            self.assertEqual(sorted(storage.slides), list(range(1, 17)))
            for prefix, programs in storage.slide_programs.items():
                self.assertEqual(len(programs), 1, 'one program a user: ten banks, 80 workers')
                chunk, program = programs[0]
                self.assertEqual(len(chunk), 10)
                self.assertEqual((program.prefix, program.history_rows, program.in_place), (prefix, 2048, True))
                self.assertEqual(list(program.banks), [bank for bank, delta in storage.banks()])
        captures = [entry for entry in self.ops.log if entry[0] == 'capture']
        self.assertEqual(len(captures), 68)
        # every (segment, prefix) program is run once before any capture (compiled), then captured: two generic_ops (one program of ten banks) each
        self.assertEqual(sum(1 for entry in self.ops.log if entry[0] == 'generic_op'), 4 * 16 * 2)

    def test_the_fallback_layout_is_two_programs_of_forty_workers(self):
        with patch.dict(os.environ, {twin.BANKS_FLAG: '5'}):
            fused = self.build()
            fused.capture(self.capture)
        for storage in fused.segments:
            for programs in storage.slide_programs.values():
                self.assertEqual([len(chunk) for chunk, program in programs], [5, 5])
        self.assertIn('layout=2x40', fused.engaged_line())

    def test_out_of_place_captures_only_the_projections(self):
        fused = self.build(inplace=False)
        fused.capture(self.capture)
        self.assertEqual(fused.trace_count(), 4)
        self.assertEqual([storage.slides for storage in fused.segments], [{}] * 4)
        self.assertEqual(len([entry for entry in self.ops.log if entry[0] == 'generic_op']), 0)

    def test_the_retainer_keeps_protected_buffers_and_refuses_a_partial_alias_on_any_of_the_four_chips(self):
        fused = self.build()
        storage = fused.segments[1]
        storage.taps = tuple(self.taps)
        owned = []
        retain = fused.retainer(storage, owned)
        for protected in (*storage.taps, storage.query, storage.tables[0], storage.deltas[3]['k'], self.weights.projection):
            self.assertIs(retain(protected), protected)
        self.assertEqual(owned, [])
        fresh = self.ops.allocate((1, 1, 32, 5120))
        self.assertIs(retain(fresh), fresh)
        self.assertEqual(owned, [fresh])
        for chip in range(CHIPS):
            shifted = self.ops.allocate((1, 1, 32, 128))
            shifted.shards[chip].address = storage.deltas[0]['v'].shards[chip].address
            with self.subTest(chip=chip), self.assertRaises(ValueError):
                retain(shifted)

    def test_the_timing_replays_measure_the_widest_trace_of_every_segment_and_are_reported(self):
        fused = self.build()
        fused.capture(self.capture)
        self.assertEqual(set(fused.timing), {'tproj_ms', 'slide_ms'})
        line = fused.engaged_line()
        self.assertTrue(line.startswith('[PINDIAG] fused commit engaged users=4 rows=16 inplace=1 live_banks=0 audit=0 kernel='), line)
        self.assertRegex(line, r' traces=68 layout=1x80 tp=4 workers=8 tproj_ms=[\d.]+ slide_ms=[\d.]+$')
        # the gate's regex is the pinned one: it reads the same line
        import lever_n_m3native_gate as gate

        report = gate.h1b_report({'QWEN_FAST_FUSED_COMMIT': '1', 'QWEN_FAST_FUSED_COMMIT_INPLACE': '1'}, line + '\n')
        self.assertNotIn('no engaged line', ' '.join(report['problems']).lower())

    def test_close_releases_every_trace_and_buffer_but_never_a_pooled_bank(self):
        fused = self.build()
        fused.capture(self.capture)
        fused.close()
        self.assertEqual(len(self.ops.released), 68)
        self.assertTrue(fused.closed)
        pooled = {id(value) for slot in self.slots for layer in slot.kv for side in layer.values() for value in side.values()}
        self.assertFalse(pooled & {id(value) for value in self.ops.deallocated})
        self.assertEqual(fused.trace_count(), 0)


def fused_drafter(fixture, fused, segment, *, position=5000, history_rows=2048, parity=False, cache_class=None):
    """A DFlashDevice and its four-card DraftKVHistory, built without their constructors, on the segment's slot."""
    ops, slot = fixture.ops, fixture.slots[segment]
    cache = object.__new__(cache_class or draft_kv_history_tp.DraftKVHistory)
    active = [dict(layer['active']) for layer in slot.kv]
    spare = [dict(layer['spare']) for layer in slot.kv]
    cache.active, cache.spare = (spare, active) if parity else (active, spare)
    cache.operations, cache.mesh, cache.parameters = ops, fixture.mesh, fused.parameters
    cache.position, cache.history_rows, cache.pending, cache.closed = position, history_rows, None, False
    cache.owned, cache.borrowed, cache.checks, cache.projection, cache.query = [], [], [], None, slot.query
    device = object.__new__(DFlashDevice)
    device.operations, device.mesh, device.kv_history = ops, fixture.mesh, cache
    device.position, device.history_rows, device.pending, device.closed = position, 2048, None, False
    device.progress, device.proposal_capture, device.pool_slot = None, object(), slot
    device.projection, device.feature_norm = fixture.weights.projection, fixture.weights.feature_norm
    device.collectives = fixture.collectives
    device.history, device.spare_history, device.published_rows = 'history', 'spare-history', 0
    return device


def taps_for(fixture, segment):
    return PackedFeatureTaps(fixture.taps, row_offset=16 * segment, rows=16)


class PublicationTests(TwinFixture):
    def setUp(self):
        super().setUp()
        self.fused = self.built()
        self.today = Mock(side_effect=lambda device, features, prefix, **options: ('today', prefix, options))

        def prepare_publication(device, features, prefix, **options):
            return self.today(device, features, prefix, **options)

        patcher = patch.object(DFlashDevice, 'prepare_publication', prepare_publication)
        patcher.start()
        self.addCleanup(patcher.stop)

    def publish(self, device, segment, prefix, *, position=None):
        restore = twin.install_fused_commit(device, self.fused, segment, merge_release=False, fused_steady_state=False)
        try:
            publication = device.prepare_publication(taps_for(self, segment), prefix,
                                                     position=device.position if position is None else position)
            if isinstance(publication, SimpleNamespace):
                device.commit_publication(publication)
            return publication
        finally:
            restore()

    def test_a_fused_publication_enqueues_t_proj_then_the_slide_with_no_fence_and_commits_without_a_swap(self):
        device = fused_drafter(self, self.fused, 1)
        self.fused.stage_tables(1, 5000)
        self.ops.log.clear()
        cache = device.kv_history
        active, spare = cache.active, cache.spare
        publication = self.publish(device, 1, 7)
        storage = self.fused.segments[1]
        self.assertEqual(self.ops.log, [('execute_trace', storage.projection_trace, False),
                                        ('execute_trace', storage.slides[7], False)])
        self.today.assert_not_called()
        self.assertEqual(publication.status, 'committed')
        self.assertEqual((device.position, cache.position, cache.history_rows), (5007, 5007, 2048))
        self.assertIs(cache.active, active)
        self.assertIs(cache.spare, spare)
        self.assertTrue(device.history_stale)
        self.assertEqual(self.lines[-1], '[PACKED-FUSED] round=3 segment=1 prefix=7 path=fused reason=- tables=window')

    def test_a_late_table_is_staged_before_t_proj(self):
        device = fused_drafter(self, self.fused, 0, position=6000)
        self.publish(device, 0, 3)
        storage = self.fused.segments[0]
        self.assertEqual([entry[0] for entry in self.ops.log], ['host_copy', 'host_copy', 'execute_trace', 'execute_trace'])
        for table, expected in zip(storage.tables, pinned.host_tables(6000, 32, 16)):
            self.assertTrue(torch.equal(table.value, expected))
        self.assertTrue(self.lines[-1].endswith('tables=late'))

    def test_every_refusal_takes_todays_path_argument_for_argument(self):
        def refuse(name, device):
            if name == 'ramp':
                device.kv_history.history_rows = 2000
            elif name == 'slot':
                device.pool_slot = self.slots[0]
            elif name == 'progress':
                device.progress = Mock()
            elif name == 'no-capture':
                device.proposal_capture = None
            elif name == 'projection':
                device.kv_history.projection = object()
            elif name == 'weights':
                device.projection = object()
            elif name == 'parameters':
                device.kv_history.parameters = tuple(dict(name='other') for layer in range(5))
            elif name == 'collectives':
                device.collectives = object()
            elif name == 'poisoned':
                self.fused.segments[2].poisoned = device.kv_history
            elif name == 'no-kv':
                device.kv_history = None
            elif name == 'frontier':
                device.position = device.kv_history.position = 262144 - 20

        for name in ('ramp', 'slot', 'progress', 'no-capture', 'projection', 'weights', 'parameters', 'collectives',
                     'poisoned', 'no-kv', 'frontier'):
            with self.subTest(reason=name):
                self.today.reset_mock()
                self.ops.log.clear()
                device = fused_drafter(self, self.fused, 2)
                refuse(name, device)
                result = self.publish(device, 2, 4)
                self.assertEqual(result, ('today', 4, {'position': device.position}))
                self.today.assert_called_once()
                self.assertEqual(self.ops.log, [])
                self.assertTrue(self.lines[-1].endswith('path=today reason=%s tables=-' % name), self.lines[-1])
                self.fused.segments[2].poisoned = None

    def test_the_scope_refusal_is_the_flag_and_the_four_card_cache_class(self):
        for name in ('flag off', 'the pairs cache class'):
            with self.subTest(name=name):
                self.today.reset_mock()
                device = fused_drafter(self, self.fused, 2, cache_class=None if name == 'flag off'
                                       else draft_kv_history.DraftKVHistory)
                environment = clean_environment() if name == 'flag off' else clean_environment(QWEN_FAST_TP_KV_SLIDE='1')
                with environment:
                    result = self.publish(device, 2, 4)
                self.assertEqual(result[0], 'today')
                self.assertTrue(self.lines[-1].endswith('path=today reason=scope tables=-'), self.lines[-1])
        self.assertIsNone(self.fused.refusal(fused_drafter(self, self.fused, 1), 1, taps_for(self, 1), 4, 5000))

    def test_features_prefix_and_pending_refusals(self):
        device = fused_drafter(self, self.fused, 2)
        restore = twin.install_fused_commit(device, self.fused, 2, merge_release=False, fused_steady_state=False)
        try:
            self.assertEqual(device.prepare_publication(taps_for(self, 1), 4, position=5000)[0], 'today')
            self.assertTrue(self.lines[-1].endswith('reason=features tables=-'))
            self.assertEqual(device.prepare_publication(taps_for(self, 2), 17, position=5000)[0], 'today')
            self.assertTrue(self.lines[-1].endswith('reason=prefix tables=-'))
            self.assertEqual(device.prepare_publication(taps_for(self, 2), 4, position=4999)[0], 'today')
            self.assertTrue(self.lines[-1].endswith('reason=pending tables=-'))
        finally:
            restore()

    def test_the_guard_order_is_the_pinned_ones(self):
        """The four-card refusal is the pinned method with the scope check replaced: the same reasons, in the same order, for the
        same states (a state that trips two guards reports the earlier)."""
        device = fused_drafter(self, self.fused, 2)
        device.kv_history.history_rows = 100          # ramp
        device.progress = Mock()                      # progress, later in the list
        self.assertEqual(self.fused.refusal(device, 2, taps_for(self, 2), 4, 5000), 'ramp')
        device.kv_history.history_rows = 2048
        device.pool_slot = self.slots[0]              # slot before progress
        self.assertEqual(self.fused.refusal(device, 2, taps_for(self, 2), 4, 5000), 'slot')
        device = fused_drafter(self, self.fused, 2, parity=True)
        with clean_environment():
            self.assertEqual(self.fused.refusal(device, 2, taps_for(self, 2), 4, 5000), 'parity', 'parity before scope')

    def test_parity_takes_today_once_and_its_swap_normalises_it(self):
        device = fused_drafter(self, self.fused, 0, parity=True)
        cache = device.kv_history

        def today(drafter, features, prefix, *, position, **options):
            publication = SimpleNamespace(position=position, prefix=prefix, rows=2048, status='prepared')
            cache.pending = publication
            drafter.pending = SimpleNamespace(position=position, prefix=prefix, rows=2048, history='spare-history',
                                              kv=publication, status='prepared')
            return drafter.pending

        self.today.side_effect = today
        self.publish(device, 0, 6)
        self.assertTrue(self.lines[-1].endswith('reason=parity tables=-'))
        self.assertIs(cache.active[0]['k'], self.slots[0].kv[0]['active']['k'], "today's commit swapped it back")
        self.publish(device, 0, 2)
        self.assertTrue(self.lines[-1].startswith('[PACKED-FUSED] round=3 segment=0 prefix=2 path=fused'))
        self.assertEqual(device.position, 5008)

    def test_out_of_place_runs_the_four_card_transport_from_the_deltas_and_keeps_the_swap(self):
        fused = self.build(inplace=False)
        fused.capture(self.capture)
        self.fused = fused
        device = fused_drafter(self, fused, 2)
        cache = device.kv_history
        active, spare = cache.active, cache.spare
        transports = []
        transport = Mock(side_effect=lambda grid, a, d, s, **options: transports.append((a, d, s, options)) or (lambda: None))
        self.ops.log.clear()
        with patch.object(twin, '_transport', return_value=transport):
            self.publish(device, 2, 9)
        storage = fused.segments[2]
        self.assertEqual(self.ops.log[-1], ('execute_trace', storage.projection_trace, False))
        self.assertEqual(transports, [(active[layer][name], storage.deltas[layer][name], spare[layer][name],
                                       dict(history_rows=2048, prefix=9)) for layer in range(5) for name in HEADS])
        self.assertIs(cache.active, spare)
        self.assertIs(cache.spare, active)
        self.assertEqual(cache.position, 5009)
        self.assertIs(twin._transport(), draft_kv_slide_tp.prepare)
        self.assertIsNot(twin._transport(), draft_kv_slide.prepare)

    def test_a_discard_after_the_in_place_slide_poisons_the_segment(self):
        device = fused_drafter(self, self.fused, 1)
        restore = twin.install_fused_commit(device, self.fused, 1, merge_release=False, fused_steady_state=False)
        try:
            publication = device.prepare_publication(taps_for(self, 1), 4, position=5000)
            device.discard_publication(publication)
        finally:
            restore()
        self.assertIs(self.fused.segments[1].poisoned, device.kv_history)
        self.assertTrue(any(line.startswith(pinned.DISCARD_MARKER) for line in self.lines))
        self.assertEqual(self.fused.refusal(device, 1, taps_for(self, 1), 4, 5000), 'poisoned')

    def test_a_failed_slide_enqueue_poisons_and_raises(self):
        device = fused_drafter(self, self.fused, 1)
        storage = self.fused.segments[1]
        storage.slides[4] = 'broken'
        original = self.ops.execute_trace

        def execute(grid, trace, cq_id=0, blocking=True):
            if trace == 'broken':
                raise RuntimeError('enqueue')
            return original(grid, trace, cq_id=cq_id, blocking=blocking)

        self.ops.execute_trace = execute
        restore = twin.install_fused_commit(device, self.fused, 1, merge_release=False, fused_steady_state=False)
        try:
            with self.assertRaises(RuntimeError):
                device.prepare_publication(taps_for(self, 1), 4, position=5000)
        finally:
            restore()
        self.assertIs(storage.poisoned, device.kv_history)
        self.assertTrue(any('path=failed poisoned=1' in line for line in self.lines))

    def test_the_round_b1_splits_are_added_when_the_sink_is_set(self):
        from dflash_traced_publish import PUBLICATION_SPLITS

        device = fused_drafter(self, self.fused, 0)
        splits = {}
        token = PUBLICATION_SPLITS.set(splits)
        try:
            self.publish(device, 0, 1)
        finally:
            PUBLICATION_SPLITS.reset(token)
        self.assertEqual(set(splits), {'kv_in', 'kv_exec'})


class WindowTests(TwinFixture):
    def test_the_window_stages_each_live_segment_for_its_next_frontier_once(self):
        fused = self.built()
        requests = [SimpleNamespace(engine=SimpleNamespace(segment=segment),
                                    session=SimpleNamespace(finished=segment == 3, position=7000 + segment)) for segment in range(4)]
        fused.stage_window(requests)
        self.assertEqual([storage.staged_position for storage in fused.segments], [7000, 7001, 7002, None])
        self.ops.log.clear()
        fused.stage_window(requests)
        self.assertEqual(self.ops.log, [], 'already staged: nothing is written twice')


# ---------------------------------------------------------------------------------------------
# A request's life: the fused path against the eager one, after every commit.
# ---------------------------------------------------------------------------------------------

def truth(segment, position, layer, name):
    """The 16 projected rows of the block at `position` for one segment, layer and head: what T_proj writes (E2: row i is the same
    whatever the prefix, because every op is row-local), as a (1, 2, 16, 128) bf16 tensor."""
    generator = torch.Generator().manual_seed(hash((segment, position, layer, name)) % (2 ** 31))
    return torch.randn((1, KV_HEADS, 16, 128), generator=generator).bfloat16()


def valid_bank(seed, history_rows):
    generator = torch.Generator().manual_seed(seed)
    bank = torch.randn(KV_SHAPE, generator=generator).bfloat16()
    bank[:, :, history_rows:] = 0     # every published bank holds zeros past its rows
    return bank


def bits(value):
    return value.contiguous().view(torch.int16)


class EagerWorld:
    """The pure eager sequence (draft_kv_history_tp.prepare with the flag off, and the real commit): the reference the fused
    commit is compared with after every commit."""

    def __init__(self, segment, seed, history_rows, position, ops):
        self.segment, self.ops = segment, ops
        make = lambda offset: [{name: ops.allocate(KV_SHAPE, value=valid_bank(seed + 10 * layer + offset + (5 if name == 'v' else 0),
                                                                                history_rows if offset == 0 else 0))
                                for name in HEADS} for layer in range(5)]
        self.cache = object.__new__(draft_kv_history_tp.DraftKVHistory)
        self.cache.__dict__.update(operations=ops, mesh='mesh', parameters=(), position=position, history_rows=history_rows,
                                   owned=[], checks=[], projection=None, pending=None, closed=False, query=None,
                                   active=make(0), spare=make(1), borrowed=[])

    def prepare(self, prefix, position):
        return eager_prepare(self.cache, self.segment, prefix, position)

    def commit(self, publication):
        self.cache.commit(publication)


def eager_prepare(cache, segment, prefix, position):
    """DraftKVHistory_tp.prepare's flag-off chain in torch, into the spare: the truth rows [:prefix] as the K/V projection's."""
    if cache.closed or cache.pending is not None or position != cache.position:
        raise ValueError('One accepted-prefix cache update at the current committed frontier required')
    rows = min(2048, cache.history_rows + prefix)
    for layer, (active, spare) in enumerate(zip(cache.active, cache.spare)):
        for name in HEADS:
            delta = truth(segment, position, layer, name)
            padded = torch.zeros(DELTA_SHAPE, dtype=torch.bfloat16)
            padded[:, :, :prefix] = delta[:, :, :prefix]     # today's project_inputs pads the rows past the prefix with zeros
            spare[name].value = eager_chain(bits(active[name].value), bits(padded), cache.history_rows, prefix).view(torch.bfloat16)
    cache.pending = SimpleNamespace(position=position, prefix=prefix, rows=rows, status='prepared')
    return cache.pending


class LifeTests(TwinFixture):
    """One request through a segment: ramp commits (today's out-of-place path and its swap), the parity refusal, fused steady
    commits, and sequential steps between packed rounds."""

    def setUp(self):
        super().setUp()
        self.fused = self.built()
        self.worlds = {}

        def today(device, features, prefix, *, position, **options):
            segment = device.pool_slot.index
            publication = eager_prepare(device.kv_history, segment, prefix, position)
            device.pending = SimpleNamespace(position=position, prefix=prefix, rows=publication.rows,
                                             history=device.spare_history, kv=publication, status='prepared')
            return device.pending

        self.today = today
        patcher = patch.object(DFlashDevice, 'prepare_publication', lambda device, features, prefix, **options:
                               self.today(device, features, prefix, **options))
        patcher.start()
        self.addCleanup(patcher.stop)
        # What the traces do on the card, from the state the fused commit itself staged: T_proj writes the block's 16 projected rows
        # into the deltas (zeros past row 16), each slide trace slides the slot's active banks in place by the kernel's rule.
        for segment, storage in enumerate(self.fused.segments):
            self.ops.effects[storage.projection_trace] = lambda storage=storage, segment=segment: self.t_proj(storage, segment)
            for prefix, trace in storage.slides.items():
                self.ops.effects[trace] = lambda storage=storage, prefix=prefix: self.slide(storage, prefix)

    def t_proj(self, storage, segment):
        self.assertIsNotNone(storage.staged_position, 'T_proj replays against the tables the commit staged')
        for layer in range(5):
            for name in HEADS:
                rows = truth(segment, storage.staged_position, layer, name)
                delta = torch.zeros(DELTA_SHAPE, dtype=torch.bfloat16)
                delta[:, :, :16] = rows
                storage.deltas[layer][name].value = delta

    def slide(self, storage, prefix):
        for (bank, delta) in storage.banks():
            bank.value = slide_tests.kernel_model(2048, prefix, bits(bank.value), bits(delta.value)).view(torch.bfloat16)

    def publish(self, device, segment, prefix):
        restore = twin.install_fused_commit(device, self.fused, segment, merge_release=False, fused_steady_state=False)
        try:
            publication = device.prepare_publication(taps_for(self, segment), prefix, position=device.position)
            device.commit_publication(publication)
        finally:
            restore()

    def assert_same(self, device, world, label):
        cache, reference = device.kv_history, world.cache
        self.assertEqual((cache.position, cache.history_rows, device.position), (reference.position, reference.history_rows,
                                                                                 reference.position), label)
        for layer in range(5):
            for name in HEADS:
                for chip, shard in enumerate(self.ops.get_device_tensors(cache.active[layer][name])):
                    self.assertTrue(torch.equal(bits(self.ops.to_torch(shard)), bits(reference.active[layer][name].value)),
                                    (label, layer, name, chip))

    def life(self, segment, *, start_rows, ramp_prefixes, steady_prefixes, sequential_after=(), position=5000):
        seed = 100 * segment + start_rows
        device = fused_drafter(self, self.fused, segment, position=position, history_rows=start_rows)
        slot = self.slots[segment]
        world = EagerWorld(segment, seed, start_rows, position, self.ops)
        for layer in range(5):
            for name in HEADS:
                slot.kv[layer]['active'][name].value = world.cache.active[layer][name].value.clone()
                slot.kv[layer]['spare'][name].value = world.cache.spare[layer][name].value.clone()
        self.assert_same(device, world, 'start')
        history = []

        def eager_round(prefix):
            publication = world.prepare(prefix, world.cache.position)
            world.commit(publication)

        for step, prefix in enumerate(ramp_prefixes):
            self.publish(device, segment, prefix)
            eager_round(prefix)
            self.assert_same(device, world, 'ramp %d prefix %d' % (step, prefix))
            history.append(self.lines[-1])
        for step, prefix in enumerate(steady_prefixes):
            if step in sequential_after:
                # a solo step between packed rounds: today's out-of-place commit and swap, outside the fused overrides
                publication = self.today(device, None, 3, position=device.position)
                device.commit_publication(publication)
                eager_round(3)
                self.assert_same(device, world, 'sequential before steady %d' % step)
            active_before = device.kv_history.active
            self.publish(device, segment, prefix)
            eager_round(prefix)
            self.assert_same(device, world, 'steady %d prefix %d' % (step, prefix))
            history.append(self.lines[-1])
            if history[-1].split('path=')[1].startswith('fused'):
                self.assertIs(device.kv_history.active, active_before, 'a fused commit never swaps')
        return device, world, history

    def paths(self, history):
        return [line.split('path=')[1].split(' ')[0] + ('/' + line.split('reason=')[1].split(' ')[0]
                if 'path=today' in line else '') for line in history]

    def test_every_segment_an_even_ramp_then_every_prefix_steady(self):
        for segment in range(4):
            with self.subTest(segment=segment):
                # 1900 + 2 x 74: the ramp fills the window after an EVEN number of swaps, so the pool's active is still active
                device, world, history = self.life(segment, start_rows=1900, ramp_prefixes=[16, 16, 16, 16, 16, 16, 16, 16],
                                                   steady_prefixes=list(range(1, 17)))
                # 1900 + 8 x 16 = 2028: still ramp (8 swaps, even); a further ramp round to reach 2048 is one more swap
                self.assertEqual(self.paths(history)[:8], ['today/ramp'] * 8)

    def test_an_odd_ramp_refuses_parity_once_then_every_round_is_fused(self):
        for segment in range(4):
            with self.subTest(segment=segment):
                device, world, history = self.life(segment, start_rows=2000, ramp_prefixes=[16, 16, 16],
                                                   steady_prefixes=[16, 5, 1, 12, 16, 9])
                # 2000 + 3 x 16 = 2048: three swaps (odd): the first steady round is the parity refusal, its swap normalises
                self.assertEqual(self.paths(history), ['today/ramp', 'today/ramp', 'today/ramp', 'today/parity'] + ['fused'] * 5)

    def test_a_sequential_step_between_packed_rounds_flips_parity_and_refuses_once(self):
        # 2032 + 2 x 8 = 2048: two ramp swaps (even), so the window opens on the pool's active bank
        device, world, history = self.life(2, start_rows=2032, ramp_prefixes=[8, 8], steady_prefixes=[16, 4, 4, 9, 2, 16, 1],
                                           sequential_after=(2, 5))
        self.assertEqual(self.paths(history), ['today/ramp'] * 2 + ['fused', 'fused', 'today/parity', 'fused', 'fused',
                                                                     'today/parity', 'fused'])
        self.assertEqual(self.fused.refusals, {'ramp': 2, 'parity': 2})

    def test_a_request_that_starts_full_is_never_refused(self):
        """A prompt of 2048 tokens or more builds its cache at 2048 rows over the pool's active bank: the coding case."""
        device, world, history = self.life(0, start_rows=2048, ramp_prefixes=[], steady_prefixes=[16, 16, 11, 1, 16, 7, 16, 16])
        self.assertEqual(set(self.paths(history)), {'fused'})
        self.assertEqual(self.fused.refusals, {})
        self.assertEqual(self.fused.counts['fused'], 8)

    def test_four_users_interleaved_do_not_disturb_each_other(self):
        devices, worlds = [], []
        for segment in range(4):
            device = fused_drafter(self, self.fused, segment, position=7000 + 100 * segment, history_rows=2048)
            world = EagerWorld(segment, 900 + segment, 2048, 7000 + 100 * segment, self.ops)
            for layer in range(5):
                for name in HEADS:
                    self.slots[segment].kv[layer]['active'][name].value = world.cache.active[layer][name].value.clone()
                    self.slots[segment].kv[layer]['spare'][name].value = world.cache.spare[layer][name].value.clone()
            devices.append(device)
            worlds.append(world)
        for round_ in range(6):
            for segment in (2, 0, 3, 1):
                prefix = 1 + (7 * round_ + 5 * segment) % 16
                self.publish(devices[segment], segment, prefix)
                worlds[segment].commit(worlds[segment].prepare(prefix, worlds[segment].cache.position))
            for segment in range(4):
                self.assert_same(devices[segment], worlds[segment], 'round %d segment %d' % (round_, segment))
        self.assertEqual(self.fused.counts['fused'], 24)

    def test_t_proj_rows_past_the_prefix_are_real_and_the_bank_never_sees_them(self):
        device = fused_drafter(self, self.fused, 1, history_rows=2048)
        world = EagerWorld(1, 5, 2048, 5000, self.ops)
        for layer in range(5):
            for name in HEADS:
                self.slots[1].kv[layer]['active'][name].value = world.cache.active[layer][name].value.clone()
        self.publish(device, 1, 3)
        storage = self.fused.segments[1]
        delta = storage.deltas[0]['k'].value
        self.assertTrue(torch.count_nonzero(delta[:, :, 3:16]) > 0, 'rows 3..15 hold real projections (today holds zeros)')
        self.assertEqual(int(torch.count_nonzero(delta[:, :, 16:])), 0)
        eager = world.prepare(3, 5000)
        world.commit(eager)
        self.assert_same(device, world, 'prefix 3 with real rows past it')


# ---------------------------------------------------------------------------------------------
# The audit at four chips.
# ---------------------------------------------------------------------------------------------

class AuditTests(TwinFixture):
    AUDIT = True

    def setUp(self):
        super().setUp()
        self.fused = self.built()

    def reference_rows(self, prefix, seed):
        generator = torch.Generator().manual_seed(seed)
        return [{name: [torch.randn((1, KV_HEADS, prefix, 128), generator=generator).bfloat16()] * CHIPS for name in HEADS}
                for layer in range(5)]

    def run_audit(self, device, prefix, reference, *, break_bank=None, break_delta=None):
        storage = self.fused.segments[device.pool_slot.index]
        order = []

        def eager(drafter, features, prefix_, position, *, slide):
            order.append(('reference', slide))
            generator = torch.Generator().manual_seed(1)
            for layer in range(5):
                for name in HEADS:
                    bank = torch.randn(KV_SHAPE, generator=generator).bfloat16()
                    drafter.kv_history.spare[layer][name].value = bank.clone()
                    drafter.kv_history.active[layer][name].value = bank.clone()
                    delta = torch.zeros(DELTA_SHAPE, dtype=torch.bfloat16)
                    delta[..., :prefix_, :] = reference[layer][name][0]
                    storage.deltas[layer][name].value = delta
            if break_bank is not None:
                layer, name = break_bank
                broken = drafter.kv_history.active[layer][name].value.clone()
                broken[0, 0, 2047, 0] += 1
                drafter.kv_history.active[layer][name].value = broken
            if break_delta is not None:
                layer, name = break_delta
                broken = storage.deltas[layer][name].value.clone()
                broken[0, 1, 0, 3] += 1
                storage.deltas[layer][name].value = broken
            return reference

        restore = twin.install_fused_commit(device, self.fused, device.pool_slot.index, merge_release=False,
                                            fused_steady_state=False)
        try:
            with patch.object(self.fused, 'eager_reference', Mock(side_effect=eager)):
                publication = device.prepare_publication(taps_for(self, device.pool_slot.index), prefix, position=device.position)
                device.commit_publication(publication)
        finally:
            restore()
        return order

    def test_the_audit_checks_eighty_items_a_publication_at_four_chips(self):
        device = fused_drafter(self, self.fused, 1)
        self.ops.log.clear()
        order = self.run_audit(device, 6, self.reference_rows(6, 5))
        self.assertEqual(order, [('reference', True)])
        executes = [entry for entry in self.ops.log if entry[0] in ('execute_trace', 'sync')]
        self.assertEqual([entry[0] for entry in executes], ['execute_trace', 'execute_trace', 'sync'])
        audit = [line for line in self.lines if line.startswith(pinned.AUDIT_MARKER)]
        self.assertEqual(audit, ['[PACKED-FUSED-AUDIT] round=3 segment=1 prefix=6 mode=inplace checked=80 mismatches=0'])
        self.assertEqual(self.fused.counts['mismatches'], 0)

    def test_a_bank_mismatch_is_logged_and_repaired_from_the_spare(self):
        device = fused_drafter(self, self.fused, 2)
        self.run_audit(device, 4, self.reference_rows(4, 7), break_bank=(3, 'v'))
        cache = device.kv_history
        self.assertTrue(torch.equal(cache.active[3]['v'].value, cache.spare[3]['v'].value))
        self.assertTrue(any(line.startswith(pinned.AUDIT_MISMATCH_MARKER + ' round=3 segment=2 prefix=4 at=bank3v.0')
                            for line in self.lines), self.lines)
        # the bank is one tensor whose four shards read one value in this fake: the chips of the mismatch are all four
        self.assertEqual(self.fused.counts['mismatches'], CHIPS)

    def test_a_mismatch_on_the_last_chip_alone_is_seen_and_named(self):
        device = fused_drafter(self, self.fused, 0)
        real = self.ops.to_torch

        def to_torch(shard):
            value = real(shard)
            if getattr(shard, 'chip3', False):
                value = value.clone()
                value[0, 0, 0, 0] += 1
            return value

        for layer in range(5):
            for name in HEADS:
                device.kv_history.active[layer][name].shards[3].chip3 = True
        self.ops.to_torch = to_torch
        try:
            self.run_audit(device, 5, self.reference_rows(5, 11))
        finally:
            self.ops.to_torch = real
        mismatch = [line for line in self.lines if line.startswith(pinned.AUDIT_MISMATCH_MARKER)]
        self.assertEqual(len(mismatch), 1)
        self.assertIn('.3', mismatch[0])
        self.assertEqual(self.fused.counts['mismatches'], 10)

    def test_a_delta_mismatch_is_the_e2_claim_failing(self):
        device = fused_drafter(self, self.fused, 0)
        self.run_audit(device, 3, self.reference_rows(3, 9), break_delta=(0, 'k'))
        audit = [line for line in self.lines if line.startswith(pinned.AUDIT_MARKER)]
        self.assertTrue(audit[-1].endswith('mismatches=4'), audit)

    def test_out_of_place_audits_the_deltas_and_reruns_todays_path_on_a_mismatch(self):
        fused = self.build(inplace=False)
        fused.capture(self.capture)
        self.fused = fused
        device = fused_drafter(self, fused, 1)
        with patch.object(twin, '_transport', return_value=Mock(return_value=lambda: None)):
            order = self.run_audit(device, 5, self.reference_rows(5, 3), break_delta=(4, 'v'))
        self.assertEqual(order, [('reference', False), ('reference', True)])
        audit = [line for line in self.lines if line.startswith(pinned.AUDIT_MARKER)]
        self.assertEqual(audit, ['[PACKED-FUSED-AUDIT] round=3 segment=1 prefix=5 mode=oop checked=40 mismatches=4'])

    def test_the_reference_is_todays_publication_through_the_four_card_transport(self):
        """eager_reference at count = prefix: project_features, project_inputs, the four-card K/V projection, and (in place) the
        four-card transport of every bank into the spare - never the pair's slide driver."""
        fused = self.fused
        device = fused_drafter(self, fused, 2)
        cache = device.kv_history
        device.project_features = Mock(side_effect=lambda features, count_: self.ops.allocate((1, 1, count_, 5120)))
        cache.temporaries = lambda owned: _Scope(lambda value: value)
        cache.project_inputs = Mock(return_value=(self.ops.allocate((1, 1, 32, 5120)), ('cos', 'sin')))
        transports = []
        transport = Mock(side_effect=lambda mesh, a, d, s, **options: transports.append((a, d, s, options)) or (lambda: None))
        for layer in range(5):
            for name in HEADS:
                self.slots[2].kv[layer]['active'][name].value = torch.zeros(KV_SHAPE, dtype=torch.bfloat16)
        with patch.object(twin, '_transport', return_value=transport):
            reference = fused.eager_reference(device, taps_for(self, 2), 9, 5000, slide=True)
        self.assertEqual(len(transports), 10)
        self.assertEqual({tuple(sorted(options.items())) for a, d, s, options in transports},
                         {(('history_rows', 2048), ('prefix', 9))})
        self.assertEqual(len(reference), 5)
        self.assertEqual({len(reference[layer][name]) for layer in range(5) for name in HEADS}, {CHIPS})


class _Scope:
    def __init__(self, retain):
        self.retain = retain

    def __enter__(self):
        return self.retain

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------------------------
# E2 at four-card shapes: T_proj is today's publication at count 16, and every op in it is row-local.
# ---------------------------------------------------------------------------------------------

class ShapeTensor4(base.ShapeTensor):
    def __init__(self, ops, shape, dtype):
        self.shape, self.dtype, self.layout = tuple(shape), dtype, 'tile'
        self.shards = [tpv.FakeShard(self, next(ops.addresses)) for chip in range(CHIPS)]


class ShapeOps4(base.ShapeOps):
    """base.ShapeOps at four chips and two KV heads (8 query heads)."""

    def tensor(self, shape, dtype='bf16'):
        return ShapeTensor4(self, shape, dtype)

    def all_gather_async(self, value, **kwargs):
        self.record('all_gather_async', value, dim=kwargs['dim'], num_links=kwargs['num_links'])
        return self.tensor((value.shape[0] * CHIPS,) + value.shape[1:], value.dtype)

    def nlp_create_qkv_heads(self, query, kv, **kwargs):
        self.record('nlp_create_qkv_heads', query, kv, **kwargs)
        rows = kv.shape[2]
        return (self.tensor((1, 8, rows, 128)), self.tensor((1, KV_HEADS, rows, 128)), self.tensor((1, KV_HEADS, rows, 128)))


class OpSequenceTests(unittest.TestCase):
    def fixture(self):
        ops = ShapeOps4()
        grid = grid_mesh(CHIPS)
        collectives = SimpleNamespace(get_and_cycle_ag_semaphore_handles=lambda: 'ag',
                                      get_and_cycle_barrier_semaphore_handle=lambda: 'barrier')
        parameters = tuple(dict(operations=ops, native_head_layout=True, kernel='kv-kernel',
                                projections=dict(k=ops.tensor((5120, 256)), v=ops.tensor((5120, 256))),
                                head_norms=dict(k=ops.tensor((1, 1, 2, 32)))) for layer in range(5))
        weights = SimpleNamespace(layers=[(parameter, 'mlp', 'weights', 'conv') for parameter in parameters],
                                  projection=ops.tensor((10240, 5120)), feature_norm=ops.tensor((1, 1, 160, 32)), tensors=[])
        taps = tuple(ops.tensor((1, 1, 64, 1280)) for tap in range(5))
        slots = [SimpleNamespace(index=index, query=ops.tensor((1, 1, 32, 1024)),
                                 kv=[{side: {name: ops.tensor(KV_SHAPE) for name in HEADS} for side in ('active', 'spare')}
                                     for layer in range(5)]) for index in range(4)]
        block = SimpleNamespace(rows_per_user=16, users=4, shape=m3_shape(PAGE_WIDTH), segment_slots=slots, taps=taps, rounds=0)
        return ops, grid, collectives, weights, parameters, taps, slots, block

    def test_t_proj_logs_todays_publication_at_prefix_sixteen_call_for_call_through_the_four_card_twins(self):
        with clean_environment(QWEN_FAST_TP_KV_SLIDE='1'), four_cards():
            ops, grid, collectives, weights, parameters, taps, slots, block = self.fixture()
            with patch('feature_collective.projection_links', return_value=4):
                fused = twin.FusedCommit(block, operations=ops, mesh=grid, pool=None, shared_weights=weights,
                                         collectives=collectives, inplace=True, audit=False)
                segment, position = 2, 131000
                storage = fused.segments[segment]
                storage.taps = taps
                ops.log.clear()
                fused.project(storage, [])
                fused_log = [entry for entry in ops.log if entry[0] != 'copy']
                copies = [entry for entry in ops.log if entry[0] == 'copy']
                ops.log.clear()
                device = SimpleNamespace(operations=ops, mesh=grid, collectives=collectives, projection=weights.projection,
                                         feature_norm=weights.feature_norm, kernel=fused.kernel)
                retain = lambda value: value
                features = PackedFeatureTaps(taps, row_offset=16 * segment, rows=16)
                projected = DFlashDevice.project_features(device, features, 16, retain=retain)
                cache = SimpleNamespace(operations=ops, mesh=grid)
                cache.upload = lambda value: draft_kv_history_tp.DraftKVHistory.upload(cache, value)
                inputs, tables = draft_kv_history_tp.DraftKVHistory.project_inputs(cache, projected, 16, position, retain)
                from draft_kv_projection_tp import project_key_value

                for parameter in parameters:
                    project_key_value(ops, inputs, slots[segment].query, tables, retain, parameters=parameter)
                today_log = list(ops.log)
        uploads = [entry for entry in today_log if entry[0] == 'from_torch']
        self.assertEqual(len(uploads), 2)
        self.assertEqual([entry for entry in today_log if entry[0] != 'from_torch'], fused_log)
        self.assertGreater(len(fused_log), 60)
        self.assertEqual(len(copies), 10)
        names = [entry[0] for entry in fused_log]
        self.assertEqual(names.count('all_gather_async'), 1, 'one four-chip fp32 all-gather of the partials')
        self.assertGreaterEqual(names.count('add'), 3, 'the chip-order adds of the four fp32 partials')
        gathers = [entry for entry in fused_log if entry[0] == 'all_gather_async']
        self.assertEqual(gathers[0][1], (('tensor', (1, 1, 32, 5120), 'f32'),))
        heads = [entry for entry in fused_log if entry[0] == 'nlp_create_qkv_heads']
        self.assertEqual(len(heads), 5, 'one per learned layer, at 8 query and 2 KV heads')
        # the uploads are the bytes the window stages
        for upload, table in zip(uploads, pinned.host_tables(position, 32, 16)):
            self.assertEqual(upload[1][0], ('host', (1, 1, 32, 128), 'torch.bfloat16', table.float().sum().item()))

    def test_every_op_is_row_local_and_no_row_dimension_is_concatenated_gathered_or_reduced(self):
        with clean_environment(QWEN_FAST_TP_KV_SLIDE='1'), four_cards():
            ops, grid, collectives, weights, parameters, taps, slots, block = self.fixture()
            with patch('feature_collective.projection_links', return_value=4):
                fused = twin.FusedCommit(block, operations=ops, mesh=grid, pool=None, shared_weights=weights,
                                         collectives=collectives, inplace=True, audit=False)
                storage = fused.segments[0]
                storage.taps = taps
                ops.log.clear()
                fused.project(storage, [])
        allowed = {'slice', 'pad', 'concat', 'matmul', 'all_gather_async', 'add', 'typecast', 'rms_norm', 'nlp_create_qkv_heads',
                   'rotary_embedding_hf', 'copy'}
        names = {entry[0] for entry in ops.log}
        self.assertLessEqual(names, allowed | {'reshape', 'permute', 'transpose', 'unsqueeze', 'to_layout', 'clone', 'sharded_to_interleaved'},
                             names - allowed)
        for name, arguments, keywords in ops.log:
            if name == 'concat':
                self.assertIn(arguments[1], (-1, 3), 'the taps are joined along the width, never the rows')
        for entry in ops.log:
            if entry[0] == 'all_gather_async':
                self.assertEqual(dict(entry[2])['dim'], 0, 'the chips are stacked on dim 0, never the rows')
            if entry[0] == 'matmul':
                shapes = [item[1] for item in entry[1] if isinstance(item, tuple) and item and item[0] == 'tensor']
                self.assertEqual(shapes[0][2], 32, 'the left operand keeps its 32 rows')

    def test_rows_below_the_prefix_are_equal_whatever_the_rows_above_it_hold_and_a_row_mixing_control_leaks(self):
        """A numeric model of T_proj's op classes at four chips: the fp32 chip partials (a per-row dot in a fixed order), the fp32
        chip-order add, the bf16 cast, the per-row rms norm and the rotary at the row's own position. Rows 0..prefix-1 of the 16-row run
        equal the prefix-row run's for every prefix, with the partner rows zero, real or a hundredfold. The control (a norm over
        the rows, which no op here does) leaks."""
        generator = torch.Generator().manual_seed(7)
        width, chips = 64, CHIPS
        features = torch.randn((16, chips, width), generator=generator).bfloat16()
        weight = torch.randn((chips, width, 48), generator=generator).bfloat16()
        norm = torch.rand(48, generator=generator).bfloat16() + 0.5
        cos, sin = torch.randn((16, 48), generator=generator).bfloat16(), torch.randn((16, 48), generator=generator).bfloat16()

        def rows(features, count, mix=False):
            padded = torch.zeros((32, chips, width), dtype=torch.bfloat16)
            padded[:count] = features[:count]
            partial = torch.stack([torch.stack([(padded[r, c].double() * weight[c].double().T).sum(-1).float() for c in range(chips)])
                                   for r in range(32)])                                # (rows, chips, 48) fp32, per-row dot
            total = partial[:, 0]
            for chip in range(1, chips):
                total = total + partial[:, chip]                                        # chip order
            value = total.bfloat16()
            if mix:
                value = (value.float() / (value.float().pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-6)).bfloat16()
            else:
                value = (value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16() * norm
            out = value[:16].clone()
            out[:, :24] = out[:, :24] * cos[:, :24] - out[:, 24:] * sin[:, :24]
            return out

        for prefix in range(1, 17):
            today = rows(features, prefix)                                              # rows past the prefix: zeros
            for label, partner in (('real', 1.0), ('hundredfold', 100.0), ('zeros', 0.0)):
                tail = features.clone()
                tail[prefix:] = tail[prefix:] * partner
                fused = rows(tail, 16)
                self.assertTrue(torch.equal(bits(fused[:prefix]), bits(today[:prefix])), (prefix, label))
            if prefix < 16:
                leaked = rows(features, 16, mix=True)[:prefix]
                quiet = rows(features, prefix, mix=True)[:prefix]
                self.assertFalse(torch.equal(bits(leaked), bits(quiet)), 'a row-mixing op leaks: the control fires')


# ---------------------------------------------------------------------------------------------
# Shipping.
# ---------------------------------------------------------------------------------------------

class ShippingTests(unittest.TestCase):
    def test_the_twin_reaches_the_image_in_all_three_lists(self):
        dockerfile = (ROOT / 'docker' / 'qwen-fast-serving.Dockerfile').read_text(encoding='utf-8')
        self.assertIn('COPY scripts/ci/fused_commit_tp.py /experiment-scripts/ci/', dockerfile)
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-fast-serving-image.yml').read_text(encoding='utf-8')
        self.assertIn('fused_commit_tp.py', workflow)
        overlay = (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').split()
        self.assertIn('scripts/ci/fused_commit_tp.py', overlay)

    def test_the_suite_runs_in_the_cpu_workflow(self):
        self.assertRegex(CPU_WORKFLOW.read_text(encoding='utf-8'), r'\btest_fused_commit_tp4\b')

    def test_the_twin_names_no_pair_literal_and_imports_only_the_served_helpers(self):
        text = (HERE / 'fused_commit_tp.py').read_text(encoding='utf-8')
        for literal in ('range(2)', 'len(parts) != 2', 'KV_SHAPE =', '(1, 4, 2048, 128)', 'WORKERS = 16'):
            self.assertNotIn(literal, text)
        self.assertNotIn('import draft_kv_slide\n', text)
        self.assertNotIn('draft_kv_history import', text.replace('draft_kv_history_tp', ''))


if __name__ == '__main__':
    unittest.main()
