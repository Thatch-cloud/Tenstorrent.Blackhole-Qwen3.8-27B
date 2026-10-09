"""Octo-T8 (QWEN_FAST_OCTO) and the lone-user packed policy (QWEN_FAST_SOLO_PACKED): the host side, none needing a device.

Layers:
  - the flags and the admission (serving_octo): off by default and strict; refused at every shape, environment, minimum and run it does not admit, every
    reason at once; and refused BY NAME for every device piece nothing has built (DEVICE_PIECES) - so the flag cannot engage on an attach that builds no
    octo block. A second set of tests patches the pieces in and shows the rest of the admission holds together;
  - the policy state (serving_octo.OctoState): live and alternate, the turn that only advances on a counted round, the lines;
  - the routing (serving_packed_step.PackedStep given `octo=`), on test_serving_packed_step's fakes: which block a round's tickets are drafted for, the epoch
    bump of a switch, the route and the lines AFTER the round ran, the fall-backs;
  - the hook (serving_worker_hook): the budget narrowing that keeps a round on the M3 blocks, the group drafted at 8 rows;
  - the attach (serving_runtime.attach_combined_runtime on test_serving_runtime's fakes): refused before anything is built;
  - the flag OFF: byte for byte today's step (a scenario run on the tree before octo existed, its output pinned here) and nothing imports serving_octo.
The real block at 8 users x 8 rows on the fake device is test_octo_block."""

import json
import os
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch  # noqa: F401 - imported before any test patches sys.modules

import octo_markers
import packed_verifier
import serving_octo as octo
import serving_packed_step
from serving_packed_step import PackedStep
import serving_solo_lane
import test_serving_packed_step as step_tests
from test_serving_packed_step import CommittingRequest, FakeBlock, entry
import verifier_engine

M3_TWO = (True, 'users=8 FOUR_AS_TWO=0 PACKED_STEP=1 M3_BLOCKS=2')
NOT_M3 = (False, 'users=4 FOUR_AS_TWO=unset PACKED_STEP=1')
ALL_PIECES = frozenset(key for key, text in octo.DEVICE_PIECES)


def gate_environment(**changes):
    """A gate run of an octo profile: every REQUIRED_ENV value, the two gate switches, the flag; `changes` over it (None removes a key)."""
    environ = {name: wanted for name, wanted, why in octo.REQUIRED_ENV}
    environ.update({octo.GATE_ENV: '1', octo.GATE_PROFILE_ENV: '1', octo.OCTO_FLAG: 'alternate'})
    for key, value in changes.items():
        if value is None:
            environ.pop(key, None)
        else:
            environ[key] = value
    return environ


def solo_packed_environment(**changes):
    environ = {'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_TP': '4', 'QWEN_FAST_EXTENT_REPLAY': '1', 'QWEN_FAST_PADDED_BLOCK': '1',
               'QWEN_FAST_SINGLE_GATEUP': '1', octo.GATE_ENV: '1', octo.GATE_PROFILE_ENV: '1', octo.SOLO_PACKED_FLAG: '1'}
    for key, value in changes.items():
        if value is None:
            environ.pop(key, None)
        else:
            environ[key] = value
    return environ


class Lines:
    def __init__(self):
        self.lines = []

    def __call__(self, template, *values):
        self.lines.append(template.format(*values))


def built(*, gdn=8):
    """Every device piece built, as a session that finished them would leave the module (the pinned module's own limit read as `gdn`)."""
    return patch.multiple(octo, BUILT=set(ALL_PIECES), gdn_batch_users=Mock(return_value=gdn))


class FlagTests(unittest.TestCase):
    def test_the_mode_is_off_live_or_alternate_and_nothing_else(self):
        self.assertEqual(octo.octo_mode({}), 'off')
        for value in ('off', 'live', 'alternate'):
            self.assertEqual(octo.octo_mode({octo.OCTO_FLAG: value}), value)
        for value in ('', 'on', '1', '0', 'LIVE', ' live', 'alternate ', 'true'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'QWEN_FAST_OCTO must be one of off\\|live\\|alternate'):
                octo.octo_mode({octo.OCTO_FLAG: value})

    def test_the_minimum_live_is_a_decimal_integer_from_one_to_seven_and_defaults_to_six(self):
        self.assertEqual(octo.octo_min_live({}), 6)
        self.assertEqual(octo.MIN_LIVE_DEFAULT, octo.USERS - 2, 'eight segments less the two idle ones page 0 can hold')
        for value in ('1', '5', '6', '7'):
            self.assertEqual(octo.octo_min_live({octo.MIN_LIVE_FLAG: value}), int(value))
        for value in ('', '0', '8', '-1', '06', '5.0', 'six', ' 6'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'QWEN_FAST_OCTO_MIN_LIVE must be a decimal integer from 1 to 7'):
                octo.octo_min_live({octo.MIN_LIVE_FLAG: value})

    def test_solo_packed_is_zero_or_one_and_nothing_else(self):
        self.assertFalse(octo.solo_packed_requested({}))
        self.assertFalse(octo.solo_packed_requested({octo.SOLO_PACKED_FLAG: '0'}))
        self.assertTrue(octo.solo_packed_requested({octo.SOLO_PACKED_FLAG: '1'}))
        for value in ('', '2', 'true', 'yes', '01', ' 1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'QWEN_FAST_SOLO_PACKED must be 0 or 1'):
                octo.solo_packed_requested({octo.SOLO_PACKED_FLAG: value})

    def test_the_gate_rule_is_the_solo_lanes_and_the_extent_paths_own(self):
        import itertools

        import packed_any_admission

        self.assertEqual((octo.GATE_ENV, octo.GATE_PROFILE_ENV), (serving_solo_lane.GATE_ENV, serving_solo_lane.GATE_PROFILE_ENV))
        for gate, marker in itertools.product((None, '0', '1'), repeat=2):
            environ = {key: value for key, value in ((octo.GATE_ENV, gate), (octo.GATE_PROFILE_ENV, marker)) if value is not None}
            self.assertEqual(octo.gate_run(environ), serving_solo_lane.gate_run(environ), (gate, marker))
            self.assertEqual(octo.gate_run(dict(environ, QWEN_FAST_TP='4')), packed_any_admission.unqualified_allowed(dict(environ, QWEN_FAST_TP='4')))

    def test_the_names_match_the_ones_the_runtime_and_the_markers_module_use(self):
        import serving_runtime

        self.assertEqual(octo.M3_BLOCKS_FLAG, serving_runtime.M3_BLOCKS_FLAG)
        self.assertEqual(octo.ROUND_LINE, octo_markers.ROUND_LINE)
        self.assertEqual(octo.PROGRAMS_LINE, octo_markers.PROGRAMS_LINE)
        self.assertEqual(octo.ADMITTED_MARKER + ' mode=', octo_markers.ADMITTED + 'mode=')
        self.assertEqual((octo.UNQUALIFIED_MARKER, octo.REFUSED_MARKER), (octo_markers.UNQUALIFIED, octo_markers.REFUSED))
        self.assertEqual((octo.USERS, octo.ROWS), (8, 8))


class OctoAdmissionTests(unittest.TestCase):
    def test_off_by_default_and_nothing_is_read_while_off(self):
        for environ in ({}, {octo.OCTO_FLAG: 'off'}):
            self.assertIsNone(octo.octo_admission(NOT_M3, environ))
            self.assertIsNone(octo.octo_admission(NOT_M3, dict(environ, **{octo.MIN_LIVE_FLAG: 'junk'})), 'the minimum is not read while off')

    def test_a_malformed_flag_is_a_configuration_error_never_a_silent_off(self):
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_OCTO must be one of'):
            octo.octo_admission(M3_TWO, gate_environment(**{octo.OCTO_FLAG: 'yes'}))
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_OCTO_MIN_LIVE must be a decimal integer'):
            octo.octo_admission(M3_TWO, gate_environment(**{octo.MIN_LIVE_FLAG: '9'}))

    def test_today_the_flag_is_refused_by_name_for_every_device_piece_and_logged_line_by_line(self):
        log = Lines()
        with self.assertRaises(ValueError) as caught:
            octo.octo_admission(M3_TWO, gate_environment(), log=log)
        message = str(caught.exception)
        self.assertEqual(sorted(key for key, text in octo.device_gaps()), sorted(key for key, text in octo.DEVICE_PIECES))
        for key, text in octo.DEVICE_PIECES:
            self.assertIn('device piece %s is not built' % key, message)
        self.assertTrue(log.lines and all(line.startswith(octo.REFUSED_MARKER + ':') for line in log.lines))
        self.assertEqual(len(log.lines), len(octo.DEVICE_PIECES), 'the environment itself was fine: only the pieces are reasons')
        self.assertFalse(any(line.startswith(octo.ADMITTED_MARKER) for line in log.lines))

    def test_the_device_pieces_are_the_named_ones_and_each_says_what_remains(self):
        self.assertEqual([key for key, text in octo.DEVICE_PIECES],
                         ['attach-build', 'gdn-batch', 'attention-8row', 'publication-8row', 'fused-third-block'])
        for key, text in octo.DEVICE_PIECES:
            self.assertGreater(len(text), 60, key)

    def test_every_piece_built_and_a_gate_run_is_admitted_and_logs_every_missing_qualification(self):
        log = Lines()
        with built():
            record = octo.octo_admission(M3_TWO, gate_environment(), log=log)
        self.assertEqual((record['mode'], record['rows'], record['users'], record['min_live']), ('alternate', 8, 8, 6))
        unqualified = [line for line in log.lines if line.startswith(octo.UNQUALIFIED_MARKER)]
        self.assertEqual(len(unqualified), len(record['unqualified']))
        self.assertTrue(all('/%d)' % len(unqualified) in line for line in unqualified))
        admitted = [line for line in log.lines if line.startswith(octo.ADMITTED_MARKER)]
        self.assertEqual(admitted, ['[OCTO] admitted mode=alternate rows=8 users=8 min_live=6 (gate only)'])
        self.assertEqual(octo_markers.admissions(admitted[0]), [dict(mode='alternate', rows=8, users=8, min_live=6)])
        self.assertTrue(all(len(line) < 200 for line in log.lines))

    def test_the_gdn_piece_cannot_be_claimed_while_the_pinned_module_says_four(self):
        import gdn_user_batch

        self.assertEqual(gdn_user_batch.MAX_USERS, 4, 'the hash-pinned pair launch: test_tp2_pins')
        with patch.object(octo, 'BUILT', set(ALL_PIECES)):
            self.assertEqual([key for key, text in octo.device_gaps()], ['gdn-batch'])
        with built(gdn=8):
            self.assertEqual(octo.device_gaps(), [])
        with patch.object(octo, 'BUILT', ALL_PIECES - {'gdn-batch'}), patch.object(octo, 'gdn_batch_users', Mock(return_value=8)):
            self.assertEqual([key for key, text in octo.device_gaps()], ['gdn-batch'])

    def test_every_refusal_is_named_at_once_and_logged_line_by_line(self):
        cases = {
            'the shape': (NOT_M3, {}, 'beside the two 64-row M3 blocks of eight seats'),
            'one m3 block': (M3_TWO, {octo.M3_BLOCKS_FLAG: '1'}, 'QWEN_FAST_M3_BLOCKS=1, not 2'),
            'packed step': (M3_TWO, {'QWEN_FAST_PACKED_STEP': None}, r'QWEN_FAST_PACKED_STEP=\(unset\), not 1'),
            'the pair': (M3_TWO, {'QWEN_FAST_TP': '2'}, 'QWEN_FAST_TP=2, not 4'),
            'four as two': (M3_TWO, {'QWEN_FAST_FOUR_AS_TWO': '1'}, 'M3 blocks of 64 rows'),
            'extent replay': (M3_TWO, {'QWEN_FAST_EXTENT_REPLAY': '0'}, 'QWEN_FAST_EXTENT_REPLAY=0, not 1'),
            'single gate/up': (M3_TWO, {'QWEN_FAST_SINGLE_GATEUP': '0'}, 'one MLP arithmetic at M = 64'),
            'quad draft': (M3_TWO, {'QWEN_FAST_QUAD_DRAFT': '0'}, 'cut on the host to 8 rows'),
            'quad blocks': (M3_TWO, {'QWEN_FAST_QUAD_DRAFT_BLOCKS': '0'}, 'one quad draft per M3 block'),
            'kv reservation': (M3_TWO, {'QWEN_FAST_KV_RESERVATION': None}, 'the KV pool is lowered'),
            'solo lane': (M3_TWO, {'QWEN_FAST_SOLO_LANE': '1'}, 'QWEN_FAST_SOLO_LANE=1: octo cannot sit beside it'),
            'fast lane': (M3_TWO, {'QWEN_FAST_LANE': '1'}, 'QWEN_FAST_LANE=1: octo cannot sit beside it'),
            'not a gate run': (M3_TWO, {octo.GATE_ENV: None}, 'gate run of a gate-only'),
            'not a gate profile': (M3_TWO, {octo.GATE_PROFILE_ENV: None}, 'gate run of a gate-only'),
            'five live': (M3_TWO, {octo.MIN_LIVE_FLAG: '5'}, 'leaves 3 idle segments and a block holds 2'),
        }
        for name, (m3, changes, text) in cases.items():
            with self.subTest(refused=name):
                log = Lines()
                with built(), self.assertRaisesRegex(ValueError, text):
                    octo.octo_admission(m3, gate_environment(**changes), log=log)
                self.assertTrue(log.lines and all(line.startswith(octo.REFUSED_MARKER + ':') for line in log.lines))
        # a traffic profile (no gate switches at all) is refused whatever else holds
        with built(), self.assertRaisesRegex(ValueError, 'gate run of a gate-only'):
            octo.octo_admission(M3_TWO, gate_environment(**{octo.GATE_ENV: None, octo.GATE_PROFILE_ENV: None}), log=Lines())

    def test_all_the_reasons_are_listed_together(self):
        with built(), self.assertRaises(ValueError) as caught:
            octo.octo_admission(NOT_M3, gate_environment(QWEN_FAST_TP='2', QWEN_FAST_SOLO_LANE='1', **{octo.GATE_ENV: None}), log=Lines())
        message = str(caught.exception)
        for text in ('beside the two 64-row M3 blocks', 'QWEN_FAST_TP=2, not 4', 'QWEN_FAST_SOLO_LANE=1', 'gate run of a gate-only'):
            self.assertIn(text, message)

    def test_five_live_is_admitted_the_day_a_block_holds_three_idle_segments(self):
        self.assertEqual(packed_verifier.PackedVerifierEngine.MAX_IDLE_SEGMENTS, 2, 'page 0 has two 32-row tile rows')
        self.assertEqual(octo.idle_capacity(), 2)
        self.assertEqual(octo.MIN_LIVE_TARGET, 5)
        with built(), patch.object(octo, 'idle_capacity', Mock(return_value=3)):
            record = octo.octo_admission(M3_TWO, gate_environment(**{octo.MIN_LIVE_FLAG: '5'}), log=Lines())
        self.assertEqual(record['min_live'], 5)

    def test_the_required_environment_is_what_a_levern_profile_with_eight_seats_carries(self):
        for name, wanted, why in octo.REQUIRED_ENV:
            self.assertTrue(why)
        names = [name for name, wanted, why in octo.REQUIRED_ENV]
        self.assertEqual(len(names), len(set(names)))


class SoloPackedAdmissionTests(unittest.TestCase):
    def test_off_by_default(self):
        for environ in ({}, {octo.SOLO_PACKED_FLAG: '0'}):
            self.assertIsNone(octo.solo_packed_admission(NOT_M3, environ))

    def test_a_malformed_flag_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_SOLO_PACKED must be 0 or 1'):
            octo.solo_packed_admission(M3_TWO, solo_packed_environment(**{octo.SOLO_PACKED_FLAG: 'yes'}))

    def test_a_lone_user_is_refused_today_because_it_leaves_three_idle_segments_and_page_zero_holds_two(self):
        log = Lines()
        with self.assertRaises(ValueError) as caught:
            octo.solo_packed_admission(M3_TWO, solo_packed_environment(), log=log)
        self.assertIn('a lone user leaves 3 idle segments of the 4-user block and a block holds 2', str(caught.exception))
        self.assertIn('padded_min_users below 2 at construction', str(caught.exception))
        self.assertTrue(log.lines and all(line.startswith('[SOLO-PACKED] refused:') for line in log.lines))

    def test_the_block_itself_refuses_a_padded_minimum_of_one_at_the_m3_shape(self):
        # the same rule, from the constructor: users - MAX_IDLE_SEGMENTS <= padded_min_users (the real block's check, run on the fake device in test_octo_block)
        users, capacity = 4, packed_verifier.PackedVerifierEngine.MAX_IDLE_SEGMENTS
        self.assertGreater(max(1, users - capacity), 1)

    def test_admitted_when_a_block_holds_three_idle_segments(self):
        log = Lines()
        with patch.object(octo, 'idle_capacity', Mock(return_value=3)):
            record = octo.solo_packed_admission(M3_TWO, solo_packed_environment(), log=log)
        self.assertEqual(record, dict(min_users=1))
        self.assertEqual(log.lines, ['[SOLO-PACKED] admitted min_users=1 (gate only)'])

    def test_every_other_refusal_is_named_at_once(self):
        cases = {
            'the shape': (NOT_M3, {}, 'the lone-user padded round is the 64-row M3 block'),
            'padded block': (M3_TWO, {'QWEN_FAST_PADDED_BLOCK': None}, r'QWEN_FAST_PADDED_BLOCK=\(unset\), not 1'),
            'extent replay': (M3_TWO, {'QWEN_FAST_EXTENT_REPLAY': '0'}, 'QWEN_FAST_EXTENT_REPLAY=0, not 1'),
            'single gate/up': (M3_TWO, {'QWEN_FAST_SINGLE_GATEUP': None}, 'QWEN_FAST_SINGLE_GATEUP'),
            'the pair': (M3_TWO, {'QWEN_FAST_TP': '2'}, 'QWEN_FAST_TP=2, not 4'),
            'solo lane': (M3_TWO, {'QWEN_FAST_SOLO_LANE': '1'}, 'two answers to one question'),
            'not a gate run': (M3_TWO, {octo.GATE_ENV: None}, 'gate run of a gate-only'),
        }
        for name, (m3, changes, text) in cases.items():
            with self.subTest(refused=name), patch.object(octo, 'idle_capacity', Mock(return_value=3)):
                with self.assertRaisesRegex(ValueError, text):
                    octo.solo_packed_admission(m3, solo_packed_environment(**changes), log=Lines())


class StateTests(unittest.TestCase):
    def state(self, mode='alternate', min_live=6, programs=None):
        self.lines, self.now = [], [100.0]
        return octo.OctoState(mode, min_live, clock=lambda: self.now[0], programs=programs, log=self.lines.append)

    def round(self, state, ran, *, live=8, rows=8, committed=32, ms=50.0, gap=10.0):
        self.now[0] += gap / 1000
        state.begin()
        self.now[0] += ms / 1000
        return state.finish(ran, live=live, rows=rows, committed=committed)

    def test_the_state_is_built_for_live_or_alternate_with_a_minimum_inside_the_block(self):
        for mode, minimum in (('off', 6), ('x', 6), ('live', 0), ('live', 8), ('live', '6'), ('live', True)):
            with self.subTest(mode=mode, minimum=minimum), self.assertRaises(ValueError):
                octo.OctoState(mode, minimum)

    def test_live_plans_octo_for_every_eligible_round_and_m3_for_the_rest(self):
        state = self.state('live')
        self.assertEqual([state.plan(True), state.plan(False), state.plan(True)], ['octo', 'm3', 'octo'])
        for _ in range(3):
            state.plan(True)
            self.round(state, 'octo')
        self.assertEqual(state.plan(True), 'octo', 'live never alternates')

    def test_alternate_takes_its_turn_only_from_a_counted_round(self):
        state = self.state('alternate')
        self.assertEqual(state.plan(True), 'octo')
        self.assertEqual(state.plan(True), 'octo', 'planning twice (an early draft discarded and drafted again) advances nothing')
        self.round(state, 'octo')
        self.assertEqual(state.plan(True), 'm3')
        self.assertEqual(state.plan(True), 'm3')
        self.round(state, 'm3')
        self.assertEqual(state.plan(True), 'octo')
        # a round that was not eligible is M3's and consumes no turn
        self.assertEqual(state.plan(False), 'm3')
        self.round(state, 'm3', live=3, rows=16)
        self.assertEqual(state.plan(True), 'octo')
        # a round planned for octo that ran as something else (a fall-back) consumes no turn either
        state.plan(True)
        self.round(state, 'seq')
        self.assertEqual(state.plan(True), 'octo')
        self.assertEqual(state.counted, dict(octo=1, m3=1))
        self.assertEqual(state.rounds, dict(octo=1, m3=2, seq=1))

    def test_the_round_line_says_where_it_ran_and_parses_back(self):
        state = self.state('alternate', programs=iter([4001, 4001, 4001, 4001]).__next__)
        state.plan(True)
        fields = self.round(state, 'octo', committed=37, ms=48.25, gap=12.5)
        self.assertEqual(fields['counted'], 1)
        line = [line for line in self.lines if line.startswith(octo.MARKER + ' round=')][0]
        self.assertEqual(line, '[OCTO] round=1 shape=octo planned=octo live=8 rows=8 eligible=1 counted=1 octo_rounds=1 m3_rounds=0 committed=37 '
                               'step_ms=48.2 gap_ms=0.0')
        self.assertLess(len(line), 180)
        parsed = octo_markers.rounds(line)
        self.assertEqual(parsed, [dict(round=1, shape='octo', planned='octo', live=8, rows=8, eligible=1, counted=1, octo_rounds=1, m3_rounds=0,
                                      committed=37, step_ms=48.2, gap_ms=0.0)])
        state.plan(True)
        self.round(state, 'm3', rows=16, committed=33, ms=96.0, gap=12.5)
        second = octo_markers.rounds('\n'.join(self.lines))[1]
        self.assertEqual((second['shape'], second['planned'], second['counted'], second['octo_rounds'], second['m3_rounds']), ('m3', 'm3', 1, 1, 1))
        self.assertAlmostEqual(second['gap_ms'], 12.5, places=0)

    def test_a_programs_line_is_written_on_the_first_round_of_each_shape_after_a_switch(self):
        counts = iter([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
        state = self.state('alternate', programs=lambda: next(counts))
        for ran in ('octo', 'm3', 'octo', 'octo', 'seq', 'octo'):
            state.plan(True)
            self.round(state, ran)
        programs = octo_markers.programs('\n'.join(self.lines))
        self.assertEqual([(item['shape'], item['switch']) for item in programs], [('octo', 0), ('m3', 1), ('octo', 1)])
        self.assertTrue(all(item['before'] == item['after'] == 100 for item in programs))

    def test_a_program_compiled_on_a_switch_shows_as_before_below_after_and_an_unreadable_count_as_none(self):
        readings = iter([10, 12])
        state = self.state('live', programs=lambda: next(readings))
        state.plan(True)
        self.round(state, 'octo')
        item = octo_markers.programs('\n'.join(self.lines))[0]
        self.assertEqual((item['before'], item['after']), (10, 12))
        state = self.state('live')
        state.plan(True)
        self.round(state, 'octo')
        item = octo_markers.programs('\n'.join(self.lines))[0]
        self.assertEqual((item['before'], item['after']), (None, None))

    def test_switched_remembers_the_last_announced_shape(self):
        state = self.state()
        self.assertEqual([state.switched(shape) for shape in ('octo', 'octo', 'm3', 'm3', 'octo')], [False, False, True, False, True])

    def test_a_round_is_octo_m3_or_seq_and_nothing_else(self):
        state = self.state()
        with self.assertRaises(ValueError):
            state.finish('solo', live=1, rows=16, committed=1)

    def test_the_program_count_reads_the_blocks_models_mesh_and_never_raises(self):
        mesh = SimpleNamespace(num_program_cache_entries=lambda: 4321)
        self.assertEqual(octo.program_count(SimpleNamespace(model=SimpleNamespace(mesh_device=mesh))), 4321)
        for block in (SimpleNamespace(), SimpleNamespace(model=SimpleNamespace(mesh_device=object())),
                      SimpleNamespace(model=SimpleNamespace(mesh_device=SimpleNamespace(num_program_cache_entries=Mock(side_effect=RuntimeError))))):
            self.assertIsNone(octo.program_count(block))


class ExtentFake(FakeBlock):
    """A packed block as the policy sees it: the extent block's frontier range, padded rounds from `lowest` live users up to one less than full."""

    def __init__(self, users, rows, lowest=2):
        super().__init__(users=users, rows=rows)
        self.extent, self.lowest, self.deferred_commits = True, lowest, None

    def pads(self, count):
        return self.lowest <= count < self.shape.users

    def padded_refusal(self, users):
        return None

    def admits(self, position):
        return 128 <= position and position + self.rows_per_user <= 131328

    def accept_limit(self, position):
        return None


class Seats:
    """Eight requests over the two M3 blocks (slots 0..3 and 4..7) and the octo block (slots 0..7), sharing each request's engine by identity."""

    def __init__(self, test, *, mode='alternate', min_live=6, octo_lowest=None):
        verifier_engine.note_prefill()
        self.stepped = []
        self.a, self.b = ExtentFake(4, 16), ExtentFake(4, 16)
        self.octo = ExtentFake(8, 8, lowest=min_live if octo_lowest is None else octo_lowest)
        self.requests = []
        for index in range(8):
            request = CommittingRequest('R%d' % index, 4100 + 50 * index, self.stepped)
            request.engine.widths = (1, 2, 4)
            (self.a if index < 4 else self.b).bind(request.engine, index % 4)
            self.octo.bind(request.engine, index)
            self.requests.append(request)
        self.lines = []
        self.state = octo.OctoState(mode, min_live, log=self.lines.append)
        self.step = PackedStep([self.a, self.b], per_block_widths=True, octo=self.octo, octo_state=self.state)
        self.bumps = []
        patcher = patch.object(serving_packed_step, 'note_fixture_writer', self.bumps.append)
        patcher.start()
        test.addCleanup(patcher.stop)

    def live(self, *indexes):
        return [self.requests[index] for index in indexes]

    def draft(self, members, blocked=None):
        """What the hook does: ask for the groups, then draft every member of a group at its width (a narrow engine width where the group has none)."""
        groups = self.step.proposal_groups(members, blocked=blocked)
        for block, rows, group in groups:
            for request in group:
                if request.session.finished or request.session.pending is not None:
                    continue
                if block is not None and rows is not None:
                    request.propose(block.predictions_for(block.segment_of(request.engine)), 3, rows)
                else:
                    request.propose(list(range(1000, 1004)), 2, 4)
        return groups

    def run(self, members, cancelled=lambda: False):
        return self.step([entry(request) for request in members], cancelled=cancelled)

    def rounds_lines(self):
        return octo_markers.rounds('\n'.join(self.lines))

    def which(self, groups):
        names = {id(self.a): 'A', id(self.b): 'B', id(self.octo): 'octo'}
        return [(names.get(id(block), '-'), rows, len(group)) for block, rows, group in groups]


class RoutingTests(unittest.TestCase):
    def test_eight_live_in_live_mode_draft_eight_rows_for_the_octo_block_and_run_one_pass(self):
        seats = Seats(self, mode='live')
        members = seats.live(*range(8))
        groups = seats.draft(members)
        self.assertEqual(seats.which(groups), [('octo', 8, 8)])
        self.assertEqual({len(request.session.pending.tokens) for request in members}, {8})
        outputs = seats.run(members[::-1])
        self.assertEqual([output.request_id for output in outputs], ['R%d' % index for index in range(7, -1, -1)], "the scheduler's own order")
        self.assertEqual([call[0] for call in seats.octo.calls[:1]], ['verify'])
        self.assertEqual((seats.a.calls, seats.b.calls), ([], []), 'neither M3 block ran')
        self.assertEqual(seats.octo.rounds, 1)
        self.assertEqual(seats.step.route, 'octo')
        self.assertEqual(seats.stepped, [])
        lines = seats.rounds_lines()
        self.assertEqual([(line['shape'], line['planned'], line['live'], line['rows'], line['eligible'], line['counted'], line['committed'])
                          for line in lines], [('octo', 'octo', 8, 8, 1, 1, 32)])

    def test_alternate_runs_octo_then_m3_then_octo_each_round_exactly_once(self):
        seats = Seats(self)
        members = seats.live(*range(8))
        shapes = []
        for number in range(6):
            groups = seats.draft(members)
            shapes.append(seats.which(groups))
            outputs = seats.run(members)
            self.assertEqual(len(outputs), 8)
        self.assertEqual(shapes[0], [('octo', 8, 8)])
        self.assertEqual(shapes[1], [('A', 16, 4), ('B', 16, 4)])
        self.assertEqual([shapes[number] for number in (0, 2, 4)], [shapes[0]] * 3)
        self.assertEqual([shapes[number] for number in (1, 3, 5)], [shapes[1]] * 3)
        self.assertEqual([(line['shape'], line['counted']) for line in seats.rounds_lines()],
                         [('octo', 1), ('m3', 1), ('octo', 1), ('m3', 1), ('octo', 1), ('m3', 1)])
        self.assertEqual((seats.octo.rounds, seats.a.rounds, seats.b.rounds), (3, 3, 3))

    def test_every_switch_bumps_the_fixture_epoch_once_at_the_plan_and_the_step_does_not_bump_again(self):
        seats = Seats(self)
        members = seats.live(*range(8))
        for number in range(5):
            seats.draft(members)
            seats.run(members)
        # five rounds: octo, m3, octo, m3, octo - four switches
        self.assertEqual(seats.bumps, ['octo-switch'] * 4)

    def test_the_step_bumps_when_the_plan_did_not_announce_the_switch(self):
        seats = Seats(self, mode='live')
        members = seats.live(*range(8))
        seats.draft(members)
        seats.run(members)
        self.assertEqual(seats.bumps, [])
        # tickets drafted for the M3 blocks without a plan (a stale ticket): the step sees an m3 round after an octo one
        for request in members[:4]:
            request.propose(seats.a.predictions_for(seats.a.segment_of(request.engine)), 3, 16)
        for request in members[4:]:
            request.propose(seats.b.predictions_for(seats.b.segment_of(request.engine)), 3, 16)
        seats.run(members)
        self.assertEqual(seats.bumps, ['octo-switch'])

    def test_below_the_minimum_live_the_groups_are_exactly_the_two_block_ones(self):
        seats = Seats(self)
        for live in ((0, 1, 2, 3), (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5)[:5], (2,), ()):
            members = seats.live(*live)
            expected = serving_packed_step.proposal_groups((seats.a, seats.b), members)
            if len(live) >= 6:
                continue
            self.assertEqual(seats.step.proposal_groups(members), expected, live)
        self.assertEqual(seats.bumps, [])
        self.assertEqual(seats.state.planned, 'm3')

    def test_six_and_seven_live_are_padded_octo_rounds_with_the_idle_segments_committed_at_prefix_zero(self):
        for live in ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 7)):
            seats = Seats(self, mode='live')
            members = seats.live(*live)
            groups = seats.draft(members)
            self.assertEqual(seats.which(groups), [('octo', 8, len(live))], live)
            seats.run(members)
            self.assertEqual(seats.octo.rounds, 1, live)
            self.assertEqual(seats.step.route, 'octo')

    def test_five_live_is_not_an_octo_round_whatever_the_minimum_says_because_the_block_cannot_pad_it(self):
        seats = Seats(self, mode='live', min_live=5, octo_lowest=6)
        members = seats.live(0, 1, 2, 3, 4)
        self.assertEqual(seats.which(seats.step.proposal_groups(members)), [('A', 16, 4), ('B', None, 1)], 'the 4 + 1 split, today\'s')

    def test_a_finished_request_rides_in_the_last_group_and_the_live_set_decides(self):
        seats = Seats(self, mode='live')
        members = seats.live(*range(8))
        members[7].session.finished = True
        groups = seats.step.proposal_groups(members)
        self.assertEqual(seats.which(groups), [('octo', 8, 7), ('-', None, 1)])
        self.assertEqual(groups[0][2], members[:7])
        self.assertEqual(groups[1][2], [members[7]])
        # all finished: no live request, no octo
        for request in members:
            request.session.finished = True
        self.assertEqual(seats.which(seats.step.proposal_groups(members)), [('-', None, 8)])

    def test_a_request_the_octo_block_does_not_hold_keeps_the_round_on_the_m3_blocks(self):
        seats = Seats(self, mode='live')
        members = seats.live(*range(8))
        seats.octo.bound.pop(id(members[3].engine))
        self.assertEqual(seats.which(seats.step.proposal_groups(members)), [('A', 16, 4), ('B', 16, 4)])

    def test_a_blocked_member_the_hooks_budget_narrowing_would_cut_keeps_the_round_on_the_m3_blocks(self):
        seats = Seats(self, mode='live')
        members = seats.live(*range(8))
        self.assertEqual(seats.which(seats.step.proposal_groups(members, blocked={id(members[5])})), [('A', 16, 4), ('B', 16, 4)])
        self.assertEqual(seats.which(seats.step.proposal_groups(members, blocked=set())), [('octo', 8, 8)])

    def test_the_blocks_own_rules_decide_tokens_left_the_frontier_and_the_count(self):
        seats = Seats(self, mode='live')
        members = seats.live(*range(8))
        members[2].session.emitted = [1] * 249                  # 7 tokens left: fewer than the 8 rows (no budget cap)
        self.assertEqual(seats.which(seats.step.proposal_groups(members)), seats.which(serving_packed_step.proposal_groups((seats.a, seats.b), members)),
                         'seven tokens left for one member: the octo block cannot take its rows, the M3 plan stands')
        with patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '1'}):
            self.assertEqual(seats.which(seats.step.proposal_groups(members)), [('octo', 8, 8)], 'any token left keeps the width under the cap')
        members[2].session.emitted = []
        members[4].session.position = 100                        # below the extent block's floor
        self.assertEqual(seats.which(seats.step.proposal_groups(members)), seats.which(serving_packed_step.proposal_groups((seats.a, seats.b), members)))

    def test_a_round_drafted_for_octo_that_lost_a_member_before_the_step_is_narrowed_and_served_sequentially(self):
        seats = Seats(self, mode='live')
        members = seats.live(*range(8))
        seats.draft(members)
        survivors = members[:5]                                   # a partner aborted after the drafts: five entries, eight-row tickets
        for request in members[5:]:
            request.session.pending = None                         # (the aborted ones are gone with their tickets)
        outputs = seats.run(survivors)
        self.assertEqual(len(outputs), 5)
        self.assertEqual(seats.octo.calls, [], 'five live is not a padded round of this block: ineligible, narrowed to the engines\' widths')
        self.assertEqual(sorted(request_id for request_id, flag in seats.stepped), ['R%d' % index for index in range(5)])
        self.assertEqual(seats.step.route, 'sequential')
        line = seats.rounds_lines()[0]
        self.assertEqual((line['shape'], line['planned'], line['counted']), ('seq', 'octo', 0), 'ran as something else than planned: no turn consumed')

    def test_a_cancelled_round_runs_nothing_on_the_device_and_is_counted_as_sequential(self):
        seats = Seats(self, mode='live')
        members = seats.live(*range(8))
        for request in members:
            request.engine.widths = (1, 2, 4, 8, 16)
        seats.draft(members)
        outputs = seats.run(members, cancelled=lambda: True)
        self.assertEqual(len(outputs), 8)
        self.assertEqual(seats.octo.calls, [])
        self.assertEqual(seats.rounds_lines()[0]['shape'], 'seq')

    def test_the_octo_round_flushes_the_other_blocks_deferred_commits_first_and_the_m3_round_flushes_the_octo_blocks(self):
        order = []
        seats = Seats(self)
        for name, block in (('A', seats.a), ('B', seats.b), ('octo', seats.octo)):
            block.deferred_commits = [object()]
            block.flush_commits = (lambda site, name=name, block=block: order.append((name, site)) or setattr(block, 'deferred_commits', None) or 1)
        members = seats.live(*range(8))
        seats.draft(members)
        seats.run(members)                                          # octo round: A and B flush before it
        self.assertEqual(order, [('A', 'verify'), ('B', 'verify')])
        order.clear()
        seats.octo.deferred_commits = [object()]
        seats.draft(members)
        seats.run(members)                                          # m3 round: the octo block flushes before the verifies
        self.assertEqual(order, [('octo', 'verify')])

    def test_all_blocks_arming_and_flushing_reach_the_octo_block_too(self):
        seats = Seats(self)
        self.assertEqual(seats.step.all_blocks(), (seats.a, seats.b, seats.octo))
        for block in (seats.a, seats.b, seats.octo):
            block.arm_deferred_commits = Mock(return_value=block is seats.octo)
            block.flush_commits = Mock(return_value=2)
        self.assertTrue(seats.step.arm_deferred_commits())
        self.assertEqual(seats.step.flush_deferred_commits('end'), 6)

    def test_the_window_for_an_octo_round_is_the_octo_blocks_own_with_its_prestage(self):
        import verify_prestage

        seats = Seats(self, mode='live')
        for block in (seats.a, seats.b, seats.octo):
            block.prestaged, block.round_fences, block.fused = object(), True, None
        groups = seats.step.proposal_groups(seats.live(*range(8)))
        window = seats.step.while_waiting_groups(groups)
        self.assertIsInstance(window, verify_prestage.WhileWaiting)
        self.assertEqual((window.block, window.prestage), (seats.octo, True))

    def test_the_construction_rules(self):
        seats = Seats(self)
        a, b, state = seats.a, seats.b, seats.state
        with self.assertRaisesRegex(ValueError, 'go together'):
            PackedStep([a, b], per_block_widths=True, octo=seats.octo)
        with self.assertRaisesRegex(ValueError, 'go together'):
            PackedStep([a, b], per_block_widths=True, octo_state=state)
        with self.assertRaisesRegex(ValueError, 'separate eight-user eight-row block'):
            PackedStep([a, b], octo=seats.octo, octo_state=state, per_block_widths=False)
        with self.assertRaises(ValueError):          # the solo block never rides beside per-block widths (that rule is older than octo)
            PackedStep([a, b], octo=seats.octo, octo_state=state, per_block_widths=True, solo=ExtentFake(1, 16))
        with self.assertRaisesRegex(ValueError, 'separate eight-user eight-row block'):
            PackedStep([a], per_block_widths=False, octo=seats.octo, octo_state=state)
        with self.assertRaisesRegex(ValueError, 'separate eight-user eight-row block'):
            PackedStep([a, seats.octo], per_block_widths=True, octo=seats.octo, octo_state=state)
        with self.assertRaisesRegex(ValueError, 'separate eight-user eight-row block'):
            PackedStep([a, b], per_block_widths=True, octo=ExtentFake(8, 16), octo_state=state)
        with self.assertRaisesRegex(ValueError, 'separate eight-user eight-row block'):
            PackedStep([a, b], per_block_widths=True, octo=ExtentFake(4, 16), octo_state=state)

    def test_the_prestage_engages_over_three_blocks_when_there_is_an_octo_block(self):
        engaged = []
        with patch('verify_prestage.engage_two_block', side_effect=lambda blocks, environ=None: engaged.append(tuple(blocks))), \
                patch('verify_prestage.hostgap_log_enabled', return_value=True), patch('verify_prestage.block_label') as label:
            seats = Seats(self)
        self.assertEqual(engaged, [(seats.a, seats.b, seats.octo)])
        self.assertEqual([call.args[1] for call in label.call_args_list], [0, 1, 2])


class HookTests(unittest.TestCase):
    """The hook's side: the blocked set, the call that carries it, and the octo group drafted at 8 rows."""

    def setUp(self):
        from test_serving_worker_hook import PerBlockDraftTests

        self.helper = PerBlockDraftTests()
        self.helper.setUp()
        self.addCleanup(self.helper.tearDown)

    def hook(self, groups, octo_block=object(), asked=None):
        import test_serving_worker_hook as hooks

        def ask(requests, **kwargs):
            if asked is not None:
                asked.append(kwargs)
            return groups(requests)

        worker, bridge, events, scheduled = hooks.WorkerHookTests().fixture()
        packed_step = SimpleNamespace(proposal_rows=Mock(side_effect=AssertionError('the single width is never asked')),
                                      proposal_groups=ask, octo=SimpleNamespace(shape=SimpleNamespace(rows_per_user=8)) if octo_block else None)
        if not octo_block:
            del packed_step.octo
        from serving_worker_hook import FastWorkerHook

        hook = FastWorkerHook(worker, bridge, cancelled=lambda: False, packed_step=packed_step)
        self.addCleanup(hook.close)
        original = hook.bridges
        self.addCleanup(setattr, hook, 'bridges', original)
        return worker, hook

    def bridges(self, *names, **kwargs):
        return {name: self.helper.bridge(name, **kwargs.get(name, {})) for name in names}

    def test_an_octo_group_drafts_every_member_at_eight_rows_and_the_finished_ones_not_at_all(self):
        bridges = self.bridges(*'abcdefgh')
        requests = {name: bridge.request for name, bridge in bridges.items()}
        bridges['h'].request.session.finished = True
        bridges['h'].drafts = Mock(return_value=None)
        octo_block = object()
        worker, hook = self.hook(lambda asked: [(octo_block, 8, [requests[name] for name in 'abcdefg']), (None, None, [requests['h']])])
        hook.bridges = bridges
        result = worker.take_draft_token_ids()
        self.assertEqual(result.req_ids, list('abcdefg'))
        for name in 'abcdefg':
            bridges[name].drafts.assert_called_once_with(packed_rows=8)
        bridges['h'].drafts.assert_called_once_with()

    def test_the_blocked_set_names_the_members_the_budget_narrowing_would_cut_at_eight_rows(self):
        bridges = self.bridges(*'abcd', a=dict(max_tokens=100, emitted=100), b=dict(max_tokens=100, emitted=93),
                               c=dict(max_tokens=100, emitted=92), d=dict(max_tokens=None))
        worker, hook = self.hook(lambda asked: [])
        values = list(bridges.values())
        with patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '0'}):
            blocked = hook.octo_blocked(values)
        self.assertEqual(blocked, {id(bridges['a'].request), id(bridges['b'].request)}, 'fewer than eight tokens left: a (0) and b (7); c has eight')
        with patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '1'}):
            blocked = hook.octo_blocked(values)
        self.assertEqual(blocked, {id(bridges['a'].request)}, 'under the cap only a request vLLM no longer owes a token')
        bridges['a'].request.session.finished = True
        with patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '1'}):
            self.assertEqual(hook.octo_blocked(values), set(), 'a finished request is not a reason')

    def test_the_room_left_in_the_context_blocks_under_the_cap(self):
        bridges = self.bridges(*'ab')
        for name, bridge in bridges.items():
            bridge.runner = SimpleNamespace(model_config=SimpleNamespace(max_model_len=4100 + (6 if name == 'a' else 9)))
            bridge.request.session.position = 4100
        worker, hook = self.hook(lambda asked: [])
        # schedulable_rows is max_model_len - 1 - position: 5 for a, 8 for b
        with patch.dict(os.environ, {'QWEN_FAST_BUDGET_CAP': '1'}):
            self.assertEqual(hook.octo_blocked(list(bridges.values())), {id(bridges['a'].request)})

    def test_block_groups_passes_the_blocked_set_only_when_the_step_has_an_octo_block(self):
        asked = []
        bridges = self.bridges(*'ab', a=dict(max_tokens=100, emitted=100))
        worker, hook = self.hook(lambda requests: [(None, None, list(requests))], asked=asked)
        hook.block_groups(list(bridges.values()))
        self.assertEqual(asked, [dict(blocked={id(bridges['a'].request)})])
        asked.clear()
        worker2, hook2 = self.hook(lambda requests: [(None, None, list(requests))], octo_block=None, asked=asked)
        hook2.block_groups(list(bridges.values()))
        self.assertEqual(asked, [{}], 'a step without an octo block is asked the one-argument question it always was')


class AttachTests(unittest.TestCase):
    """serving_runtime.attach_combined_runtime on test_serving_runtime's fakes (its `exercise` asserts, in its finally, the events the attach left: a refusal
    before the pool leaves only the attach-failed line, so the assertion IS the proof that nothing was built)."""

    def harness(self):
        import test_serving_runtime

        return test_serving_runtime.RuntimeAttachmentTests('exercise')

    EIGHT = dict(packed=True, users=8, four_as_two=False, m3_blocks=2)

    def refused(self, text, extra_env):
        harness = self.harness()
        with self.assertRaisesRegex(ValueError, text):
            harness.exercise(refused=True, extra_env=extra_env, **self.EIGHT)
        return harness

    def test_the_flag_on_is_refused_before_anything_is_built_naming_the_pieces_nothing_has_built(self):
        for flags in ({octo.OCTO_FLAG: 'live'}, {octo.OCTO_FLAG: 'alternate', octo.GATE_ENV: '1', octo.GATE_PROFILE_ENV: '1'}):
            with self.subTest(flags=flags):
                self.refused('QWEN_FAST_OCTO=.* is refused: .*device piece attach-build is not built', flags)

    def test_the_refusal_names_every_reason_at_once(self):
        with self.assertRaises(ValueError) as caught:
            self.harness().exercise(refused=True, extra_env={octo.OCTO_FLAG: 'live'}, **self.EIGHT)
        message = str(caught.exception)
        for text in ('gate run of a gate-only', 'QWEN_FAST_EXTENT_REPLAY=(unset), not 1', 'QWEN_FAST_KV_RESERVATION', 'device piece gdn-batch is not built',
                     'device piece attention-8row is not built', 'device piece fused-third-block is not built'):
            self.assertIn(text, message)

    def test_solo_packed_on_is_refused_before_anything_is_built(self):
        self.refused('QWEN_FAST_SOLO_PACKED=1 is refused',
                     {octo.SOLO_PACKED_FLAG: '1', octo.GATE_ENV: '1', octo.GATE_PROFILE_ENV: '1', 'QWEN_FAST_PADDED_BLOCK': '1'})

    def test_a_malformed_value_is_a_configuration_error_naming_the_flag(self):
        for flags, text in (({octo.OCTO_FLAG: ''}, 'QWEN_FAST_OCTO must be one of'), ({octo.OCTO_FLAG: 'maybe'}, 'QWEN_FAST_OCTO must be one of'),
                            ({octo.SOLO_PACKED_FLAG: ''}, 'QWEN_FAST_SOLO_PACKED must be 0 or 1'), ({octo.SOLO_PACKED_FLAG: '2'}, 'must be 0 or 1')):
            with self.subTest(flags=flags):
                self.refused(text, flags)

    def test_an_admitted_flag_with_no_block_built_is_a_tripwire_not_a_silent_no_op(self):
        record = dict(mode='live', rows=8, users=8, min_live=6, unqualified=[])
        with patch.object(octo, 'octo_admission', Mock(return_value=record)):
            self.refused('was admitted but this attach builds no octo block', {octo.OCTO_FLAG: 'live'})

    def test_the_flags_off_builds_exactly_the_two_blocks_it_always_did_and_never_imports_serving_octo(self):
        for extra_env in ({}, {octo.OCTO_FLAG: 'off', octo.SOLO_PACKED_FLAG: '0'}):
            harness = self.harness()
            with self.subTest(extra_env=extra_env), patch.dict('sys.modules', {'serving_octo': None}):      # an import of it now raises ImportError
                harness.exercise(admission={}, extra_env=extra_env, **self.EIGHT)
            self.assertEqual([call.kwargs['pool_slots'] for call in harness.engine_calls], [(0, 1, 2, 3), (4, 5, 6, 7)])
            self.assertEqual(harness.pool_options['packed_replicas'], {(4, 16): 2}, 'two M3 block sets: no third')


GOLDEN = json.loads('''{"bumps": [], "log": [["A:16:4", "B:16:4"], [[["verify", ["R0", "R1", "R2", "R3"]], ["commit", 0, 4], ["commit", 1, 4], ["commit", 2, 4], ["commit", 3, 4]], [["verify", ["R4", "R5", "R6", "R7"]], ["commit", 0, 4], ["commit", 1, 4], ["commit", 2, 4], ["commit", 3, 4]], [4, 4, 4, 4, 4, 4, 4, 4], null], ["A:16:4", "B:None:1"], [[["verify", ["R0", "R1", "R2", "R3"]], ["commit", 0, 4], ["commit", 1, 4], ["commit", 2, 4], ["commit", 3, 4]], [], [4, 4, 4, 4, 4], null], ["A:16:3", "B:16:2"], [[["verify", ["R0", "R1", "R2"]], ["commit", 0, 4], ["commit", 1, 4], ["commit", 2, 4]], [["verify", ["R5", "R6"]], ["commit", 1, 4], ["commit", 2, 4]], [4, 4, 4, 4, 4], null], ["A:16:4", "B:16:4"], [[["verify", ["R0", "R1", "R2", "R3"]], ["commit", 0, 4], ["commit", 1, 4], ["commit", 2, 4], ["commit", 3, 4]], [["verify", ["R4", "R5", "R6", "R7"]], ["commit", 0, 4], ["commit", 1, 4], ["commit", 2, 4], ["commit", 3, 4]], [4, 4, 4, 4, 4, 4, 4, 4], null], ["A:None:1"], [[], [], [4], null]], "rounds": [4, 3], "stepped": [["R4", false], ["R3", false]]}''')


class OffIdentityTests(unittest.TestCase):
    """Default off, byte for byte today's: the scenario below was run on the tree BEFORE octo existed (the commit this branch forks from) and its output is
    pinned in GOLDEN, which the tree with octo must reproduce exactly - the same block calls in the same order, the same widths, the same epoch bumps, the same
    fallbacks. It uses only the API that tree had."""

    def scenario(self):
        verifier_engine.note_prefill()
        a, b = step_tests.PaddedFakeBlock(), step_tests.PaddedFakeBlock()
        stepped = []
        step = PackedStep([a, b], per_block_widths=True)
        log, bumps = [], []
        with patch.object(serving_packed_step, 'note_fixture_writer', bumps.append):
            for live in ([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4], [0, 1, 2, 5, 6], [0, 1, 2, 3, 4, 5, 6, 7], [3]):
                a.bound.clear()
                b.bound.clear()
                requests = {}
                for index in live:
                    request = CommittingRequest('R%d' % index, 4100 + 50 * index, stepped)
                    request.engine.widths = (1, 2, 4)
                    (a if index < 4 else b).bind(request.engine, index % 4)
                    requests[index] = request
                members = [requests[index] for index in live]
                groups = step.proposal_groups(members)
                log.append(['%s:%s:%d' % ('A' if group[0] is a else 'B' if group[0] is b else '-', group[1], len(group[2])) for group in groups])
                for block, rows, group in groups:
                    for request in group:
                        if block is not None and rows is not None:
                            request.propose(block.predictions_for(block.segment_of(request.engine)), 3, rows)
                        elif not request.session.finished:
                            request.propose(list(range(1000, 1004)), 2, 4)
                before = (len(a.calls), len(b.calls))
                outputs = step([entry(request) for request in members], cancelled=lambda: False)
                log.append([a.calls[before[0]:], b.calls[before[1]:], [len(output.token_ids) for output in outputs], step.route])
        return json.loads(json.dumps(dict(log=log, bumps=bumps, stepped=stepped, rounds=(a.rounds, b.rounds)), default=str))

    def test_the_two_block_step_without_octo_reproduces_the_pre_octo_scenario_exactly(self):
        self.assertEqual(self.scenario(), GOLDEN)

    def test_without_octo_the_step_has_none_of_it(self):
        a, b = step_tests.PaddedFakeBlock(), step_tests.PaddedFakeBlock()
        step = PackedStep([a, b], per_block_widths=True)
        self.assertIsNone(step.octo)
        self.assertIsNone(step.octo_state)
        self.assertIs(step.all_blocks(), step.blocks, 'the very tuple it always returned')
        with patch.object(PackedStep, 'route_octo', Mock(side_effect=AssertionError('route_octo is never reached without the block'))), \
                patch.object(PackedStep, 'octo_groups', Mock(side_effect=AssertionError('octo_groups is never reached without the block'))):
            self.GOLDEN_RUN()

    def GOLDEN_RUN(self):
        self.assertEqual(self.scenario(), GOLDEN)

    def test_a_single_block_step_and_a_solo_step_are_untouched(self):
        block = step_tests.PaddedFakeBlock()
        step = PackedStep(block)
        self.assertIs(step.block, block)
        self.assertIsNone(step.octo)
        solo = step_tests.PaddedFakeBlock(1)
        stepped = PackedStep(block, solo=solo)
        self.assertEqual(stepped.all_blocks(), (block, solo))

    def test_the_hook_asks_a_step_without_an_octo_block_the_question_it_always_asked(self):
        from serving_worker_hook import FastWorkerHook

        calls = []
        hook = FastWorkerHook.__new__(FastWorkerHook)
        hook.packed_step = SimpleNamespace(proposal_groups=lambda requests: calls.append(list(requests)) or [])
        self.assertEqual(hook.block_groups([SimpleNamespace(request='r1'), SimpleNamespace(request='r2')]), [])
        self.assertEqual(calls, [['r1', 'r2']])

    def test_the_modules_the_step_imports_do_not_include_serving_octo(self):
        # serving_octo is imported lazily by the attach only, under its flags; the step takes its state as an argument
        for name in ('serving_packed_step', 'serving_worker_hook', 'packed_shapes', 'packed_verifier'):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name + '.py'), encoding='utf-8') as handle:
                source = handle.read()
            self.assertNotIn('import serving_octo', source, name)
            self.assertNotIn('from serving_octo', source, name)


if __name__ == '__main__':
    unittest.main()
