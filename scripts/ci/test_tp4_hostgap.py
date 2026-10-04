"""tp4/hostgap: the eight-seat host gap (QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE and its companions; every flag default off).

With two packed 64-row blocks the verify pre-stage is left out of the drafts' window, so 744 of 788 measured verifies took the full
stage at verify time (about 22.7 ms of device idle an eight-live round). This module pins what the levers do and, as important,
what they cannot do:

  - the flags are 0 or 1, the audit and the window validate need the pre-stage, and with every flag off the epoch, the windows, the
    entry and the page binding are today's;
  - engage_two_block engages 1a-lite (the first block to verify), then the per-block epochs, refuses by name (one block, a block
    without the pre-stage, a shared fixture, a shared replay reader, shared extent storage) and says so;
  - per-block epochs: a block's own writes move only its epoch, an external writer moves the global one and kills both snapshots, a
    shared staging address disengages the mode, and the window pre-stages the first block (lite) or every packed block (blocks);
  - on a real block over the fake device model: the staged inputs after the window and the diff are the full stage's (every
    destination, every chip, under the full audit), a stray write is found and the round runs on the full stage, and the 1c skip
    removes exactly one binding check and runs it as a shadow under the audit;
  - 1d: the storage check runs once per distinct validator, and the incremental page validation accepts and refuses exactly what
    the full validation does over random allocations;
  - stage 0: the new lines (and only the new lines) exist under the log flag;
  - the smoke rules, the four profiles (the control plus flags and nothing else), the job pack, and the shipping lists."""

import json
import os
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dflash_packed_proposal_coordinator import note_fenced, run_while_waiting
import c2_serving_job as job
import c2_smoke_check
import packed_verifier
import serving_packed_bridge
import serving_packed_step
import test_gdn_records as tgr
import test_padded_probe as tpp
import test_serving_packed_bridge as tbridge
import test_serving_packed_step as tstep
import test_serving_page_binding as tbinding
import test_verify_prestage as tvp
import verifier_engine
import verify_prestage as vp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
FOLDER = HERE / 'references' / 'tp4-hostgap-jobs'
IMAGE = 'tp4-hostgap-1'
CONTROL = 'c2-packed-tp4-8x262k-best-time-gate'
LITE, ARM = 'c2-packed-tp4-8x262k-hostgap-1', 'c2-packed-tp4-8x262k-hostgap-2'
LITE_AUDIT, ARM_AUDIT = LITE + '-audit', ARM + '-audit'
FLAGS = vp.HOSTGAP_FLAGS
NAMES = tvp.NAMES
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')


def lines_of(log, marker):
    return [line for line in log if line.startswith(marker)]


def stub_block(label=None, *, reader=True, storage=None, prestaged=True, users=4):
    """A block as far as the attach checks and the epochs look at it."""
    fixture = SimpleNamespace(replay_reader=SimpleNamespace(readers=[]) if reader else None)
    block = SimpleNamespace(fixture=fixture, extent_storage=storage, users=users, rounds=0, hostgap_label=label,
                            prestaged=None, round_fences=False, fused=None)
    if prestaged:
        block.prestaged = vp.BlockPrestage(block, audit=False)
    return block


class Clean(unittest.TestCase):
    """No QWEN_FAST_ flag from the host, and the process-wide pre-stage state reset around every test."""

    def setUp(self):
        environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
        patcher = patch.dict(os.environ, environ, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = []
        for target in (vp,):
            logger = patch.object(target, 'log_line', side_effect=self.log.append)
            logger.start()
            self.addCleanup(logger.stop)
        self.reset()
        self.addCleanup(self.reset)

    @staticmethod
    def reset():
        vp._MODE.update(first=False, blocks=False)
        vp._LOCAL.clear()
        vp._ADDRESSES.clear()
        vp._SCRATCH.clear()
        vp.uninstall_gc_log()

    def engage(self, **flags):
        environ = dict(os.environ)
        environ.update(flags)
        return environ


class FlagTests(Clean):
    def test_every_flag_is_off_by_default_zero_or_one_and_the_dependents_need_the_prestage(self):
        for name in FLAGS:
            with self.subTest(name=name):
                self.assertEqual(os.environ.get(name, '0'), '0')
                with self.assertRaises(ValueError):
                    vp._flag(name, {name: 'yes'})
        self.assertEqual([vp.two_block_enabled(), vp.block_epochs_enabled(), vp.full_audit_enabled(), vp.window_validate_enabled(),
                          vp.entry_diet_enabled(), vp.hostgap_log_enabled()], [False] * 6)
        on = {name: '1' for name in FLAGS}
        self.assertTrue(all(check(on) for check in (vp.two_block_enabled, vp.block_epochs_enabled, vp.entry_diet_enabled,
                                                    vp.hostgap_log_enabled)))
        # the audit and the 1c skip are inert without QWEN_FAST_PRESTAGE=1 (alone they audit and skip nothing)
        self.assertFalse(vp.full_audit_enabled(on))
        self.assertFalse(vp.window_validate_enabled(on))
        on['QWEN_FAST_PRESTAGE'] = '1'
        self.assertTrue(vp.full_audit_enabled(on) and vp.window_validate_enabled(on))

    def test_the_flag_names_are_the_ones_the_profiles_and_the_smoke_check_read(self):
        self.assertEqual(vp.TWO_BLOCK_FLAG, c2_smoke_check.HOSTGAP_TWO_BLOCK_FLAG)
        self.assertEqual(vp.BLOCK_EPOCHS_FLAG, c2_smoke_check.HOSTGAP_EPOCHS_FLAG)
        self.assertEqual(vp.FULL_AUDIT_FLAG, c2_smoke_check.HOSTGAP_AUDIT_FLAG)
        self.assertEqual(vp.WINDOW_VALIDATE_FLAG, c2_smoke_check.HOSTGAP_WINDOW_VALIDATE_FLAG)
        self.assertEqual(vp.TWO_BLOCK_ENGAGED_MARKER, c2_smoke_check.HOSTGAP_ENGAGED)
        self.assertEqual(vp.TWO_BLOCK_REFUSED_MARKER, c2_smoke_check.HOSTGAP_REFUSED)
        self.assertEqual(vp.BLOCK_EPOCHS_ENGAGED_MARKER, c2_smoke_check.HOSTGAP_EPOCHS_ENGAGED)
        self.assertEqual(vp.BLOCK_EPOCHS_REFUSED_MARKER, c2_smoke_check.HOSTGAP_EPOCHS_REFUSED)


class EngageTests(Clean):
    def blocks(self, **second):
        return [stub_block(), stub_block(**second)]

    def test_no_flag_engages_nothing_and_logs_nothing(self):
        self.assertIsNone(vp.engage_two_block(self.blocks()))
        self.assertEqual((vp.two_block_mode(), self.log), (None, []))

    def test_lite_engages_the_first_block_rule_and_logs_one_line_for_each_block(self):
        mode = vp.engage_two_block(self.blocks(), self.engage(QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE='1'))
        self.assertEqual((mode, vp.two_block_mode()), ('first', 'first'))
        self.assertEqual(len(lines_of(self.log, vp.TWO_BLOCK_ENGAGED_MARKER)), 2)
        self.assertRegex(self.log[0], r'block=A users=4 mode=first window_validate=0 audit=0$')
        self.assertEqual(lines_of(self.log, vp.BLOCK_EPOCHS_ENGAGED_MARKER), [])

    def test_the_block_epochs_engage_on_top_of_it(self):
        environ = self.engage(QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE='1', QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS='1',
                              QWEN_FAST_TP4_WINDOW_VALIDATE='1', QWEN_FAST_PRESTAGE='1')
        self.assertEqual(vp.engage_two_block(self.blocks(), environ), 'blocks')
        self.assertEqual(lines_of(self.log, vp.BLOCK_EPOCHS_ENGAGED_MARKER), [vp.BLOCK_EPOCHS_ENGAGED_MARKER + ' blocks=2'])
        self.assertRegex(self.log[1], r'block=A users=4 mode=blocks window_validate=1 audit=0$')

    def test_refusals_name_their_reason_and_leave_todays_rule(self):
        asked = self.engage(QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE='1')
        cases = {'one_packed_block': [stub_block()], 'built_without_the_pre-stage': [stub_block(), stub_block(prestaged=False)]}
        for reason, blocks in cases.items():
            with self.subTest(reason=reason):
                self.log.clear()
                self.assertIsNone(vp.engage_two_block(blocks, asked))
                self.assertIsNone(vp.two_block_mode())
                self.assertEqual(len(self.log), 1)
                self.assertTrue(self.log[0].startswith(vp.TWO_BLOCK_REFUSED_MARKER + ' reason='), self.log)
        shared = stub_block()
        twin = stub_block()
        twin.fixture = shared.fixture
        self.log.clear()
        self.assertIsNone(vp.engage_two_block([shared, twin], asked))
        self.assertIn('two_blocks_share_one_fixture', self.log[0])

    def test_the_block_epochs_refuse_a_shared_reader_or_shared_extent_storage_but_keep_lite(self):
        asked = self.engage(QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE='1', QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS='1')
        first, second = stub_block(), stub_block()
        second.fixture.replay_reader = first.fixture.replay_reader
        self.assertEqual(vp.engage_two_block([first, second], asked), 'first')
        self.assertIn('share_one_replay_reader', lines_of(self.log, vp.BLOCK_EPOCHS_REFUSED_MARKER)[0])
        self.log.clear()
        tensor = object()
        first, second = stub_block(storage=[[(tensor, 1)]]), stub_block(storage=[[(tensor, 2)]])
        self.assertEqual(vp.engage_two_block([first, second], asked), 'first')
        self.assertIn('share_extent_storage', lines_of(self.log, vp.BLOCK_EPOCHS_REFUSED_MARKER)[0])
        self.log.clear()
        first, second = stub_block(storage=[[(object(), 1)]]), stub_block(reader=False)
        self.assertEqual(vp.engage_two_block([first, second], asked), 'first')
        self.assertIn('no_replay_reader', lines_of(self.log, vp.BLOCK_EPOCHS_REFUSED_MARKER)[0])

    def test_the_block_epochs_without_the_two_block_flag_are_refused(self):
        self.assertIsNone(vp.engage_two_block(self.blocks(), self.engage(QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS='1')))
        self.assertEqual(len(lines_of(self.log, vp.BLOCK_EPOCHS_REFUSED_MARKER)), 1)
        self.assertEqual(vp.two_block_mode(), None)

    def test_a_misset_host_gap_flag_fails_at_the_attach_not_at_the_first_step(self):
        for name in FLAGS:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                vp.engage_two_block(self.blocks(), self.engage(**{name: 'yes'}))

    def test_engaging_again_starts_from_a_clean_state(self):
        vp._MODE.update(first=True, blocks=True)
        vp._LOCAL[1] = (4, 'x')
        vp._ADDRESSES[1] = frozenset()
        self.assertIsNone(vp.engage_two_block(self.blocks()))
        self.assertEqual((vp._MODE, vp._LOCAL, vp._ADDRESSES), (dict(first=False, blocks=False), {}, {}))

    def test_a_packed_step_over_two_blocks_engages_and_over_one_does_not(self):
        with patch.dict(os.environ, {'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE': '1'}):
            serving_packed_step.PackedStep([stub_block(), stub_block()], per_block_widths=True)
            self.assertEqual(vp.two_block_mode(), 'first')
            self.reset()
            serving_packed_step.PackedStep(stub_block())
            self.assertIsNone(vp.two_block_mode())


class PerBlockEpochTests(Clean):
    """usable() is (global, local) both standing; a block's own writes move its own epoch only."""

    def snapshot(self, block):
        block.prestaged.snapshot = vp.Snapshot(vp.epoch(), [], [], 0, 0.0, 1, local=vp.local_epoch(block.fixture))
        return block.prestaged

    def test_a_blocks_own_write_leaves_the_other_blocks_snapshot_usable(self):
        first, second = stub_block('A'), stub_block('B')
        vp._MODE['blocks'] = True
        a, b = self.snapshot(first), self.snapshot(second)
        self.assertEqual((a.usable()[1], b.usable()[1]), (None, None))
        before = vp.epoch()
        vp.bump_fixture(first.fixture, 'verify')
        self.assertEqual(vp.epoch(), before, 'the global epoch did not move')
        self.assertEqual(a.usable(), (None, 'epoch:verify'))
        self.assertIsNone(b.usable()[1])
        for reason in ('stage_packed', 'prestage'):
            vp.bump_fixture(second.fixture, reason)
            self.assertEqual(b.usable(), (None, 'epoch:%s' % reason))
        self.assertEqual(vp.epoch(), before)

    def test_an_external_writer_kills_both_snapshots(self):
        first, second = stub_block('A'), stub_block('B')
        vp._MODE['blocks'] = True
        a, b = self.snapshot(first), self.snapshot(second)
        vp.bump('admission')
        self.assertEqual((a.usable(), b.usable()), ((None, 'epoch:admission'), (None, 'epoch:admission')))

    def test_a_verify_failure_stays_global_and_kills_both(self):
        first, second = stub_block('A'), stub_block('B')
        vp._MODE['blocks'] = True
        a, b = self.snapshot(first), self.snapshot(second)
        vp.bump('verify-failed')
        self.assertEqual((a.usable()[0], b.usable()[0]), (None, None))

    def test_without_the_block_epochs_every_write_is_the_global_bump(self):
        first, second = stub_block('A'), stub_block('B')
        a, b = self.snapshot(first), self.snapshot(second)
        before = vp.epoch()
        vp.bump_fixture(first.fixture, 'verify')
        self.assertEqual((vp.epoch(), vp.last_bump(), vp._LOCAL), (before + 1, 'verify', {}))
        self.assertEqual((a.usable(), b.usable()), ((None, 'epoch:verify'), (None, 'epoch:verify')))

    def test_the_snapshot_keeps_the_pre_h1a_constructor_and_a_zero_local_epoch(self):
        snapshot = vp.Snapshot(3, [], [], 0, 0.0, 1)
        self.assertEqual((snapshot.epoch, snapshot.local), (3, 0))


class OverlapTests(Clean):
    def addresses(self, table):
        return lambda destination: table[destination]

    def test_disjoint_blocks_register_and_a_shared_address_disengages_the_mode(self):
        first, second = stub_block('A'), stub_block('B')
        vp._MODE.update(first=True, blocks=True)
        table = {'a1': (1, 2, 3, 4), 'a2': (5, 6, 7, 8), 'b1': (10, 11, 12, 13), 'shared': (1, 2, 3, 4)}
        self.assertIsNone(vp.register_destinations(first, ['a1', 'a2'], self.addresses(table)))
        self.assertIsNone(vp.register_destinations(second, ['b1'], self.addresses(table)))
        self.assertTrue(vp._MODE['blocks'])
        # once per block: a second call for a registered block reads nothing
        self.assertIsNone(vp.register_destinations(first, ['shared'], self.addresses({})))
        self.reset()
        vp._MODE.update(first=True, blocks=True)
        vp.register_destinations(first, ['a1'], self.addresses(table))
        self.assertEqual(vp.register_destinations(second, ['shared'], self.addresses(table)), 'overlap')
        self.assertEqual((vp._MODE['blocks'], vp._MODE['first']), (False, True))
        self.assertEqual(len(lines_of(self.log, vp.BLOCK_EPOCHS_REFUSED_MARKER)), 1)
        before = vp.epoch()
        vp.bump_fixture(first.fixture, 'verify')
        self.assertEqual(vp.epoch(), before + 1, 'global bumps again, the safe direction')

    def test_the_same_address_on_two_chips_is_not_an_overlap(self):
        first, second = stub_block('A'), stub_block('B')
        vp._MODE.update(first=True, blocks=True)
        self.assertIsNone(vp.register_destinations(first, ['x'], self.addresses({'x': (1, 2)})))
        self.assertIsNone(vp.register_destinations(second, ['y'], self.addresses({'y': (2, 1)})))
        self.assertTrue(vp._MODE['blocks'])


class WindowRuleTests(Clean):
    """serving_packed_step.while_waiting_groups: which blocks' windows pre-stage."""

    def setUp(self):
        super().setUp()
        verifier_engine.note_prefill()
        self.a, self.b = tstep.PaddedFakeBlock(), tstep.PaddedFakeBlock()
        for index, block in enumerate((self.a, self.b)):
            block.prestaged, block.round_fences, block.fused = SimpleNamespace(), False, None
            block.fixture = SimpleNamespace(replay_reader=SimpleNamespace(readers=[]))
            block.rounds = 0
        self.step = serving_packed_step.PackedStep([self.a, self.b], per_block_widths=True)

    def windows(self, groups):
        window = self.step.while_waiting_groups(groups)
        if window is None:
            return None
        parts = getattr(window, 'windows', [window])
        return [(part.block is self.a and 'A' or 'B', part.prestage) for part in parts]

    def both(self, rows=(16, 16)):
        return [(self.a, rows[0], ['a']), (self.b, rows[1], ['b'])]

    def test_today_two_blocks_run_no_pre_stage(self):
        self.assertEqual(self.windows(self.both()), [('A', False), ('B', False)])

    def test_lite_pre_stages_the_block_that_verifies_first_whatever_order_the_groups_come_in(self):
        vp._MODE['first'] = True
        self.assertEqual(self.windows(self.both()), [('A', True), ('B', False)])
        self.assertEqual(self.windows(list(reversed(self.both()))), [('B', False), ('A', True)])

    def test_lite_picks_the_first_block_that_runs_packed_not_the_first_that_exists(self):
        vp._MODE['first'] = True
        self.assertEqual(self.windows(self.both((None, 16))), [('B', True)], 'one packed block: today\'s rule')
        three = tstep.PaddedFakeBlock()
        three.prestaged, three.round_fences, three.fused, three.rounds = SimpleNamespace(), False, None, 0
        step = serving_packed_step.PackedStep([self.a, self.b, three], per_block_widths=True)
        vp._MODE['first'] = True    # a PackedStep resets the mode at attach
        window = step.while_waiting_groups([(self.a, None, ['a']), (self.b, 16, ['b']), (three, 16, ['c'])])
        self.assertEqual([(part.block is self.b, part.prestage) for part in window.windows], [(True, True), (False, False)])

    def test_the_block_epochs_pre_stage_every_packed_block(self):
        vp._MODE.update(first=True, blocks=True)
        self.assertEqual(self.windows(self.both()), [('A', True), ('B', True)])

    def test_one_packed_block_pre_stages_as_ever_with_or_without_the_flags(self):
        self.assertEqual(self.windows(self.both((16, None))), [('A', True)])
        vp._MODE['first'] = True
        self.assertEqual(self.windows(self.both((16, None))), [('A', True)])


class RealBlockTests(tvp.PrestageFixture):
    """The four-user block over the fake device model, with the host-gap flags: the same numbers as the control, byte for byte."""

    def open_hostgap(self, *, audit=False, mode='first', **flags):
        os.environ['QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE'] = '1'
        for name, value in flags.items():
            os.environ[name] = value
        if audit:
            os.environ['QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT'] = '1'
        block = self.open_block()
        vp._MODE.update(first=mode in ('first', 'blocks'), blocks=mode == 'blocks')
        self.addCleanup(lambda: (vp._MODE.update(first=False, blocks=False), vp._LOCAL.clear(), vp._ADDRESSES.clear()))
        return block

    def reference(self, rounds):
        """The flag-off block's predictions over the same rounds (no pre-stage at all)."""
        block = self.open_block(prestage=False)
        served = [self.round(block, users, token_base)[0] for users, token_base in rounds]
        block.close()
        self.block = None
        return served

    def schedule(self, steps=4):
        users, rounds = tvp.base_users(), []
        for step in range(steps):
            rounds.append((users, 3 * step))
            users = tvp.advanced(users, 2 + step)
        return rounds

    def run_window_rounds(self, block, rounds, window=True):
        served = []
        for number, (users, token_base) in enumerate(rounds):
            if number and window:
                self.window(block, users)
            served.append(self.round(block, users, token_base)[0])
        return served

    def test_lite_stages_what_the_full_stage_stages_and_predicts_what_the_control_predicts(self):
        rounds = self.schedule()
        block = self.open_hostgap(audit=True)
        served = self.run_window_rounds(block, rounds)
        chips = len(self.ttnn.get_device_tensors(block.fixture.tokens))
        counts = dict(block.prestaged.counts)
        self.assertEqual(served, self.reference(rounds))
        self.assertEqual(self.paths(), ['full', 'diff', 'diff', 'diff'])
        audits = self.marked(vp.FULL_AUDIT_MARKER)
        self.assertEqual(len(audits), 4, 'the first round (full) and the three diff rounds')
        self.assertEqual([re.search(r'path=(\w+)', line).group(1) for line in audits], ['full', 'diff', 'diff', 'diff'])
        for line in audits:
            self.assertRegex(line, r'^\[PACKED-PRESTAGE-FULLAUDIT\] block=- round=\d+ path=(diff|full) buffers=\d+ checked=\d+ mismatches=0$')
            buffers, checked = (int(value) for value in re.search(r'buffers=(\d+) checked=(\d+)', line).groups())
            self.assertEqual(checked, buffers * chips, 'every destination on every chip')
        self.assertEqual((counts['full_mismatches'], counts['full_audited'] > 0), (0, True))

    def test_the_full_audit_finds_a_stray_write_and_the_round_runs_on_the_full_stage(self):
        rounds = self.schedule(2)
        block = self.open_hostgap(audit=True)
        self.round(block, *rounds[0][:1], rounds[0][1])
        users, token_base = rounds[1]
        self.window(block, users)
        block.fixture.positions.value = torch.full_like(block.fixture.positions.value, 7)   # no epoch bump: only the audit can see it
        predictions, metrics, staged = self.round(block, users, token_base)
        line = self.marked(vp.FULL_AUDIT_MARKER)[-1]
        self.assertRegex(line, r'mismatches=[1-9]\d* at=')
        self.assertIn('1.0', line.split('at=')[1].split(','))
        self.assert_device_holds(block, staged)
        self.assertEqual(predictions, self.reference(rounds)[1], 'the round stayed exact')

    def test_the_audit_flag_alone_does_not_audit_and_the_audit_costs_nothing_without_a_diff(self):
        block = self.open_block()
        self.assertFalse(block.prestaged.full_audit)
        self.assertNotIn('full_audited', block.prestaged.counts)
        self.round(block, tvp.base_users())
        self.assertEqual(self.marked(vp.FULL_AUDIT_MARKER), [])

    def test_the_block_epochs_move_only_this_blocks_epoch_through_window_diff_and_full_stage(self):
        rounds = self.schedule(3)
        block = self.open_hostgap(mode='blocks')
        other = stub_block('B')
        other.prestaged.snapshot = vp.Snapshot(vp.epoch(), [], [], 0, 0.0, 1, local=vp.local_epoch(other.fixture))
        served = []
        for number, (users, token_base) in enumerate(rounds):
            before = vp.epoch()
            if number:
                self.window(block, users)
            served.append(self.round(block, users, token_base)[0])
            self.assertEqual(vp.epoch(), before, 'round %d: no global bump from this block\'s own writes' % number)
            self.assertIsNone(other.prestaged.usable()[1], 'the other block\'s snapshot is still usable')
        own, others = vp.local_epoch(block.fixture), vp.local_epoch(other.fixture)
        self.assertEqual(served, self.reference(rounds))
        self.assertEqual(self.paths(), ['full', 'diff', 'diff'])
        self.assertGreater(own, 0)
        self.assertEqual(others, 0)

    def test_the_block_epochs_keep_every_external_writer_global(self):
        block = self.open_hostgap(mode='blocks')
        users = tvp.base_users()
        self.round(block, users)
        nxt = tvp.advanced(users, 3)
        self.window(block, nxt)
        vp.bump('admission')
        self.round(block, nxt)
        self.assertEqual((self.paths()[-1], self.reasons()[-1]), ('full', 'epoch:admission'))

    def test_a_full_stage_of_the_block_bumps_its_own_epoch_only(self):
        block = self.open_hostgap(mode='blocks')
        before = (vp.epoch(), vp.local_epoch(block.fixture))
        packed_verifier.stage_packed(block.operations, block.model, block.fixture, block.shape,
                                     self.staged_users(block, self.entries(tvp.base_users())))
        self.assertEqual((vp.epoch(), vp.local_epoch(block.fixture)), (before[0], before[1] + 1))

    def test_the_first_window_registers_this_blocks_destinations_and_a_stub_sharing_one_disengages(self):
        block = self.open_hostgap(mode='blocks')
        users = tvp.base_users()
        self.round(block, users)
        self.window(block, tvp.advanced(users, 2))
        self.assertIn(id(block.fixture), vp._ADDRESSES)
        self.assertTrue(vp._ADDRESSES[id(block.fixture)])
        overlap = stub_block('B')
        shared = next(iter(vp._ADDRESSES[id(block.fixture)]))
        self.assertEqual(vp.register_destinations(overlap, ['x'], lambda destination: tuple(
            shared[1] if chip == shared[0] else 0 for chip in range(shared[0] + 1))), 'overlap')
        self.assertFalse(vp._MODE['blocks'])

    def verify_binding_checks(self, block, entries):
        """The retained block's own binding checks made inside the verify outside its replay (the audit's shadow), and the keyword
        state the verify gave its replay: whether the block's validate_bindings was wrapped (its next call skipped) while the replay ran.
        (The fake retained block's replay makes no check of its own: the real one is pinned in ReplayValidatedTests.)"""
        retained = block.fixture.retained
        replays = []
        original = retained.replay
        checker = retained.validate_bindings = Mock()

        def wrapped(operation, **kwargs):
            replays.append({'wrapped': retained.validate_bindings is not checker, **kwargs})
            return original(operation, **kwargs)

        retained.replay = wrapped
        try:
            predictions, metrics = block.verify(entries)
        finally:
            retained.replay = original
        self.assertIs(retained.validate_bindings, checker, 'the instance attribute is put back after the replay')
        return checker.call_count, replays, predictions, metrics

    def diff_round_checks(self, **flags):
        block = self.open_hostgap(**flags)
        users = tvp.base_users()
        self.round(block, users)
        self.round(block, tvp.advanced(users, 2))
        nxt = tvp.advanced(users, 5)
        self.window(block, nxt)
        entries = self.entries(nxt)
        checks, replays, predictions, metrics = self.verify_binding_checks(block, entries)
        for segment, prefix in zip(metrics['segments'], (9, 16, 0, 4)):
            block.commit_user(segment, prefix)
        return checks, replays, predictions, self.paths()[-1]

    def test_the_window_validate_tells_the_replay_to_skip_its_check_only_after_a_usable_snapshot(self):
        off = self.diff_round_checks()
        on = self.diff_round_checks(QWEN_FAST_TP4_WINDOW_VALIDATE='1')
        self.assertEqual((off[3], on[3]), ('diff', 'diff'))
        self.assertEqual(off[1], [{'wrapped': False}], 'flag off: the replay is called as it always was')
        self.assertEqual(on[1], [{'wrapped': True}])
        self.assertEqual((off[0], on[0]), (0, 0), 'no shadow without the audit')
        self.assertEqual(on[2], off[2], 'the same predictions')

    def test_under_the_audit_the_skipped_check_runs_as_a_shadow(self):
        self.h1a.clear()
        on = self.diff_round_checks(QWEN_FAST_TP4_WINDOW_VALIDATE='1', QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT='1')
        self.assertEqual(on[1], [{'wrapped': True}])
        self.assertEqual(on[0], 1, 'the skipped check ran, once, before the replay')
        self.assertTrue(any(line.startswith('[PACKED-PRESTAGE-SHADOW] round=') and line.endswith('retained_bindings=ok')
                            for line in self.h1a))

    def test_a_round_on_the_full_path_still_takes_the_replays_check(self):
        block = self.open_hostgap(QWEN_FAST_TP4_WINDOW_VALIDATE='1')
        users = tvp.base_users()
        self.round(block, users)
        checks, replays, predictions, metrics = self.verify_binding_checks(block, self.entries(tvp.advanced(users, 2)))
        self.assertEqual((checks, replays), (0, [{'wrapped': False}]), 'no snapshot: the replay keeps its own check')
        self.assertEqual(self.paths()[-1], 'full')

    def test_stage_0_writes_new_lines_only_under_the_log_flag(self):
        users = tvp.base_users()
        block = self.open_hostgap()
        self.run_window_rounds(block, self.schedule(3))
        self.assertEqual(self.marked(vp.HOSTGAP_VERIFY_MARKER), [])
        self.assertEqual(self.marked(vp.HOSTGAP_WINDOW_MARKER), [])
        self.h1a.clear()
        block = self.open_hostgap(QWEN_FAST_TP4_HOSTGAP_LOG='1')
        self.run_window_rounds(block, self.schedule(3))
        verify_lines = self.marked(vp.HOSTGAP_VERIFY_MARKER)
        self.assertEqual(len(verify_lines), 3)
        self.assertRegex(verify_lines[1], r'^\[PACKED-HOSTGAP-VERIFY\] block=- round=2 live=4 path=diff reason=- bind_ms=[0-9.]+ input_ms=[0-9.]+ '
                                          r'stage_cpu_ms=[0-9.]+ reads_ms=[0-9.]+ checks_ms=[0-9.]+ readback_ms=[0-9.]+ after_stage_cpu_ms=[0-9.]+ '
                                          r'readback_cpu_ms=[0-9.]+$')
        windows = self.marked(vp.HOSTGAP_WINDOW_MARKER)
        self.assertEqual(len(windows), 2)
        self.assertRegex(windows[0], r'^\[PACKED-HOSTGAP-WINDOW\] round=2 blocks=1 stage_window_ms=[0-9.]+ prestage_ms=[0-9.]+ prestaged=1 '
                                     r'window_ms=[0-9.]+ fence_wait_ms=[0-9.]+$')
        self.assertTrue(vp._GC['installed'], 'the gc log is installed at the first block built with the log flag')
        self.assertTrue(users)


class TwoRealBlockTests(tvp.PrestageFixture):
    """Two REAL four-user blocks over one eight-slot pool and the fake device model, in blocks mode (per-block epochs), driven as
    serving drives them: both windows, one fence, then verify and commits of A, then verify and commits of B. Review should-fix 6:
    nothing else exercises A's window, A's diff, A's verify and commits, and then B's diff over a shared process."""

    PAIR = ((0, 1, 2, 3), (4, 5, 6, 7))

    def setUp(self):
        super().setUp()
        packed = tvp.tpv
        self.sets = [packed.packed_tables(self.ttnn, users=4), packed.packed_tables(self.ttnn, users=4)]
        self.pool = packed.pool(self.ttnn, self.helpers, users=8, packed={(4, 16): self.sets[0]})
        self.pool.packed_replay = lambda count, rows: next((tables for tables in self.sets if not tables.taken), self.sets[-1])
        self.hooks = {}
        base = self.ttnn.execute_trace

        def execute_trace(mesh, trace, cq_id=0, blocking=True):
            base(mesh, trace, cq_id=cq_id, blocking=blocking)
            hook = self.hooks.get(trace)
            if hook is not None:
                hook()

        self.ttnn.execute_trace = execute_trace
        self.addCleanup(lambda: (vp._MODE.update(first=False, blocks=False), vp._LOCAL.clear(), vp._ADDRESSES.clear()))

    def open_pair(self, **flags):
        os.environ.update(flags)
        verifier_engine.note_prefill()
        for tables in self.sets:
            tables.taken = False
        self.hooks.clear()
        blocks = []
        for slots in self.PAIR:
            block = packed_verifier.PackedVerifierEngine(
                self.ttnn, self.model, self.helpers, 'sampler', pool=self.pool, shared_weights=self.weights, shape=self.shape(),
                feature_taps=tvp.tpv.TAPS, pool_slots=slots)
            self.hooks[block.trace] = tpp.DeviceModel(self, block)
            blocks.append(block)
        return blocks

    def entries_of(self, index, users, token_base):
        made = []
        for name, (position, pages) in users.items():
            slot = 4 * index + NAMES.index(name)
            owner = tvp.tpv.request('%s%d' % (name, index), self.pool.slots[slot], position, 1)
            owner.engine.pages = pages.clone()
            first = (token_base + 10 * NAMES.index(name)) % 80
            made.append(tvp.tpv.entry(owner, range(first, first + 16)))
        return made

    def requests_of(self, index, users):
        requests = []
        for name, (position, pages) in users.items():
            carry = [list(snapshot) for snapshot in self.pool.slots[4 * index + NAMES.index(name)].verifier.carry]
            requests.append(SimpleNamespace(session=SimpleNamespace(finished=False, position=position),
                                            engine=SimpleNamespace(carry=carry, pages=pages.clone())))
        return requests

    def rounds(self, steps=3):
        users = [tvp.base_users(), tvp.advanced(tvp.base_users(), 7)]
        plan = []
        for step in range(steps):
            plan.append((users, 3 * step))
            users = [tvp.advanced(item, 2 + step) for item in users]
        return plan

    def drive(self, blocks, plan, window):
        served = []
        for number, (users, token_base) in enumerate(plan):
            if number and window:
                waiting = [vp.WhileWaiting(block, self.requests_of(index, users[index])) for index, block in enumerate(blocks)]
                for item in waiting:
                    run_while_waiting(item)
                self.ttnn.synchronize_device(self.model.mesh_device)
                for item in waiting:
                    note_fenced(item)
            for index, block in enumerate(blocks):
                entries = self.entries_of(index, users[index], token_base)
                predictions, metrics = block.verify(entries)
                for segment, prefix in zip(metrics['segments'], (9, 16, 0, 4)):
                    block.commit_user(segment, prefix)
                served.append((index, number, predictions))
        return served

    def reference(self, plan):
        blocks = self.open_pair()
        served = self.drive(blocks, plan, window=False)
        for block in blocks:
            block.close()
        tvp.tpv.FakeModelBatch.instances = []
        self.h1a.clear()
        return served

    def run_arm(self, audit):
        plan = self.rounds()
        expected = self.reference(plan)
        flags = {'QWEN_FAST_PRESTAGE': '1', 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE': '1', 'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS': '1',
                 'QWEN_FAST_TP4_HOSTGAP_LOG': '1'}
        if audit:
            flags['QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT'] = '1'
        blocks = self.open_pair(**flags)
        self.assertEqual(vp.engage_two_block(blocks, dict(os.environ)), 'blocks', self.h1a)
        served = self.drive(blocks, plan, window=True)
        self.assertEqual(served, expected, 'every prediction of both blocks equals the flag-off run over the same rounds')
        verifies = [re.search(r'block=(\S+) round=(\d+) live=4 path=(\w+) reason=(\S+)', line).groups()
                    for line in self.marked(vp.HOSTGAP_VERIFY_MARKER)]
        for label in 'AB':
            paths = [path for block, number, path, reason in verifies if block == label]
            self.assertEqual(paths, ['full', 'diff', 'diff'], 'block %s: only the first round is full' % label)
        self.assertTrue(vp._MODE['blocks'], 'the blocks never shared a destination')
        return blocks

    def test_both_real_blocks_take_the_diff_path_after_the_first_round_and_predict_what_the_control_predicts(self):
        self.run_arm(audit=False)

    def test_audited_both_blocks_read_back_every_destination_with_zero_mismatches(self):
        blocks = self.run_arm(audit=True)
        chips = len(self.ttnn.get_device_tensors(blocks[0].fixture.tokens))
        audits = [re.search(r'block=(\S+) round=\d+ path=(\w+) buffers=(\d+) checked=(\d+) mismatches=(\d+)', line).groups()
                  for line in self.marked(vp.FULL_AUDIT_MARKER)]
        for label in 'AB':
            own = [item for item in audits if item[0] == label]
            self.assertEqual([item[1] for item in own], ['full', 'diff', 'diff'], label)
            self.assertTrue(all(int(item[4]) == 0 and int(item[3]) == int(item[2]) * chips for item in own), own)


class CompositeLineTests(Clean):
    def block(self, name, log):
        return tvp.CompositeWindowTests.block(self, name, log)

    def test_two_blocks_write_one_window_line_and_one_block_alone_writes_its_own(self):
        os.environ['QWEN_FAST_TP4_HOSTGAP_LOG'] = '1'
        log = []
        windows = [vp.WhileWaiting(self.block(name, log), [name], prestage=name == 'A') for name in 'AB']
        composite = vp.CompositeWindow(windows)
        composite()
        composite.fenced()
        found = lines_of(self.log, vp.HOSTGAP_WINDOW_MARKER)
        self.assertEqual(len(found), 1)
        self.assertRegex(found[0], r'blocks=2 stage_window_ms=[0-9.]+,[0-9.]+ prestage_ms=[0-9.]+,[0-9.]+ prestaged=1,0 window_ms=')
        self.assertIn(('prestage', 'A'), log)
        self.assertNotIn(('prestage', 'B'), log)
        self.log.clear()
        window = vp.WhileWaiting(self.block('A', []), ['A'])
        window()
        window.fenced()
        self.assertEqual(len(lines_of(self.log, vp.HOSTGAP_WINDOW_MARKER)), 1)

    def test_without_the_log_flag_a_window_writes_nothing(self):
        windows = [vp.WhileWaiting(self.block(name, []), [name]) for name in 'AB']
        composite = vp.CompositeWindow(windows)
        composite()
        composite.fenced()
        self.assertEqual(self.log, [])

    def test_a_window_that_raises_still_reports_the_time_it_took(self):
        os.environ['QWEN_FAST_TP4_HOSTGAP_LOG'] = '1'
        first, second = self.block('A', []), self.block('B', [])
        first.prestaged.prestage_requests = Mock(side_effect=RuntimeError('x'))
        composite = vp.CompositeWindow([vp.WhileWaiting(first, ['A']), vp.WhileWaiting(second, ['B'])])
        composite()
        composite.fenced()
        self.assertEqual(len(lines_of(self.log, vp.HOSTGAP_WINDOW_MARKER)), 1)


class StageZeroTests(Clean):
    def test_the_entry_line_is_written_once_from_what_the_bridge_left(self):
        vp.entry_line(1.0)
        self.assertEqual(self.log, [])
        vp.note_scratch('entry', dict(admit_ms=1.5, update_states_ms=2.5, storage_ms=0.5, reservation_ms=0.25, refresh_ms=0.75,
                                      refresh_writes=3, started=0.0))
        vp.entry_line(3.25)
        vp.entry_line(9.0)
        self.assertEqual(len(self.log), 1)
        self.assertRegex(self.log[0], r'^\[PACKED-ENTRY\] admit_ms=1.50 update_states_ms=2.50 storage_ms=0.50 reservation_ms=0.25 '
                                      r'refresh_ms=0.75 refresh_writes=3 checks_ms=3.25 entry_ms=[0-9.]+$')

    def test_the_collect_split_sums_the_quads_and_is_taken_once(self):
        vp.add_collect_split(1.0, 2.0)
        vp.add_collect_split(0.5, 0.25)
        self.assertEqual(vp.take_scratch('collect'), [1.5, 2.25])
        self.assertIsNone(vp.take_scratch('collect'))

    def test_the_gc_callback_only_records_and_the_lines_are_written_from_the_round_lines(self):
        import gc

        self.assertTrue(vp.install_gc_log())
        self.assertFalse(vp.install_gc_log())
        callback = vp._GC['callback']
        self.assertIn(callback, gc.callbacks)
        with patch.object(vp.time, 'perf_counter', side_effect=[10.0, 10.0 + 0.002]):
            callback('start', {})
            callback('stop', dict(generation=2, collected=7))
        self.assertEqual(vp._GC['pending'], [], 'a collection under the threshold records nothing')
        with patch.object(vp.time, 'perf_counter', side_effect=[10.0, 10.0 + 0.0123]):
            callback('start', {})
            callback('stop', dict(generation=2, collected=7))
        self.assertEqual(self.log, [], 'the callback never logs: a collection can start inside the logger\'s own emit')
        self.assertEqual(len(vp._GC['pending']), 1)
        vp.entry_line(1.0)
        self.assertEqual([line for line in self.log if 'collected=7' in line], ['[PINDIAG] gc gen=2 collected=7 ms=12.30'])
        self.assertEqual(vp._GC['pending'], [], 'drained once')
        with patch.object(vp.time, 'perf_counter', side_effect=[10.0, 10.0 + 0.02]):
            callback('start', {})
            callback('stop', dict(generation=0, collected=1))
        window = SimpleNamespace(timing=dict(ended=0.0, stage_window_ms=0.0, prestage_ms=0.0, prestaged=0), block=SimpleNamespace(rounds=0))
        vp.window_line([window])
        self.assertEqual(len([line for line in self.log if 'collected=1 ' in line]), 1, 'the window line drains too')

    def test_the_gc_record_is_bounded_and_uninstall_removes_the_callback_and_the_record(self):
        import gc

        vp.install_gc_log()
        callback = vp._GC['callback']
        for _ in range(vp.GC_PENDING_MAX + 10):
            with patch.object(vp.time, 'perf_counter', side_effect=[1.0, 1.1]):
                callback('start', {})
                callback('stop', dict(generation=1, collected=0))
        self.assertEqual(len(vp._GC['pending']), vp.GC_PENDING_MAX)
        vp.uninstall_gc_log()
        self.assertNotIn(callback, gc.callbacks)
        self.assertEqual((vp._GC['installed'], vp._GC['pending'], vp._GC['callback']), (False, [], None))
        vp.uninstall_gc_log()

    def test_thread_time_is_milliseconds_under_the_log_flag_and_not_read_without_it(self):
        with patch.object(vp.time, 'thread_time', return_value=1.5) as clock:
            self.assertEqual(vp.thread_ms(), 0.0)
            clock.assert_not_called()
            with patch.dict(os.environ, {'QWEN_FAST_TP4_HOSTGAP_LOG': '1'}):
                self.assertEqual(vp.thread_ms(), 1500.0)


class EntryTests(Clean):
    """The bridge's step entry: the stage 0 split, and 1d's one storage check per distinct validator."""

    def run_step(self, bridges, scheduled, runner, order=None, **environment):
        def packed_step(entries, *, cancelled):
            if order is not None:
                order.append('packed-step')
            return [SimpleNamespace(request_id=entry['request_id'], token_ids=[1]) for entry in entries]

        reserve = patch('serving_vllm_state.validate_runner_reservation',
                        side_effect=None if order is None else (lambda *args: order.append('reserve')))
        with patch.dict(os.environ, environment), patch('serving_vllm_state.apply_committed_output'), reserve, \
                patch('serving_packed_bridge.packed_model_runner_output', side_effect=lambda values: [v.request_id for v in values]):
            return serving_packed_bridge.execute_packed_decode(bridges, scheduled, cancelled=lambda: False, packed_step=packed_step)

    def fixture(self):
        case = tbridge.PackedBridgeTests('test_the_device_step_runs_once_and_update_states_runs_once')
        return case.fixture()

    def test_the_storage_check_runs_once_for_a_shared_validator_with_the_diet_and_per_bridge_without(self):
        for diet, expected in (('0', 2), ('1', 1)):
            runner, bridges, scheduled, _ = self.fixture()
            shared = Mock()
            for item in bridges.values():
                item.validate_storage = shared
            self.run_step(bridges, scheduled, runner, QWEN_FAST_TP4_ENTRY_DIET=diet)
            self.assertEqual(shared.call_count, expected, diet)

    def test_a_bound_method_of_one_owner_is_one_validator_and_distinct_owners_each_run(self):
        class Owner:
            def __init__(self):
                self.calls = 0

            def validate(self):
                self.calls += 1

        runner, bridges, scheduled, _ = self.fixture()
        one, other = Owner(), Owner()
        bridges['A'].validate_storage, bridges['B'].validate_storage = one.validate, one.validate
        self.run_step(bridges, scheduled, runner, QWEN_FAST_TP4_ENTRY_DIET='1')
        self.assertEqual(one.calls, 1)
        runner, bridges, scheduled, _ = self.fixture()
        bridges['A'].validate_storage, bridges['B'].validate_storage = one.validate, other.validate
        self.run_step(bridges, scheduled, runner, QWEN_FAST_TP4_ENTRY_DIET='1')
        self.assertEqual((one.calls, other.calls), (2, 1))

    def test_a_failing_validator_fails_the_step_and_poisons_every_bridge_with_or_without_the_diet(self):
        for diet in ('0', '1'):
            runner, bridges, scheduled, _ = self.fixture()
            for item in bridges.values():
                item.validate_storage = Mock(side_effect=ValueError('storage moved'))
            with self.assertRaises(ValueError):
                self.run_step(bridges, scheduled, runner, QWEN_FAST_TP4_ENTRY_DIET=diet)
            self.assertTrue(all(item.failed for item in bridges.values()))
            runner._update_states.assert_not_called()

    def test_the_step_is_the_same_with_every_flag_on_as_off_and_the_order_of_calls_is_today_s(self):
        results = []
        for flags in ({}, {'QWEN_FAST_TP4_ENTRY_DIET': '1', 'QWEN_FAST_TP4_HOSTGAP_LOG': '1'}):
            runner, bridges, scheduled, _ = self.fixture()
            order = []
            runner._update_states = Mock(side_effect=lambda value: order.append('update'))
            for name, item in bridges.items():
                item.page_binding.refresh = Mock(side_effect=lambda *a, name=name, **k: order.append('refresh-' + name))
                item.validate_storage = None
            result = self.run_step(bridges, scheduled, runner, order=order, **flags)
            results.append((result, order))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][1], ['update', 'reserve', 'refresh-B', 'reserve', 'refresh-A', 'packed-step'])

    def test_the_log_flag_leaves_the_split_for_the_packed_step_and_nothing_without_it(self):
        runner, bridges, scheduled, _ = self.fixture()
        self.run_step(bridges, scheduled, runner)
        self.assertEqual(vp._SCRATCH, {})
        runner, bridges, scheduled, _ = self.fixture()
        for item in bridges.values():
            item.page_binding.refresh = Mock(side_effect=[True])
        bridges['B'].page_binding.refresh = Mock(return_value=False)
        self.run_step(bridges, scheduled, runner, QWEN_FAST_TP4_HOSTGAP_LOG='1')
        entry = vp._SCRATCH['entry']
        self.assertEqual(sorted(entry), ['admit_ms', 'refresh_ms', 'refresh_writes', 'reservation_ms', 'started', 'storage_ms',
                                         'update_states_ms'])
        self.assertEqual(entry['refresh_writes'], 1, 'only a refresh that wrote the device counts')

    def test_a_two_block_step_writes_the_entry_line_once_before_its_first_verify(self):
        verifier_engine.note_prefill()
        case = tstep.TwoBlockStepTests('test_partitions_entries_by_block_and_runs_each_blocks_round_in_configured_order')
        case.setUp()
        os.environ['QWEN_FAST_TP4_HOSTGAP_LOG'] = '1'
        vp.note_scratch('entry', dict(admit_ms=1.0, update_states_ms=1.0, storage_ms=1.0, reservation_ms=1.0, refresh_ms=1.0,
                                      refresh_writes=0, started=0.0))
        verify_order = []
        for name, block in (('A', case.block_a), ('B', case.block_b)):
            original = block.verify
            block.verify = lambda entries, name=name, original=original: (
                verify_order.append((name, len(lines_of(self.log, vp.ENTRY_MARKER)))), original(entries))[1]
        case.step(case.four())
        self.assertEqual(verify_order, [('A', 1), ('B', 1)], 'one line, written before the first verify')
        self.assertEqual(len(lines_of(self.log, vp.ENTRY_MARKER)), 1)


class IncrementalPageValidationTests(Clean):
    """1d: checked_blocks accepts and refuses exactly what validate_blocks does, over random allocations."""

    def binding(self):
        return tbinding.PageBindingTests('test_append_refreshes_every_captured_table_without_replacing_buffers').fixture()[:2]

    def outcome(self, binding, engine, blocks, diet):
        copies = engine.operations.copy_host_to_device_tensor.call_count
        with patch.dict(os.environ, {'QWEN_FAST_TP4_ENTRY_DIET': '1' if diet else '0'}):
            try:
                result = binding.refresh(blocks, position=4096, rows=16)
            except ValueError as failure:
                result = 'refused:%s' % failure
        return result, binding.blocks, engine.operations.copy_host_to_device_tensor.call_count - copies

    def test_random_allocations_are_accepted_and_refused_exactly_as_the_full_validation_does(self):
        rng = random.Random(7)
        plain, plain_engine = self.binding()
        diet, diet_engine = self.binding()
        held = tuple(range(4, 68))
        accepted = refused = 0
        for step in range(400):
            kind = rng.choice(('same', 'grow', 'grow', 'grow', 'dup', 'range', 'bool', 'prefix', 'shrink', 'capacity', 'negative'))
            if kind == 'same':
                candidate = held
            elif kind == 'grow':
                taken = set(held)
                fresh = [value for value in range(68, 200) if value not in taken][:rng.randrange(1, 4)]
                candidate = held + tuple(fresh)
            elif kind == 'dup':
                candidate = held + (rng.choice(held),)
            elif kind == 'range':
                candidate = held + (256 + rng.randrange(5),)
            elif kind == 'bool':
                candidate = held + (True,)
            elif kind == 'prefix':
                index = rng.randrange(len(held))
                candidate = held[:index] + (199,) + held[index + 1:]
            elif kind == 'shrink':
                candidate = held[:-1]
            elif kind == 'negative':
                candidate = held + (-1,)
            else:
                candidate = held + tuple(range(200, 200 + 80))
            with self.subTest(step=step, kind=kind):
                a = self.outcome(plain, plain_engine, candidate, False)
                b = self.outcome(diet, diet_engine, candidate, True)
                self.assertEqual(a, b)
                if isinstance(a[0], str):
                    refused += 1
                else:
                    accepted += 1
                    held = tuple(plain.blocks)
        self.assertGreater(accepted, 40)
        self.assertGreater(refused, 40)

    def test_the_suffix_route_does_not_run_the_full_validation_and_a_poisoned_set_falls_back_to_it(self):
        binding, engine = self.binding()
        with patch.dict(os.environ, {'QWEN_FAST_TP4_ENTRY_DIET': '1'}):
            self.assertFalse(binding.refresh(tuple(range(4, 68)), position=4000, rows=16))
            self.assertIsNotNone(binding._known_set)
            with patch.object(binding, 'validate_blocks', wraps=binding.validate_blocks) as full:
                self.assertTrue(binding.refresh(tuple(range(4, 69)), position=4096, rows=16))
                full.assert_not_called()
                binding.blocks = tuple(list(binding.blocks))   # an allocation set from elsewhere: the held set no longer vouches
                self.assertFalse(binding.refresh(tuple(binding.blocks), position=4096, rows=16))
                full.assert_called_once()

    def test_flag_off_the_binding_never_builds_the_set(self):
        binding, engine = self.binding()
        with patch.dict(os.environ, {}):
            binding.refresh(tuple(range(4, 69)), position=4096, rows=16)
            binding.refresh(tuple(range(4, 69)), position=4096, rows=16)
        self.assertIsNone(binding._known_set)


GOOD_STEADY = {'concurrent8_steady': {}}


def smoke(*names):
    return 'SMOKE_JSON ' + json.dumps({name: {} for name in names})


def prestage_lines(count, path='diff', reason='-', live=4):
    return ['[PACKED-PRESTAGE] round=%d path=%s buffers=3 reason=%s live=%d' % (n, path, reason, live) for n in range(count)]


def audit_lines(blocks='AB', mismatches=0, count=3, full=True, full_mismatches=0, checked=596):
    lines = ['[PACKED-PRESTAGE-FULLAUDIT] block=%s round=%d path=diff buffers=149 checked=%d mismatches=%d' % (block, n, checked, mismatches)
             for block in blocks for n in range(count)]
    if full:
        lines += ['[PACKED-PRESTAGE-FULLAUDIT] block=%s round=0 path=full buffers=149 checked=%d mismatches=%d' % (
            block, checked, full_mismatches) for block in blocks]
    return lines


def verify_lines(count, blocks='AB', path='diff', reason='-', live=4):
    return ['[PACKED-HOSTGAP-VERIFY] block=%s round=%d live=%d path=%s reason=%s bind_ms=0.10 input_ms=1.00 stage_cpu_ms=0.90 reads_ms=0.5 '
            'checks_ms=0.1 readback_ms=2.0 after_stage_cpu_ms=0.3 readback_cpu_ms=0.2' % (block, n, live, path, reason)
            for block in blocks for n in range(count)]


def engaged(blocks=2, mode='blocks'):
    lines = ['[PINDIAG] verify prestage two-block engaged block=%s users=4 mode=%s window_validate=1 audit=1' % ('AB'[i], mode)
             for i in range(blocks)]
    return lines + (['[PINDIAG] verify prestage block epochs engaged blocks=2'] if mode == 'blocks' else [])


def text(*groups):
    return '\n'.join(line for group in groups for line in group)


ARM_ENV = {'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE': '1', 'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS': '1', 'QWEN_FAST_M3_BLOCKS': '2',
           'QWEN_FAST_TP4_HOSTGAP_LOG': '1', 'QWEN_FAST_TP': '4'}
LITE_ENV = {name: value for name, value in ARM_ENV.items() if name != 'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS'}


class SmokeRuleTests(unittest.TestCase):
    def judge(self, env, container, steady=True):
        return c2_smoke_check.hostgap_problems(env, container, steady)[0]

    def test_a_clean_block_epochs_arm_passes(self):
        self.assertEqual(self.judge(ARM_ENV, text(engaged(), prestage_lines(40), verify_lines(40))), [])

    def test_a_profile_without_the_flag_logs_none_of_the_lines(self):
        self.assertEqual(self.judge({}, text(prestage_lines(40, 'full', 'no-snapshot'))), [])
        self.assertTrue(self.judge({}, text(engaged())))
        self.assertTrue(self.judge({'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS': '1'}, ''))

    def test_each_block_must_log_its_engaged_line_and_a_refusal_fails(self):
        self.assertTrue(self.judge(ARM_ENV, text(engaged(1), prestage_lines(40))))
        refused = ['[PINDIAG] verify prestage two-block refused reason=a_block_was_built_without_the_pre-stage']
        self.assertTrue(any('refused at attach' in item for item in self.judge(ARM_ENV, text(refused, prestage_lines(40)))))
        disengaged = text(engaged(), ['[PINDIAG] verify prestage block epochs refused reason=staging_destinations_overlap_between_blocks'],
                          prestage_lines(40))
        self.assertTrue(any('refused or disengaged' in item for item in self.judge(ARM_ENV, disengaged)))

    def test_the_lite_arm_needs_no_epochs_line_and_may_take_the_second_block_full(self):
        self.assertEqual(self.judge(LITE_ENV, text(engaged(mode='first'), prestage_lines(20), verify_lines(20, 'A'),
                                                   verify_lines(30, 'B', 'full', 'no-snapshot'))), [])
        misses = text(engaged(mode='first'), prestage_lines(20), verify_lines(20, 'A'), verify_lines(5, 'A', 'full', 'no-snapshot'))
        self.assertTrue(any('took the diff path' in item for item in self.judge(LITE_ENV, misses)))

    def test_the_diff_share_of_the_pre_staged_blocks_is_95_percent_and_a_wrong_epoch_is_a_miss(self):
        base = (engaged(), prestage_lines(40))
        mostly = text(*base, verify_lines(30), verify_lines(10, 'AB', 'full', 'no-snapshot'))
        self.assertTrue(any('took the diff path' in item and 'no-snapshot' in item for item in self.judge(ARM_ENV, mostly)))
        # an epoch killed by the block's own verify (or any other non-external writer) is a miss, not an excused event
        for reason in ('epoch:verify', 'epoch:verify-failed', 'epoch:stage_packed', 'epoch:prestage', 'destinations'):
            with self.subTest(reason=reason):
                wrong = text(*base, verify_lines(30), verify_lines(3, 'AB', 'full', reason))
                self.assertTrue(any('took the diff path' in item and reason in item for item in self.judge(ARM_ENV, wrong)), reason)
        # the six external writers are left out of the count
        for reason in ('epoch:admission', 'epoch:detach', 'epoch:prefill', 'epoch:prefill-chunk', 'epoch:bookkeeping', 'epoch:lane-switch'):
            with self.subTest(reason=reason):
                self.assertEqual(self.judge(ARM_ENV, text(*base, verify_lines(30), verify_lines(20, 'AB', 'full', reason))), [])
        few = text(*base, verify_lines(60), verify_lines(2, 'AB', 'full', 'no-snapshot'))
        self.assertEqual(self.judge(ARM_ENV, few), [])
        self.assertTrue(any('4-live verifies' in item for item in self.judge(ARM_ENV, text(engaged(), prestage_lines(5), verify_lines(3)))))
        # a verify of fewer live users is not a 4+4 round
        self.assertEqual(self.judge(ARM_ENV, text(*base, verify_lines(40), verify_lines(30, 'AB', 'full', 'no-snapshot', live=3))), [])

    def test_without_the_log_flag_the_share_is_unread_not_a_problem(self):
        # A traffic profile drops the diagnostic log (v578): the lever is still judged engaged and on the diff path, the share is unread.
        env = {name: value for name, value in ARM_ENV.items() if name != 'QWEN_FAST_TP4_HOSTGAP_LOG'}
        problems, facts = c2_smoke_check.hostgap_problems(env, text(engaged(), prestage_lines(40)), True)
        self.assertFalse(any('HOSTGAP_LOG' in item for item in problems), problems)
        self.assertEqual(facts['hostgap_share'], 'unread (QWEN_FAST_TP4_HOSTGAP_LOG off)')

    def test_the_ramp_threshold_is_not_applied_to_an_audited_verify(self):
        self.assertTrue(c2_smoke_check.audited_verify({'QWEN_FAST_VERIFY_T1_AUDIT': '1'}))
        self.assertTrue(c2_smoke_check.audited_verify({'QWEN_FAST_VERIFY_T2_AUDIT': '1'}))
        self.assertFalse(c2_smoke_check.audited_verify({'QWEN_FAST_VERIFY_T1_AUDIT': '0', 'QWEN_FAST_VERIFY_T2_AUDIT': '0'}))
        self.assertFalse(c2_smoke_check.audited_verify(None))

    def test_the_lever_is_not_judged_without_the_eight_user_steady_smoke(self):
        self.assertTrue(any('concurrent8_steady' in item for item in self.judge(ARM_ENV, text(engaged()), steady=False)))

    def test_the_full_audit_needs_zero_mismatches_a_line_for_each_block_and_the_shadow(self):
        env = dict(ARM_ENV, QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT='1', QWEN_FAST_TP4_WINDOW_VALIDATE='1')
        shadow = ['[PACKED-PRESTAGE-SHADOW] round=3 retained_bindings=ok']
        body = engaged() + prestage_lines(40) + verify_lines(40)
        clean = text(body, audit_lines('AB'), shadow)
        self.assertEqual(self.judge(env, clean), [])
        self.assertTrue(any('after a DIFF write' in item for item in self.judge(env, text(body, audit_lines('AB', mismatches=2), shadow))))
        self.assertTrue(any('block B' in item for item in self.judge(env, text(body, audit_lines('A'), shadow))))
        self.assertTrue(any('shadow' in item for item in self.judge(env, text(body, audit_lines('AB')))))
        lite = dict(env)
        del lite['QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS']
        self.assertEqual(self.judge(lite, text(engaged(mode='first'), prestage_lines(20), verify_lines(20, 'A'), audit_lines('A'), shadow)), [])
        unaudited = dict(ARM_ENV)
        self.assertTrue(self.judge(unaudited, text(body, audit_lines('AB'))))

    def test_a_mismatch_after_a_full_stage_is_the_comparators_and_is_reported_apart_from_the_levers(self):
        env = dict(ARM_ENV, QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT='1')
        body = engaged() + prestage_lines(40) + verify_lines(40)
        found = self.judge(env, text(body, audit_lines('AB', full_mismatches=3)))
        self.assertTrue(any('FULL stage' in item and 'comparator' in item for item in found), found)
        self.assertFalse(any('DIFF write' in item for item in found), 'the diff path alone is clean')
        found = self.judge(env, text(body, audit_lines('AB', mismatches=1)))
        self.assertTrue(any('DIFF write' in item for item in found))
        self.assertFalse(any('comparator' in item for item in found))
        none_full = self.judge(env, text(body, audit_lines('AB', full=False)))
        self.assertTrue(any('path=full line' in item for item in none_full), none_full)

    def test_a_short_read_back_cannot_pass(self):
        env = dict(ARM_ENV, QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT='1')
        body = engaged() + prestage_lines(40) + verify_lines(40)
        self.assertTrue(any('fewer than buffers x 4 chips' in item for item in self.judge(env, text(body, audit_lines('AB', checked=300)))))
        self.assertTrue(any('fewer than buffers x 2 chips' in item for item in self.judge(
            dict(env, QWEN_FAST_TP='2'), text(body, audit_lines('AB', checked=596)))))
        self.assertEqual(self.judge(dict(env, QWEN_FAST_TP='2'), text(body, audit_lines('AB', checked=298))), [])

    def test_the_check_reads_the_profile_env_and_files_the_facts(self):
        container = text(engaged(), prestage_lines(40), verify_lines(40))
        problems, facts = c2_smoke_check.check(smoke('warmup', 'concurrent8_steady'), container, False, env=ARM_ENV)
        self.assertFalse([item for item in problems if 'two-block' in item or 'diff path' in item], problems)
        self.assertEqual(facts['hostgap']['hostgap_engaged'], 2)

    def test_the_real_log_lines_are_the_ones_the_rules_read(self):
        log = []
        with patch.object(vp, 'log_line', side_effect=log.append):
            blocks = [stub_block(), stub_block()]
            vp.engage_two_block(blocks, dict(ARM_ENV, QWEN_FAST_PRESTAGE='1'))
            vp._MODE.update(first=False, blocks=False)
        found = c2_smoke_check.hostgap_facts('\n'.join(log))
        self.assertEqual((found['hostgap_engaged'], found['hostgap_epochs_engaged'], found['hostgap_refused']), (2, 1, 0))
        line = ('[PACKED-PRESTAGE] round=3 path=diff buffers=1 reason=- live=4', '[PACKED-PRESTAGE] round=4 path=full buffers=149 '
                'reason=no-snapshot live=4')
        found = c2_smoke_check.hostgap_facts('\n'.join(line))
        self.assertEqual((found['hostgap_live4_diff'], found['hostgap_live4_full_no_snapshot']), (1, 1))
        found = c2_smoke_check.hostgap_facts('\n'.join(verify_lines(2, 'A', 'full', 'epoch:verify') + audit_lines('A', count=1)))
        self.assertEqual(found['hostgap_verify_live4'], [('A', 'full', 'epoch:verify')] * 2)
        self.assertEqual((found['hostgap_full_audits'], found['hostgap_full_audits_full']), (2, 1))


def load_profiles():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)['profiles']


ADDED = {
    LITE: {'QWEN_FAST_TP4_HOSTGAP_LOG': '1', 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE': '1', 'QWEN_FAST_TP4_WINDOW_VALIDATE': '1'},
    ARM: {'QWEN_FAST_TP4_HOSTGAP_LOG': '1', 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE': '1', 'QWEN_FAST_TP4_WINDOW_VALIDATE': '1',
          'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS': '1', 'QWEN_FAST_TP4_ENTRY_DIET': '1'},
}
ADDED[LITE_AUDIT] = dict(ADDED[LITE], QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT='1')
ADDED[ARM_AUDIT] = dict(ADDED[ARM], QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT='1')


class ProfileTests(unittest.TestCase):
    def test_each_arm_is_the_timed_control_plus_its_flags_and_nothing_else(self):
        found = load_profiles()
        control = found[CONTROL]
        for name, flags in ADDED.items():
            with self.subTest(name=name):
                profile = found[name]
                self.assertEqual({key: value for key, value in profile['env'].items() if key not in flags}, control['env'])
                self.assertEqual({key: profile['env'][key] for key in flags}, flags)
                self.assertEqual({key: value for key, value in profile.items() if key not in ('description', 'env')},
                                 {key: value for key, value in control.items() if key not in ('description', 'env')})
                self.assertTrue(profile['gate_only'])
                self.assertEqual(profile['env']['QWEN_FAST_VERIFY_T1_AUDIT'], '0', 'the timed arms keep the verify audits off')
                self.assertEqual(profile['env']['QWEN_FAST_262K_EVIDENCE_WAIVER'], '1')

    def test_the_flags_the_profiles_name_are_the_ones_the_runtime_reads(self):
        named = {key for flags in ADDED.values() for key in flags}
        self.assertEqual(named, set(FLAGS))
        self.assertEqual(set(ADDED[ARM_AUDIT]), set(FLAGS))

    def test_the_control_is_untouched_and_the_default_stays_the_production_profile(self):
        with open(PROFILES_PATH, encoding='utf-8') as handle:
            data = json.load(handle)
        self.assertEqual(data['default'], 'c2-packed-tp4')
        env = data['profiles'][CONTROL]['env']
        self.assertFalse(set(FLAGS) & set(env))

    def test_no_traffic_profile_carries_a_host_gap_flag(self):
        for name, profile in load_profiles().items():
            if name in ADDED or name in ('c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-audit', 'c2-packed-tp4-8x262k-w1', 'c2-packed-tp4-8x262k-w1-audit', 'c2-packed-tp4-8x262k-w1-audit-nod1', 'c2-packed-tp4-8x262k-w1-lite', 'c2-packed-tp4-8x262k-w1-nod1'):  # tp4/w1's stack arms carry the epochs arm's flags
                continue
            with self.subTest(name=name):
                self.assertFalse(set(FLAGS) & set(profile.get('env', {})))


EXPECTED = {
    'X0-status-rescan-reset': ('status rescan reset', None, 'stop'), 'B0-build': ('build', 'c2-packed-tp4', 'stop'),
    'S0c-control-attach-smoke': ('reset smoke', CONTROL, 'stop'),
    'A1-audited-attach-smoke': ('reset smoke', ARM_AUDIT, 'stop'), 'A2-audited-attach-lite': ('reset smoke', LITE_AUDIT, 'soft'),
    'H1-hang-shapes-hostgap': ('reset smoke', ARM, 'stop'), 'H2-hang-shapes-hostgap': ('reset smoke', ARM, 'stop'),
    'H3-hang-shapes-hostgap': ('reset smoke', ARM, 'stop'), 'H4-hang-shapes-hostgap': ('reset smoke', ARM, 'stop'),
    'H5-hang-shapes-hostgap': ('reset smoke', ARM, 'stop'),
    'T1-timed-A-control': ('reset smoke', CONTROL, 'soft'), 'T2-timed-B-hostgap-2': ('reset smoke', ARM, 'soft'),
    'T3-timed-A-control': ('reset smoke', CONTROL, 'soft'), 'T4-timed-B-hostgap-2': ('reset smoke', ARM, 'soft'),
    'T5-timed-C-hostgap-1': ('reset smoke', LITE, 'soft'), 'T6-timed-A-control': ('reset smoke', CONTROL, 'soft'),
    'C1-churn16-hostgap-2': ('reset gate', ARM, 'soft'),
    'Z-reset': ('status reset', None, 'soft'),
}
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}


def pack_text(name):
    return (FOLDER / (name + '.env')).read_text(encoding='utf-8')


def pack_job(name):
    return job.read_job(job.parse_env(pack_text(name)), sorted(load_profiles()), root=ROOT)


def order_lines():
    return [line.split() for line in (FOLDER / 'ORDER.txt').read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.startswith('#')]


class PackTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates_in_the_asked_order(self):
        lines = order_lines()
        self.assertEqual([line[0] for line in lines], list(EXPECTED))
        self.assertEqual(sorted(path.stem for path in FOLDER.glob('*.env')), sorted(EXPECTED))
        for name, mode, image, minutes in lines:
            self.assertEqual((mode, image), (EXPECTED[name][2], IMAGE), name)
            self.assertTrue(minutes.isdigit() and int(minutes) > 0, name)

    def test_every_template_parses_with_its_actions_profile_and_the_one_image(self):
        for name, (actions, profile, _mode) in EXPECTED.items():
            with self.subTest(name=name):
                result = pack_job(name)
                self.assertEqual((result['actions'], result['cards'], result['tag']), (actions, 'quad', IMAGE))
                if profile:
                    self.assertEqual(result['profile'], profile)

    def test_the_first_quad_job_is_status_rescan_reset_and_no_job_touches_the_agent_or_production(self):
        self.assertEqual(pack_job('X0-status-rescan-reset')['actions'], 'status rescan reset')
        self.assertEqual(order_lines()[0][0], 'X0-status-rescan-reset')
        self.assertEqual(order_lines()[-1][0], 'Z-reset')
        for name in EXPECTED:
            with self.subTest(name=name):
                result = pack_job(name)
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))
                self.assertEqual(result['bake_default_profile'] or '', '')
        self.assertNotIn(IMAGE, job.PROTECTED)
        self.assertFalse(IMAGE.startswith(job.PROTECTED_PREFIXES))

    def test_every_timed_and_attach_smoke_with_the_quad_blocks_runs_concurrent8_steady(self):
        for name, (actions, profile, _mode) in EXPECTED.items():
            if not profile or 'smoke' not in actions:
                continue
            with self.subTest(name=name):
                self.assertIn('concurrent8_steady', pack_job(name)['tests'].replace(' ', ',').split(','))

    def test_the_audited_attach_runs_the_32k_coding_users_and_the_steady_mix(self):
        tests = pack_job('A1-audited-attach-smoke')['tests'].replace(' ', ',').split(',')
        self.assertTrue({'concurrent8_steady', 'concurrent8_code_32k'} <= set(tests))

    def test_five_hang_shape_runs_on_the_audits_off_arm_carry_the_eight_seat_shapes(self):
        names = [name for name in EXPECTED if name.startswith('H')]
        self.assertEqual(len(names), 5)
        self.assertEqual(len({pack_job(name)['tests'] for name in names}), 1)
        tests = pack_job(names[0])['tests'].replace(' ', ',').split(',')
        for shape in ('concurrent8_steady', 'steady_resend', 'replay_concurrent8', 'replay_concurrent4', 'concurrent8_code_equal',
                      'concurrent8_drain'):
            self.assertIn(shape, tests)
        env = load_profiles()[ARM]['env']
        self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))

    def test_the_timing_jobs_alternate_abab_on_the_same_tests_at_32k_and_128k(self):
        timed = [name for name in EXPECTED if name.startswith('T')]
        self.assertEqual([pack_job(name)['profile'] for name in timed[:4]], [CONTROL, ARM, CONTROL, ARM])
        self.assertEqual([pack_job(name)['profile'] for name in timed[4:]], [LITE, CONTROL])
        self.assertEqual(len({pack_job(name)['tests'] for name in timed}), 1)
        tests = pack_job(timed[0])['tests'].replace(' ', ',').split(',')
        self.assertTrue({'concurrent8_steady', 'concurrent8_code_32k', 'concurrent8_code_128k'} <= set(tests))

    def test_the_dependencies_and_the_read_rules(self):
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        self.assertIn('# NEEDS A1 A2 <- S0c', order)
        self.assertIn('# NEEDS H1 H2 H3 H4 H5 <- A1', order)
        self.assertIn('# NEEDS T1 T2 T3 T4 <- A1 H1 H2 H3 H4 H5', order)
        self.assertIn('# NEEDS T5 T6 <- A2 S0c', order)
        self.assertIn('# NEEDS C1 <- A1', order)
        self.assertIn('OWN five consecutive hang-shape completions', order)
        self.assertIn('ZERO [PACKED-PRESTAGE-FULLAUDIT] mismatches', order)
        self.assertIn('five consecutive completions', order)
        self.assertIn('PAIRED', order)

    def test_no_hostname_address_registry_or_digest_and_lf_endings(self):
        for path in FOLDER.iterdir():
            text_found = path.read_text(encoding='utf-8')
            self.assertIsNone(BANNED.search(text_found), path.name)
            self.assertNotIn('\r', text_found, path.name)


class ShippingTests(unittest.TestCase):
    RUNTIME = ('verify_prestage.py', 'packed_verifier.py', 'serving_packed_step.py', 'serving_packed_bridge.py',
               'serving_page_binding.py', 'quad_draft_tp.py', 'dflash_packed_proposal_coordinator.py')

    def test_every_module_the_levers_touch_is_in_both_image_copy_lists(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        docker, context = dockerfile_modules(dockerfile_text()), context_modules()
        for name in self.RUNTIME:
            with self.subTest(module=name):
                self.assertIn(name, docker)
                self.assertIn(name, context)

    def test_the_overlay_manifest_names_them_all_so_the_image_runs_this_commits_copies(self):
        listed = {line.split()[0] for line in (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').splitlines()
                  if line.strip() and not line.startswith('#')}
        for name in self.RUNTIME:
            with self.subTest(module=name):
                self.assertIn('scripts/ci/' + name, listed)

    def test_the_suite_runs_in_the_cpu_workflow(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertRegex(workflow, r'python -B -m unittest [^\n]*\btest_tp4_hostgap\b')

    def test_the_new_and_edited_files_are_lf_and_no_pyc_is_tracked_by_this_change(self):
        for name in self.RUNTIME + ('test_tp4_hostgap.py', 'c2_smoke_check.py', 'qwen_c2_profiles.json'):
            with self.subTest(name=name):
                self.assertNotIn(b'\r\n', (HERE / name).read_bytes())


class ReplayValidatedTests(unittest.TestCase):
    """The real retained block, unedited: skip_next_binding_check drops exactly the check right before the trace."""

    def replays(self, skip):
        block = tgr.unpacked_block(48, 4)
        counted = Mock(side_effect=block.validate_bindings)
        block.validate_bindings = counted
        with patch('gdn_records.restore_prefix'):
            block.commit(2, synchronize=True)
            before = counted.call_count
            operation = Mock(return_value=None)
            if skip:
                with vp.skip_next_binding_check(block):
                    block.replay(operation)
            else:
                block.replay(operation)
        operation.assert_called_once_with()
        self.assertIs(block.validate_bindings, counted, 'the block keeps the attribute it had')
        return counted.call_count - before, block

    def test_skipping_drops_the_pre_trace_check_and_nothing_else(self):
        default, block = self.replays(False)
        skipped, skipped_block = self.replays(True)
        self.assertEqual(default, 2, 'before the trace and, without round fences, after its sync')
        self.assertEqual(skipped, 1, 'the post-sync check stays')
        self.assertEqual((block.replay_epoch, skipped_block.replay_epoch), (1, 1))
        self.assertIsNone(skipped_block.selected_prefix)

    def test_the_block_class_attribute_is_restored_and_a_raise_inside_restores_it_too(self):
        block = tgr.unpacked_block(48, 4)
        self.assertNotIn('validate_bindings', vars(block))
        with vp.skip_next_binding_check(block):
            self.assertIn('validate_bindings', vars(block))
        self.assertNotIn('validate_bindings', vars(block))
        with self.assertRaises(ValueError):
            with vp.skip_next_binding_check(block):
                block.replay(Mock())
        self.assertNotIn('validate_bindings', vars(block))

    def test_skipping_still_refuses_a_replay_with_no_synchronized_commit(self):
        block = tgr.unpacked_block(48, 4)
        with self.assertRaises(ValueError):
            with vp.skip_next_binding_check(block):
                block.replay(Mock())

    def test_a_rebound_native_buffer_still_fails_the_check_that_remains(self):
        block = tgr.unpacked_block(48, 4)
        with patch('gdn_records.restore_prefix'):
            block.commit(2, synchronize=True)
        block.validate_bindings = Mock(side_effect=[ValueError('Native layer binding changed')])
        with self.assertRaises(ValueError):
            with vp.skip_next_binding_check(block):
                block.replay(Mock(return_value=None))


class FlagOffIdentityTests(tvp.PrestageFixture):
    """With every host-gap flag off the block's rounds are the pre-hostgap block's: the same paths, the same numbers, the same lines."""

    def test_the_rounds_paths_and_marker_lines_are_the_same_with_the_flags_set_to_zero(self):
        results = []
        for flags in ({}, {name: '0' for name in FLAGS}):
            for name, value in flags.items():
                os.environ[name] = value
            block = self.open_block()
            self.h1a.clear()
            users = tvp.base_users()
            served = [self.round(block, users)[0]]
            for step in (2, 5):
                nxt = tvp.advanced(users, step)
                self.window(block, nxt)
                served.append(self.round(block, nxt)[0])
                users = nxt
            lines = [re.sub(r'ms=[0-9.]+', 'ms=X', line) for line in self.h1a]
            results.append((served, self.paths(), lines, vp.two_block_mode()))
            block.close()
            self.block = None
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][1], ['full', 'diff', 'diff'])
        self.assertFalse([line for line in results[0][2] if 'HOSTGAP' in line or 'FULLAUDIT' in line or 'SHADOW' in line])


if __name__ == '__main__':
    unittest.main()
