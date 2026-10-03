"""The four-card quad draft (QWEN_FAST_QUAD_DRAFT) as v172 D1 saw it, and as the all-levers profiles now serve it.

v172 (profile c2-packed-tp4-gate-quad): the quad marker appeared 0 times and no round was served by the quad. The cause was
upstream of the quad: users 0 and 1 came out of their prefill dead (an instant end-of-text; the prefill programs and GDN scratch
were first created after the packed captures and replays overwrote them - fixed by serving_runtime.prefill_warm_before_traces,
see test_tp4_prefill_scratch), so the groups the coordinator saw were never exactly the quad's two pairs. The coordinator is
right to wait: this pins that it never forms a quad from a broken or partial group, never disables itself for it, never writes
a fallback line, and forms one as soon as the four users are packable. The smoke check's reading of both logs is pinned too.
At 8 seats (groups beyond the first four slots) the quad stays off until a per-block quad exists; that limit is documented here.
"""
import unittest
from unittest.mock import Mock

import c2_smoke_check as check
import quad_draft
import test_quad_draft as base
from test_dflash_packed_proposal_coordinator import make_bridge, make_device

FLAG = base.FLAG
LIVE_LAYERS = [object()] * 5
PREDECESSORS, SUCCESSORS = object(), object()


def bridges_with(operations, mesh, rows_by_slot):
    out = []
    for slot, rows in rows_by_slot:
        device = make_device(operations, mesh, slot=slot, history_rows=rows)
        device.layers, device.predecessors, device.successors = LIVE_LAYERS, PREDECESSORS, SUCCESSORS
        device.kv_history.active = list(LIVE_LAYERS)
        device.block_rows, device.fused_convolution = 16, True
        device.prepare_device = Mock(return_value=True)
        out.append(make_bridge('r%d' % slot, device, seed=100 + slot))
    return out


class CoordinatorWaitsForTheFourthPackableUserTests(unittest.TestCase):
    setUp = base.CoordinatorTests.setUp
    coordinator = base.CoordinatorTests.coordinator
    prepare = base.CoordinatorTests.prepare

    def quiet(self, lines):
        return not [line for line in lines if 'fallback' in line or 'disabled' in line]

    def test_a_pair_still_ramping_never_forms_a_quad_and_never_disables_it(self):
        coordinator = self.coordinator()
        bridges = bridges_with(self.operations, self.mesh, [(0, 1024), (1, 1024), (2, 2048), (3, 2048)])
        for _ in range(3):
            _, lines = self.prepare(coordinator, bridges)
            self.assertTrue(self.quiet(lines), lines)
        self.assertEqual(base.FakeQuadTrace.instances, [])
        self.assertEqual(coordinator.quad_rounds, 0)
        self.assertFalse(coordinator.quad_disabled)
        self.assertEqual(coordinator.quad_failures, 0)

    def test_the_quad_forms_the_round_after_the_last_ramping_user_packs(self):
        coordinator = self.coordinator()
        ramp = bridges_with(self.operations, self.mesh, [(0, 2048), (1, 2048), (2, 2048), (3, 1024)])
        self.prepare(coordinator, ramp)
        self.assertEqual(base.FakeQuadTrace.instances, [])
        ramp[3].request.runtime.drafter.history_rows = 2048
        _, lines = self.prepare(coordinator, ramp)
        self.assertEqual(len(base.FakeQuadTrace.instances), 1)
        self.assertEqual(coordinator.quad_rounds, 1)
        self.assertTrue(self.quiet(lines), lines)

    def test_a_dead_slot_in_the_group_is_no_quad_either(self):
        # v172's users 0 and 1: slots 2 and 3 alone are one pair, never the quad's four.
        coordinator = self.coordinator()
        bridges = bridges_with(self.operations, self.mesh, [(2, 2048), (3, 2048)])
        _, lines = self.prepare(coordinator, bridges)
        self.assertEqual(base.FakeQuadTrace.instances, [])
        self.assertTrue(self.quiet(lines), lines)

    def test_the_quad_is_the_first_four_slots_only(self):
        # Eight seats (origin/tp4/seats8) need a per-block quad before the coordinator can form one beyond these slots.
        self.assertEqual(quad_draft.SLOTS, (0, 1, 2, 3))
        self.assertEqual(quad_draft.PAIRS, ((0, 1), (2, 3)))


def log(*, markers, rounds, audits_equal=96):
    lines = ['[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=64/16 rows=64 sdpa=fold conv=110'] * markers
    lines += ['[PACKED-SELECT] round=%d pairs=[[0, 1, 2, 3]] users=4 calls=1 collect_ms=1.0 select_ms=2.0' % n
              for n in range(1, rounds + 1)]
    lines += ['[QUAD-DRAFT] round=%d built=%d ms=3.0' % (n, int(n == 1)) for n in range(1, rounds + 1)]
    lines += ['[DRAFT-SINGLES-AUDIT] group=[0, 1, 2, 3] equal=1 checks=%d' % audits_equal] * max(rounds, 5)
    return chr(10).join(lines)


class SmokeCheckReadsBothLogsTests(unittest.TestCase):
    ENV = {check.QUAD_FLAG: '1'}

    def problems(self, text):
        return check.draft_problems(check.draft_facts(text), self.ENV, True)

    def test_the_v172_signature_fails_on_the_marker_and_on_no_quad_round(self):
        problems = self.problems(log(markers=0, rounds=0))
        self.assertEqual(len(problems), 2, problems)
        self.assertIn('appears 0 times', problems[0])
        self.assertIn('no round was served by the quad', problems[1])

    def test_the_engaged_logs_of_n1c_and_n4b_pass(self):
        for rounds in (288, 466):
            self.assertEqual(self.problems(log(markers=1, rounds=rounds)), [])

    def test_a_second_engage_a_fallback_or_a_disable_fails(self):
        self.assertTrue(self.problems(log(markers=2, rounds=5)))
        self.assertTrue(self.problems(log(markers=1, rounds=5) + chr(10) + '[QUAD-DRAFT] fallback round=9 reason=RuntimeError:x'))
        self.assertTrue(self.problems(log(markers=1, rounds=5) + chr(10) + '[PINDIAG] quad draft disabled round=9 failures=2 reason=x'))


if __name__ == '__main__':
    unittest.main()
