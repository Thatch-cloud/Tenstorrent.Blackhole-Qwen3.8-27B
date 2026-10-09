"""Engine reuse: the parked admission terms on REAL vLLM 0.25.1 objects (qwen-fast-vllm-cpu.yml; skipped where vLLM is not installed).

test_levern_prefix_scheduler_vllm builds the merged scheduler (the plugin's TTScheduler wrapped by serving_prefill_admission.install, the prefix graft on the
instance, vLLM's own Scheduler, KVCacheManager, BlockPool and Request underneath). This module runs that scheduler with the DRAM predicate the worker registers
under engine reuse (serving_prefill_admission.dram_predicate with the parked set's arrival terms) and proves, on the real scheduler:
  P1  the same reading holds an arrival under today's terms and admits it under the parked terms (the v70 point), and a parked arrival whose released
      single's rebuild does not fit is held, then admitted when the release ladder's credit arrives, with the hold's age logged;
  P2  the terms are asked at every step: switching the parked set's answer (a kill switch, an unpark, a re-park) changes the very next admission, and a
      terms reader that raises admits (the scheduler never hard-depends on the worker);
  P3  a hold never starves the decoder: while an arrival is held every step is a decode step, nothing is preempted, no block is held past its reservation
      and every block is free again at the end;
  P4  with no parked terms the predicate, the hold lines and the admissions are today's, step for step.
Run with VLLM_USE_V2_MODEL_RUNNER=0 (as the other installed-vLLM suites)."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import levern_policy  # noqa: E402
import serving_prefill_admission as admission  # noqa: E402
import serving_request_quarantine as quarantine  # noqa: E402
from test_levern_prefix_scheduler_vllm import DECODER_ANSWER, MergedDrive, MergedEnv  # noqa: E402
from test_qwen_prefix_scheduler_vllm import VLLM_ERROR, tokens  # noqa: E402
from test_qwen_prefix_scheduler_vllm import setUpModule, tearDownModule  # noqa: E402,F401  (the module-level fixtures stage the graft package)

if VLLM_ERROR is None:
    from vllm.v1.request import RequestStatus

MB = 10 ** 6
RESERVE = 256 * 1024 * 1024
V70_FREE, V70_LARGEST = 1.514e9, 1320.7 * MB     # three decoders and a pair: today's terms hold a long arrival, the parked ones admit it
LOW_FREE, LOW_LARGEST = 1.06e9, 913.2 * MB       # the worst corner's low reading: S decides
AGE = '[PINDIAG] dram hold age request='
HOLD = '[PINDIAG] dram hold '


class Reading(object):
    """The pool the predicate reads and the parked terms it asks, both changeable between steps."""

    def __init__(self, free, largest):
        self.free, self.largest, self.terms, self.raises = free, largest, None, False

    def pool(self):
        return SimpleNamespace(dram_statistics=lambda: [dict(free=int(self.free), largest_free=int(self.largest))])

    def parked(self):
        if self.raises:
            raise RuntimeError('the parked set is closing')
        return self.terms

    def predicate(self):
        return admission.dram_predicate(self.pool(), RESERVE, parked=self.parked)


@unittest.skipIf(VLLM_ERROR is not None, 'vLLM is not importable here (%s)' % VLLM_ERROR)
class ParkedAdmissionOnRealVllmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MergedEnv()

    def setUp(self):
        self.env.logs[:] = []
        self.addCleanup(levern_policy.reset_state)
        shared = quarantine.holder()
        saved = shared.installed, dict(shared.pending), shared.live
        self.addCleanup(lambda: (setattr(shared, 'installed', saved[0]), setattr(shared, 'pending', saved[1]), setattr(shared, 'live', saved[2])))
        patcher = mock.patch.dict(os.environ, {'QWEN_FAST_TP': '4'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, reading):
        scheduler, state, lines, ledger = self.env.merged()
        drive = MergedDrive(self.env, scheduler, state, 'parked', ledger)
        self.addCleanup(admission.register_dram_predicate(reading.predicate()))
        drive.add('d0', tokens(100, 'd0'), DECODER_ANSWER, salt=None)
        drive.run_until(lambda: 'd0' in drive.model.final)
        for _ in range(3):
            drive.step()
        return scheduler, drive, lines

    def held_for(self, drive, scheduler, request_id, steps=6):
        """Step `steps` times: the arrival must still be waiting and every step a decode step."""
        before = len(drive.records)
        for _ in range(steps):
            drive.step()
        self.assertEqual([kind for kind, _ in drive.records[before:]], ['decode'] * steps)
        self.assertIn(request_id, [request.request_id for request in scheduler.waiting])
        self.assertIsNone(drive.model.written.get(request_id))

    def test_the_same_reading_holds_under_todays_terms_and_admits_under_the_parked_ones(self):
        reading = Reading(V70_FREE, V70_LARGEST)
        scheduler, drive, lines = self.build(reading)
        drive.add('arrival', tokens(9000, 'arrival'), 1)
        self.held_for(drive, scheduler, 'arrival')
        self.assertTrue([line for line in lines if line.startswith(HOLD + 'prompt=')], lines)
        reading.terms = dict(rebind=admission.PARKED_REBIND_BYTES, single=0, credit=0)
        drive.run_until(drive.finished('arrival'))
        self.assertEqual(drive.preempted, set())
        self.assertTrue([line for line in lines if line.startswith(HOLD + 'released')], lines)
        self.assertFalse([line for line in lines if line.startswith(AGE)], 'a hold that began on today\'s terms logs no parked age')

    def test_a_released_singles_rebuild_that_does_not_fit_holds_until_the_ladders_credit_arrives_and_the_age_is_logged(self):
        reading = Reading(LOW_FREE, LOW_LARGEST)
        reading.terms = dict(rebind=admission.PARKED_REBIND_BYTES, single=admission.measured_single_capture_bytes(), credit=0)
        scheduler, drive, lines = self.build(reading)
        drive.add('arrival', tokens(9000, 'arrival'), 1)
        self.held_for(drive, scheduler, 'arrival')
        reading.terms = dict(reading.terms, credit=300 * MB)
        drive.run_until(drive.finished('arrival'))
        age = [line for line in lines if line.startswith(AGE)]
        self.assertEqual(len(age), 1, lines)
        self.assertIn('arrival', age[0])
        self.assertEqual(drive.preempted, set())
        self.assertLessEqual(drive.over, 0)

    def test_the_terms_are_asked_every_step_a_kill_switch_changes_the_next_admission_and_a_raising_reader_admits(self):
        reading = Reading(V70_FREE, V70_LARGEST)
        reading.terms = dict(rebind=admission.PARKED_REBIND_BYTES, single=0, credit=0)
        scheduler, drive, lines = self.build(reading)
        drive.add('first', tokens(9000, 'first'), 1)
        drive.run_until(drive.finished('first'))
        reading.terms = None                    # the kill switch: every slot is today's per-request build again
        drive.add('second', tokens(9000, 'second'), 1)
        self.held_for(drive, scheduler, 'second')
        reading.raises = True                   # a terms reader that fails: the predicate raises, the scheduler admits
        drive.run_until(drive.finished('second'))
        self.assertEqual(drive.preempted, set())

    def test_a_hold_never_starves_the_decoder_and_every_block_is_free_again_at_the_end(self):
        import serving_kv_reservation as kv

        reading = Reading(V70_FREE, V70_LARGEST)
        scheduler, drive, lines = self.build(reading)
        drive.add('arrival', tokens(9000, 'arrival'), 1)
        self.held_for(drive, scheduler, 'arrival', steps=10)
        self.assertGreater(scheduler.requests['d0'].num_output_tokens, 0)
        reading.terms = dict(rebind=admission.PARKED_REBIND_BYTES, single=0, credit=0)
        drive.run_until(drive.finished('arrival'))
        scheduler.finish_requests('d0', RequestStatus.FINISHED_ABORTED)
        drive.step()
        self.assertEqual(scheduler.get_num_unfinished_requests(), 0)
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), kv.pool_blocks(scheduler))
        self.assertEqual(drive.unwritten, [])

    def test_without_parked_terms_the_hold_lines_and_the_admissions_are_todays(self):
        reading = Reading(V70_FREE, V70_LARGEST)
        scheduler, drive, lines = self.build(reading)
        drive.add('arrival', tokens(9000, 'arrival'), 1)
        self.held_for(drive, scheduler, 'arrival')
        reading.free, reading.largest = 3e9, 2000 * MB
        drive.run_until(drive.finished('arrival'))
        self.assertFalse([line for line in lines if line.startswith(AGE)])


if __name__ == '__main__':
    unittest.main()
