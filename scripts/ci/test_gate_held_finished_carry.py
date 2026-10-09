"""A gate-held prefill pass must not swallow the finished ids (run 37868643564, HLF v589, serve-12).

A prompt aborted mid-prefill while decoders run leaves the prefill gate held by a request vLLM no longer holds. Every pass
is then a gate-held pass (hide, allowed 0) that schedules nothing: the plugin discards it and runs a decode-only pass, and
schedule() had already handed finished_req_ids to the discarded pass. The worker never saw the aborted id (the gate stayed
held for good) nor, later, a decoder's budget finish, and the next packed step found a bridge vLLM no longer schedules:
ValueError 'Live prepared request ticket required before scheduler admission' (serving_vllm_contract.prepared_ticket).
The packed-ticket contract is modelled here by Worker.execute."""
import unittest
from unittest.mock import Mock

import serving_prefill_admission as admission
import test_serving_prefill_admission as base


class TicketWorker(object):
    """The worker side: one live bridge per decoding request, detached only when a step names the request finished
    (serving_lifecycle._execute); a step must carry exactly the live bridges (serving_vllm_packed.ordered_tickets)."""

    def __init__(self, live):
        self.live = set(live)
        self.released = set()

    def execute(self, step):
        finished = set(step.finished_req_ids)
        self.live -= finished
        self.released |= finished
        cached = set(step.scheduled_cached_reqs.req_ids)
        if cached != self.live:
            raise ValueError('Live prepared request ticket required before scheduler admission: scheduled=%r live=%r'
                             % (sorted(cached), sorted(self.live)))
        return step


class GateHeldFinishedCarryTests(base.GateFreeCase):
    def scheduler(self, log):
        cls = base.plugin_class(base.FinishingVllmScheduler)
        admission.install(base.configured(cls), log=log)
        scheduler = cls()
        for request_id in 'ABC':
            scheduler.add_request(base.FakeRequest(request_id, 100))
            scheduler.schedule()
        self.assertEqual(base.names(scheduler.running), ['A', 'B', 'C'])
        return scheduler

    def test_the_abort_of_the_gate_holder_reaches_the_worker(self):
        scheduler = self.scheduler(Mock())
        worker = TicketWorker('ABC')
        scheduler.add_request(base.FakeRequest('W', 100))     # a new prompt waits behind the gate
        with base.gate('G'):
            scheduler.finished_req_ids.add('G')                # G, a long prefill, was aborted; it still holds the gate
            step = worker.execute(scheduler.schedule())
        self.assertEqual(step.finished_req_ids, {'G'}, 'the decode-only pass names it, so the worker releases the gate')

    def test_a_decoder_finishing_behind_the_gate_is_named_by_the_decode_step(self):
        """The crash: the gate is still held (the abort was lost), C hits its budget and finishes, W waits."""
        scheduler = self.scheduler(Mock())
        worker = TicketWorker('ABC')
        scheduler.add_request(base.FakeRequest('W', 100))
        with base.gate('G'):
            scheduler.finish('C')
            step = worker.execute(scheduler.schedule())    # pre-fix: ValueError Live prepared request ticket required
        self.assertEqual(base.cached_ids(step), ['A', 'B'])
        self.assertEqual(step.finished_req_ids, {'C'})
        self.assertEqual(worker.live, {'A', 'B'})

    def test_with_no_decode_left_the_returned_pass_names_the_finish_once(self):
        log = Mock()
        scheduler = self.scheduler(log)
        for request_id in 'ABC':
            scheduler.finish(request_id)
        scheduler.add_request(base.FakeRequest('W', 100))
        with base.gate('G'):
            step = scheduler.schedule()
        self.assertEqual(step.finished_req_ids, {'A', 'B', 'C'})
        self.assertEqual(base.logged(log, getattr(admission, 'GATE_CARRIED_LINE', 'unset')), [])


if __name__ == '__main__':
    unittest.main()
