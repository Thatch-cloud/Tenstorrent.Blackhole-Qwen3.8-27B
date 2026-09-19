"""The packed scheduler contract, including the ordering trap.

Probe 35436807668 measured TTScheduler scheduling both decodes in one step as
`cached=['B','A']` - both requests, and NOT in creation order. The pack's row
segments are matched to the scheduler's rows, so assembling in registry order
would hand each user the other's proposals with nothing to catch it.
"""

from types import SimpleNamespace
import unittest

from serving_vllm_packed import admit_packed_scheduler_output, ordered_tickets


def request(request_id, position, tokens):
    ticket = SimpleNamespace(request_id=request_id, position=position, tokens=list(tokens))
    session = SimpleNamespace(pending=ticket, phase='pending', request_id=request_id, position=position)
    return SimpleNamespace(session=session, engine=SimpleNamespace(phase='idle'),
                           closed=False, cancelled=False, busy=False)


def step(order, positions, counts, proposals, **overrides):
    cached = SimpleNamespace(req_ids=list(order), num_computed_tokens=list(positions),
                             resumed_req_ids=set())
    scheduled = SimpleNamespace(scheduled_cached_reqs=cached, scheduled_new_reqs=[],
        finished_req_ids=set(), num_scheduled_tokens=dict(counts),
        total_num_scheduled_tokens=sum(counts.values()),
        scheduled_spec_decode_tokens=dict(proposals))
    for name, value in overrides.items():
        setattr(scheduled, name, value)
    return scheduled


class PackedAdmissionTests(unittest.TestCase):
    def pair(self):
        return [request('A', 100, list(range(1, 17))), request('B', 5000, list(range(50, 66)))]

    def scheduled(self, order=('B', 'A')):
        positions = {'A': 100, 'B': 5000}
        tokens = {'A': list(range(1, 17)), 'B': list(range(50, 66))}
        return step(order, [positions[name] for name in order],
                    {name: 16 for name in order},
                    {name: tokens[name][1:] for name in order})

    def test_both_requests_are_admitted_in_the_schedulers_order(self):
        entries = admit_packed_scheduler_output(self.pair(), self.scheduled(('B', 'A')))
        self.assertEqual([entry['request_id'] for entry in entries], ['B', 'A'],
                         'the pack follows the scheduler, not the registry')
        self.assertEqual([entry['ticket'].position for entry in entries], [5000, 100])

    def test_creation_order_does_not_leak_into_the_pack(self):
        forward = admit_packed_scheduler_output(self.pair(), self.scheduled(('A', 'B')))
        reverse = admit_packed_scheduler_output(self.pair(), self.scheduled(('B', 'A')))
        self.assertEqual([entry['request_id'] for entry in forward], ['A', 'B'])
        self.assertEqual([entry['request_id'] for entry in reverse], ['B', 'A'])

    def test_a_new_request_in_the_step_is_refused(self):
        scheduled = self.scheduled()
        scheduled.scheduled_new_reqs = [SimpleNamespace(req_id='C')]
        with self.assertRaises(ValueError):
            admit_packed_scheduler_output(self.pair(), scheduled)

    def test_a_frontier_that_does_not_match_a_ticket_is_refused(self):
        scheduled = self.scheduled(('A', 'B'))
        scheduled.scheduled_cached_reqs.num_computed_tokens = [101, 5000]
        with self.assertRaises(ValueError):
            admit_packed_scheduler_output(self.pair(), scheduled)

    def test_a_scheduled_request_we_do_not_hold_is_refused(self):
        scheduled = self.scheduled(('A', 'B'))
        with self.assertRaises(ValueError):
            admit_packed_scheduler_output([self.pair()[0]], scheduled)

    def test_a_held_request_the_scheduler_omitted_is_refused(self):
        """Half a pack is not a pack: the block's rows would not be covered."""
        scheduled = step(('A',), [100], {'A': 16}, {'A': list(range(2, 17))})
        with self.assertRaises(ValueError):
            admit_packed_scheduler_output(self.pair(), scheduled)

    def test_token_counts_must_cover_every_request_block(self):
        scheduled = self.scheduled(('A', 'B'))
        scheduled.num_scheduled_tokens = {'A': 16, 'B': 15}
        with self.assertRaises(ValueError):
            admit_packed_scheduler_output(self.pair(), scheduled)
        scheduled = self.scheduled(('A', 'B'))
        scheduled.total_num_scheduled_tokens = 31
        with self.assertRaises(ValueError):
            admit_packed_scheduler_output(self.pair(), scheduled)

    def test_proposals_must_match_every_prepared_ticket(self):
        scheduled = self.scheduled(('A', 'B'))
        scheduled.scheduled_spec_decode_tokens['A'] = [99] * 15
        with self.assertRaises(ValueError):
            admit_packed_scheduler_output(self.pair(), scheduled)

    def test_a_resumed_request_is_refused(self):
        scheduled = self.scheduled(('A', 'B'))
        scheduled.scheduled_cached_reqs.resumed_req_ids = {'A'}
        with self.assertRaises(ValueError):
            admit_packed_scheduler_output(self.pair(), scheduled)

    def test_duplicate_ids_are_refused(self):
        pair = self.pair()
        pair[1] = request('A', 5000, list(range(50, 66)))
        with self.assertRaises(ValueError):
            ordered_tickets(pair, self.scheduled(('A', 'B')))

    def test_a_single_request_still_admits(self):
        """One user is a pack of one; the contract must not require two."""
        entries = admit_packed_scheduler_output([request('A', 100, list(range(1, 17)))],
                                                step(('A',), [100], {'A': 16},
                                                     {'A': list(range(2, 17))}))
        self.assertEqual([entry['request_id'] for entry in entries], ['A'])


if __name__ == '__main__':
    unittest.main()
