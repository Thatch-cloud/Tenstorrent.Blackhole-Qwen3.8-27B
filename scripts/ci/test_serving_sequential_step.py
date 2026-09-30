"""The sequential step: correct, ordered, and honest about what it costs."""

from types import SimpleNamespace
import unittest

from serving_sequential_step import describe, sequential_packed_step


def entry(request_id, stepped):
    ticket = SimpleNamespace(request_id=request_id)

    def step(name, *, cancelled):
        stepped.append((name, cancelled()))
        return SimpleNamespace(request_id=name, token_ids=[1, 2])

    return dict(request_id=request_id, ticket=ticket,
                request=SimpleNamespace(step=step))


class SequentialStepTests(unittest.TestCase):
    def test_every_request_steps_in_the_schedulers_order(self):
        stepped = []
        outputs = sequential_packed_step([entry('B', stepped), entry('A', stepped)],
                                         cancelled=lambda: False)
        self.assertEqual([name for name, _ in stepped], ['B', 'A'],
                         'entry order is the scheduler order, not creation order')
        self.assertEqual([output.request_id for output in outputs], ['B', 'A'])

    def test_cancellation_reaches_every_request(self):
        stepped = []
        sequential_packed_step([entry('A', stepped), entry('B', stepped)],
                               cancelled=lambda: True)
        self.assertEqual([flag for _, flag in stepped], [True, True])

    def test_a_ticket_for_another_request_is_refused(self):
        stepped = []
        broken = entry('A', stepped)
        broken['ticket'] = SimpleNamespace(request_id='someone-else')
        with self.assertRaises(ValueError):
            sequential_packed_step([broken], cancelled=lambda: False)
        self.assertEqual(stepped, [], 'refused before touching the device')

    def test_an_output_for_the_wrong_request_is_refused(self):
        stepped = []
        wrong = entry('A', stepped)
        wrong['request'] = SimpleNamespace(
            step=lambda name, *, cancelled: SimpleNamespace(request_id='B', token_ids=[1]))
        with self.assertRaises(ValueError):
            sequential_packed_step([wrong], cancelled=lambda: False)

    def test_an_empty_block_is_refused(self):
        with self.assertRaises(ValueError):
            sequential_packed_step([], cancelled=lambda: False)

    def test_it_reports_that_it_is_not_the_batched_verifier(self):
        """A benchmark that reads this must not mistake it for the goal."""
        cost = describe()
        self.assertIs(cost['batched'], False)
        self.assertEqual(cost['weight_passes_per_round'], 'one per user')


if __name__ == '__main__':
    unittest.main()
