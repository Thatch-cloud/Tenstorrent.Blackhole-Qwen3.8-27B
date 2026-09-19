"""One scheduled step, several packed decode requests, one device step.

The two things that must happen exactly ONCE are the point of the whole exercise:
`runner._update_states`, which consumes the whole SchedulerOutput rather than one
request's part of it, and the device step, because one pass over the weights
serving every user is what packing buys. Everything else is per user.

Probe 35436807668 also measured the scheduler presenting the pair as
`cached=['B','A']`, so the outputs must follow the scheduler's order rather than
the order the bridges happen to sit in.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from serving_packed_bridge import execute_packed_decode


def bridge(runner, request_id, position, tokens):
    ticket = SimpleNamespace(request_id=request_id, position=position, tokens=list(tokens))
    session = SimpleNamespace(pending=ticket, phase='pending', request_id=request_id, position=position)
    request = SimpleNamespace(session=session, engine=SimpleNamespace(phase='idle'),
                              closed=False, cancelled=False, busy=False)
    state = SimpleNamespace(block_ids=[[7]])
    return SimpleNamespace(runner=runner, request=request, state=state, failed=False,
                           validate_storage=None, page_binding=SimpleNamespace(refresh=Mock()))


def scheduled_step(order, positions, tokens):
    cached = SimpleNamespace(req_ids=list(order),
                             num_computed_tokens=[positions[name] for name in order],
                             resumed_req_ids=set())
    return SimpleNamespace(scheduled_cached_reqs=cached, scheduled_new_reqs=[], finished_req_ids=set(),
        num_scheduled_tokens={name: len(tokens[name]) for name in order},
        total_num_scheduled_tokens=sum(len(tokens[name]) for name in order),
        scheduled_spec_decode_tokens={name: tokens[name][1:] for name in order})


class PackedBridgeTests(unittest.TestCase):
    def fixture(self, order=('B', 'A')):
        runner = SimpleNamespace(_update_states=Mock())
        tokens = {'A': list(range(1, 17)), 'B': list(range(50, 66))}
        positions = {'A': 100, 'B': 5000}
        bridges = {name: bridge(runner, name, positions[name], tokens[name]) for name in ('A', 'B')}
        return runner, bridges, scheduled_step(order, positions, tokens), tokens

    def run_packed(self, runner, bridges, scheduled, outputs=None, **kwargs):
        seen = {}

        def packed_step(entries, *, cancelled):
            seen['entries'] = entries
            seen['cancelled'] = cancelled
            return outputs if outputs is not None else [
                SimpleNamespace(request_id=entry['request_id'], token_ids=[1, 2]) for entry in entries]

        with patch('serving_vllm_state.apply_committed_output') as applied, \
                patch('serving_vllm_state.validate_runner_reservation'), \
                patch('serving_packed_bridge.packed_model_runner_output',
                      side_effect=lambda values: [value.request_id for value in values]):
            result = execute_packed_decode(bridges, scheduled, cancelled=lambda: False,
                                           packed_step=kwargs.get('packed_step', packed_step))
        return result, seen, applied

    def test_the_device_step_runs_once_and_update_states_runs_once(self):
        runner, bridges, scheduled, _ = self.fixture()
        calls = []

        def packed_step(entries, *, cancelled):
            calls.append(len(entries))
            return [SimpleNamespace(request_id=entry['request_id'], token_ids=[1]) for entry in entries]

        self.run_packed(runner, bridges, scheduled, packed_step=packed_step)
        self.assertEqual(calls, [2], 'one device step covering both requests')
        self.assertEqual(runner._update_states.call_count, 1,
                         'update_states consumes the whole SchedulerOutput, so once')

    def test_outputs_follow_the_scheduler_order_not_the_bridge_order(self):
        runner, bridges, scheduled, _ = self.fixture(order=('B', 'A'))
        result, seen, _ = self.run_packed(runner, bridges, scheduled)
        self.assertEqual([entry['request_id'] for entry in seen['entries']], ['B', 'A'])
        self.assertEqual(result, ['B', 'A'])

    def test_every_user_binds_its_own_pages_at_its_own_frontier(self):
        runner, bridges, scheduled, _ = self.fixture(order=('B', 'A'))
        self.run_packed(runner, bridges, scheduled)
        for name, position in (('A', 100), ('B', 5000)):
            bridges[name].page_binding.refresh.assert_called_once_with([7], position=position, rows=16)

    def test_a_step_output_out_of_order_is_refused_and_poisons_every_bridge(self):
        runner, bridges, scheduled, _ = self.fixture(order=('B', 'A'))
        wrong = [SimpleNamespace(request_id='A', token_ids=[1]),
                 SimpleNamespace(request_id='B', token_ids=[1])]
        with self.assertRaises(ValueError):
            self.run_packed(runner, bridges, scheduled, outputs=wrong)
        self.assertTrue(all(value.failed for value in bridges.values()))

    def test_a_short_output_list_is_refused(self):
        runner, bridges, scheduled, _ = self.fixture()
        with self.assertRaises(ValueError):
            self.run_packed(runner, bridges, scheduled,
                            outputs=[SimpleNamespace(request_id='B', token_ids=[1])])

    def test_bridges_on_different_runners_are_refused(self):
        runner, bridges, scheduled, _ = self.fixture()
        bridges['A'].runner = SimpleNamespace(_update_states=Mock())
        with self.assertRaises(ValueError):
            self.run_packed(runner, bridges, scheduled)

    def test_a_failed_bridge_stops_the_block(self):
        runner, bridges, scheduled, _ = self.fixture()
        bridges['A'].failed = True
        with self.assertRaises(ValueError):
            self.run_packed(runner, bridges, scheduled)
        runner._update_states.assert_not_called()

    def test_a_missing_device_step_is_refused_before_any_binding(self):
        runner, bridges, scheduled, _ = self.fixture()
        with self.assertRaises(ValueError):
            execute_packed_decode(bridges, scheduled, cancelled=lambda: False, packed_step=None)
        runner._update_states.assert_not_called()


if __name__ == '__main__':
    unittest.main()
