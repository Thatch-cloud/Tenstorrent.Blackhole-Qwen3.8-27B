"""QWEN_FAST_LOOKUP_DRAFT beside engine reuse (tp4/stream-1): a REBOUND request attaches the prompt lookup like a built one.

serving_request_factory.attach_lookup runs after rebind_parked as it runs after a cold build, on the real factory, the real parked set, the real
engines and devices of the census world (test_parked_tp4_census); only the device kernels are fake. What is held:

  - a request rebound onto a parked slot carries a lookup (engaged line with the policy, the request, the history length and the index build time),
    the lookup's rows replace DFlash2's in the ticket the verify stages, the round lines are logged, and the close parks the slot as before;
  - flag off (unset, 'off', '0'): no lookup, no import of prompt_lookup, no lookup line, and the run is event for event the same as with the flag
    never heard of: the same tokens, epochs and log lines;
  - a built request (an unparked slot, or no parked set) attaches it as before;
  - a malformed policy raised after the rebind returns the slot to its parked state (no leak) and the capture is closed exactly once;
  - the output is the same with the lookup on and off (the lookup only changes how many tokens a round commits).

CPU only, on a fake four-chip world; the policy side of the pair (the boot's check) is test_parked_tp4_profiles, the tau-lab side test_drafter_arms.
"""

import contextlib
import io
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prompt_lookup  # noqa: E402
import serving_parked_engines as parked  # noqa: E402
from test_parked_tp4_census import CensusCapture, World  # noqa: E402
from test_parked_tp4_set import make_set  # noqa: E402

PARKED = {'QWEN_FAST_PARKED_ENGINES': '1'}
FLAG = prompt_lookup.LOOKUP_FLAG
PERIOD = (3, 4, 5, 6, 7, 8, 9, 10)          # the prompt repeats these; the seed (7) follows a 6, so the last two tokens occurred earlier
LENGTH = 300


def prompt_ids(length=LENGTH):
    """300 tokens of a period-8 pattern ending in 6: with the census seed 7 the history ends ... 5, 6, 7 and that key occurred before."""
    ids = [PERIOD[index % len(PERIOD)] for index in range(length)]
    assert ids[-1] == 6
    return ids


def admit(world, engines, request_id, *, budget=24, first_block=1):
    """world.admit (the real from_prefill beside the packed blocks) with the periodic prompt in place of the census's ones."""
    from serving_request_factory import from_prefill

    blocks = list(range(first_block, first_block + -(-(LENGTH + budget) // 64) + 1))
    state = world.state(request_id, LENGTH, budget, blocks=blocks)
    state.prompt_token_ids = prompt_ids()
    capture = world.last_capture = CensusCapture(world.ops, world.mesh, LENGTH)
    return from_prefill(world.ops, world.model, world.sampler, world.pages(state), world.helpers, state=state, capture=capture,
                        fixtures=world.fixtures, eos_ids=(99,), collectives=world.collectives, buffer_pool=world.pool,
                        shared_weights=world.weights, capture_rows=4, **({} if engines is None else dict(parked=engines)))


def drive(request):
    """Steps until the session finishes; (the emitted tokens, the epochs after each step)."""
    request_id = request.session.request_id
    epochs = []
    while not request.session.finished:
        request.step(request_id, cancelled=lambda: False)
        epochs.append(request.session.epoch)
    return list(request.session.emitted), epochs


def lookup_lines(lines):
    return [line for line in lines if '[LOOKUP' in line]


def shape(lines):
    """The log lines with every number and object address replaced: a timing or an address differs run to run, the line does not."""
    return [re.sub(r'[0-9][0-9.]*', '#', re.sub(r'@[0-9a-f]+', '@', line)) for line in lines]


def environment(value):
    """The lookup flag as given (None: unset), the rest of the process environment kept."""
    kept = {name: text for name, text in os.environ.items() if name != FLAG}
    if value is not None:
        kept[FLAG] = value
    return patch.dict(os.environ, kept, clear=True)


class RunOutcome:
    def __init__(self, tokens, epochs, lines, stdout, request):
        self.tokens, self.epochs, self.lines, self.stdout, self.request = tokens, epochs, lines, stdout, request


def run_parked(value, *, budget=24):
    """One request rebound onto a parked slot with the lookup flag at `value`, driven to its end and closed."""
    with World(environment=PARKED) as world:
        engines = make_set(world)
        engines.build()
        entry = engines.slots[0]
        with environment(value):
            request = admit(world, engines, 'request', budget=budget)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                tokens, epochs = drive(request)
                request.close('request')
        assert request.parked_slot is entry
        assert (entry.state, entry.parks, engines.unparks) == ('parked', 1, 0), (entry.state, entry.parks, engines.unparks)
        assert world.ops.violations == [], world.ops.violations
        return RunOutcome(tokens, epochs, list(world.lines), out.getvalue(), request)


class LookupOnReuseTests(unittest.TestCase):
    def setUp(self):
        # the round lines of a request the test does not read are printed by the lookup: keep them out of the runner's output
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def test_a_rebound_request_carries_the_lookup_and_its_rows_replace_dflash2s(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = engines.slots[0]
            captures = world.ops.captures
            with environment('n3m12'):
                request = admit(world, engines, 'request')
                self.assertEqual(world.ops.captures, captures, 'nothing captured: this is a rebind')
                self.assertIs(request.parked_slot, entry)
                self.assertIsNotNone(request.lookup)
                self.assertEqual(request.lookup.policy, prompt_lookup.LookupPolicy(3, 12))
                engaged = [line for line in world.lines if line.startswith(prompt_lookup.ENGAGED)]
                self.assertEqual(len(engaged), 1, world.lines)
                self.assertRegex(engaged[0], r'^\[LOOKUP-DRAFT\] engaged policy=n3m12 request=request history=%d build_ms=[0-9.]+$' % (LENGTH + 1))
                ticket = request.prepare('request')
                # the history ends 5, 6, 7 and the pattern repeats with period 8: the lookup proposes the pattern's continuation, whatever
                # DFlash2 (the census draft: zeros) proposed, and the match runs back to the cap
                self.assertEqual(ticket.source, 'lookup')
                self.assertEqual(ticket.match_length, prompt_lookup.MAX_BACK)
                self.assertEqual(ticket.tokens[0], 7)
                width = len(ticket.tokens) - 1                    # the census engines capture four rows: three proposals
                self.assertEqual(list(ticket.tokens[1:]), [8, 9, 10, 3, 4, 5, 6, 7][:width])
                self.assertEqual(width, 3)
                self.assertIs(request.session.pending, ticket)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    drive(request)
                    request.close('request')
            rounds = [line for line in out.getvalue().splitlines() if line.startswith(prompt_lookup.ROUND_MARKER)]
            self.assertGreaterEqual(len(rounds), 1)
            self.assertIn('request=request', rounds[0])
            self.assertIn('source=lookup', rounds[0])
            self.assertEqual((entry.state, entry.parks, engines.unparks), ('parked', 1, 0))
            self.assertEqual(world.ops.violations, [])

    def test_flag_off_is_the_same_run_as_a_flag_that_was_never_set(self):
        outcomes = [run_parked(value) for value in (None, 'off', '0', '')]
        first = outcomes[0]
        self.assertIsNone(first.request.lookup)
        for outcome in outcomes:
            self.assertIsNone(outcome.request.lookup)
            self.assertEqual(lookup_lines(outcome.lines), [])
            self.assertEqual(outcome.stdout, '', 'nothing printed by a request without a lookup')
            self.assertEqual(outcome.tokens, first.tokens)
            self.assertEqual(outcome.epochs, first.epochs)
            self.assertEqual(shape(outcome.lines), shape(first.lines))

    def test_flag_off_imports_nothing_of_the_lookup(self):
        import serving_request_factory as factory

        for value in (None, 'off', '0', ''):
            with self.subTest(value=value):
                request = type('Request', (), {})()
                with environment(value), patch.dict(sys.modules):
                    sys.modules.pop('prompt_lookup', None)
                    factory.attach_lookup(request, (1, 2, 3), 'request')
                    self.assertNotIn('prompt_lookup', sys.modules)
                self.assertFalse(hasattr(request, 'lookup'), 'a flag-off request is not touched')

    def test_the_output_is_the_same_with_the_lookup_on_and_off(self):
        off, on = run_parked(None), run_parked('n3m12')
        self.assertIsNotNone(on.request.lookup)
        self.assertGreaterEqual(on.request.lookup.lookup_rounds, 1, 'the periodic prompt gives the lookup rounds to win')
        self.assertEqual(on.tokens, off.tokens)

    def test_a_built_request_attaches_it_as_before(self):
        with World(environment=PARKED) as world:
            with environment('n3m12'):
                request = admit(world, None, 'request')          # no parked set: today's per-request build
                self.assertFalse(hasattr(request, 'parked_slot'))
                self.assertIsNotNone(request.lookup)
                self.assertEqual(len([line for line in world.lines if line.startswith(prompt_lookup.ENGAGED)]), 1)
                self.assertEqual(request.prepare('request').source, 'lookup')
                request.close('request')
            self.assertEqual(world.ops.violations, [])

    def test_a_slot_without_a_parked_engine_is_built_and_attaches_it(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = engines.slots[0]
            engines.unpark(entry, 'test') if hasattr(engines, 'unpark') else self.skipTest('the set has no unpark')
            with environment('n3m12'):
                request = admit(world, engines, 'request')
                self.assertIsNone(getattr(request, 'parked_slot', None), 'built, not rebound')
                self.assertIsNotNone(request.lookup)
                request.close('request')
            self.assertEqual(world.ops.violations, [])

    def test_a_malformed_policy_after_the_rebind_parks_the_slot_again_and_raises(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            entry = engines.slots[0]
            with environment('rolling'):
                with self.assertRaises(ValueError):
                    admit(world, engines, 'request')
            self.assertEqual((entry.state, entry.parks, engines.unparks), ('parked', 1, 0), 'the engine bound to the failed request is parked, not leaked')
            self.assertEqual(entry.engine.phase, 'parked')
            self.assertEqual(world.ops.violations, [])
            # the slot serves the next request
            with environment('n3m12'):
                request = admit(world, engines, 'next')
                self.assertIs(request.parked_slot, entry)
                self.assertIsNotNone(request.lookup)
                drive(request)
                request.close('next')
            self.assertEqual(world.ops.violations, [])

    def test_two_requests_on_two_slots_keep_their_own_indexes(self):
        with World(environment=PARKED) as world:
            engines = make_set(world)
            engines.build()
            with environment('n3m12'):
                first = admit(world, engines, 'first', first_block=1)
                second = admit(world, engines, 'second', first_block=71)
                self.assertIsNot(first.lookup, second.lookup)
                self.assertIsNot(first.parked_slot, second.parked_slot)
                self.assertIsNot(first.lookup.index, second.lookup.index)
                self.assertEqual(first.lookup.request_id, 'first')
                self.assertEqual(second.lookup.request_id, 'second')
                drive(first)
                self.assertEqual(second.lookup.synced, 1, 'the second request\'s index did not follow the first one\'s commits')
                drive(second)
                first.close('first')
                second.close('second')
            self.assertEqual(world.ops.violations, [])


class ShippingTests(unittest.TestCase):
    def test_the_modules_reach_the_image_and_the_suite_runs_in_the_cpu_workflow(self):
        root = HERE.parent.parent
        overlay = (root / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8')
        dockerfile = (root / 'docker' / 'qwen-fast-serving.Dockerfile').read_text(encoding='utf-8')
        for name in ('prompt_lookup.py', 'serving_request_factory.py', 'serving_fast_policy.py'):
            self.assertTrue('scripts/ci/' + name in overlay, name + ' is not in the overlay list')
            self.assertTrue('scripts/ci/' + name in dockerfile, name + ' is not in the Dockerfile copy list')
        workflow = (root / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertIn('test_lookup_reuse', workflow)


if __name__ == '__main__':
    unittest.main()
