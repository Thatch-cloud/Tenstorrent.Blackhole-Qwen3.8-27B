"""NP9: the per-request output ceiling of the four-card Swift stack, 16,384 -> 32,768 (docs/tp4-np9-answer32k.md), held on CPU (stdlib only: no torch, no device).

What is held here, in the order of the document:

1. the policy accepts a QWEN_FAST_OUTPUT_BUDGET of 256 to 32,768 in whole 256-token families, and nothing else;
2. what the old 16,384 cap protected does not apply to the serving shape: its verifier captures, proposal bucket and page tables do not depend on the budget (so the raise costs no device
   memory), and the clamp can never hand the drafter or the page table a budget they refuse;
3. the contract: the clamp, the default for an omitted max_tokens (16,384, unchanged), the prompt cap (253,920, unchanged) and the checkpoint-level raise (Swift only, a rule of its own);
4. the ship-4 profile and its waiver are byte-identical to what the owner approved, and no profile of the file asks for a budget above 16,384;
5. the KV reservation admission: the numbers of the seat trade-off (how many requests fit the 16,499-block pool at which prompt lengths) and the scheduler hold, on the real module.
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
import types
import unittest
import unittest.mock as mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_shapes  # noqa: E402
import qwen_c2_checkpoint as checkpoint  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_fast_policy as policy  # noqa: E402
import serving_kv_reservation as reservation  # noqa: E402
import serving_request_factory as factory  # noqa: E402
import verifier_engine  # noqa: E402
from dflash_proposal_inputs import proposal_contexts  # noqa: E402
from serving_page_binding import validate_initial_capture_pages  # noqa: E402

PROFILES = HERE / 'qwen_c2_profiles.json'
CATALOG = json.loads((HERE / 'qwen_c2_checkpoints.json').read_text(encoding='utf-8'))
SWIFT = CATALOG['checkpoints'][0]
PAIR = {checkpoint.REPO_ENV: SWIFT['repo'], checkpoint.REVISION_ENV: SWIFT['revision']}
SHIP = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s4'
# sha256 of the canonical JSON (sort_keys, compact separators) of the ship-4 profile entry and of its owner_traffic_waiver, as the owner approved them on 2026-10-12 and as origin/tp4/swift
# carried them when NP9 began. NP9 changes neither: the raise is the Swift catalog entry's.
SHIP_DIGEST = 'a7703528f688b0d581081d4c63325deb7c390373d124464b7f9dc92056c8cd6f'
WAIVER_DIGEST = '96d81ceea693ce8721c70fe0c249ddcb908dcbb557adb776d7b800781b6efcce'
WINDOW = 262144
POOL = 16499                 # 16,500 blocks less vLLM's null block (the ship-4 profile's pool)
EIGHT_SEATS = 8


def profiles():
    return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def params(max_tokens=None, **values):
    return types.SimpleNamespace(**dict(dict(n=1, logprobs=None, prompt_logprobs=None, structured_outputs=None, logit_bias=None, allowed_token_ids=None, bad_words=None, stop=None, min_tokens=0,
                                             stop_token_ids=None, max_tokens=max_tokens, temperature=0.7), **values))


def swift_limits():
    ship = dict(profiles()[SHIP], name=SHIP)
    with mock.patch.object(contract, 'log', lambda *args: None):
        return contract.checkpoint_limits(contract.request_limits(ship), dict(PAIR), ship)


def base_limits():
    return contract.request_limits(dict(profiles()[SHIP], name=SHIP))


def enforce(limits, prompt_tokens, max_tokens=None):
    request = params(max_tokens)
    contract.enforce_request(request, prompt_tokens=prompt_tokens, max_model_len=WINDOW, budget=limits['budget'], eos_ids=frozenset((248046, 248044)),
                             max_prompt_tokens=limits['max_prompt_tokens'], min_answer_tokens=limits['min_answer_tokens'], default_max_tokens=limits['default_max_tokens'],
                             drafter_headroom_tokens=limits.get('drafter_headroom_tokens', 0), kv_pool_blocks=limits.get('kv_pool_blocks'))
    return request.max_tokens


class PolicyBoundTests(unittest.TestCase):
    def test_the_budget_runs_from_256_to_32768_in_whole_families(self):
        for value in ('256', '4096', '16384', '16640', '32512', '32768', 32768):
            self.assertEqual(policy._output_budget(value), int(value))
        self.assertEqual(policy._output_budget(None), 256)
        self.assertEqual(policy._output_budget(''), 256)
        for value in ('0', '255', '300', '32767', '33024', '32769', '65536', '-256', 'many', '16384.0'):
            with self.subTest(value=value), self.assertRaises(ValueError) as raised:
                policy._output_budget(value)
            self.assertIn('from 256 to 32768', str(raised.exception))

    def test_the_numbers_the_modules_repeat_are_one_number(self):
        self.assertEqual(policy.MAXIMUM_OUTPUT_BUDGET, contract.MAXIMUM_OUTPUT_BUDGET)
        self.assertEqual(policy.MAXIMUM_OUTPUT_BUDGET, checkpoint.OUTPUT_BUDGET_MAX)
        self.assertEqual(policy.PACKED_FAMILY_TOKENS, contract.OUTPUT_BUDGET_STEP)
        self.assertEqual(policy.PACKED_FAMILY_TOKENS, checkpoint.OUTPUT_BUDGET_STEP)
        self.assertEqual(contract.OUTPUT_BUDGET_ENV, 'QWEN_FAST_OUTPUT_BUDGET')
        self.assertEqual(contract.LEGACY_OUTPUT_BUDGET, 16384)
        self.assertEqual(policy.MINIMUM_MODEL_LEN, contract.MINIMUM_CONTEXT + policy.OUTPUT_BUDGET)

    def run_policy(self, budget, asked):
        """validate_request_sampling in a process whose environment says `budget` (OUTPUT_BUDGET is read when the module is imported): 'ok' or the refusal."""
        code = ('import sys, types\nimport serving_fast_policy as p\n'
                'sample = types.SimpleNamespace(temperature=0, n=1, max_tokens=%d, min_tokens=0, ignore_eos=False, logprobs=None, prompt_logprobs=None, presence_penalty=0,\n'
                '    frequency_penalty=0, repetition_penalty=1, stop=None, stop_token_ids=None, _eos_token_id=None)\n'
                'try:\n    p.validate_request_sampling(sample, prompt_tokens=4096)\n    print("ok", p.OUTPUT_BUDGET, p.MINIMUM_MODEL_LEN)\n'
                'except ValueError as error:\n    print("refused", p.OUTPUT_BUDGET, error)\n' % asked)
        env = dict(os.environ, PYTHONPATH=str(HERE), PYTHONDONTWRITEBYTECODE='1')
        env.pop('QWEN_FAST_OUTPUT_BUDGET', None)
        if budget is not None:
            env['QWEN_FAST_OUTPUT_BUDGET'] = str(budget)
        result = subprocess.run([sys.executable, '-s', '-c', code], env=env, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', 'replace')[-1500:])
        return result.stdout.decode().strip()

    def test_the_fast_path_admits_a_request_up_to_the_ceiling_it_was_started_with(self):
        self.assertEqual(self.run_policy(32768, 32768), 'ok 32768 36864')
        self.assertTrue(self.run_policy(32768, 32769).startswith('refused 32768 Fast serving requires a greedy single-sequence request'))
        self.assertIn('at most 32768 output tokens', self.run_policy(32768, 32769))
        self.assertEqual(self.run_policy(16384, 16384), 'ok 16384 20480')
        self.assertIn('at most 16384 output tokens', self.run_policy(16384, 16385))
        self.assertIn('at most 16384 output tokens', self.run_policy(16384, 32768), 'a profile that keeps 16,384 still refuses 32,768')
        self.assertIn('at most 256 output tokens', self.run_policy(None, 257), 'unset is still 256')


class CaptureShapeTests(unittest.TestCase):
    """What the cap protected (policy comment, verifier_engine.capture_bucket_rows) is not allocated on the serving shape."""

    def test_the_serving_captures_are_the_sequential_widths_whatever_the_budget(self):
        rows = packed_shapes.M3_SEQUENTIAL_CAPTURE_ROWS
        self.assertTrue(factory.sequential_captures(rows), 'beside the M3 blocks the widths are narrower than a replayed block (REPLAY_MIN_ROWS)')
        for budget in range(256, 32768 + 1, 256):
            self.assertEqual(verifier_engine.capture_bucket_rows(16, budget, rows), (1, 2, 4), budget)

    def test_the_replay_shape_is_where_a_bigger_budget_costs_memory_and_the_contract_keeps_it_away(self):
        wide = packed_shapes.SEQUENTIAL_CAPTURE_ROWS
        low, high = (len(verifier_engine.capture_bucket_rows(16, budget, wide)) for budget in (16384, 32768))
        self.assertEqual((low, high), (133, 261), 'replay attention: one capture per width and per 256-position family, so the old cost model: it doubles with the budget')
        replay = dict(profiles()[SHIP], name='replay')
        replay['env'] = dict(replay['env'])
        del replay['env']['QWEN_FAST_EXTENT_REPLAY']
        for budget in (16640, 32768):
            self.assertTrue(contract.output_budget_problems(replay, budget), budget)
            with self.assertRaises(ValueError):
                contract.request_limits(dict(replay, env=dict(replay['env'], QWEN_FAST_OUTPUT_BUDGET=str(budget))))
        self.assertEqual(contract.output_budget_problems(replay, 16384), [], 'nothing new is asked of a profile that keeps the legacy budget')
        no_any = dict(profiles()[SHIP], name='no-any')
        no_any['env'] = dict(no_any['env'], QWEN_FAST_ANY_REQUEST='0')
        self.assertTrue(contract.output_budget_problems(no_any, 32768))

    def test_one_proposal_bucket_and_the_drafter_bound_never_refuse_a_clamped_budget(self):
        for position in (1, 255, 2048, 4096, 100000, 229344):
            for budget in (16384, 32768):
                self.assertEqual(factory.single_bucket_contexts(position, budget), (2048,))
                self.assertEqual(proposal_contexts(position, budget), proposal_contexts(position, 16384), 'the draft history ladder tops out at 2,048 rows whatever the answer length')
        # the clamp (request_budget) can never exceed 262,144 - position - 32, the bound proposal_contexts raises on, at any prompt the contract admits
        for prompt in list(range(1, 253921, 997)) + [229344, 229345, 253920]:
            granted = factory.request_budget(params(40000), prompt_tokens=prompt, capacity=WINDOW, ceiling=32768)
            self.assertLessEqual(granted, 262144 - prompt - 32, prompt)
            self.assertGreaterEqual(granted, 8192 if prompt <= 253920 else 1)
            proposal_contexts(prompt, granted)

    def test_the_page_table_of_the_whole_window_holds_the_longest_answer_at_every_prompt(self):
        class Row(object):
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return self.values

        class Pages(object):
            ndim = 2
            shape = (1, WINDOW // 64)

            def __getitem__(self, index):
                return Row(list(range(4096)))

        blocks = tuple(range(4096))
        for prompt in (1, 4096, 100000, 229344, 253920):
            budget = factory.request_budget(params(32768), prompt_tokens=prompt, capacity=WINDOW, ceiling=32768)
            validate_initial_capture_pages(Pages(), blocks, position=prompt, output_budget=budget)
        with self.assertRaises(ValueError):
            validate_initial_capture_pages(Pages(), blocks, position=253920, output_budget=WINDOW)     # the check itself still bites past the window

    def test_the_extent_pool_refuses_a_replay_width_so_the_budget_can_never_reach_one(self):
        source = (HERE / 'serving_buffer_pool.py').read_text(encoding='utf-8')
        self.assertIn('Extent replay pools no per-family replay tables', source)
        self.assertIn('if any(rows >= 8 for rows in bucket_rows):', source)


class ContractLimitsTests(unittest.TestCase):
    def test_swift_gets_32768_and_nothing_else_moves(self):
        base, swift = base_limits(), swift_limits()
        self.assertEqual((base['budget'], base['default_max_tokens']), (16384, 8192))
        self.assertEqual((swift['budget'], swift['default_max_tokens']), (32768, 16384))
        moved = ('budget', 'default_max_tokens')
        self.assertEqual({k: v for k, v in swift.items() if k not in moved}, {k: v for k, v in base.items() if k not in moved})
        self.assertEqual((swift['max_prompt_tokens'], swift['min_answer_tokens'], swift['drafter_headroom_tokens'], swift['kv_pool_blocks']), (253920, 8192, 32, POOL))

    def test_the_clamp(self):
        swift, base = swift_limits(), base_limits()
        for asked, granted in ((1, 1), (8192, 8192), (16384, 16384), (16385, 16385), (20000, 20000), (30000, 30000), (32768, 32768), (32769, 32768), (64000, 32768), (10 ** 6, 32768)):
            self.assertEqual(enforce(swift, 1000, asked), granted, asked)
        for asked, granted in ((20000, 16384), (32768, 16384), (5000, 5000)):
            self.assertEqual(enforce(base, 1000, asked), granted, 'the base keeps its ceiling: %d' % asked)
        # the window less the 32 positions the drafter's last proposal reaches less the prompt: a long prompt leaves less than the ceiling
        for prompt, granted in ((229344, 32768), (229345, 32767), (240000, 22112), (250000, 12112), (253920, 8192)):
            self.assertEqual(enforce(swift, prompt, 32768), granted, prompt)
            self.assertLessEqual(prompt + granted, WINDOW - 32)

    def test_the_default_for_an_omitted_max_tokens_is_still_16384_and_the_base_8192(self):
        swift, base = swift_limits(), base_limits()
        self.assertEqual(enforce(swift, 1000, None), 16384)
        self.assertEqual(enforce(swift, 1000, WINDOW - 1000), 16384, 'vLLM fills an omitted max_tokens with the rest of the window: read as omitted')
        self.assertEqual(enforce(swift, 250000, None), 12112)
        self.assertEqual(enforce(swift, 253920, None), 8192)
        self.assertEqual(enforce(base, 1000, None), 8192)

    def test_the_prompt_cap_is_253920_for_both(self):
        for limits in (base_limits(), swift_limits()):
            self.assertEqual(contract.prompt_room(WINDOW, limits['budget'], limits['max_prompt_tokens'], limits['min_answer_tokens'], limits['drafter_headroom_tokens']), 253920)
            self.assertEqual(enforce(limits, 253920, 100), 100)
            with self.assertRaises(contract.ContractError):
                enforce(limits, 253921, 100)

    def test_the_drafter_end_limit_still_holds_for_every_prompt_and_every_ceiling(self):
        swift = swift_limits()
        self.assertEqual(contract.DRAFTER_END_LIMIT, factory.DRAFTER_END_LIMIT)
        for prompt in list(range(1, 253921, 1999)) + [253920]:
            for asked in (None, 1, 32768, 40000):
                self.assertLessEqual(prompt + enforce(swift, prompt, asked), contract.DRAFTER_END_LIMIT)

    def test_the_raise_is_swifts_alone_and_a_profile_that_would_lose_prompt_room_is_refused_by_name(self):
        ship = dict(profiles()[SHIP], name=SHIP)
        limits = contract.request_limits(ship)
        self.assertEqual(contract.checkpoint_budget_problems(ship, limits, 32768), [])
        self.assertEqual(contract.checkpoint_budget_problems(ship, limits, 16384), [], 'equal is no change')
        self.assertTrue(contract.checkpoint_budget_problems(ship, limits, 8192), 'a checkpoint only raises')
        self.assertTrue(any('lower the profile' in line for line in contract.checkpoint_budget_problems(ship, limits, 8192)))
        for budget in (32769, 33024, 65536, 0, 100, '32768', None):
            self.assertTrue(contract.checkpoint_budget_problems(ship, limits, budget), budget)
        no_room = dict(ship, min_answer_tokens=None, max_prompt_tokens=None)
        problems = contract.checkpoint_budget_problems(no_room, contract.request_limits(no_room), 32768)
        self.assertTrue(any('longest admitted prompt' in line for line in problems), problems)
        small = copy.deepcopy(ship)
        small['engine']['max-model-len'] = 32768 + 4096 - 64
        self.assertTrue(any('window of at least 36864' in line for line in contract.output_budget_problems(small, 32768)))
        no_reader = copy.deepcopy(ship)
        del no_reader['env']['QWEN_FAST_EXTENT_REPLAY']
        self.assertTrue(any('QWEN_FAST_EXTENT_REPLAY' in line for line in contract.checkpoint_budget_problems(no_reader, limits, 32768)))
        with mock.patch.object(contract, 'log', lambda *args: None), self.assertRaises(ValueError) as raised:
            contract.checkpoint_limits(contract.request_limits(no_room), dict(PAIR), no_room)
        self.assertIn('longest admitted prompt', str(raised.exception))
        with mock.patch.object(contract, 'log', lambda *args: None), self.assertRaises(ValueError):
            contract.checkpoint_limits(limits, dict(PAIR))          # no profile to hold the number to

    def test_a_profile_may_name_32768_itself_on_the_ship_shape_and_nowhere_else(self):
        twin = dict(profiles()[SHIP], name='twin')
        twin['env'] = dict(twin['env'], QWEN_FAST_OUTPUT_BUDGET='32768')
        self.assertEqual(contract.request_limits(twin)['budget'], 32768)
        for key in ('QWEN_FAST_ANY_REQUEST', 'QWEN_FAST_EXTENT_REPLAY'):
            broken = dict(twin, env=dict(twin['env'], **{key: '0'}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.request_limits(broken)

    def test_the_catalog_rule_is_checked_at_load(self):
        self.assertEqual(SWIFT['request'], {'default_max_tokens': 16384, 'output_budget': 32768})
        self.assertNotIn('request', CATALOG['base'])
        self.assertEqual(checkpoint.REQUEST_KEYS, frozenset(('default_max_tokens', 'output_budget')))


class ShippedProfilesTests(unittest.TestCase):
    def test_the_approved_ship4_profile_and_its_waiver_are_byte_identical(self):
        ship = profiles()[SHIP]
        self.assertEqual(digest(ship), SHIP_DIGEST)
        self.assertEqual(digest(ship['owner_traffic_waiver']), WAIVER_DIGEST)
        self.assertEqual(ship['owner_traffic_waiver']['decision'][:8], 'APPROVED')
        self.assertEqual((ship['env']['QWEN_FAST_OUTPUT_BUDGET'], ship['default_max_tokens'], ship['min_answer_tokens'], ship['max_prompt_tokens']), ('16384', 8192, 8192, 253920))

    def test_no_profile_of_the_file_asks_for_more_than_16384(self):
        budgets = {name: int(entry['env']['QWEN_FAST_OUTPUT_BUDGET']) for name, entry in profiles().items() if 'QWEN_FAST_OUTPUT_BUDGET' in entry.get('env', {})}
        self.assertTrue(budgets)
        self.assertEqual(max(budgets.values()), 16384, [name for name, value in budgets.items() if value > 16384])
        for name in profiles():
            with self.subTest(profile=name):
                if 'QWEN_FAST_OUTPUT_BUDGET' in profiles()[name].get('env', {}):
                    contract.request_limits(dict(profiles()[name], name=name))      # every shipped profile still passes the limits check

    def test_only_the_swift_entry_of_the_catalog_carries_a_request_block(self):
        self.assertEqual([entry['repo'] for entry in CATALOG['checkpoints'] if 'request' in entry], [SWIFT['repo']])


def seats(prompt, max_tokens, pool=POOL):
    """How many identical requests the reservation lets run at once (the scheduler seats are eight): the pool over one request's blocks, with max_tokens clamped as the contract clamps it."""
    granted = min(max_tokens, WINDOW - 32 - prompt)
    return min(EIGHT_SEATS, pool // reservation.request_blocks(prompt, granted))


def longest_prompt_for(seat_count, max_tokens, pool=POOL):
    return max(prompt for prompt in range(1, 253921) if seat_count * reservation.request_blocks(prompt, min(max_tokens, WINDOW - 32 - prompt)) <= pool)


class KvAdmissionTests(unittest.TestCase):
    """The cost of a bigger ceiling is KV admission, not device memory: the reservation holds prompt + max_tokens blocks per running request (serving_kv_reservation.request_blocks)."""

    def test_the_numbers_of_the_trade_off(self):
        self.assertEqual(reservation.request_blocks(1000, 16384), 274)
        self.assertEqual(reservation.request_blocks(1000, 32768), 530)
        self.assertEqual(reservation.request_blocks(99104, 32768), 2062)
        # the longest prompt at which N identical requests all fit the pool at once
        table = {(8192, 8): 123680, (16384, 8): 115488, (30000, 8): 101872, (32768, 8): 99104,
                 (16384, 7): 134368, (32768, 7): 117984, (16384, 6): 159456, (32768, 6): 143072, (16384, 5): 194656, (32768, 5): 178272}
        for (max_tokens, seat_count), prompt in table.items():
            self.assertEqual(longest_prompt_for(seat_count, max_tokens), prompt, (max_tokens, seat_count))
        self.assertEqual(longest_prompt_for(4, 32768), 253920, 'four requests at the longest prompt and the longest answer always fit: 4 x 4,097 blocks')
        self.assertEqual(max(reservation.request_blocks(p, min(32768, WINDOW - 32 - p)) for p in range(1, 253921)), 4097)
        self.assertEqual(max(reservation.request_blocks(p, min(16384, WINDOW - 32 - p)) for p in range(1, 253921)), 4097)

    def test_the_seats_at_each_prompt_length(self):
        rows = {4096: (8, 8, 8), 32768: (8, 8, 8), 98304: (8, 8, 8), 100000: (8, 8, 7), 115488: (8, 8, 7), 120000: (8, 7, 6), 131072: (7, 7, 6), 200000: (5, 4, 4), 253920: (4, 4, 4)}
        for prompt, (small, default, big) in rows.items():
            self.assertEqual((seats(prompt, 8192), seats(prompt, 16384), seats(prompt, 32768)), (small, default, big), prompt)

    def test_a_single_request_is_never_refused_for_its_blocks_at_the_new_ceiling(self):
        swift = swift_limits()
        for prompt in list(range(1, 253921, 499)) + [253920]:
            for asked in (None, 32768, 40000):
                granted = enforce(swift, prompt, asked)           # no ContractError: the pool holds the largest request alone
                self.assertLessEqual(reservation.request_blocks(prompt, granted), swift['kv_pool_blocks'])

    def scheduler(self, running):
        pool = types.SimpleNamespace(num_gpu_blocks=POOL + 1)
        return types.SimpleNamespace(kv_cache_manager=types.SimpleNamespace(block_pool=pool), max_model_len=WINDOW, running=running)

    @staticmethod
    def request(name, prompt, max_tokens):
        return types.SimpleNamespace(request_id=name, num_prompt_tokens=prompt, max_tokens=max_tokens)

    def test_the_scheduler_hold_at_long_prompts_with_the_new_ceiling(self):
        prompt, lines = longest_prompt_for(8, 32768), []
        self.assertEqual(prompt, 99104)
        log = lambda template, *values: lines.append(template.format(*values))
        running = [self.request('r%d' % index, prompt, 32768) for index in range(7)]
        newcomer = self.request('r7', prompt, 32768)
        self.assertFalse(reservation.hold(self.scheduler(running), [newcomer], 7, {}, log), 'the eighth fits at the longest prompt that lets eight 32,768-token requests in')
        self.assertFalse(reservation.hold(self.scheduler(running), [self.request('r7', prompt + 192, 32768)], 7, {}, log), '3 blocks of the pool are left over at this prompt: +192 tokens still fits')
        longer = self.request('r7', prompt + 256, 32768)
        self.assertTrue(reservation.hold(self.scheduler(running), [longer], 7, {}, log), 'one block past the pool and the eighth waits for blocks: it is held, never preempted')
        self.assertTrue(lines and lines[-1].startswith('[PINDIAG] kv reservation hold request=r7'), lines)
        # the same eighth request at the 16,384 ceiling fits at that prompt: the trade-off is the ceiling a client asks for
        running = [self.request('r%d' % index, prompt, 16384) for index in range(7)]
        self.assertFalse(reservation.hold(self.scheduler(running), [self.request('r7', prompt + 256, 16384)], 7, {}, log))
        # a lone request is never held, whatever it asks for
        self.assertFalse(reservation.hold(self.scheduler([]), [self.request('big', 253920, 32768)], 0, {}, log))

    def test_a_client_that_names_no_max_tokens_reserves_what_it_did_before(self):
        swift = swift_limits()
        for prompt in (1000, 50000, 115000, 200000):
            self.assertEqual(reservation.request_blocks(prompt, enforce(swift, prompt, None)), reservation.request_blocks(prompt, min(16384, WINDOW - 32 - prompt)))


if __name__ == '__main__':
    unittest.main()
