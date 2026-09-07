import random
import hashlib
import json
import unittest
from unittest.mock import patch

from lookup_acceptance import score_lookup, score_report
from lookup_draft import LookupDraft


class LookupAcceptanceTests(unittest.TestCase):
    def test_report_rejects_unverified_or_mutated_native_evidence(self):
        request = dict(prompt_tokens=[0, 1, 2], emitted=[0, 1, 2], max_new_tokens=3,
            vocab_size=10, eos_ids=[], exact=True, state_exact=True, inactive_exact=True)
        for tokens, digest in (('prompt_tokens', 'prompt_sha256'), ('emitted', 'output_sha256')):
            request[digest] = hashlib.sha256(json.dumps(request[tokens]).encode()).hexdigest()
        report = dict(passed=True, request_checks=[request])
        self.assertEqual(len(score_report(report)[0]['arms']), 48)
        for key in ('exact', 'state_exact', 'inactive_exact'):
            with self.assertRaises(ValueError):
                score_report(dict(passed=True, request_checks=[dict(request, **{key: False})]))
        with self.assertRaises(ValueError):
            score_report(dict(report, passed=False))
        request['emitted'][1] = 9
        with self.assertRaisesRegex(ValueError, 'hash'):
            score_report(report)

    def test_periodic_native_tape_and_target_only_control(self):
        prompt = [0, 1, 2] * 100
        emitted = [index % 3 for index in range(65)]
        wide = score_lookup(prompt, emitted, vocab_size=100, max_new_tokens=65, prefer_full_suffix=True)
        self.assertEqual(wide['accepted'], 62)
        self.assertEqual(wide['committed'], 64)
        self.assertEqual([block['rows'] for block in wide['blocks']], [32, 32])
        serial = score_lookup(prompt, emitted, vocab_size=100, max_new_tokens=65, max_rows=1)
        self.assertEqual(serial['accepted'], 0)
        self.assertEqual(len(serial['blocks']), 64)

    def test_threshold_changes_policy_not_reference_tokens(self):
        result = score_lookup([0, 1, 2] * 10, [0, 1, 2, 0], vocab_size=10,
                              max_new_tokens=4, minimum_match=129)
        self.assertEqual([block['rows'] for block in result['blocks']], [1, 1, 1])
        self.assertEqual(result['committed'], 3)

    def test_eos_in_accepted_prefix_or_correction_and_prefill(self):
        for emitted in ([0, 1, 2], [0, 1, 9], [9]):
            result = score_lookup([0, 1, 2] * 100, emitted, vocab_size=10,
                max_new_tokens=65, eos_ids=(emitted[-1],), prefer_full_suffix=True)
            self.assertEqual(result['committed'], len(emitted) - 1)
        self.assertIsNone(result['mean_committed_per_block'])

    def test_future_native_tokens_never_enter_draft_history(self):
        prompt, emitted = [1, 2, 3], [4, 5, 6, 7, 8]
        original = LookupDraft.propose_with_match
        observed = []

        def inspect(draft, request_id, count):
            observed.append(list(draft.history))
            consumed = len(draft.history) - len(prompt)
            self.assertEqual(draft.history, prompt + emitted[:consumed])
            return original(draft, request_id, count)

        with patch.object(LookupDraft, 'propose_with_match', inspect):
            result = score_lookup(prompt, emitted, vocab_size=10, max_new_tokens=5)
        self.assertTrue(observed)
        self.assertEqual(result['accepted'], 0)

    def test_random_complete_tapes_preserve_accounting(self):
        generator = random.Random(811)
        for repeat in range(100):
            prompt = [generator.randrange(5) for index in range(50)]
            emitted = [generator.randrange(5) for index in range(35)]
            result = score_lookup(prompt, emitted, vocab_size=10, max_new_tokens=35,
                max_rows=generator.choice((1, 2, 4, 8, 16, 32)), minimum_match=generator.choice((1, 2, 4)),
                prefer_full_suffix=bool(repeat % 2))
            self.assertEqual(result['committed'], 34)
            self.assertEqual(sum(block['committed'] for block in result['blocks']), 34)
            self.assertLessEqual(result['accepted'], result['proposed'])

    def test_incomplete_or_invalid_tapes_are_rejected(self):
        for options in (dict(emitted=[]), dict(emitted=[0, 1]), dict(emitted=[0, 9, 1], eos_ids=(9,)),
                        dict(emitted=[0, 1, True]), dict(max_rows=3), dict(minimum_match=0)):
            arguments = dict(prompt=[0, 1], emitted=[0, 1, 2], vocab_size=10, max_new_tokens=3)
            arguments.update(options)
            with self.assertRaises(ValueError):
                score_lookup(**arguments)
