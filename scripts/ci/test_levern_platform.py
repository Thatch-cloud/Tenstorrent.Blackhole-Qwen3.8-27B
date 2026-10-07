"""Lever N at TP4: the platform wrap that keeps chunked prefill on (levern_platform), executed against the pinned plugin's own policy
function (fixtures/platform_chunked_prefill_policy.py, bf77cd63 platform.py's _apply_chunked_prefill_policy and its allowlist)."""

import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import levern_platform
import levern_policy

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'platform_chunked_prefill_policy.py'
BUDGET = 262144


class Logger(object):
    def __init__(self):
        self.lines, self.warnings = [], []

    def info(self, message, *values):
        self.lines.append(message % values if values else message)

    def warning(self, message, *values):
        self.warnings.append(message % values if values else message)


def platform_module(logger):
    module = types.ModuleType('vllm_tt_plugin.platform')
    module.logger = logger
    exec(compile(FIXTURE.read_text(encoding='utf-8'), str(FIXTURE), 'exec'), module.__dict__)
    return module


def config(model_type='qwen3_5', enable=True, budget=BUDGET, context=BUDGET, threshold=0):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(enable_chunked_prefill=enable, max_num_batched_tokens=budget,
                                         long_prefill_token_threshold=threshold, disable_chunked_mm_input=False),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type), max_model_len=context))


class PlatformWrapTests(unittest.TestCase):
    def test_the_stock_policy_turns_chunking_off_for_qwen3_5(self):
        logger = Logger()
        module = platform_module(logger)
        cfg = config(threshold=1024)
        module._apply_chunked_prefill_policy(cfg)
        self.assertFalse(cfg.scheduler_config.enable_chunked_prefill)
        self.assertEqual(cfg.scheduler_config.long_prefill_token_threshold, 0)
        self.assertIn('Chunked prefill is not supported', logger.lines[0])

    def test_the_wrap_keeps_chunking_on_and_the_budget_as_the_profile_set_it(self):
        logger = Logger()
        module = platform_module(logger)
        levern_platform.install(module, logger)
        cfg = config(threshold=1024)
        module._apply_chunked_prefill_policy(cfg)
        scheduler = cfg.scheduler_config
        self.assertTrue(scheduler.enable_chunked_prefill)
        self.assertEqual(scheduler.max_num_batched_tokens, BUDGET)
        self.assertEqual(scheduler.long_prefill_token_threshold, 0)
        self.assertIn('[PINDIAG] lever N: chunked prefill kept for qwen3_5 (budget=262144 threshold=0)', logger.lines)
        self.assertEqual(logger.lines[0].split('[PINDIAG]')[0], 'Chunked prefill is not supported for `model_type=qwen3_5`; disabling it.')

    def test_a_budget_below_the_context_is_not_bumped_away_from_the_request(self):
        logger = Logger()
        module = platform_module(logger)
        levern_platform.install(module, logger)
        cfg = config(budget=8192)
        module._apply_chunked_prefill_policy(cfg)
        self.assertEqual(cfg.scheduler_config.max_num_batched_tokens, 8192)
        self.assertTrue(cfg.scheduler_config.enable_chunked_prefill)

    def test_chunking_that_was_never_asked_for_stays_off(self):
        logger = Logger()
        module = platform_module(logger)
        levern_platform.install(module, logger)
        cfg = config(enable=False)
        module._apply_chunked_prefill_policy(cfg)
        self.assertFalse(cfg.scheduler_config.enable_chunked_prefill)
        self.assertFalse(any('lever N' in line for line in logger.lines))

    def test_a_model_the_platform_handles_itself_is_left_alone(self):
        logger = Logger()
        module = platform_module(logger)
        levern_platform.install(module, logger)
        cfg = config(model_type='gemma4')
        module._apply_chunked_prefill_policy(cfg)
        self.assertTrue(cfg.scheduler_config.disable_chunked_mm_input)
        self.assertTrue(cfg.scheduler_config.enable_chunked_prefill)
        self.assertFalse(any('lever N' in line for line in logger.lines))

    def test_the_marker_is_logged_once_per_process(self):
        logger = Logger()
        module = platform_module(logger)
        levern_platform.install(module, logger)
        for _ in range(3):
            module._apply_chunked_prefill_policy(config())
        self.assertEqual(sum('lever N' in line for line in logger.lines), 1)

    def test_install_is_idempotent_and_keeps_the_original(self):
        module = platform_module(Logger())
        original = module._apply_chunked_prefill_policy
        levern_platform.install(module)
        wrapped = module._apply_chunked_prefill_policy
        levern_platform.install(module)
        self.assertIs(module._apply_chunked_prefill_policy, wrapped)
        self.assertIs(wrapped.__wrapped__, original)
        self.assertTrue(getattr(wrapped, levern_platform.WRAPPED))

    def test_a_platform_without_the_policy_function_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            levern_platform.install(types.ModuleType('vllm_tt_plugin.platform'))
        self.assertIn(levern_platform.FUNCTION, str(caught.exception))

    def test_the_marker_is_the_policy_modules_line(self):
        self.assertEqual(levern_policy.PLATFORM_LINE.format(262144, 0),
                         '[PINDIAG] lever N: chunked prefill kept for qwen3_5 (budget=262144 threshold=0)')


if __name__ == '__main__':
    unittest.main()
