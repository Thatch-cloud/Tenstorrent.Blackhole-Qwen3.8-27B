"""A 262,144-token Gemma request stores 2,088 objects (design 1.4): 2,048 full-attention blocks plus 5 sliding groups x the 8-block window tail.

vLLM falls back to DENSE sliding storage when the window tail cannot be represented (a retention interval of None): every sliding block is
stored, 12,288 objects, 34.2 GB. The platform check refuses that setting; this test guards the arithmetic against the real vLLM scheduler:
it counts what the real ``OffloadingConnectorScheduler`` asks the real ``NvmeKVManager`` to store for a 262,144-token prompt, once at the
defaults and once with the dense fallback forced as a control (so a green first half cannot come from a counter that counts nothing).
"""

import tempfile
import unittest

from harness import BLOCK, Rig, tokens
import harness
from tier_kit import RealTier

from transformers import GPT2Config
from vllm.config import ModelConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.structured_output import StructuredOutputManager

from ttkvp2p.manager import estimate_objects

PROMPT_TOKENS = 262144
FULL = PROMPT_TOKENS // BLOCK                 # 2,048
WINDOW_BLOCKS = 8                             # cdiv(1023, 128)
SLIDING_GROUPS = 5
EXPECTED = FULL + SLIDING_GROUPS * WINDOW_BLOCKS          # 2,088
DENSE = FULL * (1 + SLIDING_GROUPS)                        # 12,288


MAX_MODEL_LEN = PROMPT_TOKENS + 4096          # the scheduler never schedules a token at position max_model_len - 1: the prompt must leave room to sample


def big_model_config() -> ModelConfig:
    root = tempfile.mkdtemp(prefix='ttkvp2p-model262k-')
    GPT2Config(n_positions=MAX_MODEL_LEN, n_embd=256, n_layer=1, n_head=4, architectures=['GPT2LMHeadModel']).save_pretrained(root)
    return ModelConfig(model=root, dtype='float32', max_model_len=MAX_MODEL_LEN, skip_tokenizer_init=True, seed=0)


class ObjectCount(unittest.TestCase):
    def count_stored(self, retention_interval):
        """Run one 262,144-token prompt through the real scheduler and the real manager; the objects the manager admitted."""
        saved = harness.model_config
        big = big_model_config()
        harness.model_config = lambda: big
        self.addCleanup(setattr, harness, 'model_config', saved)
        rig = Rig(real_tier=RealTier(), num_slots=14000, num_blocks=4200, max_batched=16384, checks=False, with_scheduler=False, slot_objects=64)
        self.addCleanup(rig.close)
        if retention_interval != 'default':
            rig.vllm_config.cache_config.prefix_cache_retention_interval = retention_interval
        rig.scheduler = Scheduler(vllm_config=rig.vllm_config, kv_cache_config=rig.kv_cache_config,
                                  structured_output_manager=StructuredOutputManager(rig.vllm_config), block_size=BLOCK, hash_block_size=BLOCK,
                                  include_finished_set=False, log_stats=False)
        calls = []
        manager = rig.manager
        original = manager.prepare_store

        def spy(keys, ctx):
            out = original(keys, ctx)
            calls.append((len(list(keys)), len(out.keys_to_store)))
            return out

        manager.prepare_store = spy
        rig.add('big', tokens(PROMPT_TOKENS, 'big'))
        rig.finish('big', limit=4000)
        rig.drain(limit=4000)
        return rig, calls

    def test_the_defaults_store_2088_objects_for_a_262k_request(self):
        rig, calls = self.count_stored('default')
        manager = rig.manager
        self.assertEqual(manager.written(), EXPECTED, 'the object count of a 262k Gemma request')
        self.assertEqual(manager.entries, EXPECTED)
        by_group = {}
        for key in manager.index:
            group = int.from_bytes(key[-4:], 'big')
            by_group[group] = by_group.get(group, 0) + 1
        self.assertEqual(sorted(by_group.values()), [WINDOW_BLOCKS] * SLIDING_GROUPS + [FULL])
        self.assertEqual(max(by_group.values()), FULL)
        self.assertEqual(rig.manager.stored_objects, EXPECTED)
        self.assertEqual(estimate_objects(PROMPT_TOKENS, (None, 8, 8, 8, 8, 8)), EXPECTED, 'the admission estimate is the same number')
        self.assertEqual(rig.runtime.counters['done_store'] > 0, True)

    def test_the_count_follows_the_sliding_window_so_the_counter_is_not_blind(self):
        """A control: a 2,048-token window (16 blocks) must give 2,048 + 5 x 16 = 2,128. A counter that always says 2,088 would fail here."""
        saved = harness.WINDOW
        harness.WINDOW = 2048
        self.addCleanup(setattr, harness, 'WINDOW', saved)
        rig, calls = self.count_stored('default')
        self.assertEqual(rig.manager.written(), FULL + SLIDING_GROUPS * 16)
        self.assertEqual(estimate_objects(PROMPT_TOKENS, (None, 16, 16, 16, 16, 16)), FULL + SLIDING_GROUPS * 16)

    def test_the_dense_fallback_is_12288_objects_by_arithmetic(self):
        """vLLM stores EVERY sliding block (6 x 2,048 = 12,288 objects, 34.2 GB per 262k session) when a sliding group's window tail cannot be
        represented (V single_type_kv_cache_manager.py reachable_block_mask: ``alignment_tokens % block_size != 0``). This rig has uniform
        128-token blocks, which never triggers that branch, so the 12,288 is arithmetic here; W2a measures the real Gemma KV config, and the
        platform check refuses a retention interval of None."""
        self.assertEqual(DENSE, 12288)
        self.assertEqual(estimate_objects(PROMPT_TOKENS, (None,) * 6), DENSE)
        self.assertGreater(DENSE, 5 * EXPECTED)

    def test_the_sizes_in_bytes(self):
        object_bytes = 2785280
        self.assertAlmostEqual(EXPECTED * object_bytes / 1e9, 5.82, places=2)
        self.assertAlmostEqual(DENSE * object_bytes / 1e9, 34.2, places=1)


if __name__ == '__main__':
    unittest.main()
