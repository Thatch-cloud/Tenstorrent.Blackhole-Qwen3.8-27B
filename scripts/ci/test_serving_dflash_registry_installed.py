import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from transformers import Qwen3Config
from vllm import ModelRegistry
from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig

try:
    from vllm_tt_plugin.qwen_dflash_registry import DFlash2DraftModel, register
except ModuleNotFoundError:
    from serving_dflash_registry import DFlash2DraftModel, register


class DFlash2RegistryTests(unittest.TestCase):
    def test_metadata_does_not_enable_standard_draft_execution(self):
        register()
        info, architecture = ModelRegistry.inspect_model_cls(['DFlash2DraftModel'])
        self.assertEqual(architecture, 'DFlash2DraftModel')
        self.assertTrue(info.is_text_generation_model)
        with self.assertRaisesRegex(RuntimeError, 'explicit TT combined'):
            DFlash2DraftModel(vllm_config=None)

    def test_real_speculative_config_preserves_dflash2_architecture(self):
        register()
        with TemporaryDirectory() as directory:
            target_path = Path(directory) / 'target'
            draft_path = Path(directory) / 'draft'
            target_path.mkdir()
            draft_path.mkdir()
            metadata = Qwen3Config(hidden_size=5120, intermediate_size=17408,
                num_hidden_layers=5, num_attention_heads=32, num_key_value_heads=8,
                head_dim=128, vocab_size=248320, max_position_embeddings=262144,
                sliding_window=2048, use_sliding_window=True,
                layer_types=['sliding_attention'] * 5,
                architectures=['Qwen3ForCausalLM'], eos_token_id=248044,
                pad_token_id=248044, dtype='bfloat16')
            metadata.save_pretrained(target_path)
            draft = metadata.to_dict()
            draft['architectures'] = ['DFlash2DraftModel']
            draft['dflash_config'] = dict(block_size=8, conv_group_size=16,
                conv_kernel_size=2, mask_token_id=248070, selector_rank=256,
                selector_top_k=16, target_layer_ids=[5, 19, 33, 47, 61])
            (draft_path / 'config.json').write_text(json.dumps(draft), encoding='utf-8')
            target = ModelConfig(model=str(target_path), dtype='bfloat16',
                max_model_len=4352, skip_tokenizer_init=True, seed=0)
            config = SpeculativeConfig(model=str(draft_path), method='dflash',
                num_speculative_tokens=15, draft_sample_method='greedy',
                rejection_sample_method='standard', target_model_config=target,
                target_parallel_config=ParallelConfig())
            self.assertEqual(config.draft_model_config.hf_config.architectures,
                ['DFlash2DraftModel'])
            self.assertEqual(config.draft_model_config.hf_config.dflash_config,
                draft['dflash_config'])
            self.assertEqual(config.num_speculative_tokens, 15)
            self.assertEqual(config.draft_model_config.get_vocab_size(), 248320)
