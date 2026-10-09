"""What the installed vLLM reads from the two drafter configs, before a card is involved.

The block-16 candidate differs from the served drafter's hub config.json in ONE value, dflash_config.block_size (8 against 16). The
serving profile gives vLLM 15 speculative tokens either way, and no TT source reads dflash_config (the card gates would be the first to
find out). This test loads each committed config through the installed vLLM's own config loader, holds the two equal but for that key, and
scans the installed vLLM for any read of it.

The committed-config tests run in the CPU suite; the installed-vLLM tests need vLLM (qwen-fast-vllm-cpu.yml) and are skipped without it. Not an in-image test.
"""
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import drafter_manifest as manifests

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / 'references' / 'drafter-configs'
NAMES = ('dedf8df6', 'b16-98759a49')


def config_path(name):
    return CONFIGS / ('%s.config.json' % name)


class CommittedConfigTests(unittest.TestCase):
    def test_each_config_is_the_manifests_pinned_bytes(self):
        for name in NAMES:
            digest = hashlib.sha256(config_path(name).read_bytes()).hexdigest()
            self.assertEqual(digest, manifests.load(name)['config_sha256'], name)

    def test_the_two_configs_differ_in_the_block_size_only(self):
        served, candidate = (json.loads(config_path(name).read_text(encoding='utf-8')) for name in NAMES)
        self.assertEqual((served['dflash_config']['block_size'], candidate['dflash_config']['block_size']), (8, 16))
        for document in (served, candidate):
            document['dflash_config'] = dict(document['dflash_config'], block_size=None)
        self.assertEqual(served, candidate)


@unittest.skipUnless(find_spec('vllm'), 'needs the installed vLLM (qwen-fast-vllm-cpu.yml)')
class InstalledVllmReadsTheSameValuesTests(unittest.TestCase):
    """SpeculativeConfig itself needs a target ModelConfig and the draft architecture in the model registry, which a CPU job without
    weights does not have; what matters is which keys of the draft config the installed vLLM reads, and what its own config loader
    returns for the two files."""

    def load(self, name):
        from vllm.transformers_utils.config import get_config
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        shutil.copyfile(str(config_path(name)), os.path.join(directory, 'config.json'))
        return get_config(directory, trust_remote_code=False)

    def test_vllms_own_loader_returns_the_same_config_but_the_block_size(self):
        served, candidate = self.load('dedf8df6'), self.load('b16-98759a49')
        print('[DRAFTER-CONFIG] served block_size=%s candidate block_size=%s' % (
            served.dflash_config.get('block_size'), candidate.dflash_config.get('block_size')))
        self.assertEqual((served.dflash_config['block_size'], candidate.dflash_config['block_size']), (8, 16))
        left, right = served.to_dict(), candidate.to_dict()
        for document in (left, right):
            document['dflash_config'] = dict((key, value) for key, value in document['dflash_config'].items() if key != 'block_size')
            for key in ('transformers_version', '_name_or_path'):
                document.pop(key, None)
        self.assertEqual(left, right)

    def test_no_installed_vllm_module_reads_block_size_from_the_draft_config(self):
        """The installed (pinned) vLLM reads dflash_config's causal, use_swa, swa_window_size, mask_token_id, target_layer_ids,
        use_aux_hidden_state and sink-bias keys; none of its modules asks it for block_size, so the candidate's 16 against the served 8
        reaches neither the scheduler nor the proposer. A vLLM bump that starts reading it fails here before a card is involved."""
        import re
        import vllm
        root = Path(vllm.__file__).resolve().parent
        readers, asked = [], re.compile(r"dflash_config[^\n]{0,80}block_size|block_size[^\n]{0,80}dflash_config")
        for path in root.rglob('*.py'):
            text = path.read_text(encoding='utf-8', errors='replace')
            if 'dflash_config' in text and asked.search(text):
                readers.append(str(path.relative_to(root)))
        self.assertEqual(readers, [], 'vLLM now reads the draft config block_size: the b16 arm changes what it does')


if __name__ == '__main__':
    unittest.main()
