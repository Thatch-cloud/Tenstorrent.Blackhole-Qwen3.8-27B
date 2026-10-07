"""What the installed vLLM reads from the two drafter configs, before a card is involved.

The block-16 candidate differs from the served drafter's hub config.json in ONE value, dflash_config.block_size (8 against 16). The
serving profile gives vLLM 15 speculative tokens either way, and no TT source reads dflash_config (the card gates would be the first to
find out). This test builds vLLM's own dflash SpeculativeConfig from each committed config, with the profile's speculative-config
arguments, and holds the scheduler-side values equal: the number of speculative tokens and what follows from it. A difference is not
a failure of the drafter, it is a thing the owner must read before the card window, so the assertion message prints both sides.

The committed-config tests run in the CPU suite; the installed-vLLM tests need vLLM (qwen-fast-vllm-cpu.yml) and are skipped without it. Not an in-image test.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from importlib.util import find_spec
import unittest

import drafter_manifest as manifests

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / 'references' / 'drafter-configs'
NAMES = ('dedf8df6', 'b16-98759a49')


def profile_speculative_config():
    document = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))
    return dict(document['profiles']['c2-packed-tp4']['engine']['speculative-config'])


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
    def build(self, name):
        from vllm.config import SpeculativeConfig
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        shutil.copyfile(str(config_path(name)), os.path.join(directory, 'config.json'))
        arguments = profile_speculative_config()
        arguments['model'] = directory
        return SpeculativeConfig(**arguments)

    def values(self, name):
        built = self.build(name)
        draft = built.draft_model_config.hf_config
        return dict(method=built.method, num_speculative_tokens=built.num_speculative_tokens,
                    draft_layers=getattr(draft, 'num_hidden_layers', None),
                    draft_hidden=getattr(draft, 'hidden_size', None),
                    draft_block_size=(getattr(draft, 'dflash_config', None) or {}).get('block_size'))

    def test_the_scheduler_side_values_are_the_same_for_both_drafters(self):
        served, candidate = self.values('dedf8df6'), self.values('b16-98759a49')
        print('[DRAFTER-CONFIG] served=%s candidate=%s' % (json.dumps(served, sort_keys=True), json.dumps(candidate, sort_keys=True)))
        self.assertEqual(served['num_speculative_tokens'], 15)
        for key in ('method', 'num_speculative_tokens', 'draft_layers', 'draft_hidden'):
            self.assertEqual(served[key], candidate[key], 'vLLM reads %s differently: served=%s candidate=%s' % (key, served, candidate))


if __name__ == '__main__':
    unittest.main()
