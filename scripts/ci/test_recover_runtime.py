import unittest
from unittest.mock import patch
import json
import subprocess

from recover_runtime import IMAGE_ID, REPOSITORY, main, verified_reference


class RuntimeRecoveryTests(unittest.TestCase):
    def test_existing_exact_image_requires_no_registry_or_pull(self):
        with patch('recover_runtime.Path.mkdir'), patch('recover_runtime.Path.write_text'), \
                patch('recover_runtime.subprocess.run', return_value=
                    subprocess.CompletedProcess([], 0, IMAGE_ID + '\n', '')) as run:
            main()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][:3], ['docker', 'image', 'inspect'])

    def test_wrong_registry_config_never_pulls(self):
        manifest = {'SchemaV2Manifest': {'config': {'digest': 'sha256:' + 'b' * 64}},
            'Descriptor': {'digest': 'sha256:' + 'a' * 64}}
        with patch('recover_runtime.Path.mkdir'), patch('recover_runtime.subprocess.run', side_effect=[
                subprocess.CompletedProcess([], 1, '', 'missing'),
                subprocess.CompletedProcess([], 0, json.dumps(manifest), '')]) as run:
            with self.assertRaises(ValueError):
                main()
        self.assertEqual(run.call_count, 2)

    def test_pull_uses_digest_and_checks_downloaded_image(self):
        manifest = {'SchemaV2Manifest': {'config': {'digest': IMAGE_ID}},
            'Descriptor': {'digest': 'sha256:' + 'a' * 64}}
        for downloaded in (IMAGE_ID, 'sha256:' + 'b' * 64):
            with self.subTest(downloaded=downloaded), patch('recover_runtime.Path.mkdir'), \
                    patch('recover_runtime.Path.write_text') as write, \
                    patch('recover_runtime.subprocess.run', side_effect=[
                        subprocess.CompletedProcess([], 1, '', 'missing'),
                        subprocess.CompletedProcess([], 0, json.dumps(manifest), ''),
                        subprocess.CompletedProcess([], 0),
                        subprocess.CompletedProcess([], 0, downloaded + '\n', '')]) as run:
                if downloaded == IMAGE_ID:
                    main()
                    self.assertEqual(write.call_count, 2)
                else:
                    with self.assertRaises(RuntimeError):
                        main()
                    self.assertEqual(write.call_count, 1)
                self.assertEqual(run.call_args_list[2].args[0],
                    ['docker', 'pull', REPOSITORY + '@sha256:' + 'a' * 64])

    def test_immutable_reference_for_exact_config(self):
        document = {'SchemaV2Manifest': {'config': {'digest': IMAGE_ID}},
            'Descriptor': {'digest': 'sha256:' + 'a' * 64}}
        self.assertEqual(verified_reference(document), REPOSITORY + '@sha256:' + 'a' * 64)

    def test_reject_changed_image_missing_digest_and_indexes(self):
        for document in ([], {},
                {'SchemaV2Manifest': {'config': {'digest': 'sha256:' + 'b' * 64}},
                    'Descriptor': {'digest': 'sha256:' + 'a' * 64}},
                {'SchemaV2Manifest': {'config': {'digest': IMAGE_ID}}},
                {'SchemaV2Manifest': {'config': {'digest': IMAGE_ID}},
                    'Descriptor': {'digest': 'latest'}}):
            with self.subTest(document=document), self.assertRaises(ValueError):
                verified_reference(document)
