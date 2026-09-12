import subprocess
import unittest
from unittest.mock import patch

from recover_runtime import IMAGE_ID, REFERENCE, main


class RuntimeRecoveryTests(unittest.TestCase):
    def test_existing_exact_image_requires_no_registry_or_pull(self):
        with patch('recover_runtime.Path.mkdir'), patch('recover_runtime.Path.write_text'), \
                patch('recover_runtime.subprocess.run', return_value=
                    subprocess.CompletedProcess([], 0, IMAGE_ID + '\n', '')) as run:
            main()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][:3], ['docker', 'image', 'inspect'])

    def test_pull_uses_pinned_index_not_a_mutable_tag_or_config_digest(self):
        self.assertEqual(REFERENCE, 'zot.thatch.local:5000/tt-vllm@' + IMAGE_ID)
        for downloaded in (IMAGE_ID, 'sha256:' + 'b' * 64):
            with self.subTest(downloaded=downloaded), patch('recover_runtime.Path.mkdir'), \
                    patch('recover_runtime.Path.write_text') as write, \
                    patch('recover_runtime.subprocess.run', side_effect=[
                        subprocess.CompletedProcess([], 1, '', 'missing'),
                        subprocess.CompletedProcess([], 0),
                        subprocess.CompletedProcess([], 0, downloaded + '\n', '')]) as run:
                if downloaded == IMAGE_ID:
                    main()
                    write.assert_called_once()
                else:
                    with self.assertRaises(RuntimeError):
                        main()
                    write.assert_not_called()
                self.assertEqual(run.call_args_list[1].args[0], ['docker', 'pull', REFERENCE])

    def test_failed_pull_cannot_publish_success(self):
        with patch('recover_runtime.Path.mkdir'), patch('recover_runtime.Path.write_text') as write, \
                patch('recover_runtime.subprocess.run', side_effect=[
                    subprocess.CompletedProcess([], 1, '', 'missing'),
                    subprocess.CalledProcessError(1, ['docker', 'pull', REFERENCE])]):
            with self.assertRaises(subprocess.CalledProcessError):
                main()
        write.assert_not_called()
