"""Every simulator success path must verify commit-only source stability."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class CommitCompletionTests(unittest.TestCase):
    def completion_checks(self):
        source = Path(__file__).with_name('gdn-multitoken.py')
        tree = ast.parse(source.read_text(), feature_version=(3, 10))
        checks = []
        for parent in ast.walk(tree):
            for field in ('body', 'orelse', 'finalbody'):
                statements = getattr(parent, field, [])
                if not isinstance(statements, list):
                    continue
                for index, statement in enumerate(statements):
                    if (isinstance(statement, ast.Assign)
                            and ast.unparse(statement) == "report['passed'] = True"):
                        previous = statements[index - 1]
                        self.assertIsInstance(previous, ast.If)
                        self.assertEqual(ast.unparse(previous.test), 'args.commit_only_gdn')
                        checks.append(compile(ast.Module(body=[previous], type_ignores=[]), str(source), 'exec'))
        self.assertEqual(len(checks), 2)
        return checks

    def test_all_success_paths_record_unchanged_sources(self):
        for check in self.completion_checks():
            report = {'commit_sources': {'kernel': 'original'}}
            namespace = dict(args=SimpleNamespace(commit_only_gdn=True), report=report,
                source_hashes=lambda directory: {'kernel': 'original'}, Path=Path, __file__=__file__)
            exec(check, namespace)
            self.assertEqual(report['commit_sources_after'], report['commit_sources'])

    def test_all_success_paths_reject_changed_sources(self):
        for check in self.completion_checks():
            namespace = dict(args=SimpleNamespace(commit_only_gdn=True),
                report={'commit_sources': {'kernel': 'original'}},
                source_hashes=lambda directory: {'kernel': 'changed'}, Path=Path, __file__=__file__)
            with self.assertRaisesRegex(ValueError, 'changed during simulation'):
                exec(check, namespace)


if __name__ == '__main__':
    unittest.main()
