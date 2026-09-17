import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class CombinedUploadTests(unittest.TestCase):
    def test_predecessor_is_row_major_and_successor_tiled(self):
        tree = ast.parse(Path(__file__).with_name('t32-combined-proposal-probe.py').read_text())
        upload = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'upload')
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in ('predecessor', 'successor') for target in node.targets)]
        operations = SimpleNamespace(from_torch=Mock(), bfloat16='bf16', TILE_LAYOUT='tile',
            ROW_MAJOR_LAYOUT='row', DRAM_MEMORY_CONFIG='dram', ReplicateTensorToMesh=Mock())
        namespace = dict(ttnn=operations, mesh=object(), owned=[], reader=Mock())
        module = ast.fix_missing_locations(ast.Module(body=[upload, *assignments], type_ignores=[]))
        exec(compile(module, '<combined-upload-test>', 'exec'), namespace)
        self.assertEqual([call.kwargs['layout'] for call in operations.from_torch.call_args_list], ['row', 'tile'])
        self.assertEqual(len(namespace['owned']), 2)


if __name__ == '__main__':
    unittest.main()
