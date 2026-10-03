from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from dspark_layer_mesh import INPUTS, execute


class DSparkLayerMeshTests(unittest.TestCase):
    def test_complete_layer_uses_device_partials_and_original_residual(self):
        runtime,collectives,weights = MagicMock(),MagicMock(),object()
        mesh = SimpleNamespace(shape=(1,2))
        inputs = {name:object() for name in INPUTS}
        retain = MagicMock()
        attention = {'attention_partial':object()}
        mlp = {'down_partial':object(),'attention_residual':object()}
        output = {'output':object()}
        attention_peers,down_peers = (object(),object()),(object(),object())
        with patch('dspark_layer_mesh.attention_partial',return_value=attention) as attend, \
                patch('dspark_layer_mesh.gather_partials',side_effect=(attention_peers,down_peers)) as gather, \
                patch('dspark_layer_mesh.mlp_partial',return_value=mlp) as feedforward, \
                patch('dspark_layer_mesh.finish',return_value=output) as finish:
            result = execute(runtime,mesh,collectives,inputs,weights,retain,mask_validated=True)
        self.assertEqual(result,dict(attention=attention,mlp=mlp,finish=output))
        self.assertIs(attend.call_args.kwargs['mesh'],mesh)
        self.assertIs(attend.call_args.kwargs['composed_attention'],True)
        self.assertIs(attend.call_args.kwargs['mask_validated'],True)
        self.assertEqual([entry.args for entry in gather.call_args_list],
            [(runtime,mesh,collectives,attention['attention_partial'],retain),
             (runtime,mesh,collectives,mlp['down_partial'],retain)])
        feedforward.assert_called_once_with(runtime,*attention_peers,inputs['noise'],weights,retain)
        finish.assert_called_once_with(runtime,*down_peers,mlp['attention_residual'],retain)
        self.assertEqual(runtime.mock_calls,[])

    def test_incomplete_inputs_or_unvalidated_mask_reject_before_execution(self):
        inputs = {name:object() for name in INPUTS}
        with patch('dspark_layer_mesh.attention_partial') as attend:
            for candidate,validated in (({},True),(inputs,False),(inputs,1)):
                with self.assertRaises(ValueError):
                    execute(MagicMock(),SimpleNamespace(shape=(1,2)),MagicMock(),candidate,{},lambda value:value,
                        mask_validated=validated)
            attend.assert_not_called()

    def test_single_chip_rejects_before_execution(self):
        with patch('dspark_layer_mesh.attention_partial') as attend:
            with self.assertRaises(ValueError):
                execute(MagicMock(),SimpleNamespace(shape=(1,1)),MagicMock(),
                    {name:object() for name in INPUTS},{},lambda value:value,mask_validated=True)
            attend.assert_not_called()


if __name__ == '__main__':
    unittest.main()
