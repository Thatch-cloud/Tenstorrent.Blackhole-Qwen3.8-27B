from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from dspark_backbone_mesh import PARAMETERS, execute, flatten, pack_parameter
from dspark_layer import PHASES, SPECIFICATIONS
from dspark_layer_mesh import INPUTS


class DSparkBackboneMeshTests(unittest.TestCase):
    def test_complete_parameter_inventory_and_sharding_delegation(self):
        self.assertEqual(len(PARAMETERS),56)
        value,result = object(),object()
        with patch('dspark_backbone_mesh.pack_weight',return_value=result) as pack:
            self.assertIs(pack_parameter('layers.4.mlp.down_proj.weight',value),result)
            pack.assert_called_once_with('mlp.down_proj.weight',value)
            with self.assertRaises(ValueError):
                pack_parameter('layers.5.mlp.down_proj.weight',value)

    def test_final_normalization_is_owned_replicated_and_bounded(self):
        import torch

        value = torch.ones(5120,dtype=torch.bfloat16)
        packed,sharded = pack_parameter('norm.weight',value)
        self.assertEqual(tuple(packed.shape),(1,1,1,5120))
        self.assertIs(sharded,False)
        self.assertNotEqual(packed.data_ptr(),value.data_ptr())
        for invalid in (value[:2560],value.float(),torch.full_like(value,float('nan'))):
            with self.assertRaises(ValueError):
                pack_parameter('norm.weight',invalid)

    def test_all_131_stage_bindings_are_required(self):
        result = dict(layers=tuple({phase:{stage:object() for stage in stages} for phase,stages in PHASES.items()}
            for layer in range(5)),final_norm=object())
        values = flatten(result)
        self.assertEqual(len(values),131)
        self.assertIs(values[4,'finish','output'],result['layers'][4]['finish']['output'])
        self.assertIs(values[-1,'final','final_norm'],result['final_norm'])
        with self.assertRaises(ValueError):
            flatten(dict(result,layers=result['layers'][:4]))
        del result['layers'][3]['mlp']['down_partial']
        with self.assertRaises(ValueError):
            flatten(result)

    def test_five_layers_chain_without_mutating_borrowed_inputs(self):
        runtime,collectives,retain = MagicMock(),MagicMock(),MagicMock()
        mesh = SimpleNamespace(shape=(1,2))
        inputs = {name:object() for name in INPUTS}
        original = dict(inputs)
        weights = [{name:object() for name in SPECIFICATIONS} for layer in range(5)]
        gamma,normalized = object(),object()
        outputs = [dict(finish={'output':object()}) for layer in range(5)]
        with patch('dspark_backbone_mesh.execute_layer',side_effect=outputs) as layer, \
                patch('dspark_backbone_mesh.norm',return_value=normalized) as norm:
            result = execute(runtime,mesh,collectives,inputs,weights,gamma,retain,mask_validated=True)
        self.assertEqual(result,dict(layers=tuple(outputs),final_norm=normalized))
        self.assertEqual(inputs,original)
        self.assertEqual(layer.call_count,5)
        for index,call in enumerate(layer.call_args_list):
            self.assertEqual(call.args[:3],(runtime,mesh,collectives))
            self.assertEqual(call.args[4:],(weights[index],retain))
            self.assertIs(call.kwargs['mask_validated'],True)
            borrowed = call.args[3]
            self.assertIs(borrowed['noise'],inputs['noise'] if index==0 else outputs[index-1]['finish']['output'])
            for name in INPUTS:
                if name!='noise':
                    self.assertIs(borrowed[name],inputs[name])
        norm.assert_called_once_with(runtime,outputs[-1]['finish']['output'],gamma,retain)
        self.assertEqual(runtime.mock_calls,[])

    def test_incomplete_layers_or_unvalidated_inputs_reject_before_execution(self):
        inputs = {name:object() for name in INPUTS}
        weights = [{name:object() for name in SPECIFICATIONS} for layer in range(5)]
        with patch('dspark_backbone_mesh.execute_layer') as layer:
            for candidates,borrowed,validated in ((weights[:4],inputs,True),(weights+[{}],inputs,True),
                    (weights[:4]+[{}],inputs,True),(weights,{},True),(weights,inputs,False),(weights,inputs,1)):
                with self.assertRaises(ValueError):
                    execute(MagicMock(),SimpleNamespace(shape=(1,2)),MagicMock(),borrowed,candidates,
                        object(),lambda value:value,mask_validated=validated)
            layer.assert_not_called()

    def test_wrong_mesh_rejects_before_execution(self):
        with patch('dspark_backbone_mesh.execute_layer') as layer:
            with self.assertRaises(ValueError):
                execute(MagicMock(),SimpleNamespace(shape=(1,1)),MagicMock(),
                    {name:object() for name in INPUTS},[{name:object() for name in SPECIFICATIONS} for index in range(5)],
                    object(),lambda value:value,mask_validated=True)
            layer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
