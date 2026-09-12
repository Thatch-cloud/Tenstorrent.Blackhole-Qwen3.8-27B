from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from dspark_intake import TAPS
from dspark_pipeline import INPUTS, PARAMETERS, execute, flatten, pack_parameter
from dspark_layer import PHASES


class DSparkPipelineTests(unittest.TestCase):
    def test_complete_projection_and_backbone_inventory(self):
        self.assertEqual(len(PARAMETERS),58)
        self.assertEqual(len(set(PARAMETERS)),58)
        self.assertEqual(len(INPUTS),12)
        self.assertNotIn('context',INPUTS)
        for name in ('hidden_norm.weight','norm.weight','layers.4.mlp.down_proj.weight'):
            with patch('dspark_pipeline.pack_backbone') as pack:
                value = object()
                self.assertIs(pack_parameter(name,value),pack.return_value)
                pack.assert_called_once_with('norm.weight' if name=='hidden_norm.weight' else name,value)

    def test_actual_fc_partials_feed_norm_then_complete_device_backbone(self):
        runtime,mesh,collectives,retain = MagicMock(),SimpleNamespace(shape=(1,2)),object(),object()
        retain = lambda value:value
        inputs = {name:object() for name in INPUTS}
        original = dict(inputs)
        parameters = {name:object() for name in PARAMETERS}
        layers = object()
        projected = dict(joined=object(),partial=object())
        reduced = {name:object() for name in ('sum','narrowed','unweighted_norm','context')}
        first,second,backbone = object(),object(),object()
        with patch('dspark_pipeline.project',return_value=projected) as project, \
                patch('dspark_pipeline.gather_partials',return_value=(first,second)) as gather, \
                patch('dspark_pipeline.normalize_partials',return_value=reduced) as normalize, \
                patch('dspark_pipeline.backbone',return_value=backbone) as chain:
            result = execute(runtime,mesh,collectives,inputs,parameters,layers,retain,mask_validated=True)
        project.assert_called_once_with(runtime,{tap:inputs['feature_'+str(tap)] for tap in TAPS},parameters['fc.weight'],retain)
        gather.assert_called_once_with(runtime,mesh,collectives,projected['partial'],retain)
        normalize.assert_called_once_with(runtime,first,second,parameters['hidden_norm.weight'],retain,composed_norm=True)
        self.assertIs(chain.call_args.args[3]['context'],reduced['context'])
        self.assertIs(chain.call_args.args[3]['noise'],inputs['noise'])
        self.assertEqual(chain.call_args.args[4:],(layers,parameters['norm.weight'],retain))
        self.assertEqual(result,dict(projection={**projected,**reduced},backbone=backbone))
        self.assertEqual(inputs,original)
        self.assertEqual(runtime.mock_calls,[])

    def test_missing_inputs_parameters_or_wrong_mesh_reject_before_execution(self):
        inputs = {name:object() for name in INPUTS}
        parameters = {name:object() for name in PARAMETERS}
        for values,weights,shape,valid in (({},parameters,(1,2),True),(inputs,{},(1,2),True),
                (inputs,parameters,(1,1),True),(inputs,parameters,(1,2),False)):
            with patch('dspark_pipeline.project') as project:
                with self.assertRaises(ValueError):
                    execute(MagicMock(),SimpleNamespace(shape=shape),object(),values,weights,[],lambda value:value,mask_validated=valid)
                project.assert_not_called()

    def test_every_projection_and_all_five_layer_stages_are_retained(self):
        projection = {name:object() for name in ('joined','partial','sum','narrowed','unweighted_norm','context')}
        backbone = dict(layers=tuple({phase:{stage:object() for stage in stages} for phase,stages in PHASES.items()}
            for layer in range(5)),final_norm=object())
        result = dict(projection=projection,backbone=backbone)
        stages = flatten(result)
        self.assertEqual(len(stages),137)
        self.assertIs(stages['projection.context'],projection['context'])
        self.assertIs(stages['-1.final.final_norm'],backbone['final_norm'])
        self.assertIs(stages['4.finish.output'],backbone['layers'][4]['finish']['output'])
        del projection['partial']
        with self.assertRaises(ValueError):
            flatten(result)

    def test_fc_rejects_incomplete_or_wrong_precision_weights(self):
        import torch

        for value in (torch.ones(5120,dtype=torch.bfloat16),torch.ones(1).expand(5120,25600)):
            with self.assertRaises(ValueError):
                pack_parameter('fc.weight',value)


if __name__=='__main__':
    unittest.main()
