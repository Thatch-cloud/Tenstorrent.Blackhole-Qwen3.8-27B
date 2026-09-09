from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import torch

from dspark_backbone_reference import rms_norm
from dspark_intake import TAPS
from dspark_projection import difference, norm_references, normalize_partials, project, reference_metadata


def tensor(shape, dtype='bf16'):
    return SimpleNamespace(shape=shape, dtype=dtype, layout='tile', memory_config=lambda:'dram')


def operations():
    return SimpleNamespace(bfloat16='bf16',float32='fp32',TILE_LAYOUT='tile',DRAM_MEMORY_CONFIG='dram',
        MathFidelity=SimpleNamespace(HiFi4='hifi4'), WormholeComputeKernelConfig=MagicMock(),
        MatmulMultiCoreReuseMultiCast1DProgramConfig=MagicMock(),concat=MagicMock(),matmul=MagicMock(),
        add=MagicMock(),typecast=MagicMock(),rms_norm=MagicMock(),mul=MagicMock())


class DSparkProjectionTests(unittest.TestCase):
    def test_projection_retains_complete_named_tap_order_and_precision(self):
        runtime = operations()
        features = {layer:tensor((1,1,32,2560)) for layer in reversed(TAPS)}
        weight = tensor((1,1,12800,5120))
        owned = []
        def retain(value):
            owned.append(value)
            return value
        result = project(runtime,features,weight,retain)
        self.assertEqual(runtime.concat.call_args.args[0], tuple(features[layer] for layer in TAPS))
        self.assertIs(runtime.matmul.call_args.args[0],runtime.concat.return_value)
        self.assertIs(runtime.matmul.call_args.args[1],weight)
        self.assertEqual(runtime.matmul.call_args.kwargs['dtype'],'fp32')
        self.assertEqual(runtime.MatmulMultiCoreReuseMultiCast1DProgramConfig.call_args.kwargs['in0_block_w'],8)
        self.assertEqual(runtime.WormholeComputeKernelConfig.call_args.kwargs,
            dict(math_fidelity='hifi4',math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=False))
        self.assertEqual(owned,list(result.values()))

    def test_incomplete_or_wrongly_sharded_features_reject_before_dispatch(self):
        runtime = operations()
        features = {layer:tensor((1,1,32,2560)) for layer in TAPS}
        for changed,weight in (({**features,TAPS[0]:tensor((1,1,32,5120))},tensor((1,1,12800,5120))),
                ({layer:value for layer,value in features.items() if layer != TAPS[0]},tensor((1,1,12800,5120))),
                (features,tensor((1,1,12800,32))), (features,tensor((1,1,12800,5120),'fp32'))):
            with self.assertRaises(ValueError):
                project(runtime,changed,weight,lambda value:value)
        runtime.concat.assert_not_called()
        runtime.matmul.assert_not_called()

    def test_normalization_rounds_before_gamma_without_fused_weight_argument(self):
        runtime = operations()
        first,second = [tensor((1,1,32,5120),'fp32') for unused in range(2)]
        gamma = tensor((1,1,1,5120))
        runtime.rms_norm.return_value = tensor((1,1,32,5120))
        result = normalize_partials(runtime,first,second,gamma,lambda value:value)
        self.assertEqual(runtime.add.call_args.args,(first,second))
        self.assertEqual(runtime.typecast.call_args.args,(runtime.add.return_value,'bf16'))
        self.assertIs(runtime.rms_norm.call_args.args[0],runtime.typecast.return_value)
        self.assertIsNone(runtime.rms_norm.call_args.kwargs['weight'])
        self.assertEqual(runtime.rms_norm.call_args.kwargs['epsilon'],1e-6)
        self.assertEqual(runtime.mul.call_args.args,(runtime.rms_norm.return_value,gamma))
        self.assertIs(result['context'],runtime.mul.return_value)

    def test_rounding_reference_matches_frozen_backbone_and_detects_fused_policy(self):
        generator = torch.Generator().manual_seed(38)
        first,second = [torch.randn((1,1,32,5120),generator=generator) for unused in range(2)]
        gamma = torch.randn(5120,generator=generator).bfloat16()
        result = norm_references(first,second,gamma)
        self.assertTrue(torch.equal(result['native_input_norm'],rms_norm((first+second).bfloat16(),gamma)))
        with self.assertRaises(ValueError):
            norm_references(first,second,torch.ones_like(gamma))

    def test_exact_and_numerical_diagnostics_cannot_hide_drift_or_nonfinite_values(self):
        expected = torch.ones(32,dtype=torch.bfloat16)
        actual = expected.clone()
        actual[0] = 1.0078125
        self.assertTrue(difference(actual,expected,exact=False)['passed'])
        self.assertFalse(difference(actual,expected,exact=True)['passed'])
        actual[0] = 2
        self.assertEqual(difference(actual,expected,exact=False)['failed_elements'],1)
        actual[0] = float('nan')
        with self.assertRaises(ValueError):
            difference(actual,expected,exact=False)

    def test_frozen_reference_metadata_is_available_without_local_output_tensors(self):
        result = reference_metadata()
        self.assertEqual(len(result['outputs_sha256']),64)
        self.assertEqual(len(result['tensor_sha256']),62)
        self.assertIn('fc.weight',result['tensor_sha256'])


if __name__ == '__main__':
    unittest.main()
