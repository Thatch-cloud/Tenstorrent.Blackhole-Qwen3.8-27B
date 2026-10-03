import unittest

import torch

from dspark_backbone_reference import rms_norm
from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_projection import POLICY, TOLERANCE
from dspark_projection_diagnostic import analyse_case, validate_capture


class DSparkProjectionDiagnosticTests(unittest.TestCase):
    def test_attributes_projection_and_normalization_separately(self):
        full = torch.tensor([[[[1.,2.,3.,4.]]]],dtype=torch.bfloat16)
        gamma = torch.tensor([1.,2.,3.,4.],dtype=torch.bfloat16)
        context = rms_norm(full,gamma)
        for error in ('none','projection','normalization'):
            narrowed = full.clone()
            if error == 'projection':
                narrowed[...,0] += 1.
            normalized = rms_norm(narrowed,torch.ones_like(gamma))
            if error == 'normalization':
                normalized[...,0] += 1.
            observed = dict(sum=narrowed.float(),narrowed=narrowed,
                unweighted_norm=normalized,context=normalized*gamma)
            result = analyse_case(observed,context,gamma,full,full)
            comparisons = result['comparisons']
            self.assertEqual(comparisons['native_projection_cpu_norm_vs_backbone']['passed'],error!='projection')
            self.assertEqual(comparisons['native_context_vs_own_input']['passed'],error!='normalization')
            self.assertEqual(bool(result['failures']),error!='none')
            if result['failures']:
                self.assertEqual(result['failures'][0]['coordinate'],[0,0,0,0])
                self.assertIn('cpu_projection_bits',result['failures'][0])

    def test_corrupt_intermediates_cannot_support_attribution(self):
        full = torch.tensor([[[[1.,2.,3.,4.]]]],dtype=torch.bfloat16)
        gamma = torch.ones(4,dtype=torch.bfloat16)
        context = rms_norm(full,gamma)
        valid = dict(sum=full.float(),narrowed=full,unweighted_norm=context,context=context)
        for name in valid:
            observed = {stage:value.clone() for stage,value in valid.items()}
            observed[name][...,0] += 2.
            with self.assertRaises(ValueError):
                analyse_case(observed,context,gamma,full,full)
        with self.assertRaises(ValueError):
            analyse_case(valid,context+1,gamma,full,full)

    def test_failed_closed_capture_is_allowed_but_never_forged_or_live_evidence(self):
        reference,sources,native = {'cpu':'sha'},{'probe':'sha'},{'runtime':'sha'}
        report = dict(mode='eager_diagnostic',backend='simulator',stage='failed',closed_cleanly=True,
            checkpoint_closed=True,checkpoint_sha256=CHECKPOINT_SHA256,policy=POLICY,tolerance=TOLERANCE,
            reference=reference,sources=sources,sources_after=sources,native_sources=native,native_sources_after=native,
            fabric_tested=False,full_pipeline_captured=False,target_integrated=False,eligible_for_hardware=False)
        payload = dict(checkpoint_sha256=CHECKPOINT_SHA256,policy=POLICY,reference=reference,sources=sources,native_sources=native)
        validate_capture(report,payload,sources=sources,reference=reference)
        for key,value in (('closed_cleanly',False),('checkpoint_closed',False),('stage','tail_eager_1'),
                ('eligible_for_hardware',True),('sources_after',{}),('native_sources_after',{}),('cleanup_error','failed')):
            with self.assertRaises(ValueError):
                validate_capture({**report,key:value},payload,sources=sources,reference=reference)
        for key in payload:
            with self.assertRaises(ValueError):
                validate_capture(report,{**payload,key:None},sources=sources,reference=reference)


if __name__ == '__main__':
    unittest.main()
