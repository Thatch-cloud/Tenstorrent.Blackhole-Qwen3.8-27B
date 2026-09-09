import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_intake import TAPS
from dspark_projection import EAGER_STAGES, EXACT_STAGES, HANDOFF, POLICY, PROJECTION_STAGES, TAIL_STAGES, TOLERANCE
from dspark_projection_gate import qualify


REFERENCE = dict(tensor_sha256={'fc.weight':'fc-sha','hidden_norm.weight':'norm-sha'})


def fixture():
    return dict(passed=True,closed_cleanly=True,checkpoint_closed=True,stage='complete',backend='simulator',mode='matrix',
        checkpoint_sha256=CHECKPOINT_SHA256,reference=REFERENCE,policy=POLICY,tolerance=TOLERANCE,handoff=HANDOFF,
        taps=list(TAPS),packer_compat=False,rows=32,input_width=25600,output_width=5120,fixtures=2,
        fabric_tested=False,full_pipeline_captured=False,target_integrated=False,eligible_for_hardware=False,
        parameter_sha256=REFERENCE['tensor_sha256'],sources={'probe':'sha'},sources_after={'probe':'sha'},
        native_sources={'runtime':'sha'},native_sources_after={'runtime':'sha'},
        eager_checks=[dict(pattern=pattern,chip=chip,stage=stage,passed=True,exact_required=stage in EXACT_STAGES,
            bitwise_exact=True,failed_elements=0,max_abs=0.) for pattern in range(2) for chip in range(2) for stage in EAGER_STAGES],
        replay_checks=[dict(phase=phase,ordinal=ordinal,pattern=pattern,chip=chip,stage=stage,exact=True,bindings_stable=True)
            for phase,stages in (('projection',PROJECTION_STAGES),('tail',TAIL_STAGES))
            for ordinal,pattern in enumerate((0,1,0)) for chip in range(2) for stage in stages],
        input_checks=[dict(phase=phase,mode=mode,ordinal=ordinal,pattern=pattern,chip=chip,tensor=index,exact=True)
            for phase,count in (('projection',5),('tail',3)) for mode,patterns in (('eager',(0,1)),('replay',(0,1,0)))
            for ordinal,pattern in enumerate(patterns) for chip in range(2) for index in range(count)],
        parameter_checks=[dict(phase=phase,chip=chip,tensor=name,exact=True)
            for phase in ('before','after') for chip in range(2) for name in ('weight','gamma')],
        stale_controls=[dict(phase=phase,chip=chip,detected=True) for phase in ('projection','tail') for chip in range(2)],
        rounding_controls=[dict(pattern=pattern,distinguished=True) for pattern in range(2)])


def check(report, **kwargs):
    return qualify(report,sources={'probe':'sha'},native={'runtime':'sha'},reference=REFERENCE,
        exit_status=kwargs.pop('exit_status','0'),**kwargs)


class DSparkProjectionGateTests(unittest.TestCase):
    def test_complete_matrix_still_does_not_qualify_fabric_or_integration(self):
        result = check(fixture())
        self.assertEqual(result['checks'],162)
        self.assertFalse(result['fabric_tested'])
        self.assertFalse(result['full_pipeline_captured'])
        self.assertFalse(result['eligible_for_hardware'])

    def test_every_audit_coordinate_is_mandatory(self):
        for name in ('eager_checks','replay_checks','input_checks','parameter_checks','stale_controls','rounding_controls'):
            for duplicate in (False,True):
                report = fixture()
                if duplicate:
                    report[name].append(copy.deepcopy(report[name][0]))
                else:
                    report[name].pop()
                with self.assertRaises(ValueError):
                    check(report)

    def test_accuracy_ownership_and_handoff_scope_cannot_be_relabelled(self):
        for field,value in (('fabric_tested',True),('full_pipeline_captured',True),('eligible_for_hardware',True),
                ('target_integrated',True),('checkpoint_closed',False),('closed_cleanly',False),('stage','tail_capture'),
                ('handoff','device collective'),('policy','fused RMS'),('tolerance',{'rtol':1.,'atol':1.}),
                ('sources_after',{}),('native_sources_after',{}),('parameter_sha256',{}),('reference',{}),('rows',True),
                ('mode','eager_diagnostic'),('mode',None)):
            report = fixture()
            report[field] = value
            with self.assertRaises(ValueError):
                check(report)
        for field,value in (('passed',False),('exact_required',False),('bitwise_exact',False),
                ('failed_elements',1),('failed_elements',False),('max_abs',float('nan'))):
            report = fixture()
            report['eager_checks'][0][field] = value
            with self.assertRaises(ValueError):
                check(report)
        with self.assertRaises(ValueError):
            check(fixture(),exit_status='124')
        report = fixture()
        report['packer_compat'] = True
        with self.assertRaises(ValueError):
            check(report)
        self.assertTrue(check(report,packer_compat=True)['passed'])

    def test_eager_only_requires_non_overwriting_operand_capture(self):
        import hashlib
        import importlib.util
        import torch

        path = Path(__file__).with_name('dspark-projection-probe.py')
        spec = importlib.util.spec_from_file_location('dspark_projection_capture_test',path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'observed.pt'
            values = dict(features={5:torch.tensor([1.],dtype=torch.bfloat16)},
                contexts={0:torch.tensor([2.])}, gamma=torch.tensor([3.]),
                eager={('tail',1):{'context':[torch.tensor([4.])]}})
            metadata = probe.save_operands(destination,fixture(),**values)
            payload = torch.load(destination,weights_only=True,map_location='cpu')
            self.assertEqual(metadata['sha256'],hashlib.sha256(destination.read_bytes()).hexdigest())
            self.assertTrue(torch.equal(payload['eager']['tail',1]['context'][0],values['eager']['tail',1]['context'][0]))
            self.assertEqual(payload['reference'],REFERENCE)
            self.assertEqual(payload['sources'],fixture()['sources'])
            with self.assertRaises(FileExistsError):
                probe.save_operands(destination,fixture(),**values)
            result = subprocess.run([sys.executable,'-B',str(path),'--output',str(Path(directory)/'report.json'),
                '--checkpoint','not-opened','--cpu-outputs','not-opened','--eager-only'],
                text=True,capture_output=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('require --save-operands',result.stderr)

    def test_probe_rejects_hardware_before_checkpoint_or_runtime_import(self):
        environment = {key:value for key,value in os.environ.items()
            if key not in ('TT_METAL_SIMULATOR','TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1',QWEN_CARDS_ALLOCATED='1')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'unexpected.json'
            result = subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('dspark-projection-probe.py')),
                '--output',str(output),'--checkpoint','not-opened','--cpu-outputs','not-opened'],
                env=environment,text=True,capture_output=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Simulator required',result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
