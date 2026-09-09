import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_layer import ADDITIONAL, EXACT, INPUT_COUNTS, PHASES, POLICY, WIDE_POLICY, REPLAY_CASES, SPECIFICATIONS
from dspark_layer_gate import qualify
from dspark_layer_reference import CASES, PROJECTION_OPERANDS_SHA256, PROJECTION_REPORT_SHA256
from dspark_projection import TOLERANCE


REFERENCE = dict(tensor_sha256={'layers.0.'+name:name+'-sha' for name in SPECIFICATIONS})


def fixture():
    return dict(passed=True,closed_cleanly=True,checkpoint_closed=True,stage='complete',backend='simulator',mode='matrix',
        checkpoint_sha256=CHECKPOINT_SHA256,layer=0,context_rows=32,proposal_rows=7,cases=[list(value) for value in CASES],
        policy=POLICY,tolerance=TOLERANCE,reference=REFERENCE,precise_native=True,packer_compat=True,composed_attention=False,
        projection_report_sha256=PROJECTION_REPORT_SHA256,projection_operands_sha256=PROJECTION_OPERANDS_SHA256,
        sources={'probe':'sha'},sources_after={'probe':'sha'},native_sources={'runtime':'sha'},native_sources_after={'runtime':'sha'},
        parameter_sha256={name:REFERENCE['tensor_sha256']['layers.0.'+name] for name in SPECIFICATIONS},
        fabric_tested=False,full_pipeline_captured=False,target_integrated=False,eligible_for_hardware=False,
        cpu_checks=[dict(case=case,stage=stage,exact=True) for case in range(3) for stage in ('attention_residual','output')],
        eager_checks=[dict(phase=phase,case=case,chip=chip,stage=stage,passed=True,
            exact_required=(phase,stage) in EXACT,bitwise_exact=True,failed_elements=0,max_abs=0.)
            for phase,stages in PHASES.items() for case in range(3) for chip in range(2) for stage in (*stages,*ADDITIONAL[phase])],
        replay_checks=[dict(phase=phase,ordinal=ordinal,case=case,chip=chip,stage=stage,exact=True,bindings_stable=True)
            for phase,stages in PHASES.items() for ordinal,case in enumerate(REPLAY_CASES) for chip in range(2) for stage in stages],
        input_checks=[dict(phase=phase,mode=mode,ordinal=ordinal,case=case,chip=chip,tensor=index,exact=True)
            for phase,count in INPUT_COUNTS.items() for mode,cases in (('eager',range(3)),('replay',REPLAY_CASES))
            for ordinal,case in enumerate(cases) for chip in range(2) for index in range(count)],
        parameter_checks=[dict(phase=phase,chip=chip,tensor=name,exact=True,bindings_stable=True)
            for phase in ('before','after') for chip in range(2) for name in SPECIFICATIONS],
        stale_controls=[dict(phase=phase,chip=chip,detected=True) for phase in PHASES for chip in range(2)],
        padding_checks=[dict(phase=phase,case=case,chip=chip,zero=True)
            for phase in ('attention','finish') for case in range(3) for chip in range(2)])


def check(report, exit_status='0'):
    return qualify(report,sources={'probe':'sha'},native={'runtime':'sha'},reference=REFERENCE,exit_status=exit_status)


class DSparkLayerGateTests(unittest.TestCase):
    def test_complete_layer_is_664_checks_not_fabric_or_hardware(self):
        result = check(fixture())
        self.assertEqual(result['checks'],664)
        self.assertEqual(result['counts']['eager_checks'],192)
        self.assertFalse(result['full_pipeline_captured'])
        self.assertFalse(result['eligible_for_hardware'])

    def test_all_coordinates_are_mandatory_and_unique(self):
        for name in ('cpu_checks','eager_checks','replay_checks','input_checks','parameter_checks','stale_controls','padding_checks'):
            for duplicate in (False,True):
                report = fixture()
                if duplicate:
                    report[name].append(copy.deepcopy(report[name][0]))
                else:
                    report[name].pop()
                with self.assertRaises(ValueError):
                    check(report)

    def test_composed_attention_preserves_all_664_gates_and_requires_explicit_policy(self):
        report = {**fixture(),'policy':WIDE_POLICY,'composed_attention':True,'precise_native':False,'packer_compat':False}
        arguments = dict(sources=report['sources'],native=report['native_sources'],reference=REFERENCE,
            exit_status='0',composed_attention=True)
        self.assertEqual(qualify(report,**arguments)['checks'],664)
        with self.assertRaises(ValueError):
            check(report)
        for key,value in (('composed_attention',False),('policy',POLICY),('precise_native',True),('packer_compat',True)):
            with self.assertRaises(ValueError):
                qualify({**report,key:value},**arguments)
        report['eager_checks'][0]['failed_elements'] = 1
        with self.assertRaises(ValueError):
            qualify(report,**arguments)

    def test_eager_only_missing_weights_and_changed_precision_cannot_qualify(self):
        for key,value in (('mode','eager_diagnostic'),('parameter_sha256',{}),('projection_report_sha256','wrong'),
                ('projection_operands_sha256','wrong'),('layer',1),('layer',False),('proposal_rows',15),('context_rows',4096),
                ('reference',{}),('sources_after',{}),('native_sources_after',{}),('closed_cleanly',False),('checkpoint_closed',False),
                ('policy','different'),('tolerance',{'atol':1.,'rtol':1.}),('fabric_tested',True),('full_pipeline_captured',True),
                ('target_integrated',True),('eligible_for_hardware',True),('precise_native',False),('packer_compat',False)):
            with self.assertRaises(ValueError):
                check({**fixture(),key:value})
        with self.assertRaises(ValueError):
            check(fixture(),'124')
        report = fixture()
        report['eager_checks'][0]['failed_elements'] = 1
        with self.assertRaises(ValueError):
            check(report)
        report = fixture()
        report['parameter_checks'][0]['bindings_stable'] = False
        with self.assertRaises(ValueError):
            check(report)

    def test_hardware_rejected_before_checkpoint_and_native_runtime_import(self):
        environment = {key:value for key,value in os.environ.items() if key not in ('TT_METAL_SIMULATOR','TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1',QWEN_CARDS_ALLOCATED='1')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'unexpected.json'
            arguments = [sys.executable,'-B',str(Path(__file__).with_name('dspark-layer-probe.py')),'--output',str(output)]
            for name in ('checkpoint','projection-report','projection-operands','cpu-outputs','config'):
                arguments.extend(['--'+name,'not-opened'])
            result = subprocess.run(arguments,env=environment,text=True,capture_output=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Simulator required',result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
