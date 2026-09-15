"""Weight-free full-history 64K split-K numerical and changed-input replay screen."""

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from dspark_hardware_gate import digest
from dspark_splitk_linearity import KINDS, diagnostic_scope, summarize
from dspark_ladder_backend import require_backend, require_packer_mode
from dspark_ladder_fixtures import fixture_probe
from dspark_ladder_geometry import geometry
from dspark_splitk_attention import adapter
import dspark_splitk_attention
from dspark_splitk_hardware_build import REPORT, validate_build
from dspark_splitk_hardware_scope import HEADER, kernel_scope
from native_draft_sdpa import audit_active_kernel, precise_draft_kernel


def main():
    require_backend(os.environ, hardware=True, device_present=Path('/dev/tenstorrent').exists())
    if os.environ.get('QWEN_SPLITK_ATTENTION') != '1' or '--hardware' not in sys.argv:
        raise ValueError('Explicit isolated split-K hardware probe required')
    directory = Path(__file__).resolve().parent
    root = Path(os.environ['TT_METAL_HOME'])
    simulator = directory / 'dspark-splitk-simulator.json'
    build = validate_build(root, directory, REPORT, simulator)
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh hardware probe output required')
    fixture = geometry(65536, 1024)
    original_execute = dspark_splitk_attention.execute_folded

    def execute(*args, **kwargs):
        kwargs.update(key_chunk_size=128, max_cores_per_head=8, stripe_keys=False, fp32_dest_acc=True)
        return original_execute(*args, **kwargs)

    with fixture_probe(65536) as probe:
        def fingerprints(runtime, *, packer_compat=False, precise_native=False):
            require_packer_mode(hardware=True, packer_compat=packer_compat, precise_native=precise_native)
            validate_build(runtime, directory, REPORT, simulator)
            audit_active_kernel(runtime)
            if digest(runtime / probe.NATIVE.PACKER) != probe.NATIVE.ORIGINAL_PACKER:
                raise ValueError('Stock hardware packer required')
            sources = {str(path.relative_to(runtime)): digest(path)
                for path in (runtime / 'ttnn/cpp/ttnn/operations/transformer').rglob('*')
                if path.is_file() and path.suffix in ('.cpp', '.hpp', '.h')}
            sources.update({name: digest(runtime / name) for name in (*build['binaries'], probe.NATIVE.PACKER)})
            return sources

        with precise_draft_kernel(root), kernel_scope(root) as kernel, diagnostic_scope(), \
                patch.dict(os.environ, QWEN_PRECISE_DRAFT_ACTIVE='1', QWEN_SPLITK_FP32_INTERMEDIATES='1'), \
                patch.object(dspark_splitk_attention, 'execute_folded', execute), \
                patch.object(probe.NATIVE, 'fingerprints', fingerprints), \
                patch.object(probe, 'execute', wraps=adapter(65536)) as execution:
            try:
                probe.run()
            finally:
                if output.exists():
                    report = json.loads(output.read_text())
                    if report['passed'] and execution.call_count != len(KINDS) + 3:
                        report['passed'] = False
                        report['error'] = 'Diagnostic calls, two eager calls and one capture required'
                    report.update(scope=__doc__, splitk_factory=build, splitk_kernel=kernel,
                        linearity_diagnostics=summarize(report.get('value_diagnostics', [])),
                        splitk_execution_calls=execution.call_count,
                        expected_execution_calls=len(KINDS) + 3, key_chunk_size=128,
                        max_cores_per_head=8, native_padded_keys=fixture['native_keys'],
                        local_denominator_arithmetic='sfpu-fp32-multiply-add-and-copy',
                        tree_denominator_arithmetic='sfpu-fp32-two-products-add-and-transport',
                        local_numerator_add='native-fpu',
                        correction_factor_rounding='explicit-fp32-to-bf16-rne',
                        added_masked_poison_rows=fixture['extra_masked_keys'],
                        performance_qualified=False, full_request_qualified=False,
                        wrapper_sources={name: digest(directory / name) for name in
                            (Path(__file__).name, 'dspark_splitk_hardware_scope.py', 'dspark_splitk_linearity.py')})
                    output.write_text(json.dumps(report, indent=2) + '\n')
            if not report['passed']:
                raise ValueError('Full-history split-K hardware screen failed')
        if digest(root / HEADER) != kernel['source_before']:
            raise ValueError('Hardware decode source was not restored')


if __name__ == '__main__':
    main()
