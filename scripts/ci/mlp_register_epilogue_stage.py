"""Stage a bounded numerical test; no hardware or serving route."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_recipe_context import REVISION, replace_once
from frozen_mlp_buffer_trial import adapt_probe
from mlp_register_epilogue import adapt_projection


def diagnostic_probe(source):
    anchor = "                    raise AssertionError(f'Fused multi-row output differs: rows={rows}, chip={chip}, mismatches={int((observed != reference).sum())}')"
    return replace_once(source, anchor, '''                    mismatch = observed != reference
                    indices = mismatch.flatten().nonzero().flatten()[:32]
                    report['numerical_failure'] = dict(rows=rows, chip=chip,
                        mismatches=int(mismatch.sum()), elements=reference.numel(),
                        finite=bool(torch.isfinite(observed).all()),
                        max_abs=float((observed.float() - reference.float()).abs().max()),
                        per_row=mismatch.reshape(rows, -1).sum(dim=1).tolist(),
                        columns=(indices % reference.shape[-1]).tolist(),
                        reference=reference.flatten()[indices].float().tolist(),
                        actual=observed.flatten()[indices].float().tolist())
                    raise AssertionError(f'Fused multi-row output differs: rows={rows}, chip={chip}, mismatches={int(mismatch.sum())}')''')


def stage(checkout, manifest, *, diagnose_activation=False, diagnose_rounding=False, nearest_away=False):
    if manifest.exists():
        raise ValueError('Fresh candidate manifest required')
    scripts = checkout / 'scripts/ci'
    originals = {}
    for name in ('fused_1d.py', 'fused-batch-probe.py'):
        source = subprocess.check_output(['git', '-C', str(checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text(encoding='utf-8') != source:
            raise ValueError('Exact frozen source required: ' + name)
        originals[name] = source
    payloads = {'fused_1d.py': adapt_projection(originals['fused_1d.py'],
            diagnose_activation=diagnose_activation, diagnose_rounding=diagnose_rounding, nearest_away=nearest_away),
        'fused-batch-probe.py': diagnostic_probe(replace_once(adapt_probe(originals['fused-batch-probe.py']),
            'T16 buffering only; other row widths and performance unqualified',
            'T16 MLP epilogue numerical diagnostic; no performance qualification')),
        'mlp_register_epilogue.py': Path(__file__).with_name('mlp_register_epilogue.py').read_text(encoding='utf-8')}
    if nearest_away:
        payloads['mlp_rounding_policy.py'] = Path(__file__).with_name('mlp_rounding_policy.py').read_text(encoding='utf-8')
    payloads['simulator-suite.sh'] = replace_once((scripts / 'simulator-suite.sh').read_text(encoding='utf-8'),
        'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py',
        'timeout -k 15 510 python3 -u /experiment-scripts/ci/fused-batch-probe.py')
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(source.encode()).hexdigest() for name, source in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        token_rows=16, pairs_per_worker=3, readers_changed=False, buffers_changed=False,
        activation_diagnostic=diagnose_activation,
        rounding_diagnostic=diagnose_rounding,
        nearest_away_hypothesis=nearest_away,
        accumulation_changed=False, activation_processor='MATH instead of PACK',
        bf16_rounding='native pack' if diagnose_activation else 'explicit SFPU before unchanged BF16 product',
        simulator_qualified=False, hardware_qualified=False, performance_qualified=False), indent=2) + '\n',
        encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument('--diagnose-activation', action='store_true')
    diagnostic.add_argument('--diagnose-rounding', action='store_true')
    diagnostic.add_argument('--nearest-away', action='store_true')
    options = parser.parse_args()
    stage(options.checkout, options.manifest, diagnose_activation=options.diagnose_activation,
        diagnose_rounding=options.diagnose_rounding, nearest_away=options.nearest_away)


if __name__ == '__main__':
    main()
