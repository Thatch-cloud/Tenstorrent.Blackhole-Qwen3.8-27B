"""Stage bounded T32 recurrence/scatter replay on the frozen simulator base."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_recipe_context import REVISION, replace_once
from gdn_shared_qk_t32_adapter import adapt_probe, payloads


def stage(checkout, manifest):
    if manifest.exists():
        raise ValueError('Fresh T32 GDN manifest required')
    directory = Path(__file__).parent
    scripts = checkout / 'scripts/ci'
    originals = {}
    for name in ('gdn_shared_qk_program.py', 'gdn_shared_qk_pipeline.py', 'gdn-shared-recurrence-probe.py'):
        source = subprocess.check_output(['git', '-C', str(checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text() != source:
            raise ValueError('Exact frozen GDN input required: ' + name)
        originals[name] = source
    originals['shared_qk_norm_scatter.py'] = (directory / 'shared_qk_norm_scatter.py').read_text()
    sources = payloads(originals)
    sources['gdn-shared-recurrence-probe.py'] = adapt_probe(originals['gdn-shared-recurrence-probe.py'])
    for name in ('gdn_shared_qk_t32_adapter.py', 'gdn_norm_scatter.py'):
        sources[name] = (directory / name).read_text()
    sources['simulator-suite.sh'] = replace_once((scripts / 'simulator-suite.sh').read_text(),
        'frozen_sim_phase.py --phase probe --seconds 510 ',
        'frozen_sim_phase.py --phase probe --seconds 420 ')
    for name, source in sources.items():
        (scripts / name).write_bytes(source.encode())
    manifest.write_text(json.dumps(dict(rows=32, state_math_changed=False, tile_indexing_changed=False,
        simulator_qualified=False, hardware_qualified=False, performance_qualified=False,
        before={name: hashlib.sha256(source.encode()).hexdigest() for name, source in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    stage(options.checkout, options.manifest)


if __name__ == '__main__':
    main()
