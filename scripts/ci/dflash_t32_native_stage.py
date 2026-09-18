"""Stage a weight-free T32 replay probe without changing the admitted T16 files."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once


def payloads(directory):
    directory = Path(directory)
    probe = (directory / 'dflash-t16-native-attention-probe.py').read_text()
    gate = (directory / 'dflash_t16_native_attention_gate.py').read_text()
    probe = probe.replace('dflash_t16_native_attention', 'dflash_t32_native_attention').replace('T16', 'T32')
    probe = probe.replace('block_rows=16', 'block_rows=32')
    probe = replace_once(probe, "    patterns[1][3][..., :16, :5] = float('-inf')",
        "    for values in patterns:\n        values[3][..., :, :3] = float('-inf')\n"
        "    patterns[1][3][..., :, :5] = float('-inf')")
    gate = gate.replace('dflash_t16_native_attention', 'dflash_t32_native_attention')
    gate = gate.replace('dflash-t16-native-attention-probe.py', 'dflash-t32-native-attention-probe.py')
    gate = replace_once(gate, "SOURCES = ('dflash_t32_native_attention.py',",
        "SOURCES = ('dflash_t16_native_attention.py', 'dflash_t32_native_attention.py',")
    gate = replace_once(gate, "report['block_rows'] != 16", "report['block_rows'] != 32")
    suite = (directory / 'simulator-suite.sh').read_text()
    suite = suite.replace('dflash-t16-native-attention-probe.py', 'dflash-t32-native-attention-probe.py')
    suite = suite.replace('dflash_t16_native_attention_gate', 'dflash_t32_native_attention_gate')
    suite = suite.replace('/experiment/results/dflash-t16-', '/experiment/results/dflash-t32-')
    result = {'dflash-t32-native-attention-probe.py': probe,
        'dflash_t32_native_attention_gate.py': gate, 'simulator-suite.sh': suite}
    for name, source in result.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    directory = Path(__file__).parent
    if options.manifest.exists() or (directory / 'dflash_t32_native_attention_gate.py').exists():
        raise ValueError('Fresh T32 simulator staging required')
    sources = payloads(directory)
    for name, source in sources.items():
        (directory / name).write_bytes(source.encode())
    options.manifest.write_text(json.dumps(dict(block_rows=32, learned_weights=False,
        hardware_qualified=False, performance_qualified=False,
        sources={name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}), indent=2) + '\n')


if __name__ == '__main__':
    main()
