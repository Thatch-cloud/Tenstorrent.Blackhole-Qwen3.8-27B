"""Compare CPU tables with reviewed, hash-pinned upstream YaRN functions without installing Transformers."""

import argparse
import ast
import __future__
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from dspark_intake import FILES, validate_config
from dspark_rope_tables import DSparkRotary


UPSTREAM_REVISION = 'cc832f9055ba11c8c55f918ab4bda9472b910d48'
UPSTREAM = {
    'modeling_rope_utils.py.source.txt': '200d1fb4ed77132634761e279abba73d48d8bd8d8075de9c54ccc4f7f6671553',
    'modeling_qwen3.py.source.txt': 'fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab',
}
POSITIONS = [0, 1, 6, 7, 169, 170, 4095, 4096, 8191, 8192, 16383, 16384,
    32767, 32768, 64503, 64504, 65535, 65536, 131071, 131072, 262143]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reviewed_function(path, name, *, class_name=None):
    import torch

    if path.name not in UPSTREAM or path.stat().st_size > 1000000:
        raise ValueError('Only the two reviewed, bounded upstream source files are allowed')
    data = path.read_bytes()
    if digest(data) != UPSTREAM[path.name]:
        raise ValueError('Upstream source digest differs; refuse function execution')
    nodes = ast.parse(data).body
    if class_name is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    if (path.name, class_name, name) not in (
            ('modeling_rope_utils.py.source.txt', None, '_compute_yarn_parameters'),
            ('modeling_qwen3.py.source.txt', 'Qwen3RotaryEmbedding', 'forward')):
        raise ValueError('Only static YaRN parameters and the CPU rotary forward body are reviewed')
    function.decorator_list = []
    namespace = dict(torch=torch, math=math, maybe_autocast=torch.autocast)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def compare(config_path, source_directory):
    import torch

    data = config_path.read_bytes()
    if (len(data), digest(data)) != FILES['config.json']:
        raise ValueError('Exact DSpark checkpoint configuration bytes required')
    config = json.loads(data)
    validate_config(config)
    parameter_function = reviewed_function(source_directory / 'modeling_rope_utils.py.source.txt', '_compute_yarn_parameters')
    table_function = reviewed_function(source_directory / 'modeling_qwen3.py.source.txt', 'forward',
        class_name='Qwen3RotaryEmbedding')
    configuration = SimpleNamespace(**config, standardize_rope_params=lambda: None)
    frequency, scale = parameter_function(configuration, device=torch.device('cpu'))
    upstream_rotary = SimpleNamespace(inv_freq=frequency, attention_scaling=scale)
    candidate = DSparkRotary(config)
    if (not torch.equal(frequency.view(torch.int32), candidate.inverse_frequency.view(torch.int32))
            or scale != candidate.attention_scaling):
        raise AssertionError('CPU YaRN parameters differ from pinned upstream arithmetic')
    identifiers = torch.tensor(POSITIONS, dtype=torch.int64)
    checks = []
    for dtype in (torch.float32, torch.bfloat16):
        expected = table_function(upstream_rotary, torch.zeros(1, 1, 128, dtype=dtype), identifiers[None])
        actual = candidate.positions(identifiers, dtype=dtype)
        for kind, observed, reference in zip(('cosine', 'sine'), actual, expected, strict=True):
            observed = observed.reshape_as(reference)
            bits = torch.int32 if dtype == torch.float32 else torch.int16
            if not torch.equal(observed.view(bits), reference.view(bits)):
                raise AssertionError(f'CPU {dtype} {kind} tables differ from reviewed upstream forward body')
            checks.append(dict(dtype=str(dtype), table=kind, values=observed.numel(), bitwise_exact=True,
                sha256=digest(observed.contiguous().view(torch.uint8).numpy().tobytes())))
    return dict(cpu_reference_passed=True, backend='cpu', upstream_revision=UPSTREAM_REVISION,
        upstream_sources=UPSTREAM.copy(), config_sha256=digest(data), positions=POSITIONS.copy(),
        inverse_frequency_exact=True, attention_scaling_exact=True, attention_scaling=scale,
        correction_range=list(candidate.correction_range), checks=checks,
        target_integrated=False, eligible_for_hardware=False, serving_qualified=False,
        scope='Static pre-normalized YaRN config and selected absolute-position tables only; decorators disabled, no model code imported')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--upstream-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    paths = [Path(__file__), Path(__file__).with_name('dspark_rope_tables.py'), Path(__file__).with_name('dspark_intake.py')]
    sources = {path.name: digest(path.read_bytes()) for path in paths}
    report = compare(options.config, options.upstream_directory)
    if any(digest(path.read_bytes()) != sources[path.name] for path in paths):
        raise ValueError('CPU reference sources changed during comparison')
    report['sources'] = sources
    options.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
