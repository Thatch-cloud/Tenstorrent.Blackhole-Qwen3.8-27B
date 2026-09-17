"""Compare saved CPU backbone stages with reviewed upstream function bodies; no imports of checkpoint modules."""

import argparse
import ast
import __future__
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

from dspark_intake import FILES, TAPS
from dspark_rope_reference import UPSTREAM, UPSTREAM_REVISION
from dspark_weights import VerifiedWeights


FUNCTIONS = {
    'rms': ('qwen', 'Qwen3RMSNorm', 'forward'),
    'mlp': ('qwen', 'Qwen3MLP', 'forward'),
    'eager_attention_forward': ('qwen', None, 'eager_attention_forward'),
    'repeat_kv': ('qwen', None, 'repeat_kv'),
    'rotate_half': ('qwen', None, 'rotate_half'),
    'rotary': ('qwen', 'Qwen3RotaryEmbedding', 'forward'),
    'yarn': ('rope', None, '_compute_yarn_parameters'),
    'apply_rotary_pos_emb': ('draft', None, 'apply_rotary_pos_emb'),
    'attention': ('draft', 'Qwen3DFlashAttention', 'forward'),
    'layer': ('draft', 'Qwen3DFlashDecoderLayer', 'forward'),
    'backbone': ('draft', 'DFlashDraftModel', 'forward'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reviewed_functions(draft_path, upstream_directory):
    import torch

    paths = dict(draft=draft_path, qwen=upstream_directory / 'modeling_qwen3.py.source.txt',
        rope=upstream_directory / 'modeling_rope_utils.py.source.txt')
    expected = dict(draft=FILES['dflash.py'][1], qwen=UPSTREAM['modeling_qwen3.py.source.txt'],
        rope=UPSTREAM['modeling_rope_utils.py.source.txt'])
    trees = {}
    for name, path in paths.items():
        if path.stat().st_size > 1000000:
            raise ValueError('Bounded reviewed upstream source required')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError('Upstream source hash differs; refuse function execution')
        trees[name] = ast.parse(data).body
    namespace = dict(torch=torch, nn=torch.nn, math=math, maybe_autocast=torch.autocast)
    for alias, (source, owner, name) in FUNCTIONS.items():
        nodes = trees[source]
        if owner is not None:
            nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
        function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
        function.name = alias
        function.decorator_list = []
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(paths[source]), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return {name: namespace[name] for name in FUNCTIONS}, expected


def control_forward(functions, weights, config, features, noise, start, inspect, *, attention_implementation='eager'):
    import torch

    if attention_implementation not in ('eager','qwen_fp32_reference'):
        raise ValueError('Declared eager or explicit FP32 CPU reference backend required')
    if (attention_implementation!='eager'
            and attention_implementation not in functions['attention'].__globals__.get('ALL_ATTENTION_FUNCTIONS',{})):
        raise ValueError('The explicit FP32 CPU backend must be registered before model execution')

    def linear(name):
        return lambda value: torch.nn.functional.linear(value, weights.tensor(name))

    def norm(name, stage=None, before=False):
        def apply(value):
            if before:
                inspect(stage, value)
            result = functions['rms'](SimpleNamespace(weight=weights.tensor(name), variance_epsilon=1e-6), value)
            if stage is not None and not before:
                inspect(stage, result)
            return result
        return apply

    def layer(index):
        prefix = f'layers.{index}.'
        attention = SimpleNamespace(head_dim=128, num_key_value_groups=4, scaling=128 ** -.5, training=False,
            attention_dropout=0., sliding_window=None, layer_idx=index, config=SimpleNamespace(_attn_implementation=attention_implementation),
            **{name + '_proj': linear(prefix + f'self_attn.{name}_proj.weight') for name in ('q', 'k', 'v', 'o')},
            q_norm=norm(prefix + 'self_attn.q_norm.weight'), k_norm=norm(prefix + 'self_attn.k_norm.weight'))
        mlp = SimpleNamespace(gate_proj=linear(prefix + 'mlp.gate_proj.weight'), up_proj=linear(prefix + 'mlp.up_proj.weight'),
            down_proj=linear(prefix + 'mlp.down_proj.weight'), act_fn=torch.nn.functional.silu)
        state = SimpleNamespace(input_layernorm=norm(prefix + 'input_layernorm.weight'),
            post_attention_layernorm=norm(prefix + 'post_attention_layernorm.weight', f'layer_{index}_attention_residual', True),
            self_attn=lambda **kwargs: functions['attention'](attention, **kwargs), mlp=lambda value: functions['mlp'](mlp, value))

        def apply(**kwargs):
            result = functions['layer'](state, **kwargs)
            inspect(f'layer_{index}_output', result)
            return result
        return apply

    configuration = SimpleNamespace(**config, standardize_rope_params=lambda: None)
    frequency, scale = functions['yarn'](configuration, device=torch.device('cpu'))
    rotary = SimpleNamespace(inv_freq=frequency, attention_scaling=scale)
    state = SimpleNamespace(fc=linear('fc.weight'), hidden_norm=norm('hidden_norm.weight', 'projected_context'),
        norm=norm('norm.weight', 'final_norm'), layers=[layer(index) for index in range(5)],
        rotary_emb=lambda value, positions: functions['rotary'](rotary, value, positions),
        gradient_checkpointing=False, training=False)
    packed = torch.cat([features[index] for index in TAPS], dim=-1)
    positions = torch.arange(start, start + packed.shape[1] + 7, dtype=torch.int64)[None]
    return functions['backbone'](state, position_ids=positions, noise_embedding=noise, target_hidden=packed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--draft-source', type=Path, required=True)
    parser.add_argument('--upstream-directory', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    import torch

    if options.output.exists():
        raise ValueError('Fresh independent comparison report required')
    baseline = json.loads(options.baseline_report.read_text())
    if (baseline.get('cpu_backbone_executed') is not True or baseline.get('closed_cleanly') is not True
            or baseline.get('backend') != 'cpu' or baseline.get('target_integrated') is not False
            or baseline.get('eligible_for_hardware') is not False or len(baseline.get('cases', [])) != 4
            or len(baseline.get('stages', [])) != 48 or not baseline.get('sources')
            or baseline['sources'] != baseline.get('sources_after')
            or any(digest(Path(__file__).with_name(name)) != checksum for name, checksum in baseline['sources'].items())):
        raise ValueError('Complete saved CPU execution and unchanged baseline sources required')
    tensors_path = options.baseline_report.with_suffix('.outputs.pt')
    if tensors_path.stat().st_size > 10000000 or digest(tensors_path) != baseline.get('output_tensors_sha256'):
        raise ValueError('Complete bounded and hash-matched saved CPU outputs required')
    if options.config.stat().st_size != FILES['config.json'][0] or digest(options.config) != FILES['config.json'][1]:
        raise ValueError('Pinned DSpark configuration required')
    functions, upstream = reviewed_functions(options.draft_source, options.upstream_directory)
    spec = importlib.util.spec_from_file_location('dspark_cpu_baseline', Path(__file__).with_name('dspark-backbone-cpu.py'))
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    outputs = torch.load(tensors_path, weights_only=True)
    expected_stages = ['projected_context', *[f'layer_{layer}_{stage}' for layer in range(5)
        for stage in ('attention_residual', 'output')], 'final_norm']
    recorded = {(entry['case'], entry['stage']): entry for entry in baseline['stages']}
    if (set(outputs) != set(range(4)) or set(recorded) != {(case, stage) for case in range(4) for stage in expected_stages}
            or any(set(entry['stages']) != set(expected_stages) for entry in outputs.values())):
        raise ValueError('Every saved case and full-backbone stage is required')
    for case, entry in outputs.items():
        for stage, value in entry['stages'].items():
            record = recorded[case, stage]
            if (record.get('finite') is not True or list(value.shape) != record.get('shape')
                    or str(value.dtype) != record.get('dtype') or source.tensor_digest(value) != record.get('sha256')):
                raise ValueError('Saved stage tensors disagree with baseline report')
    local_sources = {name: digest(Path(__file__).with_name(name)) for name in
        ('dspark_upstream_backbone.py', 'dspark_rope_reference.py', *baseline['sources'])}
    report = dict(upstream_cpu_exact=False, closed_cleanly=False, backend='cpu', checks=[], scope=__doc__,
        upstream_sources=upstream, transformers_revision=UPSTREAM_REVISION, sources=local_sources,
        baseline_report_sha256=digest(options.baseline_report), baseline_outputs_sha256=digest(tensors_path),
        reviewed_checkpoint_functions_executed=True, checkpoint_modules_imported=False,
        target_integrated=False, eligible_for_hardware=False, serving_qualified=False)
    weights = None
    try:
        with VerifiedWeights(options.checkpoint) as weights:
            if weights.fingerprints() != baseline.get('tensor_sha256'):
                raise ValueError('Independent control uses a different checkpoint')
            for case, (pattern, start) in enumerate(((0, 0), (0, 8190), (1, 8190), (0, 0))):
                features, noise = source.inputs(pattern)
                expected = outputs[case]['stages']
                seen = set()

                def inspect(stage, value):
                    if (stage in seen or stage not in expected or value.dtype != torch.bfloat16
                            or value.shape != expected[stage].shape or not torch.isfinite(value).all()
                            or not torch.equal(value.contiguous().view(torch.int16), expected[stage].contiguous().view(torch.int16))):
                        raise AssertionError(f'CPU port differs from reviewed upstream body: case={case}, stage={stage}')
                    seen.add(stage)
                    entry = dict(case=case, stage=stage, values=value.numel(), bitwise_exact=True, sha256=source.tensor_digest(value))
                    report['checks'].append(entry)
                    print(json.dumps(entry), flush=True)

                output = control_forward(functions, weights, json.loads(options.config.read_text()), features, noise, start, inspect)
                if seen != set(expected) or source.tensor_digest(output) != source.tensor_digest(outputs[case]['output']):
                    raise AssertionError('Incomplete upstream stage comparison')
        report['closed_cleanly'] = weights.source is None
        report['sources_after'] = {name: digest(Path(__file__).with_name(name)) for name in local_sources}
        if report['sources_after'] != local_sources:
            raise ValueError('Comparison sources changed during execution')
        report['upstream_cpu_exact'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['closed_cleanly'] = weights is None or weights.source is None
        raise
    finally:
        options.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(upstream_cpu_exact=True, stage_checks=len(report['checks']), eligible_for_hardware=False)), flush=True)


if __name__ == '__main__':
    main()
