"""Prepare the existing numerical/replay harness for block-stream transport."""

import hashlib

from frozen_recipe_context import replace_once
from frozen_mlp_buffer_trial import adapt_probe as t16_probe
from mlp_register_epilogue_stage import diagnostic_probe


def stream_fingerprints(operations, stream):
    shards = operations.get_device_tensors(stream)
    if len(shards) != 2:
        raise ValueError('Both stream shards required for byte integrity')
    return [hashlib.sha256(operations.to_torch(shard).contiguous().numpy().tobytes()).hexdigest()
        for shard in shards]


def adapt_probe(source):
    result = diagnostic_probe(t16_probe(source))
    changes = (
        ('from draft_mlp_fixture import load_mlp',
            'from mlp_block_stream import pack\n'
            'from mlp_block_stream_projection import bind_stream\n'
            'from mlp_block_stream_probe import stream_fingerprints\n'
            'from draft_mlp_fixture import load_mlp'),
        ("        report['phase'] = 'weight_checks_passed'",
            "        report['phase'] = 'pack_block_stream'\n"
            "        options.output.write_text(json.dumps(report, indent=2))\n"
            '        device_stream = pack(mesh, device_packed)\n'
            '        owned.append(device_stream)\n'
            '        ttnn.synchronize_device(mesh)\n'
            "        report['stream_bytes_before'] = stream_fingerprints(ttnn, device_stream)\n"
            "        report['phase'] = 'weight_checks_passed'"),
        ("            report['phase'] = 'fused_projection'",
            "            report['stream_binding'] = bind_stream(operation, device_stream, ttnn)\n"
            "            report['phase'] = 'fused_projection'"),
        ('(device_gate, device_up, device_packed), timing=options.timing)',
            '(device_gate, device_up, device_packed, device_stream), timing=options.timing)'),
        ("                report.setdefault('trace_replays', []).append(checked)",
            "                report.setdefault('trace_replays', []).append(checked)\n"
            "                report['stream_bytes_after'] = stream_fingerprints(ttnn, device_stream)\n"
            "                if report['stream_bytes_after'] != report['stream_bytes_before']:\n"
            "                    raise AssertionError('MLP eager/capture/replay mutated stream bytes')"),
        ('T16 buffering only; other row widths and performance unqualified',
            'T16 block-stream MLP numerical and replay checks; not combined performance or target quality'),
    )
    for before, after in changes:
        result = replace_once(result, before, after)
    compile(result, 'block-stream-fused-batch-probe.py', 'exec')
    return result
