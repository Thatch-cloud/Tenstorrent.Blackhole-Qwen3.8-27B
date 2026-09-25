"""Bind reviewed progressive replay to unchanged admitted physical MLP arithmetic."""

import copy
from pathlib import Path

from frozen_recipe_context import replace_once
from mlp_block_stream_gate import qualify as qualify_serial
from mlp_block_stream_projection import adapt_projection
from mlp_progressive_input import projection, reader
from mlp_progressive_input_report import digest, inspect
from mlp_register_epilogue_gate import SIM_PACKER


REPORT_SHA256 = '8b843215d54c68ee2f7272db1a4c2ec0d4bdab1524cb062bb02239bf9422c72a'
STAGING_SHA256 = '618ede1c41773d6c9a7f4a2412b21efdf2fd787cce4d90308d753d4dc0581fb4'
# The transformed progressive reader now carries the exit drain commit 4d890d6a
# added to fused_1d_input.cpp; this transform rebuilds the tail, so the drain has
# to be repeated in it (see mlp_progressive_input.reader) and the hash moves with it.
READER_SHA256 = '5dbe53d386ad4074c1493fc65ec5845671490adfa1f1db0c86a034b8f1947f1f'
CANDIDATE_SHA256 = 'f8e227fd3e95977b52b18c4fdb3855d78040cdad4250197935a3f89d7f33be01'


def qualify(directory, serial_evidence, evidence, register_admission):
    serial = qualify_serial(directory, serial_evidence, register_admission)
    evidence = Path(evidence)
    if (digest((evidence / 'fused-batch.json').read_bytes()) != REPORT_SHA256
            or digest((evidence / 'progressive-input-candidate.json').read_bytes()) != STAGING_SHA256):
        raise ValueError('Exact reviewed progressive simulator artifacts required')
    reviewed = inspect(evidence, Path(serial_evidence) / 'fused-batch.json', directory)
    expected, = copy.deepcopy(serial['kernels'])
    expected.update(progressive_input=True, input_buffer_tiles=160)
    expected['reader_sha256']['fused_1d_input.cpp'] = READER_SHA256
    simulated = copy.deepcopy(expected)
    simulated['rounding_runtime']['packer_header_sha256'] = SIM_PACKER
    if reviewed['kernel'] != simulated:
        raise ValueError('Physical arithmetic must retain the admitted serial baseline')
    return dict(passed=True, report_sha256=REPORT_SHA256, kernels=[expected],
        hardware_qualified=False, performance_qualified=False)


def hardware_source(source):
    candidate = projection(adapt_projection(source))
    candidate = replace_once(candidate,
        '        from mlp_progressive_input import require_simulator\n        require_simulator()\n',
        '        require_hardware(os.environ)\n')
    candidate = replace_once(candidate,
        'kernel_source=str(Path(__file__).with_name("fused_1d_input.cpp")), core_ranges=all_cores,',
        'kernel_source=progressive_reader(Path(__file__).with_name("fused_1d_input.cpp").read_text()),\n'
        '                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=all_cores,')
    compile(candidate, 'progressive_hardware_fused_1d.py', 'exec')
    return candidate


def hardware_reader(original):
    candidate = reader(original)
    if digest(candidate.encode()) != READER_SHA256:
        raise ValueError('Physical progressive reader differs from simulated source')
    return candidate
