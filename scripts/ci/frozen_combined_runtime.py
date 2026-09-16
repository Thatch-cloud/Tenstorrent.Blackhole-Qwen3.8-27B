"""Component admission for the explicitly selected offline 32K runtime candidate."""

import hashlib
import os
import subprocess
from pathlib import Path

from frozen_combined_gate import qualify as qualify_components, REPORTS
from frozen_context_geometry import selected_geometry


REPORT_SHA256 = REPORTS['draft-numerical.json']
SCRATCH_PATCH_SHA256 = 'c1aa9382e344df59975f44e172435a43322f55bcfae78e23607f36e9930c63b5'


def prepare_scratch(root):
    from sdpa_tree_scratch import audit
    if (os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1'
            or os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1'
            or os.environ.get('QWEN_HARDWARE_TESTS') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('TT_METAL_SIMULATOR')):
        raise ValueError('Explicit offline hardware scratch candidate required')
    audit(root)
    patch_file = Path('/experiment-optimisation/sim/sdpa-tree-scratch.patch')
    if hashlib.sha256(patch_file.read_bytes()).hexdigest() != SCRATCH_PATCH_SHA256:
        raise ValueError('Simulator-qualified scratch patch required')
    for arguments in (['--check'], []):
        subprocess.run(['git', '-C', str(root), 'apply', *arguments, str(patch_file)],
            check=True, timeout=10)
    return dict(sources=audit(root, patched=True),
        patch_sha256=hashlib.sha256(patch_file.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def verify_scratch(root, evidence):
    from sdpa_tree_scratch import audit
    if (os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1'
            or not isinstance(evidence, dict)
            or evidence.get('patch_sha256') != SCRATCH_PATCH_SHA256
            or evidence.get('sources') != audit(root, patched=True)
            or evidence.get('adapter_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('Matching built hardware scratch source and adapter required')


def qualified_native_reference(root, reference):
    from sdpa_tree_scratch import audit, HASHES, ROOT
    sources = audit(root, patched=True)
    result = dict(reference)
    for name, checksum in sources.items():
        path = (ROOT / name).as_posix()
        if path in reference and reference[path] != HASHES[name]:
            raise ValueError('Unexpected original native fingerprint: ' + path)
        result[path] = checksum
    return result


def validate_target_option(enabled, *, rows, position, remaining, replay, norm_batch,
                           native_sampling, group_rows, short_context):
    if type(enabled) is not bool:
        raise ValueError('Explicit T16 attention selection required')
    if not enabled:
        return
    if (any(type(value) is not int for value in (rows, position, remaining, group_rows))
            or rows != 16 or position != 32768 or not 1 <= remaining <= 256
            or replay is not True or norm_batch is not True or native_sampling is not True
            or group_rows != 4 or short_context is not False):
        raise ValueError('Qualified 32K T16 replay with four-row groups and native sampling required')


def qualify_target(directory):
    evidence = qualify(directory)
    return dict(evidence['target'], report_sha256=REPORTS['target-replay.json'])


def qualify(directory):
    if (os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1'
            or selected_geometry()['context'] != 32768
            or os.environ.get('QWEN_HARDWARE_TESTS') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1'
            or os.environ.get('TT_METAL_SIMULATOR')):
        raise ValueError('Explicit allocated offline 32K combined candidate required')
    directory = Path(directory)
    evidence = directory / 'frozen-evidence'
    draft = evidence / 'draft/scripts/ci'
    target = evidence / 'target/scripts/ci'
    result = qualify_components(evidence, draft_sources=draft, target_sources=target, context=32768)
    from frozen_combined_gate import load_reports
    reports = load_reports(evidence)
    excluded = {'frozen_probe_evidence.py', 'dspark_attention_8k_gate.py',
        'dspark_attention_value_diagnostics.py'}
    expected = {name: checksum for name, checksum in reports['draft-numerical.json']['sources'].items()
        if not name.endswith('-probe.py') and not name.startswith('../') and name not in excluded}
    for name, checksum in reports['target-replay.json']['sources'].items():
        if name.endswith('-probe.py'):
            continue
        if name in expected and expected[name] != checksum:
            raise ValueError('Component evidence disagrees on shared source: ' + name)
        expected[name] = checksum
    for name, checksum in expected.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Combined runtime component source differs: ' + name)
    result.update(report_sha256=REPORTS['draft-numerical.json'], runtime_component_sources=expected)
    return result
