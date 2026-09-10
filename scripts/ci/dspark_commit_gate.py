"""Separate T16 commit-only GDN prerequisite; the previous T8 result does not qualify this width."""

import json
from pathlib import Path

from dspark_hardware_gate import digest
from gdn_multitoken import HASHES, HANDOFF_HASHES
from gdn_commit_provenance import source_hashes


REPORT = 'dspark-commit16-simulator.json'
SHA256 = None
GENERATED = {
    'compute': '9512188a20fd63f2853f1ac427f1b7dfaf96bab18f9bfb32e73c2d647283a9e0',
    'reader': '6c56547a34384f9727c72fff1905566459b96233e703d5415d3f88328df0d072',
    'writer': '2d62a883af12d4adc2834d1eb50920dce25f910320c5432a456705bb3daef68a',
}


def qualify(directory):
    directory = Path(directory)
    if SHA256 is None:
        raise ValueError('T16 commit-only GDN has not yet been independently qualified')
    path = directory / REPORT
    if digest(path) != SHA256 or path.with_suffix('.exit-status').read_text().strip() != '0':
        raise ValueError('Pinned clean T16 commit-only simulator report required')
    report = json.loads(path.read_text())
    current_sources = source_hashes(directory)
    if report.get('commit_sources') != current_sources or report.get('commit_sources_after') != current_sources:
        raise ValueError('Complete unchanged commit-only adapter and actual publication kernel sources required')
    flags = ('passed', 'norm_gate', 'convolution', 'batched_convolution', 'dma_windows', 'packed_checkpoints',
        'continuation_enabled', 'compact_prologue', 'norm_batch_layer', 'deferred_conv_publication', 'commit_only_gdn')
    if (any(report.get(name) is not True for name in flags) or report.get('backend') != 'ttsim'
            or report.get('rows') != 16 or report.get('last_stage') != dict(stage='complete')
            or any(report.get(name) != 17 for name in ('model_adapter_checks', 'continuation_checks', 'precommit_unchanged_checks'))
            or report.get('stale_controls') != 1 or report.get('native_hashes') != HASHES
            or report.get('handoff_runtime_hashes') != HANDOFF_HASHES or report.get('generated_hashes') != GENERATED):
        raise ValueError('Every T16 accepted prefix, native-state invariant and real T2 continuation must pass')
    sources = {'batched_adapter_sha256': 'gdn_batched_conv.py', 'model_adapter_sha256': 'gdn_device_loop_state.py',
        'adapter_sha256': 'gdn_multitoken_conv.py'}
    for field, name in sources.items():
        if digest(directory / name) != report.get(field):
            raise ValueError('Changed commit-only adapter requires simulation: ' + name)
    for field, stem in (('window_prefix_hashes', 'gdn_conv_prefix_copy'), ('window_dma_hashes', 'gdn_conv_windows')):
        expected = {suffix: digest(directory / (stem + '.' + suffix)) for suffix in ('py', 'cpp')}
        if report.get(field) != expected:
            raise ValueError('Changed packed convolution publication requires simulation')
    return {REPORT: SHA256}
