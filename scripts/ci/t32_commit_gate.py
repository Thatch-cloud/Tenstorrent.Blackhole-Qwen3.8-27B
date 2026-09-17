"""Pinned T32 same-kernel serial-equivalence prerequisite, not full-request admission."""

import json
from pathlib import Path

from dspark_hardware_gate import digest
from dspark_commit_gate import GENERATED
from gdn_commit_provenance import source_hashes
from gdn_multitoken import HASHES, HANDOFF_HASHES


REPORT = 't32-commit.json'
SHA256 = '4a29010fbcc8b0ff456ce0b156370a2f3791ed496a1762a2f212a5019a2c439d'


def validate(report, sources):
    flags = ('passed', 'norm_gate', 'convolution', 'batched_convolution', 'dma_windows',
        'packed_checkpoints', 'continuation_enabled', 'compact_prologue', 'norm_batch_layer',
        'deferred_conv_publication', 'commit_only_gdn')
    counts = ('model_adapter_checks', 'continuation_checks', 'precommit_unchanged_checks')
    if (any(report.get(name) is not True for name in flags)
            or report.get('backend') != 'ttsim' or report.get('rows') != 32
            or report.get('last_stage') != {'stage': 'complete'}
            or any(report.get(name) != 33 for name in counts) or report.get('stale_controls') != 1
            or report.get('native_hashes') != HASHES or report.get('handoff_runtime_hashes') != HANDOFF_HASHES
            or report.get('generated_hashes') != GENERATED):
        raise ValueError('Complete T32 prefix, continuation and unchanged pre-commit checks required')
    if not sources or report.get('commit_sources') != sources or report.get('commit_sources_after') != sources:
        raise ValueError('Unchanged T32 adapter and publication source identity required')
    return dict(rows=32, prefixes=33, full_request_qualified=False, native_oracle_qualified=False)


def qualify(directory, *, source_directory=None):
    path = Path(directory) / REPORT
    if digest(path) != SHA256 or path.with_suffix('.exit-status').read_text().strip() != '0':
        raise ValueError('Pinned clean T32 commit-state simulator report required')
    sources = source_hashes(Path(__file__).parent if source_directory is None else source_directory)
    return validate(json.loads(path.read_text()), sources)
