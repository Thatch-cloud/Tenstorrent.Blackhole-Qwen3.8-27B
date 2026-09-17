"""Source-pinned sixteen-worker combined correctness admission."""

import json
from pathlib import Path
from unittest.mock import patch

import matched_combined_gate as baseline
from dspark_splitk_sim_gate import verify_sources
from splitk_workers_hardware_gate import qualify as qualify_hardware


SCREEN_RUN = 35041737747
SCREEN_SHA256 = '9799daa6eadefc02cbad316d21e53b1612edf6b8538101dab4b2608418ab0825'
BASE_QUALIFY = baseline.qualify


def validate_calls(calls, count):
    expected = dict(key_chunk_size=256, requested_worker_limit=8, selected_worker_limit=16,
        stripe_keys=False, fp32_dest_acc=True)
    if count <= 0 or calls != [expected] * count:
        raise ValueError('Every attention call must use the qualified sixteen-worker configuration')


def qualify(directory, report_path):
    with patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN), patch.object(baseline, 'SCREEN_SHA256', SCREEN_SHA256):
        evidence = BASE_QUALIFY(directory, report_path)
    report = json.loads(Path(report_path).read_bytes())
    worker = report['workers_combined']
    scopes = report['splitk_combined']['scopes']
    if len(scopes) != 1:
        raise ValueError('One complete combined runtime scope required')
    validate_calls(worker['calls'], scopes[0]['attention_calls'])
    if (worker.get('failure') is not None or worker['sources'] != worker['sources_after']
            or worker['hardware_admission'] != qualify_hardware(directory,
                Path(directory) / 'dspark-workers-64k-hardware.json')):
        raise ValueError('Source-stable hardware-qualified combined worker audit required')
    verify_sources(directory, worker['sources'])
    return dict(evidence, workers_combined=worker)
