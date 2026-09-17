"""Full-64K worker-limit component admission; not request or speed acceptance."""

import json
from pathlib import Path
from unittest.mock import patch

import matched_draft_gate as baseline
from dspark_splitk_sim_gate import verify_sources
from splitk_workers_gate import REPORT_SHA256 as SIMULATOR_SHA256


REPORT_SHA256 = 'b64e417601f29f302da91b0ee1a27faee877c2e7149798d6ac7fd9817914cacd'


def qualify(directory, report_path):
    with patch.object(baseline, 'REPORTS', {65536: REPORT_SHA256}):
        admission = baseline.qualify(directory, report_path, 65536)
    report = json.loads(Path(report_path).read_bytes())
    worker = report['worker_experiment']
    call = dict(key_chunk_size=256, requested_worker_limit=8, selected_worker_limit=16,
        stripe_keys=False, fp32_dest_acc=True)
    if (worker.get('failure') is not None or worker.get('calls') != [call] * 3
            or worker['sources'] != worker['sources_after']
            or worker['simulator'].get('report_sha256') != SIMULATOR_SHA256
            or worker['simulator'].get('simulator_qualified') is not True):
        raise ValueError('Exact source-stable sixteen-worker hardware execution required')
    verify_sources(directory, worker['sources'])
    return dict(admission, worker_limit=16, key_chunk_size=256,
        simulator_report_sha256=SIMULATOR_SHA256,
        geometry_scope='Same full-history allocation; worker_experiment overrides baseline worker scheduling')
