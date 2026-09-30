"""Extract explicit telemetry observations without inferring prefetch capability."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(snapshot):
    if not isinstance(snapshot, (dict, list)) or not snapshot:
        raise ValueError('Nonempty structured hardware snapshot required')
    observations = []

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                location = path + [str(key)]
                normalized = str(key).lower().replace('-', '_').replace(' ', '_')
                if any(term in normalized for term in ('firmware', 'fw_version', 'harvest', 'board_id', 'serial')):
                    observations.append(dict(path=location, value=child))
                visit(child, location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [index])

    visit(snapshot, [])
    masks = [entry for entry in observations if 'dram' in str(entry['path'][-1]).lower()
             and 'harvest' in str(entry['path'][-1]).lower()]
    return dict(observations=observations, explicit_dram_harvest_fields=masks,
        dram_harvesting_status='requires_schema_and_board_mapping_review' if masks else 'not_reported',
        prefetch_supported=None, firmware_modified=False, devices_reset=False,
        performance_qualified=False,
        scope='Raw telemetry fields only; missing masks are unknown, never zero; native capability still required')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    payload = options.snapshot.read_bytes()
    result = summarize(json.loads(payload))
    result['snapshot_sha256'] = hashlib.sha256(payload).hexdigest()
    options.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
