"""Read per-chip DRAM harvesting masks the way tt-metal reads them, without changing state.

At the pinned runtime revision, metal_env.cpp enables Blackhole DRAM programmable
cores only when every chip in the cluster reports
``get_soc_desc(chip).harvesting_masks.dram_harvesting_mask == 0``; a single-device
cluster skips that loop entirely. This probe reports those masks and nothing else.
Every strategy is attempted independently so one run records the full picture,
including which access paths are unavailable in the pinned image.
"""

import argparse
import json
import os
import traceback
from pathlib import Path

ARCHITECTURAL_DRAM_CHANNELS = 8


def _record(probes, name, run):
    entry = dict(probe=name, ok=False, value=None, error=None)
    try:
        entry['value'] = run()
        entry['ok'] = entry['value'] is not None
    except BaseException:
        entry['error'] = traceback.format_exc(limit=4).strip().splitlines()[-1][:300]
    probes.append(entry)
    return entry


def _descriptor_files(roots):
    hits = []
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for pattern in ('**/*cluster_desc*.yaml', '**/*cluster-desc*.yaml', '**/*soc_desc*.yaml'):
            for path in base.glob(pattern):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if not path.is_file() or stat.st_size > 2_000_000:
                    continue
                hits.append(dict(path=str(path), size=stat.st_size, mtime=int(stat.st_mtime),
                                 text=path.read_text(errors='replace')[:20000]))
    return hits or None


def _harvest_lines(hits):
    """Pull any dram-harvesting lines out of descriptor text without parsing yaml."""
    lines = []
    for hit in hits or []:
        for number, line in enumerate(hit['text'].splitlines(), 1):
            lowered = line.lower()
            if 'harvest' in lowered and ('dram' in lowered or 'mask' in lowered):
                lines.append(dict(path=hit['path'], line=number, text=line.strip()[:200]))
    return lines or None


def _umd_masks():
    """Ask UMD directly for each chip's SoC descriptor harvesting masks."""
    from tt_umd import Cluster  # type: ignore

    cluster = Cluster()
    masks = {}
    for chip in cluster.all_chip_ids():
        harvesting = cluster.get_soc_desc(chip).harvesting_masks
        masks[str(chip)] = dict(
            dram_harvesting_mask=getattr(harvesting, 'dram_harvesting_mask', None),
            tensix_harvesting_mask=getattr(harvesting, 'tensix_harvesting_mask', None),
            eth_harvesting_mask=getattr(harvesting, 'eth_harvesting_mask', None))
    return dict(number_of_devices=len(masks), masks=masks) if masks else None


def _ttnn_channels():
    """Cross-check: a harvested channel shows up as a reduced usable channel count."""
    import ttnn  # type: ignore

    count = ttnn.GetNumAvailableDevices()
    observed = {}
    for index in range(count):
        device = ttnn.open_device(device_id=index)
        try:
            channels = device.num_dram_channels()
            observed[str(index)] = dict(num_dram_channels=channels,
                                        fewer_than_architectural=channels < ARCHITECTURAL_DRAM_CHANNELS)
        finally:
            ttnn.close_device(device)
    return dict(num_available_devices=count, devices=observed) if observed else None


def _umd_symbol_survey(roots):
    """Name the accessor and serialized key so a follow-up run can target them."""
    found = []
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for path in base.rglob('*.h*'):
            if 'umd' not in str(path).lower():
                continue
            try:
                text = path.read_text(errors='replace')
            except OSError:
                continue
            if 'dram_harvesting_mask' not in text:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if 'dram_harvesting_mask' in line:
                    found.append(dict(path=str(path), line=number, text=line.strip()[:200]))
            if len(found) > 200:
                return found
    return found or None


def classify(masks):
    """Map observed masks onto metal's multi-device condition, never guessing an absent value."""
    if not masks:
        return 'masks_unavailable'
    values = [entry.get('dram_harvesting_mask') for entry in masks.values()]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        return 'unknown_mask_reported'
    if any(value != 0 for value in values):
        return 'harvested_dram_blocks_multi_device_prefetch'
    return 'no_dram_harvesting_observed'


def collect(source_roots, allow_device_open=False):
    probes = []
    descriptors = _record(probes, 'cluster_descriptor_files', lambda: _descriptor_files(source_roots))
    _record(probes, 'descriptor_harvest_lines', lambda: _harvest_lines(descriptors['value']))
    umd = _record(probes, 'umd_soc_descriptor_masks', _umd_masks)
    if allow_device_open:
        _record(probes, 'ttnn_dram_channel_crosscheck', _ttnn_channels)
    else:
        probes.append(dict(probe='ttnn_dram_channel_crosscheck', ok=False, value=None,
                           error='skipped: device open not authorised for this run'))
    _record(probes, 'umd_symbol_survey', lambda: _umd_symbol_survey(source_roots))

    masks = (umd['value'] or {}).get('masks') if umd['ok'] else None
    verdict = classify(masks)

    return dict(
        dram_harvest_verdict=verdict,
        chip_masks=masks,
        probes=probes,
        firmware_modified=False,
        devices_reset=False,
        firmware_bundle_gate='separate; requires bundle >= 19.12.0.0 at the pinned revision',
        prefetch_supported=None,
        performance_qualified=False,
        scope=('DRAM harvesting masks only. A zero mask does not enable prefetch on its own: '
               'the firmware bundle floor is an independent gate and no speedup is implied. '
               'An unavailable mask is unknown, never zero.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', action='append', default=[], dest='source_roots')
    parser.add_argument('--allow-device-open', action='store_true',
                        help='Additionally open each device to cross-check usable DRAM channel count')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    roots = options.source_roots or [os.environ.get('TT_METAL_HOME', '/opt/tt-metal'), '/tmp']
    result = collect(roots, allow_device_open=options.allow_device_open)
    options.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
