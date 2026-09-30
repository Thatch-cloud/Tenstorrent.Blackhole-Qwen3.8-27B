"""Checkpoint 2: get a DRAM-core prefetch kernel to run on the pair at all.

No prefetch kernel has ever run on this hardware; the capability only became true
after the 19.12.0 flash. This attempts the smallest real run — start the DRISC
prefetcher, push pages through a GCB whose senders are programmable DRAM cores,
and consume them with upstream's discard receiver — and reports exactly where it
stops if it does not get there.

It also extracts the upstream design spec the validator kernel refers to, because
the per-block source mapping in that document governs the layout a correctness run
must reproduce. Nothing here claims performance; the discard consumer exists to
measure push bandwidth, not model value.
"""

import argparse
import inspect
import json
import os
import traceback
from pathlib import Path

BEGIN = '<<<PREFETCH_RUN_JSON_BEGIN>>>'
END = '<<<PREFETCH_RUN_JSON_END>>>'
DESIGN_DOC = 'tt_metal/impl/buffers/prefetcher_matmul_design.md'


def signatures(ttnn):
    names = ('start_tensor_prefetcher', 'stop_tensor_prefetcher',
             'queue_tensor_prefetcher_request', 'wait_for_cq_on_tensor_prefetcher',
             'create_global_circular_buffer_for_tensor_prefetcher',
             'tensor_prefetcher_block_count_for_matmul_1d',
             'tensor_prefetcher_matmul', 'test_dram_prefetcher_consumer',
             'test_dram_prefetcher_validator')
    out = {}
    for name in names:
        value = getattr(ttnn.experimental, name, None)
        if value is None:
            out[name] = None
            continue
        doc = inspect.getdoc(value) or ''
        out[name] = dict(doc=doc)
    return out


def design_doc(root):
    path = Path(root) / DESIGN_DOC
    if not path.is_file():
        matches = list(Path(root).rglob('prefetcher*design*.md'))
        if matches:
            path = matches[0]
    if not path.is_file():
        return None
    return dict(path=str(path), text=path.read_text(errors='replace'))


def attempt_run(ttnn, report):
    """Smallest end-to-end push: DRISC senders -> GCB -> discard receivers."""
    mesh = None
    started = False
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        report['mesh_open'] = True
        report['supported'] = ttnn.experimental.is_tensor_prefetcher_supported(mesh)
        if not report['supported']:
            report['stopped_at'] = 'capability false; refusing to force'
            return
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]

        # One bank, a small disjoint receiver set, away from the DRAM sender coords.
        receivers = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        report['receiver_cores'] = 1
        page_size = 2048
        global_cb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(
            mesh, [(0, receivers)], page_size * 4, ttnn.BufferType.L1)
        report['gcb_created'] = True

        ttnn.experimental.start_tensor_prefetcher(mesh)
        started = True
        report['prefetcher_started'] = True
        report['stopped_at'] = 'started; request construction not yet attempted'
    except BaseException:
        report['error'] = traceback.format_exc(limit=8)[-2500:]
    finally:
        if started:
            try:
                ttnn.experimental.stop_tensor_prefetcher(mesh)
                report['prefetcher_stopped'] = True
            except BaseException:
                report['stop_error'] = traceback.format_exc(limit=4)[-800:]
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                report['mesh_closed'] = True
            except BaseException:
                report['close_error'] = traceback.format_exc(limit=4)[-800:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metal-root', default='/opt/tt-metal')
    parser.add_argument('--run', action='store_true', help='Attempt the kernel run')
    options = parser.parse_args()
    report = dict(scope=__doc__, override_set=bool(
        os.environ.get('TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES')),
        performance_claimed=False, correctness_claimed=False)
    try:
        import ttnn
        report['signatures'] = signatures(ttnn)
        doc = design_doc(options.metal_root)
        report['design_doc_found'] = doc is not None
        if doc:
            report['design_doc_path'] = doc['path']
            report['design_doc'] = doc['text'][:40000]
        if options.run:
            attempt_run(ttnn, report)
    except BaseException:
        report['fatal'] = traceback.format_exc(limit=6)[-2000:]
    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)


if __name__ == '__main__':
    main()
