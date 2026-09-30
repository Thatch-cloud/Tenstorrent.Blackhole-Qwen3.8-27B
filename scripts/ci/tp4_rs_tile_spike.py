"""The four-card reduction-order spike (qwen-c2-serving.yml 'fabric' with C2_FABRIC_PROBE=rs-tile, C2_CARDS=quad).

    python3 -B /probe/tp4_rs_tile_spike.py --fabric FABRIC_1D --output /probe-results/rs-tile-spike.json

S3a at four cards (docs/tp4-exact-ring-parity.md): the users that sit in rows 32..63 of the 64-row verify block diverge
from the sequential engine in their first rounds; the users in rows 0..31 do not. The suspect is the model's own
all-reduce (tt_transformers ccl.tt_all_reduce, which on a (1, N) mesh is the reduce_scatter_minimal_async alone: each
chip keeps its own 5120/4 = 1280-column slice of the sum, ccl.py returns straight after it), whose RING
reduce-scatter adds its four partials forward for even chunks and backward for odd ones, the parity taken from the
tile's flat index in the per-chip slice, so the block's second tile is added in another order than a one-tile call.
A collective needs four chips: this cannot be shown on one card (the sums have nothing to associate there), and at two
chips the sum commutes. So this spike opens all four cards, as the fabric probe does, and calls the model's own
tt_all_reduce with the model's own arguments.

WHAT IT MEASURES (random full-mantissa bfloat16 partials, heavy tailed so that a changed association shows in the last
bit; integers, as the fabric probe used, are exact under any order and cannot see this):
  REPRODUCE (the unfixed program): for each block height R (64, 128): the model's all-reduce on the whole
            (1, 1, R, 5120) block, compared bit for bit, per 32-row tile, with the same call on that tile alone (what the
            sequential engine runs, one tile). Predicted: tile 0 identical, every later tile differing.
  FIX       the same block through tile_collective_tp (the wrapper the four-card process installs) inside its block
            scope: every tile must equal the one-tile call. Its split count must be one per call.
            EVERY comparison covers all four chips: each chip holds a different 1280-column slice (chip k is not chip 0),
            and each chip starts its ring at another slice index, so a comparison of one chip's columns shows nothing
            about the others. A comparison joins chip 0..3's slices of a tile and compares that with the joined slices of
            the one-tile call.
  CONTROLS  the one-tile call twice (deterministic) and four rows against the tile's first four (shape-invariant).
            There is no replication control: at (1, 4) the outputs of the chips differ by construction.
  INFORMATIONAL (never a verdict input; an error here does not fail the run) Linear topology on the whole block against
            its one-tile call (the pair's topology, not the served path), and the reduce-scatter called directly on the
            block reshaped unit-major to (1, R/32, 32, 5120), the cheaper split that would need no extra launches (only
            the direct call restarts the ring's parity per unit; tt_all_reduce flattens the units back first).

VERDICT LINE (also in the report file): TP4_RS_TILE verdict=PASS|NOT_REPRODUCED|FAIL comparisons=C differing=D ...
  PASS            the unfixed tiles beyond the first differ, the first does not, the fix is bit exact, the controls hold.
                  Exit 0: the reduction order is a cause and the tile split cures it.
  NOT_REPRODUCED  the unfixed block already equals the tile calls (the ring's chunk parity does not flip on this image):
                  the collective is not the cause of S3a; the fix is harmless but do not expect it to cure. Exit 2.
  FAIL            the fix is not exact, a control failed (the baseline is not repeatable, so no comparison means anything),
                  tile 0 differs, or the mesh did not open. Exit 1.

Stdlib at import; ttnn, torch and the models tree only inside run(), where a test injects fakes.
"""

import argparse
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tp4_mesh  # noqa: E402
import tp4_fabric_probe  # noqa: E402

TILE = 32
WIDTH = 5120
HEIGHTS = (64, 128)
SEEDS = 3
GROUPS = ('unfixed', 'fixed', 'control', 'informational')


def parse_heights(text):
    heights = []
    for part in text.split(','):
        part = part.strip()
        if not part.isdigit() or int(part) <= TILE or int(part) % TILE:
            raise ValueError('block heights are whole 32-row tiles beyond one tile, got %r' % part)
        heights.append(int(part))
    if not heights or len(set(heights)) != len(heights):
        raise ValueError('at least one block height, each once')
    return tuple(heights)


def compare_bits(torch, left, right):
    """(elements, differing, first differing flat index or None) of two bfloat16 tensors by their bits."""
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError('shapes %s and %s cannot be compared' % (tuple(left.shape), tuple(right.shape)))
    different = (left.contiguous().view(torch.int16) != right.contiguous().view(torch.int16)).reshape(-1)
    count = int(different.sum())
    first = int(different.nonzero()[0]) if count else None
    return int(different.numel()), count, first


def record(group, name, elements, differing, first, **extra):
    row = dict(group=group, name=name, elements=elements, differing=differing, first=first)
    row.update(extra)
    return row


def summarise(records):
    """Counts of tensor comparisons and differing elements per group, from the comparison records."""
    summary = {}
    for group in GROUPS:
        rows = [row for row in records if row['group'] == group and 'error' not in row]
        summary[group] = dict(comparisons=len(rows), differing=sum(row['differing'] for row in rows),
                              unequal=sum(1 for row in rows if row['differing']))
    return summary


def verdict(report):
    """(verdict, reasons) for a finished report: PASS, NOT_REPRODUCED or FAIL."""
    reasons = []
    if not report.get('opened'):
        return 'FAIL', ['the (1, 4) mesh did not open: %s' % (report.get('error') or 'unknown')]
    if report.get('error'):
        reasons.append('the run stopped: %s' % report['error'])
    records = report.get('records') or []
    if not records:
        return 'FAIL', reasons + ['no comparison ran']
    errors = [row['name'] for row in records if 'error' in row and row['group'] != 'informational']
    if errors:
        reasons.append('errored: %s' % ', '.join(errors))
    controls = [row['name'] for row in records if row['group'] == 'control' and row.get('differing')]
    if controls:
        reasons.append('a control differs (the baseline is not repeatable): %s' % ', '.join(controls))
    first_tile = [row for row in records if row['group'] == 'unfixed' and row.get('tile') == 0 and 'error' not in row]
    later_tiles = [row for row in records if row['group'] == 'unfixed' and row.get('tile', 0) > 0 and 'error' not in row]
    fixed = [row for row in records if row['group'] == 'fixed' and 'error' not in row]
    if not first_tile or not later_tiles or not fixed:
        reasons.append('the unfixed and fixed comparisons did not all run')
    if any(row['differing'] for row in first_tile):
        reasons.append('the unfixed first tile differs from the one-tile call: %s (the hypothesis is not the whole story)'
                       % ', '.join(row['name'] for row in first_tile if row['differing']))
    if any(row['differing'] for row in fixed):
        reasons.append('the tile split is not exact: %s' % ', '.join(row['name'] for row in fixed if row['differing']))
    wrong_splits = [row['name'] for row in records if row.get('splits_expected') is not None
                    and row.get('splits') != row['splits_expected']]
    if wrong_splits:
        reasons.append('the wrapper split the wrong number of calls: %s' % ', '.join(wrong_splits))
    if reasons:
        return 'FAIL', reasons
    if not any(row['differing'] for row in later_tiles):
        return 'NOT_REPRODUCED', ['every later tile of the unfixed block equals its one-tile call: the ring order does not '
                                  'depend on the tile here']
    # Under the parity model a tile whose flat index lands on the same chunk parity as tile 0 (128 rows: tile 2) is
    # equal too; that is noted, not judged.
    silent = [row['name'] for row in later_tiles if not row['differing']]
    return 'PASS', (['later tiles equal to their one-tile call: %s' % ', '.join(silent)] if silent else [])


def verdict_line(report, outcome):
    summary = report.get('summary') or {}
    counted = [summary.get(group, {}) for group in ('unfixed', 'fixed', 'control')]
    unfixed, fixed, control = summary.get('unfixed', {}), summary.get('fixed', {}), summary.get('control', {})
    return ('TP4_RS_TILE verdict=%s comparisons=%d differing=%d unfixed_unequal=%s/%s unfixed_differing=%s fixed_unequal=%s/%s '
            'fixed_differing=%s control_unequal=%s/%s heights=%s seeds=%s topology=%s links=%s'
            % (outcome, sum(row.get('comparisons', 0) for row in counted), sum(row.get('differing', 0) for row in counted),
               unfixed.get('unequal'), unfixed.get('comparisons'), unfixed.get('differing'),
               fixed.get('unequal'), fixed.get('comparisons'), fixed.get('differing'),
               control.get('unequal'), control.get('comparisons'),
               ','.join(str(height) for height in report.get('heights') or ()), report.get('seeds'),
               report.get('topology'), report.get('num_links')))


def partials(torch, seed, rows, devices):
    """One partial per chip, (devices, 1, rows, WIDTH) bfloat16, heavy tailed with full mantissas."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(devices, 1, rows, WIDTH, generator=generator)
    scale = torch.exp(torch.randn(devices, 1, rows, WIDTH, generator=generator))
    return (base * scale).to(torch.bfloat16)


def run(options, log=print, modules=None):
    """The measurement. `modules` (a test): dict(torch, ttnn, TT_CCL, tt_all_reduce, get_num_links) in place of the
    image's."""
    report = dict(fabric=options.fabric, opened=False, records=[], heights=list(options.heights), seeds=options.seeds,
                  topology='Ring', cluster_type=None, descriptor=os.environ.get('TT_MESH_GRAPH_DESC_PATH'),
                  serving_contract=os.environ.get('QWEN_C2_SERVING'))
    if modules is None:
        import torch
        import ttnn
        from models.common.modules.tt_ccl import get_num_links
        from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce
        modules = dict(torch=torch, ttnn=ttnn, TT_CCL=TT_CCL, tt_all_reduce=tt_all_reduce, get_num_links=get_num_links)
        problems = None
        descriptor = report['descriptor']
        if descriptor and os.path.isfile(descriptor):
            with open(descriptor, encoding='utf-8') as handle:
                problems = tp4_mesh.descriptor_problems(handle.read())
            report['descriptor_problems'] = problems
        refusal = tp4_fabric_probe.descriptor_refusal(descriptor, problems)
        if refusal:
            report['error'] = 'refused to open: ' + refusal
            return report
    torch, ttnn = modules['torch'], modules['ttnn']
    import tile_collective_tp
    mesh = None
    try:
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, options.fabric))
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*tp4_mesh.MESH_SHAPE), l1_small_size=24576, trace_region_size=0)
        report['opened'] = True
        report['order'] = [int(device) for device in mesh.get_device_ids()]
        devices = mesh.get_num_devices()
        report['num_links'] = int(modules['get_num_links'](mesh))
        try:
            report['cluster_type'] = str(ttnn.cluster.get_cluster_type())
        except Exception as error:
            report['cluster_type'] = 'unknown (%s: %s)' % (type(error).__name__, error)
        collective = modules['TT_CCL'](mesh)
        measure(options, report, log, torch, ttnn, mesh, collective, modules['tt_all_reduce'], devices, tile_collective_tp)
    except Exception as error:
        report['error'] = '%s: %s' % (type(error).__name__, error)
        report['traceback'] = traceback.format_exc()[-4000:]
        report['known_failure'] = tp4_fabric_probe.known_signature(report['traceback'])
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                report['closed'] = True
            except Exception as error:
                report['closed'] = '%s: %s' % (type(error).__name__, error)
    return report


def measure(options, report, log, torch, ttnn, mesh, collective, all_reduce, devices, tile_collective_tp):
    mapper = ttnn.ShardTensorToMesh(mesh, dim=0)

    def upload(source):
        return ttnn.from_torch(source, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)

    def download(tensor):
        return [ttnn.to_torch(part).to(torch.bfloat16) for part in ttnn.get_device_tensors(tensor)]

    def reduce(function, source, topology):
        """The model's call, as its layers make it (attention/tp.py, mlp.py: no link keywords), consuming the input."""
        output = function(upload(source), mesh, collective, cluster_axis=0, dim=3, topology=getattr(ttnn.Topology, topology),
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
        copies = download(output)
        ttnn.deallocate(output)
        return copies

    records = report['records']

    def compare(group, name, left, right, **extra):
        try:
            elements, differing, first = compare_bits(torch, left, right)
            records.append(record(group, name, elements, differing, first, **extra))
            log('TP4_RS_TILE compare %s %s differing=%d/%d%s' % (group, name, differing, elements,
                                                                (' first=%d' % first) if first is not None else ''))
        except Exception as error:
            records.append(dict(group=group, name=name, error='%s: %s' % (type(error).__name__, error)))
            log('TP4_RS_TILE compare %s %s ERROR %s' % (group, name, error))

    def chips(copies, first=None, last=None):
        """Every chip's rows first..last side by side: chip k holds columns k*w..(k+1)*w of the sum, so the four
        together are the whole result and a comparison of them is a comparison of all of it."""
        return torch.cat([copy[:, :, first:last, :] for copy in copies], dim=3)

    def note_widths(copies):
        widths = {int(copy.shape[3]) for copy in copies} | set(report.get('output_widths') or ())
        report['output_widths'] = sorted(widths)

    def informational(label, tiles, source, singles):
        """Two arms that inform the choice of a cheaper split and never decide a verdict."""
        try:
            # The pair's topology on the whole block: no chunk parity, so every tile equals its one-tile call.
            linear_whole = reduce(all_reduce, source, 'Linear')
            for tile in range(tiles):
                linear_single = reduce(all_reduce, source[:, :, tile * TILE:(tile + 1) * TILE, :].contiguous(), 'Linear')
                compare('informational', '%s/linear-block-tile%d-vs-one-tile' % (label, tile),
                        chips(linear_whole, tile * TILE, (tile + 1) * TILE), chips(linear_single))
        except Exception as error:
            records.append(dict(group='informational', name='%s/linear' % label,
                                error='%s: %s' % (type(error).__name__, error)))
        try:
            # Unit-major (1, tiles, 32, 5120) through the reduce-scatter itself (tt_all_reduce flattens the units first,
            # so only the direct call restarts the ring's parity for each unit); each chip's slice is 1280 wide.
            reshaped = ttnn.reshape(upload(source), (1, tiles, TILE, WIDTH))
            output = ttnn.experimental.reduce_scatter_minimal_async(
                reshaped, persistent_output_buffers=None, dim=3,
                multi_device_global_semaphore=collective.get_and_cycle_rs_semaphore_handles(),
                barrier_semaphore=collective.get_and_cycle_barrier_semaphore_handle(),
                num_links=int(report['num_links']), memory_config=ttnn.DRAM_MEMORY_CONFIG,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Ring,
                chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
            copies = download(output)
            ttnn.deallocate(output)
            for tile in range(tiles):
                joined = torch.cat([copy[:, tile:tile + 1, :, :] for copy in copies], dim=3)
                compare('informational', '%s/unit-major-tile%d-vs-one-tile' % (label, tile),
                        joined.reshape(1, 1, TILE, -1), chips(singles[tile]))
        except Exception as error:
            records.append(dict(group='informational', name='%s/unit-major' % label,
                                error='%s: %s' % (type(error).__name__, error)))

    wrapped = tile_collective_tp.TileSplitAllReduce(all_reduce, ttnn)
    for height in options.heights:
        tiles = height // TILE
        for seed in range(options.seeds):
            source = partials(torch, 1000 * height + seed, height, devices)
            label = 'r%d/s%d' % (height, seed)
            # The sequential engine's call: one tile, alone, twice (deterministic), and four rows of it.
            singles = []
            for tile in range(tiles):
                piece = source[:, :, tile * TILE:(tile + 1) * TILE, :].contiguous()
                first = reduce(all_reduce, piece, 'Ring')
                singles.append(first)
                note_widths(first)
                if tile == 0:
                    again = reduce(all_reduce, piece, 'Ring')
                    compare('control', '%s/one-tile-twice' % label, chips(again), chips(first))
                    four = reduce(all_reduce, source[:, :, :4, :].contiguous(), 'Ring')
                    compare('control', '%s/four-rows-vs-tile0' % label, chips(four, 0, 4), chips(first, 0, 4))
            # The unfixed program: the whole block through the model's own call.
            whole = reduce(all_reduce, source, 'Ring')
            for tile in range(tiles):
                compare('unfixed', '%s/block-tile%d-vs-one-tile' % (label, tile),
                        chips(whole, tile * TILE, (tile + 1) * TILE), chips(singles[tile]), tile=tile, height=height)
            # The fix: the same block through the four-card process's wrapper, inside its scope.
            before = tile_collective_tp.splits()
            with tile_collective_tp.block_scope(height):
                fixed = reduce(wrapped, source, 'Ring')
            engaged = tile_collective_tp.splits() - before
            records.append(dict(group='control', name='%s/wrapper-splits' % label, elements=1, differing=0, first=None,
                                splits=engaged, splits_expected=1))
            for tile in range(tiles):
                compare('fixed', '%s/split-tile%d-vs-one-tile' % (label, tile),
                        chips(fixed, tile * TILE, (tile + 1) * TILE), chips(singles[tile]), tile=tile, height=height)
            if seed == 0:
                informational(label, tiles, source, singles)
    report['summary'] = summarise(records)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fabric', choices=tp4_mesh.FABRIC_CONFIGS, default=tp4_mesh.FABRIC_CONFIG)
    parser.add_argument('--output', required=True)
    parser.add_argument('--heights', default=','.join(str(height) for height in HEIGHTS),
                        help='block heights in rows, whole 32-row tiles beyond one (default 64,128)')
    parser.add_argument('--seeds', type=int, default=SEEDS)
    return parser


def main(argv=None, runner=run, log=print):
    options = build_parser().parse_args(argv)
    options.heights = parse_heights(options.heights)
    if options.seeds < 1:
        raise SystemExit('--seeds must be at least 1')
    report = runner(options, log=log)
    report.setdefault('summary', summarise(report.get('records') or []))
    outcome, reasons = verdict(report)
    report.update(verdict=outcome, reasons=reasons)
    line = verdict_line(report, outcome)
    report['verdict_line'] = line
    with open(options.output, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    log(line)
    for reason in reasons:
        log('TP4_RS_TILE reason: %s' % reason)
    if report.get('known_failure'):
        log('TP4_RS_TILE known failure: %s' % report['known_failure'])
    return {'PASS': 0, 'NOT_REPRODUCED': 2}.get(outcome, 1)


if __name__ == '__main__':
    sys.exit(main())
