"""MR probe: what a verify readback costs one read at a time, one mesh read at a time, and as one joined word tensor.

The packed verify reads its ids and values per chip (PackedVerifierEngine.shard_predictions: ttnn.to_torch of every get_device_tensors part,
2 x chips blocking reads a block, 1.05 to 1.65 ms measured at four chips) and the quad draft reads about 40 outputs a round the same way. Whether
ONE mesh-level read (one call, the runtime batching the shard reads) or ONE joined tensor (a packed `(column << 16) | bits` word per row, read
from every chip by one read) is cheaper than the loop is a property of the runtime this probe measures; it is host-only code on the
verify side, a copy, so no exactness question arises. The probe decides MR and the joined quad reads (2a) of the fusion plan.

Arms, each over the same data on the same mesh, N timed iterations after a warm-up, in a fixed interleaved order (the clock drifts):

  floor     one blocking read of one chip's part: the fixed cost L of a read.
  serial    the served loop: every chip's ids part, then every chip's values part (2 x chips reads).
  mesh      two mesh-level reads (ids, values): ttnn.to_torch(tensor, mesh_composer=ConcatMeshToTensor(mesh, dim=0)).
  word      ONE mesh-level read of one uint32 word tensor (the S1 word layout, ids and values joined): MR's layout.
  overlap   every chip's part non-blocking (ttnn.from_device(part, blocking=False)), one synchronise, then the host copies: whether the
            runtime overlaps shard reads when asked to.
  wide      serial and mesh again on a (1, 1, 32, 32) bfloat16 TILE tensor, the size class of the quad draft's outputs.

An arm that raises records its error and is skipped, so a missing API in one arm costs that arm only.

CHIPS. The mesh is (1, n) over the n devices the container sees. The cardm step shows ONE card: then `serial` and `mesh` differ only by the
composer's overhead and the verdict is INCONCLUSIVE-SINGLE-CHIP (the floor L and the per-call overhead are still read: serial at four chips
is about 8 L, the packed word read about L plus the composer's overhead). The four-chip answer is the first quad window's: its [PACKED-HOSTGAP-VERIFY]
reads_ms is the served loop's cost at four chips in the same runtime, to be put beside this report's floor, per-call overhead and ratios.

Run (inside the serving image, one card): python3 -B mr_probe.py --out results/mr-<stamp>.json [--iterations 400] [--warmup 40] [--rows 64]
Report: the last stdout line is one JSON object (kind mr-probe) and the line above it 'MR_PROBE verdict=...'.
Exit: 0 any verdict; 2 the mesh could not be opened (nothing measured); 3 the watchdog.
"""

import argparse
import json
import statistics
import sys
import threading
import time

KIND = 'mr-probe'
ARMS = ('floor', 'serial', 'mesh', 'word', 'overlap', 'wide_serial', 'wide_mesh')
# A mesh read must beat the loop by this fraction (of the loop's median) at two chips or more to be called a GO.
GO_GAIN = 0.25
WATCHDOG_S = 900


def summarize(samples_ns):
    """{n, median_us, p10_us, p90_us, min_us} of a list of nanosecond samples (empty: n = 0)."""
    if not samples_ns:
        return dict(n=0)
    ordered = sorted(samples_ns)

    def at(fraction):
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] / 1000.0

    return dict(n=len(ordered), median_us=round(statistics.median(ordered) / 1000.0, 2), p10_us=round(at(0.1), 2),
                p90_us=round(at(0.9), 2), min_us=round(ordered[0] / 1000.0, 2))


def verdict(chips, results):
    """(verdict text, evidence dict) from the per-arm summaries. INCONCLUSIVE-SINGLE-CHIP below two chips (one shard cannot show batching);
    GO when the mesh read beats the serial loop by GO_GAIN and the joined word read beats the mesh read pair; MESH-ONLY when only the
    two mesh reads win; NO-GO otherwise; INCOMPLETE when serial or mesh did not run."""
    median = lambda name: results.get(name, {}).get('median_us')
    serial, mesh, word, floor = median('serial'), median('mesh'), median('word'), median('floor')
    evidence = dict(chips=chips, floor_us=floor, serial_us=serial, mesh_us=mesh, word_us=word, overlap_us=median('overlap'))
    if serial is None or mesh is None:
        return 'INCOMPLETE', evidence
    evidence['mesh_over_serial'] = round(mesh / serial, 3) if serial else None
    evidence['word_over_serial'] = round(word / serial, 3) if serial and word is not None else None
    evidence['serial_over_floor'] = round(serial / floor, 2) if floor else None
    if chips < 2:
        return 'INCONCLUSIVE-SINGLE-CHIP', evidence
    if mesh <= serial * (1 - GO_GAIN) and word is not None and word <= mesh:
        return 'GO', evidence
    if mesh <= serial * (1 - GO_GAIN):
        return 'MESH-ONLY', evidence
    return 'NO-GO', evidence


def run(ttnn, mesh, chips, iterations, warmup, rows, clock=time.perf_counter_ns):
    """Time every arm on `mesh`; returns (per-arm summaries, per-arm errors). `ttnn` is the module (a fake in the tests)."""
    import torch

    errors, samples = {}, {name: [] for name in ARMS}
    dram = ttnn.DRAM_MEMORY_CONFIG
    width = 64
    ids = ttnn.from_torch(torch.arange(width, dtype=torch.int32).reshape(1, 1, 1, width),
                          dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram,
                          mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    values = ttnn.from_torch(torch.arange(width, dtype=torch.float32).reshape(1, 1, 1, width).to(torch.bfloat16),
                             dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram,
                             mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    words = ttnn.from_torch(torch.arange(width, dtype=torch.int32).reshape(1, 1, 1, width),
                            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    tile = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=mesh, memory_config=dram, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def parts(tensor):
        return ttnn.get_device_tensors(tensor)

    def floor():
        ttnn.to_torch(parts(ids)[0]).reshape(-1)[:rows]

    def serial():
        [ttnn.to_torch(part).reshape(-1)[:rows] for part in parts(ids)]
        [ttnn.to_torch(part).reshape(-1)[:rows] for part in parts(values)]

    def mesh_read():
        composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
        ttnn.to_torch(ids, mesh_composer=composer)
        ttnn.to_torch(values, mesh_composer=composer)

    def word():
        ttnn.to_torch(words, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

    def overlap():
        host = [ttnn.from_device(part, blocking=False) for part in parts(ids) + parts(values)]
        ttnn.synchronize_device(mesh)
        [ttnn.to_torch(part) for part in host]

    def wide_serial():
        [ttnn.to_torch(part) for part in parts(tile)]

    def wide_mesh():
        ttnn.to_torch(tile, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

    arms = dict(floor=floor, serial=serial, mesh=mesh_read, word=word, overlap=overlap, wide_serial=wide_serial,
                wide_mesh=wide_mesh)
    live = list(arms)
    for name in list(live):
        try:
            for _ in range(2):
                arms[name]()
        except BaseException as error:  # noqa: BLE001 - an arm's missing API is a result
            errors[name] = '%s: %s' % (error.__class__.__name__, str(error)[:300])
            live.remove(name)
    for index in range(warmup + iterations):
        for name in (live if index % 2 == 0 else list(reversed(live))):
            started = clock()
            try:
                arms[name]()
            except BaseException as error:  # noqa: BLE001
                errors[name] = '%s: %s' % (error.__class__.__name__, str(error)[:300])
                live.remove(name)
                continue
            elapsed = clock() - started
            if index >= warmup:
                samples[name].append(elapsed)
    return {name: summarize(values_) for name, values_ in samples.items()}, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--iterations', type=int, default=400)
    parser.add_argument('--warmup', type=int, default=40)
    parser.add_argument('--rows', type=int, default=64)
    options = parser.parse_args(argv)
    if not 1 <= options.rows <= 64 or options.iterations < 20 or options.warmup < 0:
        print('refusing: rows 1..64, iterations >= 20, warmup >= 0', file=sys.stderr)
        return 2
    timer = threading.Timer(WATCHDOG_S, lambda: (print('MR_PROBE watchdog', flush=True), sys.stdout.flush(), __import__('os')._exit(3)))
    timer.daemon = True
    timer.start()
    import ttnn

    report = dict(kind=KIND, iterations=options.iterations, warmup=options.warmup, rows=options.rows)
    mesh = None
    try:
        chips = int(ttnn.get_num_devices())
        report['chips'] = chips
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, chips))
        results, errors = run(ttnn, mesh, chips, options.iterations, options.warmup, options.rows)
        report.update(arms=results, errors=errors)
        text, evidence = verdict(chips, results)
        report.update(verdict=text, evidence=evidence)
        print('MR_PROBE verdict=%s chips=%d floor_us=%s serial_us=%s mesh_us=%s word_us=%s overlap_us=%s' % (
            text, chips, evidence['floor_us'], evidence['serial_us'], evidence['mesh_us'], evidence['word_us'],
            evidence['overlap_us']), flush=True)
        status = 0
    except BaseException as error:  # noqa: BLE001
        report.update(verdict='NOT-MEASURED', error='%s: %s' % (error.__class__.__name__, str(error)[:500]))
        print('MR_PROBE verdict=NOT-MEASURED error=%s' % report['error'], flush=True)
        status = 2
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
            except BaseException:  # noqa: BLE001
                pass
    with open(options.out, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    timer.cancel()
    return status


if __name__ == '__main__':
    sys.exit(main())
