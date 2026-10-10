"""Per-card host->device bandwidth on the served (1, 4) mesh: job F1 of the fabric-upload pack (docs/tp4-fabric-upload.md,
scripts/ci/references/fabric-upload-jobs). qwen-c2-serving.yml 'fabric' with C2_FABRIC_PROBE=h2d, C2_CARDS=quad:

    python3 -B /c2/scripts/ci/tp4_h2d_probe.py --fabric FABRIC_1D --output /probe-results/h2d-probe.json

WHY. The only host->device rate on record is a restore's 1.15-1.18 GB/s for the whole mesh (78.4 MB of GDN checkpoint in
66.5-68.0 ms, conversion included): about a tenth of even an x4 Gen5 link. Before any upload is moved onto the fabric this
job says, per card, what its own PCIe delivers, whether the x4 cards are slower than the x16 cards at all, whether four
cards written together overlap, how much of a restore is host conversion, and whether the host can write ONE card's
shard of a mesh tensor (the relay's host leg). fabric_upload_plan.decide reads its report.

ARMS (in this order; the report is rewritten after each):
  links     per mesh position: chip id, PCI index (UMD chips_with_mmio), PCIe speed/width from sysfs, the link's payload
            ceiling, the chain of ports above it (the two cards behind the switch share its upstream), whether the card
            sits in an IOMMU group (the pinned zero-copy write path for writes over 32 MiB needs the IOMMU on)
  subset    the relay's host leg: write ONE coordinate of a 16 MiB-per-card mesh tensor that holds a sentinel on every card,
            two ways. view: a single-device host tensor into ttnn.get_device_tensors(t)[i] (read from the pinned runtime: a
            1x1 host buffer is replicated to the whole mesh, so 'broadcast' is the expected status). mapper: a host tensor
            built with a one-coordinate mapper (MeshMapperConfig(Replicate, Replicate, mesh_shape_override (1, 1),
            mesh_offset_override (0, i))), whose other shards are absent, so only coordinate i is written. Per position and
            method: exact (only that shard changed), broadcast (the others took the bytes too), missing (nothing landed),
            corrupt (a shard holds neither its old bytes nor the new ones) or refused (an error); and how many coordinates
            the tensor still covers afterwards (point_to_point needs both of its pair's)
  alone     each card alone: through the mapper write when subset qualified it (the parent mesh, the relay's own path),
            else through a (1, 1) submesh (its own allocator: the parent holds no tensor while it runs, and the queues
            are drained before and after). A pre-converted host tensor copied into a preallocated DRAM tensor, copy +
            synchronize timed (no conversion inside the timer; the first call reported apart: pinning, if any).
            rm16k (ROW_MAJOR bf16, 16 KiB pages) at --sizes-mib (16 and 64 straddle the 32 MiB pinned-path threshold),
            tile_bf16 (2 KiB pages) and tile_bf8 (bfloat8_b, 1,088 B pages: the weights' and the KV's page) at
            --tile-sizes-mib. Every size read back once and compared (exact), and that read timed (d2h)
  together  the four cards in one call: a ShardTensorToMesh host tensor, the same bytes per card as alone at
            --together-mib (rm16k and tile_bf8), and a ReplicateTensorToMesh one (the replicated weights' path)
  restore   the production GDN checkpoint shape (48 layers; rec_state 12x128x128 and conv_carry 3x2,560 per chip, bf16,
            TILE) through _qwen_prefix_restore's h2d path: host conversion and the copies timed apart
  convert   host-only: ttnn.from_torch(device=None) of --together-mib of bf16 into each format (what a cache miss or a
            restore spends before any PCIe transfer)

VERDICT (last line 'H2D_PROBE verdict=...'): MEASURED (every arm ran, every read-back exact; exit 0), INEXACT (a read-back
differed; exit 1), PARTIAL (an arm errored; exit 2), NOT-MEASURED (refused before the open, or the mesh did not open;
exit 2); the watchdog exits 3. Timings are reported, never judged here: fabric_upload_plan.decide judges them.
"""

import argparse
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fabric_upload_plan as plan  # noqa: E402
import tp4_mesh  # noqa: E402
import tp4_upload_probe_common as common  # noqa: E402

KIND = 'h2d-probe-quad'
TAG = 'H2D_PROBE'
ROW_WIDTH = 8192
TILE_WIDTH = 1024
SIZES_MIB = (4, 16, 64, 256, 1024)
TILE_SIZES_MIB = (64, 256)
TOGETHER_MIB = 256
SUBSET_MIB = 16
REPEATS, WARM = 5, 1
# (layout attribute, dtype attribute, row width, device bytes per element as a fraction 1088/1024 for bfloat8_b).
FORMATS = (('rm16k', 'ROW_MAJOR_LAYOUT', 'bfloat16', ROW_WIDTH, (2, 1)),
           ('tile_bf16', 'TILE_LAYOUT', 'bfloat16', TILE_WIDTH, (2, 1)),
           ('tile_bf8', 'TILE_LAYOUT', 'bfloat8_b', TILE_WIDTH, (1088, 1024)))
# The production GDN checkpoint per chip (docs/tp4-fabric-upload.md section 1): 78,446,592 B for the mesh.
GDN_LAYERS = 48
GDN_REC = (12, 128, 128)
GDN_CARRY = (3, 2560)
STATUS = {'MEASURED': 0, 'INEXACT': 1}
ARMS = ('subset', 'alone', 'together', 'restore', 'convert')


def fmt(name):
    for entry in FORMATS:
        if entry[0] == name:
            return entry
    raise KeyError(name)


def shape_for(name, mib):
    """(rows, width, elements, device bytes) of a `mib` MiB bf16 source tensor in format `name` (rows whole tiles)."""
    _, layout, _, width, (num, den) = fmt(name)
    elements = int(mib) * (1 << 20) // 2
    rows = elements // width
    if layout == 'TILE_LAYOUT':
        rows -= rows % 32
    elements = rows * width
    return rows, width, elements, elements * num // den


def gdn_checkpoint_bytes(chips=tp4_mesh.DEVICES, layers=GDN_LAYERS):
    """Logical bytes of the restore arm's checkpoint: 78,446,592 at four chips and 48 layers (the production number)."""
    per_layer = 1
    for dim in GDN_REC:
        per_layer *= dim
    carry = GDN_CARRY[0] * GDN_CARRY[1]
    return chips * layers * (per_layer + carry) * 2


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fabric', choices=tp4_mesh.FABRIC_CONFIGS, default=tp4_mesh.FABRIC_CONFIG)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sizes-mib', default=','.join(str(size) for size in SIZES_MIB))
    parser.add_argument('--tile-sizes-mib', default=','.join(str(size) for size in TILE_SIZES_MIB))
    parser.add_argument('--together-mib', type=int, default=TOGETHER_MIB)
    parser.add_argument('--repeats', type=int, default=REPEATS)
    parser.add_argument('--warm', type=int, default=WARM)
    parser.add_argument('--subset-mib', type=int, default=SUBSET_MIB)
    parser.add_argument('--restore-layers', type=int, default=GDN_LAYERS)
    parser.add_argument('--arms', default='subset,alone,together,restore,convert')
    return parser


def sizes(text):
    values = [int(part) for part in (text or '').split(',') if part.strip()]
    if not values or any(value < 1 or value > 2048 for value in values):
        raise ValueError('sizes are MiB in 1..2048, got %r' % text)
    return values


def verdict(report):
    """MEASURED | INEXACT | PARTIAL | NOT-MEASURED from a finished report."""
    if not report.get('opened'):
        return 'NOT-MEASURED'
    records = list(report.get('arms', {}).values())
    if any(record.get('exact') is False for record in records):
        return 'INEXACT'
    if report.get('problems') or any(record.get('error') for record in records) or not records:
        return 'PARTIAL'
    return 'MEASURED'


def environment(environ):
    """What shapes a host->device rate besides the link: the runtime's switches, the CPUs and hugepages the container got."""
    out = dict((key, value) for key, value in environ.items() if key.startswith('TT_METAL_') or key.startswith('TTNN_'))
    try:
        out['cpus_usable'] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        out['cpus_usable'] = os.cpu_count()
    for name in ('nr_hugepages', 'free_hugepages'):
        out['hugepages_1g_' + name] = plan._read('/sys/kernel/mm/hugepages/hugepages-1048576kB/' + name)
    try:
        out['iommu_units'] = len(os.listdir('/sys/class/iommu'))
    except OSError:
        out['iommu_units'] = None
    return out


class Probe(object):
    def __init__(self, ttnn, torch, mesh, report, options, log):
        self.ttnn, self.torch, self.mesh, self.report, self.options, self.log = ttnn, torch, mesh, report, options, log
        self.positions = tp4_mesh.DEVICES

    # -- helpers ---------------------------------------------------------------------------------------------------
    def host(self, name, rows, width, offset, mapper=None, count=1):
        ttnn, torch = self.ttnn, self.torch
        _, layout, dtype, _, _ = fmt(name)
        parts = [common.pattern(torch, rows * width, offset=offset + index).reshape(rows, width) for index in range(count)]
        source = parts[0] if count == 1 else torch.cat(parts, dim=0)
        kwargs = dict(dtype=getattr(ttnn, dtype), layout=getattr(ttnn, layout), device=None)
        if mapper is not None:
            kwargs['mesh_mapper'] = mapper
        return ttnn.from_torch(source, **kwargs)

    def allocate(self, name, rows, width, device):
        ttnn = self.ttnn
        _, layout, dtype, _, _ = fmt(name)
        return ttnn.allocate_tensor_on_device(ttnn.Shape([rows, width]), getattr(ttnn, dtype), getattr(ttnn, layout),
                                              device, ttnn.DRAM_MEMORY_CONFIG)

    def coord(self, pos):
        ttnn = self.ttnn
        try:
            return ttnn.MeshCoordinate(0, int(pos))
        except TypeError:
            return ttnn.MeshCoordinate([0, int(pos)])

    def one_coordinate_host(self, name, rows, width, offset, pos):
        """A host tensor whose only shard is at (0, pos): the pinned runtime writes populated shards only."""
        ttnn = self.ttnn
        config = ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, 1),
                                       self.coord(pos))
        return self.host(name, rows, width, offset, mapper=ttnn.create_mesh_mapper(self.mesh, config))

    def copy_samples(self, host, target, sync_device):
        """(timed samples after the warm calls, the first call's seconds)."""
        ttnn = self.ttnn
        samples, first = [], None
        for index in range(self.options.warm + self.options.repeats):
            seconds = common.timed(lambda: ttnn.copy_host_to_device_tensor(host, target),
                                   lambda: ttnn.synchronize_device(sync_device))
            if first is None:
                first = seconds
            if index >= self.options.warm:
                samples.append(seconds)
        return samples, first

    def timed_copy(self, entry, host, target, sync_device, nbytes):
        samples, first = self.copy_samples(host, target, sync_device)
        entry.update(plan.summarize(samples, nbytes))
        entry['first_s'] = round(first, 6) if first is not None else None
        return entry

    def readback(self, target, composer=None):
        """(seconds for the device->host read, the torch tensor)."""
        import time
        ttnn = self.ttnn
        started = time.perf_counter()
        back = ttnn.from_device(target)
        seconds = time.perf_counter() - started
        return seconds, (ttnn.to_torch(back, mesh_composer=composer) if composer is not None else ttnn.to_torch(back))

    def shards(self, tensor):
        return [self.ttnn.to_torch(part) for part in self.ttnn.get_device_tensors(tensor)]

    def record(self, name, record):
        self.report.arm(name, record)
        self.log('%s arm=%s %s' % (TAG, name, ' '.join('%s=%s' % (key, record.get(key)) for key in (
            'pos', 'chip', 'width', 'nbytes', 'gbps_median', 'gbps_best', 'first_s', 'd2h_gbps', 'exact', 'status', 'error') if key in record)))
        return record

    # -- arms ------------------------------------------------------------------------------------------------------
    def links(self, **kwargs):
        found = common.positions(self.ttnn, self.mesh, **kwargs)
        self.report.data['positions'] = found
        for card in found['cards']:
            self.log('%s card pos=%d chip=%d width=x%s speed=%s GT/s ceiling=%s GB/s behind_switch=%s iommu_group=%s' % (
                TAG, card['pos'], card['chip'], card['width'], card['speed_gt'], card['ceiling_gbps'], len(card.get('ports') or []) > 1,
                card.get('iommu_group')))
        self.log('%s relay pairs (sender, receiver, hops): %s' % (TAG, found['pairs']))
        self.report.save()
        return found

    def subset(self):
        """The relay's host leg: write one coordinate of a mesh tensor, two ways; see the module docstring."""
        ttnn = self.ttnn
        rows, width, elements, nbytes = shape_for('rm16k', self.options.subset_mib)
        methods = {}
        for method in ('view', 'mapper'):
            methods[method] = self.subset_method(method, rows, width, elements, nbytes)
        qualified = [method for method in ('mapper', 'view') if methods[method]['qualified']]
        record = dict(methods=methods, nbytes=nbytes, qualified=bool(qualified), method=qualified[0] if qualified else None,
                      status=' '.join('%s=%s' % (method, methods[method]['status']) for method in ('view', 'mapper')))
        if any(entry.get('status') == 'corrupt' for method in methods.values() for entry in method['per_position']):
            record['exact'] = False   # a shard holds bytes nobody wrote: a transfer fault, not an API answer
        return self.record('subset', record)

    def subset_method(self, method, rows, width, elements, nbytes):
        ttnn, torch = self.ttnn, self.torch
        statuses = []
        staging = self.allocate('rm16k', rows, width, self.mesh)
        try:
            sentinel = self.host('rm16k', rows, width, 100, mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0), count=self.positions)
            ttnn.copy_host_to_device_tensor(sentinel, staging)
            ttnn.synchronize_device(self.mesh)
            readers = ttnn.get_device_tensors(staging)
            expected = [common.pattern(torch, elements, 100 + pos).reshape(rows, width) for pos in range(self.positions)]
            for pos in range(self.positions):
                offset = (200 if method == 'view' else 300) + pos
                want = common.pattern(torch, elements, offset).reshape(rows, width)
                entry = dict(pos=pos)
                try:
                    if method == 'view':
                        host = self.host('rm16k', rows, width, offset)
                    else:
                        host = self.one_coordinate_host('rm16k', rows, width, offset, pos)
                    handle = ttnn.get_device_tensors(staging)[pos]
                    entry['seconds'] = round(common.timed(lambda: ttnn.copy_host_to_device_tensor(host, handle),
                                                          lambda: ttnn.synchronize_device(self.mesh)), 6)
                except Exception as error:  # noqa: BLE001
                    entry.update(status='refused', error=common.error_text(error))
                    statuses.append(entry)
                    continue
                back = [ttnn.to_torch(ttnn.from_device(reader)).reshape(rows, width) for reader in readers]
                entry['coords_after'] = len(ttnn.get_device_tensors(staging))
                changed = [other for other in range(self.positions) if not torch.equal(back[other], expected[other])]
                as_new = [other for other in changed if torch.equal(back[other], want)]
                if len(as_new) != len(changed):
                    entry['status'] = 'corrupt'
                elif not changed:
                    entry['status'] = 'missing'
                elif changed == [pos]:
                    entry['status'] = 'exact'
                else:
                    entry['status'] = 'broadcast'
                entry['changed'] = changed
                expected = back
                statuses.append(entry)
        finally:
            ttnn.deallocate(staging)
        names = sorted(set(str(entry.get('status')) for entry in statuses))
        return dict(per_position=statuses, qualified=bool(statuses) and names == ['exact'] and all(
            entry.get('coords_after') == self.positions for entry in statuses), status=','.join(names))

    def submesh_targets(self):
        """[(1, 1) submesh per position] or None (the refusal recorded). The submeshes allocate on their own allocators,
        so the parent holds no tensor while they run, and every queue is drained before and after."""
        ttnn = self.ttnn
        try:
            if hasattr(self.mesh, 'quiesce_devices'):
                self.mesh.quiesce_devices()
            submeshes = list(self.mesh.create_submeshes(ttnn.MeshShape(1, 1)))
            by_chip = dict((int(sub.get_device_ids()[0]), sub) for sub in submeshes)
            self.submeshes = submeshes
            return [by_chip[int(chip)] for chip in self.mesh.get_device_ids()]
        except Exception as error:  # noqa: BLE001
            self.report.problem('submeshes refused: %s' % common.error_text(error))
            return None

    def alone(self, subset_method):
        """Each card alone: the mapper write when subset qualified it, else (1, 1) submeshes, else nothing."""
        ttnn = self.ttnn
        mode = 'mapper' if subset_method == 'mapper' else None
        devices = None
        if mode is None:
            devices = self.submesh_targets()
            mode = 'submesh' if devices else None
        self.report.data['alone_mode'] = mode
        if mode is None:
            self.report.problem('no per-card write path: the mapper write did not qualify and submeshes were refused')
            return
        cards = (self.report.data.get('positions') or {}).get('cards') or [{}] * self.positions
        plan_sizes = [('rm16k', size) for size in sizes(self.options.sizes_mib)]
        plan_sizes += [(name, size) for name in ('tile_bf16', 'tile_bf8') for size in sizes(self.options.tile_sizes_mib)]
        try:
            for name, mib in plan_sizes:
                rows, width, elements, nbytes = shape_for(name, mib)
                for pos in range(self.positions):
                    entry = dict(pos=pos, chip=cards[pos].get('chip'), width=cards[pos].get('width'), format=name, mib=mib, mode=mode)
                    target = None
                    try:
                        if mode == 'submesh':
                            host = self.host(name, rows, width, pos)
                            target = self.allocate(name, rows, width, devices[pos])
                            writer = reader = target
                            sync = devices[pos]
                        else:
                            host = self.one_coordinate_host(name, rows, width, pos, pos)
                            target = self.allocate(name, rows, width, self.mesh)
                            writer, reader = ttnn.get_device_tensors(target)[pos], ttnn.get_device_tensors(target)[pos]
                            sync = self.mesh
                        self.timed_copy(entry, host, writer, sync, nbytes)
                        seconds, back = self.readback(reader)
                        source = common.pattern(self.torch, elements, pos).reshape(rows, width)
                        expect = ttnn.to_torch(self.host(name, rows, width, pos)).reshape(rows, width) if name == 'tile_bf8' else source
                        entry.update(d2h_s=round(seconds, 6), d2h_gbps=plan.gbps(nbytes, seconds),
                                     exact=bool(self.torch.equal(back.reshape(rows, width), expect)))
                    except Exception as error:  # noqa: BLE001
                        entry['error'] = common.error_text(error)
                    finally:
                        if target is not None:
                            ttnn.deallocate(target)
                    self.record('alone/%s/%dMiB/pos%d' % (name, mib, pos), entry)
        finally:
            if mode == 'submesh' and hasattr(self.mesh, 'quiesce_devices'):
                self.mesh.quiesce_devices()

    def together(self):
        ttnn = self.ttnn
        mib = self.options.together_mib
        for name, mapper_kind in (('rm16k', 'shard'), ('tile_bf8', 'shard'), ('rm16k', 'replicate')):
            rows, width, elements, nbytes = shape_for(name, mib)
            arm = '%s/%s/%dMiB' % ('together' if mapper_kind == 'shard' else 'replicated', name, mib)
            entry = dict(format=name, mib=mib, nbytes_per_card=nbytes, nbytes=nbytes * self.positions)
            try:
                if mapper_kind == 'shard':
                    host = self.host(name, rows, width, 0, mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0), count=self.positions)
                else:
                    host = self.host(name, rows, width, 7, mapper=ttnn.ReplicateTensorToMesh(self.mesh))
                target = self.allocate(name, rows, width, self.mesh)
                try:
                    self.timed_copy(entry, host, target, self.mesh, nbytes * self.positions)
                    entry['per_card_gbps_median'] = plan.gbps(nbytes, entry.get('median_s'))
                    seconds, _ = self.readback(target, composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
                    entry.update(d2h_s=round(seconds, 6), d2h_gbps=plan.gbps(nbytes * self.positions, seconds))
                    back = self.shards(ttnn.from_device(target))
                    want = self.shards(host)
                    entry['exact'] = len(back) == len(want) == self.positions and all(
                        self.torch.equal(got.reshape(rows, width), expect.reshape(rows, width)) for got, expect in zip(back, want))
                finally:
                    ttnn.deallocate(target)
            except Exception as error:  # noqa: BLE001
                entry['error'] = common.error_text(error)
            self.record(arm, entry)

    def restore(self):
        """_qwen_prefix_restore's h2d path on the production checkpoint shape: conversion and copies timed apart."""
        import time
        ttnn, torch = self.ttnn, self.torch
        chips = self.positions
        layers = self.options.restore_layers
        entry = dict(layers=layers, logical_bytes=gdn_checkpoint_bytes(chips, layers))
        targets = []
        try:
            mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0)
            sources = []
            for layer in range(layers):
                for shape in (GDN_REC, GDN_CARRY):
                    full = (chips,) + shape
                    count = 1
                    for dim in full:
                        count *= dim
                    sources.append(common.pattern(torch, count, offset=layer).reshape(full))
                    targets.append(ttnn.allocate_tensor_on_device(ttnn.Shape([1] + list(shape)), ttnn.bfloat16,
                                                                  ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG))
            convert, copy = [], []
            for index in range(self.options.warm + self.options.repeats):
                started = time.perf_counter()
                hosts = [ttnn.from_torch(source, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=mapper)
                         for source in sources]
                converted = time.perf_counter()
                for host, target in zip(hosts, targets):
                    ttnn.copy_host_to_device_tensor(host, target)
                ttnn.synchronize_device(self.mesh)
                done = time.perf_counter()
                if index >= self.options.warm:
                    convert.append(converted - started)
                    copy.append(done - converted)
            entry['convert'] = plan.summarize(convert, entry['logical_bytes'])
            entry['copy'] = plan.summarize(copy, entry['logical_bytes'])
            entry.update(plan.summarize([a + b for a, b in zip(convert, copy)], entry['logical_bytes']))
            composer = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
            checks = []
            for index in (0, 1, len(sources) - 2, len(sources) - 1):
                back = ttnn.to_torch(targets[index], mesh_composer=composer)
                checks.append(bool(torch.equal(back.reshape(sources[index].shape), sources[index])))
            entry['exact'] = all(checks)
        except Exception as error:  # noqa: BLE001
            entry['error'] = common.error_text(error)
        finally:
            for target in targets:
                try:
                    ttnn.deallocate(target)
                except Exception:  # noqa: BLE001
                    pass
        return self.record('restore', entry)

    def convert(self):
        import time
        ttnn, torch = self.ttnn, self.torch
        mib = self.options.together_mib
        for name, layout, dtype, width, _ in FORMATS:
            rows, width, elements, nbytes = shape_for(name, mib)
            source = common.pattern(torch, elements).reshape(rows, width)
            samples = []
            entry = dict(format=name, mib=mib, source_bytes=elements * 2)
            try:
                for _ in range(max(1, min(3, self.options.repeats))):
                    started = time.perf_counter()
                    ttnn.from_torch(source, dtype=getattr(ttnn, dtype), layout=getattr(ttnn, layout), device=None)
                    samples.append(time.perf_counter() - started)
                entry.update(plan.summarize(samples, elements * 2))
            except Exception as error:  # noqa: BLE001
                entry['error'] = common.error_text(error)
            self.record('convert/%s/%dMiB' % (name, mib), entry)


def run(options, ttnn=None, torch=None, environ=None, log=print, report=None, link_kwargs=None):
    environ = os.environ if environ is None else environ
    report = report or common.Report(None, KIND)
    report.data.update(fabric=options.fabric, opened=False, serving_contract=environ.get('QWEN_C2_SERVING'),
                       environment=environment(environ))
    if ttnn is None:
        refusal, descriptor, problems = common.descriptor_refusal(environ)
        report.data.update(descriptor=descriptor, descriptor_problems=problems)
        if refusal:
            report.data['error'] = 'refused to open: ' + refusal
            report.save()
            return report.data
        import torch  # noqa: F811
        import ttnn  # noqa: F811
    arms = [arm for arm in options.arms.split(',') if arm]
    mesh = None
    probe = None
    try:
        mesh = common.open_mesh(ttnn, options.fabric)
        report.data['opened'] = True
        report.save()
        probe = Probe(ttnn, torch, mesh, report, options, log)
        probe.submeshes = ()
        probe.links(**(link_kwargs or {}))
        method = None
        if 'subset' in arms:
            method = (common.guarded(report, 'subset', probe.subset, {}) or {}).get('method')
        if 'alone' in arms:
            common.guarded(report, 'alone', lambda: probe.alone(method))
        for name in ('together', 'restore', 'convert'):
            if name in arms:
                common.guarded(report, name, getattr(probe, name))
    except Exception as error:  # noqa: BLE001
        report.data.update(error=common.error_text(error), traceback=traceback.format_exc()[-3000:])
        report.problem('stopped: ' + report.data['error'])
    finally:
        if mesh is not None:
            report.data['closed'] = common.close_mesh(ttnn, mesh, getattr(probe, 'submeshes', ()) if probe else ())
    report.data['decide_inputs'] = decide_inputs(report.data)
    report.save()
    return report.data


def decide_inputs(data):
    """The h2d half of fabric_upload_plan.decide's inputs, read off this report (None fields where an arm is missing)."""
    cards = ((data.get('positions') or {}).get('cards')) or []
    arms = data.get('arms') or {}
    alone = []
    for pos in range(len(cards) or tp4_mesh.DEVICES):
        best = None
        for name, record in arms.items():
            if name.startswith('alone/rm16k/') and record.get('pos') == pos and record.get('gbps_median'):
                if best is None or record.get('mib', 0) > best.get('mib', 0):
                    best = record
        alone.append(best.get('gbps_median') if best else None)
    together = next((record for name, record in arms.items() if name.startswith('together/rm16k/')), {})
    return dict(widths=[card.get('width') for card in cards], alone_gbps=alone, together_s=together.get('median_s'),
                together_bytes=together.get('nbytes_per_card'))


def main(argv=None, runner=run, log=print):
    options = build_parser().parse_args(argv)
    try:
        sizes(options.sizes_mib)
        sizes(options.tile_sizes_mib)
        if options.repeats < 1 or options.warm < 0 or not 1 <= options.together_mib <= 2048:
            raise ValueError('repeats >= 1, warm >= 0, together-mib in 1..2048')
        if not 1 <= options.subset_mib <= 256 or not 1 <= options.restore_layers <= GDN_LAYERS:
            raise ValueError('subset-mib in 1..256, restore-layers in 1..%d' % GDN_LAYERS)
        unknown = sorted(set(arm for arm in options.arms.split(',') if arm) - set(ARMS))
        if unknown:
            raise ValueError('unknown arms %s (known: %s)' % (', '.join(unknown), ', '.join(ARMS)))
    except ValueError as error:
        print('refusing: %s' % error, file=sys.stderr)
        return 2
    timer = common.start_watchdog(TAG, log=log)
    report = common.Report(options.output, KIND)
    data = runner(options, log=log, report=report)
    data['verdict'] = verdict(data)
    report.data = data
    report.save()
    inputs = data.get('decide_inputs') or {}
    log('%s verdict=%s opened=%s widths=%s alone_gbps=%s together_s=%s restore_s=%s problems=%d%s' % (
        TAG, data['verdict'], data.get('opened'), inputs.get('widths'), inputs.get('alone_gbps'), inputs.get('together_s'),
        ((data.get('arms') or {}).get('restore') or {}).get('median_s'), len(data.get('problems') or []),
        (' error=' + data['error']) if data.get('error') else ''))
    timer.cancel()
    return STATUS.get(data['verdict'], 2)


if __name__ == '__main__':
    sys.exit(main())
