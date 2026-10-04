"""U2 on ONE card: every one of the 65,536 bfloat16 bit patterns through the canonical rule of gdn_rows_dma_tp (modes 3 and 4) against the served path.

The row mover (gdn_rows_dma_tp.cpp) canonicalises exactly the halves the served round trip does: canonical_half(v) = +0 when the exponent bits are zero
(-0 and every denormal; CANON_DENORM=1), else v unchanged. That rule was MEASURED on card M's data (gdn_prefill_conv_exact.py), never held against every
pattern. V2, V1, WC and K5-B rest on it; this probe decides whether they rest on a construction (every pattern checked) or on observed data only.

One tile column holds 1,024 patterns, so the 65,536 patterns fill 64 tiles: a (1, 32, 2048) bf16 TILE tensor T whose element (row r, column c) is the pattern
r * 2048 + c. Four device results from T, each read back and compared as int16 bit patterns:

  raw      the row mover's raw modes (1 and 2): must copy every byte (the control: a difference here is the mover, not the rule);
  canon    the row mover's canonical modes (3 and 4), both halves of every tile (CANON_DENORM=1, the kernel's default; the build under test);
  trip0    the served round trip on rows 0-15: untilize (to_layout ROW_MAJOR), slice, tilize (to_layout TILE);
  trip1    the same on rows 16-31;
  slice1   the served tile slice that starts inside a tile (rows 16-31), the path users 1 and 3 take.

canon rows 0-15 must equal trip0, canon rows 16-31 must equal trip1 AND slice1. The report also holds the CPU model of the rule applied to the raw bytes that
reached the device against what the served path returned (rule_vs_served), the set of patterns the served path changes, and the input fidelity: the
runtime's bf16 upload and readback may flush or alter some patterns (the F1 probe's finding), and a pattern altered before the first op is not exercised;
`exercised` counts the patterns that reached the device unchanged.

Run with QWEN_FAST_TP=1 or unset (the row mover's chip count is overridden to the one chip a 1x1 mesh has: gdn_rows_dma_tp.tp_shapes.chip_count).
Exit: 0 PASS or PASS-PARTIAL (no differing pattern; PARTIAL when fewer than all 65,536 reached the device unchanged), 1 FAIL, 3 the watchdog,
4 NOT-RUN. The last stdout line is one JSON object (kind u2-canon-card-m) and the line above it 'U2_CANON verdict=...'.
"""

import argparse
import json
import os
import sys
import threading

KIND = 'u2-canon-card-m'
WIDTH = 2048                 # 64 tile columns x 32
PATTERNS = 65536
WATCHDOG_S = 1800
LISTED = 64                  # differing patterns listed per comparison


def host_patterns(torch):
    """(1, 32, 2048) int16 holding the pattern r * 2048 + c at (r, c), and the same bits as bfloat16."""
    raw = torch.arange(PATTERNS, dtype=torch.int32).reshape(1, 32, WIDTH)
    signed = (raw - ((raw & 0x8000) << 1)).to(torch.int16)
    return signed, signed.contiguous().view(torch.bfloat16)


def canonical_model(torch, bits, denorm=True):
    """The kernel's rule on int16 bit patterns: +0 where the exponent field is zero (denorm) or the value is -0 (not denorm), else unchanged."""
    unsigned = bits.to(torch.int32) & 0xFFFF
    flush = ((unsigned & 0x7F80) == 0) if denorm else (unsigned == 0x8000)
    return torch.where(flush, torch.zeros_like(bits), bits)


def tasks(rows_dma, canonical):
    """One task a tile: destination page p from source page p, rows 0-15 and rows 16-31 each raw or canonical (modes 1/2 or 3/4)."""
    pages = WIDTH // 32
    first, second = rows_dma.mode(0, canonical), rows_dma.mode(1, canonical)
    return [(0, page, (0, page, first), (0, page, second)) for page in range(pages)]


def pattern_ids(torch, bits_host, mask):
    """The distinct host patterns (unsigned) at the True positions of `mask`, as a sorted list."""
    return sorted(set((int(value) & 0xFFFF) for value in bits_host[mask].reshape(-1).tolist()))


def compare(torch, host, left, right):
    """differing element count and the first LISTED differing patterns (unsigned, hex; `host` is the input pattern tensor of the same shape)."""
    if tuple(left.shape) != tuple(right.shape):
        return dict(differing=-1, shape_left=list(left.shape), shape_right=list(right.shape), patterns=[])
    mask = left != right
    ids = pattern_ids(torch, host, mask) if bool(mask.any()) else []
    return dict(differing=int(mask.sum()), patterns=['0x%04X' % value for value in ids[:LISTED]], patterns_total=len(ids))


def verdict(report):
    """(PASS | PASS-PARTIAL | FAIL | NOT-RUN, exit status) from the report's comparisons."""
    needed = ('raw', 'canon_half0_vs_trip0', 'canon_half1_vs_trip1', 'canon_half1_vs_slice1')
    comparisons = report.get('compare') or {}
    if report.get('error') or any(name not in comparisons for name in needed):
        return 'NOT-RUN', 4
    if any(comparisons[name].get('differing', 1) != 0 for name in needed):
        return 'FAIL', 1
    if report.get('rule_vs_served', {}).get('differing', 1) != 0:
        return 'FAIL', 1
    if report.get('exercised', 0) < PATTERNS:
        return 'PASS-PARTIAL', 0
    return 'PASS', 0


class Rig(object):
    """The device side: upload T, run the row mover and the served paths, read back as int16. `ttnn` is the module (a fake in the tests)."""

    def __init__(self, ttnn, mesh, torch, rows_dma):
        self.ttnn, self.mesh, self.torch, self.rows_dma = ttnn, mesh, torch, rows_dma
        self.made = []

    def upload(self, tensor):
        ttnn = self.ttnn
        made = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        self.made.append(made)
        return made

    def read(self, tensor):
        ttnn, torch = self.ttnn, self.torch
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).contiguous().view(torch.int16)

    def mover(self, source, canonical, canon_denorm=True):
        destination = self.upload(self.torch.zeros(1, 32, WIDTH, dtype=self.torch.bfloat16))
        self.rows_dma.launch(self.mesh, [source], [destination], tasks(self.rows_dma, canonical), canon_denorm=canon_denorm)
        self.ttnn.synchronize_device(self.mesh)
        return destination

    def trip(self, source, first, last):
        ttnn = self.ttnn
        rows = ttnn.to_layout(source, ttnn.ROW_MAJOR_LAYOUT)
        cut = ttnn.slice(rows, [0, first, 0], [1, last, WIDTH])
        back = ttnn.to_layout(cut, ttnn.TILE_LAYOUT)
        self.made.extend([rows, cut, back])
        return back

    def tile_slice(self, source, first, last):
        cut = self.ttnn.slice(source, [0, first, 0], [1, last, WIDTH])
        self.made.append(cut)
        return cut

    def release(self):
        for tensor in self.made:
            try:
                self.ttnn.deallocate(tensor)
            except BaseException:  # noqa: BLE001
                pass
        self.made = []


def measure(rig):
    """The report body: every comparison, the rule model against the served path, the input fidelity."""
    torch = rig.torch
    signed, as_bf16 = host_patterns(torch)
    device_in = rig.upload(as_bf16)
    sent = rig.read(device_in)                                   # the bytes that reached the device (a readback with no op between)
    unchanged = sent == signed
    report = dict(input_roundtrip=dict(differing=int((~unchanged).sum()), patterns=['0x%04X' % value for value in pattern_ids(torch, signed, ~unchanged)[:LISTED]]),
                  exercised=int(unchanged.sum()))
    raw = rig.read(rig.mover(device_in, False))
    canon = rig.read(rig.mover(device_in, True))
    canon_bit15 = rig.read(rig.mover(device_in, True, canon_denorm=False))
    trip0 = rig.read(rig.trip(device_in, 0, 16))
    trip1 = rig.read(rig.trip(device_in, 16, 32))
    slice1 = rig.read(rig.tile_slice(device_in, 16, 32))
    report['compare'] = dict(
        raw=compare(torch, signed, raw, sent),
        canon_half0_vs_trip0=compare(torch, signed[:, 0:16], canon[:, 0:16], trip0[:, 0:16]),
        canon_half1_vs_trip1=compare(torch, signed[:, 16:32], canon[:, 16:32], trip1[:, 0:16]),
        canon_half1_vs_slice1=compare(torch, signed[:, 16:32], canon[:, 16:32], slice1[:, 0:16]))
    # the CPU model of the kernel's rule, applied to the bytes that reached the device, against the served round trip's two halves
    model = canonical_model(torch, sent)
    mismatch = int((model[:, 0:16] != trip0[:, 0:16]).sum()) + int((model[:, 16:32] != trip1[:, 0:16]).sum())
    report['rule_vs_served'] = dict(differing=mismatch)
    # what the served path changes: the patterns whose round trip is not the identity (the rule, as observed)
    changed0 = trip0[:, 0:16] != sent[:, 0:16]
    changed1 = trip1[:, 0:16] != sent[:, 16:32]
    changed = pattern_ids(torch, sent[:, 0:16], changed0) + pattern_ids(torch, sent[:, 16:32], changed1)
    changed = sorted(set(changed))
    report['served_changes'] = dict(patterns=len(changed), listed=['0x%04X' % value for value in changed[:LISTED]],
                                    all_exponent_zero=all((value & 0x7F80) == 0 for value in changed),
                                    all_to_positive_zero=bool(((trip0[:, 0:16][changed0] == 0).all() if bool(changed0.any()) else True)
                                                              and ((trip1[:, 0:16][changed1] == 0).all() if bool(changed1.any()) else True)),
                                    exponent_zero_patterns_in_input=len(set(value for value in range(PATTERNS) if (value & 0x7F80) == 0)))
    # information only: the bit-15-only build (CANON_DENORM=0) against the same served halves
    report['info_canon_denorm_off_vs_trip'] = dict(
        half0=compare(torch, signed[:, 0:16], canon_bit15[:, 0:16], trip0[:, 0:16])['differing'],
        half1=compare(torch, signed[:, 16:32], canon_bit15[:, 16:32], trip1[:, 0:16])['differing'])
    return report


def main(argv=None, torch=None, ttnn=None, rows_dma=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    options = parser.parse_args(argv)
    timer = threading.Timer(WATCHDOG_S, lambda: (print('U2_CANON watchdog', flush=True), os._exit(3)))
    timer.daemon = True
    timer.start()
    if torch is None:
        import torch
        import ttnn
        import gdn_rows_dma_tp as rows_dma

        rows_dma.tp_shapes.chip_count = lambda environ=None: 1        # one card, one chip: the mover's four-chip check does not apply
    report = dict(kind=KIND, patterns=PATTERNS, environment=dict(QWEN_FAST_TP=os.environ.get('QWEN_FAST_TP')))
    mesh, rig, status = None, None, 4
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        rig = Rig(ttnn, mesh, torch, rows_dma)
        report.update(measure(rig))
        text, status = verdict(report)
    except BaseException as error:  # noqa: BLE001
        report['error'] = '%s: %s' % (error.__class__.__name__, str(error)[:500])
        text, status = 'NOT-RUN', 4
    finally:
        if rig is not None:
            rig.release()
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
            except BaseException:  # noqa: BLE001
                pass
    report['verdict'] = text
    compares = report.get('compare', {})
    print('U2_CANON verdict=%s exercised=%s raw_differing=%s canon_vs_served_differing=%s rule_vs_served_differing=%s%s' % (
        text, report.get('exercised'), compares.get('raw', {}).get('differing'),
        sum(max(compares.get(name, {}).get('differing', 0), 0) for name in compares if name != 'raw') if compares else None,
        report.get('rule_vs_served', {}).get('differing'), (' error=%s' % report['error']) if report.get('error') else ''), flush=True)
    with open(options.out, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    timer.cancel()
    return status


if __name__ == '__main__':
    sys.exit(main())
