#!/usr/bin/env python3
"""The four-card packed GDN pieces on ONE card (S2T-01 / S2T-10, HW-A): user-batch and K5 at 12 value heads, the
single-user native twin, and the DMA kernels' four-card siblings at 192 / 80 pages.

Run with QWEN_FAST_TP=4 in the environment (the width is a launch variable: tp_shapes reads it at import). It opens a
1x1 mesh; the builders that serve one chip natively (gdn_user_batch_tp, K5) run on it directly, and those that loop over
chip_count() chips (the native twin, the state / conv / commit DMA builders) run under chip_view.ChipView, which shows the
one chip as four and launches chip 0's program. Sections (--sections):

  UB   gdn_user_batch_tp at H = 12, four users, 48 cores: (a) ONE batched launch == four per-user launches, bit for bit,
       every output and prefix-state byte and the untouched initial states; (b) HEAD-SLICE EQUIVALENCE: the recurrence is
       per value head (three value heads per key head at both widths), so the 12 heads' outputs and states equal the first
       12 heads of the PINNED pair launch (gdn_user_batch, 24 heads, the qualified program) fed the same twelve heads plus
       twelve others - an anchor on a program that ran at 24 heads, independent of anything four-card.
  K5   gdn_seq_block.execute (the qualified K5-A build) at H = 12 == the served user-batch launch at H = 12 (same inputs),
       and == the pinned pair K5 launch's first twelve heads (the same slice argument). The K5 reader / writer take H, Ct
       and the tile offsets as compile-time arguments and have never run at H = 12.
  NT   gdn_multitoken_tp.execute (the single-user native launch at H = 12, under ChipView) == the batched arm, per user.
  DMA  the sibling kernels at 192 recurrent-state pages and 80 conv pages, against the LOGICAL reference of what each
       kernel is (test_gdn_tp4_card_test holds each reference to a tile-level simulation of the kernel source):
         state copy (compact -> compact), conv windows (one launch, four windows), conv prefix copy (every prefix),
         commit DMA (entry / history -> native + checkpoint, every prefix).
  PARITY  (its own run, QWEN_FAST_TP UNSET: the pair's widths) the control on the templating: the same four operations at
       the pair's shapes (24 heads, 5120 channels, 384 / 160 pages) with the PINNED kernels, then with each _tp sibling
       built with the pair's page counts as defines (through the same builders and the commit twin): every output byte of
       the two arms must be equal. It needs no reference: it shows the sibling IS the pinned kernel at the pair's numbers.

Every comparison is bit for bit on bf16. The verdict line is 'GDN_TP4 verdict=PASS|FAIL|NO-DECISION scope=full|reduced|
parity'. A per-section exception is recorded and the run continues; a section that raised is NO-DECISION, never PASS.

  QWEN_FAST_TP=4 python3 gdn_tp4_card_test.py --out results.json
  python3 gdn_tp4_card_test.py --out parity.json --sections PARITY
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

VERDICT = 'GDN_TP4'
FOUR_SECTIONS = ('UB', 'K5', 'NT', 'DMA')
SECTIONS = FOUR_SECTIONS + ('PARITY',)
ROWS = 16
USERS = 4
TILE = 32
PAGE = 2048


# ---------------------------------------------------------------------------------------------
# Pure helpers (torch injected; no ttnn).
# ---------------------------------------------------------------------------------------------

def same_bits(torch, left, right):
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    width = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(left.element_size())
    if width is None:
        return torch.equal(left, right)
    return torch.equal(left.contiguous().view(width), right.contiguous().view(width))


def difference(torch, left, right):
    if tuple(left.shape) != tuple(right.shape):
        return dict(shape=[list(left.shape), list(right.shape)])
    gap = (left.float() - right.float()).abs()
    return dict(differing=int((gap > 0).sum()), of=int(gap.numel()), max_abs=float(gap.max()))


def user_inputs(torch, found, generator, rows=ROWS):
    """One user's (qkv, beta, gate, initial, z) at the width's shapes: beta in (0, 1), the log-gate negative, a small
    initial state, as gdn_user_batch_device_test draws them."""
    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    def rand(*shape):
        return torch.rand(*shape, generator=generator)

    return (randn(1, rows, found.gdn_qkv).bfloat16(), rand(1, rows, found.gdn_nv).bfloat16(),
            (-rand(1, rows, found.gdn_nv)).bfloat16(), (randn(1, found.gdn_nv, 128, 128) * 0.05).bfloat16(),
            randn(1, rows, found.gdn_z).bfloat16())


def widen_to_pair(torch, four, spare, found4, found2):
    """The pair-width inputs whose first 12 value heads are `four`'s: q and k gain four key heads, v twelve value heads,
    beta, gate, the initial state and z twelve value heads, all from `spare` (the same shapes as `four`, other data).
    The value head -> key head map is h // 3 at both widths, so heads 0-11 keep key heads 0-3."""
    qkv, beta, gate, initial, z = four
    extra_qkv, extra_beta, extra_gate, extra_initial, extra_z = spare
    key = found4.gdn_key
    q, k, v = qkv[..., :key], qkv[..., key:2 * key], qkv[..., 2 * key:]
    eq, ek, ev = extra_qkv[..., :key], extra_qkv[..., key:2 * key], extra_qkv[..., 2 * key:]
    wide_qkv = torch.cat([q, eq, k, ek, v, ev], dim=-1)
    if wide_qkv.shape[-1] != found2.gdn_qkv:
        raise AssertionError('widened qkv is %d wide, not %d' % (wide_qkv.shape[-1], found2.gdn_qkv))
    return (wide_qkv, torch.cat([beta, extra_beta], dim=-1), torch.cat([gate, extra_gate], dim=-1),
            torch.cat([initial, extra_initial], dim=1), torch.cat([z, extra_z], dim=-1))


def head_slice(torch, wide_output, wide_states, found4):
    """The first twelve value heads of a pair-width launch's (output, states)."""
    return wide_output[..., :found4.gdn_value], wide_states[:, :found4.gdn_nv]


# ---- DMA references: what each kernel is, in logical (row, column) terms -------------------------------------------

def windows_reference(torch, piece, history, rows, channels):
    """gdn_conv_windows.cpp: window `slot` row `token` is, with h = token + slot, the first row of history tensor h when
    h < 4, else row h - 4 of the projected piece's first `channels` columns. Returns four (1, rows, channels)."""
    out = []
    for slot in range(4):
        window = torch.zeros(1, rows, channels, dtype=piece.dtype)
        for token in range(rows):
            h = token + slot
            window[0, token] = history[h][0, 0, :channels] if h < 4 else piece[0, h - 4, :channels]
        out.append(window)
    return out


def prefix_reference(torch, windows, prefix, channels):
    """gdn_conv_prefix_copy.cpp: destination `slot`'s row 0 is row prefix - 1 of window `slot`. Four (1, 1, channels)."""
    return [window[:1, prefix - 1:prefix, :channels].clone() for window in windows]


def commit_reference(torch, entry, history, prefix, rows):
    """gdn_commit_dma.cpp: the record to publish is `entry` at prefix 0, else record prefix - 1 of the history.
    entry / history are five-tensor lists: (1, H, 128, 128) + four (1, 1, C) for the entry; (rows, H, 128, 128) + four
    (1, rows, C) for the history. Returns the five published tensors: (1, H, 128, 128) + four (1, 1, C) row-0 values."""
    if prefix == 0:
        return [entry[0].clone()] + [entry[slot][:1, :1].clone() for slot in range(1, 5)]
    return ([history[0][prefix - 1:prefix].clone()]
            + [history[slot][:1, prefix - 1:prefix].clone() for slot in range(1, 5)])


# ---- report -------------------------------------------------------------------------------------------------------

def comparison(section, kind, label, equal, **details):
    return dict(section=section, kind=kind, label=label, equal=bool(equal), **details)


def decide(report):
    """PASS / FAIL / NO-DECISION and the scope: every requested section decisive and equal; a raised section, a section
    that made no comparison, or a cut run is NO-DECISION."""
    sections = report.get('sections', {})
    requested = report.get('requested', [])
    comparisons = report.get('comparisons', [])
    problems = []
    for name in requested:
        state = sections.get(name) or {}
        if state.get('error'):
            problems.append('%s raised' % name)
        elif not any(entry['section'] == name for entry in comparisons):
            problems.append('%s made no comparison' % name)
    if problems:
        return dict(verdict='NO-DECISION', problems=problems)
    if any(not entry['equal'] for entry in comparisons):
        return dict(verdict='FAIL', problems=['%d comparisons differ' % sum(1 for e in comparisons if not e['equal'])])
    return dict(verdict='PASS', problems=[])


def scope_of(report):
    if report.get('requested') == ['PARITY']:
        return 'parity'
    return 'full' if set(report.get('requested', [])) == set(FOUR_SECTIONS) and not report.get('reduced') else 'reduced'


def verdict_line(report):
    decision = report['decision']
    tallies = report.get('tally', {})
    return '%s verdict=%s scope=%s tp=%s chips=1of%s comparisons=%d differing=%d sections=%s' % (
        VERDICT, decision['verdict'], scope_of(report), report.get('tp'), report.get('tp'),
        tallies.get('comparisons', 0), tallies.get('differing', 0), ','.join(report.get('requested', [])))


def tally(comparisons):
    return dict(comparisons=len(comparisons), differing=sum(1 for entry in comparisons if not entry['equal']))


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sections', default=','.join(FOUR_SECTIONS), help='%s, or PARITY alone (QWEN_FAST_TP unset)'
                        % ','.join(FOUR_SECTIONS))
    parser.add_argument('--seeds', default='17,18,19', help='UB / K5 / NT seeds')
    parser.add_argument('--prefixes', default='0,1,7,16', help='DMA prefixes (rows = 16)')
    parser.add_argument('--iterations', type=int, default=20, help='timing launches (0: no timing)')
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('TT_METAL_HOME', '/opt/tt-metal')))
    # accepted for run_card_b.sh, which passes them to every harness (the container timeout and the graft's sha are its own)
    parser.add_argument('--watchdog', type=float, default=0.0)
    parser.add_argument('--deadline-s', type=float, default=0.0)
    parser.add_argument('--expect-binary-sha256', default='')
    arguments = parser.parse_args(argv)
    arguments.sections = [name for name in arguments.sections.split(',') if name]
    arguments.seeds = [int(seed) for seed in arguments.seeds.split(',') if seed]
    arguments.prefixes = [int(value) for value in arguments.prefixes.split(',') if value]
    unknown = sorted(set(arguments.sections) - set(SECTIONS))
    if unknown:
        parser.error('unknown sections %s' % unknown)
    if any(not 0 <= value <= ROWS for value in arguments.prefixes):
        parser.error('--prefixes within 0..%d' % ROWS)
    if 'PARITY' in arguments.sections and arguments.sections != ['PARITY']:
        parser.error('PARITY runs at the pair\'s widths (QWEN_FAST_TP unset), alone')
    return arguments


# ---------------------------------------------------------------------------------------------
# The device part.
# ---------------------------------------------------------------------------------------------

class Rig:
    """The open card and the helpers every section shares."""

    def __init__(self, ttnn, torch, mesh, report):
        self.ttnn, self.torch, self.mesh, self.report = ttnn, torch, mesh, report
        self.comparisons = report['comparisons']

    def upload(self, value, dtype=None, layout=None):
        ttnn = self.ttnn
        return ttnn.from_torch(value, device=self.mesh, dtype=dtype or ttnn.bfloat16,
                               layout=layout or ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def host(self, value):
        shards = self.ttnn.get_device_tensors(value)
        return self.ttnn.to_torch(shards[0]).clone()

    def free(self, *values):
        for value in values:
            self.ttnn.deallocate(value)

    def record(self, entry):
        self.comparisons.append(entry)
        print(json.dumps(entry), flush=True)

    def compare(self, section, kind, label, expected, actual, **details):
        equal = same_bits(self.torch, expected, actual)
        self.record(comparison(section, kind, label, equal, **(details if equal else dict(
            details, **difference(self.torch, expected, actual)))))
        return equal


def upload_users(rig, four_inputs, norm_weight):
    return [(rig.upload(qkv), rig.upload(beta), rig.upload(gate), rig.upload(initial), rig.upload(z), norm_weight)
            for qkv, beta, gate, initial, z in four_inputs]


def read_launch(rig, produced):
    return [(rig.host(output), rig.host(states)) for output, states in produced]


def release(rig, produced):
    for output, states in produced:
        rig.free(output, states)


def timed(rig, arm, iterations):
    samples = []
    for _ in range(2):
        release(rig, arm())
    rig.ttnn.synchronize_device(rig.mesh)
    for _ in range(iterations):
        start = time.perf_counter()
        produced = arm()
        rig.ttnn.synchronize_device(rig.mesh)
        samples.append((time.perf_counter() - start) * 1000)
        release(rig, produced)
    samples.sort()
    return dict(median_ms=samples[len(samples) // 2], best_ms=samples[0], worst_ms=samples[-1], samples=len(samples))


def run_ub_k5(rig, arguments, found4, found2, kernels, section):
    """UB (section 'UB') or K5 ('K5'): the launch under test against the per-user controls and the pinned pair launch."""
    import gdn_seq_block as seq
    import gdn_user_batch as pair
    import gdn_user_batch_tp as quad

    torch = rig.torch
    timings = {}
    for seed in arguments.seeds:
        generator = torch.Generator().manual_seed(seed)
        four = [user_inputs(torch, found4, generator) for _ in range(USERS)]
        spare = [user_inputs(torch, found4, generator) for _ in range(USERS)]
        norm = (1 + torch.randn(1, 1, 128, generator=generator) * 0.1).bfloat16()
        norm_weight = rig.upload(norm)
        groups = upload_users(rig, four, norm_weight)
        wide_groups = upload_users(rig, [widen_to_pair(torch, mine, other, found4, found2)
                                         for mine, other in zip(four, spare)], norm_weight)
        initial_host = [inputs[3] for inputs in four]
        try:
            def per_user_quad():
                return [quad.execute(rig.mesh, [group], kernels)[0] for group in groups]

            def batched_quad():
                return quad.execute(rig.mesh, groups, kernels)

            def wide_pair():
                return [pair.execute(rig.mesh, [group], kernels)[0] for group in wide_groups]

            def k5():
                return seq.execute(rig.mesh, groups, rig.ttnn, output_memory=rig.ttnn.L1_MEMORY_CONFIG,
                                   kernels=seq.served_kernels())

            control = per_user_quad()
            rig.ttnn.synchronize_device(rig.mesh)
            control_host = read_launch(rig, control)
            release(rig, control)
            anchor = wide_pair()
            rig.ttnn.synchronize_device(rig.mesh)
            anchor_host = read_launch(rig, anchor)
            release(rig, anchor)

            arm = batched_quad if section == 'UB' else k5
            candidate = arm()
            rig.ttnn.synchronize_device(rig.mesh)
            candidate_host = read_launch(rig, candidate)
            release(rig, candidate)
            for index in range(USERS):
                for name, position in (('output', 0), ('states', 1)):
                    rig.compare(section, '%s_vs_per_user' % ('batched' if section == 'UB' else 'k5'),
                                'seed%d/user%d/%s' % (seed, index, name), control_host[index][position],
                                candidate_host[index][position], seed=seed, user=index)
                sliced = head_slice(torch, anchor_host[index][0], anchor_host[index][1], found4)
                for name, position in (('output', 0), ('states', 1)):
                    rig.compare(section, 'vs_pinned_pair_head_slice', 'seed%d/user%d/%s' % (seed, index, name),
                                sliced[position], candidate_host[index][position], seed=seed, user=index)
            if section == 'UB':
                for index, group in enumerate(groups):
                    rig.compare(section, 'initial_state_immutable', 'seed%d/user%d' % (seed, index),
                                initial_host[index], rig.host(group[3]), seed=seed, user=index)
            if seed == arguments.seeds[0] and arguments.iterations:
                timings.update({'per_user' if section == 'UB' else 'served_batched': timed(rig, per_user_quad, arguments.iterations),
                                ('batched' if section == 'UB' else 'k5'): timed(rig, arm, arguments.iterations)})
        finally:
            for group in groups + wide_groups:
                rig.free(*group[:5])
            rig.free(norm_weight)
    return dict(timings=timings, seeds=arguments.seeds)


def run_nt(rig, arguments, found4, kernels):
    """NT: the single-user native twin under ChipView against the batched launch, per user."""
    import chip_view
    import gdn_multitoken_tp as twin
    import gdn_user_batch_tp as quad

    torch = rig.torch
    for seed in arguments.seeds[:1]:
        generator = torch.Generator().manual_seed(seed)
        four = [user_inputs(torch, found4, generator) for _ in range(USERS)]
        norm_weight = rig.upload((1 + torch.randn(1, 1, 128, generator=generator) * 0.1).bfloat16())
        groups = upload_users(rig, four, norm_weight)
        try:
            batched = quad.execute(rig.mesh, groups, kernels)
            rig.ttnn.synchronize_device(rig.mesh)
            expected = read_launch(rig, batched)
            release(rig, batched)
            view = chip_view.ChipView(rig.ttnn, chips=4)
            with view.installed():
                for index, group in enumerate(groups):
                    output, states = twin.execute(rig.mesh, group[0], group[1], group[2], group[3], kernels,
                                                  z=group[4], norm_w=group[5])
                    rig.ttnn.synchronize_device(rig.mesh)
                    for name, value, reference in (('output', output, expected[index][0]),
                                                   ('states', states, expected[index][1])):
                        rig.compare('NT', 'native_twin_vs_batched', 'seed%d/user%d/%s' % (seed, index, name),
                                    reference, rig.host(value), seed=seed, user=index)
                    rig.free(output, states)
        finally:
            for group in groups:
                rig.free(*group[:5])
            rig.free(norm_weight)
    return {}


def dma_operations(rig, found, prefixes, publish, seed=23, layers=2):
    """The four DMA operations on seeded inputs at `found`'s widths, through the builders as they are: state copy,
    conv windows (from a projected piece and four history rows), conv prefix copy of those windows at every prefix, and
    commit publication at every prefix. Returns (inputs, outputs), host tensors keyed by name."""
    import gdn_conv_prefix_copy
    import gdn_conv_windows
    import gdn_state_copy

    torch, ttnn = rig.torch, rig.ttnn
    generator = torch.Generator().manual_seed(seed)
    channels, heads = found.gdn_qkv, found.gdn_nv

    def rand(*shape):
        return torch.randn(*shape, generator=generator).bfloat16()

    inputs, outputs = {}, {}
    source = [rand(1, heads, 128, 128)] + [rand(1, 1, channels) for _ in range(4)]
    inputs['state'] = source
    device_source = [rig.upload(value) for value in source]
    device_destination = [rig.upload(torch.zeros_like(value)) for value in source]
    try:
        gdn_state_copy.copy_compact(device_source, device_destination)
        ttnn.synchronize_device(rig.mesh)
        outputs['state'] = [rig.host(value) for value in device_destination]
    finally:
        rig.free(*device_source, *device_destination)

    piece = rand(1, ROWS, found.gdn_qkvzab_padded)
    history = [rand(1, 1, channels) for _ in range(4)]
    inputs['piece'], inputs['history'] = piece, history
    device_piece, device_history = rig.upload(piece), [rig.upload(value) for value in history]
    windows = []
    try:
        windows = gdn_conv_windows.build_windows(rig.mesh, device_piece, device_history)
        ttnn.synchronize_device(rig.mesh)
        outputs['windows'] = [rig.host(value) for value in windows]
        for prefix in [value for value in prefixes if value >= 1]:
            destinations = [rig.upload(torch.zeros(1, 1, channels).bfloat16()) for _ in range(4)]
            try:
                gdn_conv_prefix_copy.copy_prefix(rig.mesh, windows, destinations, prefix)
                ttnn.synchronize_device(rig.mesh)
                outputs['prefix%d' % prefix] = [rig.host(value) for value in destinations]
            finally:
                rig.free(*destinations)
    finally:
        rig.free(device_piece, *device_history, *windows)

    for prefix in prefixes:
        device_layers, host_layers = [], []
        try:
            for _ in range(layers):
                entry = [rand(1, heads, 128, 128)] + [rand(1, 1, channels) for _ in range(4)]
                hist = [rand(ROWS, heads, 128, 128)] + [rand(1, ROWS, channels) for _ in range(4)]
                native = [torch.full((8, heads, 128, 128), 3.0).bfloat16()] + [
                    torch.full((1, 8, channels), 3.0).bfloat16() for _ in range(4)]
                checkpoint = [torch.full((1, heads, 128, 128), 5.0).bfloat16()] + [
                    torch.full((1, 1, channels), 5.0).bfloat16() for _ in range(4)]
                host_layers.append((entry, hist, native, checkpoint))
                device_layers.append([rig.upload(value) for group in (entry, hist, native, checkpoint) for value in group])
            publish(rig.mesh, device_layers, prefix)
            ttnn.synchronize_device(rig.mesh)
            inputs['commit%d' % prefix] = host_layers
            outputs['commit%d' % prefix] = [[rig.host(value) for value in layer[10:20]] for layer in device_layers]
        finally:
            for values in device_layers:
                rig.free(*values)
    return inputs, outputs


def run_dma(rig, arguments, found4):
    """DMA: the sibling kernels at 192 / 80 pages against the logical references, under ChipView."""
    import chip_view
    import gdn_commit_dma_tp

    import tp_addresses

    torch = rig.torch
    channels = found4.gdn_qkv
    view = chip_view.ChipView(rig.ttnn, chips=4)
    # The builders read validate_projected / addresses through the pinned helpers, which serving rebinds at startup
    # (tp_addresses.install). Only here: the pinned pair launch the UB and K5 sections anchor on must stay the pair's.
    tp_addresses.install()
    try:
        with view.installed():
            inputs, outputs = dma_operations(rig, found4, arguments.prefixes, gdn_commit_dma_tp.publish)
    finally:
        tp_addresses.uninstall()
    for slot, value in enumerate(inputs['state']):
        rig.compare('DMA', 'state_copy', 'slot%d' % slot, value, outputs['state'][slot])
    reference = windows_reference(torch, inputs['piece'], inputs['history'], ROWS, channels)
    for slot in range(4):
        rig.compare('DMA', 'conv_windows', 'slot%d' % slot, reference[slot], outputs['windows'][slot])
    for prefix in [value for value in arguments.prefixes if value >= 1]:
        expected = prefix_reference(torch, outputs['windows'], prefix, channels)
        for slot in range(4):
            rig.compare('DMA', 'conv_prefix_copy', 'prefix%d/slot%d' % (prefix, slot), expected[slot],
                        outputs['prefix%d' % prefix][slot], prefix=prefix)
    for prefix in arguments.prefixes:
        for index, (entry, hist, native, checkpoint) in enumerate(inputs['commit%d' % prefix]):
            published = commit_reference(torch, entry, hist, prefix, ROWS)
            got_native, got_checkpoint = (outputs['commit%d' % prefix][index][:5],
                                          outputs['commit%d' % prefix][index][5:])
            label = 'prefix%d/layer%d' % (prefix, index)
            rig.compare('DMA', 'commit_native_record', label, published[0], got_native[0][:1], prefix=prefix)
            rig.compare('DMA', 'commit_native_untouched_slots', label, native[0][1:], got_native[0][1:], prefix=prefix)
            rig.compare('DMA', 'commit_checkpoint_record', label, published[0], got_checkpoint[0], prefix=prefix)
            for slot in range(1, 5):
                slot_label = '%s/slot%d' % (label, slot)
                rig.compare('DMA', 'commit_native_conv_row0', slot_label, published[slot], got_native[slot][:1, :1],
                            prefix=prefix)
                rig.compare('DMA', 'commit_native_conv_other_rows', slot_label, native[slot][:, 1:],
                            got_native[slot][:, 1:], prefix=prefix)
                rig.compare('DMA', 'commit_checkpoint_conv_row0', slot_label, published[slot],
                            got_checkpoint[slot][:1, :1], prefix=prefix)
    return dict(pages=dict(state=found4.gdn_state_pages, conv=found4.gdn_conv_pages))


def run_parity(rig, arguments, found2):
    """PARITY: at the pair's widths the four operations with the pinned kernels, then with the _tp siblings built with the
    pair's page counts as defines: every output byte must agree."""
    import chip_view
    import gdn_commit_dma
    import gdn_commit_dma_tp
    import tp_kernels

    def sibling(path, environ=None):
        path = Path(path)
        return str(path.with_name(path.stem + '_tp' + path.suffix))

    def pair_defines(environ=None):
        return [('QWEN_STATE_PAGES', str(found2.gdn_state_pages)), ('QWEN_CONV_PAGES', str(found2.gdn_conv_pages)),
                ('QWEN_CONV_TASKS', str(4 * found2.gdn_conv_pages))]

    view = chip_view.ChipView(rig.ttnn, chips=2)
    runs = {}
    original = (tp_kernels.source, tp_kernels.defines)
    with view.installed():
        try:
            runs['pinned'] = dma_operations(rig, found2, arguments.prefixes, gdn_commit_dma.publish)[1]
            tp_kernels.source, tp_kernels.defines = sibling, pair_defines
            runs['sibling'] = dma_operations(rig, found2, arguments.prefixes, gdn_commit_dma_tp.publish)[1]
        finally:
            tp_kernels.source, tp_kernels.defines = original
    for name in sorted(runs['pinned']):
        pinned, twin = runs['pinned'][name], runs['sibling'][name]
        flat = lambda value: [tensor for item in value for tensor in (item if isinstance(item, list) else [item])]
        for index, (left, right) in enumerate(zip(flat(pinned), flat(twin))):
            rig.compare('PARITY', 'sibling_vs_pinned_at_pair_counts', '%s/%d' % (name, index), left, right)
    return dict(pages=dict(state=found2.gdn_state_pages, conv=found2.gdn_conv_pages),
                launches=dict(view_realised=view.realised, phantom=view.phantom))


def run(arguments, report):
    import torch
    import ttnn

    import tp_shapes

    tp = tp_shapes.chip_count()
    report['tp'] = tp
    parity = arguments.sections == ['PARITY']
    if tp != (2 if parity else 4):
        raise SystemExit('%s: this run needs %s, got a width of %d' % (
            'PARITY compares kernels at the pair\'s widths' if parity else 'the four-card sections qualify four cards',
            'QWEN_FAST_TP unset' if parity else 'QWEN_FAST_TP=4', tp))
    found4, found2 = tp_shapes.geometry(4), tp_shapes.geometry(2)
    import gdn_user_batch as pair

    kernels = pair.load_kernels(arguments.root)
    report['generated_sha256'] = {role: hashlib.sha256(source.encode()).hexdigest() for role, source in kernels.items()}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        rig = Rig(ttnn, torch, mesh, report)
        for name in arguments.sections:
            print('--- section %s' % name, flush=True)
            state = report['sections'].setdefault(name, {})
            try:
                if name in ('UB', 'K5'):
                    state.update(run_ub_k5(rig, arguments, found4, found2, kernels, name))
                elif name == 'NT':
                    state.update(run_nt(rig, arguments, found4, kernels))
                elif name == 'DMA':
                    state.update(run_dma(rig, arguments, found4))
                elif name == 'PARITY':
                    state.update(run_parity(rig, arguments, found2))
            except SystemExit:
                raise
            except BaseException as error:  # noqa: BLE001 - recorded; the run continues
                state['error'] = repr(error)
                state['traceback'] = traceback.format_exc()
                print(state['traceback'], flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


def main(argv=None):
    arguments = parse(argv)
    report = dict(scope='single-card four-card-geometry GDN qualification (S2T-01 / S2T-10)', argv=list(sys.argv[1:]),
                  requested=list(arguments.sections), seeds=arguments.seeds, prefixes=arguments.prefixes,
                  reduced=len(arguments.seeds) < 3 or (set(arguments.sections) != set(FOUR_SECTIONS)
                                                       and arguments.sections != ['PARITY']),
                  sections={}, comparisons=[], env={name: os.environ.get(name) for name in ('QWEN_FAST_TP',)})

    def write():
        report['tally'] = tally(report['comparisons'])
        arguments.out.write_text(json.dumps(report, indent=2, default=str))

    try:
        run(arguments, report)
    except SystemExit as stop:
        report['error'] = str(stop)
    except BaseException as error:  # noqa: BLE001
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
    report['decision'] = decide(report) if not report.get('error') else dict(verdict='NO-DECISION',
                                                                            problems=[report['error']])
    report['tally'] = tally(report['comparisons'])
    report['verdict_line'] = verdict_line(report)
    write()
    print(report['verdict_line'], flush=True)
    return 0 if report['decision']['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
