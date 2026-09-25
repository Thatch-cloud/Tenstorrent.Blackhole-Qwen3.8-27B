"""Simulator-only functional check of gdn_prefill_conv_exact against the served FIR (lever #2).

On the local ttsim 1x2 Blackhole mesh (run-gdn-prefill-conv.sh), with different data per chip,
the op's q / k / v / new_state are compared byte for byte (int16 views) with the tree's own

    _causal_conv1d_fir(qkv_L1, None, None, 4, mesh, memory_config=L1, conv_state=carry,
                       weight_taps=taps, bias_dev=None, valid_len=vl)   + the three q/k/v slices

at a reduced width (C = 160: q, k one tile, v three tiles) so the simulator finishes. It covers
the kernels' index maths (units, ring, halo, face-row copies, state rows, routing), the
program-cache rewrites, the negative controls and the carry chain. It is a FUNCTIONAL result on
a simulator: the SFPU's rounding, denormal and tie behaviour, timing and the full 5120 width are
the card-M test's (optimisation/ttnn-op/gdn_prefill_conv).
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'ci'))
sys.path.insert(0, str(ROOT / 'optimisation' / 'ttnn-op' / 'gdn_prefill_conv'))

C, KD, K = 160, 32, 4


def main():
    spec = importlib.util.spec_from_file_location('sim_guard', Path(__file__).with_name('gdn-multitoken.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.require_simulator(os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sections', default='equality,data,negative,mirror,chain,cache,shift')
    args = parser.parse_args()
    sections = set(args.sections.split(','))
    path = Path(os.environ['QWEN_SIM_REPORT'])
    report = dict(passed=False, backend='ttsim', width=C, key_dim=KD, checks=[], failures=[],
                  scope='Functional simulator check at C=160; not a hardware exactness or timing claim.')

    def save():
        path.write_text(json.dumps(report, indent=2, default=str))

    def stage(name):
        report['last_stage'] = name
        save()
        print(json.dumps(dict(stage=name)), flush=True)

    mesh = None
    try:
        import torch
        import ttnn
        import gdn_prefill_conv_exact as pcx
        import gdn_prefill_conv_card_m as card
        from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir

        report['op_source_sha'] = pcx.source_sha()
        stage('mesh-open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]

        def upload(per_chip, memory_config=None):
            """Two host tensors (one per chip) -> one mesh tensor sharded on dim 0 (each chip [1, ...])."""
            stacked = torch.stack(per_chip).contiguous()
            return ttnn.from_torch(stacked, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))

        def read(tensor):
            return [ttnn.to_torch(part).to(torch.bfloat16) for part in ttnn.get_device_tensors(tensor)]

        def inputs(T, carry_kind, data, seed):
            xs = [card.make_x(torch, data, T, C, seed + 17 * chip) for chip in range(2)]
            carries = [card.make_carry(torch, carry_kind, C, seed + 17 * chip) for chip in range(2)]
            taps = [card.make_taps(torch, C, seed + 17 * chip) for chip in range(2)]
            qkv = upload(xs, ttnn.L1_MEMORY_CONFIG)
            carry = upload(carries) if carries[0] is not None else None
            tap_tensors = [upload([taps[chip][j:j + 1] for chip in range(2)]) for j in range(K)]
            return xs, carries, qkv, carry, tap_tensors

        def release(*tensors):
            for tensor in tensors:
                if tensor is None:
                    continue
                if isinstance(tensor, (list, tuple)):
                    release(*tensor)
                else:
                    ttnn.deallocate(tensor)

        def reference(qkv, carry, taps, vl):
            T = qkv.shape[1]
            conv, state = _causal_conv1d_fir(qkv, None, None, K, mesh, memory_config=ttnn.L1_MEMORY_CONFIG,
                                             conv_state=carry, weight_taps=taps, bias_dev=None, valid_len=vl)
            out = (ttnn.slice(conv, (0, 0, 0), (1, T, KD)), ttnn.slice(conv, (0, 0, KD), (1, T, 2 * KD)),
                   ttnn.slice(conv, (0, 0, 2 * KD), (1, T, C)), state)
            ttnn.deallocate(conv)
            return out

        def candidate(qkv, carry, taps, vl, **variant):
            return pcx.gdn_prefill_conv_exact(mesh, qkv, carry, taps, valid_len=vl, key_dim_tp=KD, **variant)

        def compare(mine, theirs):
            results = {}
            for name, a, b in zip(('q', 'k', 'v', 'new_state'), mine, theirs):
                for chip, (x, y) in enumerate(zip(read(a), read(b))):
                    results['%s/chip%d' % (name, chip)] = card.compare(torch, x, y)
            return results

        def exact(results):
            return all(r['exact'] for r in results.values())

        def check(name, results, want_exact=True, **extra):
            ok = exact(results) == want_exact
            entry = dict(name=name, ok=ok, exact=exact(results), **extra,
                         diff={k: v for k, v in results.items() if not v['exact']})
            report['checks'].append(entry)
            if not ok:
                report['failures'].append(name)
            print('%-48s %s' % (name, 'ok' if ok else 'FAIL %s' % entry['diff']), flush=True)
            save()
            return ok

        started = time.time()
        if 'equality' in sections:
            stage('equality')
            Ts = (32, 64) if args.quick else (32, 64, 96)
            for T in Ts:
                vls = [vl for vl in (1, 2, 3, 31, 32, 33, 34, 63, 64, 65, 95, 96) if vl <= T] + [None]
                if args.quick:
                    vls = [vl for vl in vls if vl in (1, 3, 32, 33, T, None)]
                for vl in vls:
                    for carry_kind in (('none', 'randn') if args.quick else card.CARRIES):
                        xs, carries, qkv, carry, taps = inputs(T, carry_kind, 'randn', args.seed)
                        before = read(qkv)
                        ref = reference(qkv, carry, taps, vl)
                        out = candidate(qkv, carry, taps, vl)
                        check('T%d vl=%s carry=%s' % (T, vl, carry_kind), compare(out, ref))
                        if not all(torch.equal(card.int16(torch, a), card.int16(torch, b)) for a, b in zip(before, read(qkv))):
                            report['failures'].append('T%d vl=%s: qkv changed' % (T, vl))
                        release(out, ref, qkv, carry, taps)
        if 'data' in sections:
            stage('data')
            for data in card.DATA:
                for vl in (None, 33):
                    xs, carries, qkv, carry, taps = inputs(64, 'special', data, args.seed + 1)
                    ref = reference(qkv, carry, taps, vl)
                    out = candidate(qkv, carry, taps, vl)
                    results = compare(out, ref)
                    qkv_ok = all(r['exact'] for n, r in results.items() if not n.startswith('new_state'))
                    extra = {}
                    if vl is not None and not exact(results) and qkv_ok:
                        other = candidate(qkv, carry, taps, vl, canon_denorm=not pcx.CANON_DENORM_DEFAULT)
                        extra['other_canon_exact'] = exact(compare(other, ref))
                        release(other)
                    check('data=%s vl=%s carry=special' % (data, vl), results, **extra)
                    release(out, ref, qkv, carry, taps)
        if 'negative' in sections:
            stage('negative')
            xs, carries, qkv, carry, taps = inputs(64, 'randn', 'randn', args.seed + 2)
            ref = reference(qkv, carry, taps, 33)
            for negative in card.NEGATIVES:
                out = candidate(qkv, carry, taps, 33, negative=negative)
                check('negative %s differs' % negative, compare(out, ref), want_exact=False)
                release(out)
            release(ref, qkv, carry, taps)
        if 'mirror' in sections:
            stage('mirror')
            for vl in (None, 33):
                xs, carries, qkv, carry, taps = inputs(64, 'randn', 'randn', args.seed + 3)
                ref = reference(qkv, carry, taps, vl)
                out = candidate(qkv, carry, taps, vl, mirror_pack=True)
                check('MIRROR_PACK vl=%s' % vl, compare(out, ref))
                words = candidate(qkv, carry, taps, vl, shift_words=True)
                check('PCX_SHIFT_WORDS vl=%s' % vl, compare(words, ref))
                release(out, words, ref, qkv, carry, taps)
        if 'chain' in sections:
            stage('chain')
            for full in (64, None):
                taps_host = [card.make_taps(torch, C, 50 + chip) for chip in range(2)]
                taps = [upload([taps_host[chip][j:j + 1] for chip in range(2)]) for j in range(K)]
                mine = theirs = None
                for step, vl in enumerate((full, full, full, 33)):
                    xs = [card.make_x(torch, 'randn', 64, C, 60 + step + 17 * chip) for chip in range(2)]
                    qkv = upload(xs, ttnn.L1_MEMORY_CONFIG)
                    ref = reference(qkv, theirs, taps, vl)
                    out = candidate(qkv, mine, taps, vl)
                    check('chain full=%s step %d vl=%s' % (full, step, vl), compare(out, ref))
                    release(qkv, out[:3], ref[:3], mine, theirs)
                    mine, theirs = out[3], ref[3]
                release(mine, theirs, taps)
        if 'cache' in sections:
            stage('cache')
            counts = []
            for index, vl in enumerate((64, 33, 2)):
                xs, carries, qkv, carry, taps = inputs(64, 'randn', 'randn', args.seed + 70 + index)
                ref = reference(qkv, carry, taps, vl)
                entries = mesh.num_program_cache_entries()
                out = candidate(qkv, carry, taps, vl)
                counts.append(mesh.num_program_cache_entries() - entries)
                check('cache call %d vl=%d' % (index, vl), compare(out, ref))
                release(out, ref, qkv, carry, taps)
            report['program_cache_new_entries'] = counts
            if any(counts[1:]):
                report['failures'].append('cached calls added program-cache entries: %s' % counts)
        if 'shift' in sections:
            stage('shift')
            for words in (False, True):
                xs, carries, qkv, carry, taps = inputs(96, 'randn', 'randn', args.seed + 80)
                out = candidate(qkv, carry, taps, None, shift_only=True, shift_words=words)
                got = read(out)
                results = {'shift/chip%d' % chip: card.compare(torch, got[chip][0], card.shifted_reference(torch, xs[chip], carries[chip]))
                           for chip in range(2)}
                check('shift-only words=%s' % words, results)
                release(out, qkv, carry, taps)
        report['seconds'] = round(time.time() - started, 1)
        report['descriptor_cache'] = pcx.cache_size()
        stage('mesh-close')
        ttnn.close_mesh_device(mesh)
        mesh = None
        report['passed'] = not report['failures'] and bool(report['checks'])
        stage('done')
    finally:
        if mesh is not None:
            import ttnn
            ttnn.close_mesh_device(mesh)
        save()
    print('PASSED' if report['passed'] else 'FAILED %s' % report['failures'], flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
