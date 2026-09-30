#!/usr/bin/env python3
"""Does the batched GDN launch shrink the L1 a captured trace leaves for everything else?

The batched program is the fused 24-core kernel on four disjoint core shares. The fused
circular-buffer plan reserves 630,784 bytes per core against the value split's 282,624
for its recurrence stage and 204,800 for its norm stage
(`gdn_multitoken.cb_plan(True)` and `gdn_vsplit.cb_plan`), and the batched program puts
the larger figure on 96 cores rather than 24. Inside a captured trace that reservation is
made once and never re-validated on replay, so anything allocated in L1 afterwards that
lands under the line would be overwritten by every round.

That is the leading hypothesis for the segfault at the second user's admission (task #48),
and this measures it instead of arguing it. Per arm - four sequential single-user launches,
or one batched launch - it:

  1. captures the launches into a trace, exactly as packed_verifier.py:433-434 does;
  2. allocates interleaved L1 canaries of known content until the allocator refuses, which
     is a direct measurement of the L1 the arm leaves free;
  3. replays the trace, then reads every canary back and compares it bit for bit.

Three outcomes, all informative. Canaries corrupted: the hypothesis is right and the
reservation is the bug. Allocation refused with the canaries intact: the allocator
protects the reserved region, so the serving path would have failed loudly rather than
corrupted, and the segfault is something else. Same free L1 under both arms: the
reservation does not grow and the hypothesis is wrong.

  python3 gdn_user_batch_l1_probe.py --out probe.json --chips 1
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

import gdn_multitoken as native
import gdn_user_batch as batch


def same_bits(torch, left, right):
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    width = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(left.element_size())
    if width is None:
        return torch.equal(left, right)
    return torch.equal(left.view(width), right.view(width))


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chips', type=int, default=1, choices=(1, 2))
    parser.add_argument('--users', type=int, default=4)
    parser.add_argument('--rows', type=int, default=16)
    parser.add_argument('--replays', type=int, default=8)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--canary-mb', type=float, default=2.0,
                        help='size of each interleaved L1 canary tensor, in MiB')
    parser.add_argument('--canary-limit', type=int, default=80,
                        help='stop after this many canaries even if the allocator keeps saying yes')
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('TT_METAL_HOME', '/opt/tt-metal')))
    return parser.parse_args()


def main():
    arguments = parse()
    report = dict(scope='Uncertified L1 headroom and trace-replay integrity probe for the batched GDN launch; '
                        'no model, no projection, no collective',
                  users=arguments.users, rows=arguments.rows, chips=arguments.chips,
                  replays=arguments.replays, canary_mb=arguments.canary_mb, seed=arguments.seed,
                  native_sha256=native.HASHES,
                  module_sha256={name: hashlib.sha256((Path(__file__).with_name(name)).read_bytes()).hexdigest()
                                 for name in ('gdn_user_batch.py',)},
                  result='did-not-run', stages=[], arms={})

    def stage(name, **details):
        report['stages'].append(dict(stage=name, **details))
        arguments.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['stages'][-1]), flush=True)

    mesh = None
    try:
        stage('import-runtime')
        import torch
        import ttnn
        report['ttnn_path'] = ttnn.__file__

        stage('load-kernels', root=str(arguments.root))
        kernels = batch.load_kernels(arguments.root)

        io, fp32 = native.cb_plan(True)
        report['fused_cb_bytes_per_core'] = sum(io.values()) * 2048 + sum(fp32.values()) * 4096

        stage('mesh-open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, arguments.chips), l1_small_size=24576,
                                     trace_region_size=134217728)
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]

        def upload(value, memory=None):
            return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=memory or ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def host(value):
            return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(value)]

        stage('fixture-upload')
        torch.manual_seed(arguments.seed)
        rows = arguments.rows
        norm_w = upload((1 + torch.randn(1, 1, 128) * 0.1).bfloat16())
        groups = [(upload(torch.randn(1, rows, 5120).bfloat16()),
                   upload(torch.rand(1, rows, 24).bfloat16()),
                   upload((-torch.rand(1, rows, 24)).bfloat16()),
                   upload((torch.randn(1, 24, 128, 128) * 0.05).bfloat16()),
                   upload(torch.randn(1, rows, 3072).bfloat16()),
                   norm_w) for index in range(arguments.users)]

        # One canary is a whole number of 32x32 BF16 tiles so the shape is exact.
        tiles = max(1, int(arguments.canary_mb * 1024 * 1024) // 2048)
        canary_shape = (1, 32, tiles * 32)

        def release(produced):
            for output, states in produced or ():
                ttnn.deallocate(output)
                ttnn.deallocate(states)

        def run_arm(name, launch):
            """Capture `launch` into a trace, fill L1 with canaries, replay, then check them."""
            # Warm first, exactly as packed_verifier.py:422-423 does before its capture:
            # a kernel compiled for the first time inside a capture is not something the
            # serving path ever asks for, and it is not what this probe is measuring.
            stage(name + '-warm')
            release(launch())
            ttnn.synchronize_device(mesh)
            stage(name + '-capture')
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            produced = None
            try:
                produced = launch()
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.synchronize_device(mesh)

            stage(name + '-fill-l1')
            canaries, expected, refusal = [], [], None
            try:
                while len(canaries) < arguments.canary_limit:
                    value = torch.randn(*canary_shape).bfloat16()
                    canaries.append(upload(value, ttnn.L1_MEMORY_CONFIG))
                    expected.append(value)
            except BaseException as error:
                refusal = repr(error)[:400]

            def survey():
                damage = []
                for index, (value, want) in enumerate(zip(canaries, expected)):
                    for chip, shard in enumerate(host(value)):
                        if not same_bits(torch, shard, want):
                            gap = (shard.float() - want.float()).abs()
                            damage.append(dict(canary=index, chip=chip, differing=int((gap > 0).sum()),
                                               of=int(gap.numel())))
                return damage

            # BEFORE the replay. Without this the probe cannot tell the GDN trace apart
            # from its own canary uploads: every `from_torch` to L1 runs a program whose
            # own circular buffers are placed in the same descending region, so filling
            # L1 to exhaustion damages canaries on its own. Only damage that APPEARS
            # across the replay is attributable to the launch under test.
            stage(name + '-check-before-replay')
            before = survey()

            stage(name + '-replay', canaries=len(canaries), damaged_before=len(before))
            for unused in range(arguments.replays):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)

            stage(name + '-check')
            after = survey()
            known = {(entry['canary'], entry['chip']) for entry in before}
            caused = [entry for entry in after if (entry['canary'], entry['chip']) not in known]
            report['arms'][name] = dict(
                canaries_allocated=len(canaries),
                l1_free_bytes_measured=len(canaries) * tiles * 2048,
                l1_free_mib_measured=len(canaries) * tiles * 2048 / (1024 * 1024),
                allocator_refusal=refusal,
                damaged_by_the_uploads=before, damaged_after_replay=after,
                caused_by_the_replay=caused, replay_is_clean=not caused)
            for value in canaries:
                ttnn.deallocate(value)
            release(produced)
            ttnn.release_trace(mesh, trace)
            ttnn.synchronize_device(mesh)

        run_arm('per_user', lambda: [batch.execute(mesh, [group], kernels)[0] for group in groups])
        run_arm('batched', lambda: batch.execute(mesh, groups, kernels))

        per_user, batched = report['arms']['per_user'], report['arms']['batched']
        report['l1_free_delta_mib'] = per_user['l1_free_mib_measured'] - batched['l1_free_mib_measured']
        report['replay_caused_corruption'] = bool(per_user['caused_by_the_replay'] or batched['caused_by_the_replay'])
        report['batched_is_worse'] = len(batched['caused_by_the_replay']) > len(per_user['caused_by_the_replay'])
        report['verdict'] = (
            'the batched replay corrupts L1 the per-user replay does not: the reservation is the bug'
            if report['batched_is_worse'] else
            'both replays corrupt L1 equally: a property of the fused plan, not of batching'
            if report['replay_caused_corruption'] else
            'neither replay changed a single canary byte: the captured launch does not overwrite L1 '
            'allocated after capture, and this is not the segfault mechanism')
        report['result'] = 'pass'
        stage('done', verdict=report['verdict'], l1_free_delta_mib=report['l1_free_delta_mib'])
        print('VERDICT', report['verdict'], flush=True)
        print(json.dumps(report['arms'], indent=2), flush=True)
        return 0
    except BaseException as error:
        report['result'] = 'error'
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], flush=True)
        return 2
    finally:
        arguments.out.write_text(json.dumps(report, indent=2))
        if mesh is not None:
            import ttnn
            ttnn.close_mesh_device(mesh)


if __name__ == '__main__':
    raise SystemExit(main())
