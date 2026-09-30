#!/usr/bin/env python3
"""Exactness test for the batch-64 ttnn.experimental.nlp_concat_heads_decode.

The op is a pure permutation (each output core gathers one head row per user, no math), so every
check here is bit-exact: whatever bf16 value went in must come out, at the right row.

Shapes are Qwen3.8-27B attention at TP=2: NH=12 heads of HD=256, so the gated SDPA output is
[1, B, 12, 256] (heads padded to 32 by the tile layout), height-sharded one user per core, and the
op must emit [1, 1, B, 3072] width-sharded by head. The input memory config and the core grid are
attention/tp.py:_concat_heads_decode (lines 375-409; the grid choice at 387-391) reproduced:
shard (TILE_SIZE, HD), HEIGHT, ROW_MAJOR, use_height_and_width_as_shard_shape -- 8x4 at B=32,
8x8 at B=64.

Cases:
  B=8   the batch-padded path (output batch padded up to one 32-row tile): a regression check that
        the <= 32 path is untouched.
  B=32  the production one-batch-tile path, unchanged.
  B=64  two batch tiles in ONE call: what this patch adds.
  B=64 vs 2x32: the batch-64 output must equal the two 32-user halves concatenated on the user
        axis, which is what scripts/ci/two_tile_decode.TwoTileConcatHeads computes today.

Run it on the rig through build-and-test-b64.sh (TT_METAL_WATCHER=5). Exit 0 iff every case passed.
"""
import sys
import traceback

import torch
import ttnn

NH, HD = 12, 256
TILE = 32


def core_grid_for(dev, B):
    """The model's own grid choice (attention/tp.py:387-391): the widest x <= grid.x that divides B
    with B // x rows on the device, one core per user, rectangle anchored at (0, 0)."""
    grid = dev.compute_with_storage_grid_size()
    gx = min(B, grid.x)
    if B >= gx and B % gx != 0:
        gx = max(x for x in range(gx, 0, -1) if B % x == 0 and B // x <= grid.y)
    if B % gx:
        raise RuntimeError("no core grid for B=%d on a %dx%d grid" % (B, grid.x, grid.y))
    gy = B // gx
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return gx, gy, cores


def shard_cfg(cores):
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, HD),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def out_shard_info(tensor):
    """Best-effort description of the width-sharded output; reported, never required (the binding
    spelling of shard_spec is not worth failing a correct kernel over)."""
    try:
        spec = tensor.memory_config().shard_spec
        if callable(spec):
            spec = spec()
        return {"out_shard": tuple(int(v) for v in spec.shape), "out_cores": int(spec.grid.num_cores())}
    except Exception as exc:
        return {"out_shard": "unavailable (%s)" % type(exc).__name__, "out_cores": None}


def run_op(dev, x, B):
    """One native call. x is [1, B, NH, HD] bf16 on the host; returns the [1, 1, rows, NH*HD] output
    as torch (rows = max(B, 32)) and what the op chose for the grids."""
    gx, gy, cores = core_grid_for(dev, B)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    sh = ttnn.to_memory_config(t, shard_cfg(cores))
    ttnn.deallocate(t)
    out_sh = ttnn.experimental.nlp_concat_heads_decode(sh, num_heads=NH)
    ttnn.deallocate(sh)
    info = {"in_grid": "%dx%d" % (gx, gy), "in_cores": gx * gy}
    info.update(out_shard_info(out_sh))
    out = ttnn.sharded_to_interleaved(out_sh, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out_sh)
    got = ttnn.to_torch(out).clone()
    ttnn.deallocate(out)
    return got, info


def reference(x, B):
    """User b's NH heads laid end to end at output row b."""
    return x.reshape(1, 1, B, NH * HD)


def run_case(dev, B, seed):
    try:
        torch.manual_seed(seed)
        x = torch.randn(1, B, NH, HD).to(torch.bfloat16)
        got, info = run_op(dev, x, B)
        res = dict(info)
        res["out_shape"] = tuple(int(v) for v in got.shape)
        res["exact"] = bool(torch.equal(got[:, :, :B, :], reference(x, B)))
        res["rows_ok"] = int(got.shape[-2]) == max(B, TILE)
        res["width_ok"] = int(got.shape[-1]) == NH * HD
        res["one_core_per_user"] = info["in_cores"] == B
        ok = res["exact"] and res["rows_ok"] and res["width_ok"] and res["one_core_per_user"]
        print("CASE B=%d: %s %s" % (B, "PASS" if ok else "FAIL", res), flush=True)
        return ok, x, got
    except Exception as exc:
        print("CASE B=%d: FAIL %r" % (B, exc), flush=True)
        traceback.print_exc()
        return False, None, None


def run_halves_case(dev, x64, got64):
    """The current workaround's answer: the same input through two B=32 calls, joined on users."""
    try:
        halves = []
        for first in range(0, int(x64.shape[1]), TILE):
            half = x64[:, first : first + TILE].contiguous()
            got, _ = run_op(dev, half, TILE)
            halves.append(got[:, :, :TILE, :])
        joined = torch.cat(halves, dim=-2)
        ok = joined.shape == got64.shape and bool(torch.equal(got64, joined))
        print(
            "CASE B=64 vs 2x32 halves: %s {'joined_shape': %s, 'native_shape': %s}"
            % ("PASS" if ok else "FAIL", tuple(int(v) for v in joined.shape), tuple(int(v) for v in got64.shape)),
            flush=True,
        )
        return ok
    except Exception as exc:
        print("CASE B=64 vs 2x32 halves: FAIL %r" % (exc,), flush=True)
        traceback.print_exc()
        return False


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        grid = dev.compute_with_storage_grid_size()
        print("GRID: compute_with_storage %dx%d" % (grid.x, grid.y), flush=True)

        ok8, _, _ = run_case(dev, 8, 0)
        ok32, _, _ = run_case(dev, 32, 1)
        ok64, x64, got64 = run_case(dev, 64, 2)
        ok_halves = run_halves_case(dev, x64, got64) if ok64 else False

        ok = ok8 and ok32 and ok64 and ok_halves
        print("RESULT", "PASS" if ok else "FAIL", flush=True)
        sys.exit(0 if ok else 1)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
