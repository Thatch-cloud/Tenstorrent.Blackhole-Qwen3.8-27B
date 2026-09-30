"""P0.1/P0.2 bandwidth microbenchmarks: PCIe per card and fabric over four links.

Uses the pinned runtime's device APIs when available; without devices it can
only exercise argument handling and summary math (host tests cover the pure
parts). Summary statistics follow the repo's repeat discipline: >=9 repeats,
p50/p95 reported, never a single winning run.
"""

import argparse
import json
import math
import time
from pathlib import Path


def summarize(samples_gib_s):
    ordered = sorted(samples_gib_s)
    if len(ordered) < 9:
        raise ValueError(f"Need at least 9 repeats, got {len(ordered)}")
    def quantile(fraction):
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]
    return {
        "repeats": len(ordered),
        "p50_GiB_s": round(quantile(0.50), 3),
        "p95_GiB_s": round(quantile(0.95), 3),
        "max_GiB_s": round(ordered[-1], 3),
    }


def transfer_bandwidth(copy_fn, payload_bytes, repeats=9):
    """copy_fn(payload_bytes) -> seconds; returns per-repeat GiB/s samples."""
    gib = payload_bytes / (1024 ** 3)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        copy_fn(payload_bytes)
        elapsed = time.perf_counter() - started
        if elapsed <= 0:
            raise RuntimeError("Non-positive transfer time")
        samples.append(gib / elapsed)
    return samples


def _device_copy(mode):
    """Build a copy_fn against the pinned runtime; raises without devices."""
    import ttnn  # pinned runtime only; import kept inside so host tests need no device

    device_0 = ttnn.open_device(device_id=0)
    try:
        if mode == "pcie":
            def copy_fn(payload_bytes):
                # Host -> card-0 DRAM over the local PCIe attachment.
                host = ttnn.from_torch(
                    __import__("torch").zeros(payload_bytes // 4, dtype=torch.float32),
                    device=device_0)
                ttnn.synchronize_device(device_0)
            return copy_fn
        if mode == "fabric":
            def copy_fn(payload_bytes):
                # Card-0 DRAM -> card-1 DRAM over the four-link fabric.
                raise NotImplementedError("Fabric copy lands with the P1 harness")
            return copy_fn
        raise ValueError(f"Unknown mode {mode!r}")
    finally:
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["pcie", "fabric"])
    parser.add_argument("--card", type=int, default=1, help="card whose attachment is measured (pcie mode)")
    parser.add_argument("--payload-mib", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    copy_fn = _device_copy(args.mode)
    samples = transfer_bandwidth(copy_fn, args.payload_mib * 1024 * 1024, args.repeats)
    summary = summarize(samples)
    summary["mode"] = args.mode
    summary["card"] = args.card
    text = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
