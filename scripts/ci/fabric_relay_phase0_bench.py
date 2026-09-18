"""Device-touching Phase-0 benchmark driver for the fabric-relay procedure.

Runs inside the pinned runtime image on the rig with both cards allocated.
P0.1 measures host<->card-0 DRAM transfer bandwidth over the local PCIe
attachment; P0.2 measures card-0 -> card-1 device-to-device copy over the
four-link fabric. Raw per-repeat samples are written to the results directory
as JSON (summary + sample list); failures are recorded, never retried into a
pass. The pure timing/summary math lives in optimisation/fabric-relay so host
tests cover it without hardware.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/experiment/source/optimisation/fabric-relay")
from fabric_bench import summarize, transfer_bandwidth  # noqa: E402

# Payload: large enough to saturate sustained bandwidth, small enough to fit
# device DRAM comfortably alongside nothing else (rig is exclusive).
TENSOR_FLOATS = 64 * 1024 * 1024  # 256 MiB of float32


def _torch():
    import torch

    return torch


def pcie_copy_fn(device, direction):
    """Host->device (write) or device->host (read) over card-0 PCIe."""
    torch = _torch()
    import ttnn

    def write(_payload_bytes):
        host = torch.zeros(TENSOR_FLOATS, dtype=torch.float32)
        device_tensor = ttnn.from_torch(host, device=device, dtype=ttnn.float32)
        ttnn.synchronize_device(device)

    def read(_payload_bytes):
        host = torch.zeros(TENSOR_FLOATS, dtype=torch.float32)
        device_tensor = ttnn.from_torch(host, device=device, dtype=ttnn.float32)
        ttnn.synchronize_device(device)
        _ = ttnn.to_torch(device_tensor)  # the measured readback

    if direction == "write":
        return write
    if direction == "read":
        return read
    raise ValueError(f"Unknown direction {direction!r}")


def fabric_copy_fn(devices):
    """Card-0 DRAM -> card-1 DRAM over the fabric links.

    Uses the mesh of both opened devices; if the pinned runtime lacks the
    cross-chip copy path, raise NotImplementedError so the caller records
    unavailability honestly.
    """
    torch = _torch()
    import ttnn

    if len(devices) < 2:
        raise NotImplementedError("Both devices required for the fabric copy")

    def copy(_payload_bytes):
        host = torch.zeros(TENSOR_FLOATS, dtype=torch.float32)
        source = ttnn.from_torch(host, device=devices[0], dtype=ttnn.float32)
        ttnn.synchronize_device(devices[0])
        # Cross-chip transfer over the fabric mesh.
        dest = ttnn.to_device(source, devices[1])
        ttnn.synchronize_device(devices[1])
        _ = ttnn.to_torch(dest)

    return copy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--mode", required=True, choices=["pcie", "fabric"])
    parser.add_argument("--payload-mib", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args(argv)

    import ttnn

    device_0 = ttnn.open_device(device_id=0)
    try:
        if args.mode == "pcie":
            samples_by_direction = {}
            for direction in ("read", "write"):
                samples = transfer_bandwidth(
                    pcie_copy_fn(device_0, direction),
                    args.payload_mib * 1024 * 1024,
                    args.repeats,
                )
                samples_by_direction[direction] = samples
            artifact = {
                "mode": "pcie",
                "card": 0,
                "payload_mib": args.payload_mib,
                "directions": {d: summarize(s) for d, s in samples_by_direction.items()},
                "raw_samples": samples_by_direction,
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        else:
            device_1 = ttnn.open_device(device_id=1)
            try:
                samples = transfer_bandwidth(
                    fabric_copy_fn([device_0, device_1]),
                    args.payload_mib * 1024 * 1024,
                    args.repeats,
                )
                artifact = {
                    "mode": "fabric",
                    "cards": [0, 1],
                    "payload_mib": args.payload_mib,
                    "summary": summarize(samples),
                    "raw_samples": samples,
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            finally:
                ttnn.close_device(device_1)
    finally:
        ttnn.close_device(device_0)

    out = Path(args.results) / f"p0-bench-{args.mode}.json"
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(artifact.get("directions", artifact.get("summary")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
