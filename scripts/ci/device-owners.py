"""Read-only host PID namespace check before opening the two allocated cards."""

import json
from pathlib import Path
import stat


def main():
    devices = {}
    for device in Path("/host-dev/tenstorrent").iterdir():
        metadata = device.stat()
        if stat.S_ISCHR(metadata.st_mode):
            devices[metadata.st_rdev] = device.name
    if len(devices) != 2:
        raise RuntimeError(f"Expected two card device nodes, found {len(devices)}")
    expected = {"blackhole-3707293C249A5E67": "0", "blackhole-CEF5729692C19E6D": "2"}
    for board, node in expected.items():
        if (Path("/host-dev/tenstorrent/by-id") / board).resolve().name != node:
            raise RuntimeError("Physical board mapping changed; review device allocation")
    owners, denied = [], []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            for descriptor in (process / "fd").iterdir():
                try:
                    metadata = descriptor.stat()
                    if stat.S_ISCHR(metadata.st_mode) and metadata.st_rdev in devices:
                        owners.append({"pid": int(process.name), "device": devices[metadata.st_rdev]})
                except (FileNotFoundError, ProcessLookupError):
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            denied.append(int(process.name))
    print(json.dumps({"owners": owners, "unreadable_pids": denied,
                      "scope": "FD ownership snapshot only; not a persistent hardware reservation"}, indent=2))
    if owners or denied:
        raise SystemExit("Device ownership cannot be proven clear; refusing accelerator execution")


if __name__ == "__main__":
    main()
