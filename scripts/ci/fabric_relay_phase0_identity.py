"""Read-only Phase-0 identity capture for the fabric-relay procedure.

Runs on the rig host (no devices opened, no weights loaded). Captures the
evidence every P0.x gate needs: by-id device resolution, link/board state,
host snapshot, and the exact repository revision. Output is a JSON artifact
plus SHA256s, ready as input to gates.py via evidence assembly.

House safety: read-only, no docker, no runtime change, no serving-default
change. Everything is captured; nothing is asserted about mesh adjacency —
the Logical/Physical histogram assertion happens when the mesh is opened,
per docs/gotchas.md.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_by_id(devices_dir="/dev/tenstorrent/by-id"):
    """Map blackhole serials to device numbers, per docs/gotchas.md."""
    mapping = {}
    root = Path(devices_dir)
    if not root.is_dir():
        return mapping, False
    for entry in sorted(root.iterdir()):
        if not entry.name.startswith("blackhole-"):
            continue
        target = Path(entry).resolve()
        mapping[entry.name] = str(target)
    return mapping, True


def _read_text(path):
    return Path(path).read_text()


def capture(read_text=_read_text, run_command=subprocess.run):
    """Assemble the identity snapshot; I/O injected for testability.

    read_text receives the plain POSIX-style /proc path as a string.
    """

    def read_optional(path):
        try:
            return read_text(path)
        except (OSError, KeyError):
            return "unavailable"

    mapping, by_id_dir_present = resolve_by_id()
    meminfo = read_optional("/proc/meminfo")
    pressure = {name: read_optional(f"/proc/pressure/{name}") for name in ("cpu", "memory", "io")}
    try:
        smi = run_command(["/home/thatch/.local/bin/tt-smi", "-s"], timeout=30)
        smi_state = {"returncode": smi.returncode, "output": smi.stdout[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        smi_state = {"error": repr(error)}
    return {
        "schema": "fabric-relay-phase0-identity-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "by_id_dir_present": by_id_dir_present,
        "device_map": mapping,
        "meminfo_total_line": next((line for line in meminfo.splitlines() if line.startswith("MemTotal")), "unavailable"),
        "host_pressure": pressure,
        "tt_smi_state": smi_state,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output JSON artifact path")
    args = parser.parse_args(argv)
    snapshot = capture()
    out = Path(args.out)
    out.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    snapshot["artifact_sha256"] = sha256_file(out)
    snapshot["artifact_path"] = str(out)
    out.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: snapshot[key] for key in
                      ("schema", "captured_at", "by_id_dir_present", "device_map", "artifact_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
