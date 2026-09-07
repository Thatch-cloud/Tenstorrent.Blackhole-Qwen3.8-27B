#!/usr/bin/env bash
set -euo pipefail
test "${RUNNER_NAME:-}" = thatch-build-amd64-02-cp-temp
test "${QWEN_CARDS_ALLOCATED:-0}" = 1
test "${QWEN_RESET_AUTHORIZED:-0}" = 1
mkdir -p experiment-results
exec > >(tee experiment-results/card-reset.log) 2>&1
printf 'stage=preflight utc=%s\n' "$(date -u +%FT%TZ)"
python3 - <<'PY'
import json
from pathlib import Path
import stat

root = Path('/dev/tenstorrent')
expected = {'blackhole-3707293C249A5E67': '0', 'blackhole-CEF5729692C19E6D': '2'}
nodes = {path.name for path in root.iterdir() if stat.S_ISCHR(path.stat().st_mode)}
if nodes != set(expected.values()):
    raise RuntimeError(f'Refusing reset: unexpected device nodes {sorted(nodes)}')
for board, node in expected.items():
    if (root / 'by-id' / board).resolve(strict=True) != root / node:
        raise RuntimeError('Refusing reset: physical board mapping changed')
pci = [path.name for path in Path('/sys/bus/pci/devices').iterdir()
       if (path / 'vendor').read_text().strip() == '0x1e52']
if len(pci) != 2:
    raise RuntimeError(f'Refusing all-device reset: expected exactly two TT PCI functions, found {pci}')
print(json.dumps({'boards': expected, 'pci': sorted(pci), 'allocation': 'operator-confirmed',
                  'scope': 'Exact two-board inventory; not a host process ownership proof'}))
PY
running=$(docker ps -q)
if [ -n "$running" ]; then
    docker inspect $running | python3 -c '
import json, sys
for container in json.load(sys.stdin):
    config = container["HostConfig"]
    mapped = [item.get("PathOnHost", "") for item in config.get("Devices") or []]
    mounts = [item.get("Source", "") for item in container.get("Mounts", [])]
    risky = config.get("Privileged") or any(path == "/dev" or path.startswith("/dev/tenstorrent") for path in mapped + mounts)
    if risky:
        raise SystemExit("Refusing reset: running container has accelerator-capable access: " + container["Name"])
print("Running-container access preflight passed; no containers stopped")
'
fi
smi=$(command -v tt-smi || true)
if [ -z "$smi" ] && [ -x /home/thatch/.local/bin/tt-smi ]; then smi=/home/thatch/.local/bin/tt-smi; fi
if [ -z "$smi" ]; then
    echo 'stage=blocked: host tt-smi not installed/on the bounded search path; no reset attempted'
    exit 1
fi
printf 'tt_smi=%s\n' "$smi"
timeout -k 5 30 "$smi" --help > experiment-results/tt-smi-help.txt
grep -q -- '--reset' experiment-results/tt-smi-help.txt
prefix=()
if [ "$(id -u)" != 0 ]; then
    sudo -n "$smi" --help >/dev/null
    prefix=(sudo -n)
fi
printf 'stage=reset-start utc=%s targets=all-exactly-two-verified-boards\n' "$(date -u +%FT%TZ)"
timeout -k 15 180 "${prefix[@]}" "$smi" -r all
printf 'stage=reset-command-succeeded utc=%s\n' "$(date -u +%FT%TZ)"
