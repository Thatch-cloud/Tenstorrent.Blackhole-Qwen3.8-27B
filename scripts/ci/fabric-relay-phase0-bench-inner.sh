#!/usr/bin/env bash
# In-container Phase-0 benchmark inner script: runs on the pinned image with
# both cards. Captures mesh/link identity first (gate evidence), then P0.1
# PCIe per-card and P0.2 fabric bandwidth repeats. Every artifact lands in
# /experiment/results; every failure is a recorded result, not a retry.
set -euo pipefail
results=/experiment/results
# Device numbers are not stable across resets and the host has three cards;
# the fabric-linked pair is passed explicitly (default 0,2 per the rig's
# known cabling) and must be re-confirmed by-id before invoking.
devices=${FABRIC_RELAY_DEVICES:-0,2}
python3 -B /experiment-scripts/ci/fabric_relay_phase0_identity.py --out "$results/identity.json" || true
python3 -B /experiment-scripts/ci/fabric_relay_phase0_bench.py \
    --results "$results" --mode pcie --payload-mib 256 --repeats 9 --devices "$devices"
python3 -B /experiment-scripts/ci/fabric_relay_phase0_bench.py \
    --results "$results" --mode fabric --payload-mib 256 --repeats 9 --devices "$devices" \
    || echo "P0.2 fabric copy unavailable on this harness (recorded, not retried)" \
        > "$results/p02-fabric-unavailable.txt"
ls -la "$results"
