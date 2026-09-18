"""P0.4 card core inventory: count reclaimable card-1 tensix cores under a
dispatch-config candidate.

Input is a JSON dump of per-card logical core maps, e.g. from the pinned
runtime's core descriptor / device grid introspection:
{"0": {"tensix": [...], "eth": [...], "dispatch_reserved": [...]}, "1": {...}}

Reclaimable = card-1 tensix cores currently reserved for dispatch/host I/O
that the eth-dispatch candidate no longer needs.
"""

import argparse
import json
from pathlib import Path


def inventory(cluster_map):
    cards = sorted(cluster_map, key=int)
    if "1" not in cards:
        raise ValueError("Cluster map must include card 1")
    result = {}
    for card in cards:
        entry = cluster_map[card]
        tensix = set(entry.get("tensix", []))
        reserved = set(entry.get("dispatch_reserved", []))
        if not reserved <= tensix:
            raise ValueError(f"Card {card}: reserved cores outside tensix grid")
        result[card] = {
            "tensix_total": len(tensix),
            "dispatch_reserved": len(reserved),
            "compute_available": len(tensix) - len(reserved),
        }
    result["reclaimable_card1"] = result["1"]["dispatch_reserved"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, help="cluster core-map JSON")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    summary = inventory(json.loads(Path(args.map).read_text(encoding="utf-8")))
    text = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
