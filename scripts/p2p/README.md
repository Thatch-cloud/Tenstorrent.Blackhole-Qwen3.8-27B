# Peer-to-peer restore of KV blocks from NVMe into card DRAM: tools and models

CPU-only groundwork for restoring evicted KV blocks from an NVMe drive straight into a card's DRAM (no bounce through host RAM). Nothing here is
imported by the serving engine; nothing touches a card unless `p2p_probe.py` is run as root on a host that has the patched driver
(`patches/tt-kmd-p2pmem`) loaded.

| File | What it is |
|---|---|
| `p2p_topology.py` | Read-only. For each card and drive: will the kernel allow peer-to-peer DMA between them (`BUS_ADDR` inside a switch, `THRU_HOST_BRIDGE` through the root complex, or not at all), which ports redirect it (ACS), and the boot parameter and `setpci` command that clear the redirect |
| `p2p_probe.py` | The feasibility probe: `O_DIRECT`-read a file or block-device range into the card through p2pmem, verify it through a second path, time it against the host-bounce path. `--dry-run` prints the plan, `--mock` runs the whole flow on fakes |
| `kv_shard_layout.py` | The on-drive layout (one object per card, sector-aligned, header page + shards), the final-layout arithmetic, the card-side digest specification, and a max-min-fair bandwidth model of drives, cards, a switch and ACS redirect |
| `test_*.py` | CPU tests of all three (`python -B -m unittest discover -s scripts/p2p -p 'test_*.py'`; allowlisted in `qwen-integration-cpu.yml`) |

Safety properties of the probe, each tested (and each checked by mutating the code and seeing a test fail): nothing is opened under `--dry-run`; the
slice must be free (the PCI core's `p2pmem/available` equals `size`) and no other process may have the card open; the card is woken (the busy power state
the UMD sends: the DRAM PHY is asleep otherwise) before anything touches it; the 4 GiB TLB window must come up for this process and is pinned (a dma-buf
export, so a reset is refused and FREE_TLB or close cannot pull it away) or the run stops; a source with a hole, an unwritten or delayed extent is refused
before the window is configured; a pattern written through the window must read back, inside the destination; the mapping must sit at the physical
address the slice has; the destination is painted with a sentinel before every timed run so a DMA that wrote nothing cannot pass on leftovers; a run that
compared nothing (`--verify none`) is UNVERIFIED, never PASS. `--prepare` onto a block device needs `--overwrite-device`, a device that nothing mounts,
holds, swaps or arrays, and a first MiB of zeros.

Exit status of `p2p_probe.py`: 0 PASS, 1 FAIL (a difference, or below `--min-gbps`), 2 could not run, 3 UNVERIFIED. The window is mapped uncached by the
kernel once the range is p2pmem, so CPU verification is slow and the host-bounce `--baseline` is indicative only (the real host path is the UMD's DMA).

Typical use (root, the patched driver loaded with `p2pmem_devices=<this card>`, nothing serving from the card; the card's DRAM is overwritten):

    p2p_topology.py --list
    p2p_probe.py --prepare /dev/<blank spare drive> --size 4G --overwrite-device      # or a file on a spare filesystem
    p2p_probe.py --source /dev/<that drive> --size 1G --chunk 16M --threads 4 --baseline --verify sample
    p2p_probe.py --source /dev/<that drive> --size 1G --baseline-only                 # the host-bounce path alone, for CPU-time comparison
