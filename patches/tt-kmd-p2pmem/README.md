# tt-kmd: publish a slice of BAR4 as PCI peer-to-peer memory (optional, off by default)

A Blackhole card's BAR4 is eight 4 GiB windows onto the NOC. A TLB register per window (the driver's `CONFIGURE_TLB`) says which NOC endpoint and
address the window shows, and a window aimed at address 0 of a DRAM tile shows that DRAM bank (the card has eight banks of 4 GiB). So a PCIe write
to `BAR4 + k * 4 GiB + x` lands in DRAM bank-for-window-k at address `x`, with no CPU involved.

`pci_p2pdma_add_resource()` gives such a range struct pages. With them, Linux 6.2 or later lets a process `mmap` a piece of it from
`/sys/bus/pci/devices/<bdf>/p2pmem/allocate` and `O_DIRECT`-read a file or a block device into it: the NVMe drive DMAs into the card, and the data
never passes through host RAM. This patch is the card's half of that: it publishes the slice. It programs no TLB window.

`0001-*.patch` applies to `tenstorrent/tt-kmd` `main` (2.11.1-pre) and to the `ttkmd-2.11.0` tag with `git apply`. `build-check.sh` applies it to a
checkout and builds it in a scratch directory, nothing installed:

    patches/tt-kmd-p2pmem/build-check.sh <tt-kmd checkout> /lib/modules/$(uname -r)/build [--no-p2pdma]

Built (W=1, no warnings from the new files) against the Linux 7.0.0-34, 6.17 and 6.11 header trees, with and without `CONFIG_PCI_P2PDMA`. The
build host is aarch64, so the x86-64 object has not been built; the code is architecture independent. The module has not been loaded anywhere.

## Parameters (all read-only after load)

| Parameter | Default | Meaning |
|---|---|---|
| `p2pmem_devices` | empty = nothing published | `all`, or a comma-separated list of PCI addresses (`0000:41:00.0,0000:42:00.0`); Blackhole only |
| `p2pmem_offset_mb` | 0 | start of the slice inside BAR4, MiB, even |
| `p2pmem_size_mb` | 4096 | length of the slice, MiB, even. The kernel spends 64 bytes of host RAM per 4 KiB of slice (64 MiB for 4 GiB) |
| `p2pmem_publish` | off | also `pci_p2pmem_publish()`, so in-kernel users (nvmet) may pick the memory on their own. Userspace `allocate` does not need it |

BAR0 (TLB registers, the 2 MiB windows onto device registers) is never offered: the BAR is fixed at 4 in the code.

## What it adds to a device

* `/sys/class/tenstorrent/tenstorrent!<N>/p2pmem/{bar,offset,size,bus_addr,tlb_first,tlb_last}`, root-readable. `tlb_first`/`tlb_last` are the
  4 GiB window ids the slice lies in (202 for the first window of BAR4). Whether any of the slice is allocated is the PCI core's own
  `/sys/bus/pci/devices/<bdf>/p2pmem/size` and `available` (equal when nothing is).
* The user of the memory must own those windows (`ALLOCATE_TLB`) and point them (`CONFIGURE_TLB`) at memory, for as long as the memory is mapped, and
  must **pin** them: a dma-buf export of the window (`EXPORT_TLB_DMABUF`, nothing need import it) keeps it allocated past `FREE_TLB` and `close()`.
  A window left unconfigured, freed, or reconfigured during a DMA sends the data wherever the NOC is told to send it.
* `RESET_DEVICE` with any destructive flag returns `EBUSY` while any of the slice is allocated, as it already does while a TLB window is exported
  as a dma-buf, and holds the whole slice for the length of the reset (`p2pmem/allocate` gives `ENOMEM` meanwhile): a reset under in-flight
  peer-to-peer DMA can wedge the host. System suspend is refused while any of it is allocated. Removing a device or unloading the module with the
  slice allocated logs a warning only: the PCI core can unmap the user mappings and wait for the pages only after this driver has stopped the card, and
  offers no way to revoke them sooner. Nothing may be allocated when the module is unloaded.
* A one-line `dmesg` record at load, and a warning, never an error, when the parameters are refused or the kernel cannot do it: the card carries on
  without p2pmem.

## Using it

`scripts/p2p/p2p_probe.py` is the user: it takes the window, aims it at a DRAM bank, maps the slice, reads a file or a range of a block device with
`O_DIRECT`, and checks the bytes through a second path. `scripts/p2p/p2p_topology.py` says beforehand, read-only, whether the kernel will allow the
drive and the card to talk (ACS redirects, root complex). The kernel needs `CONFIG_PCI_P2PDMA`, Linux 6.2 or later, the NVMe driver (not md, not
device-mapper: block stacking does not carry peer-to-peer capability), and a source with no holes (a hole is zero-filled by the CPU, which the
kernel must not do into device memory; the probe checks with FIEMAP).

## Rollback

Nothing to undo when the parameter is not set. To go back to the stock module: stop everything that has `/dev/tenstorrent/*` open, make sure
`p2pmem/available` equals `p2pmem/size`, `rmmod tenstorrent`, `modprobe tenstorrent`.

## Known limits

* The kernel maps the published pages uncached and reserves the BAR range as UC- in the PAT (`memremap_pages`, `MEMORY_DEVICE_PCI_P2PDMA`), so a
  CPU mapping of the same range made later (the driver's write-combining window mapping, in UMD or the probe) comes out UC- instead of
  write-combining: correct, but CPU reads and writes through that window are slow. Use a window nobody else maps. The user mapping from `allocate` is
  for DMA only; the CPU must not load or store through it, and the probe never does.
* `pci_p2pdma_add_resource()` has error-path bugs (a use-after-free when its last step fails) that are fixed in later kernels, not in 7.0; the
  patch validates the slice before calling it, which leaves only that last, unlikely, step.
* A slice in a BAR placed above the CPU's physical address limit cannot be given struct pages; the load then logs the error and publishes nothing.
