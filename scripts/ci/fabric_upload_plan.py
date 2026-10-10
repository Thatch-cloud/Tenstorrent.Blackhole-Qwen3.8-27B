"""The fabric upload plan's arithmetic and decision rule (docs/tp4-fabric-upload.md).

The (1, 4) mesh's four p150a cards do not share one kind of PCIe link: two train at x16 and two (behind a switch) at x4.
The owner's directive is that the x4 cards receive their bytes over the ethernet fabric from the x16 cards rather than
over their own PCIe. Whether that pays depends on numbers no run has measured yet (per-card host->device bandwidth,
whether the four cards' writes overlap, the fabric's unicast rate x16 -> x4), so this module holds:

  - the link facts a probe reads from sysfs (parse_link_speed, pcie_ceiling_gbps) and the UMD cluster descriptor's
    chip -> PCI device map (parse_mmio_chips);
  - the relay pairing on the (1, 4) line (relay_pairs): every x4 position gets an x16 sender, fewest fabric hops first;
  - the timing model (direct_seconds, relay_seconds, restore_bytes_per_card) the doc's estimates come from;
  - the pre-registered decision rule (decide) the two measurement jobs feed: RELAY only when the model, fed with the
    MEASURED rates, says the relay saves at least RELAY_MIN_GAIN of the direct time and the cards' writes overlap.

Stdlib only, Python 3.7 syntax: the probes import it inside the serving image and the CPU suite tests it.
"""

import itertools
import re
import statistics

# Line encoding per PCIe generation (the payload fraction of the raw GT/s). Gen6 FLIT mode is 242/256.
GEN_BY_GT = ((2.5, 1), (5.0, 2), (8.0, 3), (16.0, 4), (32.0, 5), (64.0, 6))
ENCODING = {1: 8.0 / 10.0, 2: 8.0 / 10.0, 3: 128.0 / 130.0, 4: 128.0 / 130.0, 5: 128.0 / 130.0, 6: 242.0 / 256.0}
# A link at or above this width is a sender; below it, a receiver of relayed bytes.
WIDE_LANES = 16
# The relay is adopted only when it saves at least this fraction of the direct time (and the absolute floor below).
RELAY_MIN_GAIN = 0.20
# Per-request bytes of a prefix restore per chip (docs/tp4-fabric-upload.md, section 1): the full-attention KV is
# 16 layers x K and V x one 256-wide head per chip in bfloat8_b (1,088 B per 32 x 32 tile), so 8,704 B per token per
# chip; a GDN checkpoint is 78,446,592 B for the whole mesh (bf16 state and carry), a quarter per chip.
KV_BYTES_PER_TOKEN_PER_CHIP = 16 * 2 * 256 * 1088 // 1024
GDN_CHECKPOINT_BYTES = 78446592
CHIPS = 4


BDF = re.compile(r'^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$')


class PlanError(ValueError):
    """Inputs the model cannot use (a missing rate, a negative byte count)."""


def parse_link_speed(text):
    """GT/s from a sysfs current_link_speed ('32.0 GT/s PCIe', '16 GT/s', '8.0 GT/s PCIe'), None when unreadable."""
    match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*GT/s', text or '')
    return float(match.group(1)) if match else None


def pcie_generation(gt_per_s):
    for rate, generation in GEN_BY_GT:
        if gt_per_s is not None and abs(gt_per_s - rate) < 0.01:
            return generation
    return None


def pcie_ceiling_gbps(gt_per_s, lanes):
    """The link's payload ceiling in decimal GB/s per direction after line encoding (TLP and DLLP overheads are NOT
    subtracted, so a real transfer lands below it). None when the speed or width is unknown."""
    generation = pcie_generation(gt_per_s)
    if generation is None or not lanes:
        return None
    return round(gt_per_s * int(lanes) * ENCODING[generation] / 8.0, 3)


def parse_mmio_chips(text):
    """{chip id: PCI device index} from a UMD cluster descriptor's chips_with_mmio section (block '- 0: 0' or flow
    '[{0: 0}, ...]' YAML), None when the section is absent. The index is the N of /dev/tenstorrent/N."""
    match = re.search(r'^chips_with_mmio:[^\n]*\n?', text or '', flags=re.M)
    if match is None:
        return None
    head = (text or '')[match.start():match.end()]
    rest = (text or '')[match.end():]
    end = re.search(r'^\S', rest, flags=re.M)
    section = head.split(':', 1)[1] + (rest[:end.start()] if end else rest)
    pairs = re.findall(r'(\d+)\s*:\s*(\d+)', section)
    return dict((int(chip), int(index)) for chip, index in pairs)


def _read(path):
    try:
        with open(path, encoding='utf-8') as handle:
            return handle.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def char_device_pci(node, sys_root='/sys', rdev_of=None):
    """The PCI address (0000:bb:dd.f) behind a character device node, through /sys/dev/char/MAJ:MIN/device (what
    qual_card.sh's qual_pci_of reads on the host; inside a container the node keeps its numbers). None when unknown."""
    import os
    try:
        rdev = rdev_of(node) if rdev_of else os.stat(node).st_rdev
    except OSError:
        return None
    link = os.path.join(sys_root, 'dev', 'char', '%d:%d' % (os.major(rdev), os.minor(rdev)), 'device')
    try:
        target = os.path.realpath(link)
    except OSError:
        return None
    name = os.path.basename(target)
    return name if BDF.match(name) else None


def pci_link(pci, sys_root='/sys'):
    """{speed_gt, width, max_speed_gt, max_width, ceiling_gbps, upstream, ports, iommu_group} for a PCI address from sysfs. upstream is
    the parent directory's PCI address (a root port, or a switch's downstream port for a card behind a switch); ports is
    the whole chain of PCI bridges above the card, nearest first, so two cards behind one switch share its later entries."""
    import os
    base = os.path.join(sys_root, 'bus', 'pci', 'devices', pci)
    speed = parse_link_speed(_read(os.path.join(base, 'current_link_speed')))
    width_text = _read(os.path.join(base, 'current_link_width'))
    max_speed = parse_link_speed(_read(os.path.join(base, 'max_link_speed')))
    max_width_text = _read(os.path.join(base, 'max_link_width'))
    width = int(width_text) if width_text and width_text.isdigit() else None
    max_width = int(max_width_text) if max_width_text and max_width_text.isdigit() else None
    ports = []
    try:
        path = os.path.dirname(os.path.realpath(base))
    except OSError:
        path = ''
    while path and BDF.match(os.path.basename(path)):
        ports.append(os.path.basename(path))
        path = os.path.dirname(path)
    # The pinned runtime's zero-copy write path (writes over 32 MiB read straight from the user buffer) needs the IOMMU:
    # a card in an IOMMU group says it is on for that device.
    return dict(speed_gt=speed, width=width, max_speed_gt=max_speed, max_width=max_width,
                ceiling_gbps=pcie_ceiling_gbps(speed, width), upstream=ports[0] if ports else None, ports=ports,
                iommu_group=os.path.exists(os.path.join(base, 'iommu_group')))


def card_links(order, mmio, dev_root='/dev/tenstorrent', sys_root='/sys', rdev_of=None):
    """[{pos, chip, pci_index, pci, speed_gt, width, ...}] for the mesh's chip ids in (1, 4) order, joining the UMD
    descriptor's chip -> PCI index map (parse_mmio_chips) to /dev/tenstorrent/<index> and its sysfs link. Fields stay
    None where a step is unknown: the probe reports what it could read and never guesses a width."""
    import os
    rows = []
    for pos, chip in enumerate(order):
        row = dict(pos=pos, chip=int(chip), pci_index=None, pci=None, speed_gt=None, width=None, max_speed_gt=None,
                   max_width=None, ceiling_gbps=None, upstream=None, ports=[], iommu_group=None)
        index = (mmio or {}).get(int(chip))
        if index is not None:
            row['pci_index'] = index
            pci = char_device_pci(os.path.join(dev_root, str(index)), sys_root=sys_root, rdev_of=rdev_of)
            row['pci'] = pci
            if pci:
                row.update(pci_link(pci, sys_root=sys_root))
        rows.append(row)
    return rows


def line_hops(sender, receiver):
    """Fabric hops between two positions of the (1, 4) line. FABRIC_1D routes along the line; the pinned
    point_to_point's Ring branch adds the mesh width to an already absolute hop count, so it never takes the wrap
    (docs/tp4-fabric-upload.md), and the closing edge (3, 0) is three hops."""
    return abs(int(receiver) - int(sender))


def relay_pairs(widths, wide=WIDE_LANES):
    """[(sender, receiver, hops)]: every position whose link is narrower than `wide` lanes gets a sender among the wide
    positions, fewest total hops first, then the smallest worst hop, then the lowest positions. A sender serves at most
    one receiver while there are at least as many wide positions as narrow ones. [] when any width is unknown or no
    position is narrow or none is wide."""
    widths = list(widths)
    if not widths or any(width is None for width in widths):
        return []
    narrow = [pos for pos, width in enumerate(widths) if int(width) < wide]
    senders = [pos for pos, width in enumerate(widths) if int(width) >= wide]
    if not narrow or not senders:
        return []
    best = None
    if len(senders) >= len(narrow):
        choices = itertools.permutations(senders, len(narrow))
    else:
        choices = itertools.product(senders, repeat=len(narrow))
    for chosen in choices:
        hops = [line_hops(sender, receiver) for sender, receiver in zip(chosen, narrow)]
        key = (sum(hops), max(hops), tuple(chosen))
        if best is None or key < best[0]:
            best = (key, chosen)
    return [(sender, receiver, line_hops(sender, receiver)) for sender, receiver in zip(best[1], narrow)]


def gbps(nbytes, seconds):
    """Decimal GB/s, None for a zero or missing time."""
    if not seconds:
        return None
    return round(float(nbytes) / float(seconds) / 1e9, 3)


def summarize(samples, nbytes):
    """The timing summary every probe arm reports: median, best and worst seconds, and the GB/s they imply."""
    samples = [float(sample) for sample in samples]
    if not samples:
        return dict(n=0, nbytes=int(nbytes))
    median = statistics.median(samples)
    return dict(n=len(samples), nbytes=int(nbytes), median_s=round(median, 6), min_s=round(min(samples), 6),
                max_s=round(max(samples), 6), gbps_median=gbps(nbytes, median), gbps_best=gbps(nbytes, min(samples)))


def _rate(rates, pos):
    rate = rates.get(pos) if isinstance(rates, dict) else rates[pos]
    if not rate or rate <= 0:
        raise PlanError('no host->device rate for position %s' % pos)
    return float(rate)


def _host_term(card_bytes, host_gbps):
    """Seconds the host needs for every card's bytes at its measured aggregate rate (0 without a cap)."""
    return float(sum(card_bytes)) / (float(host_gbps) * 1e9) if host_gbps else 0.0


def direct_seconds(card_bytes, h2d_gbps, overlap=True, host_gbps=None):
    """Every card takes its own bytes over its own PCIe. card_bytes and h2d_gbps are per position (lists or dicts).
    overlap: the four cards' writes run together (the time is the slowest card's), else one after another (the sum).
    host_gbps: the host's aggregate rate with every card written at once (F1's together arm): no upload of the mesh can
    beat sum(card_bytes) / host_gbps, whichever links the bytes take."""
    times = []
    for pos, nbytes in enumerate(card_bytes):
        if nbytes < 0:
            raise PlanError('negative bytes for position %d' % pos)
        times.append(nbytes / (_rate(h2d_gbps, pos) * 1e9) if nbytes else 0.0)
    return max(max(times) if overlap else sum(times), _host_term(card_bytes, host_gbps))


def relay_seconds(card_bytes, h2d_gbps, pairs, fabric_gbps, overlap=True, chunk_bytes=64 << 20, hop_penalty=1.0,
                  host_gbps=None):
    """The relay: each sender takes its own bytes plus its receivers' over PCIe (into a staging buffer), and the fabric
    moves each receiver's bytes on, pipelined chunk by chunk, so a pair costs max(host time, fabric time) plus one
    chunk's fabric time to fill the pipe. A receiver takes nothing over PCIe. fabric_gbps: one rate (every pair) or
    {(sender, receiver): rate}; a pair of more than one hop is divided by hop_penalty ** (hops - 1) when no per-pair
    rate is given (1.0: a multi-hop route keeps the rate, which the p2p job measures). overlap and host_gbps as in
    direct_seconds: the relay moves the same bytes across the host, so the host's aggregate cap binds it the same way."""
    receivers = dict((receiver, (sender, hops)) for sender, receiver, hops in pairs)
    load = dict((pos, float(nbytes)) for pos, nbytes in enumerate(card_bytes) if pos not in receivers)
    for receiver, (sender, _) in receivers.items():
        if sender in receivers:
            raise PlanError('position %d is both a sender and a receiver' % sender)
        load[sender] = load.get(sender, 0.0) + float(card_bytes[receiver])
    host = dict((pos, nbytes / (_rate(h2d_gbps, pos) * 1e9) if nbytes else 0.0) for pos, nbytes in load.items())
    fabric = {}
    for receiver, (sender, hops) in receivers.items():
        if isinstance(fabric_gbps, dict):
            rate = fabric_gbps.get((sender, receiver))
        else:
            rate = fabric_gbps / (hop_penalty ** max(0, hops - 1)) if fabric_gbps else None
        if not rate or rate <= 0:
            raise PlanError('no fabric rate for %d -> %d' % (sender, receiver))
        nbytes = float(card_bytes[receiver])
        fill = min(nbytes, float(chunk_bytes)) / (rate * 1e9)
        fabric[sender] = fabric.get(sender, 0.0) + nbytes / (rate * 1e9) + fill
    capped = _host_term(card_bytes, host_gbps)
    per_sender = dict((pos, max(host[pos], fabric.get(pos, 0.0))) for pos in host)
    if overlap:
        return max([capped] + list(per_sender.values()))
    # Serialized host writes: the PCIe parts add up; the fabric of the last sender drains after the last write.
    return max(capped, sum(host.values())) + max([fabric.get(pos, 0.0) - host[pos] for pos in host] + [0.0])


def replicated_seconds(nbytes, h2d_gbps, fabric_gbps, chips=CHIPS, overlap=True):
    """A tensor every card holds whole. Direct: each card takes it over its PCIe (chips uploads). Dedup: the fastest
    card takes it once and the fabric copies it on along the line (a pipelined chain: one fabric time plus a fill per
    extra hop is the floor; this charges a full fabric time per copy unless the copies overlap). Returns
    (direct_s, dedup_s)."""
    rates = [float(rate) for rate in (h2d_gbps.values() if isinstance(h2d_gbps, dict) else h2d_gbps)]
    if len(rates) != chips or min(rates) <= 0:
        raise PlanError('need %d positive rates' % chips)
    each = [nbytes / (rate * 1e9) for rate in rates]
    direct = max(each) if overlap else sum(each)
    once = nbytes / (max(rates) * 1e9)
    copies = (chips - 1) * nbytes / (float(fabric_gbps) * 1e9)
    return direct, once + (copies / (chips - 1) if overlap else copies)


def restore_bytes_per_card(tokens, kv_bytes_per_token=KV_BYTES_PER_TOKEN_PER_CHIP, gdn_bytes=GDN_CHECKPOINT_BYTES,
                           chips=CHIPS):
    """Bytes one card takes for a warm-tier restore of `tokens` prefix tokens: its quarter of the KV plus its quarter
    of the GDN checkpoint (logical bytes; tile padding of the conv carry adds about a third to the GDN part)."""
    return int(tokens) * int(kv_bytes_per_token) + int(gdn_bytes) // int(chips)


def decide(h2d, fabric, card_bytes, min_saving_s=0.0, wide=WIDE_LANES):
    """The decision the two jobs feed (docs/tp4-fabric-upload.md, section 4).

    h2d: {'widths': [lanes per position], 'alone_gbps': [GB/s per position, each card written alone],
          'together_s': seconds for the four cards written together with `together_bytes` each,
          'together_bytes': bytes per card in that arm}
    fabric: {(sender, receiver): GB/s} measured x16 -> x4 (both pairs at once, or the faster exact socket).
    card_bytes: bytes per position the decision is for (a weight load, a restore).

    The host's aggregate rate (host_gbps = cards x together_bytes / together_s) caps every upload of the mesh whichever
    links its bytes take; both times are computed under it. (Revised after F1, run 38036227448: the first version
    classed the together arm as overlapping or serial; four cards together reached 23.9 GB/s, more than one card alone
    and far less than the 68 GB/s the four alone rates add up to, which neither class described.)

    Returns dict(verdict, reasons, direct_s, relay_s, pairs, host_gbps, overlap). Verdicts:
      RELAY            the relay saves >= RELAY_MIN_GAIN of the direct time and >= min_saving_s
      DIRECT           it does not (the fabric, or the senders' doubled PCIe load, eats the gain)
      HOST-BOUND       the host, not the x4 links, sets the direct time: the host's aggregate cap is at least the
                       slowest card's own link time, or the x4 cards alone are within RELAY_MIN_GAIN of the x16 cards.
                       Moving bytes between links cannot help; the host path is the fix
      NOT-MEASURED     a rate, a width or a pair is missing
    overlap: host_gbps over the sum of the alone rates (1.0 = the cards' writes overlap fully)."""
    reasons = []
    widths = h2d.get('widths') or []
    alone = h2d.get('alone_gbps') or []
    pairs = relay_pairs(widths, wide=wide)
    out = dict(verdict='NOT-MEASURED', reasons=reasons, direct_s=None, relay_s=None, pairs=pairs, host_gbps=None, overlap=None)
    if len(widths) != len(card_bytes) or len(alone) != len(card_bytes) or any(not rate for rate in alone):
        reasons.append('per-position widths and alone rates are required for every card')
        return out
    if not pairs:
        reasons.append('no x%d sender for an x4 receiver (widths %s)' % (wide, widths))
        return out
    together, together_bytes = h2d.get('together_s'), h2d.get('together_bytes')
    if not together or not together_bytes:
        reasons.append('the four-cards-together arm is missing')
        return out
    host_gbps = len(card_bytes) * float(together_bytes) / float(together) / 1e9
    out.update(host_gbps=round(host_gbps, 3), overlap=round(host_gbps / sum(float(rate) for rate in alone), 3))
    direct = direct_seconds(card_bytes, alone, host_gbps=host_gbps)
    out['direct_s'] = round(direct, 6)
    link_term = direct_seconds(card_bytes, alone)
    host_term = _host_term(card_bytes, host_gbps)
    narrow = max(float(alone[receiver]) for _, receiver, _ in pairs)
    wide_rate = min(float(alone[sender]) for sender, _, _ in pairs)
    if host_term >= link_term:
        out['verdict'] = 'HOST-BOUND'
        reasons.append('the host moves the mesh at %.1f GB/s: %.3f s for these bytes against %.3f s for the slowest card on its own '
                       'link; every relayed byte still crosses the host' % (host_gbps, host_term, link_term))
        return out
    if narrow >= (1.0 - RELAY_MIN_GAIN) * wide_rate:
        out['verdict'] = 'HOST-BOUND'
        reasons.append('the x4 cards alone reach %.3f GB/s against %.3f GB/s for the x16 senders: the links do not set the rate'
                       % (narrow, wide_rate))
        return out
    missing = [(sender, receiver) for sender, receiver, _ in pairs if not (fabric or {}).get((sender, receiver))]
    if missing:
        reasons.append('no fabric rate for %s' % ', '.join('%d->%d' % pair for pair in missing))
        return out
    relay = relay_seconds(card_bytes, alone, pairs, fabric, host_gbps=host_gbps)
    out['relay_s'] = round(relay, 6)
    saving = direct - relay
    if direct > 0 and saving >= RELAY_MIN_GAIN * direct and saving >= min_saving_s:
        out['verdict'] = 'RELAY'
        reasons.append('relay %.3f s against direct %.3f s (saves %.0f%%)' % (relay, direct, 100.0 * saving / direct))
    else:
        out['verdict'] = 'DIRECT'
        reasons.append('relay %.3f s against direct %.3f s: under the %.0f%% / %.3f s bar'
                       % (relay, direct, 100 * RELAY_MIN_GAIN, min_saving_s))
    return out
