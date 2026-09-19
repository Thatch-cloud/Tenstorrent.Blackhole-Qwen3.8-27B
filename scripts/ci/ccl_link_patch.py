"""Make get_num_links count the cables that are actually fitted.

Upstream issue 55125, filed by this project on 2026-09-02. `tt_ccl.py` resolves a
product name from the DEVICE COUNT alone - two Blackhole devices become "P300" -
and then looks that name up in a fixed table where "P300" is (2, 2), because a
real p300 is two dies on one package with two links. Our pair is two boards with
two QSFP-DD cables, which is four links. The link count is a property of the
cabling, not of how many boards there are.

The cluster descriptor already carries the truth. `ttnn.cluster` will serialise it
and it lists every ethernet connection; run 35425330364 read four between chip 0
and chip 1 on this rig. So the fix is to read it rather than to guess from a name.

This patch is deliberately conservative:

  - discovery failing for ANY reason falls back to the original table, so a bad
    descriptor cannot stop the model starting
  - the discovered count is only ever used when it EXCEEDS the table value, so
    this cannot reduce links below what upstream would have chosen
  - the result is cached, because get_num_links is called per collective and
    serialising the descriptor each time would cost more than the links save
  - it logs what it found, so the arm can prove the lever moved rather than be
    assumed to have

WHAT IT IS BEING USED TO TEST. Prefill loses about 280 ms, 7.6% of device time,
to collectives waiting: the two chips do identical non-collective work (2322.5
against 2351.0 ms) and identical total collective time, but alternate stalling
inside them, and the imbalance GROWS through a run. If that drift is a
consequence of collectives being starved at half their links, this removes it. If
the stall survives at four links, it is an independent defect and the upstream
report gets that as evidence.
"""

import io
import sys

MARKER = '_qwen_discovered_links'

HELPER = '''

_QWEN_LINK_CACHE = {}


def _qwen_count_descriptor_links(text, a=0, b=1):
    """Count ethernet links between two chips in a UMD cluster descriptor."""
    import re
    parts = text.split("ethernet_connections:", 1)
    if len(parts) < 2:
        return None
    block = parts[1].split("\\nethernet_connections_to_remote_devices:", 1)[0]
    links = 0
    for record in re.split(r"\\n\\s*-\\s*\\n", block):
        chips = [int(m) for m in re.findall(r"chip:\\s*(\\d+)", record)]
        if len(chips) == 2 and set(chips) == set([a, b]):
            links += 1
    return links


def _qwen_discovered_links(mesh_device):
    """Links actually cabled between the mesh's chips, or None if unknowable.

    Upstream issue 55125: the table below is keyed by a name derived purely from
    device count, so cabling is invisible to it. The cluster descriptor is not.
    """
    key = id(mesh_device)
    if key in _QWEN_LINK_CACHE:
        return _QWEN_LINK_CACHE[key]
    found = None
    try:
        if mesh_device.get_num_devices() == 2:
            path = ttnn.cluster.serialize_cluster_descriptor()
            with open(path, "r") as handle:
                found = _qwen_count_descriptor_links(handle.read())
            if found:
                logger.info("[CCLLINKS] cluster descriptor reports %d links", found)
    except BaseException as error:
        logger.warning("[CCLLINKS] discovery failed, keeping the table: %s", error)
        found = None
    _QWEN_LINK_CACHE[key] = found
    return found

'''

OLD = '''    device_links = link_dict[device_name]'''
NEW = '''    device_links = link_dict[device_name]
    # Issue 55125: prefer what the cabling actually provides, and only ever
    # upward, so this cannot pick fewer links than upstream would have.
    _found = _qwen_discovered_links(mesh_device)
    if _found and _found > min(device_links):
        logger.info("[CCLLINKS] overriding %s %s with %d discovered links",
                    device_name, device_links, _found)
        device_links = (_found, _found)'''


def patch_ccl(source):
    source = source.replace('\r\n', '\n')
    if MARKER in source:
        raise SystemExit('tt_ccl.py already patched')
    if source.count(OLD) != 1:
        raise SystemExit('expected exactly one link_dict lookup, found %d'
                         % source.count(OLD))
    source = source.replace(OLD, NEW, 1)

    anchor = 'def get_num_links(mesh_device'
    if anchor not in source:
        raise SystemExit('get_num_links anchor missing')
    source = source.replace(anchor, HELPER.lstrip('\n') + '\n' + anchor, 1)

    if 'from loguru import logger' not in source:
        source = source.replace('import ttnn', 'import ttnn\nfrom loguru import logger', 1)
    return source


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: ccl_link_patch.py <in> <out>')
    source = io.open(sys.argv[1], encoding='utf-8').read()
    patched = patch_ccl(source)
    import ast
    ast.parse(patched)
    io.open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(patched)
    # Grep for the NEW behaviour, never the absence of the old.
    assert 'discovered links' in patched
    assert '_found > min(device_links)' in patched
    print('patched %s (%d -> %d bytes)' % (sys.argv[1], len(source), len(patched)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
