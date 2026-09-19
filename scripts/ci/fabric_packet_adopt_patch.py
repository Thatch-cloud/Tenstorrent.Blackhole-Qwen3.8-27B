"""Adopt the 8192 B fabric packet payload in the serving worker.

Authorised 2026-09-19. Measured +8.6% all-gather at the large shape and +22.8%
at the small one (runs 35425948829, 35426073865), lever verified by reading
`get_tt_fabric_max_payload_size_bytes()` back and by the runtime's own
"suboptimal for transporting" warning falling from 6 occurrences to 0.

Why 8192: `ccl_common.cpp::validate_packet_size` computes the ideal as
`min(hw_max / page_size, 4) * page_size`. Blackhole allows a 15232 B payload and
scatter-write caps a packet at 4 chunks, so for the 2048 B page of a bfloat16
32x32 tile the ideal is 4 * 2048 = 8192. The default 4352 carries two pages and
wastes a further 256 B. Filed upstream as tenstorrent/tt-metal#57083.

The patch is a single keyword on the one call that brings the fabric up for a
multi-device mesh, `worker.py:673`. Deliberately NOT applied to the DISABLED
call below it, which tears the fabric down and takes no router config.

Reads and writes with explicit encoding and newline, and normalises CRLF before
matching, because the Write tool emits CRLF and bash on the rig then fails with
"unexpected end of file".
"""

import io
import sys

CALL = 'ttnn.set_fabric_config(fabric_config, reliability_mode)'
PATCHED = ('ttnn.set_fabric_config(\n'
           '            fabric_config,\n'
           '            reliability_mode,\n'
           '            router_config=_qwen_router_config(),\n'
           '        )')

HELPER = '''

def _qwen_router_config():
    """Request the packet payload size the CCL validator computes as ideal.

    Blackhole allows a 15232 B payload and scatter-write caps a packet at four
    chunks, so a 2048 B tile page wants 4 * 2048 = 8192. The stack default of
    4352 fits two pages and wastes 256 B; the runtime warns about it and then
    nothing acts on the warning. Upstream: tenstorrent/tt-metal#57083.

    Returns None when this build has no FabricRouterConfig, so an older image
    keeps working rather than failing to start.
    """
    builder = getattr(ttnn, "FabricRouterConfig", None)
    if builder is None:
        return None
    try:
        return builder(max_packet_payload_size_bytes=8192)
    except TypeError:
        config = builder()
        config.max_packet_payload_size_bytes = 8192
        return config

'''


def patch_worker(source):
    """Insert the helper and route the multi-device fabric call through it."""
    source = source.replace('\r\n', '\n')
    if '_qwen_router_config' in source:
        raise SystemExit('worker.py already patched')
    if source.count(CALL) != 1:
        raise SystemExit('expected exactly one %r, found %d'
                         % (CALL, source.count(CALL)))
    source = source.replace(CALL, PATCHED, 1)

    anchor = 'def get_fabric_config(tt_config, num_devices):'
    if anchor not in source:
        raise SystemExit('get_fabric_config anchor missing')
    return source.replace(anchor, HELPER.lstrip('\n') + '\n' + anchor, 1)


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: fabric_packet_adopt_patch.py <in> <out>')
    source = io.open(sys.argv[1], encoding='utf-8').read()
    patched = patch_worker(source)
    import ast
    ast.parse(patched)
    io.open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(patched)
    # Grep for the NEW behaviour, never for the absence of the old.
    assert 'router_config=_qwen_router_config()' in patched
    assert 'max_packet_payload_size_bytes=8192' in patched
    print('patched %s -> %s (%d -> %d bytes)'
          % (sys.argv[1], sys.argv[2], len(source), len(patched)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
