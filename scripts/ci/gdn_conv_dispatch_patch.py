"""Let a full chunk take the native causal conv1d instead of the MAC FIR.

Run 35427384650 logged the dispatch and found `valid_len` is never None: 432 of
576 calls arrive with valid_len == T == 2048, a full chunk with nothing to mask,
and the rest are genuine tails. So this condition in gdn/tp.py never fires:

    if self._gdn_conv1d and valid_len is None:
        conv, conv_new_state = self._conv1d_prefill(qkv, T, _cstate)
    else:
        conv, conv_new_state = _causal_conv1d_fir(qkv, ...)

`_gdn_conv1d` is True unconditionally, so the native ttnn.conv1d path is dead
code in serving. The FIR it falls back to unrolls the K=4 convolution into three
shifted windows; a shift of 1, 2 or 3 tokens is never tile-aligned, so each tap
forces untilize/slice/tilize over the whole [1, S, 5120] activation. In the
device profile that is 1,152 cycles, 473 ms, 12.8% of prefill.

The fix is one condition, and tp.py's own comment already states the equivalence
it relies on:

    Masked buckets still pass a real valid_len (< T) so their exact masking is
    unchanged, and for a full chunk the None slice and the valid_len==T one-hot
    select the identical rows.

Only the carry extraction ever depended on valid_len. The convolution did not.

Because that equivalence is an argument and not a measurement, the gate that
uses this patch compares generated token ids between arms. If the two paths are
not numerically identical the tokens diverge and the gate fails, which is the
only thing that makes the change safe to keep.

The marker log stays in so the gate can prove the lever moved: an arm that
silently kept taking the FIR would otherwise report a speedup of zero and read
as a physical result.
"""

import io
import sys

MARKER = '[CONVPATH]'
OLD = '        if self._gdn_conv1d and valid_len is None:'
NEW = ('        _full_chunk = valid_len is None or valid_len == T\n'
       '        logger.info("%s gdn_conv1d=%%s valid_len=%%s T=%%s full=%%s"\n'
       '                    %% (self._gdn_conv1d, valid_len, T, _full_chunk))\n'
       '        if self._gdn_conv1d and _full_chunk:' % MARKER)


def patch_tp(source):
    source = source.replace('\r\n', '\n')
    if MARKER in source:
        raise SystemExit('tp.py already carries the marker')
    if source.count(OLD) != 1:
        raise SystemExit('expected exactly one dispatch, found %d' % source.count(OLD))
    source = source.replace(OLD, NEW, 1)
    if 'from loguru import logger' not in source:
        source = source.replace('import ttnn', 'import ttnn\nfrom loguru import logger', 1)
    return source


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: gdn_conv_dispatch_patch.py <in> <out>')
    source = io.open(sys.argv[1], encoding='utf-8').read()
    patched = patch_tp(source)
    import ast
    ast.parse(patched)
    io.open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(patched)
    # Grep for the NEW behaviour, never the absence of the old.
    assert 'valid_len is None or valid_len == T' in patched
    assert 'if self._gdn_conv1d and _full_chunk:' in patched
    print('patched %s (%d -> %d bytes)' % (sys.argv[1], len(source), len(patched)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
