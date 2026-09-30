"""Log which causal-conv path each GDN prefill chunk takes.

Static tracing got this far and no further. The prefill profile contains 1,152
`Untilize -> Slice -> Tilize -> Ternary` cycles costing 473 ms, 12.8% of prefill
device time, of which 357 ms is pure layout conversion. Ternary appears at
exactly two sites in the whole model, and 1,152 / 3 taps = 384 = 48 GDN layers x
8 chunks, which rules out the vision splice (once per chunk, not per layer) and
points at `ttnn.mac` in the causal-conv FIR.

The dispatch is `gdn/tp.py`:

    if self._gdn_conv1d and valid_len is None:
        conv, conv_new_state = self._conv1d_prefill(qkv, T, _cstate)   # native conv1d
    else:
        conv, conv_new_state = _causal_conv1d_fir(qkv, ...)            # MAC FIR

`_gdn_conv1d` is set True unconditionally, so the fallback can only be taken
because `valid_len is not None`. `forward_prefill_batched` is worse: it calls the
FIR with no condition at all.

What static reading cannot settle is which branch the SERVING path actually
takes per chunk, and how often. model.py passes `valid_len=None` for full chunks
and a real value for masked and tail chunks, so the answer decides whether this
is worth 12.8% of prefill or a few tenths.

So: log it, and count. Grep the run for the NEW marker rather than reasoning
about which branch should have been taken - three levers this month were inert
and every one of them looked correct in the source.
"""

import io
import sys

MARKER = '[CONVPATH]'

DISPATCH_OLD = """        if self._gdn_conv1d and valid_len is None:"""
DISPATCH_NEW = """        logger.info("%s single gdn_conv1d=%%s valid_len=%%s T=%%s"
                    %% (self._gdn_conv1d, valid_len, T))
        if self._gdn_conv1d and valid_len is None:""" % MARKER

BATCHED_OLD = """        conv, conv_new_state = _causal_conv1d_fir("""
BATCHED_NEW = """        logger.info("%s batched FIR unconditional valid_lens=%%s" %% (valid_lens,))
        conv, conv_new_state = _causal_conv1d_fir(""" % MARKER


def patch_tp(source):
    source = source.replace('\r\n', '\n')
    if MARKER in source:
        raise SystemExit('tp.py already patched')

    if source.count(DISPATCH_OLD) != 1:
        raise SystemExit('expected 1 single-prefill dispatch, found %d'
                         % source.count(DISPATCH_OLD))
    source = source.replace(DISPATCH_OLD, DISPATCH_NEW, 1)

    # The batched variant calls the FIR with no condition; patch only the call
    # that sits inside forward_prefill_batched, which is the LAST one.
    count = source.count(BATCHED_OLD)
    if count < 1:
        raise SystemExit('no _causal_conv1d_fir call found')
    head, _, tail = source.rpartition(BATCHED_OLD)
    source = head + BATCHED_NEW + tail

    if 'from loguru import logger' not in source and 'import logger' not in source:
        source = source.replace('import ttnn', 'import ttnn\nfrom loguru import logger', 1)
    return source


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: gdn_conv_path_patch.py <in> <out>')
    source = io.open(sys.argv[1], encoding='utf-8').read()
    patched = patch_tp(source)
    import ast
    ast.parse(patched)
    io.open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(patched)
    assert patched.count(MARKER) >= 2
    print('patched %s (%d -> %d bytes), %d markers'
          % (sys.argv[1], len(source), len(patched), patched.count(MARKER)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
