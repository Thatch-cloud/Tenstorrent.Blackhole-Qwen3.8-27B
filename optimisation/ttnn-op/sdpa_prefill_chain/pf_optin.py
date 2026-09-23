"""Python opt-in for the [QWEN-SDPA-PF] G6 K/V chain (sdpa-prefill-share-spec.md 3.6).

The model's chunked prefill SDPA (attention/tp.py, forward_prefill_paged) builds its
SDPAProgramConfig at the call site. With QWEN_SDPA_PREFILL_KVCHAIN=1 the grafted tp.py sets
max_cores_per_head_batch = 0x5EFA0000 | flags on the calls the chain is QUALIFIED for: the flexible
path (a device chunk_start tensor), bf8 K/V (QWEN_SDPA_BF8=1), q/k chunk 128 and S in
apply_factory_pf.QUALIFIED_ROWS (512, 1024, 2048: exactly the row counts the card-M Q1 sweep
covers). Everything else - the environment off, the legacy int chunk_start path, bf16 mode, a
256-1792-row tail chunk or S > 2048 (unequal chain groups) - builds exactly the served kwargs. To
admit another S, add it to QUALIFIED_ROWS and to the card-M test's ROWS (the CPU tests keep them
equal), so it is swept on card M before the model can send it.

    QWEN_SDPA_PREFILL_KVCHAIN=1               opt in
    QWEN_SDPA_PREFILL_KVCHAIN_FLAGS=0x1       production flags 0x1..0x7 (default 0x1, see below)

DEFAULT FLAGS: 0x1, the chain alone. The spec's served value was 0x3 (0x2: the injector's K/V
read barrier every 32 tiles), but K0 on card M (image A' 1b9b6445) measured K0b4 (the 16
injectors alone at the served cadence) 0.2114 ms/1k keys against K0b32 0.2116 and the compute
floor K0a 0.2111: one injector per group already reaches the floor at the served read-ahead, so
0x2 buys nothing and stays an A/B knob only (Q2 arm chain_b).

A stock _ttnncpp.so silently ignores the word (the served factory never reads that field): the
opt-in is refused unless the loaded binary carries the chain factory's '[QWEN-SDPA-PF] flags='
literal, before any config changes.

TP_HELPERS is the code the graft adds to attention/tp.py (self-contained: the image has no copy of
this module); the CPU tests exec it and drive it with a fake environment and /proc/self/maps.
patch_tp() applies the two anchored edits plus the helpers to attention/tp.py text; unpatch_tp()
inverts it. Wiring patch_tp into the arm's graft staging (lever_n_m3native_patch.stage, a
scripts/ci change: check both image copy lists) is the Q3 step and is NOT done here.
"""

import argparse
import ast
import hashlib
import sys
from pathlib import Path

NL = chr(10)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import apply_factory_pf as factory  # noqa: E402

ENV = 'QWEN_SDPA_PREFILL_KVCHAIN'
FLAGS_ENV = 'QWEN_SDPA_PREFILL_KVCHAIN_FLAGS'
DEFAULT_FLAGS = factory.FLAG_KV_CHAIN
BINARY_MARKER = factory.LOG_MARKER.encode('utf-8')
PINDIAG = '[PINDIAG] sdpa prefill kvchain flags='
# attention/tp.py of the image (the probe 35503727180 dump), sha256 prefix. Q3 applies patch_tp AFTER
# lever_n_m3native_patch.patch_attention_tp, whose output changes with every m3native revision, so the
# patch itself refuses drift the way stage() does (every anchor exactly once, ast.parse of the result)
# rather than by a base sha; the CLI reports whether its input is this pinned file or not.
TP_PINNED_PREFIX = 'e0c685a43796f6f8'
QUALIFIED_ROWS = factory.QUALIFIED_ROWS
QUALIFIED_CHUNK = factory.QUALIFIED_CHUNK

TP_HELPERS = NL.join((
    '',
    '',
    '# [QWEN-SDPA-PF] prefill lever #1 (optimisation/ttnn-op/sdpa_prefill_chain/pf_optin.py): the per-call opt-in',
    '# to the G6 K/V chain of the chunked prefill SDPA. Off unless %s=1.' % ENV,
    '_QWEN_PF_TAG = 0x%08X' % factory.PF_TAG,
    '_QWEN_PF_PRODUCTION = 0x%X' % factory.PRODUCTION_FLAGS,
    '_QWEN_PF_MARKER = %r' % BINARY_MARKER,
    '_QWEN_PF_ROWS = %r  # the card-M-qualified S (apply_factory_pf.QUALIFIED_ROWS)' % (QUALIFIED_ROWS,),
    '_QWEN_PF_CHUNK = %d' % QUALIFIED_CHUNK,
    '',
    '',
    'def _qwen_pf_pindiag(text):',
    '    try:',
    '        from loguru import logger',
    '    except ImportError:',
    '        print(text, flush=True)',
    '        return',
    '    logger.info(text)',
    '',
    '',
    'def _qwen_pf_binary_has_marker(maps="/proc/self/maps"):',
    '    """The _ttnncpp.so this process mapped carries the chain factory (a stock .so ignores the word)."""',
    '    import mmap',
    '',
    '    with open(maps) as handle:',
    '        paths = sorted({line.split()[-1] for line in handle if line.rstrip().endswith("_ttnncpp.so")})',
    '    if len(paths) != 1:',
    '        raise RuntimeError("[QWEN-SDPA-PF] expected exactly one mapped _ttnncpp.so, found %r" % (paths,))',
    '    with open(paths[0], "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:',
    '        return view.find(_QWEN_PF_MARKER) >= 0',
    '',
    '',
    'def _qwen_pf_word(environ=None, binary_check=None):',
    '    """None (served config) or 0x5EFA0000 | flags. Raises on bad flags or a binary without the chain."""',
    '    environ = os.environ if environ is None else environ',
    '    if environ.get("%s", "0") != "1":' % ENV,
    '        return None',
    '    flags = int(environ.get("%s", "0x%X"), 0)' % (FLAGS_ENV, DEFAULT_FLAGS),
    '    if not (flags & 0x1) or (flags & ~_QWEN_PF_PRODUCTION):',
    '        raise RuntimeError("%s must be production flags 0x1..0x7, got %%#x" %% flags)' % FLAGS_ENV,
    '    if not (_qwen_pf_binary_has_marker if binary_check is None else binary_check)():',
    '        raise RuntimeError("%s=1 but the loaded _ttnncpp.so lacks the [QWEN-SDPA-PF] chain factory "' % ENV,
    '                           "(mount the K64g graft)")',
    '    word = _QWEN_PF_TAG | flags',
    '    _qwen_pf_pindiag("%s%%#x" %% word)' % PINDIAG,
    '    return word',
    '',
    '',
    'def _qwen_pf_cfg_kwargs(grid, qk_chunk, word, flexible, sdpa_bf8, S):',
    '    """The chunked SDPA program-config kwargs: the served ones, plus the chain word when it applies."""',
    '    kwargs = dict(compute_with_storage_grid_size=grid, exp_approx_mode=False, q_chunk_size=qk_chunk,',
    '                  k_chunk_size=qk_chunk)',
    '    if word is not None and flexible and sdpa_bf8 and qk_chunk == _QWEN_PF_CHUNK and S in _QWEN_PF_ROWS:',
    '        kwargs["max_cores_per_head_batch"] = word  # flexible path, bf8 K/V, a card-M-qualified S',
    '    return kwargs',
    ''))

INIT_ANCHOR = '        self._sdpa_bf8 = os.environ.get("QWEN_SDPA_BF8", "0") == "1"' + NL
INIT_NEW = INIT_ANCHOR + (
    '        self._sdpa_pf_word = _qwen_pf_word()  # [QWEN-SDPA-PF] None unless %s=1 (refuses a stock .so)' % ENV + NL)
CALL_ANCHOR = NL.join((
    '        sdpa_cfg = ttnn.SDPAProgramConfig(',
    '            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),',
    '            exp_approx_mode=False,',
    '            q_chunk_size=qk_chunk,',
    '            k_chunk_size=qk_chunk,',
    '        )',
    ''))
CALL_NEW = NL.join((
    '        sdpa_cfg = ttnn.SDPAProgramConfig(  # [QWEN-SDPA-PF] the served kwargs unless the chain applies',
    '            **_qwen_pf_cfg_kwargs(self.mesh.compute_with_storage_grid_size(), qk_chunk,',
    '                                  getattr(self, "_sdpa_pf_word", None), chunk_start_idx_tensor is not None,',
    '                                  self._sdpa_bf8, S)',
    '        )',
    ''))
FUNCTION = '    def forward_prefill_paged('


def function_span(text, header=FUNCTION):
    """[start, end) character span of forward_prefill_paged (to the next method at the same indent)."""
    start = text.index(header)
    following = text.find(NL + '    def ', start + len(header))
    return start, (len(text) if following < 0 else following + 1)


def patch_tp(text):
    """attention/tp.py text -> text with the opt-in: the init line, the call site (inside
    forward_prefill_paged only) and TP_HELPERS appended. Every anchor must occur exactly once."""
    if '_qwen_pf_word' in text:
        raise ValueError('attention/tp.py already carries the [QWEN-SDPA-PF] opt-in')
    if text.count(INIT_ANCHOR) != 1:
        raise ValueError('init anchor occurs %d times' % text.count(INIT_ANCHOR))
    start, end = function_span(text)
    body = text[start:end]
    if body.count(CALL_ANCHOR) != 1 or text.count(CALL_ANCHOR) != 1:
        raise ValueError('call-site anchor occurs %d times in forward_prefill_paged (%d in the file)'
                         % (body.count(CALL_ANCHOR), text.count(CALL_ANCHOR)))
    body = body.replace(CALL_ANCHOR, CALL_NEW)
    text = text[:start] + body + text[end:]
    text = text.replace(INIT_ANCHOR, INIT_NEW)
    if not text.endswith(NL):
        text += NL
    text += TP_HELPERS
    ast.parse(text)                              # as lever_n_m3native_patch.stage(): a broken result fails here
    return text


def unpatch_tp(text):
    if not text.endswith(TP_HELPERS):
        raise ValueError('the helpers are not at the end of the file')
    text = text[:-len(TP_HELPERS)]
    for new, old in ((INIT_NEW, INIT_ANCHOR), (CALL_NEW, CALL_ANCHOR)):
        if text.count(new) != 1:
            raise ValueError('edited text occurs %d times' % text.count(new))
        text = text.replace(new, old)
    return text


def helpers_namespace(os_module=None):
    """TP_HELPERS executed in a fresh namespace (the CPU tests drive the helpers through it)."""
    import os as real_os

    namespace = dict(os=os_module or real_os)
    exec(compile(TP_HELPERS, 'attention/tp.py [QWEN-SDPA-PF] helpers', 'exec'), namespace)
    return namespace


def main(argv=None):
    parser = argparse.ArgumentParser(description='apply (or check) the [QWEN-SDPA-PF] opt-in on attention/tp.py')
    parser.add_argument('tp', type=Path)
    parser.add_argument('--out', type=Path, help='write the patched file here (default: print a summary only)')
    options = parser.parse_args(argv)
    text = options.tp.read_text(encoding='utf-8')
    patched = patch_tp(text)
    if unpatch_tp(patched) != text:
        print('edits do not invert', file=sys.stderr)
        return 1
    digest = hashlib.sha256(patched.encode('utf-8')).hexdigest()
    base = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if options.out:
        options.out.write_text(patched, encoding='utf-8', newline=NL)
    print('attention/tp.py %s (%s) -> [QWEN-SDPA-PF] opt-in %s%s' % (
        base[:16], 'the pinned image file' if base.startswith(TP_PINNED_PREFIX) else
        'NOT the pinned image file %s: anchors matched (e.g. after patch_attention_tp)' % TP_PINNED_PREFIX,
        digest[:16], (' written ' + str(options.out)) if options.out else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
