"""Python opt-in for the [QWEN-SDPA-PF] G6 K/V chain (sdpa-prefill-share-spec.md 3.6 and 6.4).

WIRED. The opt-in is lever_n_m3native_patch section I (scripts/ci), the one copy the model gate's
graft stages: PATCHES['attention/tp.py'] is patch_attention_tp_full, the decode-side
patch_attention_tp followed by patch_attention_tp_sdpa_pf. This module re-exports it under the
names this directory's tests and README use, and keeps the CLI that applies it to a tp.py file.
It carries no second copy of the helpers: what is tested here is what the gate mounts.

The model's chunked prefill SDPA (attention/tp.py, forward_prefill_paged) builds its
SDPAProgramConfig at the call site. With QWEN_FAST_SDPA_PF=1 the grafted tp.py rebuilds that config
with max_cores_per_head_batch = 0x5EFA0000 | flags on the calls the chain is QUALIFIED for: the
flexible path (a device chunk_start tensor), bf8 K/V (QWEN_SDPA_BF8=1), q/k chunk 128 and S in
apply_factory_pf.QUALIFIED_ROWS (512, 1024, 2048: exactly the row counts the card-M Q1 sweep
covers). Everything else - the flag off, the legacy int chunk_start path, bf16 mode, a
256-1792-row tail chunk - keeps the served config: the served statement stays verbatim and the
opt-in is inserted after it.

    QWEN_FAST_SDPA_PF=1                 opt in (the arm: M3NATIVE_SDPA_PF=1 with KOPGRAFT64 a K64g graft)
    QWEN_FAST_SDPA_PF_FLAGS=0x3         a production flag set 0x1 / 0x3 / 0x5 / 0x7 (default 0x3)

DEFAULT FLAGS: 0x3, the chain plus the injector's read-ahead 32. Q2 on card M (graft K64g, Q in L1,
the model's 2080-block page table) measured per-step 21.4 us for the baseline, 20.7 for 0x1, 16.5
for 0x3 (0.770x, exact), 22.3 for 0x5 and 18.9 for 0x7. (K0 had suggested 0x1: at K0's bench
config one injector already reached the compute floor at the served read-ahead. Q2's config is the
model's, and there the injector's read-ahead is what pays.)

A stock _ttnncpp.so silently ignores the word (the served factory never reads that field), so the
opt-in is refused unless the loaded binary carries the chain factory's '[QWEN-SDPA-PF] flags='
literal, before any config changes.
"""

import argparse
import hashlib
import sys
from pathlib import Path

NL = chr(10)
HERE = Path(__file__).resolve().parent
SCRIPTS_CI = HERE.parents[2] / 'scripts' / 'ci'
for _path in (str(SCRIPTS_CI),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import lever_n_m3native_patch as graft  # noqa: E402

ENV = graft.SDPA_PF_FLAG
FLAGS_ENV = graft.SDPA_PF_FLAGS_FLAG
DEFAULT_FLAGS = graft.SDPA_PF_DEFAULT_FLAGS
PRODUCTION_FLAGS = graft.SDPA_PF_PRODUCTION_FLAGS
BINARY_MARKER = graft.SDPA_PF_BINARY_MARKER.encode('ascii')
PINDIAG = graft.MARKER_SDPA_PF
QUALIFIED_ROWS = graft.SDPA_PF_ROWS
QUALIFIED_CHUNK = graft.SDPA_PF_CHUNK
TP_HELPERS = graft.SDPA_PF_HELPERS
INIT_NEW = graft.SDPA_PF_INIT_NEW
CALL_NEW = graft.SDPA_PF_CALL_NEW
# attention/tp.py of the image, sha256 prefix: the probe 35503727180 dump, and every m3native graft
# through v143 staged exactly this file (scripts/ci/fixtures/qwen36_attention_tp.py is a copy).
TP_PINNED_PREFIX = 'e0c685a43796f6f8'

patch_tp = graft.patch_attention_tp_sdpa_pf
unpatch_tp = graft.unpatch_attention_tp_sdpa_pf
patch_tp_full = graft.patch_attention_tp_full


def helpers_namespace(os_module=None, ttnn_module=None):
    """TP_HELPERS executed in a fresh namespace (the CPU tests drive the helpers through it)."""
    import os as real_os

    namespace = dict(os=os_module or real_os, ttnn=ttnn_module)
    exec(compile(TP_HELPERS, 'attention/tp.py [QWEN-SDPA-PF] helpers', 'exec'), namespace)
    return namespace


def main(argv=None):
    parser = argparse.ArgumentParser(description='apply (or check) the [QWEN-SDPA-PF] opt-in on attention/tp.py')
    parser.add_argument('tp', type=Path)
    parser.add_argument('--out', type=Path, help='write the patched file here (default: print a summary only)')
    parser.add_argument('--full', action='store_true',
                        help='the whole graft (patch_attention_tp, then the opt-in), as the gate stages it')
    options = parser.parse_args(argv)
    text = options.tp.read_text(encoding='utf-8')
    patched = (patch_tp_full if options.full else patch_tp)(text)
    expected = graft.patch_attention_tp(text) if options.full else text
    if unpatch_tp(patched) != expected:
        print('edits do not invert', file=sys.stderr)
        return 1
    digest = hashlib.sha256(patched.encode('utf-8')).hexdigest()
    base = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if options.out:
        options.out.write_text(patched, encoding='utf-8', newline=NL)
    print('attention/tp.py %s (%s) -> %s %s%s' % (
        base[:16], 'the pinned image file' if base.startswith(TP_PINNED_PREFIX) else
        'NOT the pinned image file %s: anchors matched' % TP_PINNED_PREFIX,
        'the full graft' if options.full else '[QWEN-SDPA-PF] opt-in', digest[:16],
        (' written ' + str(options.out)) if options.out else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
