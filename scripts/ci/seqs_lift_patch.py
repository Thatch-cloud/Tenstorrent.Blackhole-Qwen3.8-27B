"""Lift the max_num_seqs == 1 clause so the fast path will admit a second request.

Authorised 2026-09-19. Run 35432202174 drove two users at the qualified 4096/256
shape with no graft and the server refused at config construction:

    Synchronous single-request single-worker serving required; TT mesh supplies TP2

That is `validate_fast_config`, which the programme's state describes as already
lifted. It is not: `scheduler.max_num_seqs != 1` is still in the condition. The
lift that was recorded must have covered the context clause further down the same
function, and the concurrency clause was never touched.

This lifts ONLY the concurrency term. Everything else in that condition stays:
async_scheduling must still be False, and host-side tensor, pipeline and data
parallel must still be 1, because the TT mesh supplies TP2 itself and a second
host-side axis would be a different thing entirely.

It is a policy change, not a serving default: the probe mounts the patched file
for one run to find what fails NEXT. The four blockers on record - the two
unlifted device pins, the singular session state in serving_lifecycle, and
FastRunnerBridge bound to one request - are assumptions until one of them is
observed firing, and the first thing this session assumed about this file turned
out to be wrong.

The marker carries the admitted value, so the run proves what it let through
rather than only that the line executed.
"""

import io
import sys

MARKER = '[SEQLIFT]'
OLD = '    if (scheduler.max_num_seqs != 1 or scheduler.async_scheduling is not False'
NEW = ('    _qwen_seqs = scheduler.max_num_seqs\n'
       '    _qwen_logger.info("%s admitting max_num_seqs={}", _qwen_seqs)\n'
       '    if (not isinstance(_qwen_seqs, int) or _qwen_seqs < 1\n'
       '            or scheduler.async_scheduling is not False' % MARKER)


def patch_policy(source):
    source = source.replace('\r\n', '\n')
    if MARKER in source:
        raise SystemExit('policy already patched')
    if source.count(OLD) != 1:
        raise SystemExit('expected exactly one max_num_seqs clause, found %d'
                         % source.count(OLD))
    source = source.replace(OLD, NEW, 1)
    if 'from loguru import logger as _qwen_logger' not in source:
        source = 'from loguru import logger as _qwen_logger\n' + source
    return source


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: seqs_lift_patch.py <in> <out>')
    source = io.open(sys.argv[1], encoding='utf-8').read()
    patched = patch_policy(source)
    import ast
    ast.parse(patched)
    io.open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(patched)
    # Grep for the NEW behaviour, and for what must NOT have changed.
    assert '_qwen_seqs < 1' in patched
    assert 'max_num_seqs != 1' not in patched
    assert 'async_scheduling is not False' in patched, 'async gate must survive'
    assert 'tensor_parallel_size != 1' in patched, 'host TP gate must survive'
    assert '%d' not in NEW and '%s' not in NEW.split(MARKER)[1], (
        'loguru formats with braces; a percent placeholder logs literally')
    print('patched %s -> %s (%d -> %d bytes)'
          % (sys.argv[1], sys.argv[2], len(source), len(patched)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
