"""Env-gated LLK zone instrumentation of the K5-A (gdn_seq_block) kernels, for the gate's llk-* arms only.

QWEN_LLK_ZONES (default unset): 'tag' or 'stages' (llk_zones.LEVELS). Unset - which it is on every served
profile and every non-LLK arm - serving_runtime does not even import this module, and nothing changes.

Set, install() wraps gdn_seq_block.served_kernels once: the qualified build is generated and checked against
QUALIFIED exactly as before, then its compute, reader and writer texts are instrumented in memory
(llk_kernels.instrument_generated: remove() of each copy is byte-identical to the qualified text, or the part
keeps its served text). The Build that comes back carries the instrumented texts; its tag(role), the SRC_TAG
compile arg, is the sha256 prefix of the instrumented text, so the JIT compiles the copies under their own
cache keys and never reuses - or overwrites - a served binary. No recipe file is edited: gdn_seq_block.py,
its pins and the pinned native prefix stay as they are.

Refused (ValueError, the attach fails) unless TT_METAL_DEVICE_PROFILER=1 is set too: without the profiler the
zones compile to nothing, so QWEN_LLK_ZONES alone could only mean a misconfigured serving container - and a
profiled container is never a served one (the gate's llk-* arms are the only place both are set).
"""

import json

ENV = 'QWEN_LLK_ZONES'
SUMS_ENV = 'TT_METAL_PROFILER_SUM'
LEVELS = ('tag', 'stages')
LOG_PREFIX = '[LLK]'


def requested(environ=None):
    """The level asked for, None when unset or empty; any other value is refused."""
    if environ is None:
        import os

        environ = os.environ
    value = environ.get(ENV)
    if not value:
        return None
    if value not in LEVELS:
        raise ValueError('%s must be one of %s, got %r' % (ENV, ', '.join(LEVELS), value))
    return value


def install(*, log, environ=None, module=None):
    """Wrap `module`.served_kernels (gdn_seq_block's) so it returns the instrumented build, or return None when
    QWEN_LLK_ZONES is unset. Idempotent. Returns {env, level, wrapped}. `log` is brace-formatted (pindiag): every
    line goes through it as one '{}' value, so a brace in a kernel anchor never reaches a format string."""
    def say(text):
        log('{}', text)

    if environ is None:
        import os

        environ = os.environ
    level = requested(environ)
    if level is None:
        return None
    if environ.get('TT_METAL_DEVICE_PROFILER') != '1':
        raise ValueError('%s=%s without TT_METAL_DEVICE_PROFILER=1: LLK zones are a profiled gate arm\'s, never a '
                         'serving container\'s' % (ENV, level))
    if module is None:
        import gdn_seq_block as module
    import llk_kernels
    sums = environ.get(SUMS_ENV) == '1'
    if not hasattr(module, '_llk_served_kernels'):
        original = module.served_kernels
        cache = {}

        def served_kernels_llk(root=None, environ=None):
            build = original(root, environ)
            key = id(build)
            if key not in cache:
                texts, records = llk_kernels.instrument_generated(dict(build), level, sums_supported=sums, log=say)
                instrumented = module.Build(texts, build.level, build.variant, build.diag, build.qualified)
                instrumented.llk_level = level
                instrumented.llk_records = records
                instrumented.served = build
                cache[key] = instrumented
                for record in records:
                    # The gate reads these back (llk_profile_plan.generated_records): the K5-A zone names, their
                    # static reconfiguration counts, and the proof that the override ran in the worker.
                    say('%s record %s' % (LOG_PREFIX, json.dumps(record, sort_keys=True)))
                    if 'refused' in record:
                        continue
                    say('%s %s instrumented at level %s: source %s -> %s, zones %s, sums %s (in %d, out %d), %d '
                        'markers per RISC per program' % (
                            LOG_PREFIX, record['key'], level, record['source_sha256'][:12],
                            record['instrumented_sha256'][:12], ','.join(zone['name'] for zone in record['zones']),
                            'on' if record['sync']['enabled'] else 'off', record['sync']['wait_in'],
                            record['sync']['wait_out'], record['markers']['per_risc_per_program']))
            return cache[key]

        module._llk_served_kernels = original
        module.served_kernels = served_kernels_llk
    say('%s zones installed on gdn_seq_block (level %s, sums %s): K5-A builds from here are instrumented copies, '
        'profiled attribution only' % (LOG_PREFIX, level, 'on' if sums else 'off'))
    return dict(env=ENV, level=level, wrapped=True, sums=sums)
