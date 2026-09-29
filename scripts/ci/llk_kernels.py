"""The C2 hotspot kernels the LLK profiling pass instruments, and how each instrumented copy reaches the
JIT without editing a stock or pinned source in place (docs/llk-profiling-harness.md).

Two delivery routes, the two the model already uses to put its own kernels in front of the JIT:
  generated  the source is text the model builds in-process and hands to generic_op as SOURCE_CODE
             (gdn_seq_block: the served native prefix + gdn_seq_block_*.cpp, qualified by sha256 triple).
             llk_zone_override instruments the qualified text AFTER qualification, in memory, when
             QWEN_LLK_ZONES is set; the compile arg SRC_TAG (the first 32 bits of the source's sha256)
             keys the JIT cache on the instrumented content, so the served binaries are never reused.
  file       the op (compiled into _ttnncpp.so) names a kernel file under /opt/tt-metal. The gate reads
             the image's own bytes in a throwaway container, instruments them on the host and bind-mounts
             each copy read-only over its path, on llk-* arms only; the image is unchanged. Not
             TT_METAL_KERNEL_PATH: the image's WORKDIR is /opt/tt-metal and tt-metal resolves a relative
             kernel path against the working directory first (tt_metal/impl/kernels/kernel.cpp
             resolve_path), so an overlay directory would never be read. A file mount also keeps the
             copy's relative #includes resolving beside it.
  discover   as file, but the path is found in the image by pattern (ops of the PR stack whose kernel
             paths this repo does not pin yet; task L0-01 pins them).
A 'file' entry with a sha256 is instrumented with its stage regions only if the image's bytes match the
pin; otherwise (and for sha256 None) it gets the generic envelope and sync sums, which need no anchors.

Stdlib only, Python 3.7 syntax.
"""
import posixpath
import re

import llk_zones

# Where an instrumented copy may be mounted: kernel sources of ttnn ops, nothing else. Never a model file,
# a graft, a pinned recipe file (tp_common.py, frozen_combined_runtime.py, extent_attention_replay.py) or a
# binary: memory qualification-pins-model-sources.
MOUNTABLE = re.compile(r'ttnn/cpp/ttnn/operations/(?:[A-Za-z0-9_]+/)+kernels/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.(?:cpp|hpp|h)')
PINNED_NAMES = ('tp_common.py', 'frozen_combined_runtime.py', 'extent_attention_replay.py', 'graft.sha256',
                'source.sha256', '_ttnncpp.so', '_ttnn.so')
IMAGE_ROOT = '/opt/tt-metal'
PHASES = ('decode', 'prefill')

# K5-A (gdn_seq_block_compute.cpp): one core runs one (user, head) for a 16-token block. The chain's seven
# steps are five regions (T2+T3 and T4+T5 share one, to stay well inside the marker budget: 16 tokens x 5
# regions x 2 + prologue, epilogue, envelope and the sums = 168 of 250 per TRISC per program).
END_OF_ENTRY = None
K5A_STAGES = (
    ('PRO', '    // ---- prologue, once per block', '    // ---- the chain, token by token ----', 1),
    ('T1', '        // T1: bf16 state -> fp32 [415].', '        if constexpr (SB_DIAG == SB_DIAG_PASSTHROUGH) {', 16),
    ('T23', '        // T2: h = S * exp(g) [466]', "        // T4: D'_j = delta row 0 broadcast down every row [492].", 16),
    ('T45', "        // T4: D'_j = delta row 0 broadcast down every row [492].",
     '        // T6: the bf16 snapshot for the writer', 16),
    ('T6', '        // T6: the bf16 snapshot for the writer', '        // T7: o = q~ @ h_new [506]', 16),
    ('T7', '        // T7: o = q~ @ h_new [506]', '    }\n\n    // ---- epilogue, once per block', 16),
    ('EPI', '    // ---- epilogue, once per block', END_OF_ENTRY, 1),
)
# K64j SDPA decode (sdpa_flash_decode_qwen.cpp): per head a core owns - the K-chunk loop (QK, max, exp,
# sum, PV, the lazy-softmax correction), the tree reduction, and the final normalisation or hand-off.
# The multiplicity is the heads per core, declared at most 8 (decode: 64 q heads over the TP2 pair's grid).
SDPA_DEC_STAGES = (
    ('KLOOP', '        // Loop through all K chunks', '        /* END OF FLASH ATTENTION LOOP */', 8),
    ('TREE', '        // Tree reduction: receive from children and combine', '        // Finalize output based on tree role', 8),
    ('FINAL', '        // Finalize output based on tree role', '    }\n\n    // Free up cb_q_in after Q chunks', 8),
)

KERNELS = (
    dict(key='K5A', route='generated', part='compute', phase='decode', required=True, stages=K5A_STAGES,
         repo='scripts/ci/gdn_seq_block_compute.cpp',
         title='GDN recurrence, K5-A sequential block (gdn_seq_block) compute'),
    dict(key='K5A_RD', route='generated', part='reader', phase='decode', required=False, stages=(),
         repo='scripts/ci/gdn_seq_block_reader.cpp', title='K5-A reader (data movement, NoC 1)'),
    dict(key='K5A_WR', route='generated', part='writer', phase='decode', required=False, stages=(),
         repo='scripts/ci/gdn_seq_block_writer.cpp', title='K5-A writer (data movement, NoC 0)'),
    dict(key='SDPA_DEC', route='file', phase='decode', required=True, stages=SDPA_DEC_STAGES,
         path='ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode_qwen.cpp',
         repo='optimisation/ttnn-op/k64j/kernels/compute/sdpa_flash_decode_qwen.cpp',
         sha256='409a1aafc3ffaaca2c6afba0e999525b7d141491f0a52e5583efa70447ee5c0e',
         title='K64j SDPA decode compute (graft K64j, build_k64j.sh K64J_COMPUTE_QWEN)'),
    dict(key='SDPA_DEC_RD', route='file', phase='decode', required=False, stages=(),
         path='ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_qwen.cpp',
         repo='optimisation/ttnn-op/k64j/kernels/dataflow/reader_decode_qwen.cpp',
         sha256='adb6091878ba3f0a0805846ff56f05352610d7fe779b5ae96320c437c095db49',
         title='K64j SDPA decode reader (K64J_READER_QWEN)'),
    dict(key='SDPA_COMMON', route='file', phase='both', required=False, stages=(), header=True,
         path='ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp', sha256=None,
         title='SDPA compute helpers (QK/PV matmul blocks, reduce, sub_exp): sync sums only, attributed to the '
               'including kernel'),
    dict(key='ATTN_PREP', route='file', phase='decode', required=False, stages=(),
         path='ttnn/cpp/ttnn/operations/transformer/attn_prep/device/kernels/compute/attn_prep.cpp',
         repo='optimisation/ttnn-op/kernels-batch64/attn_prep/device/kernels/compute/attn_prep.cpp',
         sha256='90090093c05df47a68b1bc8860c45c90f58f9adc5571d9d2432355bcde6d4be8',
         title='AttnPrep compute (norm, rope; the pin is the repo copy - the image\'s is unverified)'),
    dict(key='MM', route='file', phase='both', required=False, stages=(), sha256=None,
         path='ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp',
         title='stock matmul compute (decode projections and MLP; every op that uses it)'),
    dict(key='SDPA_PF', route='file', phase='prefill', required=False, stages=(), sha256=None,
         path='ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp',
         title='SDPA prefill compute (graft K64j sdpa tree)'),
    dict(key='CONV_GATES', route='discover', phase='decode', required=False, stages=(),
         patterns=('ttnn/cpp/ttnn/operations/**/*conv_gates*/**/kernels/compute/*.cpp',),
         title='GdnConvGates compute (ttnn.transformer.gdn_decode_conv_gates, PR stack; path unpinned)'),
    dict(key='AGMM', route='discover', phase='prefill', required=False, stages=(),
         patterns=('ttnn/cpp/ttnn/operations/experimental/**/*minimal_matmul*/**/kernels/**/*compute*.cpp',),
         title='prefill gate/up all_gather_minimal_matmul_async compute (C1e keeps the op; path unpinned)'),
)
BY_KEY = dict((entry['key'], entry) for entry in KERNELS)
GENERATED_PARTS = dict((entry['part'], entry) for entry in KERNELS if entry['route'] == 'generated')
DISCOVER_LIMIT = 8   # files per pattern


class KernelError(ValueError):
    """A kernel the pass cannot instrument or mount as asked."""


def in_phase(entry, phase):
    return entry['phase'] in (phase, 'both')


def check_destination(path):
    """A path an instrumented copy may be mounted at (relative to IMAGE_ROOT), or KernelError."""
    normal = posixpath.normpath(path)
    if normal != path or path.startswith('/') or '..' in path.split('/'):
        raise KernelError('%r is not a normalised relative path' % path)
    if posixpath.basename(path) in PINNED_NAMES or not MOUNTABLE.fullmatch(path):
        raise KernelError('%r is not a ttnn op kernel source: only those may be overlaid (MOUNTABLE)' % path)
    return posixpath.join(IMAGE_ROOT, path)


def discovered_key(entry, path):
    """KEY for one discovered file: the entry's key, then the file's stem ([A-Z0-9_])."""
    stem = re.sub(r'[^A-Z0-9_]', '_', posixpath.splitext(posixpath.basename(path))[0].upper())
    return ('%s_%s' % (entry['key'], stem))[:40].rstrip('_')


def stage_regions(entry, text):
    """[(stage, begin, end, multiplicity)]; an end of END_OF_ENTRY (None) is the entry point's closing line."""
    return [tuple(stage) for stage in entry.get('stages') or ()]


def instrument_entry(entry, text, level, key=None, sums_supported=True, budget=llk_zones.MARKER_BUDGET):
    """(instrumented text, record) for one registry entry's source. A pinned entry whose bytes differ from
    its pin (or an unpinned one) gets no stage regions - only the envelope and sums - and says so."""
    key = key or entry['key']
    pinned = entry.get('sha256')
    actual = llk_zones.sha256(text)
    exact = pinned is not None and actual == pinned
    use_stages = bool(entry.get('stages')) and (exact or entry['route'] == 'generated')
    regions = stage_regions(entry, text) if use_stages else ()
    header = bool(entry.get('header'))
    result, record = llk_zones.instrument(text, key, level, regions=regions, envelope=not header,
                                          sync=True, sums_supported=sums_supported, budget=budget, what=key)
    record.update(kernel=entry['key'], route=entry['route'], phase=entry['phase'], header=header,
                  pinned_sha256=pinned, pin_match=None if pinned is None else exact,
                  stages_applied=[stage for stage, _, _, _ in regions],
                  note=None if (exact or not entry.get('stages') or entry['route'] == 'generated') else
                  'image bytes differ from the pin: envelope and sums only (L0-01 re-pins the anchors)')
    return result, record


def instrument_generated(kernels, level, sums_supported=True, log=None):
    """{part: text} of a generated build -> ({part: text}, [record]). A part whose transform is refused keeps
    its served text and its record carries the reason (the arm's coverage then shows it missing)."""
    result, records = dict(kernels), []
    for part, entry in sorted(GENERATED_PARTS.items()):
        if part not in kernels:
            continue
        try:
            result[part], record = instrument_entry(entry, kernels[part], level, sums_supported=sums_supported)
        except llk_zones.ZoneError as error:
            record = dict(kernel=entry['key'], route='generated', level=level, refused=str(error))
            if log is not None:
                log('[LLK] %s not instrumented: %s' % (entry['key'], error))
        records.append(record)
    return result, records


def file_requests(phase):
    """(paths, patterns) the gate reads from the image for one phase's file and discover entries."""
    paths, patterns = [], []
    for entry in KERNELS:
        if not in_phase(entry, phase):
            continue
        if entry['route'] == 'file':
            check_destination(entry['path'])
            paths.append(entry['path'])
        elif entry['route'] == 'discover':
            patterns.extend(entry['patterns'])
    return paths, patterns


def plan_files(phase, image_files, globs, level, sums_supported=True, budget=llk_zones.MARKER_BUDGET):
    """[(path, instrumented text or None, record)] for one phase: every file and discover entry, from the
    image's bytes (image_files: path -> text or None; globs: pattern -> [paths])."""
    planned = []
    for entry in KERNELS:
        if not in_phase(entry, phase) or entry['route'] == 'generated':
            continue
        if entry['route'] == 'file':
            targets = [(entry['path'], entry['key'])]
        else:
            found = []
            for pattern in entry['patterns']:
                found.extend(globs.get(pattern) or [])
            targets = [(path, discovered_key(entry, path)) for path in sorted(set(found))[:DISCOVER_LIMIT]]
            if not targets:
                planned.append((None, None, dict(kernel=entry['key'], route='discover', level=level,
                                                 refused='no file in the image matches %s' % ', '.join(entry['patterns']))))
        for path, key in targets:
            text = image_files.get(path)
            record = dict(kernel=entry['key'], key=key, route=entry['route'], path=path, level=level)
            if text is None:
                record['refused'] = 'not in the image'
                planned.append((path, None, record))
                continue
            try:
                check_destination(path)
                if level == 'tag' and entry.get('header'):
                    raise llk_zones.ZoneError('a header has no envelope: nothing to do at level tag')
                result, detail = instrument_entry(entry, text, level, key=key, sums_supported=sums_supported,
                                                  budget=budget)
                detail.update(path=path, key=key)
                planned.append((path, result, detail))
            except (llk_zones.ZoneError, KernelError) as error:
                record['refused'] = str(error)
                planned.append((path, None, record))
    return planned


def zone_index(records):
    """zone name -> (kernel key, stage or None, kind) over manifest records; the report's key."""
    index = {}
    for record in records:
        for zone in record.get('zones') or ():
            index[zone['name']] = dict(kernel=record.get('key') or record.get('kernel'), stage=zone['stage'],
                                       kind=zone['kind'], multiplicity=zone['multiplicity'],
                                       reconfig_static=zone['reconfig_static'])
    return index
