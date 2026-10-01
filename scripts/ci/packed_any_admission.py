"""S2 C2-packed-any attach-time admission (s2-design.md W7): whether this process may serve packed
rounds at any position through the extent replay path (QWEN_FAST_EXTENT_REPLAY=1), decided once, on the
host, before the attach allocates anything.

WHY. The extent path serves every packed round through one captured K64j program whose extent comes from
a per-bundle cur_pos tensor (flag 0x20). Its exactness rests on evidence gathered outside the serving
process: K64j's card pass (CB1), the kernel sections that decide the exactness policy (CB2a: K2, X7, Z)
and the reader harness (CB2b), each against one binary, four kernels and one reader source. Serving is
exact only where every one of those holds for the bytes this process would run, so the attach checks
each of them and refuses otherwise. There is no fallback here: a refused c2-packed attach fails closed,
and the c2 profile (S1) keeps serving (design section 5, Admission).

WHAT IS CHECKED, under QWEN_FAST_EXTENT_REPLAY=1 only (unset or '0': admit is never called and nothing
here runs; any other value is refused by extent_replay_enabled):
  1. the shape (check_environment): the 64-row M3 block (serving_runtime.m3_shape: four scheduler
     requests, QWEN_FAST_FOUR_AS_TWO=0, QWEN_FAST_PACKED_STEP=1), QWEN_FAST_ANY_REQUEST=1,
     QWEN_FAST_REPLAY_GROUP_ROWS=8 (G8B2, the one geometry CB1 qualified 0x27 at) and
     QWEN_SDPA_TREE_SCRATCH_ROUNDS=1, the pinned reader's G8 precondition (attention_replay.py:23-24;
     design B7), which the image sets but nothing checked before;
  2. the modes: QWEN_FAST_SDPA_MODES is exactly tail, share and slice - the extent reader adds extent
     itself and serves 0x27 (tail | share | slice | extent), the one flag set CB2a and CB2b qualified, so
     a missing mode or readahead (0x2F) is refused here, host only, and not by the reader after the pool,
     the runtime and the weights are built; extent is refused by name (any pinned or pooled reader in the
     process would refuse it, design 1.4 #5);
  3. the binary (check_runtime): QWEN_FAST_RUNTIME_BINARY_SHA256 names K64j's _ttnncpp.so, both paths
     dflash_combined_sim_runtime.BINARIES names hash to it and carry the K64j literals (a graft .so
     replaces the whole binary: memory graft-so-drops-image-patches), the four K64j kernels are their
     recorded bytes, and sdpa_tree_scratch.audit(root, patched=True) holds;
  4. the evidence (check_evidence): packed_any_evidence.json at its pinned sha256, naming that binary and
     those kernels, recording CB1, CB2a and CB2b as PASS with the coverage each must have (evidence_problems:
     the design's seeds, variants, families, starts and geometries, and a count floor per section), and the
     sha256 of every reader source they qualified - which must be the live file's.
After the pool is built, admit_pool requires it to hold the extent storage (extent_replay True: the S2
block keys on that storage alone, so a pool built without it would serve through the per-family block,
packed only in [131072, 131312], and nothing would say so) and its DRAM statistics to be readable (design
B4: the scheduler-side DRAM hold reads them per request). After the packed block is built, admit_blocks
requires it to be the extent block, every segment reader reporting runtime_extent. Each fails the attach
before the lifecycle admits a request.

ONE LINE PER PROBLEM. Every refusal names each failed condition on its own [PINDIAG] line (the server
log's capture cuts lines at about 250 characters), and the exception carries them all.

The result is cached for the process (admitted()). Under the flag the pool refuses to build the extent
storage unless it holds (serving_buffer_pool calls require_admitted), so no serving process builds an
extent path it did not admit. The guard is in the pool - the one source of the storage every extent
reader is built over - and never in extent_attention_replay.py, whose bytes the evidence pins: CB2b
qualifies the reader as every S2 branch holds it.

RECORDING CB2b (done for run 36260236826, v52; a re-qualified reader is recorded the same way). Transcribe
its full-scope run into sections.CB2b, from the 'K64J_READER verdict=PASS
scope=full ...' line and the report JSON of optimisation/ttnn-op/k64j/extent_reader_card_b.py: status PASS,
run, failures 0, scope, chips, capacity, seeds and variants run, r1_geometries (the report's r1_run:
geometry -> words), r2_families (r2_families_replayed), idle_starts (idle_starts_run), counts R1 (the line's
r1 plus r1_reader), S (staging), R2 (r2 plus r2_trace), R4 (r4) and liveness (live), and sources (the
line's extent_sha256, which must be the top-level sources' own). Re-pin EVIDENCE_SHA256 in the same
commit. Nothing else changes: the tests that read the checked-in file follow its CB2b status.

FOUR CARDS (QWEN_FAST_TP=4, the c2-packed-tp4 profiles). The same admission with the four-card numbers and its own
record. K64j at one KV head serves 0x23 (tail | share | extent; the slice needs a second KV head), so the modes are exactly
tail and share and CB1 must hold a G8B2 0x23 combo; the reader is extent_attention_replay_tp.py, whose sha256 CB2b records;
CB2b's harness presents one chip as chip 1of4; CB1 and CB2a must say kv_heads 1 (and CB1 hold no q-slice combo), CB2b's reader
must have served 0x23, so the pair's two-head sections copied in never qualify four cards (_one_head_problems); the record is
packed_any_evidence_tp4.json at its own pin. Until that record
holds a qualifying pass, `admit` refuses - except for a GATE-ONLY profile (QWEN_C2_GATE=1: the contract boots gate_only
profiles only for a gate and never for traffic), where the evidence problems are logged one per line as UNQUALIFIED and the
attach proceeds, so the first four-card rounds can run at all. Every other condition (the shape, the modes, the K64j binary
and kernels, the tree-scratch patch) still refuses. tp_guard is the same rule applied by serving_request_factory's attach
source check, so a process that never reaches admit cannot serve four cards on the pair's evidence either.

Stdlib only at import: the image build runs this module's in-image test (check_runtime on /opt/tt-metal).
"""

import hashlib
import json
import os
from pathlib import Path

import tp_shapes


FLAG = 'QWEN_FAST_EXTENT_REPLAY'
MARKER = '[PINDIAG] packed-any admission'
HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'packed_any_evidence.json'
# The sha256 of packed_any_evidence.json as reviewed. A new record (CB2b's result, a re-qualified reader)
# changes the file, and then this pin, in the same commit: the pin is what makes the file evidence.
EVIDENCE_SHA256 = 'a9407e9a54bfbaa7a0264456644ef6136e376f600ce8f0490417f2b5406515ee'
EVIDENCE_SCHEMA = 'qwen-c2-packed-any-evidence/1'

RUNTIME_BINARY_ENV = 'QWEN_FAST_RUNTIME_BINARY_SHA256'
# K64j (optimisation/ttnn-op/k64j/build_k64j.sh, card-b-v7 run 36222920898): K64i's contents, the decode
# factory with F19-F22 and the four kernels below. K64i's is kept for the provenance succession.
K64J_TTNNCPP_SHA256 = '152951c1c0de5c9dfad2d62c295393a43b2ecf353965c55c709da7e539b975b7'
K64I_TTNNCPP_SHA256 = 'cf54d716669be6b71f1d627e74892c90f562495dc9500589408a72b4ddccf4a4'
# Format literals the served binary must carry (pooled_attention_replay.required_binary_markers for
# tail, share, slice and extent, plus the tree-scratch patch's environment name).
BINARY_LITERALS = (
    b'[QWEN-SDPA] flags=',                     # F4: the [QWEN-SDPA] branch
    b'[QWEN-SDPA] KV-share twin bands',        # F9: stage 3, share (0x2)
    b'[QWEN-SDPA] q-slice rows_per_kv=',       # F18: stage 4, slice (0x4)
    b'[QWEN-SDPA] runtime-extent entries=',    # F22: K64j, extent (0x20)
    b'QWEN_SDPA_TREE_SCRATCH_ROUNDS',          # the decode factory's tree-scratch patch (3e0a69af)
)
KERNEL_ROOT = 'ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels'
# optimisation/ttnn-op/k64j/build_k64j.sh K64J_READER_QWEN, K64J_READER_SLICE, K64J_COMPUTE_QWEN, K64J_WRITER_SLICE.
K64J_KERNELS = {
    'dataflow/reader_decode_qwen.cpp': 'adb6091878ba3f0a0805846ff56f05352610d7fe779b5ae96320c437c095db49',
    'dataflow/reader_decode_qwen_slice.cpp': '518d8096e3cceb160eaef8ab4f0ae976ccbffd3904d31176b7f9d02828c37f8a',
    'compute/sdpa_flash_decode_qwen.cpp': '409a1aafc3ffaaca2c6afba0e999525b7d141491f0a52e5583efa70447ee5c0e',
    'dataflow/writer_decode_qwen_slice.cpp': '642c36f809be0f1ad1664deb405dc310dabaa32d5628710320a118eb37a6cc7a',
}
# (variable, the only value admitted, why).
REQUIRED_ENV = (
    ('QWEN_FAST_ANY_REQUEST', '1', 'the extent path serves the any-request engines (C2-any)'),
    ('QWEN_FAST_REPLAY_GROUP_ROWS', '8', 'CB1 qualified 0x27 at G8B2 only (eight-row groups)'),
    ('QWEN_SDPA_TREE_SCRATCH_ROUNDS', '1', 'the pinned reader\'s G8 precondition (attention_replay.py:23-24)'),
)
SDPA_MODES_ENV = 'QWEN_FAST_SDPA_MODES'
# The modes the environment must name, exactly: with the extent the reader adds, 0x27 - the one flag set
# CB2a (K2, X7, Z) and CB2b ran and the extent reader asserts at G8B2. The image's v235 value.
QUALIFIED_MODES = frozenset(('tail', 'share', 'slice'))
# Four cards: one KV head, so no slice; the extent reader serves 0x23.
QUALIFIED_MODES_TP4 = frozenset(('tail', 'share'))

# The reader sources the evidence qualified, next to this module: the live bytes must be the recorded ones.
QUALIFIED_SOURCES = ('extent_attention_replay.py',)
QUALIFIED_SOURCES_TP4 = ('extent_attention_replay_tp.py',)
# The four-card record, its own pin, and the flag set its CB1 combo must hold.
EVIDENCE_TP4 = HERE / 'packed_any_evidence_tp4.json'
EVIDENCE_TP4_SHA256 = 'd221f68f494e7b1a7571cefdfc28a986458d1ae4511e00a8bf56bfc79aa262cf'
CB1_FLAG = '0x27'
CB1_FLAG_TP4 = '0x23'
# Four cards: the evidence must be the ONE-KV-HEAD runs. K64j is the same binary at both widths and 0x23 is a legal flag set at
# two KV heads too, so without these the pair's CB1 (card B, two heads, G8B2 0x21/0x23/0x27/0x2F) and CB2a (K2 at two heads)
# copied into the four-card record would qualify four cards on the pair's shape. The recorder
# (record_packed_any_evidence_tp4.py) writes kv_heads=1 only from a report that says so, and the reader's served flags.
KV_HEADS_TP4 = 1
SLICE_FLAGS = ('0x27', '0x2f')              # the q-slice needs a second KV head: never in a one-head record
CB2B_SERVED_FLAGS_TP4 = '0x23'
# A gate-only profile boots only with this set (serving_c2_contract.GATE_SWITCH).
GATE_ENV = 'QWEN_C2_GATE'
# Set by the gate-only four-card profile's own env and by no traffic profile: QWEN_C2_GATE=1 is the workflow's switch for EVERY
# gate boot, so on its own it cannot tell a gate-only profile from a traffic profile run as a gate.
GATE_PROFILE_ENV = 'QWEN_C2_GATE_PROFILE'
SECTIONS = ('CB1', 'CB2a', 'CB2b')
SEEDS = (0, 1, 2, 3, 4)
K1_EXTENTS = (2304, 16896, 33024, 65792, 98560, 131328)       # K1's six families (probe_k64j_card_b.EXTENTS)
CB1_COUNTS = ('extent', 'mixed', 'share_slot0', 'trace', 'skip')
CB2A_VARIANTS = ('normal', 'peaky')
CB2A_K2_TICKETS = 1980                                         # k64j_card_b.k2_coverage at seeds 0-4, both variants
CB2_EXTENTS = (2304, 4352, 16640, 65792, 131328)               # K2's and X7's families (k64j_card_b.CB2_EXTENTS)
CB2_STARTS = (0, 7, 127, 240, 255)                             # their s mod 256 (k64j_card_b.CB2_STARTS)
Z_FAMILIES = tuple(range(256, 4096, 256))                      # the 15 families with E / 256 < 16 cores per head
Z_STARTS = (0, 32, 7, 127, 240, 255)                           # the idle starts, then the boundary ones (k64j_card_b)
# The count floors, as k64j_card_b.verdict_line counts them over the design's set (seeds 0-4): X7 compares
# 0x7 narrow and 0x27 each against 0x7 wide, per family, start, seed and variant; Z compares each replay with
# the eager call and with the compile-time 0x7 call, per family, start and seed (normal queries only).
X7_FLOOR = len(CB2_EXTENTS) * len(CB2_STARTS) * len(SEEDS) * len(CB2A_VARIANTS) * 2          # 500
Z_FLOOR = len(Z_FAMILIES) * len(Z_STARTS) * len(SEEDS) * 2                                   # 900
# CB2b (W10b, extent_reader_card_b.py): only its full scope is evidence, and the admission reads the coverage
# itself instead of trusting the word: R1 at three geometries and the design residues, R2 over more than 50
# families including the five named, R4 at both idle starts, seeds 0-2 and both variants, at C = 131,328.
CB2B_SCOPE = 'full'
CB2B_CHIPS = ('1of2',)            # TwoChipView on a one-chip p150a: chip 1 is left to the extent audit and G3/G3b
CB2B_CHIPS_TP4 = ('1of4',)        # the four-card harness's ChipView: one chip of four, the rest left to the extent audit
CB2B_CAPACITY = 131328
CB2B_SEEDS = (0, 1, 2)
CB2B_R1_GEOMETRIES = ('G8B2', 'G4B3', 'G4B1')
CB2B_RESIDUES = (0, 7, 127, 240, 255)
CB2B_R2_MIN_FAMILIES = 51         # design W10b R2: "restaged across more than 50 families"
CB2B_R2_NAMED = (256, 2304, 16640, 65792, 131328)
CB2B_IDLE_STARTS = (0, 32)
# Its sections as the harness names them (R1, S = the construction and restage staging, R2, R4) and its
# liveness controls; from the verdict line: R1 = r1 + r1_reader, S = staging, R2 = r2 + r2_trace, R4 = r4,
# liveness = live.
CB2B_COUNTS = ('R1', 'S', 'R2', 'R4', 'liveness')

_STATE = {}


class AdmissionRefused(ValueError):
    """The c2-packed attach may not proceed. The message names every failed condition; `problems` holds
    them one per entry, which is how the attach logs them."""

    def __init__(self, message, problems=None):
        super().__init__(message)
        self.problems = list(problems) if problems else [message]


def width(environ=None):
    """The tensor-parallel width this process serves at (QWEN_FAST_TP; the pair when unset)."""
    return tp_shapes.requested_tp(os.environ if environ is None else environ)


def qualified_modes(tp):
    return QUALIFIED_MODES if tp == tp_shapes.PAIR else QUALIFIED_MODES_TP4


def qualified_sources(tp):
    return QUALIFIED_SOURCES if tp == tp_shapes.PAIR else QUALIFIED_SOURCES_TP4


def evidence_path(tp):
    return EVIDENCE if tp == tp_shapes.PAIR else EVIDENCE_TP4


def evidence_pin(tp):
    return EVIDENCE_SHA256 if tp == tp_shapes.PAIR else EVIDENCE_TP4_SHA256


def gate_only(environ=None):
    """Whether this process is a gate run of a gate-only profile: the contract's boot condition for such profiles."""
    return (os.environ if environ is None else environ).get(GATE_ENV) == '1'


def _log(template, *values):
    """One server-log line, brace-formatted (dflash_device.pindiag's signature): loguru where the engine
    has it, print otherwise."""
    text = template.format(*values)
    try:
        from loguru import logger
    except ImportError:
        print(text, flush=True)
        return
    logger.info('{}', text)


def _refuse(log, what, problems):
    """Log each problem on its own line, then raise one AdmissionRefused carrying them all."""
    for index, problem in enumerate(problems, 1):
        log('{} refused ({}/{}): {}', MARKER, index, len(problems), problem)
    raise AdmissionRefused('%s=1 refused %s: %s' % (FLAG, what, ' | '.join(problems)), problems)


def extent_replay_enabled(environ=None):
    """QWEN_FAST_EXTENT_REPLAY, strictly: unset or '0' is off, '1' is on, anything else is refused (a
    typo must not silently serve the path the operator did not ask for, nor skip the one they did)."""
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (FLAG, value))
    return value == '1'


def admitted():
    """Whether admit() passed in this process."""
    return 'record' in _STATE


def require_admitted(what, environ=None):
    """For what builds the extent path's storage (serving_buffer_pool under extent_replay): under
    QWEN_FAST_EXTENT_REPLAY=1 nothing is built unless this process's attach admitted it. With the flag
    unset (the card harnesses, the CPU tests) this checks nothing: the flag is what makes a process a
    c2-packed server."""
    if extent_replay_enabled(environ) and not admitted():
        raise AdmissionRefused('%s under %s=1 needs the attach\'s packed-any admission first '
                               '(packed_any_admission.admit, serving_runtime); none passed in this process'
                               % (what, FLAG))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def _hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def check_environment(environ, m3):
    """Refusal strings for the shape and the modes (items 1-2); [] when they hold. `m3` is
    serving_runtime.m3_shape's (met, description)."""
    problems = []
    met, shape = m3
    if not met:
        problems.append('the extent path is the 64-row M3 block\'s (users=4 FOUR_AS_TWO=0 PACKED_STEP=1), not %s'
                        % shape)
    for name, wanted, why in REQUIRED_ENV:
        if environ.get(name) != wanted:
            problems.append('%s=%s, not %s: %s' % (name, environ.get(name, '(unset)'), wanted, why))
    from pooled_attention_replay import sdpa_modes

    tp = width(environ)
    value = environ.get(SDPA_MODES_ENV, '')
    try:
        modes = sdpa_modes(environ)
    except ValueError as error:
        problems.append(str(error))
    else:
        wanted = qualified_modes(tp)
        missing = sorted(wanted - modes)
        served = '0x27 (tail, share, slice and extent)' if tp == tp_shapes.PAIR else '0x23 (tail, share and extent)'
        if missing:
            problems.append('%s=%s lacks %s: the extent reader serves %s only, '
                            'and K64j refuses 0x20 without 0x1' % (SDPA_MODES_ENV, value, ','.join(missing), served))
        if 'extent' in modes:
            problems.append('%s=%s names extent: the extent reader adds it itself, and any pinned or pooled reader '
                            'in the process would refuse it' % (SDPA_MODES_ENV, value))
        others = sorted(modes - wanted - {'extent'})
        if others:
            problems.append('%s=%s names %s: CB2a and CB2b qualified %s only (readahead makes 0x2F, which the '
                            'extent reader refuses after the attach has built everything)'
                            % (SDPA_MODES_ENV, value, ','.join(others), served.split(' ')[0]))
    return problems


def literals_missing(path, literals=BINARY_LITERALS):
    """The literals a binary does not carry (searched in a read-only map, as
    pooled_attention_replay.loaded_binary_has_modes searches the mapped one)."""
    import mmap

    with open(str(path), 'rb') as handle:
        if os.fstat(handle.fileno()).st_size == 0:
            return list(literals)
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
            return [literal for literal in literals if view.find(literal) < 0]


def check_runtime(root, binaries=None):
    """Item 3 without the environment: the K64j binary at every path BINARIES names, carrying
    BINARY_LITERALS; the four K64j kernels; the audited tree-scratch sources. `binaries`, when given, is
    {path: sha256} already read this process (runtime_binary_override.install's record), so the attach
    does not hash the binaries twice. Returns a record; raises AdmissionRefused with every problem, each
    naming its file relative to `root` (one short line each)."""
    from dflash_combined_sim_runtime import BINARIES

    root = Path(root)
    problems, record = [], dict(binaries={}, kernels={}, literals=[literal.decode() for literal in BINARY_LITERALS])
    for name in BINARIES:
        path = root / name
        if not path.is_file():
            problems.append('%s is missing' % name)
            continue
        digest = (binaries or {}).get(name) or sha256_file(path)
        record['binaries'][name] = digest
        if digest != K64J_TTNNCPP_SHA256:
            problems.append('%s is %s, not K64j\'s %s' % (name, digest[:16], K64J_TTNNCPP_SHA256[:16]))
        missing = literals_missing(path)
        if missing:
            problems.append('%s lacks %s' % (name, ', '.join(repr(literal.decode()) for literal in missing)))
    for name, wanted in sorted(K64J_KERNELS.items()):
        path = root / KERNEL_ROOT / name
        digest = sha256_file(path) if path.is_file() else None
        record['kernels'][name] = digest
        if digest != wanted:
            problems.append('kernel %s is %s, not the K64j kernel %s' % (name, (digest or 'absent')[:16], wanted[:16]))
    from sdpa_tree_scratch import audit

    try:
        record['tree_scratch'] = audit(root, patched=True)
    except OSError as error:
        problems.append('sdpa_tree_scratch.audit(patched=True): %s %s' % (
            type(error).__name__, Path(error.filename).name if getattr(error, 'filename', None) else error))
    except ValueError as error:
        problems.append('sdpa_tree_scratch.audit(patched=True): %s' % error)
    if problems:
        raise AdmissionRefused('the runtime under %s is not K64j as qualified: %s' % (root.as_posix(), '; '.join(problems)),
                               ['runtime: %s' % problem for problem in problems])
    return record


def _count(value):
    """(passed, total) from a [passed, total] pair, or None."""
    if (isinstance(value, list) and len(value) == 2 and all(type(item) is int for item in value)
            and 0 <= value[0] <= value[1]):
        return tuple(value)
    return None


def _full(value, floor=1):
    """A full pass of at least `floor` comparisons."""
    counted = _count(value)
    return counted is not None and counted[1] >= floor and counted[0] == counted[1]


def _missing(found, wanted):
    """What of `wanted` a recorded list lacks (all of it when the record is not a list)."""
    found = set(found) if isinstance(found, list) else set()
    return [item for item in wanted if item not in found]


def _passed(section, name, problems):
    if not isinstance(section, dict):
        problems.append('%s: the section is missing' % name)
        return False
    if section.get('status') != 'PASS':
        problems.append('%s: status %s, not PASS' % (name, section.get('status')))
        return False
    if type(section.get('run')) is not int or section['run'] <= 0:
        problems.append('%s: no run id' % name)
    if section.get('failures') != 0:
        problems.append('%s: failures %r, not 0' % (name, section.get('failures')))
    return True


def _cover(problems, section, key, wanted, what):
    missing = _missing(section.get(key), wanted)
    if missing:
        problems.append('%s: %s lack %s' % (what, key, missing))


def _cb1_problems(cb1, problems, flag=CB1_FLAG):
    _cover(problems, cb1, 'seeds', SEEDS, 'CB1')
    missing = _missing(cb1.get('extents'), K1_EXTENTS)
    if missing:
        problems.append('CB1: extents lack K1\'s %s' % missing)
    combos = cb1.get('combos') or []
    if not any(isinstance(combo, dict) and combo.get('geometry') == 'G8B2' and flag in (combo.get('flags') or [])
               for combo in combos):
        problems.append('CB1: no G8B2 %s combo, the one the extent reader serves' % flag)
    counts = cb1.get('counts') or {}
    for key in CB1_COUNTS:
        if not _full(counts.get(key)):
            problems.append('CB1: %s %s is not a full pass' % (key, counts.get(key)))


def _cb2a_problems(cb2a, problems):
    _cover(problems, cb2a, 'seeds', SEEDS, 'CB2a')
    _cover(problems, cb2a, 'variants', CB2A_VARIANTS, 'CB2a')
    k2 = cb2a.get('k2') or {}
    if k2.get('verdict') != 'PASS':
        problems.append('CB2a: K2 verdict %s, not PASS (a failed or coverage-reduced K2 leaves the exactness '
                        'policy to the user, design 6.1 D-c)' % k2.get('verdict'))
    if not _full(k2.get('tickets'), CB2A_K2_TICKETS):
        problems.append('CB2a: K2 tickets %s, not a full pass of the %d the design asks' % (k2.get('tickets'),
                                                                                         CB2A_K2_TICKETS))
    if not _full(k2.get('rows')):
        problems.append('CB2a: K2 rows %s are not all bitwise equal' % k2.get('rows'))
    if not _full(cb2a.get('x7'), X7_FLOOR):
        problems.append('CB2a: X7 %s, not a full pass of the %d the design asks' % (cb2a.get('x7'), X7_FLOOR))
    z = cb2a.get('z') or {}
    if not _full(z.get('passed'), Z_FLOOR):
        problems.append('CB2a: Z %s, not a full pass of the %d the design asks' % (z.get('passed'), Z_FLOOR))
    missing = _missing(z.get('families'), Z_FAMILIES)
    if missing:
        problems.append('CB2a: Z families lack %s of the 15 below 4096' % missing)


def _cb2b_problems(cb2b, sources, problems, chips=CB2B_CHIPS, source_names=QUALIFIED_SOURCES):
    if cb2b.get('scope') != CB2B_SCOPE:
        problems.append('CB2b: scope %s, not full (a reduced run - the watcher pass, one seed - is never CB2b\'s '
                        'evidence)' % cb2b.get('scope'))
    if cb2b.get('chips') not in chips:
        problems.append('CB2b: chips %s, not the harness\'s %s' % (cb2b.get('chips'), '/'.join(chips)))
    if cb2b.get('capacity') != CB2B_CAPACITY:
        problems.append('CB2b: capacity %s, not the served C = %d' % (cb2b.get('capacity'), CB2B_CAPACITY))
    _cover(problems, cb2b, 'seeds', CB2B_SEEDS, 'CB2b')
    _cover(problems, cb2b, 'variants', CB2A_VARIANTS, 'CB2b')
    geometries = cb2b.get('r1_geometries') if isinstance(cb2b.get('r1_geometries'), dict) else {}
    missing = [name for name in CB2B_R1_GEOMETRIES if name not in geometries]
    if missing:
        problems.append('CB2b: R1 ran no %s (the design asks G8B2, G4B3 and G4B1)' % ','.join(missing))
    for name in CB2B_R1_GEOMETRIES:
        short = _missing(geometries.get(name), CB2B_RESIDUES) if name in geometries else []
        if short:
            problems.append('CB2b: R1 %s lacks words %s' % (name, short))
    families = cb2b.get('r2_families')
    if (not isinstance(families, list) or any(type(family) is not int or family % 256 or not 256 <= family <= CB2B_CAPACITY
                                              for family in families) or len(set(families)) != len(families)):
        problems.append('CB2b: r2_families is not a list of distinct 256-key families up to %d' % CB2B_CAPACITY)
    else:
        if len(families) < CB2B_R2_MIN_FAMILIES:
            problems.append('CB2b: R2 replayed %d families, not more than 50' % len(families))
        named = _missing(families, CB2B_R2_NAMED)
        if named:
            problems.append('CB2b: R2 families lack the named %s' % named)
    _cover(problems, cb2b, 'idle_starts', CB2B_IDLE_STARTS, 'CB2b')
    counts = cb2b.get('counts') or {}
    for key in CB2B_COUNTS:
        if not _full(counts.get(key)):
            problems.append('CB2b: %s %s is not a full pass' % (key, counts.get(key)))
    ran = cb2b.get('sources') or {}
    for name in source_names:
        if ran.get(name) != sources.get(name):
            problems.append('CB2b: ran %s at %s, not the qualified %s' % (name, str(ran.get(name))[:16],
                                                                         str(sources.get(name))[:16]))


def _one_head_problems(sections, problems):
    """Four cards only: CB1 and CB2a ran at one KV head per chip (kv_heads 1, no q-slice combo) and CB2b's reader served
    0x23. A section that is not a PASS is already refused by _passed."""
    passed = {name: section for name, section in sections.items()
              if name in SECTIONS and isinstance(section, dict) and section.get('status') == 'PASS'}
    for name in ('CB1', 'CB2a'):
        heads = passed[name].get('kv_heads') if name in passed else KV_HEADS_TP4
        if type(heads) is not int or heads != KV_HEADS_TP4:
            problems.append('%s: kv_heads %r, not %d: a two-head (pair) run is not four-card evidence'
                            % (name, heads, KV_HEADS_TP4))
    if 'CB1' in passed:
        held = sorted({str(flag).lower() for combo in passed['CB1'].get('combos') or [] if isinstance(combo, dict)
                       for flag in combo.get('flags') or []} & set(SLICE_FLAGS))
        if held:
            problems.append('CB1: combos hold %s: the q-slice needs a second KV head, so this is not a one-head run'
                            % ','.join(held))
    if 'CB2b' in passed:
        served = passed['CB2b'].get('served') if isinstance(passed['CB2b'].get('served'), dict) else {}
        if served.get('flags') != CB2B_SERVED_FLAGS_TP4:
            problems.append('CB2b: the reader served %s, not %s (one KV head)' % (served.get('flags'), CB2B_SERVED_FLAGS_TP4))


def evidence_problems(evidence, sources_root=HERE, tp=None):
    """Every reason the evidence does not qualify the extent path for these bytes; [] when it does. `tp` is the
    width the record is for (default: the width this process serves at); four cards read the reader twin's sha256,
    a G8B2 0x23 CB1 combo, CB2b's 1of4 chip view, and one KV head in CB1, CB2a and CB2b (_one_head_problems)."""
    tp = width() if tp is None else tp
    problems = []
    if not isinstance(evidence, dict) or evidence.get('schema') != EVIDENCE_SCHEMA:
        return ['the evidence is not a %s record' % EVIDENCE_SCHEMA]
    binary = evidence.get('binary') or {}
    if binary.get('ttnncpp_sha256') != K64J_TTNNCPP_SHA256:
        problems.append('binary: the evidence qualified %s, not K64j %s' % (str(binary.get('ttnncpp_sha256'))[:16],
                                                                           K64J_TTNNCPP_SHA256[:16]))
    if evidence.get('kernels') != K64J_KERNELS:
        problems.append('kernels: the evidence names other kernel bytes than K64j\'s four')
    sources = evidence.get('sources') or {}
    for name in qualified_sources(tp):
        recorded = sources.get(name)
        path = Path(sources_root) / name
        live = sha256_file(path) if path.is_file() else None
        if not _hex64(recorded):
            problems.append('sources: no sha256 recorded for %s' % name)
        elif recorded != live:
            problems.append('sources: %s is %s, but the evidence qualified %s (re-run CB2b on these bytes)'
                            % (name, (live or 'absent')[:16], recorded[:16]))
    sections = evidence.get('sections') or {}
    for name in sorted(set(sections) - set(SECTIONS)):
        problems.append('%s: not a section this admission knows' % name)
    if _passed(sections.get('CB1'), 'CB1', problems):
        _cb1_problems(sections['CB1'], problems, CB1_FLAG if tp == tp_shapes.PAIR else CB1_FLAG_TP4)
    if _passed(sections.get('CB2a'), 'CB2a', problems):
        _cb2a_problems(sections['CB2a'], problems)
    if _passed(sections.get('CB2b'), 'CB2b', problems):
        _cb2b_problems(sections['CB2b'], sources, problems,
                       CB2B_CHIPS if tp == tp_shapes.PAIR else CB2B_CHIPS_TP4, qualified_sources(tp))
    if tp != tp_shapes.PAIR:
        _one_head_problems(sections, problems)
    return problems


def check_evidence(path=None, *, expected_sha256=None, sources_root=HERE, tp=None):
    """Item 4: the evidence file at its pinned sha256, and evidence_problems empty. Returns the parsed
    record; raises AdmissionRefused naming every problem (one entry each in its `problems`). The file and its pin
    are the pair's or the four-card record's by `tp` (default: the width this process serves at)."""
    tp = width() if tp is None else tp
    expected_sha256 = evidence_pin(tp) if expected_sha256 is None else expected_sha256
    path = Path(evidence_path(tp) if path is None else path)
    if not path.is_file():
        raise AdmissionRefused('the extent path has no evidence: %s is missing' % path,
                               ['evidence: %s is missing' % path.name])
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise AdmissionRefused('%s is %s, not the reviewed %s' % (path.name, digest[:16], expected_sha256[:16]),
                               ['evidence: %s is %s, not the reviewed %s' % (path.name, digest[:16], expected_sha256[:16])])
    try:
        evidence = json.loads(payload.decode('utf-8'))
    except ValueError as error:
        raise AdmissionRefused('%s does not parse: %s' % (path.name, error),
                               ['evidence: %s does not parse: %s' % (path.name, str(error)[:120])])
    problems = evidence_problems(evidence, sources_root, tp)
    if problems:
        raise AdmissionRefused('the evidence does not qualify the extent path: ' + '; '.join(problems),
                               ['evidence: %s' % problem for problem in problems])
    return evidence


UNQUALIFIED_MARKER = '[PINDIAG] packed-any admission UNQUALIFIED (gate only, four cards)'


def unqualified_allowed(environ=None):
    """Whether missing four-card evidence may be waved through: only at four cards, only in a gate run of a
    gate-only profile (the contract never boots one for traffic; the profile itself sets QWEN_C2_GATE_PROFILE=1)."""
    environ = os.environ if environ is None else environ
    return (width(environ) != tp_shapes.PAIR and gate_only(environ)
            and environ.get(GATE_PROFILE_ENV) == '1')


def _log_unqualified(log, problems):
    for index, problem in enumerate(problems, 1):
        log('{} ({}/{}): {}', UNQUALIFIED_MARKER, index, len(problems), problem)


def tp_guard(environ=None, *, log=None, sources_root=HERE):
    """The four-card rule for a process that has not reached admit (serving_request_factory.attach_source_check):
    at the pair nothing (returns None); at four cards the four-card record must qualify, or the process must be a
    gate run, in which case the problems are logged as UNQUALIFIED and returned. Otherwise AdmissionRefused."""
    environ = os.environ if environ is None else environ
    tp = width(environ)
    if tp == tp_shapes.PAIR:
        return None
    log = _log if log is None else log
    try:
        check_evidence(tp=tp, sources_root=sources_root)
    except AdmissionRefused as refusal:
        if not unqualified_allowed(environ):
            raise AdmissionRefused('QWEN_FAST_TP=%d serves on the pair\'s evidence otherwise, and %s is not a qualifying '
                                   'four-card record (%s): %s' % (tp, EVIDENCE_TP4.name, GATE_ENV + ' is not 1',
                                                                 '; '.join(refusal.problems)), refusal.problems)
        _log_unqualified(log, refusal.problems)
        return list(refusal.problems)
    return []


def admit(runtime_root, *, m3, binary_record=None, environ=None, log=None, evidence=None):
    """Items 1-4, once per process: the record on success (cached; a second call returns it), else
    AdmissionRefused naming every failed condition, each logged on its own line before it is raised.
    `binary_record` is runtime_binary_override.install's return value (None when it admitted nothing)."""
    if 'record' in _STATE:
        return _STATE['record']
    environ = os.environ if environ is None else environ
    log = _log if log is None else log
    problems = []
    if not extent_replay_enabled(environ):
        problems.append('%s is not 1' % FLAG)
    problems.extend(check_environment(environ, m3))
    requested = (environ.get(RUNTIME_BINARY_ENV) or '').lower()
    if requested != K64J_TTNNCPP_SHA256:
        problems.append('%s=%s, not K64j\'s %s' % (RUNTIME_BINARY_ENV, requested[:16] or '(unset)',
                                                  K64J_TTNNCPP_SHA256[:16]))
    if binary_record is not None and binary_record.get('override') != K64J_TTNNCPP_SHA256:
        problems.append('the runtime binary override admitted %s, not K64j' % str(binary_record.get('override'))[:16])
    tp = width(environ)
    record = dict(flag=FLAG, shape=m3[1])
    unqualified = []
    for name, check in (('runtime', lambda: check_runtime(runtime_root, (binary_record or {}).get('binaries'))),
                        ('evidence', lambda: check_evidence(evidence) if tp == tp_shapes.PAIR
                         else check_evidence(evidence, tp=tp))):
        try:
            record[name] = check()
        except AdmissionRefused as refusal:
            if name == 'evidence' and unqualified_allowed(environ):
                unqualified.extend(refusal.problems)
                record[name] = None
            else:
                problems.extend(refusal.problems)
    if problems:
        _refuse(log, 'at attach', problems)
    if unqualified:
        # A gate run at four cards, before any four-card evidence exists: every missing piece is on the record, one
        # line each, and the process may never take traffic (its profile is gate_only).
        record['unqualified'] = unqualified
        _log_unqualified(log, unqualified)
        log('{} passed UNQUALIFIED: K64j {} x{}; kernels {}; {} evidence problems (gate only)',
            MARKER, K64J_TTNNCPP_SHA256[:16], len(record['runtime']['binaries']),
            ','.join(sha[:8] for _, sha in sorted(K64J_KERNELS.items())), len(unqualified))
        _STATE['record'] = record
        return record
    sections = record['evidence']['sections']
    log('{} passed: K64j {} x{}; kernels {}; evidence {}; CB1 {} CB2a {} CB2b {}; reader {}',
        MARKER, K64J_TTNNCPP_SHA256[:16], len(record['runtime']['binaries']),
        ','.join(sha[:8] for _, sha in sorted(K64J_KERNELS.items())), evidence_pin(tp)[:16],
        sections['CB1']['run'], sections['CB2a']['run'], sections['CB2b']['run'],
        ','.join(record['evidence']['sources'][name][:16] for name in qualified_sources(tp)))
    _STATE['record'] = record
    return record


def admit_statistics(pool, *, log=None):
    """Design B4: the pool's DRAM statistics (serving_buffer_pool.dram_statistics, the figures the DRAM
    admission hold and the coordinator read) must be readable once the pool exists, with a largest free
    block per chip. Refuses the attach otherwise; returns the statistics."""
    log = _log if log is None else log
    statistics = pool.dram_statistics()
    if isinstance(statistics, dict):
        reason = str(statistics.get('unavailable', 'no statistics'))[:160]
    elif (not isinstance(statistics, list) or not statistics
          or any(not isinstance(chip, dict) or type(chip.get('largest_free')) is not int or chip['largest_free'] < 0
                 for chip in statistics)):
        reason = 'no per-chip largest free block in %s' % (repr(statistics)[:120],)
    else:
        log('{} DRAM statistics readable: largest_free={}', MARKER,
            ','.join('%.1fMB' % (chip['largest_free'] / 1e6) for chip in statistics))
        return statistics
    log('{} refused: DRAM statistics unavailable ({})', MARKER, reason)
    raise AdmissionRefused('%s=1 needs readable DRAM statistics (the DRAM admission hold reads them per request): %s'
                           % (FLAG, reason))


def admit_pool(pool, *, log=None):
    """Right after the pool (design W2, B4): it holds the extent storage - extent_replay exactly True, the
    storage the S2 block keys on (without it the block would be the per-family one, packed only in
    [131072, 131312], every other round sequential, and nothing at attach would say so) - and its DRAM
    statistics are readable (admit_statistics). Refuses the attach otherwise; returns the statistics."""
    log = _log if log is None else log
    if getattr(pool, 'extent_replay', None) is not True:
        _refuse(log, 'after the pool', ['the pool holds no extent storage (extent_replay %r): the block would '
                                        'serve the per-family path, not the admitted one'
                                        % (getattr(pool, 'extent_replay', None),)])
    return admit_statistics(pool, log=log)


def admit_blocks(blocks, *, log=None):
    """After the packed blocks are built (design W3): each is the extent block (PackedVerifierEngine.extent
    True) and every segment reader of its fixture's replay reader reports runtime_extent - the executed
    path is the one admitted (memory graft-mounted-is-not-graft-executed). Refuses the attach otherwise,
    before the lifecycle admits a request; returns the segment count per block."""
    log = _log if log is None else log
    blocks, problems, segments = list(blocks), [], []
    if not blocks:
        problems.append('no packed block was built: the extent path serves its rounds through the block')
    for index, block in enumerate(blocks):
        if getattr(block, 'extent', None) is not True:
            problems.append('block %d is not the extent block (extent %r)' % (index, getattr(block, 'extent', None)))
        replay = getattr(getattr(block, 'fixture', None), 'replay_reader', None)
        readers = list(getattr(replay, 'readers', None) or ())
        segments.append(len(readers))
        if not readers:
            problems.append('block %d has no segment readers to check (fixture.replay_reader.readers)' % index)
        wrong = [segment for segment, reader in enumerate(readers) if getattr(reader, 'runtime_extent', None) is not True]
        if wrong:
            problems.append('block %d segments %s do not report runtime_extent: not the extent readers' % (index, wrong))
    if problems:
        _refuse(log, 'after the block', problems)
    log('{} extent block engaged: blocks={} segments={}', MARKER, len(blocks), ','.join(map(str, segments)))
    return segments
