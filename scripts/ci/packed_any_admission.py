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
     requests, QWEN_FAST_FOUR_AS_TWO=0, QWEN_FAST_PACKED_STEP=1; or, under QWEN_FAST_M3_BLOCKS=2, two of
     them for eight requests - each block is that same block, so the evidence's geometry is unchanged and the
     record and the passed line name blocks=2), QWEN_FAST_ANY_REQUEST=1,
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
     recorded bytes, and sdpa_tree_scratch.audit(root, patched=True) holds. The binary may also be K64j-OQ (K64J_OQ_TTNNCPP_SHA256: K64j plus
     the prefill SDPA's oneq edits, nothing else), named by the same variable and carrying its two extra literals; every record below was made
     on K64j, so a K64j-OQ boot borrows K64j's record in a GATE boot only (equivalent) and a traffic boot needs a record of its own;
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

A 262,144-TOKEN WINDOW (QWEN_FAST_MAX_POSITION=262144, the c2-packed-tp4-8x262k profiles; page-table width 4,096). The same
admission at the second capacity, behind its OWN evidence and never the 131,328 record: packed_any_evidence_tp4_262144.json at its own
pin (EVIDENCE_TP4_262K_SHA256), whose CB1 (K1 plus the 196,864 and 262,144 families), CB2a (K2 2,030 tickets, X7 600, Z 900, K2/X7
over the 131k families plus the full window) and CB2b (R2 over the named families plus 262,144, capacity 262,144) are the design set
at that capacity (capacity_design), AND ordered_writer_evidence_tp4.json (E1: page_width_tp4 admits width 4,096). Either record
failing refuses the attach, in a gate run too: the gate-only UNQUALIFIED waiver is the 131,328 record's alone and is never applied
at 262,144 (a 262k gate arm exercises the machinery, it never runs on the 131k evidence). The ONE exception is explicit and named:
QWEN_FAST_262K_EVIDENCE_WAIVER=1 (page_width_tp4.waiver_active), honoured only in a gate run of a gate-only profile (QWEN_C2_GATE=1 and
QWEN_C2_GATE_PROFILE=1), refused by name anywhere else, logged once as '262k evidence WAIVED (gate-only): ...' and followed by
'passed UNQUALIFIED ... (262k waiver, gate only)': every result of such a run is unqualified. The capacity is QWEN_FAST_MAX_POSITION
rounded up to a whole 64-key page (unset: 131,328, exactly today's path and today's passed line); any other capacity is refused by
name. admit_pool then holds the pool to the admitted capacity (pool.page_width * 64).

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
# K64j-OQ (optimisation/ttnn-op/sdpa_prefill_oneq/build_k64j_oq.sh, card-M runs W1 and Q1 of tp4/prefill-sdpa): K64j's contents byte for byte
# (the decode factory, its four kernels, the chain reader, _ttnn.so and every op directory) with the PREFILL SDPA factory's oneq edits, flag 0x8 of the per-call
# chain word (one q chunk per core). The build proves the delta: K64j alone rebuilt in the same ttbuild, every QWEN string of K64j kept, only the two oneq literals
# new, the same exported symbols, one manifest line. The edits are inert unless a call carries the 0x8 bit (QWEN_FAST_SDPA_PF_ONEQ=1).
K64J_OQ_TTNNCPP_SHA256 = '2b81e28f017ccf0ab50028fbae5eb31dd61cfd8a3159233a1ed785712d024a57'
# The binaries this admission serves on, with the name the logs use. Every evidence record below was recorded on K64j, and a record names ONE binary
# (binary.ttnncpp_sha256): see equivalent() for what that means for K64j-OQ.
SERVED_BINARIES = {K64J_TTNNCPP_SHA256: 'K64j', K64J_OQ_TTNNCPP_SHA256: 'K64j-OQ'}
ONEQ_BINARY_LITERALS = (b'[QWEN-SDPA-PF] oneq needs one q chunk per core', b'[QWEN-SDPA-PF] oneq=1 q_chunks=')
# served binary -> the binary whose evidence it may borrow, and only in a GATE boot (QWEN_C2_GATE=1: never traffic). K64j-OQ differs from K64j in the prefill
# factory alone, which no decode-extent section (CB1, CB2a, CB2b) exercises; a traffic boot still needs a record of its own binary
# (record_packed_any_evidence_tp4.py --binary k64j-oq, then the pins re-set), the requalification list in optimisation/ttnn-op/sdpa_prefill_oneq/README.md.
EVIDENCE_BORROWS = {K64J_OQ_TTNNCPP_SHA256: K64J_TTNNCPP_SHA256}
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

# THE SECOND CAPACITY. The design set by served capacity (capacity_design): the 131,328 entry is exactly the constants above.
CAPACITY_131K = CB2B_CAPACITY
CAPACITY_262K = 262144
CAPACITIES = (CAPACITY_131K, CAPACITY_262K)
CAPACITY_ENV = 'QWEN_FAST_MAX_POSITION'
K2_SWEEP_TICKETS = 173                                          # k64j_card_b.K2_SWEEP (128, 300): every start of families 256 and 512
K1_EXTENTS_262K = K1_EXTENTS + (196864, 262144)
CB2_EXTENTS_262K = CB2_EXTENTS + (CAPACITY_262K,)
CB2B_R2_NAMED_262K = CB2B_R2_NAMED + (CAPACITY_262K,)
CB2A_K2_TICKETS_262K = len(SEEDS) * len(CB2A_VARIANTS) * (K2_SWEEP_TICKETS + len(CB2_EXTENTS_262K) * len(CB2_STARTS))      # 2,030
X7_FLOOR_262K = len(CB2_EXTENTS_262K) * len(CB2_STARTS) * len(SEEDS) * len(CB2A_VARIANTS) * 2                          # 600
# The 262,144 record and its own pin, a skeleton until E2 records its sections (a name the 131k recorder's repin never matches).
EVIDENCE_TP4_262K = HERE / 'packed_any_evidence_tp4_262144.json'
EVIDENCE_TP4_262K_SHA256 = '422bfbaa03d1615b1a42d22ca566138b2c1929fc2ed917ba8adc7cd58d68a3d8'

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


def evidence_path(tp, capacity=CAPACITY_131K):
    if capacity == CAPACITY_262K:
        return EVIDENCE_TP4_262K
    return EVIDENCE if tp == tp_shapes.PAIR else EVIDENCE_TP4


def evidence_pin(tp, capacity=CAPACITY_131K):
    if capacity == CAPACITY_262K:
        return EVIDENCE_TP4_262K_SHA256
    return EVIDENCE_SHA256 if tp == tp_shapes.PAIR else EVIDENCE_TP4_SHA256


def gate_only(environ=None):
    """Whether this process is a gate run of a gate-only profile: the contract's boot condition for such profiles."""
    return (os.environ if environ is None else environ).get(GATE_ENV) == '1'


def served_binary(environ=None):
    """The binary this process serves on: QWEN_FAST_RUNTIME_BINARY_SHA256 when it names one of SERVED_BINARIES, else K64j (the default every record and
    test below was written for; admit refuses any other value by name)."""
    requested = ((os.environ if environ is None else environ).get(RUNTIME_BINARY_ENV) or '').lower()
    return requested if requested in SERVED_BINARIES else K64J_TTNNCPP_SHA256


def binary_label(sha):
    return SERVED_BINARIES.get(sha, 'K64j')


def equivalent(recorded, served, environ=None):
    """Whether a record made on binary `recorded` may stand for binary `served` here: only the reviewed pair of EVIDENCE_BORROWS and only in a gate boot."""
    return recorded != served and EVIDENCE_BORROWS.get(served) == recorded and gate_only(environ)


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


M3_BLOCKS_ENV = 'QWEN_FAST_M3_BLOCKS'


def m3_blocks(environ=None):
    """QWEN_FAST_M3_BLOCKS as the attach reads it (serving_runtime.m3_blocks, which ships beside this module): 1 unset
    or '1', 2 for '2', ValueError naming the flag for anything else."""
    value = (os.environ if environ is None else environ).get(M3_BLOCKS_ENV, '1')
    if value not in ('1', '2'):
        raise ValueError('%s must be 1 or 2, got %r' % (M3_BLOCKS_ENV, value))
    return int(value)


def check_environment(environ, m3):
    """Refusal strings for the shape and the modes (items 1-2); [] when they hold. `m3` is
    serving_runtime.m3_shape's (met, description)."""
    problems = []
    met, shape = m3
    try:
        m3_blocks(environ)
    except ValueError as error:
        problems.append(str(error))
    if not met:
        problems.append('the extent path is the 64-row M3 block\'s (users=4 FOUR_AS_TWO=0 PACKED_STEP=1; or users=8 '
                        'with %s=2, two such blocks), not %s' % (M3_BLOCKS_ENV, shape))
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


def check_runtime(root, binaries=None, served=None):
    """Item 3 without the environment: the K64j binary (or, with `served` naming it, K64j-OQ) at every path BINARIES names, carrying
    BINARY_LITERALS (and K64j-OQ's own); the four K64j kernels; the audited tree-scratch sources. `binaries`, when given, is
    {path: sha256} already read this process (runtime_binary_override.install's record), so the attach
    does not hash the binaries twice. Returns a record; raises AdmissionRefused with every problem, each
    naming its file relative to `root` (one short line each)."""
    from dflash_combined_sim_runtime import BINARIES

    root = Path(root)
    served = K64J_TTNNCPP_SHA256 if served is None else served
    label = binary_label(served)
    literals = BINARY_LITERALS + (ONEQ_BINARY_LITERALS if served == K64J_OQ_TTNNCPP_SHA256 else ())
    problems, record = [], dict(binaries={}, kernels={}, literals=[literal.decode() for literal in literals])
    for name in BINARIES:
        path = root / name
        if not path.is_file():
            problems.append('%s is missing' % name)
            continue
        digest = (binaries or {}).get(name) or sha256_file(path)
        record['binaries'][name] = digest
        if digest != served:
            problems.append('%s is %s, not %s\'s %s' % (name, digest[:16], label, served[:16]))
        missing = literals_missing(path, literals)
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


def capacity_design(capacity):
    """The design set the evidence must cover at a served capacity: K1's families, K2's and X7's families, K2's ticket count, X7's
    and Z's floors, and CB2b's named R2 families. 131,328 is exactly the constants above; 262,144 adds the families that reach
    the full window (K1 196,864 and 262,144; K2/X7 and R2 262,144). ValueError for any other capacity."""
    if capacity == CAPACITY_131K:
        return dict(capacity=capacity, k1_extents=K1_EXTENTS, cb2_extents=CB2_EXTENTS, k2_tickets=CB2A_K2_TICKETS,
                    x7_floor=X7_FLOOR, z_floor=Z_FLOOR, r2_named=CB2B_R2_NAMED)
    if capacity == CAPACITY_262K:
        return dict(capacity=capacity, k1_extents=K1_EXTENTS_262K, cb2_extents=CB2_EXTENTS_262K,
                    k2_tickets=CB2A_K2_TICKETS_262K, x7_floor=X7_FLOOR_262K, z_floor=Z_FLOOR, r2_named=CB2B_R2_NAMED_262K)
    raise ValueError('capacity %r is not one this admission has evidence for (%s)' % (capacity, ', '.join(map(str, CAPACITIES))))


def served_capacity(environ=None):
    """The window this process serves: QWEN_FAST_MAX_POSITION rounded up to a whole 64-key page; 131,328 when unset or empty
    (today's path, byte for byte). ValueError naming the variable for anything that is not a positive decimal integer."""
    value = (os.environ if environ is None else environ).get(CAPACITY_ENV, '')
    if value == '':
        return CAPACITY_131K
    if not value.isdigit() or int(value) < 1 or str(int(value)) != value:
        raise ValueError('%s must be a positive decimal integer, got %r' % (CAPACITY_ENV, value))
    return -(-int(value) // 64) * 64


def capacity_problem(capacity):
    """The refusal text for a capacity this admission has no evidence for; None when it has."""
    if capacity in CAPACITIES:
        return None
    return ('capacity %d (%s=%s) is neither %d nor %d: this admission has evidence for those two windows only'
            % (capacity, CAPACITY_ENV, capacity, CAPACITY_131K, CAPACITY_262K))


def _cb1_problems(cb1, problems, flag=CB1_FLAG, design=None):
    design = design or capacity_design(CAPACITY_131K)
    _cover(problems, cb1, 'seeds', SEEDS, 'CB1')
    missing = _missing(cb1.get('extents'), design['k1_extents'])
    if missing:
        problems.append('CB1: extents lack K1\'s %s' % missing)
    if design['capacity'] != CAPACITY_131K and cb1.get('capacity') != design['capacity']:
        problems.append('CB1: capacity %s, not the served C = %d' % (cb1.get('capacity'), design['capacity']))
    combos = cb1.get('combos') or []
    if not any(isinstance(combo, dict) and combo.get('geometry') == 'G8B2' and flag in (combo.get('flags') or [])
               for combo in combos):
        problems.append('CB1: no G8B2 %s combo, the one the extent reader serves' % flag)
    counts = cb1.get('counts') or {}
    for key in CB1_COUNTS:
        if not _full(counts.get(key)):
            problems.append('CB1: %s %s is not a full pass' % (key, counts.get(key)))


def _cb2a_problems(cb2a, problems, design=None):
    design = design or capacity_design(CAPACITY_131K)
    _cover(problems, cb2a, 'seeds', SEEDS, 'CB2a')
    _cover(problems, cb2a, 'variants', CB2A_VARIANTS, 'CB2a')
    if design['capacity'] != CAPACITY_131K:
        if cb2a.get('capacity') != design['capacity']:
            problems.append('CB2a: capacity %s, not the served C = %d' % (cb2a.get('capacity'), design['capacity']))
        _cover(problems, cb2a, 'cb2_extents', design['cb2_extents'], 'CB2a')
    k2 = cb2a.get('k2') or {}
    if k2.get('verdict') != 'PASS':
        problems.append('CB2a: K2 verdict %s, not PASS (a failed or coverage-reduced K2 leaves the exactness '
                        'policy to the user, design 6.1 D-c)' % k2.get('verdict'))
    if not _full(k2.get('tickets'), design['k2_tickets']) or (_count(k2.get('tickets')) or (0, 0))[1] != design['k2_tickets']:
        problems.append('CB2a: K2 tickets %s, not a full pass of the %d the design asks' % (k2.get('tickets'),
                                                                                         design['k2_tickets']))
    if not _full(k2.get('rows')):
        problems.append('CB2a: K2 rows %s are not all bitwise equal' % k2.get('rows'))
    if not _full(cb2a.get('x7'), design['x7_floor']):
        problems.append('CB2a: X7 %s, not a full pass of the %d the design asks' % (cb2a.get('x7'), design['x7_floor']))
    z = cb2a.get('z') or {}
    if not _full(z.get('passed'), design['z_floor']):
        problems.append('CB2a: Z %s, not a full pass of the %d the design asks' % (z.get('passed'), design['z_floor']))
    missing = _missing(z.get('families'), Z_FAMILIES)
    if missing:
        problems.append('CB2a: Z families lack %s of the 15 below 4096' % missing)


def _cb2b_problems(cb2b, sources, problems, chips=CB2B_CHIPS, source_names=QUALIFIED_SOURCES, design=None):
    design = design or capacity_design(CAPACITY_131K)
    # the 131,328 record is held to CB2B_CAPACITY itself (the card-less evidence chain narrows it to its fake runs' capacity)
    capacity = CB2B_CAPACITY if design['capacity'] == CAPACITY_131K else design['capacity']
    if cb2b.get('scope') != CB2B_SCOPE:
        problems.append('CB2b: scope %s, not full (a reduced run - the watcher pass, one seed - is never CB2b\'s '
                        'evidence)' % cb2b.get('scope'))
    if cb2b.get('chips') not in chips:
        problems.append('CB2b: chips %s, not the harness\'s %s' % (cb2b.get('chips'), '/'.join(chips)))
    if cb2b.get('capacity') != capacity:
        problems.append('CB2b: capacity %s, not the served C = %d' % (cb2b.get('capacity'), capacity))
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
    if (not isinstance(families, list) or any(type(family) is not int or family % 256 or not 256 <= family <= capacity
                                              for family in families) or len(set(families)) != len(families)):
        problems.append('CB2b: r2_families is not a list of distinct 256-key families up to %d' % capacity)
    else:
        if len(families) < CB2B_R2_MIN_FAMILIES:
            problems.append('CB2b: R2 replayed %d families, not more than 50' % len(families))
        named = _missing(families, design['r2_named'])
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


def evidence_problems(evidence, sources_root=HERE, tp=None, capacity=CAPACITY_131K, environ=None):
    """Every reason the evidence does not qualify the extent path for these bytes; [] when it does. `tp` is the
    width the record is for (default: the width this process serves at); four cards read the reader twin's sha256,
    a G8B2 0x23 CB1 combo, CB2b's 1of4 chip view, and one KV head in CB1, CB2a and CB2b (_one_head_problems).
    `capacity` is the window the record is for (default 131,328, today's set; 262,144 only at four cards: the design set of
    capacity_design, and the record names its capacity). `environ` names the binary served (served_binary: K64j unless
    QWEN_FAST_RUNTIME_BINARY_SHA256 names K64j-OQ) and whether this is a gate boot, in which K64j-OQ may borrow K64j's record (equivalent)."""
    tp = width() if tp is None else tp
    problems = []
    if not isinstance(evidence, dict) or evidence.get('schema') != EVIDENCE_SCHEMA:
        return ['the evidence is not a %s record' % EVIDENCE_SCHEMA]
    design = capacity_design(capacity)
    if capacity != CAPACITY_131K:
        if tp == tp_shapes.PAIR:
            return ['the %d-token window is a four-card window: there is no pair record for it' % capacity]
        if evidence.get('capacity') != capacity:
            problems.append('capacity: the record is for %s, not %d' % (evidence.get('capacity'), capacity))
    binary = evidence.get('binary') or {}
    served = served_binary(environ)
    recorded = binary.get('ttnncpp_sha256')
    if recorded != served and not equivalent(recorded, served, environ):
        problems.append('binary: the evidence qualified %s, not %s %s%s' % (
            str(recorded)[:16], binary_label(served), served[:16],
            '' if served not in EVIDENCE_BORROWS or EVIDENCE_BORROWS[served] != recorded else
            ' (a record made on K64j stands for K64j-OQ in a gate boot only: re-record it on K64j-OQ before traffic)'))
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
        _cb1_problems(sections['CB1'], problems, CB1_FLAG if tp == tp_shapes.PAIR else CB1_FLAG_TP4, design)
    if _passed(sections.get('CB2a'), 'CB2a', problems):
        _cb2a_problems(sections['CB2a'], problems, design)
    if _passed(sections.get('CB2b'), 'CB2b', problems):
        _cb2b_problems(sections['CB2b'], sources, problems,
                       CB2B_CHIPS if tp == tp_shapes.PAIR else CB2B_CHIPS_TP4, qualified_sources(tp), design)
    if tp != tp_shapes.PAIR:
        _one_head_problems(sections, problems)
    return problems


def check_evidence(path=None, *, expected_sha256=None, sources_root=HERE, tp=None, capacity=CAPACITY_131K, environ=None):
    """Item 4: the evidence file at its pinned sha256, and evidence_problems empty. Returns the parsed
    record; raises AdmissionRefused naming every problem (one entry each in its `problems`). The file and its pin
    are the pair's or the four-card record's by `tp` (default: the width this process serves at), or the 262,144-token
    window's own (EVIDENCE_TP4_262K at EVIDENCE_TP4_262K_SHA256) by `capacity`."""
    tp = width() if tp is None else tp
    expected_sha256 = evidence_pin(tp, capacity) if expected_sha256 is None else expected_sha256
    path = Path(evidence_path(tp, capacity) if path is None else path)
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
    problems = evidence_problems(evidence, sources_root, tp, capacity, environ)
    if problems:
        raise AdmissionRefused('the evidence does not qualify the extent path%s: ' % (
            '' if capacity == CAPACITY_131K else ' at capacity %d' % capacity) + '; '.join(problems),
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


def check_window_262k(*, sources_root=HERE, evidence_path_=None, expected_sha256=None, writer_state=None, environ=None):
    """The 262,144-token window's prerequisites, none of them waivable: the window's own evidence record at its own pin
    (check_evidence at capacity 262,144) and E1, the ordered writers' page-table width 4,096 (page_width_tp4). Returns the
    parsed 262k record; raises AdmissionRefused naming every problem with the capacity."""
    problems, record = [], None
    try:
        record = check_evidence(evidence_path_, expected_sha256=expected_sha256, sources_root=sources_root, tp=4,
                                capacity=CAPACITY_262K, environ=environ)
    except AdmissionRefused as refusal:
        problems.extend('capacity %d: %s' % (CAPACITY_262K, problem) for problem in refusal.problems)
    if writer_state is None:
        import page_width_tp4

        writer_state = page_width_tp4.evidence_state()
    ok, writer_problems = writer_state
    if not ok:
        problems.extend('capacity %d: page-table width 4096 is not qualified (E1, ordered_writer_evidence_tp4.json): %s'
                        % (CAPACITY_262K, problem) for problem in writer_problems)
    if problems:
        raise AdmissionRefused('the %d-token window has no qualifying evidence: %s' % (CAPACITY_262K, ' | '.join(problems)),
                               problems)
    return record


def waiver_for_262k(environ, what, log=None):
    """QWEN_FAST_262K_EVIDENCE_WAIVER=1 in a gate run of a gate-only profile: log it (loud, once per process) and return True. Unset: False.
    Set anywhere else: AdmissionRefused naming the flag."""
    import page_width_tp4

    try:
        active = page_width_tp4.waiver_active(environ)
    except page_width_tp4.WaiverRefused as error:
        raise AdmissionRefused(str(error), [str(error)])
    if active:
        page_width_tp4.log_waiver_once(what, log)
    return active


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
        check_evidence(tp=tp, sources_root=sources_root, environ=environ)
    except AdmissionRefused as refusal:
        if not unqualified_allowed(environ):
            raise AdmissionRefused('QWEN_FAST_TP=%d serves on the pair\'s evidence otherwise, and %s is not a qualifying '
                                   'four-card record (%s): %s' % (tp, EVIDENCE_TP4.name, GATE_ENV + ' is not 1',
                                                                 '; '.join(refusal.problems)), refusal.problems)
        _log_unqualified(log, refusal.problems)
        return list(refusal.problems)
    capacity = served_capacity(environ)
    if capacity == CAPACITY_262K:
        # The window's own evidence and E1, in a gate run too: the waiver above is the 131,328 record's alone. Only the explicit
        # QWEN_FAST_262K_EVIDENCE_WAIVER=1 (waiver_for_262k: a gate run of a gate-only profile, else refused) passes without them.
        try:
            check_window_262k(sources_root=sources_root, environ=environ)
        except AdmissionRefused as refusal:
            if not waiver_for_262k(environ, 'tp_guard: %d evidence problems, one UNQUALIFIED line each' % len(refusal.problems), log):
                raise
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
    if requested not in SERVED_BINARIES:
        problems.append('%s=%s, not K64j\'s %s or K64j-OQ\'s %s' % (RUNTIME_BINARY_ENV, requested[:16] or '(unset)',
                                                                   K64J_TTNNCPP_SHA256[:16], K64J_OQ_TTNNCPP_SHA256[:16]))
    served = requested if requested in SERVED_BINARIES else K64J_TTNNCPP_SHA256
    label = binary_label(served)
    if binary_record is not None and binary_record.get('override') != served:
        problems.append('the runtime binary override admitted %s, not %s' % (str(binary_record.get('override'))[:16], label))
    tp = width(environ)
    record = dict(flag=FLAG, shape=m3[1])
    capacity = CAPACITY_131K
    waived_262k = False
    import page_width_tp4

    try:
        waived_262k = page_width_tp4.waiver_active(environ)
    except page_width_tp4.WaiverRefused as error:
        problems.append(str(error))
    try:
        capacity = served_capacity(environ)
    except ValueError as error:
        problems.append(str(error))
    else:
        refusal_text = capacity_problem(capacity)
        if refusal_text is not None:
            problems.append(refusal_text)
        elif capacity == CAPACITY_262K and tp == tp_shapes.PAIR:
            problems.append('capacity %d is a four-card window: %s is not 4' % (capacity, tp_shapes.TP_SWITCH))
    wide = capacity == CAPACITY_262K
    if wide:
        record['capacity'] = capacity
    # QWEN_FAST_M3_BLOCKS=2: the record and the passed line name the block count (and only then, so a one-block attach
    # reads and records exactly what it always did). A malformed value is already a problem of check_environment above.
    blocks = 1
    try:
        blocks = m3_blocks(environ)
    except ValueError:
        pass
    suffix = '' if blocks == 1 else ' blocks=%d' % blocks
    if wide:
        suffix += ' capacity=%d' % capacity
    if blocks != 1:
        record['blocks'] = blocks
    unqualified, waived = [], []
    # The evidence checks take the environment only when this process serves K64j-OQ (the binary named there decides what a record must say); on K64j the calls
    # are exactly what they always were.
    env_kw = {} if served == K64J_TTNNCPP_SHA256 else dict(environ=environ)
    if wide:
        window_check = lambda: check_window_262k(evidence_path_=evidence, **env_kw)
    elif tp == tp_shapes.PAIR:
        window_check = lambda: check_evidence(evidence, **env_kw)
    else:
        window_check = lambda: check_evidence(evidence, tp=tp, **env_kw)
    # K64j: the call is exactly today's (the two arguments); K64j-OQ names itself as the third.
    runtime_args = ((runtime_root, (binary_record or {}).get('binaries')) if served == K64J_TTNNCPP_SHA256
                    else (runtime_root, (binary_record or {}).get('binaries'), served))
    for name, check in (('runtime', lambda: check_runtime(*runtime_args)),
                        ('evidence', window_check)):
        if name == 'evidence' and capacity not in CAPACITIES:
            continue                       # already refused by name above; there is no record to read for it
        try:
            record[name] = check()
        except AdmissionRefused as refusal:
            if name == 'evidence' and wide and waived_262k:
                # QWEN_FAST_262K_EVIDENCE_WAIVER=1 in a gate run of a gate-only profile: the 262k records are not there, say so.
                waived.extend(refusal.problems)
                record[name] = None
            elif name == 'evidence' and not wide and unqualified_allowed(environ):
                unqualified.extend(refusal.problems)
                record[name] = None
            else:
                problems.extend(refusal.problems)
    if problems:
        _refuse(log, 'at attach', problems)
    if waived:
        import page_width_tp4

        page_width_tp4.log_waiver_once('admission at capacity %d passes without the 262k evidence (%d problems, one UNQUALIFIED line each)' % (capacity, len(waived)), log)
        record['waived'] = waived
        _log_unqualified(log, waived)
        log('{} passed UNQUALIFIED: {} {} x{}; kernels {}; {} evidence problems (262k waiver, gate only){}',
            MARKER, label, served[:16], len(record['runtime']['binaries']),
            ','.join(sha[:8] for _, sha in sorted(K64J_KERNELS.items())), len(waived), suffix)
        _STATE['record'] = record
        return record
    if unqualified:
        # A gate run at four cards, before any four-card evidence exists: every missing piece is on the record, one
        # line each, and the process may never take traffic (its profile is gate_only).
        record['unqualified'] = unqualified
        _log_unqualified(log, unqualified)
        log('{} passed UNQUALIFIED: {} {} x{}; kernels {}; {} evidence problems (gate only){}',
            MARKER, label, served[:16], len(record['runtime']['binaries']),
            ','.join(sha[:8] for _, sha in sorted(K64J_KERNELS.items())), len(unqualified), suffix)
        _STATE['record'] = record
        return record
    sections = record['evidence']['sections']
    recorded = (record['evidence'].get('binary') or {}).get('ttnncpp_sha256')
    if equivalent(recorded, served, environ):
        record['binary_equivalence'] = dict(recorded=recorded, served=served)
        log('{} evidence recorded on {} {} stands for {} {} by the reviewed equivalence (gate boot only; traffic needs a record on {})',
            MARKER, binary_label(recorded), recorded[:16], label, served[:16], label)
    log('{} passed: {} {} x{}; kernels {}; evidence {}; CB1 {} CB2a {} CB2b {}; reader {}{}',
        MARKER, label, served[:16], len(record['runtime']['binaries']),
        ','.join(sha[:8] for _, sha in sorted(K64J_KERNELS.items())), evidence_pin(tp, capacity)[:16],
        sections['CB1']['run'], sections['CB2a']['run'], sections['CB2b']['run'],
        ','.join(record['evidence']['sources'][name][:16] for name in qualified_sources(tp)), suffix)
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
    admitted_capacity = (_STATE.get('record') or {}).get('capacity')
    if admitted_capacity is not None:
        # A window other than 131,328 was admitted (the record names it): the pool's table must be exactly that wide - a pool
        # built at the 131k width under the 262k admission (or the reverse) is the evidence's geometry on another one.
        page_width = getattr(pool, 'page_width', None)
        if type(page_width) is not int or page_width * 64 != admitted_capacity:
            _refuse(log, 'after the pool', ['the pool\'s page table is %r pages (%r keys), not the admitted capacity %d (%d pages)'
                                            % (page_width, page_width * 64 if type(page_width) is int else None,
                                               admitted_capacity, admitted_capacity // 64)])
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
