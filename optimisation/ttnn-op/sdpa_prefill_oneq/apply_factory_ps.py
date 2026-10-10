"""Apply the [QWEN-SDPA-PF] oneq edits PS0-PS5 to the prefill SDPA factory: ONE q chunk per core, no causal pairing.

Prefill lever S1 (optimisation/ttnn-op/sdpa_prefill_oneq/README.md). The served causal chunked SDPA hands the flat
(batch, head, q chunk) space out in PAIRS (sdpa_program_factory.cpp: `global_q_pair_distribute = is_causal &&
(q_num_chunks % 2 == 0)`), and the kernels' shared zigzag remap (q_chunk_remapping.hpp) turns a pair of consecutive flat
positions into the light q chunk k and the heavy q chunk N-1-k. At TP4 (6 local Q heads, 1 KV head, 2048 rows, q/k chunk
128) that is 96 q chunks = 48 pairs: 48 of the grid's 110/130 cores work, each for 2K+17 128x128 blocks (K = context
tokens / 128). This patch adds one flag to the existing per-call word:

    SDPAProgramConfig.max_cores_per_head_batch = 0x5EFA0000 | flags         flags: 0x1 chain, 0x2 inj_batch, 0x4 noc_order,
                                                                             0x8 oneq (NEW)

With 0x8 (and the chain flag 0x1) the factory turns the pair distribution off, so the stock non-pair split of the same
file gives core i the single flat position i (96 cores for 96 q chunks, 1 chunk each, blocks K+q+1 for q chunk q), the
kernels' zigzag remap stays a bijection inside each head (every q chunk is visited exactly once, on some core), and the
F4 grouping of PF keys the cores by their whole unit list: 16 chains of 6 (one per q chunk, the 6 heads that share the
KV head). Nothing else moves: the compute, writer and chain-reader kernels are the very same files, the per-q-chunk
arithmetic and the k-chunk order are untouched, so the output is byte-identical to the paired chain's and to the stock
path's (exact by construction; the card-M sweep proves it on hardware).

    PS0  the decode of the oneq bits and the work split's pair switch (before the split, PF:397)
    PS1  the constant kQwenPfOneQ = 0x8 (F1's block)
    PS2  the flag 0x8 in F1's accepted set
    PS3  the F4 envelope takes `global_q_pair_distribute || qwen_one_q`
    PS4  the oneq envelope: at most one q chunk per core (TT_FATAL otherwise: TP2's 192 chunks, any shape above the grid)
    PS5  the log line `[QWEN-SDPA-PF] oneq=1 ...` after the PF log line

Input must be the PF factory (apply_factory_pf.py output, sha256 bfab8558...) or the served fd8c0676 (PF is applied
first); any other input is refused. Every anchor occurs exactly once and starts at the line recorded below (in the PF
file); the result must hash to PS_FACTORY; the edits invert; every new line lies inside create_descriptor (nothing at
namespace scope: the unity-build collision trap); the qwen_draft_fp32_intermediates block is byte-identical; no CR.

    python3 apply_factory_ps.py <factory.cpp> --out X.cpp     # write the patched factory to X.cpp
    python3 apply_factory_ps.py <factory.cpp>                 # in place, keeps .orig-<sha8>
    python3 apply_factory_ps.py <factory.cpp> --record        # print the output sha, do not enforce it
"""

import argparse
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'sdpa_prefill_chain'))

import apply_factory_pf as pf  # noqa: E402

NL = chr(10)

SERVED_FACTORY = pf.BASE_FACTORY
PF_FACTORY = pf.PF_FACTORY
PS_FACTORY = '37b9d966b9c85a09eb1e8941b6af789b5a2c9263fea9a8b929284560ede689ce'

# Constants that cross file boundaries (the planner, the card test, the Python twin, the build script; the CPU tests
# check each against the C++ text).
PF_TAG = pf.PF_TAG
FLAG_ONEQ = 0x8
ONEQ_BITS = pf.FLAG_KV_CHAIN | FLAG_ONEQ          # the early decode needs both
PRODUCTION_FLAGS = (0x1, 0x3, 0x5, 0x7)           # the chain's own sets (PF)
ONEQ_FLAGS = tuple(flags | FLAG_ONEQ for flags in PRODUCTION_FLAGS)   # 0x9, 0xB, 0xD, 0xF
FATAL_TEXT = '[QWEN-SDPA-PF] oneq needs one q chunk per core: {} q chunks on {} cores'
LOG_TEXT = '[QWEN-SDPA-PF] oneq=1 q_chunks={} cores={} chunks_per_core=1 chains={} members={}'
NEW_QWEN_STRINGS = (FATAL_TEXT, LOG_TEXT)         # the only QWEN strings this patch adds to the binary
MARKERS = (FATAL_TEXT, LOG_TEXT, 'qwen_one_q', 'kQwenPfOneQ')
FUNCTION_LINE = pf.FUNCTION_LINE


def lines(*parts):
    return ''.join(part + NL for part in parts)


# ---- PS0: the decode and the pair switch (PF:397) ----------------------------------------------
PS0_OLD = lines('    const bool global_q_pair_distribute = is_causal && (q_num_chunks % 2 == 0);')
PS0_NEW = lines(
    '    // [QWEN-SDPA-PF] oneq (flag 0x8 with the chain flag 0x1): ONE q chunk per core, no causal pairing. Decoded here, ahead of the',
    '    // work split, from the very word F1 parses below (F1 refuses every word this does not recognise). Any other call: unchanged.',
    '    constexpr uint32_t kQwenOqTagMask = 0xFFFF0000u, kQwenOqTag = 0x5EFA0000u, kQwenOqBits = 0x9u;',
    '    const uint32_t qwen_oq_word =',
    '        program_config.has_value() ? static_cast<uint32_t>(program_config->max_cores_per_head_batch) : 0u;',
    '    const bool qwen_one_q = (qwen_oq_word & kQwenOqTagMask) == kQwenOqTag && (qwen_oq_word & kQwenOqBits) == kQwenOqBits;',
    '    const bool global_q_pair_distribute = is_causal && (q_num_chunks % 2 == 0) && !qwen_one_q;')

# ---- PS1: the constant (F1's block) ------------------------------------------------------------
PS1_OLD = lines('    constexpr uint32_t kQwenPfTestMutate = 0x100u, kQwenPfTestHang = 0x200u;')
PS1_NEW = PS1_OLD + lines('    constexpr uint32_t kQwenPfOneQ = 0x8u;')

# ---- PS2: the accepted set ---------------------------------------------------------------------
PS2_OLD = lines('        TT_FATAL(qwen_kv_chain && (qwen_pf_flags & ~(kQwenPfKvChain | kQwenPfInjBatch | kQwenPfNocOrder |')
PS2_NEW = lines(
    '        TT_FATAL(qwen_kv_chain && (qwen_pf_flags & ~(kQwenPfKvChain | kQwenPfInjBatch | kQwenPfNocOrder | kQwenPfOneQ |')

# ---- PS3: the envelope's pair predicate --------------------------------------------------------
PS3_OLD = lines(
    '                     NKH == NVH && NQH % NKH == 0 && global_q_pair_distribute && Sq_chunk_t == Sk_chunk_t &&')
PS3_NEW = lines(
    '                     NKH == NVH && NQH % NKH == 0 && (global_q_pair_distribute || (qwen_one_q && q_num_chunks % 2 == 0)) &&',
    '                     Sq_chunk_t == Sk_chunk_t &&')

# ---- PS4: the oneq envelope --------------------------------------------------------------------
PS4_OLD = lines('                 "[QWEN-SDPA-PF] kv_chain outside its qualified envelope");')
PS4_NEW = PS4_OLD + lines(
    '        if (qwen_one_q) {  // the non-pair split of the file: min(total, cores) cores with ONE chunk each, or more than one and refused',
    '            TT_FATAL(max_global_q_chunks_per_core == 1,',
    '                     "%s", total_q_chunks, num_cores);' % FATAL_TEXT.replace('%', '%%'),
    '        }')

# ---- PS5: the log line -------------------------------------------------------------------------
PS5_OLD = lines('                 qwen_chains, qwen_members, (qwen_pf_flags & kQwenPfNocOrder) ? "noc" : "raster");')
PS5_NEW = PS5_OLD + lines(
    '        if (qwen_one_q) {',
    '            log_info(tt::LogOp, "%s",' % LOG_TEXT,
    '                     total_q_chunks, num_cores, qwen_chains, qwen_members);',
    '        }')

# (label, PF line the anchor starts on, old text, new text), in file order.
EDITS = (
    ('PS0', 397, PS0_OLD, PS0_NEW),
    ('PS1', 537, PS1_OLD, PS1_NEW),
    ('PS2', 544, PS2_OLD, PS2_NEW),
    ('PS3', 1380, PS3_OLD, PS3_NEW),
    ('PS4', 1383, PS4_OLD, PS4_NEW),
    ('PS5', 1471, PS5_OLD, PS5_NEW),
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class Refusal(ValueError):
    """The input is not the PF (or served) factory, or an anchor drifted: nothing is written."""


def check_anchors(text):
    for label, line, old, _new in EDITS:
        count = text.count(old)
        if count != 1:
            raise Refusal('anchor %s occurs %d times (need exactly 1)' % (label, count))
        found = text[:text.index(old)].count(NL) + 1
        if found != line:
            raise Refusal('anchor %s starts at line %d, recorded PF:%d; refusing on drift' % (label, found, line))


def to_pf(source):
    """The PF factory bytes from `source` (the PF factory itself, or the served fd8c0676 with PF applied)."""
    digest = sha256(source)
    if digest == PF_FACTORY:
        return source
    if digest == SERVED_FACTORY:
        try:
            return pf.patch(source)
        except pf.Refusal as error:
            raise Refusal('PF edits refused: %s' % error)
    raise Refusal('unexpected factory %s (need the PF factory %s or the served %s)' % (digest, PF_FACTORY, SERVED_FACTORY))


def patch(source):
    """The PS factory from the PF (or served) bytes; Refusal on any other input or on drift."""
    base = to_pf(source)
    text = base.decode('utf-8')
    check_anchors(text)
    for _label, _line, old, new in EDITS:
        text = text.replace(old, new)
    for marker in MARKERS:
        if marker not in text:
            raise Refusal('patched factory lacks %r' % marker)
    if pf.protected_block(text) != pf.protected_block(base.decode('utf-8')):
        raise Refusal('the qwen_draft_fp32_intermediates block moved')
    body = text.index(FUNCTION_LINE)
    for label, _line, _old, new in EDITS:
        if text.index(new) < body:
            raise Refusal('edit %s lies outside create_descriptor' % label)
    out = text.encode('utf-8')
    if chr(13).encode() in out:
        raise Refusal('output has a CR; files must be LF')
    return out


def unpatch(patched):
    """The inverse, last edit first: proves the output is the PF factory plus exactly these edits."""
    text = patched.decode('utf-8')
    for label, _line, old, new in reversed(EDITS):
        if text.count(new) != 1:
            raise Refusal('edit %s occurs %d times' % (label, text.count(new)))
        text = text.replace(new, old)
    return text.encode('utf-8')


KNOWN_FLAGS = pf.KNOWN_FLAGS | FLAG_ONEQ


def decode_word(word, test_env=None):
    """F1 plus PS2 in Python: (chain, flags, oneq) for SDPAProgramConfig.max_cores_per_head_batch = word, or ValueError
    with the TT_FATAL text F1 raises. test_env is the value of QWEN_SDPA_PF_TEST."""
    word &= 0xFFFFFFFF
    if (word & pf.PF_TAG_MASK) != PF_TAG:
        return False, 0, False
    flags = word & ~pf.PF_TAG_MASK & 0xFFFFFFFF
    if not flags & pf.FLAG_KV_CHAIN or flags & ~KNOWN_FLAGS:
        raise ValueError('%s %#x' % (pf.FLAGS_MARKER, flags))
    if flags & pf.TEST_FLAGS and test_env != '1':
        raise ValueError('%s %#x need %s=1' % (pf.TEST_MARKER, flags, pf.TEST_ENV))
    return True, flags, bool(flags & FLAG_ONEQ)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('factory')
    parser.add_argument('--out', help='write here instead of patching in place')
    parser.add_argument('--record', action='store_true', help='print the output sha instead of enforcing it')
    args = parser.parse_args(argv)
    path = Path(args.factory)
    source = path.read_bytes()
    if PS_FACTORY != '__RECORD__' and sha256(source) == PS_FACTORY:
        print('factory already [QWEN-SDPA-PF]-oneq-patched %s' % PS_FACTORY)
        if args.out:
            Path(args.out).write_bytes(source)
        return 0
    try:
        patched = patch(source)
    except Refusal as error:
        print('refused: %s' % error)
        return 1
    digest = sha256(patched)
    if not args.record and digest != PS_FACTORY:
        print('reconstruction mismatch: built %s, recorded %s' % (digest, PS_FACTORY))
        return 1
    if unpatch(patched) != to_pf(source):
        print('edits do not invert')
        return 1
    if args.out:
        Path(args.out).write_bytes(patched)
        target = Path(args.out)
    else:
        backup = path.with_name(path.name + '.orig-' + sha256(source)[:8])
        if not backup.exists():
            backup.write_bytes(source)
        path.write_bytes(patched)
        target = path
    print('factory %s -> [QWEN-SDPA-PF] oneq %s written %s' % (sha256(source)[:16], digest, target))
    print(digest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
