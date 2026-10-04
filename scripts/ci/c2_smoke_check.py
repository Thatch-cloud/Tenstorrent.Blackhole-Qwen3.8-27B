"""c2_smoke_check: the smoke's stop conditions, enforced (the four-card speed window's K1 and K2).

c2_serving_smoke.py records every test's exception as an `error` entry and exits 0, and the quad smoke step fails only
when the container exits or never becomes ready, so a bad smoke used to let a window chain go on. This reads the smoke's
own SMOKE_JSON line and the container log and exits non-zero on:
  - an `error` entry in a core test (warmup, warm_lifecycle, coding, concurrent4, long_real_text), a warmup status other
    than 200, or a stream (coding, a concurrent user, a warm_lifecycle row) that produced no tokens or no finish;
  - garbage text: a coding or concurrent stream whose kept text is empty, mostly non-printable, or a repetition of a few
    characters;
  - an audit mismatch line in the container log ('audit mismatch': the verify t1 audit, round b1, the extent audit);
  - a '[PINDIAG] tp4 vglue fell back' line: a verify-glue lever declined and its served path ran, so its timing is not the lever's;
  - a ramp commit above --max-ramp-kv-ms (default 50) when the served profile has the drafter's K/V slide on
    (QWEN_FAST_TP_KV_SLIDE=1): the MEDIAN, over the [PACKED-PUBLISH] rounds with a commit, of each round's largest
    prepare_history entry. The median, so the attach's few compile-bearing rounds do not fail it; the eager chain's ~310 ms
    per ramp user in v140 would;
  - the batched draft, when the smoke ran the `concurrent4_steady` test (four users past the 2,048-row draft window from the
    first round, the only mix a pair or the quad can serve): what the profile asked for must have run. QWEN_FAST_QUAD_DRAFT=1:
    the quad's marker exactly once, at least one round the quad served, no fallback and no disable line. Off: no quad line, and
    at least one round two packed pairs served (a comparison arm that never batched compares nothing). A
    [DRAFT-SINGLES-AUDIT] line with equal=0 or a [QUAD-AUDIT] line with equal=0 fails; a profile with
    QWEN_FAST_DRAFT_SINGLES_AUDIT set and no audit line fails;
  - the fused commit (QWEN_FAST_FUSED_COMMIT=1, the round-fence plan's H1b; at four cards fused_commit_tp), by the gate's own H1b
    rules (lever_n_m3native_gate.h1b_report: a refused build, an engaged line that disagrees with the flags, no fused publication,
    a refusal reason but `ramp` and `parity`, a discard after an in-place slide, an audit count that is not the fused count, an audit
    mismatch) and, here, that the engaged line is there once with the trace count the flags imply (users x (1 + 16 prefixes) in
    place, users otherwise), every audit line checked the items the width implies (10 banks x chips, twice in place: 80 at four
    cards), and - in a smoke that ran `concurrent4_steady` - that a round of four fused publications happened and, under
    QWEN_FAST_FUSED_COMMIT_LIVE_BANKS, that the live-bank marker was logged. A profile with the flag off must log no fused line.

  - the eight-seat host-gap levers (tp4/hostgap, hostgap_problems): QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE=1 must log its engaged line once
    for each M3 block and no refusal line; QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS=1 its block-epochs line and, in a smoke that ran
    concurrent8_steady (and QWEN_FAST_TP4_HOSTGAP_LOG=1), at least 95% of the 4-live verifies of the pre-staged blocks (A and B under the
    epochs, A under the lite arm) on the diff path, counting every full stage as a miss but those after an external writer
    (epoch:admission, detach, prefill, prefill-chunk, bookkeeping, lane-switch: a 'no-snapshot' or an 'epoch:verify' is a miss); the audit flag (QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT=1) at least one [PACKED-PRESTAGE-FULLAUDIT] line for
    each block that pre-stages, at least one path=full line (the comparator checked against the full stage itself), buffers x chips
    checked on every line, and zero mismatches in every one (a path=full mismatch is the comparator's artifact, a path=diff one the lever's, and the
    two are reported apart); a profile without the flag logs none of these lines;
  - a code-prompt answer that is not text (coding, concurrent4_steady, steady_resend): a stream that finished with `stop` at
    its first token (an instant EOS: v172's users 0 and 1) or before MIN_ANSWER_TOKENS, or whose sample is mostly non-Latin
    script (v172's users 2 and 3: mixed-script symbols), and, at four cards (QWEN_FAST_TP not 2), a first prefill whose [MEMLEDGER] item=model_after_prefill
    line reports buffers: the model allocated device state at request time, after the packed traces were captured
    (serving_runtime.prefill_warm_before_traces), which is the hazard v172 fell into; also at four cards the eager prefill warm
    line must precede Metal's 'Allocating device buffers is unsafe due to the existence of an active trace' warning, and no
    prefill may compile a program beyond its window snapshot's ('[PINDIAG] four-card prefill programs=A->B window=W': B-A-W
    above zero is the #48536 sequence). Every prefill is counted, the ones that end at their first token (the max_tokens=1
    warmup, the first prefill after the attach) included; the engine builds' own lines ('[PINDIAG] four-card engine
    programs=A->B') are facts, not rules: they compile after the capture by design;
  - the speed window's full-text reference (opt-in tests concurrent4_solo, concurrent4_v164order, concurrent4_code,
    concurrent4_code_equal, replay_concurrent4): every concurrent4 (and concurrent4_v164order) user's content hash, reasoning hash,
    completion tokens and finish reason must equal the same prompt's solo run on this image (concurrent4_solo, 800 tokens each); a
    difference names the user's packed segment, read from the container log's [PACKED] lines (the user's prompt length is the
    position of its request's first line). The two fields are hashed apart because the step that carries </think> splits the
    delta differently at 16 packed rows than at 4 solo. The stream rules above apply to the new concurrent tests, the code tests
    are judged as code answers, and replay_concurrent4's four non-streamed requests must each return 200 with tokens;
  - steady_resend (opt-in, after concurrent4_steady): the first steady prompt again, alone, judged as a code answer (above).
    Its prefill reuses, after packed and pair replays, the window-snapshot programs the steady test compiled after the
    capture - the one prefill program set no attach warm covers;
  - the two STREAMED parser tests (opt-in): stream_tool_call (a streamed tool_choice auto request must arrive as tool_calls
    deltas with a JSON call and no <tool_call> marker in the content) and stream_reasoning (reasoning_content non-empty, no
    <think> or </think> in the content): parser M runs on the streaming path only, so the non-streaming tool_call test does
    not exercise it;
  - a TRAFFIC profile (one that is not gate only, serving the extent replay), judged when the check is given the profile's
    entry (--profile always does): its attach must log '[PINDIAG] packed-any admission passed:' and no 'UNQUALIFIED' or
    'refused' admission line (the record qualified these bytes, no waiver), and a parser_rechunk profile must log the
    contract's 'parser M armed' line (the parser fix is live in the server that took the requests);

  python c2_smoke_check.py --smoke-log smoke.log --container-log container.log --profile P [--profiles qwen_c2_profiles.json]
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

SOLO_TEST = 'concurrent4_solo'
REPLAY_TEST = 'replay_concurrent4'
# The eight-seat tests (tp4/seats8: QWEN_FAST_M3_BLOCKS=2), judged as the four-user ones are: every user a stream that ends in tokens,
# the code ones code answers too; the eight-user replay is judged as the four-user one.
EIGHT_TESTS = ('concurrent8_code_equal', 'concurrent8_code_32k', 'concurrent8_code_128k', 'concurrent8_skew', 'concurrent5_split', 'concurrent8_drain',
               'concurrent8_steady')
# The 262k stall shape (tp4/seats262k): seven decoding users and one cold 253,920-token arrival; its numbers are recorded, not gated, but
# every stream must end in tokens (the arrival's too) and the arrival's time to first token must exist.
STALL_TEST = 'stall8_cold262k'
REPLAY_TESTS = (REPLAY_TEST, 'replay_concurrent8')
# Four-user streamed tests: their users get the stream rules; the code ones are code answers too (TEXT_TESTS).
CONCURRENT_TESTS = ('concurrent4', 'concurrent4_v164order', 'concurrent4_steady', 'concurrent4_code', 'concurrent4_code_equal',
                    'concurrent4_code_32k', 'concurrent8_code') + EIGHT_TESTS
# The tests whose users are the four concurrent4 prompts, comparable with their solo runs.
SOLO_COMPARED = ('concurrent4', 'concurrent4_v164order')
CORE = ('warmup', 'warm_lifecycle', 'coding', 'concurrent4', 'concurrent4_v164order', 'concurrent4_steady', 'concurrent4_code',
        'concurrent4_code_equal', 'concurrent4_code_32k', 'concurrent8_code') + EIGHT_TESTS + (SOLO_TEST, REPLAY_TEST, 'replay_concurrent8', 'long_real_text', 'steady_resend')
PACKED_LINE = re.compile(r'\[PACKED\] request=(\S+) segment=(\d+) position=(\d+) ')
SOLO_FIELDS = (('content_sha256', 'content'), ('reasoning_sha256', 'reasoning'), ('completion_tokens', 'tokens'),
               ('finish', 'finish'))
PUBLISH = re.compile(r'\[PACKED-PUBLISH\] round=\d+ stages=\{.*?prepare_history: \[([0-9.,\s]*)\]')
MISMATCH = re.compile(r'audit mismatch', re.IGNORECASE)
VGLUE_FELL_BACK = '[PINDIAG] tp4 vglue fell back'
# tp4/next-3-cheap (draft_wide_tp): the drafter's wide norms. A fall-back line fails the smoke (the timing is not the lever's); a profile that asks for the lever
# (QWEN_FAST_TP4_DRAFT_WIDE=1) and logs no engaged line fails too.
DRAFT_WIDE_FLAG = 'QWEN_FAST_TP4_DRAFT_WIDE'
DRAFT_WIDE_ENGAGED = '[PINDIAG] tp4 draft wide engaged'
DRAFT_WIDE_FELL_BACK = '[PINDIAG] tp4 draft wide fell back'
# tp4/gluefix (gdn_pair_slice_tp): the packed GDN block's odd-user slices from one row-major conversion. Same rule as the wide norms: a fall-back line fails
# the smoke, and a profile that sets QWEN_FAST_GDN_PAIR_SLICE=1 without V2 (QWEN_FAST_TP4_GDN_GLUE, which replaces the slices and leaves it idle) must log the engaged line.
PAIR_SLICE_FLAG = 'QWEN_FAST_GDN_PAIR_SLICE'
PAIR_SLICE_ENGAGED = '[PINDIAG] tp4 pair slice engaged'
PAIR_SLICE_FELL_BACK = '[PINDIAG] tp4 pair slice fell back'
VGLUE_AUDIT_FLAG = 'QWEN_FAST_TP4_VGLUE_AUDIT'
VGLUE_AUDIT_PASSED = '[PINDIAG] tp4 vglue audit '
DISPATCH_DIAG_FLAG = 'QWEN_FAST_GDN_DISPATCH_DIAG'
DISPATCH_DIAG_LINE = '[PINDIAG] tp4 gdn dispatch diag'
# tp4/samp-draft (tp4_sampdraft): the sampler's shard argmax kernels, the drafter's rewritten conv I/O and its tile-copy head ops. Per lever the
# same three rules as the wide norms: a fall-back line fails the smoke (the served path ran, the timing is not the lever's), a profile that asks
# for the lever and logs no engaged line fails, and an audit flag with no 'exact=True' audit line fails (an audited arm must have audited).
# (flag, engaged marker, fell-back marker, what it is); (audit flag, audit marker) per audited lever. tests hold these equal to tp4_sampdraft's.
SAMPDRAFT_LEVERS = (('QWEN_FAST_TP4_SHARD_ARGMAX', '[PINDIAG] tp4 shard argmax engaged', '[PINDIAG] tp4 shard argmax fell back',
                     'shard argmax kernels'),
                    ('QWEN_FAST_TP4_DRAFT_CONV', '[PINDIAG] tp4 draft conv engaged', '[PINDIAG] tp4 draft conv fell back',
                     'drafter conv I/O kernels'),
                    ('QWEN_FAST_TP4_DRAFT_HEADS', '[PINDIAG] tp4 draft heads engaged', '[PINDIAG] tp4 draft heads fell back',
                     'drafter head copies'))
SAMPDRAFT_AUDITS = (('QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT', '[PINDIAG] tp4 shard argmax audit', 'shard argmax'),
                    ('QWEN_FAST_TP4_DRAFT_CONV_AUDIT', '[PINDIAG] tp4 draft conv audit', 'drafter conv'),
                    ('QWEN_FAST_TP4_DRAFT_HEADS_AUDIT', '[PINDIAG] tp4 draft heads audit', 'drafter heads'))
# tp4/v5split (gdn_seq_block_split): the K5-A recurrence launch with each head's value columns split over two cores. A profile that asks for it
# (QWEN_FAST_GDN_SPLIT_V=2) and logs no engaged line ran K5-A, and its timing says nothing about the lever; the line is logged once per user count.
GDN_SPLIT_FLAG = 'QWEN_FAST_GDN_SPLIT_V'
GDN_SPLIT_ENGAGED = '[PINDIAG] gdn split_v build split=2'

# tp4/sdpa-long (sdpa_long_tp): a profile that names an SDPA configuration (QWEN_FAST_TP4_SDPA, anything but unset, 0 or off) and logs no engaged
# line ran the served call unchanged, so its timing is not the configuration's.
SDPA_LONG_FLAG = 'QWEN_FAST_TP4_SDPA'
SDPA_LONG_ENGAGED = '[PINDIAG] tp4 sdpa engaged'
SDPA_LONG_OFF = ('', '0', 'off')


def sdpa_long_problems(env, container_text):
    """[problem] when the profile's env names an SDPA configuration and the container log has no engaged line."""
    value = (env or {}).get(SDPA_LONG_FLAG)
    if value is None or value.strip().lower() in SDPA_LONG_OFF or SDPA_LONG_ENGAGED in container_text:
        return []
    return ['%s=%s is set and no engaged line (%s) was logged: the configuration never ran' % (SDPA_LONG_FLAG, value, SDPA_LONG_ENGAGED)]


# tp4/sdpa-multi (sdpa_multi_tp): QWEN_FAST_TP4_SDPA=multi needs ITS engaged line (config=multi, flags=0x21: the attach built the launch),
# the executed path's own line (the first multi call in a forward: a launch that was built and never called proves nothing), and, under
# QWEN_FAST_TP4_SDPA_AUDIT=1, at least one audit line that says exact=True and NONE that says otherwise: the audit compares the multi
# launch's block output with the per-user launches' word for word, so a mismatch line (or an exact=False one) fails the arm.
SDPA_MULTI_NAME = 'multi'
SDPA_MULTI_CALL = '[PINDIAG] tp4 sdpa multi call'
SDPA_AUDIT_FLAG = 'QWEN_FAST_TP4_SDPA_AUDIT'
SDPA_AUDIT_LINE = '[PINDIAG] tp4 sdpa audit'
SDPA_AUDIT_MISMATCH = '[PINDIAG] tp4 sdpa audit MISMATCH'


def sdpa_multi_problems(env, container_text):
    """[problem] for a profile that names QWEN_FAST_TP4_SDPA=multi: the engaged line with config=multi flags=0x21, the call line, and
    (audit flag) every audit line exact with at least one logged, none a MISMATCH."""
    env = env or {}
    if (env.get(SDPA_LONG_FLAG) or '').strip() != SDPA_MULTI_NAME:
        return []
    lines = container_text.splitlines()
    problems = []
    engaged = [line for line in lines if SDPA_LONG_ENGAGED in line]
    if not any('config=%s' % SDPA_MULTI_NAME in line and 'flags=0x21' in line for line in engaged):
        problems.append('%s=multi is set and no engaged line with config=multi flags=0x21 was logged: the one-launch path never attached'
                        % SDPA_LONG_FLAG)
    if SDPA_MULTI_CALL not in container_text:
        problems.append('%s=multi is set and no call line (%s) was logged: the launch was built and never called' % (SDPA_LONG_FLAG, SDPA_MULTI_CALL))
    audits = [line for line in lines if SDPA_AUDIT_LINE in line]
    mismatched = [line.strip()[:200] for line in audits if SDPA_AUDIT_MISMATCH in line or 'exact=False' in line]
    problems += ['the multi SDPA audit found a difference: %s' % line for line in mismatched[:4]]
    if (env.get(SDPA_AUDIT_FLAG) or '').strip() == '1' and not any('exact=True' in line for line in audits if SDPA_AUDIT_MISMATCH not in line):
        problems.append('%s=1 is set and no passing audit line (%s <n> exact=True) was logged: nothing was compared'
                        % (SDPA_AUDIT_FLAG, SDPA_AUDIT_LINE))
    return problems


SLIDE_FLAG = 'QWEN_FAST_TP_KV_SLIDE'
QUAD_FLAG = 'QWEN_FAST_QUAD_DRAFT'
# tp4/next-5: QWEN_FAST_QUAD_DRAFT_BLOCKS=2, the eight-seat quad (two quads of four). The smoke that judges it is concurrent8_steady.
QUAD_BLOCKS_FLAG = 'QWEN_FAST_QUAD_DRAFT_BLOCKS'
QUAD_BLOCKS_VALUE = '2'
QUAD_BLOCK_SLOTS = ('0,1,2,3', '4,5,6,7')
STEADY_EIGHT_TEST = 'concurrent8_steady'
SINGLES_AUDIT_FLAG = 'QWEN_FAST_DRAFT_SINGLES_AUDIT'
DRAFTER_BF16_FLAG = 'QWEN_FAST_DRAFTER_BF16'
DRAFTER_BF16_ENGAGED = '[DRAFTER_BF16] engaged'
DRAFTER_BF8_LEND = re.compile(r'\[PINDIAG\] draft weights lent to .*projections dtype=\S*bf8')
FUSED_FLAG = 'QWEN_FAST_FUSED_COMMIT'
FUSED_INPLACE_FLAG = 'QWEN_FAST_FUSED_COMMIT_INPLACE'
FUSED_LIVE_BANKS_FLAG = 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'
FUSED_AUDIT_FLAG = 'QWEN_FAST_FUSED_COMMIT_AUDIT'
FUSED_PREFIXES = 16          # one slide trace per (segment, accepted prefix 1..rows_per_user) in place
FUSED_LINES = ('[PINDIAG] fused commit engaged', '[PINDIAG] fused commit refused', '[PACKED-FUSED] round=',
               '[PACKED-FUSED-AUDIT] round=')
# tp4/tpub (verifier_engine_tp): the sequential step's carry copies as traces. The engaged line is logged once per request engine; a declined line
# fails (the timing is then not the lever's); the audit's lines say how many tensors were compared on every chip (48 layers x 5 tensors x chips).
TPUB_FLAG = 'QWEN_FAST_TP4_TRACED_PUBLISH'
TPUB_AUDIT_FLAG = 'QWEN_FAST_TP4_TRACED_PUBLISH_AUDIT'
TPUB_ENGAGED = '[TPUB] carry traces engaged'
TPUB_DECLINED = '[TPUB] carry traces declined'
TPUB_AUDIT_LINE = re.compile(r'\[TPUB-AUDIT\] op=(save|restore) checked=(\d+) mismatches=(\d+)')
TPUB_LAYERS, TPUB_TENSORS_PER_LAYER = 48, 5
# tp4/lookup (prompt_lookup, serving_fast_request.prepare): QWEN_FAST_LOOKUP_DRAFT names a policy (n<N>m<M>); the engaged line is logged once per
# request, and one round line per user per round says which source proposed it and what the round committed.
LOOKUP_FLAG = 'QWEN_FAST_LOOKUP_DRAFT'
LOOKUP_ENGAGED = '[LOOKUP-DRAFT] engaged'
LOOKUP_ROUND = re.compile(r'\[LOOKUP-ROUND\] request=(\S+) position=(\d+) source=(lookup|dflash2) match=(\d+) offered=(\d+) proposed=(\d+) committed=(\d+)')
STEADY_TEST = 'concurrent4_steady'
RESEND_TEST = 'steady_resend'
# One line per packed round the coordinator selected (dflash_packed_proposal_coordinator.SELECT_LINE, QWEN_FAST_PACKED_AUDIT):
# the quad's one group of four slots, or two packed pairs.
QUAD_ROUND = re.compile(r'\[PACKED-SELECT\] round=\d+ pairs=\[\[0, 1, 2, 3\]\] users=4 ')
# Both quads in one round: eight users, one select line (the quad's group of four slots, twice).
QUADS_ROUND = re.compile(r'\[PACKED-SELECT\] round=\d+ pairs=\[\[0, 1, 2, 3\], \[4, 5, 6, 7\]\] users=8 ')
QUAD_MARKER_SLOTS = re.compile(r'\[PINDIAG\] quad draft engaged slots=\[([0-9,]*)\]')
PAIR_ROUND = re.compile(r'\[PACKED-SELECT\] round=\d+ pairs=\[\[0, 1\], \[2, 3\]\] users=4 ')
QUAD_MARKER = '[PINDIAG] quad draft engaged'
QUAD_DISABLED = '[PINDIAG] quad draft disabled'
QUAD_FALLBACK = '[QUAD-DRAFT] fallback'
QUAD_LINE = re.compile(r'\[QUAD-DRAFT\] round=\d+ built=')
QUAD_AUDIT = re.compile(r'\[QUAD-AUDIT\] round=\S+ equal=([01]) ')
SINGLES_GROUP_EQUAL = re.compile(r'\[DRAFT-SINGLES-AUDIT\] round=\S+ group=\[([0-9, ]*)\] equal=1 ')
# A draft build that borrowed the pool's pre-trace mask / output set (dflash_proposal_trace.POOLED_*_LINE) and one the pool could not serve.
POOLED_MASK_SLOTS = re.compile(r'\[PINDIAG\] draft mask pooled slots=\[([0-9,]*)\]')
POOLED_OUTPUT_SLOTS = re.compile(r'\[PINDIAG\] draft outputs pooled slots=\[([0-9,]*)\]')
POOLED_REFUSED = ('[PINDIAG] draft mask pooled refused', '[PINDIAG] draft outputs pooled refused')
SINGLES_AUDIT_LINE = re.compile(r'\[DRAFT-SINGLES-AUDIT\] round=\S+ group=\[[0-9, ]*\] equal=([01]) stage=(\S+) ')
# A code prompt asked to be explained and rewritten (800 or 1500 tokens out) does not end by itself in a few tokens.
MIN_ANSWER_TOKENS = 16
TEXT_TESTS = ('coding', STEADY_TEST, RESEND_TEST, 'concurrent4_code', 'concurrent4_code_equal', 'concurrent4_code_32k', 'concurrent8_code') + EIGHT_TESTS
STREAM_PARSER_TESTS = ('stream_tool_call', 'stream_reasoning')
EXTENT_FLAG = 'QWEN_FAST_EXTENT_REPLAY'
GATE_PROFILE_FLAG = 'QWEN_C2_GATE_PROFILE'
ADMISSION_PASSED = '[PINDIAG] packed-any admission passed:'
ADMISSION_UNQUALIFIED = 'packed-any admission passed UNQUALIFIED'
WAIVER_FLAG = 'QWEN_FAST_262K_EVIDENCE_WAIVER'
WAIVER_MARKER = '[PINDIAG] 262k evidence WAIVED (gate-only)'   # page_width_tp4.WAIVER_MARKER
WAIVER_CAPACITY = 'capacity=262144'
WAIVER_STAMP = 'UNQUALIFIED (262k waiver)'
ADMISSION_REFUSED = '[PINDIAG] packed-any admission refused'
PARSER_ARMED = 'parser M armed'
FOREIGN_SHARE = 0.3
FIRST_PREFILL_ITEM = re.compile(r'\[MEMLEDGER\] phase=prefill point=after \S+ item=model_after_prefill chip0=\S+ .*?buffers=(\d+) ')
# Four cards: the eager prefill is warmed before the packed traces (serving_runtime.prefill_warm_before_traces) and every prefill
# reports the programs it compiled (serving_runtime.prefill_tripwire). Metal prints its warning once per process, at the first
# allocation made with a trace live, so it is an order marker and not a count.
WARM_LINE = '[PINDIAG] four-card eager prefill warmed before the packed traces'
UNSAFE_ALLOCATION = 'Allocating device buffers is unsafe due to the existence of an active trace'
PREFILL_PROGRAMS = re.compile(r'\[PINDIAG\] four-card prefill programs=(\d+|None)->(\d+|None) window=(\d+) prompt=(\d+)')
ENGINE_PROGRAMS = re.compile(r'\[PINDIAG\] four-card engine programs=(\d+|None)->(\d+|None) ')
# QWEN_FAST_M3_REQUEST_WARM=1 or even (the tp4/warm4 profiles): the request widths are warmed before the one block's capture. The rule is
# on the warm itself (the line exists, precedes the capture anchor and compiled something). The anchor is block 0's capture line when
# the log has one (two-block attaches), else the first 'verify t1 engaged site=packed_verify' line, which is logged right after the
# single block's capture. The first engine build's program delta is RECORDED, not bounded: that build also constructs the per-request
# drafter, which the warm does not cover. Read the warm arm against its control.
REQUEST_WARM_LINE = re.compile(r'\[PINDIAG\] request widths warmed before the packed traces: '
                               r'rows=\((?:1, 2, 4|1, 2, 4, 1)\) programs=(\d+|None)->(\d+|None)')
BLOCK0_CAPTURE_LINE = '[PINDIAG] packed blocks capture block=0'
BLOCK_CAPTURE_ANCHOR = '[PINDIAG] verify t1 engaged site=packed_verify'
DEFAULT_PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'


def smoke_results(text):
    """The dict of the last SMOKE_JSON line of a smoke log, or None."""
    found = None
    for line in text.splitlines():
        marker = line.find('SMOKE_JSON ')
        if marker >= 0:
            try:
                found = json.loads(line[marker + len('SMOKE_JSON '):])
            except ValueError:
                pass
    return found


def garbage(text):
    """A reason the kept text of a stream is not an answer, or None."""
    if not text or not text.strip():
        return 'empty text'
    sample = text[:300]
    printable = sum(1 for char in sample if char.isprintable() or char in '\n\t')
    if printable < 0.9 * len(sample):
        return 'mostly non-printable'
    if len(sample) >= 100 and len(set(sample)) <= 4:
        return 'a repetition of %d characters' % len(set(sample))
    return None


def foreign_script(text):
    """The share of a sample's letters outside Latin script (code and English are Latin), or 0.0 when it has none."""
    sample = [char for char in text[:300] if char.isalpha()]
    if not sample:
        return 0.0
    return sum(1 for char in sample if ord(char) > 0x2FF) / len(sample)


def answer_problems(name, stream):
    """Why a code-prompt stream is not an answer: an instant or early stop, or mostly foreign-script text."""
    if not isinstance(stream, dict) or 'error' in stream:
        return []
    problems = []
    tokens = stream.get('tokens')
    if stream.get('finish') == 'stop' and isinstance(tokens, int) and tokens < MIN_ANSWER_TOKENS:
        problems.append('%s: finished with stop after %d token(s), under %d (an instant EOS is not an answer to a code prompt)'
                        % (name, tokens, MIN_ANSWER_TOKENS))
    share = foreign_script(stream['text']) if isinstance(stream.get('text'), str) else 0.0
    if share > FOREIGN_SHARE:
        problems.append('%s: %.0f%% of the letters are outside Latin script (garbage, not an answer)' % (name, 100 * share))
    return problems


def stream_problems(name, stream, text_needed=True):
    problems = []
    if not isinstance(stream, dict):
        return ['%s: no result' % name]
    if 'error' in stream:
        return ['%s: %s' % (name, stream['error'])]
    if not stream.get('tokens'):
        problems.append('%s: no tokens' % name)
    if stream.get('finish') is None and 'text' in stream:
        problems.append('%s: no finish reason' % name)
    if text_needed and 'text' in stream:
        reason = garbage(stream['text'])
        if reason:
            problems.append('%s: garbage text (%s)' % (name, reason))
    return problems


def packed_segments(container_text):
    """{request id: (segment, position of its first [PACKED] line)}: the segment the packed block served each request in."""
    found = {}
    for request, segment, position in PACKED_LINE.findall(container_text):
        found.setdefault(request, (int(segment), int(position)))
    return found


def segment_of(prompt_tokens, container_text):
    """The packed segment a user ran in, as text: its request's first [PACKED] line is at the prompt's length (or one past it,
    the seed token), and a prompt length that one request matches names one segment. Otherwise 'unread' (no such line or the
    audit off) or the candidates, when several requests of that length ran in different segments."""
    if not isinstance(prompt_tokens, int):
        return 'unread'
    segments = sorted({segment for segment, position in packed_segments(container_text).values()
                       if position in (prompt_tokens, prompt_tokens + 1)})
    return ', '.join(str(segment) for segment in segments) if segments else 'unread'


def solo_problems(results, container_text=''):
    """A concurrent4 user whose full answer differs from its solo run's. The answer is its content hash, its reasoning hash,
    its completion tokens and its finish reason (all four must be equal: the same greedy tokens split into the two fields
    equally). Skipped when concurrent4_solo did not run, errored, or was recorded without hashes."""
    solo = results.get(SOLO_TEST)
    if not isinstance(solo, dict) or 'error' in solo:
        return []
    references = solo.get('users') or []
    problems = []
    for name in SOLO_COMPARED:
        entry = results.get(name)
        if not isinstance(entry, dict) or 'error' in entry:
            continue
        for index, user in enumerate(entry.get('users') or []):
            reference = references[index] if index < len(references) else None
            if not isinstance(user, dict) or not isinstance(reference, dict) or 'error' in user or 'error' in reference:
                continue
            if 'content_sha256' not in user or 'content_sha256' not in reference:
                continue
            differing = [label for key, label in SOLO_FIELDS if user.get(key) != reference.get(key)]
            if differing:
                problems.append('%s user %d: diverged from solo (segment %s): %s differ (%s tokens against %s solo)'
                                % (name, index, segment_of(user.get('prompt_tokens'), container_text), ', '.join(differing),
                                   user.get('completion_tokens'), reference.get('completion_tokens')))
    return problems


def smoke_problems(results, container_text=''):
    problems = []
    if results is None:
        return ['no SMOKE_JSON line in the smoke log']
    for name in CORE:
        entry = results.get(name)
        if entry is None:
            continue
        if 'error' in entry:
            problems.append('%s: %s' % (name, entry['error']))
    if 'warmup' in results and 'error' not in results['warmup'] and results['warmup'].get('value') != 200:
        problems.append('warmup: status %s' % results['warmup'].get('value'))
    for name in ('coding', RESEND_TEST):
        if name in results and 'error' not in results[name]:
            problems += stream_problems(name, results[name])
            problems += answer_problems(name, results[name])
    for name in CONCURRENT_TESTS + (SOLO_TEST,):
        if name in results and 'error' not in results[name]:
            for index, user in enumerate(results[name].get('users') or []):
                problems += stream_problems('%s user %d' % (name, index), user)
                if name in TEXT_TESTS:
                    problems += answer_problems('%s user %d' % (name, index), user)
    stall = results.get(STALL_TEST)
    if isinstance(stall, dict) and 'error' not in stall:
        for index, user in enumerate(stall.get('users') or []):
            problems += stream_problems('%s user %d' % (STALL_TEST, index), user)
        if not isinstance(stall.get('arrival_ttft_s'), (int, float)):
            problems.append('%s: the cold arrival has no time to first token' % STALL_TEST)
        if not stall.get('seat_gaps'):
            problems.append('%s: no seat gap was recorded' % STALL_TEST)
    elif isinstance(stall, dict):
        problems.append('%s: %s' % (STALL_TEST, stall['error']))
    for replay in REPLAY_TESTS:
        if replay in results and 'error' not in results[replay]:
            for index, user in enumerate(results[replay].get('users') or []):
                name = '%s user %d' % (replay, index)
                if 'error' in user:
                    problems.append('%s: %s' % (name, user['error']))
                elif user.get('status') != 200:
                    problems.append('%s: status %s' % (name, user.get('status')))
                elif not user.get('tokens'):
                    problems.append('%s: no tokens' % name)
                elif user.get('finish') is None:
                    problems.append('%s: no finish reason' % name)
    problems += solo_problems(results, container_text)
    for name in STREAM_PARSER_TESTS:
        entry = results.get(name)
        if entry is None:
            continue
        if 'error' in entry:
            problems.append('%s: %s' % (name, entry['error']))
        elif entry.get('ok') is not True:
            problems += ['%s: %s' % (name, reason) for reason in (entry.get('problems') or ['not ok'])[:4]]
    if 'warm_lifecycle' in results and 'error' not in results['warm_lifecycle']:
        for label, rows in results['warm_lifecycle'].items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not row.get('tokens'):
                    problems.append('warm_lifecycle %s at %s characters: no tokens' % (label, row.get('chars')))
    return problems


def ramp_kv_median(log_text):
    """(median of each round's largest prepare_history entry, rounds counted), or (None, 0)."""
    largest = []
    for match in PUBLISH.finditer(log_text):
        values = [float(part) for part in match.group(1).replace(' ', '').split(',') if part]
        if any(values):
            largest.append(max(values))
    return (statistics.median(largest), len(largest)) if largest else (None, 0)


def slide_on(profile, profiles_path):
    document = json.loads(Path(profiles_path).read_text(encoding='utf-8'))
    entry = (document.get('profiles') or {}).get(profile)
    if entry is None:
        raise ValueError('profile %s is not in %s' % (profile, profiles_path))
    return (entry.get('env') or {}).get(SLIDE_FLAG) == '1'


def profile_env(profile, profiles_path):
    document = json.loads(Path(profiles_path).read_text(encoding='utf-8'))
    entry = (document.get('profiles') or {}).get(profile)
    if entry is None:
        raise ValueError('profile %s is not in %s' % (profile, profiles_path))
    return entry.get('env') or {}


def profile_entry(profile, profiles_path):
    document = json.loads(Path(profiles_path).read_text(encoding='utf-8'))
    entry = (document.get('profiles') or {}).get(profile)
    if entry is None:
        raise ValueError('profile %s is not in %s' % (profile, profiles_path))
    return entry


def first_prefill_buffers(container_text):
    """Buffers the model held after the FIRST prefill of the container log ([MEMLEDGER] item=model_after_prefill), or None."""
    match = FIRST_PREFILL_ITEM.search(container_text)
    return int(match.group(1)) if match else None


def late_program_problems(container_text):
    """(problems, facts) for the four-card prefill tripwire: the warm line must exist and precede Metal's 'unsafe allocation'
    warning (the packed traces' first allocation), and no prefill may compile a program beyond its window snapshot's
    (B - A - W above zero is the #48536 sequence: a program compiled after the traces were captured)."""
    problems, lines = [], container_text.splitlines()
    warm = next((i for i, line in enumerate(lines) if WARM_LINE in line), None)
    unsafe = next((i for i, line in enumerate(lines) if UNSAFE_ALLOCATION in line), None)
    if warm is None:
        problems.append('the four-card eager prefill warm never ran before the packed traces (no "%s" line): every prefill '
                        'would compile after the capture' % WARM_LINE)
    elif unsafe is not None and unsafe < warm:
        problems.append('the eager prefill warm (log line %d) came after the first allocation made with a trace live '
                        '(line %d, Metal "%s")' % (warm + 1, unsafe + 1, UNSAFE_ALLOCATION))
    prefills = [m.groups() for m in map(PREFILL_PROGRAMS.search, lines) if m]
    late = []
    for before, after, window, prompt in prefills:
        if before == 'None' or after == 'None':
            continue
        extra = int(after) - int(before) - int(window)
        if extra > 0:
            late.append((int(prompt), extra))
    problems += ['the prefill of %d tokens compiled %d program(s) beyond its window snapshot after the packed traces were '
                 'captured ([PINDIAG] four-card prefill programs=): the warm did not cover its shape' % item for item in late[:4]]
    if warm is not None and not prefills:
        problems.append('no "[PINDIAG] four-card prefill programs=" line: the prefill tripwire did not run')
    engines = [(before, after) for before, after in (m.groups() for m in map(ENGINE_PROGRAMS.search, lines) if m)
               if before != 'None' and after != 'None']
    return problems, dict(prefill_warm_line=None if warm is None else warm + 1, unsafe_allocation_line=None if unsafe is None else unsafe + 1,
                          prefills_counted=len(prefills), prefills_with_late_programs=len(late),
                          prefill_window_programs=sum(int(window) for _, _, window, _ in prefills),
                          engines_counted=len(engines), engine_programs=sum(int(after) - int(before) for before, after in engines))


def request_warm_problems(container_text, flag='1'):
    """(problems, facts) under QWEN_FAST_M3_REQUEST_WARM in ('1', 'even'): the warm line must exist and precede the capture anchor,
    and the warm must have compiled something (B - A of its programs=A->B above 0). The first four-card engine build's delta is
    recorded as a fact (it includes the drafter, so it is not bounded)."""
    problems, lines = [], container_text.splitlines()
    warm = next((i for i, line in enumerate(lines) if REQUEST_WARM_LINE.search(line)), None)
    anchor_text = next((text for text in (BLOCK0_CAPTURE_LINE, BLOCK_CAPTURE_ANCHOR)
                        if any(text in line for line in lines)), None)
    capture = None if anchor_text is None else next(i for i, line in enumerate(lines) if anchor_text in line)
    if warm is None:
        problems.append('QWEN_FAST_M3_REQUEST_WARM=%s but no "[PINDIAG] request widths warmed before the packed traces" line: the warm '
                        'never ran' % flag)
    elif capture is None:
        problems.append('no "%s" or "%s" line to order the request warm against' % (BLOCK0_CAPTURE_LINE, BLOCK_CAPTURE_ANCHOR))
    elif warm > capture:
        problems.append('the request warm (log line %d) came after the packed block captured its trace (line %d)' % (warm + 1, capture + 1))
    engines = [(int(before), int(after)) for before, after in (m.groups() for m in map(ENGINE_PROGRAMS.search, lines) if m)
               if before != 'None' and after != 'None']
    first = None if not engines else engines[0][1] - engines[0][0]
    warm_programs = warm_ms = None
    if warm is not None:
        counts = REQUEST_WARM_LINE.search(lines[warm]).groups()
        timing = re.search(r' ms=(\d+)', lines[warm])
        warm_ms = None if timing is None else int(timing.group(1))
        if 'None' not in counts:
            warm_programs = int(counts[1]) - int(counts[0])
            if warm_programs <= 0:
                problems.append('the request warm compiled %d programs: it did not run the request widths (or the program count is '
                                'not live)' % warm_programs)
    return problems, dict(request_warm_programs=warm_programs, request_warm_ms=warm_ms,
                          request_warm_line=None if warm is None else warm + 1,
                          capture_anchor_line=None if capture is None else capture + 1, first_engine_programs=first)


def draft_facts(container_text):
    """What the container log says about the batched draft: rounds served by the quad and by two packed pairs, the quad's marker,
    fallback and disable lines, and the two audits' lines."""
    singles = SINGLES_AUDIT_LINE.findall(container_text)
    quad_audit = QUAD_AUDIT.findall(container_text)
    return dict(quad_rounds=len(QUAD_ROUND.findall(container_text)),
                quads_rounds=len(QUADS_ROUND.findall(container_text)),
                quad_marker_slots=sorted(QUAD_MARKER_SLOTS.findall(container_text)),
                pair_rounds=len(PAIR_ROUND.findall(container_text)),
                quad_markers=container_text.count(QUAD_MARKER), quad_disabled=container_text.count(QUAD_DISABLED),
                quad_fallbacks=container_text.count(QUAD_FALLBACK), quad_lines=len(QUAD_LINE.findall(container_text)),
                quad_audits=len(quad_audit), quad_audits_unequal=sum(1 for equal in quad_audit if equal != '1'),
                singles_groups_equal=sorted({''.join(group.split()) for group in SINGLES_GROUP_EQUAL.findall(container_text)}),
                pooled_mask_slots=sorted(set(POOLED_MASK_SLOTS.findall(container_text))),
                pooled_output_slots=sorted(set(POOLED_OUTPUT_SLOTS.findall(container_text))),
                pooled_refused=sum(container_text.count(line) for line in POOLED_REFUSED),
                singles_audits=len(singles), singles_audits_unequal=sum(1 for equal, _ in singles if equal != '1'),
                singles_audit_stages=sorted({stage for equal, stage in singles if equal != '1'})[:4])


FAST_PATH_KEYS = ('QWEN_FAST_ANY_REQUEST', 'QWEN_FAST_EXTENT_REPLAY', 'QWEN_FAST_TP')


def drafter_bf16_problems(env, container_text):
    """The problems the profile's QWEN_FAST_DRAFTER_BF16 setting leaves. Off: no '[DRAFTER_BF16]' line at all. On (the value 1): at least one engaged
    line (it is printed once per engine process) and NO draft-weights lend line naming a bf8 projection dtype (a bf8 lend line under the flag means an
    upload did not take it; with every projection bf16 the lend line carries no dtype)."""
    on = (env or {}).get(DRAFTER_BF16_FLAG) == '1'
    engaged = container_text.count(DRAFTER_BF16_ENGAGED)
    if not on:
        if '[DRAFTER_BF16]' in container_text:
            return ['%s is not 1 and the log holds a [DRAFTER_BF16] line: the bf16 drafter ran on a profile without it' % DRAFTER_BF16_FLAG]
        return []
    problems = []
    if not engaged:
        problems.append('%s=1 and no "%s" line was logged: the drafter projections never took the flag' % (DRAFTER_BF16_FLAG, DRAFTER_BF16_ENGAGED))
    bf8 = [line.strip()[:200] for line in container_text.splitlines() if DRAFTER_BF8_LEND.search(line)]
    problems += ['%s=1 and a draft-weights lend line still reports bf8 projections: %s' % (DRAFTER_BF16_FLAG, line) for line in bf8[:4]]
    return problems


def fast_path(env):
    """Whether the profile drafts: it serves the speculative fast path (S2) or names a batched-draft flag. The G1
    profiles (general-*) set none of these keys."""
    return (any(env.get(key) not in (None, '', '0') for key in FAST_PATH_KEYS)
            or any(key in env for key in (QUAD_FLAG, QUAD_BLOCKS_FLAG, SINGLES_AUDIT_FLAG)))


def blocks_problems(facts, env, steady_eight):
    """QWEN_FAST_QUAD_DRAFT_BLOCKS: what a profile that asks for the eight-seat quad must show. A value other than '2' fails whatever
    ran (the coordinator refuses it by name). Held to the log only by a smoke that ran concurrent8_steady (eight users past the
    2,048-row draft window together, the only mix both quads can serve): exactly two engaged markers, one for each block of slots, at
    least one round both quads served, no fallback and no disable line."""
    problems = []
    value = env.get(QUAD_BLOCKS_FLAG, '')
    if value in ('', '0'):
        return problems
    if value != QUAD_BLOCKS_VALUE:
        return ['%s=%r: only %s is served' % (QUAD_BLOCKS_FLAG, value, QUAD_BLOCKS_VALUE)]
    # The image sets QWEN_FAST_QUAD_DRAFT=1, so a profile that does not name it relies on that default: only an explicit other value refuses.
    if env.get(QUAD_FLAG, '1') != '1':
        problems.append('%s=%s without %s=1: no quad can form' % (QUAD_BLOCKS_FLAG, value, QUAD_FLAG))
    if not steady_eight:
        problems.append('the profile asks for the eight-seat quad and the smoke did not run %s (or it errored): the blocks were not judged'
                        % STEADY_EIGHT_TEST)
        return problems
    if facts['quad_markers'] != 2 or list(facts['quad_marker_slots']) != sorted(QUAD_BLOCK_SLOTS):
        problems.append('the quad marker (%s) appears %d times with slots %s, not once for each of %s: a block never engaged or '
                        'engaged twice' % (QUAD_MARKER, facts['quad_markers'], facts['quad_marker_slots'],
                                           ' and '.join(QUAD_BLOCK_SLOTS)))
    if not facts['quads_rounds']:
        problems.append('no round was served by both quads (no [PACKED-SELECT] pairs=[[0, 1, 2, 3], [4, 5, 6, 7]] users=8 line): the '
                        'eight steady users drafted some other way')
    if facts['quad_fallbacks'] or facts['quad_disabled']:
        problems.append('a quad fell back %d times and was disabled %d times' % (facts['quad_fallbacks'], facts['quad_disabled']))
    # The pre-capture buffers of both quads: a quad that built without them uploaded its own mask and read its own outputs after the
    # request traces existed (the post-capture allocation hazard of v86), which no text comparison can see.
    for label, key in (('mask', 'pooled_mask_slots'), ('outputs', 'pooled_output_slots')):
        missing = [slots for slots in QUAD_BLOCK_SLOTS if slots not in facts.get(key, ())]
        if missing:
            problems.append('no "[PINDIAG] draft %s pooled slots=[...]" line for the quad over slots %s: it built without the pool pre-trace '
                            '%s' % (label, ' and '.join(missing), label))
    if facts.get('pooled_refused'):
        problems.append('%d "pooled refused" lines: the pool could not serve a draft mask or output set' % facts['pooled_refused'])
    # With the singles audit on, each block's quad must have been audited, not any one line.
    if env.get(SINGLES_AUDIT_FLAG) not in (None, '', '0'):
        missing = [slots for slots in QUAD_BLOCK_SLOTS if slots not in facts.get('singles_groups_equal', ())]
        if missing:
            problems.append('%s is set and no [DRAFT-SINGLES-AUDIT] group=[%s] equal=1 line was logged for the quad over slots %s'
                            % (SINGLES_AUDIT_FLAG, '...', ' and '.join(missing)))
    return problems


def draft_problems(facts, env, steady, steady_eight=False):
    """The problems the profile's batched-draft settings leave: see the module docstring. Only a smoke that ran the steady
    four-user mix can be held to it; the audits' unequal lines fail whatever ran. With QWEN_FAST_QUAD_DRAFT_BLOCKS set the quad
    rules are blocks_problems' (the eight-user mix), not the four-user ones."""
    problems = []
    if facts['quad_audits_unequal']:
        problems.append('%d [QUAD-AUDIT] lines with equal=0: the quad differs from the pair traces' % facts['quad_audits_unequal'])
    if facts['singles_audits_unequal']:
        problems.append('%d [DRAFT-SINGLES-AUDIT] lines with equal=0 (%s): a batched draft differs from the single-user draft'
                        % (facts['singles_audits_unequal'], ', '.join(facts['singles_audit_stages'])))
    if env.get(QUAD_BLOCKS_FLAG, '') not in ('', '0'):
        # blocks_problems holds the singles audit to one equal=1 line per quad (more than the any-line rule below).
        problems += blocks_problems(facts, env, steady_eight)
        return problems
    if not steady:
        return problems
    if env.get(QUAD_FLAG) == '1':
        if facts['quad_markers'] != 1:
            problems.append('the quad marker (%s) appears %d times, not once: the quad never engaged or engaged twice'
                            % (QUAD_MARKER, facts['quad_markers']))
        if not facts['quad_rounds']:
            problems.append('no round was served by the quad (no [PACKED-SELECT] pairs=[[0, 1, 2, 3]] line): the four '
                            'steady users drafted some other way')
        if facts['quad_fallbacks'] or facts['quad_disabled']:
            problems.append('the quad fell back %d times and was disabled %d times' % (facts['quad_fallbacks'],
                                                                                        facts['quad_disabled']))
    else:
        if facts['quad_markers'] or facts['quad_lines'] or facts['quad_rounds']:
            problems.append('the quad ran (marker %d, round lines %d, rounds %d) on a profile with %s off'
                            % (facts['quad_markers'], facts['quad_lines'], facts['quad_rounds'], QUAD_FLAG))
        if not facts['pair_rounds']:
            problems.append('no round was served by two packed pairs: the comparison arm never batched a draft')
    if env.get(SINGLES_AUDIT_FLAG) not in (None, '', '0') and not facts['singles_audits']:
        problems.append('%s is set and no [DRAFT-SINGLES-AUDIT] line was logged' % SINGLES_AUDIT_FLAG)
    return problems


def traffic_problems(container_text, entry):
    """What a TRAFFIC profile's boot must show (`entry`: the profile's whole record). A profile that is gate only, or that
    does not serve the extent replay, is not judged. The attach's packed-any admission must have PASSED - on a record that
    qualifies these bytes, so neither the UNQUALIFIED waiver (which only a gate-only profile is given) nor a refusal line -
    and a parser_rechunk profile must have armed parser M."""
    env = entry.get('env') or {}
    if entry.get('gate_only') or env.get(GATE_PROFILE_FLAG) or env.get(EXTENT_FLAG) != '1':
        return []
    problems = []
    if ADMISSION_PASSED not in container_text:
        problems.append('no "%s" line in the container log: the traffic profile was not admitted by the packed-any '
                        'admission' % ADMISSION_PASSED)
    if ADMISSION_UNQUALIFIED in container_text:
        problems.append('the packed-any admission passed UNQUALIFIED on a traffic profile: the four-card record does not '
                        'qualify these bytes')
    if ADMISSION_REFUSED in container_text:
        problems.append('a "%s" line in the container log' % ADMISSION_REFUSED)
    if entry.get('parser_rechunk') and PARSER_ARMED not in container_text:
        problems.append('the profile sets parser_rechunk but the container log has no "%s" line' % PARSER_ARMED)
    return problems


def waiver_active_in_log(container_text):
    """Whether the container log shows the 262k evidence waiver (its loud line or the waived admission's own text)."""
    return WAIVER_MARKER in container_text or '(262k waiver, gate only)' in container_text


def waiver_problems(container_text, entry):
    """The 262k evidence waiver, judged from the container log. A traffic (non gate-only) profile must show none of it, so a leak
    stands out even if a code-level guard changes later. A gate-only profile that sets QWEN_FAST_262K_EVIDENCE_WAIVER must show the
    loud line exactly once and an admission that passed UNQUALIFIED at capacity 262144 (and one that does not set it, no waiver)."""
    env = entry.get('env') or {}
    flagged = env.get(WAIVER_FLAG) not in (None, '', '0')
    marker_lines = container_text.count(WAIVER_MARKER)
    waived_pass = [line for line in container_text.splitlines()
                   if ADMISSION_UNQUALIFIED in line and '(262k waiver, gate only)' in line]
    problems = []
    if not entry.get('gate_only'):
        if marker_lines or waived_pass or flagged:
            problems.append('the 262k evidence waiver is active on a traffic profile (%d "%s" lines, %d waived admission lines, flag %s): '
                            'it exists only in a gate run of a gate-only profile' % (marker_lines, WAIVER_MARKER, len(waived_pass),
                                                                                    'set' if flagged else 'unset'))
        return problems
    if not flagged:
        if marker_lines or waived_pass:
            problems.append('the log shows the 262k evidence waiver but the profile does not set %s' % WAIVER_FLAG)
        return problems
    if marker_lines != 1:
        problems.append('"%s" appears %d times, not once (%s is set)' % (WAIVER_MARKER, marker_lines, WAIVER_FLAG))
    if not any(WAIVER_CAPACITY in line for line in waived_pass):
        problems.append('no "packed-any admission passed UNQUALIFIED ... (262k waiver, gate only) %s" line (%s is set)'
                        % (WAIVER_CAPACITY, WAIVER_FLAG))
    return problems


def fused_facts(env, container_text):
    """lever_n_m3native_gate.h1b_summary of the container log (the gate's own reading of the H1b lines), or None when the profile has
    no fused-commit flag on and the log holds none of its lines."""
    import lever_n_m3native_gate as gate

    if env.get(FUSED_FLAG) != '1' and not any(line in container_text for line in FUSED_LINES):
        return None
    return gate.h1b_summary(container_text)


def m3_blocks(env):
    """QWEN_FAST_M3_BLOCKS as the attach reads it: 2 for two 64-row blocks (eight seats), else 1."""
    return 2 if env.get('QWEN_FAST_M3_BLOCKS') == '2' else 1


def fused_problems(env, container_text, steady, facts=None):
    """The problems the profile's fused-commit settings leave: see the module docstring."""
    import lever_n_m3native_gate as gate

    problems = []
    if env.get(FUSED_FLAG) != '1':
        if any(line in container_text for line in FUSED_LINES):
            problems.append('%s is not set and the log holds fused-commit lines: the fused commit ran on a profile without it'
                            % FUSED_FLAG)
        return problems
    report = gate.h1b_report(env, container_text)
    problems += report['problems']
    facts = report if facts is None else facts
    engaged_lines = container_text.count(gate.FUSED_ENGAGED_MARKER)
    # Every 64-row M3 block builds its own fused commit and logs its own engaged line (packed_verifier, once per block): one line a block.
    blocks = m3_blocks(env)
    if engaged_lines != blocks:
        problems.append('the fused commit engaged line (%s) appears %d times, not %s' % (
            gate.FUSED_ENGAGED_MARKER, engaged_lines, 'once' if blocks == 1 else 'once per M3 block (%d blocks)' % blocks))
    engaged = facts.get('engaged')
    inplace = env.get(FUSED_INPLACE_FLAG) == '1'
    for later in list(gate.FUSED_ENGAGED_LINE.finditer(container_text))[1:]:
        wanted = int(later.group(1)) * (1 + FUSED_PREFIXES) if inplace else int(later.group(1))
        if int(later.group(7)) != wanted:
            problems.append('a further fused-commit block captured %d traces, not %d (%s users)' % (int(later.group(7)), wanted, later.group(1)))
    if engaged is not None:
        wanted = engaged['users'] * (1 + FUSED_PREFIXES) if inplace else engaged['users']
        if engaged['traces'] != wanted:
            problems.append('the fused commit captured %d traces, not %d (%d users, %s)' % (
                engaged['traces'], wanted, engaged['users'], 'a T_proj and %d slides each' % FUSED_PREFIXES if inplace
                else 'a T_proj each'))
    if env.get(FUSED_AUDIT_FLAG) == '1':
        chips = int(env.get('QWEN_FAST_TP') or 2)
        expected = 10 * chips * (2 if inplace else 1)    # ten deltas a chip, and in place ten banks a chip
        wrong = sorted({int(match.group(5)) for match in gate.FUSED_AUDIT_LINE.finditer(container_text)
                        if int(match.group(5)) != expected})
        if wrong:
            problems.append('audit lines checked %s items, not %d (10 K/V pieces x %d chips%s)' % (
                wrong, expected, chips, ', deltas and banks' if inplace else ''))
        if not facts.get('audits'):
            problems.append('%s is set and no [PACKED-FUSED-AUDIT] line was logged' % FUSED_AUDIT_FLAG)
    if steady:
        # Under two M3 blocks both log the same round number (each block counts its own rounds): a round of both blocks has eight lines.
        both_blocks = 0
        if blocks > 1:
            rounds = {}
            for match in gate.FUSED_LINE.finditer(container_text):
                rounds.setdefault(match.group(1), []).append(match.group(4))
            both_blocks = sum(1 for paths in rounds.values() if len(paths) == 4 * blocks and all(path == 'fused' for path in paths))
        if not (facts.get('four_fused_rounds') or both_blocks):
            problems.append('no round had all four users on the fused path (%d fused publications, %d today; reasons %s)' % (
                facts.get('fused', 0), facts.get('today', 0), facts.get('today_reasons')))
        if env.get(FUSED_LIVE_BANKS_FLAG) == '1' and not facts.get('live_banks'):
            problems.append('%s is set and the live-bank marker (%s) was never logged' % (FUSED_LIVE_BANKS_FLAG,
                                                                                            gate.FUSED_LIVE_BANKS_MARKER))
    return problems


# tp4/hostgap: the two-block verify pre-stage (verify_prestage), judged on the eight-seat steady mix.
HOSTGAP_TWO_BLOCK_FLAG = 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE'
HOSTGAP_EPOCHS_FLAG = 'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS'
HOSTGAP_AUDIT_FLAG = 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT'
HOSTGAP_WINDOW_VALIDATE_FLAG = 'QWEN_FAST_TP4_WINDOW_VALIDATE'
HOSTGAP_ENGAGED = '[PINDIAG] verify prestage two-block engaged'
HOSTGAP_REFUSED = '[PINDIAG] verify prestage two-block refused'
HOSTGAP_EPOCHS_ENGAGED = '[PINDIAG] verify prestage block epochs engaged'
HOSTGAP_EPOCHS_REFUSED = '[PINDIAG] verify prestage block epochs refused'
HOSTGAP_FULL_AUDIT = re.compile(r'\[PACKED-PRESTAGE-FULLAUDIT\] block=(\S+) round=\d+ path=(diff|full) buffers=(\d+) checked=(\d+) mismatches=(\d+)')
HOSTGAP_VERIFY_LINE = re.compile(r'\[PACKED-HOSTGAP-VERIFY\] block=(\S+) round=\d+ live=(\d+) path=(\w+) reason=(\S+)')
HOSTGAP_LOG_FLAG = 'QWEN_FAST_TP4_HOSTGAP_LOG'
HOSTGAP_DIFF_MIN_SHARE = 0.95
# The only reasons a pre-staged block may legitimately take the full stage: an external writer between the window and the verify
# (a membership or prefill event). Every other 'full' (no-snapshot, epoch:verify, destinations) counts against the share.
HOSTGAP_EXCUSED_REASONS = ('epoch:admission', 'epoch:detach', 'epoch:prefill', 'epoch:prefill-chunk', 'epoch:bookkeeping',
                           'epoch:lane-switch')
HOSTGAP_SHADOW = '[PACKED-PRESTAGE-SHADOW]'
HOSTGAP_PRESTAGE_LINE = re.compile(r'\[PACKED-PRESTAGE\] round=\d+ path=(\w+) buffers=\d+ reason=(\S+) live=(\d+)')
HOSTGAP_LINE_PREFIXES = (HOSTGAP_ENGAGED, HOSTGAP_REFUSED, HOSTGAP_EPOCHS_ENGAGED, HOSTGAP_EPOCHS_REFUSED,
                         '[PACKED-PRESTAGE-FULLAUDIT]')
HOSTGAP_NO_SNAPSHOT_MAX_SHARE = 0.10
HOSTGAP_MIN_LIVE4_VERIFIES = 20


def hostgap_facts(container_text):
    """What the log says about the two-block pre-stage: the 4-live verify paths, the audited blocks and their mismatches."""
    live4 = [(path, reason) for path, reason, live in HOSTGAP_PRESTAGE_LINE.findall(container_text) if live == '4']
    audits = HOSTGAP_FULL_AUDIT.findall(container_text)
    verifies = [(block, path, reason) for block, live, path, reason in HOSTGAP_VERIFY_LINE.findall(container_text) if live == '4']
    return dict(hostgap_engaged=container_text.count(HOSTGAP_ENGAGED), hostgap_refused=container_text.count(HOSTGAP_REFUSED),
                hostgap_epochs_engaged=container_text.count(HOSTGAP_EPOCHS_ENGAGED),
                hostgap_live4_verifies=len(live4),
                hostgap_live4_full_no_snapshot=sum(1 for path, reason in live4 if path == 'full' and reason == 'no-snapshot'),
                hostgap_live4_diff=sum(1 for path, reason in live4 if path == 'diff'),
                hostgap_verify_live4=verifies, hostgap_audit_lines=audits,
                hostgap_full_audits=len(audits), hostgap_full_audit_blocks=sorted({item[0] for item in audits}),
                hostgap_full_audits_full=sum(1 for item in audits if item[1] == 'full'),
                hostgap_audit_mismatches_diff=sum(int(item[4]) for item in audits if item[1] == 'diff'),
                hostgap_audit_mismatches_full=sum(int(item[4]) for item in audits if item[1] == 'full'),
                hostgap_full_audit_mismatches=sum(int(item[4]) for item in audits),
                hostgap_shadow_checks=container_text.count(HOSTGAP_SHADOW))


def hostgap_share_problems(verifies, per_block):
    """The share of the pre-staged blocks' 4-live verifies on the diff path: blocks A and B under the per-block epochs, block A
    alone under the lite arm (B takes the full stage there by design). Only a full stage after an external writer
    (HOSTGAP_EXCUSED_REASONS) is left out of the count; 'no-snapshot', 'epoch:verify' and the rest count as misses.
    At least HOSTGAP_DIFF_MIN_SHARE must be on the diff path."""
    judged = None if per_block else {'A'}
    eligible = [(block, path, reason) for block, path, reason in verifies
                if (judged is None or block in judged) and not (path == 'full' and reason in HOSTGAP_EXCUSED_REASONS)]
    if len(eligible) < HOSTGAP_MIN_LIVE4_VERIFIES:
        return ['only %d 4-live verifies of the pre-staged block(s) were logged (%d needed to judge the diff share)' % (
            len(eligible), HOSTGAP_MIN_LIVE4_VERIFIES)]
    diff = sum(1 for item in eligible if item[1] == 'diff')
    if diff < HOSTGAP_DIFF_MIN_SHARE * len(eligible):
        misses = {}
        for block, path, reason in eligible:
            if path != 'diff':
                key = '%s:%s:%s' % (block, path, reason)
                misses[key] = misses.get(key, 0) + 1
        return ['%d of %d 4-live verifies of the pre-staged block(s) took the diff path (%d%% needed); the rest: %s' % (
            diff, len(eligible), int(HOSTGAP_DIFF_MIN_SHARE * 100), ', '.join('%s x%d' % item for item in sorted(misses.items())))]
    return []


def hostgap_problems(env, container_text, steady_eight):
    """(problems, facts) for the eight-seat host-gap levers: see the module docstring. Judged only where the profile asked for one;
    a profile without QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE logs none of the lever's lines."""
    facts = hostgap_facts(container_text)
    problems = []
    asked = env.get(HOSTGAP_TWO_BLOCK_FLAG) == '1'
    if not asked:
        if any(prefix in container_text for prefix in HOSTGAP_LINE_PREFIXES):
            problems.append('%s is not set and the log holds two-block pre-stage lines: the lever ran on a profile without it'
                            % HOSTGAP_TWO_BLOCK_FLAG)
        if env.get(HOSTGAP_EPOCHS_FLAG) == '1':
            problems.append('%s=1 without %s=1: the per-block epochs need the two-block pre-stage' % (
                HOSTGAP_EPOCHS_FLAG, HOSTGAP_TWO_BLOCK_FLAG))
        return problems, facts
    blocks = m3_blocks(env)
    if facts['hostgap_refused']:
        problems.append('the two-block pre-stage was refused at attach (%d line(s) %s): the arm measures nothing' % (
            facts['hostgap_refused'], HOSTGAP_REFUSED))
    elif facts['hostgap_engaged'] != blocks:
        problems.append('the two-block engaged line (%s) appears %d times, not once per M3 block (%d)' % (
            HOSTGAP_ENGAGED, facts['hostgap_engaged'], blocks))
    per_block = env.get(HOSTGAP_EPOCHS_FLAG) == '1'
    if per_block:
        if HOSTGAP_EPOCHS_REFUSED in container_text:
            problems.append('the per-block epochs were refused or disengaged (%s): the second block was not pre-staged'
                            % HOSTGAP_EPOCHS_REFUSED)
        elif facts['hostgap_epochs_engaged'] != 1:
            problems.append('the block-epochs engaged line (%s) appears %d times, not once' % (
                HOSTGAP_EPOCHS_ENGAGED, facts['hostgap_epochs_engaged']))
    if not steady_eight:
        problems.append('the profile asks for the two-block pre-stage and the smoke did not run %s (or it errored): the lever was '
                        'not judged' % STEADY_EIGHT_TEST)
        return problems, facts
    if not facts['hostgap_live4_diff']:
        problems.append('no 4-live verify took the diff path: the pre-stage never served a verify')
    if env.get(HOSTGAP_LOG_FLAG) == '1':
        problems.extend(hostgap_share_problems(facts['hostgap_verify_live4'], per_block))
    elif per_block:
        problems.append('%s=1 is not set: the per-block epochs arm cannot be judged on the share of verifies that took the diff path'
                        % HOSTGAP_LOG_FLAG)
    if env.get(HOSTGAP_AUDIT_FLAG) == '1':
        wanted = ['A', 'B'] if per_block else ['A']
        if facts['hostgap_audit_mismatches_diff']:
            problems.append('%d staged buffers differed from the full stage after a DIFF write in the full audit '
                            '([PACKED-PRESTAGE-FULLAUDIT] path=diff): the lever staged something other than the full stage'
                            % facts['hostgap_audit_mismatches_diff'])
        if facts['hostgap_audit_mismatches_full']:
            problems.append('%d buffers differed after a FULL stage in the full audit ([PACKED-PRESTAGE-FULLAUDIT] path=full): the '
                            'comparator itself disagrees with the full stage (layout, padding or dtype), so a diff-path mismatch '
                            'in this run says nothing about the lever' % facts['hostgap_audit_mismatches_full'])
        if not facts['hostgap_full_audits_full']:
            problems.append('%s is set and no [PACKED-PRESTAGE-FULLAUDIT] path=full line was logged: the comparator was never '
                            'checked against the full stage' % HOSTGAP_AUDIT_FLAG)
        chips = int(env.get('QWEN_FAST_TP') or 2)
        short = [item for item in facts['hostgap_audit_lines'] if not int(item[2]) or int(item[3]) != int(item[2]) * chips]
        if short:
            problems.append('%d [PACKED-PRESTAGE-FULLAUDIT] lines checked fewer than buffers x %d chips (first: buffers=%s checked=%s)'
                            % (len(short), chips, short[0][2], short[0][3]))
        missing = [label for label in wanted if label not in facts['hostgap_full_audit_blocks']]
        if missing:
            problems.append('%s is set and no [PACKED-PRESTAGE-FULLAUDIT] line was logged for block %s' % (
                HOSTGAP_AUDIT_FLAG, ' and '.join(missing)))
        if env.get(HOSTGAP_WINDOW_VALIDATE_FLAG) == '1' and not facts['hostgap_shadow_checks']:
            problems.append('%s and %s are set and the skipped binding check never ran as a shadow (no %s line)' % (
                HOSTGAP_WINDOW_VALIDATE_FLAG, HOSTGAP_AUDIT_FLAG, HOSTGAP_SHADOW))
    elif facts['hostgap_full_audits']:
        problems.append('[PACKED-PRESTAGE-FULLAUDIT] lines on a profile without %s' % HOSTGAP_AUDIT_FLAG)
    return problems, facts


def tpub_problems(env, container_text):
    """The problems the profile's traced-carry settings leave: with the flag off no [TPUB line at all; on, at least one engaged line, no
    declined line, and (audited) at least one audit line, every one with mismatches=0 and the item count the width implies."""
    on = env.get(TPUB_FLAG) == '1'
    audit_lines = TPUB_AUDIT_LINE.findall(container_text)
    engaged, declined = container_text.count(TPUB_ENGAGED), container_text.count(TPUB_DECLINED)
    if not on:
        if engaged or declined or audit_lines or '[TPUB-AUDIT]' in container_text:
            return ['%s is not set and the log holds traced-carry lines: the traced carry ran on a profile without it' % TPUB_FLAG]
        return []
    problems = []
    if declined:
        problems.append("the traced carry declined %d time(s) (%s): its timing is not the lever" % (declined, TPUB_DECLINED))
    if not engaged and not declined:
        problems.append('%s is set and no engaged line (%s) was logged: the carry copies were never traced' % (TPUB_FLAG, TPUB_ENGAGED))
    chips = int(env.get('QWEN_FAST_TP') or 2)
    if env.get(TPUB_AUDIT_FLAG) == '1':
        if not audit_lines:
            problems.append('%s is set and no [TPUB-AUDIT] line was logged' % TPUB_AUDIT_FLAG)
        elif not any(line[0] == 'restore' for line in audit_lines):
            problems.append('%s is set and no [TPUB-AUDIT] op=restore line was logged: the restore writes the live slot and was never compared'
                            % TPUB_AUDIT_FLAG)
        unequal = [line for line in audit_lines if int(line[2]) != 0]
        if unequal:
            problems.append('%d [TPUB-AUDIT] lines with mismatches>0 (first: op=%s mismatches=%s): a traced carry copy differs from the eager one'
                            % (len(unequal), unequal[0][0], unequal[0][2]))
        expected = TPUB_LAYERS * TPUB_TENSORS_PER_LAYER * chips
        wrong = sorted({int(line[1]) for line in audit_lines if int(line[1]) != expected})
        if wrong:
            problems.append('[TPUB-AUDIT] lines checked %s items, not %d (%d layers x %d tensors x %d chips)' % (
                wrong, expected, TPUB_LAYERS, TPUB_TENSORS_PER_LAYER, chips))
    elif audit_lines:
        problems.append('[TPUB-AUDIT] lines on a profile without %s' % TPUB_AUDIT_FLAG)
    return problems


def sampdraft_problems(container_text, env):
    """The tp4/samp-draft stop conditions (see SAMPDRAFT_LEVERS): fall-backs always; a missing engaged line for a lever the profile
    sets; a missing 'exact=True' audit line for an audit the profile sets."""
    problems = []
    lines = container_text.splitlines()
    for flag, engaged, fell_back, what in SAMPDRAFT_LEVERS:
        fell = [line.strip()[:200] for line in lines if fell_back in line]
        problems += ['the %s fell back (the served path ran, it saved nothing): %s' % (what, line) for line in fell[:4]]
        if env is not None and env.get(flag) == '1' and engaged not in container_text:
            problems.append('%s is set and no engaged line (%s) was logged: the %s never ran' % (flag, engaged, what))
    for flag, marker, what in SAMPDRAFT_AUDITS:
        if env is not None and env.get(flag) == '1' and not any(marker in line and 'exact=True' in line for line in lines):
            problems.append('%s is set and no passing audit line (%s ... exact=True) was logged: the %s was never audited'
                            % (flag, marker, what))
    return problems


def lookup_problems(env, container_text):
    """The problems the profile's lookup setting leaves: with the flag off no [LOOKUP line at all; on, at least one engaged line naming the
    policy and at least one well-formed round line (a user, a source, a proposed count of 0..15 (not 0 for a lookup round) and a committed
    count of 1..16), and every [LOOKUP-ROUND] line must parse."""
    value = (env.get(LOOKUP_FLAG) or '').strip().lower()
    on = value not in ('', '0', 'off')
    rounds = LOOKUP_ROUND.findall(container_text)
    marked = container_text.count('[LOOKUP-ROUND]')
    if not on:
        if container_text.count(LOOKUP_ENGAGED) or marked or '[LOOKUP-DRAFT]' in container_text:
            return ['%s is not set and the log holds lookup lines: the lookup ran on a profile without it' % LOOKUP_FLAG]
        return []
    problems = []
    if not container_text.count(LOOKUP_ENGAGED):
        problems.append('%s=%s and no engaged line (%s) was logged: the lookup was never built' % (LOOKUP_FLAG, value, LOOKUP_ENGAGED))
    elif 'policy=%s' % value not in container_text:
        problems.append('the engaged line does not name the policy of the profile (policy=%s)' % value)
    if not marked:
        problems.append('%s=%s and no [LOOKUP-ROUND] line was logged: no round carried the lookup bookkeeping' % (LOOKUP_FLAG, value))
    elif len(rounds) != marked:
        problems.append('%d [LOOKUP-ROUND] lines, %d of them malformed' % (marked, marked - len(rounds)))
    bad = [line for line in rounds if not (0 <= int(line[5]) <= 15 and 1 <= int(line[6]) <= 16 and (line[2] != 'lookup' or int(line[5]) > 0))]
    if bad:
        problems.append('%d [LOOKUP-ROUND] lines with an impossible count (first: request=%s source=%s proposed=%s committed=%s)'
                        % (len(bad), bad[0][0], bad[0][2], bad[0][5], bad[0][6]))
    return problems


# tp4/u1 (tile_collective_tp): the packed verify's all-reduces as one reduce-scatter on the unit-major view. A profile that sets the flag must log the
# engaged line with at least one unit-major call, no fall-back line (every served call is in the census, so a fall-back is a lever that saved nothing) and
# no audit mismatch; under the audit flag an exact=True audit line must exist for every served shape.
U1_FLAG = 'QWEN_FAST_TP4_RS_UNIT_MAJOR'
U1_AUDIT_FLAG = 'QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT'
U1_ENGAGED = '[PINDIAG] tp4 u1 engaged'
U1_FELL_BACK = '[PINDIAG] tp4 u1 fell back'
U1_AUDIT = '[PINDIAG] tp4 u1 audit '
U1_MISMATCH = '[PINDIAG] tp4 u1 audit mismatch'
U1_SERVED_SHAPES = ('64x5120',)
U1_CHIPS = 4


def u1_audit_problems(env, lines, engaged):
    """The audit must have compared something: every audit line names four chips and some elements; each served shape (the rows= of
    the engaged lines that ran the lever, else U1_SERVED_SHAPES) has an exact line from a replay (round >= 1, not only the warm
    forward's round 0), and at least one block owner per QWEN_FAST_M3_BLOCKS has one."""
    problems = []
    parsed = []
    for line in lines:
        if U1_AUDIT not in line or U1_MISMATCH in line:
            continue
        match = re.search(r' shape=(\d+x\d+) (?:owner=(\S+) )?round=(\d+) calls=(\d+) chips=(\d+) elements=(\d+) exact=True', line)
        if match is None:
            problems.append('an audit line of an unknown form: %s' % line.strip()[:200])
            continue
        shape, owner, round_number, calls, chips, elements = match.groups()
        if int(chips) != U1_CHIPS or int(elements) < 1 or int(calls) < 1:
            problems.append('an audit line compared nothing (need chips=%d, calls and elements above 0): %s' % (U1_CHIPS, line.strip()[:200]))
            continue
        parsed.append((shape, owner, int(round_number)))
    served = sorted({'%sx5120' % match.group(1) for match in (re.search(r' rows=(\d+) .* unit_major=[1-9]', line) for line in engaged) if match})         or list(U1_SERVED_SHAPES)
    for shape in served:
        replays = [owner for found, owner, round_number in parsed if found == shape and round_number >= 1]
        if not replays:
            problems.append('%s is set and no replay audit line (%sshape=%s ... round>=1 ... exact=True) was logged: no replay was compared'
                            % (U1_AUDIT_FLAG, U1_AUDIT, shape))
            continue
        blocks = env.get('QWEN_FAST_M3_BLOCKS', '1')
        wanted = int(blocks) if blocks.isdigit() and int(blocks) > 0 else 1
        if len(set(replays)) < wanted:
            problems.append('%s: shape %s replay audits came from %d block owner(s), %d blocks are served'
                            % (U1_AUDIT_FLAG, shape, len(set(replays)), wanted))
    return problems


def u1_problems(env, container_text):
    lines = container_text.splitlines()
    problems = ['the unit-major all-reduce audit found a difference: %s' % line.strip()[:200] for line in lines if U1_MISMATCH in line][:4]
    problems += ['the unit-major all-reduce fell back (the split ran, it saved nothing): %s' % line.strip()[:200]
                 for line in lines if U1_FELL_BACK in line][:4]
    if env is None or env.get(U1_FLAG) != '1':
        if any(U1_ENGAGED in line or U1_AUDIT in line for line in lines):
            problems.append('unit-major all-reduce lines on a profile without %s' % U1_FLAG)
        return problems
    engaged = [line for line in lines if U1_ENGAGED in line]
    if not engaged:
        problems.append('%s is set and no engaged line (%s) was logged: the unit-major all-reduce never ran' % (U1_FLAG, U1_ENGAGED))
    elif not any(re.search(r' unit_major=[1-9]\d*( |$)', line) for line in engaged):
        problems.append('%s engaged lines report no unit-major call: %s' % (U1_FLAG, engaged[0].strip()[:200]))
    if env.get(U1_AUDIT_FLAG) == '1':
        problems += u1_audit_problems(env, lines, engaged)
    return problems


# tp4/w2 (gdn_conv_gates_spread, F1): the block conv-gates launch with its gate tiles on cores of their own. Three rules: a fall-back line fails
# the smoke (the served launch ran, the timing is not the lever's), a profile that asks for it and logs no engaged line (with at least one gate
# core) fails, and an audit flag with no 'exact=True' audit line (or any mismatch line) fails. tests hold these equal to the module's.
SPREAD_FLAG = 'QWEN_FAST_TP4_CONV_GATES_SPREAD'
SPREAD_AUDIT_FLAG = 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT'
SPREAD_ENGAGED = '[PINDIAG] tp4 conv gates spread engaged'
SPREAD_FELL_BACK = '[PINDIAG] tp4 conv gates spread fell back'
SPREAD_AUDIT = '[PINDIAG] tp4 conv gates spread audit'
SPREAD_MISMATCH = '[PINDIAG] tp4 conv gates spread audit mismatch'


def spread_problems(env, container_text):
    """[problem] for the F1 conv-gates launch: any mismatch or fall-back line; with the flag, the engaged line (gate_cores at least 1) and, with the
    audit flag, a passing audit line (the marker, '<n> exact=True')."""
    lines = container_text.splitlines()
    problems = ['the conv-gates spread audit found a difference: %s' % line.strip()[:200] for line in lines if SPREAD_MISMATCH in line][:4]
    problems += ['the conv-gates spread fell back (the served launch ran, it saved nothing): %s' % line.strip()[:200]
                 for line in lines if SPREAD_FELL_BACK in line][:4]
    if env is None or env.get(SPREAD_FLAG) != '1':
        if any(SPREAD_ENGAGED in line or SPREAD_AUDIT in line for line in lines):
            problems.append('conv-gates spread lines on a profile without %s' % SPREAD_FLAG)
        return problems
    engaged = [line for line in lines if SPREAD_ENGAGED in line]
    if not engaged:
        problems.append('%s is set and no engaged line (%s) was logged: the spread launch never ran' % (SPREAD_FLAG, SPREAD_ENGAGED))
    elif not any(re.search(r' gate_cores=[1-9]\d*( |$)', line) for line in engaged):
        problems.append('%s engaged lines report no gate core: %s' % (SPREAD_FLAG, engaged[0].strip()[:200]))
    if env.get(SPREAD_AUDIT_FLAG) == '1' and not any(
            SPREAD_AUDIT in line and 'exact=True' in line and SPREAD_MISMATCH not in line for line in lines):
        problems.append('%s=1 is set and no passing audit line (%s <n> exact=True) was logged: nothing was compared' % (SPREAD_AUDIT_FLAG, SPREAD_AUDIT))
    return problems


def check(smoke_text, container_text, slide, max_ramp_ms=50.0, env=None, entry=None):
    """(problems, facts) for a smoke log and a container log. `env` (the served profile's) adds the batched-draft
    stop conditions, `entry` (its whole record) the traffic profile's admission and parser conditions."""
    smoke = smoke_results(smoke_text)
    problems = smoke_problems(smoke, container_text)
    mismatches = [line.strip()[:200] for line in container_text.splitlines() if MISMATCH.search(line)]
    problems += ['audit mismatch in the container log: %s' % line for line in mismatches[:4]]
    fell = [line.strip()[:200] for line in container_text.splitlines() if VGLUE_FELL_BACK in line]
    problems += ['a vglue lever fell back (its served path ran, it saved nothing): %s' % line for line in fell[:4]]
    wide_fell = [line.strip()[:200] for line in container_text.splitlines() if DRAFT_WIDE_FELL_BACK in line]
    problems += ['a drafter norm fell back from the wide grid (the plain call ran, it saved nothing): %s' % line for line in wide_fell[:4]]
    if env is not None and env.get(DRAFT_WIDE_FLAG) == '1' and DRAFT_WIDE_ENGAGED not in container_text:
        problems.append('%s is set and no engaged line (%s) was logged: the wide norms never ran' % (DRAFT_WIDE_FLAG, DRAFT_WIDE_ENGAGED))
    slice_fell = [line.strip()[:200] for line in container_text.splitlines() if PAIR_SLICE_FELL_BACK in line]
    problems += ['the GDN pair slice fell back (the plain slices ran, it saved nothing): %s' % line for line in slice_fell[:4]]
    if (env is not None and env.get(PAIR_SLICE_FLAG) == '1' and env.get('QWEN_FAST_TP4_GDN_GLUE') != '1'
            and PAIR_SLICE_ENGAGED not in container_text):
        problems.append('%s is set and no engaged line (%s) was logged: the shared conversion never ran' % (PAIR_SLICE_FLAG, PAIR_SLICE_ENGAGED))
    if (env is not None and env.get(PAIR_SLICE_FLAG) == '1' and env.get('QWEN_FAST_TP4_GDN_GLUE') != '1' and env.get(VGLUE_AUDIT_FLAG) == '1'
            and not any(VGLUE_AUDIT_PASSED in line and 'exact=True' in line for line in container_text.splitlines())):
        problems.append('%s and %s are set and no passing audit line (%s<n> exact=True) was logged: nothing was compared' % (PAIR_SLICE_FLAG, VGLUE_AUDIT_FLAG, VGLUE_AUDIT_PASSED))
    if env is not None and env.get(DISPATCH_DIAG_FLAG) == '1' and DISPATCH_DIAG_LINE not in container_text:
        problems.append('%s is set and no diagnostic line (%s) was logged' % (DISPATCH_DIAG_FLAG, DISPATCH_DIAG_LINE))
    if env is not None and env.get(GDN_SPLIT_FLAG) == '2' and GDN_SPLIT_ENGAGED not in container_text:
        problems.append('%s=2 is set and no engaged line (%s) was logged: the split recurrence never ran' % (GDN_SPLIT_FLAG, GDN_SPLIT_ENGAGED))
    problems += sampdraft_problems(container_text, env)
    problems += u1_problems(env, container_text)
    problems += sdpa_long_problems(env, container_text)
    problems += sdpa_multi_problems(env, container_text)
    problems += spread_problems(env, container_text)
    median, rounds = ramp_kv_median(container_text)
    facts = dict(audit_mismatches=len(mismatches), publish_rounds=rounds, largest_prepare_history_median_ms=median)
    if env is not None and env.get('QWEN_FAST_TP', '2') != '2':
        late = first_prefill_buffers(container_text)
        facts['first_prefill_model_buffers'] = late
        programs, program_facts = late_program_problems(container_text)
        problems += programs
        facts.update(program_facts)
        if late:
            problems.append('the first prefill left %d model buffers allocated after the packed traces were captured '
                            '([MEMLEDGER] item=model_after_prefill): the prefill warm must run before the traces' % late)
    if env is not None and env.get('QWEN_FAST_M3_REQUEST_WARM') in ('1', 'even'):
        warm_problems, warm_facts = request_warm_problems(container_text, env['QWEN_FAST_M3_REQUEST_WARM'])
        problems += warm_problems
        facts.update(warm_facts)
    if env is not None:
        problems += drafter_bf16_problems(env, container_text)
        drafts = draft_facts(container_text)
        facts['draft'] = drafts
        steady = STEADY_TEST in (smoke or {}) and 'error' not in smoke[STEADY_TEST]
        # The batched-draft conditions are the S2 fast path's: a G1 profile (general-*) never drafts, so they would fail it.
        if fast_path(env):
            steady_eight = STEADY_EIGHT_TEST in (smoke or {}) and 'error' not in smoke[STEADY_EIGHT_TEST]
            problems += draft_problems(drafts, env, steady, steady_eight)
        fused = fused_facts(env, container_text)
        if fused is not None:
            facts['fused'] = fused
        problems += fused_problems(env, container_text, steady, fused)
        problems += tpub_problems(env, container_text)
        problems += lookup_problems(env, container_text)
        hostgap, hostgap_found = hostgap_problems(
            env, container_text, STEADY_EIGHT_TEST in (smoke or {}) and 'error' not in smoke[STEADY_EIGHT_TEST])
        problems += hostgap
        if any(hostgap_found[key] for key in ('hostgap_engaged', 'hostgap_refused', 'hostgap_live4_diff', 'hostgap_full_audits')):
            facts['hostgap'] = hostgap_found
    if entry is not None:
        problems += traffic_problems(container_text, entry)
        problems += waiver_problems(container_text, entry)
    if waiver_active_in_log(container_text):
        facts['unqualified'] = WAIVER_STAMP
    if slide:
        if median is None:
            problems.append('no [PACKED-PUBLISH] round with a commit: the ramp commit time is unread (QWEN_FAST_PACKED_AUDIT?)')
        elif median > max_ramp_ms:
            problems.append('ramp commit: median of the largest prepare_history per round is %.1f ms, above %.0f ms '
                            '(the slide is not taking effect)' % (median, max_ramp_ms))
    return problems, facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoke-log', type=Path, required=True)
    parser.add_argument('--container-log', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES)
    parser.add_argument('--max-ramp-kv-ms', type=float, default=50.0)
    options = parser.parse_args(argv)
    try:
        smoke = options.smoke_log.read_text(encoding='utf-8', errors='replace')
        container = options.container_log.read_text(encoding='utf-8', errors='replace')
        slide = slide_on(options.profile, options.profiles)
        env = profile_env(options.profile, options.profiles)
        entry = profile_entry(options.profile, options.profiles)
    except (OSError, ValueError) as error:
        print('SMOKE_CHECK unreadable: %s' % error, file=sys.stderr)
        return 2
    problems, facts = check(smoke, container, slide, options.max_ramp_kv_ms, env, entry)
    print('SMOKE_CHECK profile=%s slide=%s %s' % (options.profile, 'on' if slide else 'off', json.dumps(facts)))
    if facts.get('unqualified'):
        print('SMOKE_CHECK %s: the 262k evidence records were waived; this result is a measurement, never evidence' % WAIVER_STAMP)
    for problem in problems:
        print('SMOKE_CHECK FAILED: %s' % problem)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
