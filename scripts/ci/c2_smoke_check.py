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
  - octo-T8 (QWEN_FAST_OCTO=live|alternate, tp4/octo-t8; octo_judge, called by octo_container_problems): the shape must have EXECUTED, not been mounted - a
    flag-off profile logs no [OCTO] line; a flagged one logs its admission once and at least 16 rounds with shape=octo (written after the octo block's own round
    counter moved), at rows=8 with live at least min_live; under `alternate` at least 16 counted rounds of EACH shape and the counted rounds strictly alternate;
    no program compiled on the first round of a shape after a switch (the [OCTO] programs line: after must equal before, and the counter must be readable);
    eligible rounds that ran as something else than planned stay under a tenth. QWEN_FAST_SOLO_PACKED=1: a padded round of a lone live user ran and none
    went to the per-request engines. The paired timing (octo_judge --verdict) is a separate read, not a stop condition;
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

  - the round-host levers (tp4/round-host, round_host_problems): any QWEN_FAST_TP4_ROUND_HOST_* flag must log its engaged line once, with
    the flags it names equal to the profile's, no refusal and no declined line; the ledger (LOG or LEAN) must log a [PACKED-ROUND-HOST] line a
    step with every field; each lever must have RUN, from the ledger's own counters (SELECT and READ on 95% of the eight-live steps that drafted, KEYED on
    half of the verifies of a block that can be keyed, READ's feature guard skipped on some steps once 64 reads have passed); LEAN must have
    written none of the lines it drops; the audit must have logged an equal=1 line for each audited lever and no equal=0; a profile without
    the flags logs none of these lines;
  - Lever N (QWEN_FAST_LEVER_N=1, tp4/lever-n; levern_problems): the lever must have ENGAGED, not merely been installed. The scheduler's install line, the
    platform wrap's "chunked prefill kept" line, the route's install line and its warm line (before Metal's unsafe-allocation warning, the packed
    traces' first allocation, and with exactly the plan's three steps) must each be logged; no "lever N REFUSED" line; and every prompt that was
    split must show the ledger the exactness argument rests on: one route line per step, each start equal to the previous end, every end but the
    last a multiple of 2,048, the last end the prompt, a decode slot written ONLY by the last step, and no program compiled by a step. While
    decoders and a partial prefill coexist the alternation must be visible: between two prefill steps of one request that ran with decoders
    there is a decode step that served at least one seat. In the stall tests every decoding seat must have progressed inside the arrival's prefill
    window (the control arm freezes them; this arm's point). The digest instrument (QWEN_FAST_LEVERN_AUDIT=1, also alone as the control) must have
    logged a digest line for every prompt of a levern_equal test that ran. The hang shapes (levern_*) are judged as streams, and the exact-length
    rows as completions that counted the prompt they were sent. Nothing here compares the two arms: levern_compare.py does.

  python c2_smoke_check.py --smoke-log smoke.log --container-log container.log --profile P [--profiles qwen_c2_profiles.json]
"""

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

import levern_policy
import round_host

SOLO_TEST = 'concurrent4_solo'
REPLAY_TEST = 'replay_concurrent4'
# The eight-seat tests (tp4/seats8: QWEN_FAST_M3_BLOCKS=2), judged as the four-user ones are: every user a stream that ends in tokens,
# the code ones code answers too; the eight-user replay is judged as the four-user one.
EIGHT_TESTS = ('concurrent8_code_equal', 'concurrent8_code_32k', 'concurrent8_code_128k', 'concurrent8_skew', 'concurrent5_split', 'concurrent8_drain',
               'concurrent8_steady')
# The 262k stall shape (tp4/seats262k): seven decoding users and one cold 253,920-token arrival; its numbers are recorded, not gated, but
# every stream must end in tokens (the arrival's too) and the arrival's time to first token must exist.
STALL_TEST = 'stall8_cold262k'
STALL_TESTS = (STALL_TEST, 'stall8_cold128k')
# The two-arrival shape (the combined window): six decoders and TWO simultaneous cold arrivals. Recorded, not gated, but every stream (the arrivals' too) must end
# in tokens, each arrival must have a time to first token and no error, and the seat gaps must exist (HX-C's and S's reads depend on them).
COLD2_TEST = 'cold2_254k'
# Lever N (tp4/lever-n): the hang shapes are streams of several users (the arrival and the follow-ups included); the equal tests are rows of exact-length completions.
LEVERN_USER_TESTS = ('levern_equal_busy', 'levern_decoder_finishes', 'levern_all_decoders_finish', 'levern_cancel_mid_prefill',
                     'levern_arrival_during_prefill', 'levern_seed_stops')
LEVERN_ROW_TESTS = ('levern_equal', 'levern_equal_long', 'levern_equal_busy')
# Engine reuse (tp4/engine-reuse): the exact-length ladder and the budget ladder are rows (keyed by prompt length, by budget); the churn, the abort-and-reuse
# and the turn loop are streams of users.
PARKED_ROW_TESTS = ('parked_equal', 'parked_budgets')
PARKED_USER_TESTS = ('parked_churn', 'parked_churn_long', 'parked_abort_reuse', 'parked_turns')
PARKED_TESTS = PARKED_ROW_TESTS + PARKED_USER_TESTS
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
# The op-fusion programme (docs/tp4-fusion.md): one table row per lever of the work packages, generated by make_fusion_profiles.py from scripts/ci/fusion-wp/*.json (never
# hand-edited). FUSION_LEVERS rows are (flag, flag value, engaged marker, fell-back marker, what), FUSION_AUDITS rows (audit flag, audit marker, what); fusion_problems() holds
# the same three rules as the sampdraft levers. A lever the static tables above already dispatch is not repeated here. FUSION_RULES names the packages' own stricter rules: host-side
# modules of scripts/ci with problems(env, container_text) -> [problem], called once per arm by fusion_problems (a module that cannot be imported is a problem, not a crash).
# fusion-wp begin: generated by scripts/ci/make_fusion_profiles.py from scripts/ci/fusion-wp/*.json; do not edit by hand
FUSION_LEVERS = (
    ('QWEN_FAST_KV_PAGE_WRITER', '1', '[PINDIAG] tp4 kv page writer engaged', '[PINDIAG] tp4 kv page writer fell back', 'K/V page writer'),  # WP2
    ('QWEN_FAST_MLP_CFG', 'l1', '[PINDIAG] tp4 mlp gateup engaged route=cfg', '[PINDIAG] tp4 mlp gateup fell back', 'MLP streaming config and L1 multiply'),  # WP4
    ('QWEN_FAST_MLP_GATEUP', '1', '[PINDIAG] tp4 mlp gateup engaged route=fused', '[PINDIAG] tp4 mlp gateup fell back', 'MLP fused gate|up with SwiGLU epilogue'),  # WP4
    ('QWEN_FAST_CCL_OPTIONS', 'rs-c1', '[PINDIAG] tp4 ccl options engaged', '[PINDIAG] tp4 ccl options fell back', 'CCL options'),  # WP5
    ('QWEN_FAST_DRAFT_REDUCE', '1', '[PINDIAG] tp4 draft reduce engaged', '[PINDIAG] tp4 draft reduce fell back', 'WP6 F-F1 drafter local reduce'),  # WP6
    ('QWEN_FAST_DRAFT_TAIL', '1', '[PINDIAG] tp4 draft tail engaged', '[PINDIAG] tp4 draft tail fell back', 'WP6 F-F3a drafter SwiGLU and residual kernels'),  # WP6
    ('QWEN_FAST_DRAFT_GATEUP1', '1', '[PINDIAG] tp4 draft gateup1 engaged', '[PINDIAG] tp4 draft gateup1 fell back', 'WP6 F-F3c drafter fused gate|up matmul'),  # WP6
    ('QWEN_FAST_DRAFT_MM_GRID', '1', '[PINDIAG] tp4 draft mmgrid engaged', '[PINDIAG] tp4 draft mmgrid fell back', 'WP6 R2 drafter matmuls on device-width grids'),  # WP6
    ('QWEN_FAST_DRAFT_PERMUTE', '1', '[PINDIAG] tp4 draft permute engaged', '[PINDIAG] tp4 draft permute fell back', 'F-F2 drafter permutation kernels'),  # WP7
    ('QWEN_FAST_DRAFT_QKV1', '1', '[PINDIAG] tp4 draft qkv1 engaged', '[PINDIAG] tp4 draft qkv1 fell back', 'F-F3c fused drafter q|k|v projection'),  # WP7
    ('QWEN_FAST_DRAFT_HEAD64', '1', '[PINDIAG] tp4 draft head64 engaged', '[PINDIAG] tp4 draft head64 fell back', 'F-F4 one 64-row drafter head'),  # WP7
    ('QWEN_FAST_PRESTAGE_DIFF', '1', '[PINDIAG] tp4 prestage diff engaged', '[PINDIAG] tp4 prestage diff fell back', 'WPH-1 pre-stage diff'),  # WPH
    ('QWEN_FAST_WRITE_PACKED_LEAN', '1', '[PINDIAG] tp4 write packed lean engaged', '[PINDIAG] tp4 write packed lean fell back', 'WPH-2 lean write_packed'),  # WPH
    ('QWEN_FAST_BATCHED_READS', '1', '[PINDIAG] tp4 batched reads engaged', '[PINDIAG] tp4 batched reads fell back', 'WPH-3 batched verify and collect read-backs'),  # WPH
    ('QWEN_FAST_MLP_CFG', 'g3u4d3', '[PINDIAG] tp4 mlp gateup engaged route=cfg', '[PINDIAG] tp4 mlp gateup fell back', 'MLP streaming config g3u4d3 (the card-M sweep winner)'),  # WP0
    ('QWEN_FAST_CCL_OPTIONS', 'served', '[PINDIAG] tp4 ccl options engaged', '[PINDIAG] tp4 ccl options fell back', 'CCL options (the served set)'),  # WP5
    ('QWEN_FAST_BATCHED_READS', 'async', '[PINDIAG] tp4 batched reads engaged', '[PINDIAG] tp4 batched reads fell back', 'WPH-3 batched read-backs (asynchronous copies)'),  # WPH
)
FUSION_AUDITS = (
    ('QWEN_FAST_KV_PAGE_WRITER_AUDIT', '[PINDIAG] tp4 kv page writer audit', 'K/V page writer'),  # WP2
    ('QWEN_FAST_MLP_CFG_AUDIT', '[PINDIAG] tp4 mlp gateup audit', 'MLP streaming config and L1 multiply'),  # WP4
    ('QWEN_FAST_MLP_GATEUP_AUDIT', '[PINDIAG] tp4 mlp gateup audit', 'MLP fused gate|up with SwiGLU epilogue'),  # WP4
    ('QWEN_FAST_CCL_OPTIONS_AUDIT', '[PINDIAG] tp4 ccl options audit', 'CCL options'),  # WP5
    ('QWEN_FAST_DRAFT_REDUCE_AUDIT', '[PINDIAG] tp4 draft reduce audit', 'WP6 F-F1 drafter local reduce'),  # WP6
    ('QWEN_FAST_DRAFT_TAIL_AUDIT', '[PINDIAG] tp4 draft tail audit', 'WP6 F-F3a drafter SwiGLU and residual kernels'),  # WP6
    ('QWEN_FAST_DRAFT_GATEUP1_AUDIT', '[PINDIAG] tp4 draft gateup1 audit', 'WP6 F-F3c drafter fused gate|up matmul'),  # WP6
    ('QWEN_FAST_DRAFT_MM_GRID_AUDIT', '[PINDIAG] tp4 draft mmgrid audit', 'WP6 R2 drafter matmuls on device-width grids'),  # WP6
    ('QWEN_FAST_DRAFT_PERMUTE_AUDIT', '[PINDIAG] tp4 draft permute audit', 'F-F2 drafter permutation kernels'),  # WP7
    ('QWEN_FAST_DRAFT_QKV1_AUDIT', '[PINDIAG] tp4 draft qkv1 audit', 'F-F3c fused drafter q|k|v projection'),  # WP7
    ('QWEN_FAST_DRAFT_HEAD64_AUDIT', '[PINDIAG] tp4 draft head64 audit', 'F-F4 one 64-row drafter head'),  # WP7
    ('QWEN_FAST_PRESTAGE_DIFF_AUDIT', '[PINDIAG] tp4 prestage diff audit', 'WPH-1 pre-stage diff'),  # WPH
    ('QWEN_FAST_WRITE_PACKED_LEAN_AUDIT', '[PINDIAG] tp4 write packed lean audit', 'WPH-2 lean write_packed'),  # WPH
    ('QWEN_FAST_BATCHED_READS_AUDIT', '[PINDIAG] tp4 batched reads audit', 'WPH-3 batched verify and collect read-backs'),  # WPH
)
FUSION_RULES = (
    'tp4_shard_argmax_smoke',  # WP0
    'ccl_options_smoke',  # WP0
    'draft_permute_smoke',  # WP0
    'draft_wp6_smoke',  # WP0
    'kv_page_writer_tp4_smoke',  # WP0
    'tp4_mlp_gateup_smoke',  # WP4
    'hostgap_wph_smoke',  # WPH
)
# fusion-wp end
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
SDPA_MULTI_UNQUALIFIED = '[PINDIAG] tp4 sdpa multi UNQUALIFIED'
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
    if env.get('QWEN_FAST_MAX_POSITION') == '262144' and SDPA_MULTI_UNQUALIFIED not in container_text:
        problems.append('%s=multi is set at 262,144 and no UNQUALIFIED line (%s) was logged: the attach no longer says the G16 launch is outside the 262k '
                        'evidence' % (SDPA_LONG_FLAG, SDPA_MULTI_UNQUALIFIED))
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
# The multi-user SDPA launch (G16 flags 0x21) is outside packed_any_evidence_tp4_262144.json (G4B1/G4B3/G8B2, 0x23): an arm that runs it is a measurement at 262k, never
# qualified evidence, with or without the waiver. The attach logs SDPA_MULTI_UNQUALIFIED at 262,144; every summary the gates and the smoke write carries this stamp then.
MULTI_STAMP = 'UNQUALIFIED (multi G16, gate only)'


def unqualified_stamp(container_text):
    """The stamp a result carries for what its server log shows: the 262k waiver's, the multi launch's, both joined by ' + ', or '' when neither."""
    text = container_text or ''
    stamps = []
    if waiver_active_in_log(text):
        stamps.append(WAIVER_STAMP)
    if SDPA_MULTI_UNQUALIFIED in text:
        stamps.append(MULTI_STAMP)
    return ' + '.join(stamps)
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


def cold2_problems(entry):
    """[problem] for the cold2_254k result (two simultaneous cold arrivals beside six decoders): every user a stream that ends in tokens, an arrival record with
    a numeric time to first token and no error for each of the two arrivals, and a recorded gap for every decoder; [] for no such test."""
    if entry is None:
        return []
    if not isinstance(entry, dict):
        return ['%s: no result' % COLD2_TEST]
    if 'error' in entry:
        return ['%s: %s' % (COLD2_TEST, entry['error'])]
    problems = []
    for index, user in enumerate(entry.get('users') or []):
        problems += stream_problems('%s user %d' % (COLD2_TEST, index), user)
    arrivals = entry.get('arrivals')
    if not isinstance(arrivals, list) or len(arrivals) != 2:
        problems.append('%s: %s arrival records, two were asked for' % (COLD2_TEST, len(arrivals) if isinstance(arrivals, list) else 'no'))
        arrivals = arrivals if isinstance(arrivals, list) else []
    for index, arrival in enumerate(arrivals):
        arrival = arrival if isinstance(arrival, dict) else {}
        if arrival.get('error'):
            problems.append('%s arrival %d: %s' % (COLD2_TEST, index, arrival['error']))
        if not isinstance(arrival.get('ttft_s'), (int, float)):
            problems.append('%s arrival %d: no time to first token' % (COLD2_TEST, index))
    gaps = entry.get('seat_gaps')
    if not gaps:
        problems.append('%s: no seat gap was recorded' % COLD2_TEST)
    for gap in gaps or ():
        if isinstance(gap, dict) and gap.get('error'):
            problems.append('%s seat %s: %s' % (COLD2_TEST, gap.get('seat'), gap['error']))
        elif isinstance(gap, dict) and gap.get('longest_gap_s') is None:
            problems.append('%s seat %s: no gap could be read (the decoder streamed fewer than two chunks)' % (COLD2_TEST, gap.get('seat')))
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
    for stall_name in STALL_TESTS:
        stall = results.get(stall_name)
        if isinstance(stall, dict) and 'error' not in stall:
            for index, user in enumerate(stall.get('users') or []):
                problems += stream_problems('%s user %d' % (stall_name, index), user)
            if not isinstance(stall.get('arrival_ttft_s'), (int, float)):
                problems.append('%s: the cold arrival has no time to first token' % stall_name)
            if not stall.get('seat_gaps'):
                problems.append('%s: no seat gap was recorded' % stall_name)
        elif isinstance(stall, dict):
            problems.append('%s: %s' % (stall_name, stall['error']))
    problems += cold2_problems(results.get(COLD2_TEST))
    problems += levern_smoke_problems(results)
    problems += parked_smoke_problems(results)
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


def levern_smoke_problems(results):
    """The Lever N tests' own results (levern_* in c2_serving_smoke): the exact-length rows and the hang shapes' streams."""
    problems = []
    for name in LEVERN_ROW_TESTS + LEVERN_USER_TESTS:
        entry = results.get(name)
        if entry is None:
            continue
        if 'error' in entry:
            problems.append('%s: %s' % (name, entry['error']))
            continue
        for length, row in sorted((entry.get('prompts') or {}).items(), key=lambda item: int(item[0])):
            label = '%s prompt %s' % (name, length)
            if 'error' in row:
                problems.append('%s: %s' % (label, row['error']))
                continue
            if row.get('prompt_tokens') != row.get('prompt_tokens_sent'):
                problems.append('%s: the server counted %r tokens of the %r it was sent' % (label, row.get('prompt_tokens'),
                                                                                           row.get('prompt_tokens_sent')))
            if not row.get('tokens'):
                problems.append('%s: no tokens' % label)
            if row.get('finish') is None:
                problems.append('%s: no finish reason' % label)
            if not row.get('content_sha256'):
                problems.append('%s: no answer hash' % label)
        users = entry.get('users') or []
        for index, user in enumerate(users):
            if name == 'levern_seed_stops' and index == len(users) - 2:
                # the max_tokens=1 request: it must end at its seed, with exactly that token
                if isinstance(user, dict) and 'error' not in user and user.get('tokens') != 1:
                    problems.append('%s: the one-token request produced %r tokens' % (name, user.get('tokens')))
                elif isinstance(user, dict) and 'error' in user:
                    problems.append('%s user %d: %s' % (name, index, user['error']))
                continue
            problems += stream_problems('%s user %d' % (name, index), user)
        if name == 'levern_cancel_mid_prefill':
            dropped = entry.get('dropped') or {}
            if dropped.get('outcome') != 'dropped':
                problems.append('%s: the cold prefill was not dropped by the client (%r): the abort shape never ran' % (name, dropped.get('outcome')))
    return problems


def parked_smoke_problems(results):
    """The engine-reuse tests' own results (parked_* in c2_serving_smoke): the ladder rows and the streams of the churn, the abort-and-reuse and the turns."""
    problems = []
    for name in PARKED_TESTS:
        entry = results.get(name)
        if entry is None:
            continue
        if 'error' in entry:
            problems.append('%s: %s' % (name, entry['error']))
            continue
        for key, row in sorted((entry.get('prompts') or {}).items(), key=lambda item: int(item[0])):
            label = '%s %s %s' % (name, 'budget' if name == 'parked_budgets' else 'prompt', key)
            if 'error' in row:
                problems.append('%s: %s' % (label, row['error']))
                continue
            if row.get('prompt_tokens') != row.get('prompt_tokens_sent'):
                problems.append('%s: the server counted %r tokens of the %r it was sent' % (label, row.get('prompt_tokens'), row.get('prompt_tokens_sent')))
            if not row.get('tokens'):
                problems.append('%s: no tokens' % label)
            elif name == 'parked_budgets' and row['tokens'] > int(key):
                problems.append('%s: %d tokens past the budget' % (label, row['tokens']))
            elif name == 'parked_budgets' and int(key) == 1 and row['tokens'] != 1:
                problems.append('%s: a one-token budget produced %r tokens' % (label, row['tokens']))
            if row.get('finish') is None:
                problems.append('%s: no finish reason' % label)
            if not row.get('content_sha256'):
                problems.append('%s: no answer hash' % label)
        users = entry.get('users') or []
        for index, user in enumerate(users):
            if name == 'parked_abort_reuse' and index % 3 == 1:
                # the max_tokens=1 request of each round: it ends at its seed, with exactly that token
                if isinstance(user, dict) and 'error' not in user and user.get('tokens') != 1:
                    problems.append('%s user %d: the one-token request produced %r tokens' % (name, index, user.get('tokens')))
                elif isinstance(user, dict) and 'error' in user:
                    problems.append('%s user %d: %s' % (name, index, user['error']))
                continue
            problems += stream_problems('%s user %d' % (name, index), user)
    return problems


# The one lever of the op-fusion programme that the static sampdraft tables above dispatch (WP1's S1); the sampdraft DRAFT_CONV and DRAFT_HEADS levers are part of the
# production profile, so they are not fusion levers and must never move the production bands.
FUSION_SAMPDRAFT_FLAGS = ('QWEN_FAST_TP4_SHARD_ARGMAX',)


def fusion_arm_flags(env):
    """([lever flags], [audit flags]) of the op-fusion programme (FUSION_LEVERS, FUSION_AUDITS and S1 from the sampdraft tables) that the profile env turns on. A lever counts at
    the value its table row judges (a lever the profile sets to '0' is off); an audit flag counts at '1'. Production and every profile without a lever give ([], []), which keeps
    parked_judge.ENGINE_GB, the production band, exactly."""
    env = env or {}
    levers = set(flag for flag, value, _engaged, _fell, _what in FUSION_LEVERS if env.get(flag) == value)
    levers |= set(flag for flag in FUSION_SAMPDRAFT_FLAGS if env.get(flag) == '1')
    audits = set(flag for flag, _marker, _what in FUSION_AUDITS if env.get(flag) == '1')
    audits |= set(flag + '_AUDIT' for flag in FUSION_SAMPDRAFT_FLAGS if env.get(flag + '_AUDIT') == '1')
    return sorted(levers), sorted(audits)


def parked_container_problems(env, container_text):
    """(problems, facts) of the engine-reuse markers in the container log (parked_judge.judge): what a parked profile must show, and that any other
    profile shows none. An op-fusion lever arm is judged against parked_judge's lever band for the engines' DRAM (derivation there); production is not."""
    if env is None:
        return [], {}
    import parked_judge

    levers, audits = fusion_arm_flags(env)
    return parked_judge.judge(env, container_text, levers=levers, lever_audits=audits)


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


def octo_blocks(env):
    """The packed blocks octo-T8 adds beside the M3 ones: 1 when the profile asks for QWEN_FAST_OCTO live or alternate (a third block, which the host-gap pre-stage
    engages over too), else 0."""
    return 1 if str(env.get('QWEN_FAST_OCTO', 'off')) in ('live', 'alternate') else 0


def octo_container_problems(env, container_text):
    """(problems, facts) of the octo-T8 markers in the container log (octo_judge.judge): what an octo profile must show, and that any other profile shows none."""
    if env is None:
        return [], {}
    import octo_judge

    return octo_judge.judge(env, container_text)


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
    # (and the octo-T8 block, a third packed block with its own fused commit: users x (1 + rows) traces in place, 8 x 9 = 72)
    blocks = m3_blocks(env) + octo_blocks(env)
    if engaged_lines != blocks:
        problems.append('the fused commit engaged line (%s) appears %d times, not %s' % (
            gate.FUSED_ENGAGED_MARKER, engaged_lines, 'once' if blocks == 1 else 'once per %s block (%d blocks)' % ('packed' if octo_blocks(env) else 'M3', blocks)))
    engaged = facts.get('engaged')
    inplace = env.get(FUSED_INPLACE_FLAG) == '1'
    for later in list(gate.FUSED_ENGAGED_LINE.finditer(container_text))[1:]:
        wanted = int(later.group(1)) * (1 + int(later.group(2))) if inplace else int(later.group(1))     # (one slide per accepted prefix 1..rows: 16 at M3, 8 at octo)
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
            both_blocks = sum(1 for paths in rounds.values()
                              if (len(paths) == 4 * m3_blocks(env) or (octo_blocks(env) and len(paths) % 4 == 0 and len(paths) >= 4 * m3_blocks(env)))
                              and all(path == 'fused' for path in paths))
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
    elif facts['hostgap_engaged'] != blocks + octo_blocks(env):
        problems.append('the two-block engaged line (%s) appears %d times, not once per M3 block (%d)' % (
            HOSTGAP_ENGAGED, facts['hostgap_engaged'], blocks + octo_blocks(env)))
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
        # The share of diff-path verifies is a measurement (W1's timing arms carry the log). A traffic profile drops the diagnostic
        # log; the lever's function is still judged above (epochs engaged once, 4-live verifies on the diff path) and its
        # exactness by the verify audits. Unread, not a problem (v578, 2026-10-05).
        facts['hostgap_share'] = 'unread (%s off)' % HOSTGAP_LOG_FLAG
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


# tp4/round-host: the host work inside a verified round (round_host.py), judged on the ledger the arm writes.
ROUND_HOST_LEDGER = re.compile(r'\[PACKED-ROUND-HOST\] round=(\d+) live=(\S+) pos=(\S+) ((?:\w+=\S+ )+)sel=(\d+) read=(\d+) keyed=(\d+) guard=(\d+)')
ROUND_HOST_AUDIT_LINE = re.compile(r'\[ROUND-HOST-AUDIT\] kind=(\w+) equal=(\d)')
ROUND_HOST_ENGAGED_LINE = re.compile(r'\[PINDIAG\] round host engaged ((?:\w+=\d ?)+)')
# The lines LEAN drops (round_host.LEAN_PHASES and the three audit lines behind the commits): none may appear.
ROUND_HOST_LEAN_ABSENT = (re.compile(r'\[PHASE\] (?:propose|prepare_proposals|propose_quad|early_draft|packed_commit) '),
                          re.compile(r'\[PACKED-PUBLISH-SPLIT\] '), re.compile(r'\[PACKED-COMMIT-HOST\] '),
                          re.compile(r'\[PACKED-COMMIT\] '))
ROUND_HOST_MIN_STEPS = 50
ROUND_HOST_FAST_SHARE = 0.95
ROUND_HOST_KEYED_SHARE = 0.5
ROUND_HOST_GUARD_AFTER = 64
ROUND_HOST_AUDIT_KINDS = {round_host.SELECT_FLAG: 'select', round_host.READ_FLAG: 'merge', round_host.KEYED_FLAG: 'keyed'}


def round_host_facts(container_text):
    """The ledger's steps ({field: value or None, live, sel, read, keyed, guard}), the audit lines [(kind, equal)] and the engaged line's flags."""
    steps = []
    for number, live, position, fields, sel, read, keyed, guard in ROUND_HOST_LEDGER.findall(container_text):
        values = {}
        for item in fields.split():
            name, _, value = item.partition('=')
            values[name] = None if value == '-' else float(value)
        steps.append(dict(round=int(number), live=None if live == '-' else int(live), position=None if position == '-' else int(position),
                          fields=values, sel=int(sel), read=int(read), keyed=int(keyed), guard=int(guard)))
    engaged = ROUND_HOST_ENGAGED_LINE.findall(container_text)
    return dict(steps=steps, audits=[(kind, int(equal)) for kind, equal in ROUND_HOST_AUDIT_LINE.findall(container_text)],
                engaged=engaged)


def round_host_problems(env, container_text, steady_eight):
    """(problems, facts) for the round-host levers: see the module docstring. Judged only where the profile set a flag; a profile with none
    logs none of the lever's lines. Where the eight-seat steady mix did not run, the run-length rules are not judged (the lines still are)."""
    env = env or {}
    found = round_host_facts(container_text)
    set_flags = {name: env.get(name, '0') for name in round_host.FLAGS if env.get(name, '0') != '0'}
    steps, audits = found['steps'], found['audits']
    problems = []
    marked = (container_text.count(round_host.ENGAGED_MARKER) or container_text.count(round_host.REFUSED_MARKER) or steps or audits
              or container_text.count(round_host.DECLINED_MARKER))
    if not set_flags:
        if marked:
            problems.append('no QWEN_FAST_TP4_ROUND_HOST_* flag is set and the log holds round-host lines: the lever ran on a profile without it')
        return problems, found
    for name, value in set_flags.items():
        if value != '1':
            problems.append('%s=%s: the flag is 0 or 1' % (name, value))
    if container_text.count(round_host.REFUSED_MARKER):
        problems.append('a round-host refusal line (%s) was logged' % round_host.REFUSED_MARKER)
    if container_text.count(round_host.DECLINED_MARKER):
        problems.append('a round-host fast path declined to the reference (%s): %s' % (
            round_host.DECLINED_MARKER, next(line.strip()[:160] for line in container_text.splitlines() if round_host.DECLINED_MARKER in line)))
    if len(found['engaged']) != 1:
        problems.append('%d "%s" lines were logged for one attach (one needed)' % (len(found['engaged']), round_host.ENGAGED_MARKER))
    else:
        engaged = dict(item.split('=') for item in found['engaged'][0].split())
        for name in round_host.FLAGS:
            wanted = '1' if set_flags.get(name) == '1' else '0'
            if engaged.get(name.rsplit('_', 1)[-1].lower()) != wanted:
                problems.append('the engaged line says %s=%s and the profile has %s=%s' % (
                    name.rsplit('_', 1)[-1].lower(), engaged.get(name.rsplit('_', 1)[-1].lower()), name, wanted))
    ledger_on = '1' in (set_flags.get(round_host.LOG_FLAG), set_flags.get(round_host.LEAN_FLAG))
    if ledger_on and not steps:
        problems.append('%s or %s is set and no %s line was logged: the ledger never ran' % (
            round_host.LOG_FLAG, round_host.LEAN_FLAG, round_host.LEDGER_MARKER))
    if not ledger_on and steps:
        problems.append('neither %s nor %s is set and the ledger wrote lines' % (round_host.LOG_FLAG, round_host.LEAN_FLAG))
    for step in steps:
        absent = [name for name in round_host.Ledger.FIELDS if name not in step['fields']]
        if absent:
            problems.append('a %s line lacks the fields %s' % (round_host.LEDGER_MARKER, ','.join(absent)))
            break
    judged = [step for step in steps if step['live'] is not None and step['fields'].get('step') is not None]
    facts = dict(round_host_steps=len(steps), round_host_judged=len(judged))
    found.update(facts)
    # The run-length rules need a long enough run of eight-live steps that drafted (a ledger step with a draft has an `ed` field): both quads
    # serve exactly those, so the quad readback (READ) and the batched selection (SELECT) run on every one of them; a ramp-up or a tail step
    # with fewer live users drafts through the pairs and says nothing about the lever.
    drafted = [step for step in judged if step['fields'].get('ed') is not None and step['live'] == 8]
    if steady_eight and len(drafted) >= ROUND_HOST_MIN_STEPS:
        for flag, field, label in ((round_host.SELECT_FLAG, 'sel', 'selection'), (round_host.READ_FLAG, 'read', 'quad merge')):
            if set_flags.get(flag) == '1':
                taken = sum(1 for step in drafted if step[field] >= 1)
                if taken < ROUND_HOST_FAST_SHARE * len(drafted):
                    problems.append('%s is set and only %d of %d eight-live drafting steps took the fast %s (%d%% needed)' % (
                        flag, taken, len(drafted), label, int(ROUND_HOST_FAST_SHARE * 100)))
        if set_flags.get(round_host.READ_FLAG) == '1' and len(drafted) > ROUND_HOST_GUARD_AFTER:
            # round_host.guard_full: READ checks the replicated selector feature on every chip for the first 64 reads and every 64th, and the
            # ledger's `guard` counts the reads that skipped it. Under AUDIT the guard runs on EVERY read (it is part of what the audit compares),
            # so the counter must stay 0 and a skip there is a code bug; without AUDIT the sampling must show skips once 64 reads have passed
            # (run 38019684412 and 38020932079: 678 skips in 752 guards; run 38018599272, audited: 0 skips in 375).
            skipped = sum(step['guard'] for step in drafted)
            if set_flags.get(round_host.AUDIT_FLAG) == '1':
                if skipped:
                    problems.append('%s and %s are set and the replicated-feature guard was skipped %d time(s) over %d steps: the audit runs it on every read' % (
                        round_host.AUDIT_FLAG, round_host.READ_FLAG, skipped, len(drafted)))
            elif not skipped:
                problems.append('%s is set and the replicated-feature guard was never skipped over %d steps' % (round_host.READ_FLAG, len(drafted)))
        if set_flags.get(round_host.KEYED_FLAG) == '1':
            possible = sum(2 if step['live'] == 8 else 1 for step in judged if step['live'] in (4, 8))
            taken = sum(step['keyed'] for step in judged if step['live'] in (4, 8))
            facts['round_host_keyed_share'] = round(taken / possible, 3) if possible else None
            if possible >= ROUND_HOST_MIN_STEPS and taken < ROUND_HOST_KEYED_SHARE * possible:
                problems.append('%s is set and only %d of %d verifies took the keyed write (%d%% needed)' % (
                    round_host.KEYED_FLAG, taken, possible, int(ROUND_HOST_KEYED_SHARE * 100)))
    if set_flags.get(round_host.LEAN_FLAG) == '1':
        for pattern in ROUND_HOST_LEAN_ABSENT:
            match = pattern.search(container_text)
            if match:
                problems.append('%s is set and the log holds a line it drops: %s' % (round_host.LEAN_FLAG, match.group(0).strip()))
    unequal = [kind for kind, equal in audits if not equal]
    if unequal:
        problems.append('%d round-host audit line(s) read equal=0 (%s): a fast path and its reference differ' % (len(unequal), ','.join(sorted(set(unequal)))))
    if set_flags.get(round_host.AUDIT_FLAG) == '1':
        for flag, kind in ROUND_HOST_AUDIT_KINDS.items():
            if set_flags.get(flag) == '1' and not any(item == (kind, 1) for item in audits) and (steady_eight or kind != 'keyed'):
                problems.append('%s and %s are set and no [ROUND-HOST-AUDIT] kind=%s equal=1 line was logged: nothing was compared' % (
                    round_host.AUDIT_FLAG, flag, kind))
    elif audits:
        problems.append('%s is not set and the log holds [ROUND-HOST-AUDIT] lines' % round_host.AUDIT_FLAG)
    return problems, found


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


def fusion_problems(container_text, env):
    """The op-fusion programme's stop conditions (FUSION_LEVERS, FUSION_AUDITS): a fall-back line always; a missing engaged line for a lever the profile sets
    (to the lever's value); a missing 'exact=True' audit line for an audit flag the profile sets. An audit MISMATCH line is already `audit mismatch` to check()."""
    problems = []
    lines = container_text.splitlines()
    for flag, value, engaged, fell_back, what in FUSION_LEVERS:
        fell = [line.strip()[:200] for line in lines if fell_back in line]
        problems += ['the %s fell back (the served path ran, it saved nothing): %s' % (what, line) for line in fell[:4]]
        if env is not None and env.get(flag) == value and engaged not in container_text:
            problems.append('%s=%s is set and no engaged line (%s) was logged: the %s never ran' % (flag, value, engaged, what))
    for flag, marker, what in FUSION_AUDITS:
        if env is not None and env.get(flag) == '1' and not any(marker in line and 'exact=True' in line for line in lines):
            problems.append('%s is set and no passing audit line (%s ... exact=True) was logged: the %s was never audited' % (flag, marker, what))
    for module in FUSION_RULES:
        try:
            import importlib

            problems.extend(importlib.import_module(module).problems(env, container_text))
        except Exception as error:      # a package's rule that cannot run must fail the arm loudly, never pass it silently
            problems.append('the smoke rule of %s could not run (%s: %s)' % (module, type(error).__name__, error))
    return problems


# tp4/upload-p0 (qwen_device_zeros, qwen_lazy_shard): the engine-start upload levers. Per lever the same rules as the other levers: a profile that asks for it
# and logs no engaged line ran the served path, a refusal or a mismatch line fails the arm (the lever latched off and the host path ran), an audit flag with no
# 'exact=True' line was never audited, and an engaged line on a profile that did not ask means the lever ran without its switch. The lines are pinned equal to the
# modules' by test_upload_p0.
DEVICE_ZEROS_FLAG = 'QWEN_FAST_DEVICE_ZEROS'
DEVICE_ZEROS_AUDIT_FLAG = 'QWEN_FAST_DEVICE_ZEROS_AUDIT'
DEVICE_ZEROS_ENGAGED = '[PINDIAG] tp4 device zeros engaged'
DEVICE_ZEROS_REFUSED = '[PINDIAG] tp4 device zeros refused'
DEVICE_ZEROS_AUDIT = '[PINDIAG] tp4 device zeros audit'
DEVICE_ZEROS_MISMATCH = '[PINDIAG] tp4 device zeros audit mismatch'
DEVICE_ZEROS_TAGS = ('kv_cache', 'buffer_pool')
LAZY_SHARD_FLAG = 'QWEN_FAST_LAZY_SHARD_W'
LAZY_SHARD_AUDIT_FLAG = 'QWEN_FAST_LAZY_SHARD_W_AUDIT'
LAZY_SHARD_ENGAGED = '[PINDIAG] tp4 lazy shard engaged'
LAZY_SHARD_REFUSED = '[PINDIAG] tp4 lazy shard refused'
LAZY_SHARD_AUDIT = '[PINDIAG] tp4 lazy shard audit'
LAZY_SHARD_MISMATCH = '[PINDIAG] tp4 lazy shard audit mismatch'


def upload_p0_problems(env, container_text):
    """[problem] for the engine-start upload levers a profile's `env` asks for (or does not ask for), read from the container log."""
    env = env or {}
    lines = container_text.splitlines()
    problems = []
    for flag, audit_flag, engaged, refused, audit, mismatch, what, tags in (
            (DEVICE_ZEROS_FLAG, DEVICE_ZEROS_AUDIT_FLAG, DEVICE_ZEROS_ENGAGED, DEVICE_ZEROS_REFUSED, DEVICE_ZEROS_AUDIT, DEVICE_ZEROS_MISMATCH,
             'device zero fill', DEVICE_ZEROS_TAGS),
            (LAZY_SHARD_FLAG, LAZY_SHARD_AUDIT_FLAG, LAZY_SHARD_ENGAGED, LAZY_SHARD_REFUSED, LAZY_SHARD_AUDIT, LAZY_SHARD_MISMATCH,
             'lazy shard loader', (None,))):
        on = env.get(flag) == '1'
        bad = [line.strip()[:240] for line in lines if refused in line or mismatch in line]
        problems += ['the %s latched off or was refused (the host path ran, it saved nothing): %s' % (what, line) for line in bad[:4]]
        if not on:
            if any(engaged in line for line in lines):
                problems.append('%s is not set and the log holds %s lines: the %s ran on a profile without it' % (flag, engaged, what))
            continue
        for tag in tags:
            needle = engaged if tag is None else engaged + ' tag=' + tag
            if not any(needle in line for line in lines):
                problems.append('%s=1 is set and no engaged line (%s) was logged: the %s never ran%s' % (
                    flag, needle, what, '' if tag is None else ' for ' + tag))
        if env.get(audit_flag) == '1':
            for tag in tags:
                if not any(audit in line and 'exact=True' in line and (tag is None or 'tag=' + tag in line) for line in lines):
                    problems.append('%s=1 is set and no passing audit line (%s exact=True%s) was logged: the %s was never audited' % (
                        audit_flag, audit, '' if tag is None else ' tag=' + tag, what))
        elif any(audit in line and 'exact=' in line for line in lines):
            problems.append('%s is not set and the log holds audit lines' % audit_flag)
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


def audited_verify(env):
    """True when either verify audit is on in the served profile's env (they inflate the ramp's prepare_history)."""
    env = env or {}
    return env.get('QWEN_FAST_VERIFY_T1_AUDIT') == '1' or env.get('QWEN_FAST_VERIFY_T2_AUDIT') == '1'
LEVERN_FLAG = 'QWEN_FAST_LEVER_N'
LEVERN_AUDIT_FLAG = 'QWEN_FAST_LEVERN_AUDIT'
LEVERN_CHUNK = 2048
LEVERN_WARM_REQUEST = '__levern_warm__'
LEVERN_WARM_STEPS = 3
LEVERN_INSTALLED = '[PINDIAG] lever N installed on '
LEVERN_PLATFORM = '[PINDIAG] lever N: chunked prefill kept for qwen3_5'
LEVERN_ROUTE_INSTALLED = '[PINDIAG] lever N route installed: route=1'
LEVERN_WARMED = '[PINDIAG] lever N route warmed before the packed traces: steps=%d' % LEVERN_WARM_STEPS
LEVERN_REFUSED = '[PINDIAG] lever N REFUSED'
LEVERN_ROUTE = re.compile(r'\[PINDIAG\] lever N route req=(\S+) start=(\d+) end=(\d+) prompt=(\d+) final=([01]) wrote_slot=([01]) '
                          r'ms=([0-9.]+) programs=(\d+|None)->(\d+|None) window=(\d+)(?: source=(COLD|CHECKPOINT|SCRATCH|PARKED) captured=(\S+))?')
# The merged route (Lever N beside prefix reuse, docs/lever-n-prefix-merged-route.md): its warm runs the four sources (four steps, no route lines), the
# graft says it runs chunked beside the cap, and a step that is not the first of its request continues a scratch or a parked state.
LEVERN_MERGED_WARM_STEPS = 4
LEVERN_GRAFT_CHUNKED = 'chunked=levern'
LEVERN_PARK_OUT = re.compile(r'\[PINDIAG\] lever N park out req=(\S+) at=(\d+)')
LEVERN_PARK_IN = re.compile(r'\[PINDIAG\] lever N park in req=(\S+) at=(\d+)')
LEVERN_STEP = re.compile(r'\[PINDIAG\] lever N step n=(\d+) kind=(prefill|decode) seats=(\d+) req=(\S+) start=(\S+) tokens=(\d+) '
                         r'end=(\S+) prompt=(\S+) final=(\S+) reason=(\S+) prev=(\S+):(\S+)ms owed_ms=(\S+) owed_rounds=(\d+)'
                         r'(?: f_eff=(\S+))?(?: need=(\S+))?(?: gap_ms=(\S+))?')
LEVERN_DIGEST = re.compile(r'\[PINDIAG\] lever N digest req=(\S+) prompt=(\d+) tokens_sha=([0-9a-f]{32}) slot_sha=([0-9a-f]{32}) '
                           r'logits_sha=([0-9a-f]{32}) kv_sha=([0-9a-f]{32})')


# The audit digest's region read (QWEN_FAST_LEVERN_KV_READ, levern_policy.KV_READ_LINE and KV_CROSS_LINE; docs/prefix-audit-cost.md): one 'kv read mode=' line per
# digest that read the KV (a prompt over KV_DIGEST_MAX_PROMPT skips it), one 'kv read cross' line per cross-checked digest.
LEVERN_KV_READ_FLAG = 'QWEN_FAST_LEVERN_KV_READ'
LEVERN_KV_CROSS_FLAG = 'QWEN_FAST_LEVERN_KV_CROSS_STEPS'
LEVERN_KV_PREFIX = '[PINDIAG] lever N kv read '
LEVERN_KV_SKIPPED = '0' * 32
LEVERN_KV_READ = re.compile(r'\[PINDIAG\] lever N kv read mode=(region|cross) req=(\S+) prompt=(\d+) reads=(\d+) blocks_read=(\d+) '
                            r'region_ms=([0-9.]+) fallback=(\S.*)$', re.M)
LEVERN_KV_CROSS = re.compile(r'\[PINDIAG\] lever N kv read cross req=(\S+) tensors=(\d+) mismatched=(\d+) region_ms=([0-9.]+) '
                             r'whole_read_ms=([0-9.]+) blocks_read=(\d+) fallback=(\S.*)$', re.M)
LEVERN_KV_READY = '[PINDIAG] lever N kv read ready mode='


def levern_kv_read_problems(env, container_text, digests):
    """[problem] for the audit digest's read mode. Unset or full: no 'lever N kv read' line (a line says a region read ran under a profile that never
    asked). region or cross: the attach's ready line, one read line per digest that read the KV (a fallback to the whole-cache read in any of them is
    a failure: the region read did not answer), and under cross one cross line per cross-checked digest (the first QWEN_FAST_LEVERN_KV_CROSS_STEPS,
    default 1) with mismatched=0 and no fallback: an absent line is NOT EXERCISED and fails, a mismatch is the extension's finding."""
    mode = env.get(LEVERN_KV_READ_FLAG, 'full')
    lines = [line for line in container_text.splitlines() if LEVERN_KV_PREFIX in line]
    if mode not in ('region', 'cross'):
        return ['%d "%s" line(s) on a profile without %s=region or cross (the first: %s)' % (len(lines), LEVERN_KV_PREFIX.strip(), LEVERN_KV_READ_FLAG,
                                                                                              lines[0].strip()[:160])] if lines else []
    problems = []
    if LEVERN_KV_READY + mode not in container_text:
        problems.append('%s=%s and no "%s%s" line was logged: the attach never checked the extension' % (LEVERN_KV_READ_FLAG, mode, LEVERN_KV_READY, mode))
    reads = [m for m in LEVERN_KV_READ.finditer(container_text)]
    wanted = len([row for row in digests if row['kv'] != LEVERN_KV_SKIPPED])
    if len(reads) != wanted:
        problems.append('%s=%s: %d region-read line(s) for %d digest(s) that read the KV (one per digest)' % (LEVERN_KV_READ_FLAG, mode, len(reads), wanted))
    fell = [m for m in reads if m.group(7).strip() != '-']
    if fell:
        problems.append('the region read fell back to the whole-cache read (req %s: %s): the extension did not answer' % (fell[0].group(2), fell[0].group(7)[:160]))
    crosses = [m for m in LEVERN_KV_CROSS.finditer(container_text)]
    if mode == 'region':
        if crosses:
            problems.append('%d cross-check line(s) under %s=region' % (len(crosses), LEVERN_KV_READ_FLAG))
        return problems
    try:
        steps = int(env.get(LEVERN_KV_CROSS_FLAG, '1'))
    except ValueError:
        steps = 1
    expected = min(steps, wanted)
    if steps and not crosses:
        problems.append('%s=cross and no "%scross" line was logged: the region read was never compared with the whole-cache read (NOT EXERCISED)'
                        % (LEVERN_KV_READ_FLAG, LEVERN_KV_PREFIX))
    elif len(crosses) != expected:
        problems.append('%d cross-check line(s), %d expected (%s=%d over %d digest(s))' % (len(crosses), expected, LEVERN_KV_CROSS_FLAG, steps, wanted))
    for m in crosses:
        if int(m.group(3)) or m.group(7).strip() != '-':
            problems.append('cross-check of req %s: %s tensor(s) compared, %s MISMATCHED, fallback %s: the region read is NOT QUALIFIED'
                            % (m.group(1), m.group(2), m.group(3), m.group(7)[:120]))
        elif not int(m.group(2)):
            problems.append('cross-check of req %s compared no tensor' % m.group(1))
    return problems


def _float_or(text, default):
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def levern_facts(container_text):
    """The Lever N lines of a container log, parsed in order: route steps, scheduler steps and digests."""
    routes = [dict(req=m.group(1), start=int(m.group(2)), end=int(m.group(3)), prompt=int(m.group(4)), final=m.group(5) == '1',
                   wrote=m.group(6) == '1', programs=(m.group(8), m.group(9)), window=int(m.group(10)), source=m.group(11), captured=m.group(12))
              for m in LEVERN_ROUTE.finditer(container_text)]
    steps = [dict(n=int(m.group(1)), kind=m.group(2), seats=int(m.group(3)), req=m.group(4), start=m.group(5), tokens=int(m.group(6)),
                  end=m.group(7), prompt=m.group(8), final=m.group(9), reason=m.group(10), prev_kind=m.group(11), prev_ms=_float_or(m.group(12), 0.0),
                  f_eff=_float_or(m.group(15), None), need=_float_or(m.group(16), None), gap_ms=_float_or(m.group(17), None))
             for m in LEVERN_STEP.finditer(container_text)]
    digests = [dict(req=m.group(1), prompt=int(m.group(2)), tokens=m.group(3), slot=m.group(4), logits=m.group(5), kv=m.group(6))
               for m in LEVERN_DIGEST.finditer(container_text)]
    return dict(routes=routes, steps=steps, digests=digests)


def levern_route_problems(routes):
    """The ledger of every split prompt: continuity, alignment, one final step, the slot written only by it (wrote_slot is the measured count of
    slot writes), and no program compiled by a step beyond the drafter-window snapshot's own (after - before - window > 0, the four-card
    tripwire's B-A-W rule: the window's programs are keyed on the prompt's geometry and cannot be warmed). The warm requests are exempt from the
    program rule: they run before any trace is captured."""
    problems, by_request = [], {}
    for row in routes:
        by_request.setdefault(row['req'], []).append(row)
    for request, rows in by_request.items():
        # A prefix hit's first step starts at its Q (source CHECKPOINT, the merged route); every other first step starts at 0.
        cursor, label = (rows[0]['start'] if rows[0].get('source') == 'CHECKPOINT' else 0), 'prefill of %s' % request
        for index, row in enumerate(rows):
            source = row.get('source')
            if source is not None and (source in ('SCRATCH', 'PARKED')) != (index > 0):
                problems.append('%s: step %d came from source %s (a continuation is SCRATCH or PARKED, a first step COLD or CHECKPOINT)'
                                % (label, index + 1, source))
        for index, row in enumerate(rows):
            last = index == len(rows) - 1
            if row['start'] != cursor:
                problems.append('%s: step %d starts at %d, the previous ended at %d (a chunk was skipped or replayed)' % (label, index + 1, row['start'], cursor))
            if row['final'] != (row['end'] == row['prompt']):
                problems.append('%s: step %d ends at %d of %d with final=%d' % (label, index + 1, row['end'], row['prompt'], row['final']))
            if not row['final'] and row['end'] % LEVERN_CHUNK:
                problems.append('%s: a non-final step ends at %d, off the %d-token chunk boundary' % (label, row['end'], LEVERN_CHUNK))
            if row['wrote'] != row['final']:
                problems.append('%s: step %d (%d to %d) wrote_slot=%d: the decode slot is written by the final step alone'
                                % (label, index + 1, row['start'], row['end'], row['wrote']))
            if row['final'] and not last:
                problems.append('%s: a step follows its final step' % label)
            before, after = row['programs']
            if request != LEVERN_WARM_REQUEST and before != 'None' and after != 'None' and int(after) - int(before) - row['window'] > 0:
                problems.append('%s: step %d compiled %d program(s) beyond its %d window program(s) after the traces (the #48536 hang class)'
                                % (label, index + 1, int(after) - int(before) - row['window'], row['window']))
            cursor = row['end']
        if not rows[-1]['final'] and request != LEVERN_WARM_REQUEST:
            # an unfinished prompt is an aborted one (the cancel shape): legal, but nothing may follow it under this request
            pass
    return problems


LEVERN_GAP_SLACK_MS = 1000.0
LEVERN_CLIENT_DEADLINE_S = 240.0     # the client limit the owner accepted for the skew shape's slowest user
LEVERN_DEADLINE_MARGIN_S = 2.0
LEVERN_PINNED_SHARE = 0.9995
LEVERN_GOVERNOR = re.compile(r'\[PINDIAG\] lever N governor: ttft=(\S+) gap_floor=(\S+)')


def _step_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pinned(step):
    """Whether the deadline governor held the step at all the device: its unclamped need is at least 1.0 (logged beside f_eff since the floor
    landed); a pre-floor line has only the clamped f_eff, which reads 1.000 for a pinned share."""
    if step.get('need') is not None:
        return step['need'] >= LEVERN_PINNED_SHARE     # need prints with three decimals: 1.000 stands for [0.9995, 1.0005)
    return (step.get('f_eff') or 0.0) >= LEVERN_PINNED_SHARE


def levern_alternation_problems(steps, env, launched_gap_s=None):
    """Three rules over the scheduler's step lines.
    (1) Whenever a prefill step ran with decoders (seats > 0) and its request has another prefill step later, a decode step that served at least
    one seat must lie between them, unless the governor pinned the earlier step at all the device (need >= 1.0, or f_eff >= 0.9995 on a line
    that predates `need`: nothing is owed to the decoders there except the floor, rule 2) or the STATIC share is 1.0 (back-to-back chunks,
    today's stall chunked).
    (2) The decode-gap floor (G = QWEN_FAST_LEVERN_MAX_DECODE_GAP_S, default 8, 0 off; `launched_gap_s` is the value the container logged on its
    governor line and wins over the profile's): ONE stretch is every consecutive step, across requests, in which decoders were running and no
    decode round served a seat. Its length is the sum of its prefill steps' durations (a step's own duration is the NEXT line's prev_ms). It must
    not exceed G + the longest single step of the stretch (a step cannot be split) + 1 s. The problem is written when the stretch ENDS (a decode
    that served a seat, a step with no decoder, or the end of the log) so it states the whole stall and its n range, not the moment it crossed.
    Pre-governor lines (no f_eff) carry no floor.
    (3) A SHORT prefill (its request's first step had at most QWEN_FAST_LEVERN_SHORT_TOKENS, default 16384, left to compute) that ran with
    decoders must be followed by a decode round that served a seat before the next prefill step of any request (shorts are paced at one round
    whatever the share is, pinned or not). The decode round of a pass with no prefill pending is not logged as a step line; it shows as the next
    prefill line's prev=decode, which counts as the round."""
    share_one = str(env.get('QWEN_FAST_LEVERN_PREFILL_SHARE', '')) in ('1', '1.0') and not env.get('QWEN_FAST_LEVERN_ROUNDS')
    if share_one:
        return []
    try:
        gap_s = float(env.get('QWEN_FAST_LEVERN_MAX_DECODE_GAP_S', 8)) if launched_gap_s is None else float(launched_gap_s)
    except (TypeError, ValueError):
        gap_s = 8.0
    try:
        short_tokens = int(env.get('QWEN_FAST_LEVERN_SHORT_TOKENS', 16384))
    except (TypeError, ValueError):
        short_tokens = 16384
    problems, last_prefill, decoded = [], None, False
    previous, classes, short_owed = None, {}, None
    stretch = None      # dict(ms, longest, first_n, last_n, floor) while a decode-less stretch with decoders is open

    def close(end_label):
        if stretch is not None and gap_s and stretch['floor'] and stretch['ms'] > gap_s * 1000.0 + stretch['longest'] + LEVERN_GAP_SLACK_MS:
            problems.append('decoders went %.1f s without a decode round that served a seat (n=%d..%d, ended by %s; the floor is %g s + the longest '
                            'step %.1f s + %.0f s): the decode-gap floor did not hold'
                            % (stretch['ms'] / 1000.0, stretch['first_n'], stretch['last_n'], end_label, gap_s, stretch['longest'] / 1000.0,
                               LEVERN_GAP_SLACK_MS / 1000.0))

    for step in steps:
        if stretch is not None and step.get('prev_kind') == 'decode':
            # an unlogged decode round (a pass with no prefill pending) served the seats between the two lines: the stretch ended there
            close('an unlogged decode round before n=%d' % step['n'])
            stretch = None
        if previous is not None and previous['kind'] == 'prefill' and previous['seats'] > 0 and step.get('prev_kind') == 'prefill':
            took = step.get('prev_ms') or 0.0
            if stretch is None:
                stretch = dict(ms=0.0, longest=0.0, first_n=previous['n'], last_n=previous['n'], floor=False)
            stretch['ms'] += took
            stretch['longest'] = max(stretch['longest'], took)
            stretch['last_n'] = step['n']
            stretch['floor'] = stretch['floor'] or previous.get('f_eff') is not None
        if step['kind'] == 'decode':
            if step['seats'] >= 1:
                decoded, short_owed = True, None
                close('the decode round at n=%d' % step['n'])
                stretch = None
            previous = step
            continue
        if step['seats'] == 0:
            close('a step with no decoder at n=%d' % step['n'])
            stretch, short_owed = None, None
        prompt, start = _step_int(step.get('prompt')), _step_int(step.get('start'))
        if step['req'] not in classes and prompt is not None and start is not None:
            classes[step['req']] = prompt - start <= short_tokens
        if step['seats'] > 0 and short_owed is not None and not decoded and step.get('prev_kind') == 'prefill':
            problems.append('the short prefill step n=%d (%s) ran with %d decoder(s) and the next prefill step n=%d came with no decode round '
                            'between them: a short is owed its one round whatever the share' % (short_owed['n'], short_owed['req'], short_owed['seats'], step['n']))
        short_owed = None
        pinned = _pinned(step)      # the yield decision before this step used THIS line's share (govern() runs before the step is logged)
        if last_prefill is not None and last_prefill['req'] == step['req'] and last_prefill['seats'] > 0 and not decoded and not pinned:
            problems.append('two prefill steps of %s (n=%d and n=%d) ran back to back with %d decoder(s) running: the alternation did not yield'
                            % (step['req'], last_prefill['n'], step['n'], last_prefill['seats']))
        if step['seats'] > 0 and classes.get(step['req']):
            short_owed = step
        last_prefill, decoded = step, False
        previous = step
    close('the end of the log')
    return problems


def levern_governor_floor(container_text):
    """The decode-gap floor (seconds) the container logged on its governor line, or None when the line is missing (an image without the floor)."""
    found = LEVERN_GOVERNOR.findall(container_text)
    if not found:
        return None
    try:
        return float(found[-1][1])
    except ValueError:
        return None


LEVERN_SPLIT_FLOOR = 4096


def levern_exercise_problems(env, results, facts):
    """NOT_EXERCISED is not a pass. A levern_equal, levern_equal_long or levern_equal_busy row of at least 4,096 tokens must have been SPLIT:
    some request of that prompt length has route lines whose (start, end) sequence equals levern_policy.plan(P, decoding=...) (decoding False
    for the single-user rows, True for the busy ones), and levern_equal_busy must show a decode step that served a seat between two prefill
    steps of one of its prompts. A scheduler that never capped, or a route never reached, leaves no lines and used to read clean, and then the
    A1 comparison against the whole-prompt control compared two whole prompts."""
    try:
        cfg = levern_policy.config(env)
    except ValueError as failure:
        return ['the Lever N flags do not parse, so the expected split cannot be computed: %s' % failure]
    problems = []
    by_request = {}
    for row in facts['routes']:
        if row['req'] != LEVERN_WARM_REQUEST:
            by_request.setdefault(row['req'], []).append(row)
    share_one = str(env.get('QWEN_FAST_LEVERN_PREFILL_SHARE', '')) in ('1', '1.0') and not env.get('QWEN_FAST_LEVERN_ROUNDS')
    for name in LEVERN_ROW_TESTS:
        entry = results.get(name)
        if not isinstance(entry, dict) or 'error' in entry:
            continue
        busy = name == 'levern_equal_busy'
        for length in sorted(entry.get('prompts') or {}, key=int):
            prompt = int(length)
            if prompt < LEVERN_SPLIT_FLOOR:
                continue
            expected = [list(pair) for pair in levern_policy.plan(prompt, decoding=busy, step=cfg.step, solo=cfg.solo)]
            sequences = [[[row['start'], row['end']] for row in rows] for rows in by_request.values()
                         if rows and rows[0]['prompt'] == prompt]
            if expected not in sequences:
                problems.append('%s: the prompt of %d tokens was not split as the plan says (decoding=%s: %d steps, last %s); %d request(s) of that '
                                'length logged route lines of %s step(s): the lever never split it (NOT_EXERCISED)'
                                % (name, prompt, busy, len(expected), expected[-1], len(sequences),
                                   [len(sequence) for sequence in sequences] or 'none'))
        if busy and not share_one:
            lengths = {int(length) for length in entry.get('prompts') or {} if int(length) >= LEVERN_SPLIT_FLOOR}
            if lengths and not levern_interleaved(facts['steps'], lengths):
                problems.append('%s: no decode step that served a seat lay between two prefill steps of one prompt of %s tokens: nothing was interleaved'
                                % (name, sorted(lengths)))
    return problems


def levern_interleaved(steps, lengths):
    """Whether, for some prompt whose length is in `lengths`, a decode step with seats >= 1 lies between two prefill steps of one request."""
    requests = {step['req'] for step in steps if step['kind'] == 'prefill' and step['prompt'].isdigit() and int(step['prompt']) in lengths}
    for request in requests:
        index = [i for i, step in enumerate(steps) if step['kind'] == 'prefill' and step['req'] == request]
        for first, second in zip(index, index[1:]):
            if any(step['kind'] == 'decode' and step['seats'] >= 1 for step in steps[first + 1:second]):
                return True
    return False


def levern_problems(env, container_text, smoke):
    """(problems, facts) for the Lever N flags of the served profile `env` (the module docstring); ([], {}) when neither is set."""
    on = env.get(LEVERN_FLAG) == '1'
    audit = env.get(LEVERN_AUDIT_FLAG) == '1'
    if not on and not audit:
        return [], {}
    problems, facts = [], levern_facts(container_text)
    results = smoke or {}
    if on:
        merged = env.get('QWEN_PREFIX_REUSE') == '1'
        warmed = LEVERN_WARMED if not merged else LEVERN_WARMED.replace('steps=%d' % LEVERN_WARM_STEPS, 'steps=%d' % LEVERN_MERGED_WARM_STEPS)
        expected_markers = [(LEVERN_INSTALLED, 'the scheduler install line'), (LEVERN_PLATFORM, 'the platform wrap line (the policy that turns chunking '
                                                                                                 'off ran unwrapped)'),
                            (LEVERN_ROUTE_INSTALLED, 'the route install line'), (warmed, 'the route warm line with %d steps' % (
                                LEVERN_MERGED_WARM_STEPS if merged else LEVERN_WARM_STEPS))]
        if merged:
            expected_markers.append((LEVERN_GRAFT_CHUNKED, 'the install line of the prefix graft saying it runs chunked beside the cap'))
        for marker, what in expected_markers:
            if marker not in container_text:
                problems.append('Lever N is on and %s ("%s") was never logged: the lever did not engage' % (what, marker))
        lines = container_text.splitlines()
        warm = next((i for i, line in enumerate(lines) if warmed in line), None)
        unsafe = next((i for i, line in enumerate(lines) if UNSAFE_ALLOCATION in line), None)
        if warm is not None and unsafe is not None and unsafe < warm:
            problems.append('the Lever N route warm (log line %d) came after the first allocation made with a trace live (line %d)' % (warm + 1, unsafe + 1))
        refused = [line.strip()[:200] for line in lines if LEVERN_REFUSED in line]
        problems += ['a Lever N REFUSED line: %s' % line for line in refused[:4]]
        warm_steps = [row for row in facts['routes'] if row['req'] == LEVERN_WARM_REQUEST]
        if warm is not None and not merged and len(warm_steps) != LEVERN_WARM_STEPS:
            problems.append('the route warm logged %d step line(s), not %d' % (len(warm_steps), LEVERN_WARM_STEPS))
        problems += levern_route_problems(facts['routes'])
        served = [row for row in facts['routes'] if row['req'] != LEVERN_WARM_REQUEST]
        facts['split_prompts'] = len({row['req'] for row in served})
        facts['route_steps'] = len(served)
        problems += levern_exercise_problems(env, results, facts)
        if served:
            if not facts['steps']:
                problems.append('prompts were split (%d route lines) and no "lever N step" line was logged: the scheduler\'s alternation never ran'
                                % len(served))
            launched = levern_governor_floor(container_text) if merged else None
            if merged and env.get('QWEN_FAST_LEVERN_MAX_DECODE_GAP_S') is not None and launched is None:
                problems.append('the profile sets QWEN_FAST_LEVERN_MAX_DECODE_GAP_S and no "lever N governor: ttft= gap_floor=" line was logged: the image '
                                'does not carry the decode-gap floor, or the flag never reached the container')
            problems += levern_alternation_problems(facts['steps'], env, launched)
            if not any(step['kind'] == 'decode' and step['seats'] >= 1 for step in facts['steps']) and any(
                    step['kind'] == 'prefill' and step['seats'] >= 1 for step in facts['steps']):
                problems.append('decoders ran beside split prefills and no decode step was ever yielded between them')
        if merged:
            skew = results.get('concurrent8_skew')
            if isinstance(skew, dict) and 'error' not in skew:
                worst = max(skew.get('ttft_max_s') or 0.0, skew.get('last_first_token_s') or 0.0)
                if worst > LEVERN_CLIENT_DEADLINE_S - LEVERN_DEADLINE_MARGIN_S:
                    problems.append('concurrent8_skew: the slowest user waited %.1f s for a first token, over the %.0f s client deadline minus a %.0f s margin'
                                    % (worst, LEVERN_CLIENT_DEADLINE_S, LEVERN_DEADLINE_MARGIN_S))
            outs = [(m.group(1), int(m.group(2))) for m in LEVERN_PARK_OUT.finditer(container_text) if m.group(1) != LEVERN_WARM_REQUEST]
            ins = [(m.group(1), int(m.group(2))) for m in LEVERN_PARK_IN.finditer(container_text) if m.group(1) != LEVERN_WARM_REQUEST]
            facts['parks'] = len(outs)
            facts['unparks'] = len(ins)
            for request, at in ins:
                if (request, at) not in outs:
                    problems.append('prefill of %s was restored from a park at %d that was never taken (no matching "lever N park out" line)' % (request, at))
        for stall_name in STALL_TESTS:
            stall = results.get(stall_name)
            if isinstance(stall, dict) and 'error' not in stall:
                window = stall.get('window') or {}
                span = max([seat.get('window_s') or 0.0 for seat in stall.get('seat_windows') or [{}]] or [0.0])
                if window.get('seats') and span > 5.0 and window.get('seats_progressing') != window.get('seats'):
                    problems.append('%s: %s of %s decoding seats progressed inside the %.0f s arrival window: the prefill still froze the others'
                                    % (stall_name, window.get('seats_progressing'), window.get('seats'), span))
    if audit:
        problems += levern_kv_read_problems(env, container_text, facts['digests'])
        seen = {row['prompt'] for row in facts['digests']}
        for name in LEVERN_ROW_TESTS:
            entry = results.get(name)
            if isinstance(entry, dict) and 'error' not in entry:
                for length in sorted(entry.get('prompts') or {}, key=int):
                    if int(length) not in seen:
                        problems.append('%s: no digest line for the prompt of %s tokens (%s=1 logs one per finished prefill)' % (name, length, LEVERN_AUDIT_FLAG))
    return problems, facts


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


# tp4/w2 kill switch (w2_switch.py, docs/tp4-w2-kill-switch.md): the file w2.off latches the sequential step for the packed blocks W2 runs on (live) and keeps W2 out
# of the next attach. Outside the drill arm (QWEN_FAST_W2_OFF_AFTER, gate only) ANY kill line is a problem: a leftover w2.off made this arm run without W2 and it would
# read clean. In the drill arm: one drill line (the server wrote the file), then exactly one latch line, packed rounds before it and NONE after it (every packed round
# replays a trace that carries W2; the "[PINDIAG] packed extent round" line is logged once per packed round), a routed line per block that went sequential, and the
# engaged lines the W2 rules already demand (the attach before the file did not skip W2). tests hold these equal to w2_switch's.
W2_KILL_PREFIX = '[PINDIAG] w2 kill switch'
W2_KILL_LATCH = re.compile(r'\[PINDIAG\] w2 kill switch \S+ present: ')
W2_KILL_ATTACH = re.compile(r'\[PINDIAG\] w2 kill switch \S+ present at attach')
W2_KILL_ROUTED = '[PINDIAG] w2 kill switch routed the round to the sequential step'
W2_KILL_DRILL = '[PINDIAG] w2 kill switch drill (gate only): wrote '
W2_OFF_AFTER_FLAG = 'QWEN_FAST_W2_OFF_AFTER'
PACKED_ROUND_LINE = '[PINDIAG] packed extent round'


def w2_kill_problems(env, container_text):
    """[problem] for the W2 kill switch lines of a server log (see the block comment above): none for a log with no kill line and a profile without the drill."""
    lines = container_text.splitlines()
    kill = [index for index, line in enumerate(lines) if W2_KILL_PREFIX in line]
    drill_wanted = str((env or {}).get(W2_OFF_AFTER_FLAG) or '')
    if not kill:
        if drill_wanted:
            return ['%s=%s is set and no w2 kill switch line was logged: the drill never latched (fewer packed rounds than the trigger, or W2 never ran)' % (
                W2_OFF_AFTER_FLAG, drill_wanted)]
        return []
    if not drill_wanted:
        return ['the W2 kill switch logged a line outside the drill arm: a w2.off file was present, so this arm did not run W2 on all its rounds '
                '(a leftover file from an earlier arm?): %s' % lines[kill[0]].strip()[:200]]
    problems = []
    latch = [index for index in kill if W2_KILL_LATCH.search(lines[index])]
    attach = [index for index in kill if W2_KILL_ATTACH.search(lines[index])]
    drill = [index for index in kill if W2_KILL_DRILL in lines[index]]
    routed = [index for index in kill if W2_KILL_ROUTED in lines[index]]
    if attach:
        problems.append('the W2 kill switch line says the file was present at the attach (%d line(s)): the drill must write it mid-run, not before the engine started' % len(attach))
    if len(latch) != 1:
        problems.append('%d W2 kill switch latch lines under %s=%s: the switch latches exactly once' % (len(latch), W2_OFF_AFTER_FLAG, drill_wanted))
    if len(drill) != 1:
        problems.append('%d W2 kill drill lines under %s=%s: the server writes the flag file exactly once' % (len(drill), W2_OFF_AFTER_FLAG, drill_wanted))
    if latch and drill and drill[0] > latch[0]:
        problems.append('the W2 kill latch line precedes the drill line: the switch latched without the drill writing the file')
    if latch:
        rounds = [index for index, line in enumerate(lines) if PACKED_ROUND_LINE in line]
        before = [index for index in rounds if index < latch[0]]
        after = [index for index in rounds if index > latch[0]]
        if not before:
            problems.append('no packed round (%s) before the W2 kill latch: the drill proves nothing without W2 rounds first' % PACKED_ROUND_LINE)
        if after:
            problems.append('%d packed round line(s) (%s) after the W2 kill latch: a trace that carries W2 still replayed (first at line %d, latch at %d)' % (
                len(after), PACKED_ROUND_LINE, after[0] + 1, latch[0] + 1))
        if not [index for index in routed if index > latch[0]]:
            problems.append('no round was routed to the sequential step after the W2 kill latch (%s): nothing served the traffic that followed' % W2_KILL_ROUTED)
    return problems


LEVERN_KILL_PREFIX = '[PINDIAG] lever N kill switch'
DRAFTER_CHECKPOINT_FLAG = 'QWEN_FAST_DRAFTER_CHECKPOINT'
DRAFTER_CHECKPOINT_MARKER = '[PINDIAG] drafter checkpoint'


def drafter_checkpoint_problems(env, container_text):
    """[problem] for a profile that names a drafter checkpoint (drafter_checkpoint): the boot's verified line must name that id with verified=1. A profile
    that names none needs no line (the production log is unchanged)."""
    env = env or {}
    name = str(env.get(DRAFTER_CHECKPOINT_FLAG, '')).strip()
    if not name:
        return []
    lines = [line for line in container_text.splitlines() if DRAFTER_CHECKPOINT_MARKER in line]
    # A candidate's line says verified=1 (its config, manifests and weights hashed to the pins); the default has no pins and says verified=default.
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'drafter_checkpoints.json'), encoding='utf-8') as handle:
            default = json.load(handle).get('default')
    except (OSError, ValueError):
        default = None
    want = 'verified=default' if name == default else 'verified=1'
    if not any(('id=%s ' % name) in line and (want + ' ') in (line + ' ') for line in lines):
        return ['%s=%s is set and no "%s id=%s ... %s" line was logged: the pinned bytes of the candidate were never checked' % (
            DRAFTER_CHECKPOINT_FLAG, name, DRAFTER_CHECKPOINT_MARKER, name, want)]
    return []


def lever_engagement_problems(env, container_text, smoke=None, drill=False):
    """[problem] for the levers a served profile's `env` asks for, read from a gate arm's server log: the smoke's own per-lever rules (the
    engaged markers, no fall-back line, the audit lines exact, Lever N's install, warm and route ledger), applied by the prefix and serving gates
    too. A gate arm's texts can match with a lever silently not engaged (a graft mounted and never executed), and neither gate read the
    levers' markers before. Lever N's kill switch line is a problem outside the drill arm (`drill` True): a leftover levern.off would
    otherwise turn the lever off for every later arm and read as a clean run."""
    env = env or {}
    # (.extend, not +=: test_tp4_w2 holds that check() itself calls each rule exactly once.)
    problems = []
    problems.extend(sampdraft_problems(container_text, env))
    problems.extend(fusion_problems(container_text, env))
    problems.extend(u1_problems(env, container_text))
    problems.extend(sdpa_long_problems(env, container_text))
    problems.extend(sdpa_multi_problems(env, container_text))
    problems.extend(spread_problems(env, container_text))
    problems.extend(drafter_checkpoint_problems(env, container_text))
    problems.extend(w2_kill_problems(env, container_text))
    problems.extend(upload_p0_problems(env, container_text))
    lever, _ = levern_problems(env, container_text, smoke)
    problems.extend(lever)
    if env.get(LEVERN_FLAG) == '1' and not drill and LEVERN_KILL_PREFIX in container_text:
        problems.append('Lever N logged its kill switch line outside the drill arm: a levern.off file was present, so this arm did not run the lever '
                        '(a leftover file from an earlier arm?)')
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
    problems += fusion_problems(container_text, env)
    problems += u1_problems(env, container_text)
    problems += sdpa_long_problems(env, container_text)
    problems += sdpa_multi_problems(env, container_text)
    problems += spread_problems(env, container_text)
    problems += drafter_checkpoint_problems(env, container_text)
    problems += w2_kill_problems(env, container_text)
    problems += upload_p0_problems(env, container_text)
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
        round_host_found, round_host_facts_found = round_host_problems(
            env, container_text, STEADY_EIGHT_TEST in (smoke or {}) and 'error' not in smoke[STEADY_EIGHT_TEST])
        problems += round_host_found
        if round_host_facts_found.get('engaged') or round_host_facts_found.get('steps'):
            facts['round_host'] = dict(steps=len(round_host_facts_found['steps']), audits=len(round_host_facts_found['audits']),
                                       judged=round_host_facts_found.get('round_host_judged'),
                                       keyed_share=round_host_facts_found.get('round_host_keyed_share'))
        levern, levern_facts_found = levern_problems(env, container_text, smoke)
        problems += levern
        if levern_facts_found:
            facts['levern'] = dict((key, value) for key, value in levern_facts_found.items() if not isinstance(value, list))
        parked_found, parked_facts = parked_container_problems(env, container_text)
        problems += parked_found
        if parked_facts and smoke and 'parked_churn_long' in smoke:
            import parked_judge

            problems += parked_judge.memory_problems(container_text)
        if parked_facts:
            facts['parked'] = parked_facts
        octo_found, octo_facts = octo_container_problems(env, container_text)
        problems += octo_found
        if octo_facts:
            facts['octo'] = octo_facts
    if entry is not None:
        problems += traffic_problems(container_text, entry)
        problems += waiver_problems(container_text, entry)
    stamp = unqualified_stamp(container_text)
    if stamp:
        facts['unqualified'] = stamp
    if slide:
        if median is None:
            problems.append('no [PACKED-PUBLISH] round with a commit: the ramp commit time is unread (QWEN_FAST_PACKED_AUDIT?)')
        elif median > max_ramp_ms and audited_verify(env):
            # The verify audits inflate prepare_history (v423, v578: 51.7 ms with zero mismatches); the threshold is judged on the
            # audits-off arms of the same stack.
            facts['ramp_over_threshold_audited'] = round(median, 2)
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
        print('SMOKE_CHECK %s: the 262k evidence records were waived or a launch outside them ran; this result is a measurement, never evidence' % facts['unqualified'])
    for problem in problems:
        print('SMOKE_CHECK FAILED: %s' % problem)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
