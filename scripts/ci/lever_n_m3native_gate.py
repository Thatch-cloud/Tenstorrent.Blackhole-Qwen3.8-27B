"""Lever N M3native gate: is the native 64-row decode graft token-exact against each
user's own single-stream reference, at four packed users, while retiring the two-call
MLP and GDN-output wrappers?

Serves the SAME four-user packed round the qwen-fp2u-image.yml lane measures (the
base-1000..1003 prompt scheme, longctx_cycle_bench.py's own streaming/parsing logic
reused here via import) with the M3native graft mounted over the model sources
(lever_n_m3native_run_arm.sh), then asserts three things design section 5 (this
graft's own scope note) requires:

  1. Each stream's text is a byte-exact PREFIX match against its single-user
     reference (runner-evidence.local/packed-gate/single-user-*.json), compared up
     to the reference's own length - the packed round may run more decode tokens
     than the reference did, but everything the reference covers must agree exactly.
     Anything else means the native path diverged from the two-call one it replaces.
  2. The round's [PACKED-PHASE] trace_ms is reported (min/mean/max over every packed
     round in the log) - not asserted against a threshold here (that is a benchmark
     question, not a correctness one), just surfaced for the run to be read against
     the two-call baseline (run 35544598063: 1453 ms/round).
  3. The [PINDIAG] native_m3 marker is present (the positive control that the graft
     actually engaged, not that the model happened to produce the right bytes some
     other way) and every "[PINDIAG] native_m3 binder calls this round" line
     (model_batch.ModelBatch.run) reports zero calls for the retired binders - a
     silent fallback to the two-call MLP/GDN-output wrappers is exactly the failure
     mode this control exists to catch instead of measuring a graft that is not
     actually native.

Runs on the fast T16 path with speculation, like the four-user cycle bench: this is
what the target 200 tok/s/user work measures, and the packed-step/audit env vars
that produce [PACKED-PHASE] lines are set by lever_n_m3native_run_arm.sh at
`docker run`, not here - they are inherited by this process and by the vLLM server
subprocess it starts.

K64 KERNEL GRAFT (KOPGRAFT64, run_arm's optional mount block). A second, independent
graft: the batch-64 attn_decode_prep and nlp_concat_heads_decode C++ kernels
themselves, mounted only when KOPGRAFT64 is set (QWEN_FAST_NATIVE_ATTN=1 then reaches
the container). Not part of the pass/fail criteria above - this gate is run once
without it (the plain graft) and once with it, so `native_attn_engaged`
(NATIVE_ATTN_MARKER, the "[PINDIAG] native_attn engaged" line two_tile_bindings logs
once) is recorded in the JSON for the run to be read against, not asserted here.

REAL TEXT (--prompt-source real-text; default synthetic, which leaves every request, the
server argv, the report and the verdict exactly as before). Each user's prompt is a real coding
request built in the container by real_text_prompts.py from the image's own vLLM source, EXACTLY
--prompt-tokens long: the image pins every request's position to the arm's request context
(QWEN_DSPARK_REQUEST_CONTEXT = --prompt-tokens; frozen_combined_runtime.validate_target_option
refuses any other length), so --prompt-tokens + --max-tokens must fit --context. The
single-user-*.json references are synthetic, so none is loaded (--allow-missing-references is
required) and exactness is checked OFFLINE: each stream, its finish_reason, its completion and
served prompt token counts and its prompt's sha256 go into the report, and real_text_compare.py
compares a concurrent arm against a sequential (--users 1 --sequential-users N) arm on the same
image. --eos stop (the real-text default, and required there) lets a stream end at EOS. Outside
the default mode (real text, or --eos stop) the report also carries the acceptance report
(acceptance_report.py, from the full server.log after the server has stopped), the per-user
decode rate and the QWEN_* configuration, all diagnostic: none can fail an otherwise passing arm.
"""

import argparse
import ast
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from longctx_cycle_bench import stream_once  # reuse the exact streaming/parsing logic

BEGIN = '<<<M3NATIVE_GATE_JSON_BEGIN>>>'
END = '<<<M3NATIVE_GATE_JSON_END>>>'
LOG_BEGIN = '<<<M3NATIVE_GATE_LOG_BEGIN>>>'
LOG_END = '<<<M3NATIVE_GATE_LOG_END>>>'

BLOCK_SIZE = 64
MODEL = ('/models/hub/models--Qwen--Qwen3.8-27B/snapshots/'
         '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
NATIVE_M3_MARKER = '[PINDIAG] native_m3'
NATIVE_ATTN_MARKER = '[PINDIAG] native_attn engaged'
PACKED_PHASE_TRACE_MS = re.compile(r'\[PACKED-PHASE\][^\n]*\btrace_ms=([0-9.]+)')
BINDER_CALLS_LINE = re.compile(r'\[PINDIAG\] native_m3 binder calls this round: (\{.*\})')
# The binder-calls diagnostic (model_batch.ModelBatch.run) lists EVERY two-tile binder,
# retired or not: 'decode norm' (129 = two per layer plus the final norm) and
# 'full-attention forward' (16) are expected to be non-zero in every arm, and run
# 35559199392 was reported NOT PASSED on exactly those two. Only the binders the
# native path retires must be zero: the MLP and GDN-output wrappers under native_m3,
# and the sliced prep / two-tile concat guards once QWEN_FAST_NATIVE_ATTN retires them.
RETIRED_LABELS = ('MLP forward', 'GDN output projection', 'sliced attn_decode_prep', 'two-tile head concat')
# single-user-<run>.json (base 1000), single-user-<base>-<run>.json, or with a prompt-length
# segment single-user-<base>-p<tokens>-<run>.json. Unsegmented references are 32,768-token
# prompts; a segmented one is only used for a run of exactly that prompt length.
REFERENCE_NAME = re.compile(r'^single-user-(?:(\d{4})-)?(?:p(\d+)-)?\d+\.json$')


# The server log lines the gate's stdout keeps. Every '[PACKED' family passes (the per-user
# '[PACKED] request=' audit, '[PACKED-PHASE]', '[PACKED-COMMIT]', '[PACKED-COMMIT-HOST]' and
# any later '[PACKED-...]' line): run 35578180747 lost the commit-host attribution and the
# per-user audit because the old filter named four exact prefixes. The cap keeps a whole
# 33-round four-user run (about 50 kept lines per round) instead of cutting rounds 16-21
# out of the middle at 800. '[SEQ-PUBLISH]' is the sequential step's publication log, only
# under QWEN_FAST_SEQ_PUBLISH_LOG (serving_sequential_step): without it no line carries it.
DIAGNOSTIC_CAP = 4000
# A native death leaves no Python traceback: TT_FATAL / TT_THROW text, the C++ runtime's
# terminate message, the shell's signal report, or vLLM's engine-death notice are the
# only record (run 35579223088 had none of them in the kept lines).
CRASH_TEXT = ('FATAL', 'Segmentation', 'Aborted', 'Killed', 'terminate called', 'what():',
              'core dumped', 'died', 'Bus error', 'Illegal instruction', 'TT_THROW')


def select_diagnostic(lines, cap=DIAGNOSTIC_CAP):
    diagnostic = [line[:300] for line in lines
                  if '[PINDIAG]' in line or '[PACKED' in line or '[PHASE]' in line or '[GDN-SEQ-BLOCK' in line
                  or '[SEQ-PUBLISH]' in line
                  or 'ERROR' in line or 'Traceback' in line or any(crash in line for crash in CRASH_TEXT)]
    if len(diagnostic) > cap:
        omitted = len(diagnostic) - cap
        diagnostic = diagnostic[:cap // 2] + ['... %d diagnostic lines omitted' % omitted] + diagnostic[-(cap // 2):]
    return diagnostic


GENERIC_REFERENCE_TOKENS = 32768


# Image A capacity flags: each one's marker, required whenever the flag reaches this process.
# With QWEN_FAST_SINGLE_GATEUP set it takes precedence over QWEN_FAST_SKIP_BLOCK_STREAM
# (serving_runtime.register_reader_reason), so the 64-row skip marker is then not expected.
SINGLE_GATEUP_MARKERS = (
    '[PINDIAG] block stream skipped for the single gate/up copy: w_gate_up present on 0 of',
    '[PINDIAG] single gate/up copy: FusedT16Arm not installed; w_gate_up present on 0 of',
    '[PINDIAG] single gate/up copy: w_gate_up not built',
    '[PINDIAG] single gate/up copy: ff_norm gathers its own input',
)
# C1d (QWEN_FAST_C1_AGMM=1 with QWEN_FAST_SINGLE_GATEUP=1, lever_n_m3native_patch section F): the
# ff_norm skips its gather again, so its gathers-for-itself marker cannot fire; the layer's C1d
# marker and the MLP's C1d branch marker (logged inside the executed branch) take its place.
C1D_MARKERS = ('[PINDIAG] C1d: ff_norm skips its all-gather for the MLP AGMM',
               '[PINDIAG] prefill MLP C1d: two all_gather_matmul_prefill')


# Which prefill MLP branch ran is proved by the marker logged inside it (once per process):
# C1c is the default under QWEN_FAST_SINGLE_GATEUP=1 (lever_n_m3native_patch section F), so
# a rerun of a SINGLE_GATEUP tag (v104, v108, v115, v116) measures C1c, and a graft that fell
# through to C1 must not pass. QWEN_FAST_C1_LEGACY=1 keeps C1, whose 2D-branch marker is then
# required instead. C1d's branch precedes both, so under QWEN_FAST_C1_AGMM=1 only its markers.
C1C_MARKER = '[PINDIAG] prefill MLP C1c: one slice of x per 1024 rows'
C1_LEGACY_MARKER = '[PINDIAG] prefill MLP via w1/w3 2D branch'


# C1e (QWEN_FAST_C1_EXACT=1 with QWEN_FAST_SINGLE_GATEUP=1, lever_n_m3native_patch section J): the
# served fused SwiGLU AGMM on a per-layer rebuild of its packed weight in one scratch. The ff_norm skips
# its gather again, so its gathers-for-itself marker cannot fire; the scratch allocation (once, at
# load), the layer's C1e marker and the MLP's C1e branch marker (once, inside the branch) take its
# place. Every K-sharded prefill MLP call must take the C1e branch: one that fell through runs C1c (the
# default under SINGLE_GATEUP) and logs its marker, so under C1e the C1c, C1 and C1d markers are
# failures. QWEN_FAST_C1_EXACT_AUDIT=n (0..64) byte-checks the served weight of the first n layers at
# load and the scratch after the first n packs: the n-th line of each must say exact=True (a mismatch
# raises in the engine, before its line). C1e needs SINGLE_GATEUP (alone it is inert: the served path
# runs) and excludes C1_AGMM and C1_LEGACY (the graft refuses them at construction).
C1E_FLAG = 'QWEN_FAST_C1_EXACT'
C1E_AUDIT_FLAG = 'QWEN_FAST_C1_EXACT_AUDIT'
C1E_AUDIT_MAX = 64
C1E_MARKERS = ('[PINDIAG] C1e scratch allocated',
               '[PINDIAG] C1e: ff_norm skips its all-gather for the served fused op',
               '[PINDIAG] prefill MLP C1e: the served fused SwiGLU AGMM')
C1E_AUDIT_MARKER = '[PINDIAG] C1e audit'
C1E_PREMISE_MARKER = '[PINDIAG] C1e premise audit'
C1E_FORBIDDEN = (C1C_MARKER, C1_LEGACY_MARKER) + C1D_MARKERS


def c1e_audit_count(environ):
    """QWEN_FAST_C1_EXACT_AUDIT as the graft reads it (unset or empty: 0), or None if it would refuse it."""
    text = (environ.get(C1E_AUDIT_FLAG) or '').strip() or '0'
    try:
        count = int(text)
    except ValueError:
        return None
    return count if 0 <= count <= C1E_AUDIT_MAX else None


def c1e_markers(environ):
    markers = list(C1E_MARKERS)
    audit = c1e_audit_count(environ)
    if audit:
        markers += ['%s %d exact=True' % (C1E_PREMISE_MARKER, audit), '%s %d exact=True' % (C1E_AUDIT_MARKER, audit)]
    return markers


def c1e_problems(environ, log_text):
    """What a QWEN_FAST_C1_EXACT=1 run must not show, beyond its missing markers."""
    problems = []
    if environ.get('QWEN_FAST_SINGLE_GATEUP') != '1':
        problems.append('%s: needs QWEN_FAST_SINGLE_GATEUP=1 (alone it is inert and the served path runs)' % C1E_FLAG)
    for other in ('QWEN_FAST_C1_AGMM', 'QWEN_FAST_C1_LEGACY'):
        if environ.get(other) == '1':
            problems.append('%s: excludes %s=1 (the graft refuses the pair)' % (C1E_FLAG, other))
    if c1e_audit_count(environ) is None:
        problems.append('%s: an integer 0..%d, not %r' % (C1E_AUDIT_FLAG, C1E_AUDIT_MAX, environ.get(C1E_AUDIT_FLAG)))
    for marker in C1E_FORBIDDEN:
        if marker in log_text:
            problems.append('%s: every prefill MLP takes the C1e branch (%s logged)' % (C1E_FLAG, marker))
    return problems


def single_gateup_markers(environ):
    """SINGLE_GATEUP_MARKERS plus the executed prefill MLP branch's marker: C1c by default, C1's
    2D branch under QWEN_FAST_C1_LEGACY=1, or (QWEN_FAST_C1_AGMM=1) C1d's two in place of the
    ff_norm's gathers-for-itself marker, or (QWEN_FAST_C1_EXACT=1) C1e's three (and its audit lines)
    in place of it."""
    markers = list(SINGLE_GATEUP_MARKERS)
    if environ.get(C1E_FLAG) == '1':
        markers.remove('[PINDIAG] single gate/up copy: ff_norm gathers its own input')
        markers.extend(c1e_markers(environ))
    elif environ.get('QWEN_FAST_C1_AGMM') == '1':
        markers.remove('[PINDIAG] single gate/up copy: ff_norm gathers its own input')
        markers.extend(C1D_MARKERS)
    elif environ.get('QWEN_FAST_C1_LEGACY') == '1':
        markers.append(C1_LEGACY_MARKER)
    else:
        markers.append(C1C_MARKER)
    return markers


# QWEN_PREFILL_PROFILE_FLUSH=1 (M3NATIVE_PROFILE, lever_n_m3native_patch section G): the layer.py
# hook logs this once per process at its first ReadDeviceProfiler drain, inside the drain.
PREFILL_FLUSH_MARKER = '[PINDIAG] prefill profile flush: first flush'


SKIP_BLOCK_STREAM_MARKER = '[PINDIAG] block stream skipped for the 64-row block'
# QWEN_FAST_ROUND_B1=1 (M3NATIVE_ROUND_B1; build 1 of the round host-phase cuts): logged once
# per process by dflash_packed_proposal.note_round_b1 at the first B1 path that runs - the
# first feature publication, pair update or batched selection. Any run that commits a token
# publishes, so an arm with the flag and without this line ran an image without build 1.
ROUND_B1_MARKER = '[PINDIAG] round b1 engaged'
# QWEN_FAST_ROUND_B1_AUDIT=1 beside it (M3NATIVE_ROUND_B1_AUDIT; a correctness arm, never a
# timed one): dflash_packed_proposal's shadow audit re-does each B1 cut the flag-off way on
# the host and compares, because a wrong draft only lowers acceptance and the final text
# cannot see it. At four users the first audited round's line must be there (it says
# exact=True: a mismatch raises before it); any mismatch line fails, since the pair-proposal
# fallback can swallow the raise; and the last line's running counts must all be non-zero,
# so every audited cut was actually compared.
ROUND_B1_AUDIT_MARKER = '[PINDIAG] round b1 audit'
ROUND_B1_AUDIT_MISMATCH = '[PINDIAG] round b1 audit mismatch'
ROUND_B1_AUDIT_LINE = re.compile(r'\[PINDIAG\] round b1 audit ([0-9]+) exact=True select=([0-9]+) rope=([0-9]+) '
                                 r'retain=([0-9]+) borrowed=([0-9]+) release=([0-9]+)')
ROUND_B1_AUDIT_COUNTS = ('select', 'rope', 'retain', 'borrowed', 'release')
# QWEN_FAST_VERIFY_T1=1 (M3NATIVE_VERIFY_T1; the verify-trace tuning cuts of waves 1 and 2,
# verify_trace_t1.py): logged by the model_config graft where the two re-partitioned M = 64
# matmul configs are built (site=matmul_configs, lever_n_m3native_patch section A2) and by
# packed_verifier once per captured verify trace with the cuts that capture engaged
# (site=packed_verify). Either proves the flag reached code that reads it; the report lists
# which sites logged. At four users the wave-2 cuts must also have ENGAGED, or an ABAB would
# credit cuts that never ran (a stale image, the user batch off, the sampler guard tripped):
# every packed_verify line must report mask_once=1, shard_argmax=1 and, per GDN layer,
# direct_carry, last_carry and coalesced (no coalesce_fallback) - each cut named in
# QWEN_FAST_VERIFY_T1_SKIP must instead report 0. A wave-1 arm on an image without wave 2
# declares itself by skipping every VERIFY_T1_WAVE2_CUTS name; then only the graft's line is
# needed. An unknown skip name fails (the image refuses it too).
# QWEN_FAST_VERIFY_T1_AUDIT=1 beside it (a correctness arm, never a timed one): packed_verifier
# runs the pinned sampler beside the per-shard argmax in the same trace and compares every row
# every round; at four users the first round's line is required and any mismatch line fails.
VERIFY_T1_FLAG = 'QWEN_FAST_VERIFY_T1'
VERIFY_T1_AUDIT_FLAG = 'QWEN_FAST_VERIFY_T1_AUDIT'
VERIFY_T1_SKIP_FLAG = 'QWEN_FAST_VERIFY_T1_SKIP'
VERIFY_T1_MARKER = '[PINDIAG] verify t1 engaged'
VERIFY_T1_AUDIT_MARKER = '[PINDIAG] verify t1 audit'
VERIFY_T1_AUDIT_MISMATCH = '[PINDIAG] verify t1 audit mismatch'
VERIFY_T1_KEPT_SAMPLER = '[PINDIAG] verify t1 kept the pinned sampler'
VERIFY_T1_SITE = re.compile(r'\[PINDIAG\] verify t1 engaged site=([a-z_]+)')
VERIFY_T1_PACKED = re.compile(r'\[PINDIAG\] verify t1 engaged site=packed_verify((?: [a-z_]+=[0-9]+)*)')
VERIFY_T1_CUTS = ('matmul_configs', 'mask_once', 'direct_carry', 'last_carry', 'coalesce', 'shard_argmax')
VERIFY_T1_WAVE2_CUTS = ('mask_once', 'direct_carry', 'coalesce', 'shard_argmax')
# QWEN_FAST_VERIFY_T2=1 (M3NATIVE_VERIFY_T2; verify-trace cuts #1 packed conv windows and #2
# per-user K/V chains, verify_trace_t2.py): packed_verifier logs the engaged line once per
# captured verify trace with what the captured forward engaged. At four users every line must
# report windows=48 windows_fallback=0 (windows=0 when QWEN_FAST_VERIFY_T2_SKIP names it; #1
# engages only on the user-batched path, so it needs QWEN_FAST_GDN_USER_BATCH=1), kv_chains=32
# kv_fallback=0 kv_rows=<QWEN_FAST_VERIFY_T2_KV_ROWS, default 64> warm_chain=single (the warm
# forward's one chain; kv_chains=0 kv_rows=0 warm_chain=none when skipped). Any fell-back,
# kv-shared or audit-mismatch line fails the arm at any user count: a kv-shared round seen
# before drafting (site=proposal_rows) is served by the exact sequential step, one first seen
# at the step (site=ineligible) is refused - its requests fail, nothing is written - and
# either way the premise broke and the timing is contaminated. QWEN_FAST_VERIFY_T2_AUDIT=1 (a
# correctness arm) needs the first audited round's line at four users.
VERIFY_T2_FLAG = 'QWEN_FAST_VERIFY_T2'
VERIFY_T2_AUDIT_FLAG = 'QWEN_FAST_VERIFY_T2_AUDIT'
VERIFY_T2_SKIP_FLAG = 'QWEN_FAST_VERIFY_T2_SKIP'
VERIFY_T2_KV_ROWS_FLAG = 'QWEN_FAST_VERIFY_T2_KV_ROWS'
VERIFY_T2_MARKER = '[PINDIAG] verify t2 engaged'
VERIFY_T2_AUDIT_MARKER = '[PINDIAG] verify t2 audit'
VERIFY_T2_AUDIT_MISMATCH = '[PINDIAG] verify t2 audit mismatch'
VERIFY_T2_FALLBACK = '[PINDIAG] verify t2 fell back'
VERIFY_T2_KV_SHARED = '[PINDIAG] verify t2 kv shared'
VERIFY_T2_PACKED = re.compile(r'\[PINDIAG\] verify t2 engaged site=packed_verify((?: [a-z_]+=[a-z0-9_]+)*)')
VERIFY_T2_CUTS = ('windows', 'kv_chains')
DRAFT_BF8_MARKER = 'projections dtype=bf8 x36'
LEDGER_MARKERS = ('[MEMLEDGER] phase=P7 ', ' check=residual status=')
# QWEN_FAST_SDPA_MODES (optimisation/ttnn-op/sdpa_decode_qwen: 'tail' stage 1, 'share' stage 3).
# The two [PINDIAG] lines are emitted inside pooled_attention_replay.apply_sdpa_modes (the
# loaded _ttnncpp.so was checked for the factory branch; a replay reader's configs were
# rewritten), never at install or mount time. The [QWEN-SDPA] line is the grafted factory's own
# log_info (F4), printed when it builds a program in that mode: the C++ branch itself ran, not
# just the Python that selects it. With share the line must carry flag 0x2 (0x3 with tail),
# which the reader sets only on bundles of more than one entry, so kv_share=true was built.
# Stage 4 (optimisation/ttnn-op/sdpa_decode_slice, graft K64i): 'slice' is flag 0x4 (0x7 with
# tail,share), 'readahead' 0x8 (needs share). With either, the factory's F18 line, logged once
# per program built with 0x4 or 0x8, is required as well: the stage-4 branch that selects the
# slice kernels ran (a stage-3 .so refuses both flags by TT_FATAL and never logs it).
SDPA_MODES_MARKERS = ('[PINDIAG] sdpa qwen-modes binary ', '[PINDIAG] sdpa qwen-modes modes=tail ',
                      '[QWEN-SDPA] flags=0x1 ')
SDPA_MODE_FLAGS = {'tail': 0x1, 'share': 0x2, 'slice': 0x4, 'readahead': 0x8}
SDPA_SLICE_MARKER = '[QWEN-SDPA] q-slice rows_per_kv='   # apply_factory_slice.SLICE_LOG_MARKER (F18)
SDPA_STAGE4_MODES = frozenset({'slice', 'readahead'})
GDN_ALL_BATCHED = re.compile(r'gdn user_batched calls this captured forward: ([1-9][0-9]*) of ([0-9]+) GDN layers')
# K5-A (QWEN_FAST_GDN_SEQ_BLOCK=1, gdn_seq_block; needs QWEN_FAST_GDN_USER_BATCH=1): model_batch logs how
# many of a captured forward's GDN layers ran the sequential-block launch, and at which level,
# counted from the result dicts gdn_user_batch_conv marks only after the launch returned (execution,
# not just a flag read). At four users one capture must show every GDN layer at
# QWEN_FAST_GDN_SEQ_BLOCK_LEVEL (default 0), and no capture may show another level or only some layers.
# QWEN_FAST_GDN_SEQ_BLOCK_AUDIT=<layers> (a correctness arm): packed_verifier logs
# '[GDN-SEQ-BLOCK-AUDIT] layer=L user=U mismatches=N' per audited layer and user after every replay;
# every line must say 0 at any user count, and at four users each listed layer needs lines for all
# four users (an audit that logged nothing is not a pass).
GDN_SEQ_BLOCK_FLAG = 'QWEN_FAST_GDN_SEQ_BLOCK'
GDN_SEQ_BLOCK_LEVEL_FLAG = 'QWEN_FAST_GDN_SEQ_BLOCK_LEVEL'
GDN_SEQ_BLOCK_AUDIT_FLAG = 'QWEN_FAST_GDN_SEQ_BLOCK_AUDIT'
GDN_SEQ_BLOCK = re.compile(r'gdn seq_block calls this captured forward: ([1-9][0-9]*) of ([0-9]+) GDN layers '
                           r'level=([0-9]+)')
GDN_SEQ_BLOCK_AUDIT_LINE = re.compile(r'\[GDN-SEQ-BLOCK-AUDIT\] layer=([0-9]+) user=([0-9]+) mismatches=([0-9]+)')
GDN_SEQ_BLOCK_LEVEL_BITS = 4
LEDGER_RESIDUAL = re.compile(r'\[MEMLEDGER\] phase=P7 [^\n]*check=residual status=([a-zA-Z]+)')
# QWEN_FAST_PUBLISH_PREWARM=1 (M3NATIVE_PUBLISH_PREWARM; publish_prewarm.py, the k5dbg verdict's fix): every
# request's admission logs one '[PINDIAG] publish prewarm pairs=P count=N ms=M program_cache=A->B' line (pairs=none
# count=0 once every captured width was warmed earlier in the process), or '[PINDIAG] publish prewarm skipped' in the
# prefill ramp. The flag needs such a line, and one that warmed something (count >= 1): an image without the module,
# or a hook that never ran, logs neither. QWEN_FAST_SEQ_PUBLISH_LOG=1 (M3NATIVE_SEQ_PUBLISH_LOG; serving_sequential_
# step, logging only): every sequential step that returned logs one '[SEQ-PUBLISH] request=R rows=' line, and under
# QWEN_FAST_PHASE_LOG (every m3native arm) each one also ends with '[PHASE] step R end', so the two counts must be
# equal - an image without the logging logs steps and no lines; without the phase log nothing is compared. Both are
# reported under 'publish_prewarm' (every prewarm line parsed, both counts) only when one is set.
PUBLISH_PREWARM_FLAG = 'QWEN_FAST_PUBLISH_PREWARM'
PUBLISH_PREWARM_MARKER = '[PINDIAG] publish prewarm pairs='
PUBLISH_PREWARM_SKIPPED = '[PINDIAG] publish prewarm skipped'
PUBLISH_PREWARM_LINE = re.compile(r'\[PINDIAG\] publish prewarm pairs=(\S+) count=([0-9]+) ms=([0-9.]+) '
                                  r'program_cache=(\S+)->(\S+)')
SEQ_PUBLISH_LOG_FLAG = 'QWEN_FAST_SEQ_PUBLISH_LOG'
SEQ_PUBLISH_MARKER = '[SEQ-PUBLISH] request='
SEQ_PUBLISH_STEP = re.compile(r'\[SEQ-PUBLISH\] request=\S+ rows=')
PHASE_STEP_END = re.compile(r'\[PHASE\] step \S+ end ')


# Lever #2 (QWEN_FAST_GDN_PREFILL_CONV=1, lever_n_m3native_patch section H): the graft logs the
# engaged marker inside the op's own branch, once per prefill chunk, with the calls the previous
# chunk made, and (from chunk 2 on) a completion line when a chunk reaches chunk 1's count. Under
# QWEN_FAST_GDN_PREFILL_CONV_AUDIT=<n> the first audit line must say exact=True (a mismatch
# raises in the engine). The flag fails on: any FIR-fallback line; a completed chunk that did not
# engage every GDN layer (GDN_LAYERS); a last chunk with no completion line; and fewer engaged
# chunks than users x ceil(prompt_tokens / PREFILL_CONV_CHUNK_TOKENS) - a prefill that engaged
# only its tail or masked chunks while the full chunks kept the FIR would otherwise pass with
# every count equal. A partly-replaced prefill is not the thing under test.
PREFILL_CONV_FLAG = 'QWEN_FAST_GDN_PREFILL_CONV'
PREFILL_CONV_AUDIT_FLAG = 'QWEN_FAST_GDN_PREFILL_CONV_AUDIT'
PREFILL_CONV_MARKER = '[PINDIAG] GDN prefill conv engaged'
PREFILL_CONV_FALLBACK = '[PINDIAG] GDN prefill conv fell back to the FIR'
PREFILL_CONV_AUDIT_MARKER = '[PINDIAG] GDN prefill conv audit'
PREFILL_CONV_COMPLETE_MARKER = '[PINDIAG] GDN prefill conv chunk complete'
PREFILL_CONV_CHUNK = re.compile(r'\[PINDIAG\] GDN prefill conv engaged: chunk ([0-9]+) previous_chunk_calls=([0-9]+)')
PREFILL_CONV_COMPLETE = re.compile(r'\[PINDIAG\] GDN prefill conv chunk complete: chunk ([0-9]+) calls=([0-9]+)')
GDN_LAYERS = 48                   # GDN layers per forward: the spec gate steps the call counter by 48
PREFILL_CONV_CHUNK_TOKENS = 2048  # the model's prefill chunk (forward_prefill T)


# Prefill lever #1 (QWEN_FAST_SDPA_PF=1, M3NATIVE_SDPA_PF on the arm; lever_n_m3native_patch section I):
# the grafted attention/tp.py logs its [PINDIAG] line once per process, after its binary check passed,
# with the word it sends; the K64g factory logs '[QWEN-SDPA-PF] flags=0x<flags> kv_chain=1 chains=..
# members=.. order=..' each time it builds a chain program - the C++ branch itself ran, not just the
# Python that selects it (a stock or pre-K64g _ttnncpp.so never prints it). Both must carry the flag set
# QWEN_FAST_SDPA_PF_FLAGS names (default 0x3); a value that is not a production set is itself a problem
# (the graft refuses it at construction). Spec 6.4's factory marker also names the 2048-row topology,
# 'chains=16 members=96': the model's 2048-token prefill chunks carry almost all of the saving, and
# neither the Python eligibility test nor the factory's envelope pins the grid or the local head counts,
# so that tail is the runtime proof the model built the topology card M qualified. It is required
# whenever a prompt holds a 2048-token chunk, and every factory line must carry the requested flags and
# one of the Q1-qualified topologies (SDPA_PF_TOPOLOGY, test_sdpa_prefill_chain_card_m.expected_chains).
SDPA_PF_FLAG = 'QWEN_FAST_SDPA_PF'
SDPA_PF_FLAGS_FLAG = 'QWEN_FAST_SDPA_PF_FLAGS'
SDPA_PF_TAG = 0x5EFA0000
SDPA_PF_PRODUCTION_FLAGS = (0x1, 0x3, 0x5, 0x7)
SDPA_PF_DEFAULT_FLAGS = 0x3
SDPA_PF_PINDIAG = '[PINDIAG] sdpa prefill kvchain flags='
SDPA_PF_FACTORY_MARKER = '[QWEN-SDPA-PF] flags='
SDPA_PF_TOPOLOGY = {2048: (16, 96), 1024: (8, 48), 512: (4, 24)}   # rows -> (chains, members), card-M Q1
SDPA_PF_FACTORY_LINE = re.compile(r'\[QWEN-SDPA-PF\] flags=(0x[0-9a-f]+) kv_chain=1 chains=([0-9]+) members=([0-9]+) ')


def sdpa_pf_flags(environ):
    """The flag set QWEN_FAST_SDPA_PF_FLAGS names (unset or empty: 0x3), or None if not a production set."""
    text = (environ.get(SDPA_PF_FLAGS_FLAG) or '').strip() or '0x%x' % SDPA_PF_DEFAULT_FLAGS
    try:
        flags = int(text, 0)
    except ValueError:
        return None
    return flags if flags in SDPA_PF_PRODUCTION_FLAGS else None


def sdpa_pf_markers(environ, prompt_tokens=None):
    """The graft's [PINDIAG] line with the word, and the factory's line with the flags - and, when a
    prompt of `prompt_tokens` (None: assume one) holds a 2048-token chunk, its 2048-row topology."""
    flags = sdpa_pf_flags(environ)
    if flags is None:
        return [SDPA_PF_PINDIAG, SDPA_PF_FACTORY_MARKER]
    factory = '%s%#x kv_chain=1 ' % (SDPA_PF_FACTORY_MARKER, flags)
    if prompt_tokens is None or int(prompt_tokens) >= PREFILL_CONV_CHUNK_TOKENS:
        factory += 'chains=%d members=%d ' % SDPA_PF_TOPOLOGY[PREFILL_CONV_CHUNK_TOKENS]
    return ['%s%#x ' % (SDPA_PF_PINDIAG, SDPA_PF_TAG | flags), factory]


def sdpa_pf_problems(environ, log_text):
    """Every factory line carries the requested flags and a card-M-qualified (chains, members)."""
    flags = sdpa_pf_flags(environ)
    qualified = set(SDPA_PF_TOPOLOGY.values())
    problems = set()
    for match in SDPA_PF_FACTORY_LINE.finditer(log_text):
        line_flags, chains, members = int(match.group(1), 16), int(match.group(2)), int(match.group(3))
        if flags is not None and line_flags != flags:
            problems.add('%s: every chain program carries flags %#x (a factory line has %#x)' % (SDPA_PF_FLAG, flags,
                                                                                              line_flags))
        if (chains, members) not in qualified:
            problems.add('%s: chains=%d members=%d is not a card-M-qualified topology (%s)' % (
                SDPA_PF_FLAG, chains, members, ', '.join('%d rows %d/%d' % (rows, pair[0], pair[1])
                                                         for rows, pair in sorted(SDPA_PF_TOPOLOGY.items()))))
    return sorted(problems)


def prefill_conv_markers(environ):
    markers = [PREFILL_CONV_MARKER]
    try:
        audited = int(environ.get(PREFILL_CONV_AUDIT_FLAG) or 0)
    except ValueError:
        audited = 0
    if audited > 0:
        markers.append(PREFILL_CONV_AUDIT_MARKER + ' 1 exact=True')
    return markers


def prefill_conv_chunk_calls(log_text):
    """Each completed chunk's call count, from the NEXT chunk's marker (the first marker has none)."""
    return [int(match.group(2)) for match in PREFILL_CONV_CHUNK.finditer(log_text) if match.group(1) != '1']


def prefill_conv_required_chunks(users, prompt_tokens, chunk_tokens=PREFILL_CONV_CHUNK_TOKENS):
    """The engaged-chunk markers a run of `users` prompts of `prompt_tokens` must at least log."""
    if not prompt_tokens:
        return 0
    return users * -(-int(prompt_tokens) // chunk_tokens)


def prefill_conv_summary(log_text, gdn_layers=GDN_LAYERS, required_chunks=0):
    """What the markers say: each completed chunk's calls, the engaged chunks, and the last chunk."""
    chunks = [int(match.group(1)) for match in PREFILL_CONV_CHUNK.finditer(log_text)]
    complete = {int(match.group(1)): int(match.group(2)) for match in PREFILL_CONV_COMPLETE.finditer(log_text)}
    last = max(chunks) if chunks else None
    return dict(chunk_calls=prefill_conv_chunk_calls(log_text), engaged_chunks=len(chunks),
                required_chunks=required_chunks, gdn_layers=gdn_layers, last_chunk=last,
                last_chunk_calls=complete.get(last) if last is not None and last > 1 else None,
                complete_calls=sorted(set(complete.values())))


def prefill_conv_problems(log_text, gdn_layers=GDN_LAYERS, required_chunks=0):
    problems = []
    if PREFILL_CONV_FALLBACK in log_text:
        problems.append('%s: no FIR fallback (%s)' % (PREFILL_CONV_FLAG, PREFILL_CONV_FALLBACK))
    summary = prefill_conv_summary(log_text, gdn_layers, required_chunks)
    counts = summary['chunk_calls']
    if any(count != gdn_layers for count in counts):
        problems.append('%s: every completed prefill chunk engages all %d GDN layers (previous_chunk_calls %s)'
                        % (PREFILL_CONV_FLAG, gdn_layers, sorted(set(counts))))
    if any(calls != gdn_layers for calls in summary['complete_calls']):
        problems.append('%s: completion lines report %s calls, not %d' % (PREFILL_CONV_FLAG, summary['complete_calls'],
                                                                         gdn_layers))
    last = summary['last_chunk']
    if last is not None and last > 1 and summary['last_chunk_calls'] != gdn_layers:
        problems.append('%s: the last chunk (%d) never completed %d GDN layer calls (%s)'
                        % (PREFILL_CONV_FLAG, last, gdn_layers, PREFILL_CONV_COMPLETE_MARKER))
    if summary['engaged_chunks'] < required_chunks:
        problems.append('%s: %d engaged prefill chunks, at least %d expected (users x ceil(prompt / %d)): the '
                        'other chunks kept the FIR' % (PREFILL_CONV_FLAG, summary['engaged_chunks'], required_chunks,
                                                       PREFILL_CONV_CHUNK_TOKENS))
    return problems


# Variable-user packed rounds, M0 and M1 (arms R1, diagnostic, and R2, timed). Each line is logged
# only once the path its flag names has run, so a flag set on an image without it fails here:
#   QWEN_FAST_PAIR_MASK_REFRESH=1  dflash_proposal_trace logs '[PINDIAG] pair mask refresh' once per
#       process, at the first refresh of a packed pair's mask (a pair exists from two users).
#   QWEN_FAST_PAIR_MASK_AUDIT=1    '[PACKED-PROPOSE] mask round=R pair=[a,b] intact=0|1 mismatched=N
#       chip=K' per chip per pair update. A clobbered mask is the finding, never a failure: the report
#       lists it (pair_mask_audit: the first round each pair read intact=0).
#   QWEN_FAST_PAIRS_PACKED_ONLY=1  dflash_packed_proposal_coordinator logs '[PINDIAG] pairs packed
#       only' once, the first round it splits a pair the policy serves sequentially.
#   QWEN_FAST_PADDED_PROBE=1       padded_probe's lines on packed rounds 3 and 20; round 3's are
#       required. Any exact=0 or exact=error pattern, any idle_carry_intact=0 and any page-0 hit fails
#       the arm (G-pad: a mismatch stops Option A). exact=refused - a pattern needing three idle
#       segments, which page 0 cannot hold - is listed, never a failure.
PAIR_MASK_REFRESH_FLAG = 'QWEN_FAST_PAIR_MASK_REFRESH'
PAIR_MASK_AUDIT_FLAG = 'QWEN_FAST_PAIR_MASK_AUDIT'
PAIRS_PACKED_ONLY_FLAG = 'QWEN_FAST_PAIRS_PACKED_ONLY'
PADDED_PROBE_FLAG = 'QWEN_FAST_PADDED_PROBE'
PAIR_MASK_REFRESH_MARKER = '[PINDIAG] pair mask refresh'
PAIR_MASK_AUDIT_MARKER = '[PACKED-PROPOSE] mask round='
PAIR_MASK_AUDIT_LINE = re.compile(r'\[PACKED-PROPOSE\] mask round=([0-9]+|None) pair=\[([0-9]+|None),([0-9]+|None)\] '
                                  r'intact=([01]) mismatched=([0-9]+) chip=([0-9]+)')
PAIRS_PACKED_ONLY_MARKER = '[PINDIAG] pairs packed only'
PADDED_PROBE_MARKER = '[PINDIAG] padded probe round=3 '
PADDED_PROBE_LINE = re.compile(r'\[PINDIAG\] padded probe round=([0-9]+) live=([0-9,]+) exact=([a-z0-9]+) '
                               r'trace_ms=(\S+) idle_carry_intact=(\S+)(?: idle=(\S+))?(?: differ=(\S+))?')
PADDED_PROBE_PAGE0_HIT = '[PINDIAG] padded probe page0 hit'


def variable_user_markers(environ, users):
    """{flag: [marker]} for the M0/M1 flags set in `environ` (see above), and M2's (below)."""
    on = lambda name: environ.get(name) == '1'
    required = {}
    if users >= 2:
        for flag, marker in ((PAIR_MASK_REFRESH_FLAG, PAIR_MASK_REFRESH_MARKER),
                             (PAIR_MASK_AUDIT_FLAG, PAIR_MASK_AUDIT_MARKER),
                             (PAIRS_PACKED_ONLY_FLAG, PAIRS_PACKED_ONLY_MARKER),
                             (PADDED_PROBE_FLAG, PADDED_PROBE_MARKER)):
            if on(flag):
                required[flag] = [marker]
    if on(PADDED_BLOCK_FLAG):
        # M2: the block's admission at attach, at any stream count (refused off the M3 block), and at
        # least one round it served padded - an arm that never padded a round has not tested M2, even
        # when nothing else it logged is wrong (the admission alone proves the block was built, not used).
        required[PADDED_BLOCK_FLAG] = [PADDED_ADMITTED_MARKER, PADDED_ROUND_MARKER]
    return required


def pair_mask_audit_summary(log_text):
    """The audit lines: how many, how many read intact=0, and per pair the first clobbered round."""
    lines = [match.groups() for match in PAIR_MASK_AUDIT_LINE.finditer(log_text)]
    if not lines:
        return None
    first = {}
    for round_number, slot_a, slot_b, intact, mismatched, chip in lines:
        if intact == '0':
            first.setdefault('%s,%s' % (slot_a, slot_b), round_number)
    return dict(lines=len(lines), clobbered=sum(1 for line in lines if line[3] == '0'),
                max_mismatched=max(int(line[4]) for line in lines), first_clobbered_round=first)


def padded_probe_lines(log_text):
    """Every probe line as a dict (round, live, exact, trace_ms, idle_carry_intact, idle, differ)."""
    return [dict(round=int(match.group(1)), live=match.group(2), exact=match.group(3), trace_ms=match.group(4),
                 idle_carry_intact=match.group(5), idle=match.group(6), differ=match.group(7))
            for match in PADDED_PROBE_LINE.finditer(log_text)]


def variable_user_report(environ, users, log_text):
    """What the M0/M1 flags' lines say, and M2's under 'padded_block' when its flag is set; their
    problems under 'problems', for the caller."""
    problems = []
    audit = pair_mask_audit_summary(log_text) if environ.get(PAIR_MASK_AUDIT_FLAG) == '1' else None
    probe = None
    if environ.get(PADDED_PROBE_FLAG) == '1':
        probe = padded_probe_lines(log_text)
        for line in probe:
            if line['exact'] in ('0', 'error'):
                problems.append('%s: round %d live=%s exact=%s differ=%s' % (
                    PADDED_PROBE_FLAG, line['round'], line['live'], line['exact'], line['differ']))
            if line['idle_carry_intact'] == '0':
                problems.append('%s: round %d live=%s idle_carry_intact=0' % (PADDED_PROBE_FLAG, line['round'],
                                                                              line['live']))
        hits = log_text.count(PADDED_PROBE_PAGE0_HIT)
        if hits:
            problems.append('%s: no live table holds page 0 in its used range (%d %s lines)' % (
                PADDED_PROBE_FLAG, hits, PADDED_PROBE_PAGE0_HIT))
    report = dict(pair_mask_audit=audit, padded_probe=probe, problems=problems)
    padded = padded_block_report(environ, log_text)
    if padded is not None:
        problems.extend(padded.pop('problems'))
        if padded:
            report['padded_block'] = padded
    return report


# Variable-user packed rounds, M2 (QWEN_FAST_PADDED_BLOCK=1; arms R3, G1 and G2): the 64-row block also
# serves padded_min_users..3 live users as one pass, the missing segments idle on page 0 (packed_verifier.py,
# VARIABLE-USER ROUNDS). The block logs its admission once at attach - required, and its min_users must be
# the QWEN_FAST_PADDED_BLOCK_MIN_USERS asked for (default 2) - and one line per padded round (at least one
# required: variable_user_markers); each round of
# that many live users served sequentially instead logs why, and whether it was eligible (every live user
# with a block round of budget left, inside the block's family: serving_packed_step.note_padded_skip).
# Failures: any page-0 line (a live table holding page 0 in its used range, or one the check could not
# map), any idle segment asked to commit a prefix, any refusal by a backstop (site=ineligible or
# site=verify: that round's requests failed), any round refused outright (serving_packed_step.
# refuse_round), and padded rounds under PADDED_ENGAGEMENT of the eligible rounds (the padded ones plus the
# eligible skips). A refusal at site=proposal_rows by the idle slot rule is a skip, never a failure.
PADDED_BLOCK_FLAG = 'QWEN_FAST_PADDED_BLOCK'
PADDED_MIN_USERS_FLAG = 'QWEN_FAST_PADDED_BLOCK_MIN_USERS'
PADDED_MIN_USERS_DEFAULT = 2
PADDED_ADMITTED_MARKER = '[PINDIAG] packed padded block admitted'
PADDED_ADMITTED_LINE = re.compile(r'\[PINDIAG\] packed padded block admitted min_users=([0-9]+) users=([0-9]+) '
                                  r'max_idle=([0-9]+) carries_in_place=([01])')
PADDED_ROUND_MARKER = '[PINDIAG] packed padded round'
PADDED_ROUND_LINE = re.compile(r'\[PINDIAG\] packed padded round live=([0-9]+) round=([0-9]+) segments=(\S+) '
                               r'idle=(\S+) padded=([0-9]+)')
PADDED_SKIPPED_MARKER = '[PINDIAG] packed padded skipped'
PADDED_SKIPPED_LINE = re.compile(r'\[PINDIAG\] packed padded skipped live=([0-9]+) eligible=([01]) reason=(\S*)')
PADDED_REFUSED_MARKER = '[PINDIAG] packed padded refused'
PADDED_REFUSED_LINE = re.compile(r'\[PINDIAG\] packed padded refused site=(\S+) ([^\n]*)')
PADDED_PAGE0_MARKER = '[PINDIAG] packed padded page0'
PADDED_IDLE_COMMIT_MARKER = '[PINDIAG] packed padded idle commit'
REFUSED_ROUND_MARKER = 'A round the block cannot serve'
PADDED_ENGAGEMENT = 0.8


def _first_line(log_text, marker):
    return log_text[log_text.index(marker):].split(chr(10), 1)[0][:200]


def padded_block_summary(log_text):
    """What the M2 lines say: the admission, the padded rounds by live count, the sequential rounds of
    that many live users (eligible or not, and why), and the padded share of the eligible rounds."""
    admitted = PADDED_ADMITTED_LINE.search(log_text)
    rounds = [match.groups() for match in PADDED_ROUND_LINE.finditer(log_text)]
    skipped = [match.groups() for match in PADDED_SKIPPED_LINE.finditer(log_text)]
    by_live, reasons = {}, {}
    for live, *rest in rounds:
        by_live[live] = by_live.get(live, 0) + 1
    for live, eligible, reason in skipped:
        kind = re.match(r'[a-z0-9]*', reason).group(0) or '-'
        key = '%s:%s' % ('eligible' if eligible == '1' else 'ineligible', kind)
        reasons[key] = reasons.get(key, 0) + 1
    missed = sum(1 for live, eligible, reason in skipped if eligible == '1')
    eligible_rounds = len(rounds) + missed
    return dict(admitted=None if admitted is None else dict(
                    min_users=int(admitted.group(1)), users=int(admitted.group(2)), max_idle=int(admitted.group(3)),
                    carries_in_place=admitted.group(4) == '1'),
                padded_rounds=len(rounds), padded_by_live={live: by_live[live] for live in sorted(by_live, reverse=True)},
                skipped_eligible=missed, skipped_ineligible=len(skipped) - missed, skip_reasons=reasons,
                eligible_rounds=eligible_rounds,
                engagement=round(len(rounds) / eligible_rounds, 4) if eligible_rounds else None)


def padded_block_report(environ, log_text):
    """Under QWEN_FAST_PADDED_BLOCK=1 the summary and its problems (under 'problems'), else None -
    and a minimum set without the flag is itself a problem (it pads nothing)."""
    if environ.get(PADDED_BLOCK_FLAG) != '1':
        if environ.get(PADDED_MIN_USERS_FLAG):
            return dict(problems=['%s=%s without %s=1 pads nothing' % (PADDED_MIN_USERS_FLAG,
                                                                       environ.get(PADDED_MIN_USERS_FLAG),
                                                                       PADDED_BLOCK_FLAG)])
        return None
    summary = padded_block_summary(log_text)
    problems = []
    wanted = (environ.get(PADDED_MIN_USERS_FLAG) or str(PADDED_MIN_USERS_DEFAULT)).strip()
    if summary['admitted'] is not None and str(summary['admitted']['min_users']) != wanted:
        problems.append('%s: the block admitted min_users=%d, not the %s asked for' % (
            PADDED_MIN_USERS_FLAG, summary['admitted']['min_users'], wanted))
    for marker, what in ((PADDED_PAGE0_MARKER, 'no live table holds page 0 in its used range'),
                         (PADDED_IDLE_COMMIT_MARKER, 'no idle segment commits'),
                         (REFUSED_ROUND_MARKER, 'no round refused outright')):
        count = log_text.count(marker)
        if count:
            problems.append('%s: %s (%d lines; first: %s)' % (PADDED_BLOCK_FLAG, what, count,
                                                               _first_line(log_text, marker)))
    backstops = [match.group(0)[:200] for match in PADDED_REFUSED_LINE.finditer(log_text)
                 if match.group(1) != 'proposal_rows']
    if backstops:
        problems.append('%s: no refusal past proposal_rows (%d lines; first: %s)' % (PADDED_BLOCK_FLAG, len(backstops),
                                                                                  backstops[0]))
    if summary['eligible_rounds'] and summary['engagement'] < PADDED_ENGAGEMENT:
        problems.append('%s: padded rounds %d of %d eligible (%.2f), under %.2f' % (
            PADDED_BLOCK_FLAG, summary['padded_rounds'], summary['eligible_rounds'], summary['engagement'],
            PADDED_ENGAGEMENT))
    summary['problems'] = problems
    return summary


# Round-fence plan H1a (verify_prestage.py; QWEN_FAST_PRESTAGE, QWEN_FAST_PRESTAGE_AUDIT,
# QWEN_FAST_ROUND_FENCES; every one default off). Each flag's block logs its engagement once at attach
# and one line per round once its path ran, so a flag set on an image without it fails here:
#   QWEN_FAST_PRESTAGE        '[PINDIAG] verify prestage engaged', '[PACKED-PRESTAGE-WINDOW] round=' per
#       drafts window (or 'dropped=' when the pre-stage raised: the verify then goes full), and
#       '[PACKED-PRESTAGE] round=R path=diff|full buffers=N reason=R live=K' per verify. Required: at least
#       one diff round, and - over at least PRESTAGE_FLOOR_ROUNDS four-live rounds - the diff path in
#       PRESTAGE_DIFF_FLOOR of them (the plan's H1a threshold; every full round names its reason).
#   QWEN_FAST_PRESTAGE_AUDIT  '[PACKED-PRESTAGE-AUDIT] round=R path=P checked=8 first=I mismatches=M' after
#       every verify-time write; any mismatch fails the arm (the round itself was restaged in full, so its
#       text stays exact). Without QWEN_FAST_PRESTAGE it audits nothing: a problem.
#   QWEN_FAST_ROUND_FENCES    '[PINDIAG] round fences engaged' and, under QWEN_FAST_PACKED_AUDIT,
#       '[PACKED-FENCES] round=R fence=f9|replay|first validated=0|1 ...' per round. With the packed
#       proposal coordinator's window (QWEN_FAST_PACKED_PROPOSAL and QWEN_FAST_PIPELINED_PROPOSALS) at
#       least one replay must be armed by the drafts' fence (fence=f9).
PRESTAGE_FLAG = 'QWEN_FAST_PRESTAGE'
PRESTAGE_AUDIT_FLAG = 'QWEN_FAST_PRESTAGE_AUDIT'
ROUND_FENCES_FLAG = 'QWEN_FAST_ROUND_FENCES'
PRESTAGE_ENGAGED_MARKER = '[PINDIAG] verify prestage engaged'
PRESTAGE_WINDOW_MARKER = '[PACKED-PRESTAGE-WINDOW] round='
PRESTAGE_MARKER = '[PACKED-PRESTAGE] round='
PRESTAGE_AUDIT_MARKER = '[PACKED-PRESTAGE-AUDIT] round='
FENCES_ENGAGED_MARKER = '[PINDIAG] round fences engaged'
FENCES_MARKER = '[PACKED-FENCES] round='
PRESTAGE_LINE = re.compile(r'\[PACKED-PRESTAGE\] round=([0-9]+) path=(diff|full) buffers=([0-9]+) reason=(\S+) '
                           r'live=([0-9]+)')
PRESTAGE_WINDOW_LINE = re.compile(r'\[PACKED-PRESTAGE-WINDOW\] round=([0-9]+) (?:buffers=([0-9]+) ms=([0-9.]+)'
                                  r'|dropped=(\S+))')
PRESTAGE_AUDIT_LINE = re.compile(r'\[PACKED-PRESTAGE-AUDIT\] round=([0-9]+) path=(diff|full) checked=([0-9]+) '
                                 r'first=(-?[0-9]+) mismatches=([0-9]+)')
FENCES_LINE = re.compile(r'\[PACKED-FENCES\] round=([0-9]+) fence=(\S+) validated=([01]) replay_ms=([0-9.]+) '
                         r'commit_sync_ms=([0-9.]+) path=(\S+) prestage_ms=([0-9.]+) diff_ms=([0-9.]+) '
                         r'write_ms=([0-9.]+)')
PRESTAGE_DIFF_FLOOR = 0.65
PRESTAGE_FLOOR_ROUNDS = 10


def h1a_markers(environ):
    """{flag: [marker]} for the round-fence plan H1a flags set in `environ`."""
    on = lambda name: environ.get(name) == '1'
    required = {}
    if on(PRESTAGE_FLAG):
        required[PRESTAGE_FLAG] = [PRESTAGE_ENGAGED_MARKER, PRESTAGE_WINDOW_MARKER, PRESTAGE_MARKER]
        if on(PRESTAGE_AUDIT_FLAG):
            required[PRESTAGE_AUDIT_FLAG] = [PRESTAGE_AUDIT_MARKER]
    if on(ROUND_FENCES_FLAG):
        required[ROUND_FENCES_FLAG] = [FENCES_ENGAGED_MARKER]
        if on('QWEN_FAST_PACKED_AUDIT'):
            required[ROUND_FENCES_FLAG].append(FENCES_MARKER)
    return required


def h1a_summary(log_text):
    """What the H1a lines say: the verify paths (all rounds and four-live rounds), the full-path
    reasons, the window's pre-stages and drops, the audit, and how each replay was armed."""
    rounds = [match.groups() for match in PRESTAGE_LINE.finditer(log_text)]
    windows = [match.groups() for match in PRESTAGE_WINDOW_LINE.finditer(log_text)]
    audits = [match.groups() for match in PRESTAGE_AUDIT_LINE.finditer(log_text)]
    fences = [match.groups() for match in FENCES_LINE.finditer(log_text)]
    reasons = {}
    for round_number, path, buffers, reason, live in rounds:
        if path == 'full':
            reasons[reason[:48]] = reasons.get(reason[:48], 0) + 1
    four = [line for line in rounds if line[4] == '4']
    diff_buffers = sorted(int(line[2]) for line in rounds if line[1] == 'diff')
    by_fence = {}
    for line in fences:
        by_fence[line[1]] = by_fence.get(line[1], 0) + 1
    return dict(
        rounds=len(rounds), diff=sum(1 for line in rounds if line[1] == 'diff'),
        full=sum(1 for line in rounds if line[1] == 'full'), full_reasons=reasons,
        four_live=len(four), four_live_diff=sum(1 for line in four if line[1] == 'diff'),
        diff_share_four_live=(round(sum(1 for line in four if line[1] == 'diff') / len(four), 4) if four else None),
        diff_buffers_median=diff_buffers[len(diff_buffers) // 2] if diff_buffers else None,
        windows=sum(1 for line in windows if line[3] is None), dropped=[line[3] for line in windows if line[3]][:8],
        window_ms_median=(statistics.median(float(line[2]) for line in windows if line[3] is None)
                          if any(line[3] is None for line in windows) else None),
        audits=len(audits), audited_buffers=sum(int(line[2]) for line in audits),
        audit_mismatches=sum(int(line[4]) for line in audits),
        fences=len(fences), fence_kinds=by_fence, validated=sum(1 for line in fences if line[2] == '1'))


def h1a_report(environ, log_text):
    """Under any H1a flag the summary and its problems (under 'problems'), else None."""
    on = lambda name: environ.get(name) == '1'
    if not (on(PRESTAGE_FLAG) or on(PRESTAGE_AUDIT_FLAG) or on(ROUND_FENCES_FLAG)):
        return None
    summary = h1a_summary(log_text)
    problems = []
    if on(PRESTAGE_AUDIT_FLAG) and not on(PRESTAGE_FLAG):
        problems.append('%s=1 without %s=1 audits nothing' % (PRESTAGE_AUDIT_FLAG, PRESTAGE_FLAG))
    if on(PRESTAGE_FLAG):
        if summary['rounds'] and not summary['diff']:
            problems.append('%s: no verify took the diff path (%d full; reasons %s)' % (
                PRESTAGE_FLAG, summary['full'], summary['full_reasons']))
        if (summary['four_live'] >= PRESTAGE_FLOOR_ROUNDS
                and summary['diff_share_four_live'] < PRESTAGE_DIFF_FLOOR):
            problems.append('%s: diff path in %d of %d four-live rounds (%.2f), under %.2f (full reasons %s)' % (
                PRESTAGE_FLAG, summary['four_live_diff'], summary['four_live'], summary['diff_share_four_live'],
                PRESTAGE_DIFF_FLOOR, summary['full_reasons']))
    if on(PRESTAGE_FLAG) and on(PRESTAGE_AUDIT_FLAG) and summary['audit_mismatches']:
        first = next(match.group(0) for match in PRESTAGE_AUDIT_LINE.finditer(log_text) if match.group(5) != '0')
        problems.append('%s: no mismatch (%d; first: %s)' % (PRESTAGE_AUDIT_FLAG, summary['audit_mismatches'],
                                                             first[:200]))
    # Round-fence plan H2 (QWEN_FAST_GDN_AFTER_PAIRS): the GDN commits are enqueued after the drafts' fence,
    # so every replay pays the owed fence itself (fence=replay) - no f9 is expected.
    if (on(ROUND_FENCES_FLAG) and on('QWEN_FAST_PACKED_AUDIT') and on('QWEN_FAST_PACKED_PROPOSAL')
            and on('QWEN_FAST_PIPELINED_PROPOSALS') and not on(GDN_AFTER_PAIRS_FLAG)
            and summary['fences'] and not summary['fence_kinds'].get('f9')):
        problems.append('%s: no replay armed by the drafts\' fence (fence kinds %s)' % (
            ROUND_FENCES_FLAG, summary['fence_kinds']))
    summary['problems'] = problems
    return summary


# Round-fence plan H1b (fused_commit.py; QWEN_FAST_FUSED_COMMIT, _INPLACE, _LIVE_BANKS, _AUDIT; every one
# default off). The block logs its engagement once at attach (or '[PINDIAG] fused commit refused reason=' -
# a host check turned it down and today's publication served every round), one line per packed user's
# publication and, under the audit, one per audited publication:
#   QWEN_FAST_FUSED_COMMIT          '[PINDIAG] fused commit engaged ... inplace=I live_banks=L audit=A ...' and
#       '[PACKED-FUSED] round=R segment=S prefix=P path=fused|today reason=-|WHY tables=window|late|-'.
#       Required: at least one fused publication, no refusal, and every path=today publication's reason
#       one of FUSED_EXPECTED_REFUSALS (the prefill ramp; one parity normalisation per device) - any other
#       reason (slot, weights, scope, features, ...) is a round the fused path should have served.
#   QWEN_FAST_FUSED_COMMIT_INPLACE  engaged with inplace=1.
#   QWEN_FAST_FUSED_COMMIT_LIVE_BANKS  '[PINDIAG] pair live banks engaged' (the first pair bound to the live
#       banks); needs _INPLACE.
#   QWEN_FAST_FUSED_COMMIT_AUDIT    '[PACKED-FUSED-AUDIT] round=R segment=S prefix=P mode=M checked=N
#       mismatches=0' for every fused publication - one per path=fused line - and no mismatch anywhere
#       ('[PINDIAG] fused commit audit mismatch'), on either chip, in any round.
#   Any flag: no '[PINDIAG] fused commit discard after in-place slide' (an in-place publication discarded
#       after its slide moved the live banks: its request failed mid-commit, and the line names why the
#       segment refused that request's later rounds as 'poisoned').
# Every sub-flag without QWEN_FAST_FUSED_COMMIT (and _LIVE_BANKS without _INPLACE) is inert: a problem.
FUSED_FLAG = 'QWEN_FAST_FUSED_COMMIT'
FUSED_INPLACE_FLAG = 'QWEN_FAST_FUSED_COMMIT_INPLACE'
FUSED_LIVE_BANKS_FLAG = 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'
FUSED_AUDIT_FLAG = 'QWEN_FAST_FUSED_COMMIT_AUDIT'
FUSED_ENGAGED_MARKER = '[PINDIAG] fused commit engaged'
FUSED_REFUSED_MARKER = '[PINDIAG] fused commit refused'
FUSED_MARKER = '[PACKED-FUSED] round='
FUSED_AUDIT_MARKER = '[PACKED-FUSED-AUDIT] round='
FUSED_AUDIT_MISMATCH_MARKER = '[PINDIAG] fused commit audit mismatch'
FUSED_LIVE_BANKS_MARKER = '[PINDIAG] pair live banks engaged'
FUSED_DISCARD_MARKER = '[PINDIAG] fused commit discard after in-place slide'
FUSED_EXPECTED_REFUSALS = ('ramp', 'parity')
FUSED_ENGAGED_LINE = re.compile(r'\[PINDIAG\] fused commit engaged users=([0-9]+) rows=([0-9]+) inplace=([01]) '
                                r'live_banks=([01]) audit=([01]) kernel=(\S+) traces=([0-9]+)')
FUSED_LINE = re.compile(r'\[PACKED-FUSED\] round=([0-9]+) segment=([0-9]+) prefix=(\S+) path=(fused|today) '
                        r'reason=(\S+) tables=(\S+)')
FUSED_AUDIT_LINE = re.compile(r'\[PACKED-FUSED-AUDIT\] round=([0-9]+) segment=([0-9]+) prefix=([0-9]+) mode=(\S+) '
                              r'checked=([0-9]+) mismatches=([0-9]+)')


def h1b_markers(environ):
    """{flag: [marker]} for the round-fence plan H1b flags set in `environ`."""
    on = lambda name: environ.get(name) == '1'
    required = {}
    if not on(FUSED_FLAG):
        return required
    required[FUSED_FLAG] = [FUSED_ENGAGED_MARKER, FUSED_MARKER]
    if on(FUSED_INPLACE_FLAG):
        required[FUSED_INPLACE_FLAG] = [FUSED_ENGAGED_MARKER]
        if on(FUSED_LIVE_BANKS_FLAG):
            required[FUSED_LIVE_BANKS_FLAG] = [FUSED_LIVE_BANKS_MARKER]
    if on(FUSED_AUDIT_FLAG):
        required[FUSED_AUDIT_FLAG] = [FUSED_AUDIT_MARKER]
    return required


def h1b_summary(log_text):
    """What the H1b lines say: the engagement, the publications by path and refusal reason, the rounds
    whose every publication was fused, where the RoPE tables were staged, and the audit."""
    engaged = FUSED_ENGAGED_LINE.search(log_text)
    lines = [match.groups() for match in FUSED_LINE.finditer(log_text)]
    audits = [match.groups() for match in FUSED_AUDIT_LINE.finditer(log_text)]
    reasons, tables, rounds = {}, {}, {}
    for round_number, segment, prefix, path, reason, staged in lines:
        rounds.setdefault(round_number, []).append(path)
        if path == 'today':
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            tables[staged] = tables.get(staged, 0) + 1
    unexpected = {reason: count for reason, count in reasons.items() if reason not in FUSED_EXPECTED_REFUSALS}
    fused_rounds = [paths for paths in rounds.values() if all(path == 'fused' for path in paths)]
    return dict(
        engaged=None if engaged is None else dict(users=int(engaged.group(1)), rows=int(engaged.group(2)),
                                                  inplace=int(engaged.group(3)), live_banks=int(engaged.group(4)),
                                                  audit=int(engaged.group(5)), kernel=engaged.group(6),
                                                  traces=int(engaged.group(7))),
        refused=FUSED_REFUSED_MARKER in log_text,
        publications=len(lines), fused=sum(1 for line in lines if line[3] == 'fused'),
        today=sum(1 for line in lines if line[3] == 'today'), today_reasons=reasons, unexpected_reasons=unexpected,
        rounds=len(rounds), fused_rounds=len(fused_rounds),
        four_fused_rounds=sum(1 for paths in fused_rounds if len(paths) == 4), tables=tables,
        audits=len(audits), audit_checked=sum(int(line[4]) for line in audits),
        audit_mismatches=sum(int(line[5]) for line in audits),
        audit_mismatch_lines=log_text.count(FUSED_AUDIT_MISMATCH_MARKER),
        live_banks=FUSED_LIVE_BANKS_MARKER in log_text, discards=log_text.count(FUSED_DISCARD_MARKER),
        live_banks_normalised=log_text.count('[PACKED-PROPOSE] live banks pair='))


def h1b_report(environ, log_text):
    """Under any H1b flag the summary and its problems (under 'problems'), else None."""
    on = lambda name: environ.get(name) == '1'
    flags = (FUSED_FLAG, FUSED_INPLACE_FLAG, FUSED_LIVE_BANKS_FLAG, FUSED_AUDIT_FLAG)
    if not any(on(flag) for flag in flags):
        return None
    summary = h1b_summary(log_text)
    problems = []
    for flag in flags[1:]:
        if on(flag) and not on(FUSED_FLAG):
            problems.append('%s=1 without %s=1 does nothing' % (flag, FUSED_FLAG))
    if on(FUSED_LIVE_BANKS_FLAG) and not on(FUSED_INPLACE_FLAG):
        problems.append('%s=1 without %s=1 does nothing (the live bank moves every commit)'
                        % (FUSED_LIVE_BANKS_FLAG, FUSED_INPLACE_FLAG))
    if on(FUSED_FLAG):
        if summary['refused']:
            first = next(line for line in log_text.splitlines() if FUSED_REFUSED_MARKER in line)
            problems.append('%s: the block refused the fused commit (%s)' % (FUSED_FLAG, first[-200:]))
        engaged = summary['engaged']
        if engaged is not None and engaged['inplace'] != int(on(FUSED_INPLACE_FLAG)):
            problems.append('%s: engaged inplace=%d with the flag %s' % (FUSED_INPLACE_FLAG, engaged['inplace'],
                                                                         environ.get(FUSED_INPLACE_FLAG, 'unset')))
        if summary['publications'] and not summary['fused']:
            problems.append('%s: no publication took the fused path (%d today; reasons %s)' % (
                FUSED_FLAG, summary['today'], summary['today_reasons']))
        if summary['unexpected_reasons']:
            problems.append('%s: publications took today\'s path for reasons outside %s: %s' % (
                FUSED_FLAG, FUSED_EXPECTED_REFUSALS, summary['unexpected_reasons']))
        if summary['discards']:
            first = next(line for line in log_text.splitlines() if FUSED_DISCARD_MARKER in line)
            problems.append('%s: %d in-place publication(s) discarded after the slide moved the live banks (%s)' % (
                FUSED_FLAG, summary['discards'], first[-160:]))
        if on(FUSED_AUDIT_FLAG):
            if summary['audits'] != summary['fused']:
                problems.append('%s: %d audits for %d fused publications' % (FUSED_AUDIT_FLAG, summary['audits'],
                                                                           summary['fused']))
            if summary['audit_mismatches'] or summary['audit_mismatch_lines']:
                first = next((match.group(0) for match in FUSED_AUDIT_LINE.finditer(log_text)
                              if match.group(6) != '0'), '')
                problems.append('%s: no mismatch (%d; %d mismatch lines; first: %s)' % (
                    FUSED_AUDIT_FLAG, summary['audit_mismatches'], summary['audit_mismatch_lines'], first[:200]))
    summary['problems'] = problems
    return summary


# Round-fence plan H2 (early_draft.py; QWEN_FAST_EARLY_DRAFT, QWEN_FAST_GDN_AFTER_PAIRS; both default off):
#   QWEN_FAST_EARLY_DRAFT       '[PINDIAG] early draft engaged gdn_after_pairs=G' once, and per round the
#       drafts were drafted inside the step '[PACKED-EARLY-DRAFT] round=R path=reuse|redo|failed|untaken
#       live=K draft_ms=X reason=Y' (reuse: take_draft_token_ids handed vLLM the cached drafts; redo: something
#       they were drafted from changed and they were drafted again, reason names it; failed: the draft raised
#       and take_draft_token_ids re-raised it; untaken: vLLM never asked). Required: no failed or untaken line
#       and - over at least EARLY_FLOOR_ROUNDS four-live rounds - reuse in EARLY_REUSE_FLOOR of them (the plan's
#       H2 bar).
#   QWEN_FAST_GDN_AFTER_PAIRS   '[PINDIAG] gdn after pairs engaged users=U pipelined=P' at attach (or '...
#       refused reason=': a problem), and per round whose commits were deferred '[PACKED-GDN-AFTER-PAIRS]
#       round=R commits=N site=S enqueue_ms=X segments=...' - every site inside the step that decided them
#       (window: after the pairs' readback; end: the end of the early draft); reconcile or verify is R1
#       broken, and dropped= a block that failed with commits held. Under QWEN_FAST_ROUND_B1 at least one
#       flush must come after the pairs' readback (site=window). Without QWEN_FAST_EARLY_DRAFT: does nothing.
EARLY_DRAFT_FLAG = 'QWEN_FAST_EARLY_DRAFT'
GDN_AFTER_PAIRS_FLAG = 'QWEN_FAST_GDN_AFTER_PAIRS'
EARLY_ENGAGED_MARKER = '[PINDIAG] early draft engaged'
EARLY_MARKER = '[PACKED-EARLY-DRAFT] round='
GDN_ENGAGED_MARKER = '[PINDIAG] gdn after pairs engaged'
GDN_REFUSED_MARKER = '[PINDIAG] gdn after pairs refused'
GDN_MARKER = '[PACKED-GDN-AFTER-PAIRS] round='
EARLY_LINE = re.compile(r'\[PACKED-EARLY-DRAFT\] round=([0-9]+) path=(reuse|redo|failed|untaken) live=([0-9]+) '
                        r'draft_ms=([0-9.]+) reason=(\S+)')
GDN_LINE = re.compile(r'\[PACKED-GDN-AFTER-PAIRS\] round=([0-9]+) commits=([0-9]+) site=(\S+) enqueue_ms=([0-9.]+) '
                      r'segments=(\S+)(?: dropped=([0-9]+) reason=(\S+))?')
GDN_IN_STEP_SITES = ('window', 'end')
EARLY_REUSE_FLOOR = 0.95
EARLY_FLOOR_ROUNDS = 10


def h2_markers(environ):
    """{flag: [marker]} for the round-fence plan H2 flags set in `environ`."""
    on = lambda name: environ.get(name) == '1'
    required = {}
    if on(EARLY_DRAFT_FLAG):
        required[EARLY_DRAFT_FLAG] = [EARLY_ENGAGED_MARKER, EARLY_MARKER]
        if on(GDN_AFTER_PAIRS_FLAG):
            required[GDN_AFTER_PAIRS_FLAG] = [GDN_ENGAGED_MARKER, GDN_MARKER]
    return required


def h2_summary(log_text):
    """What the H2 lines say: the early drafts by path (all and four-live), the redo reasons, the drafting
    time, and the deferred GDN commits by flush site."""
    drafts = [match.groups() for match in EARLY_LINE.finditer(log_text)]
    flushes = [match.groups() for match in GDN_LINE.finditer(log_text)]
    paths, reasons, sites = {}, {}, {}
    for round_number, path, live, draft_ms, reason in drafts:
        paths[path] = paths.get(path, 0) + 1
        if path in ('redo', 'failed', 'untaken'):
            key = '%s:%s' % (path, reason[:48])
            reasons[key] = reasons.get(key, 0) + 1
    for line in flushes:
        sites[line[2]] = sites.get(line[2], 0) + 1
    four = [line for line in drafts if line[2] == '4' and line[1] in ('reuse', 'redo')]
    reuse_ms = sorted(float(line[3]) for line in drafts if line[1] == 'reuse' and line[2] == '4')
    return dict(
        drafts=len(drafts), paths=paths, reasons=reasons, four_live=len(four),
        four_live_reuse=sum(1 for line in four if line[1] == 'reuse'),
        reuse_share_four_live=(round(sum(1 for line in four if line[1] == 'reuse') / len(four), 4) if four else None),
        draft_ms_median_four_live=reuse_ms[len(reuse_ms) // 2] if reuse_ms else None,
        engaged=EARLY_ENGAGED_MARKER in log_text,
        gdn_engaged=GDN_ENGAGED_MARKER in log_text, gdn_refused=GDN_REFUSED_MARKER in log_text,
        flushes=len(flushes), flushed_commits=sum(int(line[1]) for line in flushes), flush_sites=sites,
        dropped=sum(int(line[5]) for line in flushes if line[5]),
        late=sum(count for site, count in sites.items() if site not in GDN_IN_STEP_SITES))


def h2_report(environ, log_text):
    """Under any H2 flag the summary and its problems (under 'problems'), else None."""
    on = lambda name: environ.get(name) == '1'
    if not (on(EARLY_DRAFT_FLAG) or on(GDN_AFTER_PAIRS_FLAG)):
        return None
    summary = h2_summary(log_text)
    problems = []
    if on(GDN_AFTER_PAIRS_FLAG) and not on(EARLY_DRAFT_FLAG):
        problems.append('%s=1 without %s=1 does nothing (nobody flushes inside the step)'
                        % (GDN_AFTER_PAIRS_FLAG, EARLY_DRAFT_FLAG))
    if on(EARLY_DRAFT_FLAG):
        if summary['paths'].get('failed'):
            first = next(match.group(0) for match in EARLY_LINE.finditer(log_text) if match.group(2) == 'failed')
            problems.append('%s: %d early draft(s) raised (%s)' % (EARLY_DRAFT_FLAG, summary['paths']['failed'],
                                                                   first[:160]))
        if summary['paths'].get('untaken'):
            # vLLM calls take_draft_token_ids after every step that ran the model, so a cache still held at the
            # next execute_model means the scheduler never saw drafts whose tickets are pending: never expected.
            first = next(match.group(0) for match in EARLY_LINE.finditer(log_text) if match.group(2) == 'untaken')
            problems.append('%s: %d early draft(s) never taken by vLLM (%s)' % (
                EARLY_DRAFT_FLAG, summary['paths']['untaken'], first[:160]))
        if summary['four_live'] >= EARLY_FLOOR_ROUNDS and summary['reuse_share_four_live'] < EARLY_REUSE_FLOOR:
            problems.append('%s: early drafts reused in %d of %d four-live rounds (%.2f), under %.2f (reasons %s)' % (
                EARLY_DRAFT_FLAG, summary['four_live_reuse'], summary['four_live'],
                summary['reuse_share_four_live'], EARLY_REUSE_FLOOR, summary['reasons']))
        if on(GDN_AFTER_PAIRS_FLAG):
            if summary['gdn_refused']:
                first = next(line for line in log_text.splitlines() if GDN_REFUSED_MARKER in line)
                problems.append('%s: the block refused it (%s)' % (GDN_AFTER_PAIRS_FLAG, first[-160:]))
            if summary['late']:
                first = next(match.group(0) for match in GDN_LINE.finditer(log_text)
                             if match.group(3) not in GDN_IN_STEP_SITES)
                problems.append('%s: %d flush(es) outside the step that decided them (R1; first: %s)' % (
                    GDN_AFTER_PAIRS_FLAG, summary['late'], first[:160]))
            if summary['dropped']:
                problems.append('%s: %d deferred commit(s) dropped by a failed block' % (
                    GDN_AFTER_PAIRS_FLAG, summary['dropped']))
            if on('QWEN_FAST_ROUND_B1') and summary['flushes'] and not summary['flush_sites'].get('window'):
                problems.append("%s: no flush after the pairs' readback (sites %s)" % (
                    GDN_AFTER_PAIRS_FLAG, summary['flush_sites']))
    summary['problems'] = problems
    return summary


# The pair drafter's row-1 fix (pair_row_exact.py; default off). QWEN_FAST_PAIR_ROW_EXACT=1 promises
# '[PINDIAG] pair row exact engaged pair=[a,b] context=2048,2048 heads=32/8 keys=2080', logged once per process
# when the first (2048, 2048) pair bucket has folded its draft SDPA and captured its trace. Promised with two or
# more concurrent users only: a single-stream arm builds no pair.
PAIR_ROW_EXACT_FLAG = 'QWEN_FAST_PAIR_ROW_EXACT'
PAIR_ROW_EXACT_MARKER = '[PINDIAG] pair row exact engaged'


def pair_row_exact_markers(environ, users):
    """{flag: [marker]} for QWEN_FAST_PAIR_ROW_EXACT when `environ` sets it and the arm packs pairs."""
    if environ.get(PAIR_ROW_EXACT_FLAG) == '1' and users >= 2:
        return {PAIR_ROW_EXACT_FLAG: [PAIR_ROW_EXACT_MARKER]}
    return {}


def required_flag_markers(environ, users, prompt_tokens=None):
    """The markers the flags in `environ` promise, as {flag: [marker, ...]}. prompt_tokens (each user's
    prompt; None: unknown) decides whether QWEN_FAST_SDPA_PF's 2048-row topology is promised."""
    on = lambda name: environ.get(name) == '1'
    required = {}
    if on('QWEN_FAST_SINGLE_GATEUP'):
        required['QWEN_FAST_SINGLE_GATEUP'] = single_gateup_markers(environ)
    elif on('QWEN_FAST_SKIP_BLOCK_STREAM'):
        required['QWEN_FAST_SKIP_BLOCK_STREAM'] = [SKIP_BLOCK_STREAM_MARKER]
    if on('QWEN_FAST_DRAFT_BF8'):
        required['QWEN_FAST_DRAFT_BF8'] = [DRAFT_BF8_MARKER]
    if on('QWEN_FAST_ROUND_B1'):
        required['QWEN_FAST_ROUND_B1'] = [ROUND_B1_MARKER]
        if on('QWEN_FAST_ROUND_B1_AUDIT') and users == 4:
            required['QWEN_FAST_ROUND_B1_AUDIT'] = [ROUND_B1_AUDIT_MARKER + ' 1 exact=True']
    if on(VERIFY_T1_FLAG):
        required[VERIFY_T1_FLAG] = [VERIFY_T1_MARKER]
        if on(VERIFY_T1_AUDIT_FLAG) and users == 4:
            required[VERIFY_T1_AUDIT_FLAG] = [VERIFY_T1_AUDIT_MARKER + ' 1 exact=True']
    if on(VERIFY_T2_FLAG):
        required[VERIFY_T2_FLAG] = [VERIFY_T2_MARKER]
        if on(VERIFY_T2_AUDIT_FLAG) and users == 4 and 'windows' not in verify_t2_skipped(environ):
            required[VERIFY_T2_AUDIT_FLAG] = [VERIFY_T2_AUDIT_MARKER + ' 1 exact=True']
    required.update(variable_user_markers(environ, users))
    required.update(h1a_markers(environ))
    required.update(h1b_markers(environ))
    required.update(h2_markers(environ))
    required.update(pair_row_exact_markers(environ, users))
    if on(PUBLISH_PREWARM_FLAG):
        required[PUBLISH_PREWARM_FLAG] = [PUBLISH_PREWARM_MARKER]
    if on('QWEN_FAST_MEMORY_LEDGER'):
        required['QWEN_FAST_MEMORY_LEDGER'] = list(LEDGER_MARKERS)
    if on('QWEN_PREFILL_PROFILE_FLUSH'):
        required['QWEN_PREFILL_PROFILE_FLUSH'] = [PREFILL_FLUSH_MARKER]
    if on(PREFILL_CONV_FLAG):
        required[PREFILL_CONV_FLAG] = prefill_conv_markers(environ)
    if on(SDPA_PF_FLAG):
        required[SDPA_PF_FLAG] = sdpa_pf_markers(environ, prompt_tokens)
    markers = sdpa_mode_markers(sdpa_mode_names(environ))
    if markers:
        required['QWEN_FAST_SDPA_MODES'] = markers
    return required


def sdpa_mode_markers(names):
    """The markers a QWEN_FAST_SDPA_MODES value promises, or [] when it names no served mode.
    The modes line is apply_sdpa_modes' sorted join; the factory line carries every requested
    flag (tail alone: 0x1, the stage-1 markers exactly; tail,share: 0x3; tail,share,slice: 0x7);
    slice or readahead adds the stage-4 factory's q-slice line as a fourth."""
    served = sorted(names.intersection(SDPA_MODE_FLAGS))
    if not served:
        return []
    flags = 0
    for name in served:
        flags |= SDPA_MODE_FLAGS[name]
    markers = [SDPA_MODES_MARKERS[0], '[PINDIAG] sdpa qwen-modes modes=%s ' % ','.join(sorted(names)),
               '[QWEN-SDPA] flags=0x%x ' % flags]
    if SDPA_STAGE4_MODES.intersection(served):
        markers.append(SDPA_SLICE_MARKER)
    return markers


def sdpa_mode_names(environ):
    """QWEN_FAST_SDPA_MODES as pooled_attention_replay.sdpa_modes splits it (no validation here:
    a value that reader refuses fails the run on its own)."""
    return {name.strip() for name in (environ.get('QWEN_FAST_SDPA_MODES') or '').split(',') if name.strip()}


def flag_marker_report(environ, users, log_text, prompt_tokens=None, gdn_layers=GDN_LAYERS):
    """Which promised markers the server log carries, and which flags left theirs out.
    prompt_tokens (each user's prompt) sets the engaged-chunk floor of QWEN_FAST_GDN_PREFILL_CONV."""
    required = required_flag_markers(environ, users, prompt_tokens)
    found = {flag: {marker: marker in log_text for marker in markers} for flag, markers in required.items()}
    missing = sorted('%s: %s' % (flag, marker) for flag, markers in found.items()
                     for marker, present in markers.items() if not present)
    if environ.get('QWEN_FAST_GDN_USER_BATCH') == '1' and users == 4:
        # At least one captured forward batched EVERY GDN layer (the 64-row block).
        complete = [m for m in GDN_ALL_BATCHED.finditer(log_text) if m.group(1) == m.group(2)]
        found['QWEN_FAST_GDN_USER_BATCH'] = {'n of n GDN layers batched': bool(complete)}
        if not complete:
            missing.append('QWEN_FAST_GDN_USER_BATCH: a captured forward batching every GDN layer')
    seq_block = gdn_seq_block_report(environ, users, log_text, gdn_layers)
    if seq_block is not None:
        if environ.get(GDN_SEQ_BLOCK_FLAG) == '1' and users == 4:
            found[GDN_SEQ_BLOCK_FLAG] = {'n of n GDN layers at the level': seq_block['complete']}
        missing.extend(seq_block['problems'])
    if environ.get(C1E_FLAG) == '1':
        missing.extend(c1e_problems(environ, log_text))
    if environ.get('QWEN_FAST_ROUND_B1') == '1' and environ.get('QWEN_FAST_ROUND_B1_AUDIT') == '1':
        missing.extend(round_b1_audit_problems(log_text))
    verify_t1_sites = verify_t1_packed = None
    if environ.get(VERIFY_T1_FLAG) == '1':
        verify_t1_sites = sorted(set(VERIFY_T1_SITE.findall(log_text)))
        verify_t1_packed = verify_t1_packed_counts(log_text)
        missing.extend(verify_t1_problems(environ, users, log_text, gdn_layers))
        if VERIFY_T1_AUDIT_MISMATCH in log_text:
            line = log_text[log_text.index(VERIFY_T1_AUDIT_MISMATCH):].split(chr(10), 1)[0]
            missing.append('%s: no mismatch (%s)' % (VERIFY_T1_AUDIT_FLAG, line[:200]))
    verify_t2_packed = None
    if environ.get(VERIFY_T2_FLAG) == '1':
        verify_t2_packed = verify_t2_packed_counts(log_text)
        missing.extend(verify_t2_problems(environ, users, log_text, gdn_layers))
    if environ.get(SDPA_PF_FLAG) == '1' and sdpa_pf_flags(environ) is None:
        missing.append('%s: a production flag set (%s), not %r' % (
            SDPA_PF_FLAGS_FLAG, ', '.join('%#x' % flags for flags in SDPA_PF_PRODUCTION_FLAGS),
            environ.get(SDPA_PF_FLAGS_FLAG)))
    if environ.get(SDPA_PF_FLAG) == '1':
        missing.extend(sdpa_pf_problems(environ, log_text))
    if environ.get(SDPA_PF_FLAG) != '1' and SDPA_PF_FACTORY_MARKER in log_text:
        # The off arm of an A/B on a K64g graft must be the served prefill: no call may carry the word.
        missing.append('%s unset: no chain program (%s logged)' % (SDPA_PF_FLAG, SDPA_PF_FACTORY_MARKER))
    prefill_conv = summary = None
    if environ.get(PREFILL_CONV_FLAG) == '1':
        required_chunks = prefill_conv_required_chunks(users, prompt_tokens)
        missing.extend(prefill_conv_problems(log_text, gdn_layers, required_chunks))
        summary = prefill_conv_summary(log_text, gdn_layers, required_chunks)
        prefill_conv = summary['chunk_calls']
    variable_user = variable_user_report(environ, users, log_text)
    missing.extend(variable_user.pop('problems'))
    # Round-fence plan H1a: under 'round_fence_h1a' only when one of its flags is set.
    h1a = h1a_report(environ, log_text)
    if h1a is not None:
        missing.extend(h1a.pop('problems'))
        variable_user['round_fence_h1a'] = h1a
    # Round-fence plan H1b: under 'round_fence_h1b' only when one of its flags is set.
    h1b = h1b_report(environ, log_text)
    if h1b is not None:
        missing.extend(h1b.pop('problems'))
        variable_user['round_fence_h1b'] = h1b
    # Round-fence plan H2: under 'round_fence_h2' only when one of its flags is set.
    h2 = h2_report(environ, log_text)
    if h2 is not None:
        missing.extend(h2.pop('problems'))
        variable_user['round_fence_h2'] = h2
    # K5-A: under 'gdn_seq_block' only when one of its flags is set.
    if seq_block is not None:
        variable_user['gdn_seq_block'] = {key: value for key, value in seq_block.items() if key != 'problems'}
    # Publish prewarm and the sequential publication log: under 'publish_prewarm' only when one is set.
    prewarm = publish_prewarm_report(environ, log_text)
    if prewarm is not None:
        missing.extend(prewarm.pop('problems'))
        variable_user['publish_prewarm'] = prewarm
    residual = LEDGER_RESIDUAL.search(log_text)
    return dict(found=found, missing=missing, ledger_residual=residual.group(1) if residual else None,
                prefill_conv_chunk_calls=prefill_conv, prefill_conv=summary, verify_t1_sites=verify_t1_sites,
                verify_t1_packed=verify_t1_packed, verify_t2_packed=verify_t2_packed, **variable_user)


def gdn_seq_block_report(environ, users, log_text, gdn_layers=GDN_LAYERS):
    """Under any K5-A flag (QWEN_FAST_GDN_SEQ_BLOCK, _LEVEL, _AUDIT): the captures' K5-A counts and
    levels, the audit lines, and every problem (under 'problems'); None when no flag is set."""
    flag = environ.get(GDN_SEQ_BLOCK_FLAG)
    level_text = environ.get(GDN_SEQ_BLOCK_LEVEL_FLAG)
    audit_text = environ.get(GDN_SEQ_BLOCK_AUDIT_FLAG) or ''
    if flag in (None, '', '0') and not level_text and not audit_text:
        return None
    on = flag == '1'
    problems = []
    if flag not in (None, '', '0', '1'):
        problems.append('%s: 0 or 1, not %r' % (GDN_SEQ_BLOCK_FLAG, flag))
    if on and environ.get('QWEN_FAST_GDN_USER_BATCH') != '1':
        problems.append('%s=1 needs QWEN_FAST_GDN_USER_BATCH=1 (it replaces the user-batched launch)'
                        % GDN_SEQ_BLOCK_FLAG)
    level = level_text or '0'
    if not (level.isdigit() and level == str(int(level)) and int(level) < 1 << GDN_SEQ_BLOCK_LEVEL_BITS):
        problems.append('%s: a decimal bitmask below %d, not %r' % (GDN_SEQ_BLOCK_LEVEL_FLAG,
                                                                     1 << GDN_SEQ_BLOCK_LEVEL_BITS, level_text))
        level = None
    else:
        level = int(level)
    items = audit_text.split(',') if audit_text else []
    layers = [int(item) for item in items if item.isdigit() and item == str(int(item)) and int(item) < gdn_layers]
    if len(layers) != len(items) or len(set(layers)) != len(layers):
        problems.append('%s: distinct GDN layers 0..%d, not %r' % (GDN_SEQ_BLOCK_AUDIT_FLAG, gdn_layers - 1,
                                                                  audit_text))
    if (level_text or audit_text) and not on:
        problems.append('%s / %s without %s=1 do nothing: set it or neither' % (
            GDN_SEQ_BLOCK_LEVEL_FLAG, GDN_SEQ_BLOCK_AUDIT_FLAG, GDN_SEQ_BLOCK_FLAG))
    captures = [(int(calls), int(total), int(built)) for calls, total, built in GDN_SEQ_BLOCK.findall(log_text)]
    complete = level is not None and any(calls == total == gdn_layers and built == level
                                         for calls, total, built in captures)
    if on and users == 4 and not complete:
        problems.append('%s: a captured forward running K5-A in all %d GDN layers at level %s (captures %s)' % (
            GDN_SEQ_BLOCK_FLAG, gdn_layers, level, ['%d of %d level=%d' % capture for capture in captures]))
    # Every GDN layer of one forward has the same segment widths, so K5-A runs in all of them or in
    # none (a 0 is not a capture); a partial count is a per-layer fallback, whatever else passed.
    partial = ['%d of %d level=%d' % capture for capture in captures if not capture[0] == capture[1] == gdn_layers]
    if partial:
        problems.append('%s: captures running K5-A in only some GDN layers: %s' % (GDN_SEQ_BLOCK_FLAG,
                                                                                ', '.join(partial)))
    other = sorted({built for calls, total, built in captures if built != level})
    if other:
        problems.append('%s: captures at level %s, not the %s requested' % (
            GDN_SEQ_BLOCK_FLAG, ','.join(str(value) for value in other), level))
    lines = [(int(layer), int(user), int(count)) for layer, user, count in GDN_SEQ_BLOCK_AUDIT_LINE.findall(log_text)]
    bad = [line for line in lines if line[2]]
    if bad:
        first = next(match.group(0) for match in GDN_SEQ_BLOCK_AUDIT_LINE.finditer(log_text) if match.group(3) != '0')
        problems.append('%s: %d audit line(s) with mismatches (first: %s)' % (GDN_SEQ_BLOCK_AUDIT_FLAG, len(bad),
                                                                                first[:200]))
    if on and audit_text and users == 4:
        seen = {(layer, user) for layer, user, count in lines}
        absent = ['%d/%d' % (layer, user) for layer in layers for user in range(users) if (layer, user) not in seen]
        if absent:
            problems.append('%s: no audit line for layer/user %s' % (GDN_SEQ_BLOCK_AUDIT_FLAG, ','.join(absent)))
    return dict(level=level, captures=captures, complete=complete, audit_layers=layers, audit_lines=len(lines),
                audit_mismatch_lines=len(bad), audit_mismatches=sum(line[2] for line in lines), problems=problems)


def publish_prewarm_report(environ, log_text):
    """Under QWEN_FAST_PUBLISH_PREWARM or QWEN_FAST_SEQ_PUBLISH_LOG: every prewarm line parsed (its pairs,
    count, wall ms and program-cache entries before and after), the skipped lines, the sequential steps
    logged and ended, and every problem (under 'problems'); None when neither flag is set. That a prewarm
    line is there at all is required_flag_markers' check; here, that one of them warmed something - a
    marker the line pattern cannot parse counts as none (a changed line format must not pass unread)."""
    names = (PUBLISH_PREWARM_FLAG, SEQ_PUBLISH_LOG_FLAG)
    flags = tuple(environ.get(name) for name in names)
    if all(flag in (None, '', '0') for flag in flags):
        return None
    problems = ['%s: 0 or 1, not %r' % (name, flag) for name, flag in zip(names, flags)
                if flag not in (None, '', '0', '1')]
    lines = [dict(pairs=pairs, count=int(count), ms=float(ms), program_cache_before=before, program_cache_after=after)
             for pairs, count, ms, before, after in PUBLISH_PREWARM_LINE.findall(log_text)]
    skipped = log_text.count(PUBLISH_PREWARM_SKIPPED)
    if flags[0] == '1' and PUBLISH_PREWARM_MARKER in log_text and not any(line['count'] for line in lines):
        problems.append('%s: a prewarm that warmed something (%d parsed line(s), none with count>0; %d skipped)' % (
            PUBLISH_PREWARM_FLAG, len(lines), skipped))
    logged, ended = len(SEQ_PUBLISH_STEP.findall(log_text)), len(PHASE_STEP_END.findall(log_text))
    if flags[1] == '1' and ended and logged != ended:
        problems.append('%s: a [SEQ-PUBLISH] line for every sequential step (%d ended, %d logged)' % (
            SEQ_PUBLISH_LOG_FLAG, ended, logged))
    return dict(prewarm=lines, skipped=skipped, seq_publish_steps=logged, phase_step_ends=ended, problems=problems)


def verify_t1_skipped(environ):
    return {name.strip() for name in (environ.get(VERIFY_T1_SKIP_FLAG) or '').split(',') if name.strip()}


def verify_t1_packed_counts(log_text):
    """Every site=packed_verify line's counts, one dict per captured verify trace."""
    return [{name: int(value) for name, value in (field.split('=') for field in match.group(1).split())}
            for match in VERIFY_T1_PACKED.finditer(log_text)]


def verify_t1_problems(environ, users, log_text, gdn_layers=GDN_LAYERS):
    """QWEN_FAST_VERIFY_T1: the skip list names only cuts and, at four users, every wave-2
    cut not skipped engaged in every captured verify trace (and every skipped one did not)."""
    skipped = verify_t1_skipped(environ)
    problems = []
    unknown = sorted(skipped.difference(VERIFY_T1_CUTS))
    if unknown:
        problems.append('%s: names only cuts (%s is none of %s)' % (VERIFY_T1_SKIP_FLAG, ','.join(unknown),
                                                                     ','.join(VERIFY_T1_CUTS)))
    if users != 4 or not set(VERIFY_T1_WAVE2_CUTS).difference(skipped):
        return problems
    captures = verify_t1_packed_counts(log_text)
    if not captures:
        problems.append('%s: a captured packed verify reporting its cuts (site=packed_verify); a wave-1 arm on an '
                        'image without wave 2 sets %s=%s' % (VERIFY_T1_FLAG, VERIFY_T1_SKIP_FLAG,
                                                             ','.join(VERIFY_T1_WAVE2_CUTS)))
        return problems
    user_batch = environ.get('QWEN_FAST_GDN_USER_BATCH') == '1'
    per_layer = [name for name in ('direct_carry', 'coalesce') if name not in skipped]
    if per_layer and not user_batch:
        problems.append('%s: %s engage only under QWEN_FAST_GDN_USER_BATCH=1 (or skip them)'
                        % (VERIFY_T1_FLAG, ','.join(per_layer)))
        return problems
    direct = 'direct_carry' not in skipped
    expected = dict(mask_once=int('mask_once' not in skipped), shard_argmax=int('shard_argmax' not in skipped),
                    direct_carry=gdn_layers if direct else 0,
                    last_carry=gdn_layers if direct and 'last_carry' not in skipped else 0,
                    coalesced=gdn_layers if 'coalesce' not in skipped else 0, coalesce_fallback=0)
    if 'coalesce' in skipped or not user_batch:
        expected.pop('coalesce_fallback')
    for index, counts in enumerate(captures):
        wrong = ['%s=%s (expected %d)' % (name, counts.get(name), value) for name, value in expected.items()
                 if counts.get(name) != value]
        if wrong:
            problems.append('%s: capture %d engaged %s' % (VERIFY_T1_FLAG, index + 1, ', '.join(wrong)))
    if 'shard_argmax' not in skipped and VERIFY_T1_KEPT_SAMPLER in log_text:
        line = log_text[log_text.index(VERIFY_T1_KEPT_SAMPLER):].split(chr(10), 1)[0]
        problems.append('%s: the per-shard argmax engaged (%s)' % (VERIFY_T1_FLAG, line[:200]))
    return problems


def verify_t2_skipped(environ):
    return {name.strip() for name in (environ.get(VERIFY_T2_SKIP_FLAG) or '').split(',') if name.strip()}


def verify_t2_packed_counts(log_text):
    """Every site=packed_verify T2 line's fields, one dict per captured verify trace (numbers as
    ints, the warm-chain mode as its word)."""
    captures = []
    for match in VERIFY_T2_PACKED.finditer(log_text):
        fields = {}
        for field in match.group(1).split():
            name, value = field.split('=', 1)
            fields[name] = int(value) if value.isdigit() else value
        captures.append(fields)
    return captures


def verify_t2_problems(environ, users, log_text, gdn_layers=GDN_LAYERS, kv_writes=32):
    """QWEN_FAST_VERIFY_T2: the skip list names only cuts; the kv-rows knob is 64 or 32; no
    fell-back, kv-shared or audit-mismatch line at any user count; and at four users every
    captured verify trace engaged each cut not skipped in every layer (and no skipped one). Every
    problem is reported: one never hides another."""
    skipped = verify_t2_skipped(environ)
    problems = []
    unknown = sorted(skipped.difference(VERIFY_T2_CUTS))
    if unknown:
        problems.append('%s: names only cuts (%s is none of %s)' % (VERIFY_T2_SKIP_FLAG, ','.join(unknown),
                                                                     ','.join(VERIFY_T2_CUTS)))
    knob = environ.get(VERIFY_T2_KV_ROWS_FLAG)
    if knob not in (None, '64', '32'):
        problems.append('%s: 64 or 32, not %r' % (VERIFY_T2_KV_ROWS_FLAG, knob))
    kv_rows = int(knob) if knob in ('64', '32') else 64
    for marker, flag in ((VERIFY_T2_FALLBACK, VERIFY_T2_FLAG), (VERIFY_T2_KV_SHARED, VERIFY_T2_FLAG),
                         (VERIFY_T2_AUDIT_MISMATCH, VERIFY_T2_AUDIT_FLAG)):
        if marker in log_text:
            line = log_text[log_text.index(marker):].split(chr(10), 1)[0]
            problems.append('%s: no %s line (%s)' % (flag, marker, line[:200]))
    if users != 4:
        return problems
    captures = verify_t2_packed_counts(log_text)
    if not captures:
        problems.append('%s: a captured packed verify reporting its cuts (site=packed_verify)' % VERIFY_T2_FLAG)
        return problems
    windows, chains = 'windows' not in skipped, 'kv_chains' not in skipped
    if windows and environ.get('QWEN_FAST_GDN_USER_BATCH') != '1':
        # Reported, then the captures are held to what that configuration CAN engage
        # (windows=0), so the kv fields are still checked.
        problems.append('%s: windows engages only under QWEN_FAST_GDN_USER_BATCH=1 (or skip it)' % VERIFY_T2_FLAG)
        windows = False
    expected = dict(windows=gdn_layers if windows else 0, windows_fallback=0, kv_chains=kv_writes if chains else 0,
                    kv_fallback=0, kv_rows=kv_rows if chains else 0,
                    warm_chain='single' if chains else 'none')
    for index, counts in enumerate(captures):
        wrong = ['%s=%s (expected %s)' % (name, counts.get(name), value) for name, value in expected.items()
                 if counts.get(name) != value]
        if wrong:
            problems.append('%s: capture %d engaged %s' % (VERIFY_T2_FLAG, index + 1, ', '.join(wrong)))
    return problems


def round_b1_audit_problems(log_text):
    """QWEN_FAST_ROUND_B1_AUDIT: no mismatch line, and the last audit line compared every cut."""
    problems = []
    for line in log_text.splitlines():
        if ROUND_B1_AUDIT_MISMATCH in line:
            problems.append('QWEN_FAST_ROUND_B1_AUDIT: no mismatch (%s)' % line[line.index(ROUND_B1_AUDIT_MISMATCH):][:200])
            break
    lines = list(ROUND_B1_AUDIT_LINE.finditer(log_text))
    if lines:
        counts = dict(zip(ROUND_B1_AUDIT_COUNTS, (int(value) for value in lines[-1].groups()[1:])))
        idle = [name for name, count in counts.items() if count == 0]
        if idle:
            problems.append('QWEN_FAST_ROUND_B1_AUDIT: every cut compared (none for %s)' % ','.join(idle))
    return problems


def load_references(directory, prompt_tokens=GENERIC_REFERENCE_TOKENS):
    """Each user's single-stream reference text, keyed by its prompt base.

    single-user-35492921706.json (base 1000, no offset) carries no digit group; the
    others are named single-user-<base>-<run>.json, or single-user-<base>-p<tokens>-<run>.json
    for a reference recorded at one prompt length.

    A reference recorded at exactly `prompt_tokens` wins; otherwise the generic
    (unsegmented, 32,768-token) one is used and `context` says so, because a mismatch
    against a reference of another length is not conclusive. Among several references of
    the chosen kind the longest text wins, and any shorter one that is not its prefix is
    listed under `conflicts` - two references disagreeing is itself a finding.
    """
    references = {}
    directory = Path(directory)
    if not directory.is_dir():
        return references
    candidates = {}
    for path in sorted(directory.glob('single-user*.json')):
        match = REFERENCE_NAME.match(path.name)
        if not match:
            continue
        base = int(match.group(1)) if match.group(1) else 1000
        tokens = int(match.group(2)) if match.group(2) else GENERIC_REFERENCE_TOKENS
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            continue
        streams = data.get('streams') or []
        if not streams:
            continue
        candidates.setdefault(base, []).append(dict(
            path=str(path), text=streams[0].get('text', '') or '',
            text_sha256=streams[0].get('text_sha256'), tokens=tokens, segmented=bool(match.group(2))))
    for base, found in candidates.items():
        exact = [c for c in found if c['tokens'] == prompt_tokens]
        generic = [c for c in found if not c['segmented']]
        chosen = exact or generic
        if not chosen:
            continue
        best = max(chosen, key=lambda c: (len(c['text']), c['path']))
        conflicts = sorted(c['path'] for c in chosen if not best['text'].startswith(c['text']))
        references[base] = dict(path=best['path'], text=best['text'], text_sha256=best['text_sha256'],
                                context='exact' if exact else 'generic-%d' % GENERIC_REFERENCE_TOKENS,
                                conflicts=conflicts)
    return references


def write_candidate(directory, base, prompt_tokens, entry, prompt_sha256=None):
    """This run's stream for `base`, in the tracked reference format, for promotion to
    scripts/ci/references/packed-gate after review. Never read back by the gate.

    With `prompt_sha256` (a real-text prompt) the file is named by that hash instead of the
    base, reference-candidate-realtext-<sha12>-p<tokens>.json, which REFERENCE_NAME never
    matches: a real-text stream promoted by mistake can never become a synthetic base's
    reference."""
    if not entry or entry.get('error') or not entry.get('text'):
        return None
    if prompt_sha256:
        name = 'reference-candidate-realtext-%s-p%d.json' % (prompt_sha256[:12], prompt_tokens)
        fields = dict(label='candidate', prompt_source='real-text', prompt_sha256=prompt_sha256,
                      prompt_tokens=prompt_tokens, streams=[entry])
    else:
        name = 'reference-candidate-%d-p%d.json' % (base, prompt_tokens)
        fields = dict(label='candidate', prompt_base=base, prompt_tokens=prompt_tokens, streams=[entry])
    path = Path(directory) / name
    try:
        path.write_text(json.dumps(fields, indent=2), encoding='utf-8')
    except OSError:
        return None
    return path


def engine_argv(port, users, context, trace_region_bytes=1073741824):
    """The vLLM argv this gate serves, as a list, with nothing launched.

    Split out of start_server so test_m3native_engine_argv can assert it. Two rig
    slots were spent on argv mistakes that a CPU test would have caught in
    milliseconds: run 35679222511 served --no-enable-chunked-prefill on the
    chunked arm, and run 35681324335 was refused at startup because nothing
    zeroed the phantom multimodal item.
    """
    """The fast T16 + speculation recipe the four-user cycle bench serves
    (qwen-fp2u-image.yml), so the packed round under test is the one the 200
    tok/s/user work actually measures.

    --served-model-name is 'qwen-longctx', not this gate's own name: stream_once
    (longctx_cycle_bench.py, reused here exactly) hard-codes model='qwen-longctx' in
    its request payload, so any other served name 404s every stream in milliseconds
    (gate 1, run 35556533480 - a false negative that looked like readiness with zero
    decode rounds actually run)."""
    # trace_region_bytes: the recipe's 1 GiB unless the profile arm shrinks it - the device
    # profiler reserves ~0.72 GB of DRAM per chip (run 35563019626: 32.38 GB allocatable
    # instead of 33.10, and the fourth user's first proposal hit Out of Memory), while
    # decode traces are command streams that this repo's own probes run at 256 MiB.
    recipe = dict(tt=dict(trace_mode='decode_only', trace_region_size=int(trace_region_bytes),
                          l1_small_size=24576),
                 qwen_fast_t16=True,
                 qwen_fast_runtime=dict(directory='/experiment-scripts/ci', runtime_root='/opt/tt-metal',
                                        fixtures='/experiment-dflash-fixture', target_snapshot=MODEL))
    blocks = -(-(users * context) // BLOCK_SIZE)
    # Chunked prefill is what gives a prefill somewhere to yield, and it is opt-in so
    # every existing arm is byte-identical without it. Two invariants come with it and
    # both are load-bearing: max_num_batched_tokens must EQUAL the model chunk size,
    # because a larger budget hands the model a window it cannot replay as whole traced
    # chunks and a smaller one starts a continuation mid-chunk, breaking
    # start % chunk_size == 0; and long_prefill_token_threshold must not sub-chunk
    # inside vLLM, which would produce the same unaligned starts.
    chunk = os.environ.get('M3NATIVE_PREFILL_CHUNK_TOKENS')
    if chunk is None:
        batched_tokens, chunked_flags = context, ['--no-enable-chunked-prefill']
    else:
        size = int(chunk)
        if size not in (1024, 2048, 4096):
            raise ValueError('M3NATIVE_PREFILL_CHUNK_TOKENS must be 1024, 2048 or 4096')
        # CORRECTED by run 35688313093. The rule above USED to be that
        # max_num_batched_tokens equals the model chunk size, on the reasoning that a
        # larger budget hands the model a window it cannot replay as whole traced
        # chunks. What matters is the WINDOW, and long_prefill_token_threshold is what
        # sets it; the equality was a sufficient way to get there that speculative
        # decoding breaks. vLLM reserves draft-token slots out of the batched budget and
        # says so itself:
        #
        #   num_scheduled_tokens is set to 1992 based on the speculative decoding
        #   settings ... Consider increasing max_num_batched_tokens to accommodate
        #   the additional draft token slots
        #
        # 1992 is not a multiple of 128, so the graft's own
        # 'assert start % chunk_size == 0' would have fired on the next chunk. Giving
        # the budget headroom lets the threshold bind at exactly the chunk size.
        batched_tokens = 2 * size
        chunked_flags = ['--enable-chunked-prefill',
                         '--long-prefill-token-threshold', str(size)]
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
               '--model', MODEL, '--served-model-name', 'qwen-longctx',
               '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
               '--max-model-len', str(context), '--max-num-seqs', str(users),
               '--max-num-batched-tokens', str(batched_tokens),
               '--block-size', str(BLOCK_SIZE), '--num-gpu-blocks-override', str(blocks),
               '--no-enable-prefix-caching', '--no-async-scheduling',
               # Qwen3_5ForConditionalGeneration declares a 16384-token image item that
               # Qwen36ForCausalLM, the text-only TT class it resolves to, can never
               # consume. Left declared, vLLM sizes an encoder cache for it and, with
               # disable_chunked_mm_input set, refuses to start at all when the batched
               # budget is smaller (run 35681324335). lever_n_m1_gate has passed these
               # exact zeros since its v8 run, so the keys are established rather than
               # guessed; this is parity with a lane that works, and it makes
               # compute_mm_encoder_budget return (0, 0) instead of sizing a cache for
               # a modality this model has no weights for.
               '--limit-mm-per-prompt', json.dumps(dict(image=0, video=0)),
               *chunked_flags, '--shutdown-timeout', '30',
               '--additional-config', json.dumps(recipe),
               '--speculative-config', json.dumps(
                   dict(model='/draft-config', method='dflash', num_speculative_tokens=15,
                        draft_sample_method='greedy', rejection_sample_method='standard'))]
    return command


def start_server(port, users, context, results, log_name, readiness_seconds=900,
                 trace_region_bytes=1073741824):
    """The fast T16 + speculation recipe the four-user cycle bench serves
    (qwen-fp2u-image.yml), so the packed round under test is the one the 200
    tok/s/user work actually measures.

    --served-model-name is 'qwen-longctx', not this gate's own name: stream_once
    (longctx_cycle_bench.py, reused here exactly) hard-codes model='qwen-longctx' in
    its request payload, so any other served name 404s every stream in milliseconds
    (gate 1, run 35556533480 - a false negative that looked like readiness with zero
    decode rounds actually run)."""
    command = engine_argv(port, users, context, trace_region_bytes)
    log_path = results / log_name
    handle = log_path.open('w')
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + readiness_seconds
    while True:
        if process.poll() is not None:
            raise RuntimeError('server exited before readiness: %s' % process.returncode)
        try:
            with urlopen('http://127.0.0.1:%d/health' % port, timeout=2) as response:
                if response.status == 200:
                    return process, handle, log_path, command
        except (URLError, TimeoutError, OSError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError('readiness exceeded %ds' % readiness_seconds)
        time.sleep(2)


def stop_server(process, handle):
    if process is not None and process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=120)
        except BaseException:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except BaseException:
                pass
    if handle is not None:
        try:
            handle.close()
        except BaseException:
            pass


def prompt_for(base, offset, user, tokens):
    """Exact token ids [base' + i % 64], matching longctx_cycle_bench's own scheme
    (a per-user offset, so a corrupted user is told apart from a coincidence)."""
    start = base + user * offset
    return [start + (index % 64) for index in range(tokens)]


def packed_phase_stats(text):
    values = [float(v) for v in PACKED_PHASE_TRACE_MS.findall(text)]
    if not values:
        return None
    return dict(rounds=len(values), trace_ms_min=round(min(values), 3),
               trace_ms_mean=round(statistics.fmean(values), 3), trace_ms_max=round(max(values), 3))


def compare_prefix(actual, reference_text):
    """Whether `actual` (this run's stream) is consistent with `reference_text` (the
    single-user reference).

    A full-length or longer stream must match the reference exactly over the
    reference's own length - unchanged from the original all-length semantics. A
    stream shorter than the reference (as a profiling run capped at a small
    --max-tokens produces) cannot be checked that way, since `actual` never reaches
    the reference's length; instead it must itself be an exact prefix of the
    reference. Returns (identical_prefix, partial)."""
    if len(actual) < len(reference_text):
        return reference_text.startswith(actual), True
    return actual[:len(reference_text)] == reference_text, False


def retired_binder_leaks(rounds):
    """The rounds in which a RETIRED binder (RETIRED_LABELS) saw a call; every other
    label in the payload is a binder that is meant to run and is ignored here."""
    return [{label: calls for label, calls in payload.items() if label in RETIRED_LABELS and calls}
            for payload in rounds
            if any(payload.get(label) for label in RETIRED_LABELS)]


def evaluate_gate(*, ready, users, checked, allow_missing_references, native_m3_marker_present,
                  packed_phase, binder_rounds, retired_binder_calls_nonzero, full_output_required=True,
                  reference_run=False, missing_markers=()):
    """Whether the run passes, given the pieces `main` already computed.

    Full reference coverage (`len(checked) == users`) is required unless
    `allow_missing_references` - for an arm with no single-stream reference yet (e.g. 131k),
    where `checked` may be empty or short by design. Any reference that IS present must
    still match exactly (the `all(...)` term below is never relaxed): this only widens what
    counts as complete coverage, not what counts as a match.

    `reference_run` (--sequential-users): lone streams never form a packed round, so the
    native_m3 marker, [PACKED-PHASE] and binder terms cannot apply; every reference term
    still does. `missing_markers`: a capacity flag reached the server but its own marker
    never appeared, so the run did not do what its flags claim - never a pass."""
    coverage_ok = True if allow_missing_references else len(checked) == users
    # A stream that errored, or one cut short of its reference when the arm asked for the
    # full 256 tokens, is not a pass: run 35585107688 died of DRAM after three rounds with
    # every stream at three tokens and errored, yet its prefixes 'matched'.
    streams_ok = all(
        not c.get('error')
        and (not full_output_required or 'actual_len' not in c or 'reference_len' not in c
             or c['actual_len'] >= c['reference_len'])
        for c in checked)
    references_ok = bool(
        ready
        and coverage_ok
        and streams_ok
        and all(c.get('identical_prefix') for c in checked)
        # Two references for one base that disagree leave nothing to be exact against.
        and not any(c.get('reference_conflicts') for c in checked)
    )
    if missing_markers:
        return False
    if reference_run:
        return references_ok
    return bool(
        references_ok
        and native_m3_marker_present
        and packed_phase is not None
        and bool(binder_rounds)
        and not retired_binder_calls_nonzero
    )


def retired_binder_rounds(text):
    """Every per-round '[PINDIAG] native_m3 binder calls this round: {...}' payload
    (model_batch.ModelBatch.run), each mapping a retired binder's label to how many
    calls leaked through it this round - every value here must be zero."""
    rounds = []
    for match in BINDER_CALLS_LINE.finditer(text):
        try:
            payload = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            continue
        if isinstance(payload, dict):
            rounds.append(payload)
    return rounds


PROMPT_SOURCES = ('synthetic', 'real-text')
EOS_MODES = ('ignore', 'stop')
REAL_TEXT_PROMPTS = 'real-text-prompts.json'


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--users', type=int, default=4)
    parser.add_argument('--context', type=int, default=33024)
    parser.add_argument('--prompt-tokens', type=int, default=32768)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--stream-timeout', type=int, default=600)
    parser.add_argument('--prompt-base', type=int, default=1000)
    parser.add_argument('--prompt-user-offset', type=int, default=1)
    parser.add_argument('--stagger', type=float, default=0.0)
    parser.add_argument('--start-order', default=None,
                        help='comma-separated user indices, a permutation of 0..users-1: the order the request '
                             'threads start in, --stagger apart (needs --stagger > 0). The admission order fixes '
                             "each user's slot and pair row; two arms of one flag set in two orders compare each "
                             "user's packed fingerprints (QWEN_FAST_PAIR_ROW_EXACT's field check). Unset: 0..users-1")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--results', type=Path, default=Path('/tmp/m3native-gate'))
    parser.add_argument('--trace-region-bytes', type=int, default=1073741824,
                        help='device trace region per chip (the recipe uses 1 GiB; the profile arm passes less)')
    parser.add_argument('--references', type=Path,
                        default=Path('runner-evidence.local/packed-gate'),
                        help='directory holding single-user-*.json single-stream references')
    parser.add_argument('--sequential-users', type=int, default=0,
                        help='with --users 1: this many single-stream requests one after another'
                             ' on one server (bases base + i * offset), each alone - a reference run')
    parser.add_argument('--allow-missing-references', action='store_true',
                        help='for arms with no single-stream reference yet (e.g. 131k): do not '
                             'require every user to have a reference for gate_passed. Comparisons '
                             'still record reference_present, and any reference that IS present '
                             'must still match exactly; the dram/packed_phase reporting is unchanged.')
    parser.add_argument('--prompt-source', choices=PROMPT_SOURCES, default='synthetic',
                        help='synthetic (default): token ids [base + i %% 64] per user, exactly as before. '
                             'real-text: a real coding request per user from the image\'s own vLLM source '
                             '(real_text_prompts.py), exactly --prompt-tokens tokens (the image pins the request '
                             'position), so --prompt-tokens + --max-tokens must fit --context; needs '
                             '--allow-missing-references and --eos stop')
    parser.add_argument('--eos', choices=EOS_MODES, default=None,
                        help='ignore: ignore_eos=True, every stream runs to --max-tokens (the synthetic '
                             'default). stop: ignore_eos=False, a stream may end at EOS (the real-text '
                             'default, and the only mode real text accepts)')
    return parser


def parse_options(argv=None):
    """The gate's options, with --eos resolved and the real-text combinations it refuses refused."""
    parser = build_parser()
    options = parser.parse_args(argv)
    if options.sequential_users and options.users != 1:
        parser.error('--sequential-users needs --users 1: each request must run alone')
    if options.start_order is not None:
        try:
            options.start_order = start_order(options.start_order, options.users)
        except ValueError as error:
            parser.error(str(error))
        if options.sequential_users or not options.stagger > 0:
            parser.error('--start-order needs --stagger > 0 and concurrent users: without a stagger the requests race')
    real_text = options.prompt_source == 'real-text'
    if options.eos is None:
        options.eos = 'stop' if real_text else 'ignore'
    if real_text and not options.allow_missing_references:
        parser.error('--prompt-source real-text needs --allow-missing-references: the single-user-*.json '
                     'references are synthetic (base-keyed [base + i % 64] prompts), so no real-text prompt has '
                     'one; exactness is checked offline against a sequential real-text arm (real_text_compare.py)')
    if real_text and options.eos != 'stop':
        parser.error('--prompt-source real-text needs --eos stop: the fast path\'s GreedySession finishes a '
                     'request at the snapshot EOS whatever ignore_eos says (serving_request_factory passes eos_ids, '
                     'greedy_verify.select_prefix stops there), and real text reaches EOS')
    if real_text and real_text_target(options) != options.prompt_tokens:
        parser.error('--prompt-source real-text needs --prompt-tokens + --max-tokens <= --context (got %d + %d > %d): '
                     'every prompt must be exactly --prompt-tokens long, because the image pins each request\'s '
                     'position to the request context the arm sets from it (QWEN_DSPARK_REQUEST_CONTEXT; '
                     'frozen_combined_runtime.validate_target_option refuses any other position)'
                     % (options.prompt_tokens, options.max_tokens, options.context))
    return options


def start_order(text, users):
    """--start-order as a list: a permutation of 0..users-1, or ValueError."""
    try:
        order = [int(part) for part in str(text).split(',')]
    except ValueError:
        raise ValueError('--start-order must be comma-separated user indices, got %r' % (text,))
    if sorted(order) != list(range(users)):
        raise ValueError('--start-order must be a permutation of 0..%d, got %r' % (users - 1, text))
    return order


def request_order(options):
    """The order the request threads start in: --start-order, or 0..users-1."""
    return list(options.start_order) if getattr(options, 'start_order', None) else list(range(options.users))


def start_order_report(report, options):
    """With --start-order: the order given and whether the packed fingerprints' admission order (segment ->
    user, acceptance_report.packed_fingerprints) is that order. Diagnostic; nothing here fails a run."""
    if not getattr(options, 'start_order', None):
        return
    admitted = (report.get('packed_fingerprints') or {}).get('admission_order')
    report['start_order'] = dict(requested=list(options.start_order), admitted=admitted,
                                  matches=admitted == list(options.start_order))
    print('[START-ORDER] requested=%s admitted=%s matches=%s' % (
        report['start_order']['requested'], admitted, report['start_order']['matches']), flush=True)


def real_text_target(options):
    """A real-text prompt's length: min(--prompt-tokens, --context - --max-tokens), which
    parse_options requires to BE --prompt-tokens (33024 - 256 = 32768, 131328 - 256 = 131072).
    The arms keep their --context: KV sizing and the 4 x 131k fit are computed from it."""
    return min(options.prompt_tokens, options.context - options.max_tokens)


def detail_mode(options):
    """Anything but the default mode (synthetic prompts, --eos ignore). Only here does the gate
    ask for detail streams and add the acceptance, rate and configuration diagnostics; the
    default mode's requests, report and stdout stay exactly what they were."""
    return not (options.prompt_source == 'synthetic' and options.eos == 'ignore')


def stream_kwargs(options):
    """stream_once's keywords. None at all in the default mode (synthetic, --eos ignore), so every
    existing call is unchanged - the payload stream_once sends and the fields it records."""
    if not detail_mode(options):
        return {}
    return dict(ignore_eos=options.eos == 'ignore', detail=True)


def real_text_stream_problems(results, prompt_lengths=None):
    """A real-text arm has no reference to fail a bad stream, so each stream must itself have
    completed: no error, some text, a finish_reason of stop (EOS) or length (the budget), and -
    given the built lengths - a usage.prompt_tokens equal to its prompt's length, the cheap proof
    that the server served the built ids with no template or truncation of its own."""
    problems = []
    for index, entry in enumerate(results):
        entry = entry or {}
        if entry.get('error'):
            problems.append('user %d: stream error %s' % (index, str(entry['error'])[:200]))
        elif not entry.get('text'):
            problems.append('user %d: no text' % index)
        elif entry.get('finish_reason') not in ('stop', 'length'):
            problems.append('user %d: finish_reason %r, not stop or length' % (index, entry.get('finish_reason')))
        elif prompt_lengths is not None and entry.get('prompt_tokens') != prompt_lengths[index]:
            problems.append('user %d: the server reports usage.prompt_tokens %r for a %d-token prompt'
                            % (index, entry.get('prompt_tokens'), prompt_lengths[index]))
    return problems


def marker_prompt_tokens(options, report):
    """The ONE prompt length the flag-marker floors are computed from. Synthetic: --prompt-tokens.
    Real text: the shortest built prompt, which parse_options and real_text_prompts make every
    prompt's length, --prompt-tokens (the image pins it), so the floors are the synthetic arm's:
    users x ceil(L / 2048) engaged QWEN_FAST_GDN_PREFILL_CONV chunks (16 per user at 32k, 64 at
    131k) and the SDPA_PF 2048-row topology iff L >= 2048. The minimum is kept so a prompt set
    that ever differed per user could only lower a floor, never over-require one."""
    lengths = (report.get('real_text') or {}).get('prompt_lengths')
    if options.prompt_source == 'real-text' and lengths:
        return min(lengths)
    return options.prompt_tokens


def qwen_configuration(environ):
    """The QWEN_* flags the server process inherits: what real_text_compare.py diffs between a
    concurrent and a sequential arm, whose flag sets differ (v157/v159 vs v158/v160)."""
    return {name: environ[name] for name in sorted(environ) if name.startswith('QWEN_')}


def _round_trip(value):
    """`value` as JSON would carry it; raises (inside the caller's guard) if it cannot."""
    return json.loads(json.dumps(value))


def add_run_diagnostics(report, log_path, sequential):
    """The acceptance report and the per-user decode rate (acceptance_report.py), from the FULL
    server log once the server has stopped: the last requests' fast_serving_phases records print
    after SIGTERM. Diagnostic, like the TTFT profile: a failure here - including a value JSON
    cannot carry, which would otherwise break the report print after BEGIN - is recorded in the
    report and never reaches gate_passed."""
    streams = report.get('streams')
    if streams is None:
        return
    try:
        from acceptance_report import report as acceptance
        log_text = log_path.read_text(errors='replace') if log_path.is_file() else ''
        lengths = [entry['prompt_tokens'] for entry in (report.get('real_text') or {}).get('users') or []] or None
        report['acceptance'] = _round_trip(acceptance(log_text, streams, lengths, sequential=sequential))
        print(report['acceptance']['summary_line'], flush=True)
    except Exception as error:
        report['acceptance'] = dict(error='%s: %s' % (type(error).__name__, error))
        print('[ACCEPT] report unavailable: %s' % report['acceptance']['error'], flush=True)
    try:
        from acceptance_report import decode_rates, rate_line
        accepted = report['acceptance'] if 'error' not in report['acceptance'] else None
        report['decode_rate'] = _round_trip(decode_rates(streams, concurrent=not sequential, acceptance=accepted))
        print(rate_line(report['decode_rate']), flush=True)
    except Exception as error:
        report['decode_rate'] = dict(error='%s: %s' % (type(error).__name__, error))
        print('[RATE] unavailable: %s' % report['decode_rate']['error'], flush=True)
    try:
        # Round-fence plan H2's draft check, offline: each user's [PACKED] sequence hashed, with the admission
        # order - two arms of one image with the order fixed (M3NATIVE_STAGGER) compare report to report, and
        # acceptance_report.py --packed-reference compares their logs round by round.
        from acceptance_report import packed_fingerprints
        log_text = log_path.read_text(errors='replace') if log_path.is_file() else ''
        report['packed_fingerprints'] = _round_trip(packed_fingerprints(log_text, streams))
    except Exception as error:
        report['packed_fingerprints'] = dict(error='%s: %s' % (type(error).__name__, error))


def main():
    options = parse_options()
    real_text = options.prompt_source == 'real-text'
    try:
        options.results.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    report = dict(scope=__doc__, users=options.users, context=options.context,
                 prompt_tokens=options.prompt_tokens, prompt_base=options.prompt_base,
                 prompt_user_offset=options.prompt_user_offset, ready=False)
    detail = detail_mode(options)
    if detail:
        # Only outside the default mode: a default arm's report keeps exactly today's keys.
        report.update(prompt_source=options.prompt_source, eos=options.eos, max_tokens=options.max_tokens,
                      qwen_configuration=qwen_configuration(os.environ))
    process = handle = None
    streams = options.sequential_users or options.users
    prompts = None
    try:
        # Real text: the references are synthetic, so none is loaded (parse_options required
        # --allow-missing-references).
        references = {} if real_text else load_references(options.references, options.prompt_tokens)
        report['sequential_users'] = options.sequential_users
        report['references_loaded'] = sorted(references)
        if real_text:
            # Built before the server starts, so a tokenizer or corpus problem fails in seconds
            # rather than after the engine's startup. N is the number of streams either way, so a
            # concurrent arm and a sequential arm of the same N on one image build the same set.
            import real_text_prompts
            report['real_text_target'] = real_text_target(options)
            built = real_text_prompts.build_prompts(streams, report['real_text_target'], model=MODEL,
                                                    log=lambda line: print(line, flush=True))
            real_text_prompts.write_prompts(options.results / REAL_TEXT_PROMPTS, built)
            report['real_text'] = real_text_prompts.summary(built)
            prompts = [entry['tokens'] for entry in built['users']]
            print('[REALTEXT] %d prompts of %s tokens (target %d) in %.1f s, %d tokenizer calls; corpus %s files, '
                  '%s characters, sha256 %s' % (
                      streams, report['real_text']['prompt_lengths'], report['real_text_target'],
                      built['seconds']['total'], built['tokenizer_calls'], built['corpus'].get('files'),
                      built['corpus'].get('characters'), str(built['corpus'].get('sha256'))[:16]), flush=True)

        def prompt(index):
            if prompts is not None:
                return prompts[index]
            return prompt_for(options.prompt_base, options.prompt_user_offset, index, options.prompt_tokens)

        process, handle, log_path, command = start_server(
            options.port, options.users, options.context, options.results, 'server.log',
            trace_region_bytes=options.trace_region_bytes)
        report['command'] = command
        report['ready'] = True

        results = [None] * streams
        kwargs = stream_kwargs(options)
        if options.sequential_users:
            # One request at a time: each is the only stream on the server, which is
            # what a single-stream reference means.
            for index in range(streams):
                stream_once(options.port, prompt(index), options.max_tokens, results, index, options.stream_timeout,
                            **kwargs)
        threads = [] if options.sequential_users else [threading.Thread(
            target=stream_once,
            args=(options.port, prompt(index), options.max_tokens, results, index, options.stream_timeout),
            kwargs=kwargs)
            for index in range(options.users)]
        for position, index in enumerate(request_order(options) if threads else []):
            if position and options.stagger:
                time.sleep(options.stagger)
            threads[index].start()
        for thread in threads:
            thread.join()
        report['streams'] = results
        # Derived, not measured: the gate already records ttft_s and gaps_ms per
        # stream, and run 35658854824 showed those carry a precise serial-prefill
        # story no gate asserted. Reporting is unconditional; the thresholds are
        # opt-in, so this cannot fail a run until someone sets a ceiling.
        # Never let a diagnostic fail a run that would otherwise pass. Run
        # 35668593700 lost a whole rig run because this import raised
        # ModuleNotFoundError inside the container - the module is mounted at
        # /bench now, but the guard stays, because the profile is reporting and
        # the gate's verdict does not depend on it.
        try:
            from m3native_ttft_profile import profile as ttft_profile, render as ttft_render
            report['ttft_profile'] = ttft_profile(results)
            print(ttft_render(report['ttft_profile']), flush=True)
        except Exception as error:
            report['ttft_profile'] = dict(error='%s: %s' % (type(error).__name__, error))
            print('[TTFT] profile unavailable: %s' % report['ttft_profile']['error'], flush=True)

        comparisons = []
        for index, entry in enumerate(results):
            base = options.prompt_base + index * options.prompt_user_offset
            reference = references.get(base)
            comparison = dict(user=index, prompt_base=base, reference_present=bool(reference))
            user_prompt = report['real_text']['users'][index] if real_text else None
            if user_prompt is not None:
                # What real_text_compare.py matches the sequential arm's stream on.
                comparison.update(prompt_sha256=user_prompt['prompt_sha256'],
                                  prompt_tokens=user_prompt['prompt_tokens'],
                                  served_prompt_tokens=(entry or {}).get('prompt_tokens'),
                                  completion_tokens=(entry or {}).get('completion_tokens'),
                                  max_tokens=options.max_tokens,
                                  finish_reason=(entry or {}).get('finish_reason'),
                                  actual_len=len((entry or {}).get('text') or ''))
                if entry and entry.get('error'):
                    comparison['error'] = entry['error']
            if options.users == 1:
                # Only a stream that ran alone is a single-stream reference candidate.
                if user_prompt is not None:
                    write_candidate(options.results, base, user_prompt['prompt_tokens'], entry,
                                    prompt_sha256=user_prompt['prompt_sha256'])
                else:
                    write_candidate(options.results, base, options.prompt_tokens, entry)
            if reference:
                comparison['reference_path'] = reference['path']
                comparison['reference_context'] = reference['context']
                if reference.get('conflicts'):
                    comparison['reference_conflicts'] = reference['conflicts']
                comparison['reference_len'] = len(reference['text'])
                actual = (entry or {}).get('text') or ''
                comparison['actual_len'] = len(actual)
                identical_prefix, partial = compare_prefix(actual, reference['text'])
                comparison['identical_prefix'] = identical_prefix
                if partial:
                    comparison['partial'] = True
                if entry and entry.get('error'):
                    comparison['error'] = entry['error']
            comparisons.append(comparison)
        report['comparisons'] = comparisons

        log_text = log_path.read_text(errors='replace') if log_path.is_file() else ''
        report['native_m3_marker_present'] = NATIVE_M3_MARKER in log_text
        report['native_attn_engaged'] = NATIVE_ATTN_MARKER in log_text
        report['packed_phase'] = packed_phase_stats(log_text)
        binder_rounds = retired_binder_rounds(log_text)
        report['retired_binder_rounds_observed'] = len(binder_rounds)
        report['retired_binder_calls_nonzero'] = retired_binder_leaks(binder_rounds)

        report['flag_markers'] = flag_marker_report(os.environ, streams, log_text,
                                                    prompt_tokens=marker_prompt_tokens(options, report))
        checked = [c for c in comparisons if c.get('reference_present')]
        report['users_checked'] = len(checked)
        report['allow_missing_references'] = options.allow_missing_references
        report['gate_passed'] = evaluate_gate(
            ready=report['ready'], users=streams, checked=checked,
            allow_missing_references=options.allow_missing_references,
            native_m3_marker_present=report['native_m3_marker_present'],
            packed_phase=report['packed_phase'], binder_rounds=binder_rounds,
            retired_binder_calls_nonzero=report['retired_binder_calls_nonzero'],
            full_output_required=options.max_tokens >= 256,
            reference_run=bool(options.sequential_users),
            missing_markers=report['flag_markers']['missing'])
        if real_text:
            report['real_text_stream_problems'] = real_text_stream_problems(
                results, report['real_text']['prompt_lengths'])
            report['gate_passed'] = bool(report['gate_passed'] and not report['real_text_stream_problems'])
    except BaseException as error:
        report['fatal'] = '%s: %s' % (type(error).__name__, str(error)[:600])
    finally:
        stop_server(process, handle)
        log_path = options.results / 'server.log'
        if detail:
            add_run_diagnostics(report, log_path, sequential=bool(options.sequential_users))
        try:
            start_order_report(report, options)
        except Exception as error:
            report['start_order'] = dict(error='%s: %s' % (type(error).__name__, error))
        print(BEGIN)
        print(json.dumps(report, indent=2))
        print(END)
        print(LOG_BEGIN)
        if log_path.is_file():
            lines = log_path.read_text(errors='replace').splitlines()
            for line in select_diagnostic(lines):
                print(line)
            for line in lines[-200:]:
                print(line[:300])
        print(LOG_END)
        sys.stdout.flush()
    return 0 if report.get('gate_passed') else 1


if __name__ == '__main__':
    sys.exit(main())
