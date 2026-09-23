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
# out of the middle at 800.
DIAGNOSTIC_CAP = 4000
# A native death leaves no Python traceback: TT_FATAL / TT_THROW text, the C++ runtime's
# terminate message, the shell's signal report, or vLLM's engine-death notice are the
# only record (run 35579223088 had none of them in the kept lines).
CRASH_TEXT = ('FATAL', 'Segmentation', 'Aborted', 'Killed', 'terminate called', 'what():',
              'core dumped', 'died', 'Bus error', 'Illegal instruction', 'TT_THROW')


def select_diagnostic(lines, cap=DIAGNOSTIC_CAP):
    diagnostic = [line[:300] for line in lines
                  if '[PINDIAG]' in line or '[PACKED' in line or '[PHASE]' in line
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


def single_gateup_markers(environ):
    """SINGLE_GATEUP_MARKERS plus the executed prefill MLP branch's marker: C1c by default, C1's
    2D branch under QWEN_FAST_C1_LEGACY=1, or (QWEN_FAST_C1_AGMM=1) C1d's two in place of the
    ff_norm's gathers-for-itself marker."""
    markers = list(SINGLE_GATEUP_MARKERS)
    if environ.get('QWEN_FAST_C1_AGMM') == '1':
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
DRAFT_BF8_MARKER = 'projections dtype=bf8 x36'
LEDGER_MARKERS = ('[MEMLEDGER] phase=P7 ', ' check=residual status=')
# QWEN_FAST_SDPA_MODES (optimisation/ttnn-op/sdpa_decode_qwen: 'tail' stage 1, 'share' stage 3).
# The two [PINDIAG] lines are emitted inside pooled_attention_replay.apply_sdpa_modes (the
# loaded _ttnncpp.so was checked for the factory branch; a replay reader's configs were
# rewritten), never at install or mount time. The [QWEN-SDPA] line is the grafted factory's own
# log_info (F4), printed when it builds a program in that mode: the C++ branch itself ran, not
# just the Python that selects it. With share the line must carry flag 0x2 (0x3 with tail),
# which the reader sets only on bundles of more than one entry, so kv_share=true was built.
SDPA_MODES_MARKERS = ('[PINDIAG] sdpa qwen-modes binary ', '[PINDIAG] sdpa qwen-modes modes=tail ',
                      '[QWEN-SDPA] flags=0x1 ')
SDPA_MODE_FLAGS = {'tail': 0x1, 'share': 0x2}
GDN_ALL_BATCHED = re.compile(r'gdn user_batched calls this captured forward: ([1-9][0-9]*) of ([0-9]+) GDN layers')
LEDGER_RESIDUAL = re.compile(r'\[MEMLEDGER\] phase=P7 [^\n]*check=residual status=([a-zA-Z]+)')


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
    """The three markers a QWEN_FAST_SDPA_MODES value promises, or [] when it names no served
    mode. The modes line is apply_sdpa_modes' sorted join; the factory line carries every
    requested flag (tail alone: 0x1, the stage-1 markers exactly; tail,share: 0x3)."""
    served = sorted(names.intersection(SDPA_MODE_FLAGS))
    if not served:
        return []
    flags = 0
    for name in served:
        flags |= SDPA_MODE_FLAGS[name]
    return [SDPA_MODES_MARKERS[0], '[PINDIAG] sdpa qwen-modes modes=%s ' % ','.join(sorted(names)),
            '[QWEN-SDPA] flags=0x%x ' % flags]


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
    residual = LEDGER_RESIDUAL.search(log_text)
    return dict(found=found, missing=missing, ledger_residual=residual.group(1) if residual else None,
                prefill_conv_chunk_calls=prefill_conv, prefill_conv=summary, verify_t1_sites=verify_t1_sites,
                verify_t1_packed=verify_t1_packed)


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


def write_candidate(directory, base, prompt_tokens, entry):
    """This run's stream for `base`, in the tracked reference format, for promotion to
    scripts/ci/references/packed-gate after review. Never read back by the gate."""
    if not entry or entry.get('error') or not entry.get('text'):
        return None
    path = Path(directory) / ('reference-candidate-%d-p%d.json' % (base, prompt_tokens))
    try:
        path.write_text(json.dumps(dict(label='candidate', prompt_base=base, prompt_tokens=prompt_tokens,
                                        streams=[entry]), indent=2), encoding='utf-8')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--users', type=int, default=4)
    parser.add_argument('--context', type=int, default=33024)
    parser.add_argument('--prompt-tokens', type=int, default=32768)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--stream-timeout', type=int, default=600)
    parser.add_argument('--prompt-base', type=int, default=1000)
    parser.add_argument('--prompt-user-offset', type=int, default=1)
    parser.add_argument('--stagger', type=float, default=0.0)
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
    options = parser.parse_args()
    if options.sequential_users and options.users != 1:
        parser.error('--sequential-users needs --users 1: each request must run alone')
    try:
        options.results.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    report = dict(scope=__doc__, users=options.users, context=options.context,
                 prompt_tokens=options.prompt_tokens, prompt_base=options.prompt_base,
                 prompt_user_offset=options.prompt_user_offset, ready=False)
    process = handle = None
    try:
        references = load_references(options.references, options.prompt_tokens)
        report['sequential_users'] = options.sequential_users
        report['references_loaded'] = sorted(references)

        process, handle, log_path, command = start_server(
            options.port, options.users, options.context, options.results, 'server.log',
            trace_region_bytes=options.trace_region_bytes)
        report['command'] = command
        report['ready'] = True

        streams = options.sequential_users or options.users
        results = [None] * streams
        if options.sequential_users:
            # One request at a time: each is the only stream on the server, which is
            # what a single-stream reference means.
            for index in range(streams):
                stream_once(options.port, prompt_for(options.prompt_base, options.prompt_user_offset,
                                                     index, options.prompt_tokens),
                            options.max_tokens, results, index, options.stream_timeout)
        threads = [] if options.sequential_users else [threading.Thread(
            target=stream_once,
            args=(options.port, prompt_for(options.prompt_base, options.prompt_user_offset, index,
                                           options.prompt_tokens),
                  options.max_tokens, results, index, options.stream_timeout))
            for index in range(options.users)]
        for index, thread in enumerate(threads):
            if index and options.stagger:
                time.sleep(options.stagger)
            thread.start()
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
            if options.users == 1:
                # Only a stream that ran alone is a single-stream reference candidate.
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
                                                    prompt_tokens=options.prompt_tokens)
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
    except BaseException as error:
        report['fatal'] = '%s: %s' % (type(error).__name__, str(error)[:600])
    finally:
        stop_server(process, handle)
        print(BEGIN)
        print(json.dumps(report, indent=2))
        print(END)
        print(LOG_BEGIN)
        log_path = options.results / 'server.log'
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
