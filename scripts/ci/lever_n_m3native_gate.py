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

PLATFORM ARGV (--server-argv platform; default gate, which leaves the argv above untouched). For the
C2 serving image (docker/qwen-c2-serving.Dockerfile, run by scripts/ci/c2_serving_gate.py in the
node agent's container shape): the gate launches vLLM with the platform's argv (PLATFORM_ARGV, the
smoke's), so the image's serving contract (serving_c2_contract, booted by QWEN_C2_SERVING=1)
rewrites it to its profile's exactly as it does for the agent. What was actually served is read
back from the contract's own '[QWEN-C2] profile <name>: vLLM argv [...]' line (read-the-launched-
argv) into report['platform'], which a run without that line, or with another profile than
--expect-profile, fails. Requests name --served-model-name (the platform's). The report also
carries every '[PINDIAG] dram after ...' line parsed (report['dram'], G5) and the memory ledger
read on its own (report['ledger']: the P7 residual status and the idle allocation before the first
prefill and after every stream). report['command'] is then the argv that served (the contract's),
report['command_requested'] the platform's. --prompt-lengths L1,L2,... (real text only) gives each
user its own prompt length (real_text_prompts targets).

LIFECYCLE (the C2 serving gate's lifecycle arms, detail mode): --drops (DROP_GRAMMAR), --user-max-tokens
and --user-ignore-eos act per user; a StreamWatch puts every stream and the server log's admission
markers on one clock, fires the drops, and records the phase each one hit (report['lifecycle']).
--alive-check N then sends N requests at once (the engine's seats) that must all answer.

S2 (C2-packed-any, s2-design.md W11; detail mode). Under QWEN_FAST_EXTENT_REPLAY=1 the promised SDPA modes gain
'extent' (flags 0x27 and K64j's F22 line), and an arm with that flag, a gate-only knob (QWEN_FAST_EXTENT_AUDIT,
QWEN_FAST_PACKED_CAPTURE_POSITION, QWEN_FAST_GATE_FORCE_CAP) or any S2 line gets report['s2'] (s2_report): the
extent rounds and their families, the extent audit, cap-refused / deadline / idle-commit / refuse_round / narrowed
/ aborted / dram hold / quarantine / release lines, the ledger's before-points and their floor, the per-user path
records (P packed or S sequential per round, acceptance_report.path_records) and the four-live rate. Its
problems join flag_markers['missing']. Every other arm's report is unchanged.
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


# --server-argv platform: what the platform hands vLLM (the smoke step of qwen-c2-serving.yml, which
# mirrors the Thatch runtime's launch), before the image's contract replaces every engine flag its
# profile owns. The served name, host, port and parsers are the platform's and survive the rewrite.
SERVER_ARGV_MODES = ('gate', 'platform')
PLATFORM_SERVED_NAME = 'Qwen/Qwen3.8-27B'
PLATFORM_ADDITIONAL_CONFIG = {'tt': {'l1_small_size': 24576, 'fabric_config': 'FABRIC_1D',
                                     'trace_region_size': 1073741824}}
QWEN_C2_ARGV = re.compile(r'\[QWEN-C2\] profile (\S+): vLLM argv (\[.*\])[ \t]*$', re.M)
QWEN_C2_MARKER = '[QWEN-C2] '
QWEN_C2_CONTRACT = '[QWEN-C2] request contract installed'
QWEN_C2_LINES_CAP = 40
# '[PINDIAG] dram after attach: ...' once, then 'dram after engine <request>: ...' per admitted request
# (serving_runtime.py): each chip's allocated / free / largest free block (G5 reads the last engine's).
DRAM_LINE = re.compile(r'\[PINDIAG\] dram after (attach|engine (\S+)): ([^\n]*)')
DRAM_CHIP = re.compile(r'chip([0-9]+) allocated=([0-9.]+)GB free=([0-9.]+)GB largest_free=([0-9.]+)MB of ([0-9.]+)GB')


def platform_argv(port, served_name=PLATFORM_SERVED_NAME):
    """The vLLM argv a platform launch passes, as a list (nothing launched). The contract keeps
    --served-model-name, --host, --port and the parsers, and replaces the rest with its profile's."""
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', 'Qwen/Qwen3.8-27B', '--served-model-name', served_name,
            '--host', '0.0.0.0', '--port', str(port),
            '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_xml', '--enable-auto-tool-choice',
            '--max-model-len', '65536', '--max-num-seqs', '2', '--block-size', '64', '--no-enable-prefix-caching',
            '--additional-config', json.dumps(PLATFORM_ADDITIONAL_CONFIG)]


def platform_report(log_text, expect_profile=None):
    """What the contract served, from its own log lines: the profile, the launched argv (the
    rewritten one), whether the request contract was installed, every [QWEN-C2] line (capped), and
    the problems that fail a platform run (no argv line: the contract never ran in the API server;
    another profile than expected; two different argv lines)."""
    launches = [(match.group(1), match.group(2)) for match in QWEN_C2_ARGV.finditer(log_text)]
    lines = [line.strip()[:400] for line in log_text.splitlines() if QWEN_C2_MARKER in line]
    problems = []
    served_argv = served_profile = None
    if not launches:
        problems.append('no "[QWEN-C2] profile <name>: vLLM argv" line: the serving contract did not rewrite the '
                        'API server argv (QWEN_C2_SERVING unset, or its boot failed)')
    else:
        served_profile, text = launches[-1]
        try:
            served_argv = json.loads(text)
        except ValueError:
            served_argv = text
            problems.append('the launched argv line is not a JSON list: %s' % text[:200])
        if len(set(launches)) > 1:
            problems.append('%d different launched argv lines (a restart with another profile?)' % len(set(launches)))
        if expect_profile and served_profile != expect_profile:
            problems.append('served profile %r, expected %r' % (served_profile, expect_profile))
    return dict(served_profile=served_profile, served_argv=served_argv, launches=len(launches),
                contract_installed=QWEN_C2_CONTRACT in log_text, qwen_c2_lines=lines[:QWEN_C2_LINES_CAP],
                problems=problems)


def dram_report(log_text):
    """Every '[PINDIAG] dram after attach|engine <request>' line, parsed per chip, and the floor
    over the engine lines (the G5 reading: DRAM left once every admitted request built its engine)."""
    events = []
    for match in DRAM_LINE.finditer(log_text):
        chips = [dict(chip=int(chip), allocated_gb=float(allocated), free_gb=float(free),
                      largest_free_mb=float(largest), total_gb=float(total))
                 for chip, allocated, free, largest, total in DRAM_CHIP.findall(match.group(3))]
        events.append(dict(event='attach' if match.group(1) == 'attach' else 'engine', request=match.group(2),
                           chips=chips, line=match.group(0)[:400]))
    engine_chips = [chip for event in events if event['event'] == 'engine' for chip in event['chips']]
    return dict(events=events, engines=sum(1 for event in events if event['event'] == 'engine'),
                min_free_gb=min((chip['free_gb'] for chip in engine_chips), default=None),
                min_largest_free_mb=min((chip['largest_free_mb'] for chip in engine_chips), default=None))


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
# K64j (optimisation/ttnn-op/k64j): 'extent' is flag 0x20 (0x21 with tail; it needs tail). The factory's F22
# line, logged once per program built with 0x20, is required as well: the runtime-extent branch that hands the
# cur_pos word to every kernel ran (K64i and older refuse 0x20 as an unknown flag and never log it).
SDPA_MODES_MARKERS = ('[PINDIAG] sdpa qwen-modes binary ', '[PINDIAG] sdpa qwen-modes modes=tail ',
                      '[QWEN-SDPA] flags=0x1 ')
SDPA_MODE_FLAGS = {'tail': 0x1, 'share': 0x2, 'slice': 0x4, 'readahead': 0x8, 'extent': 0x20}
SDPA_SLICE_MARKER = '[QWEN-SDPA] q-slice rows_per_kv='   # apply_factory_slice.SLICE_LOG_MARKER (F18)
SDPA_EXTENT_MARKER = '[QWEN-SDPA] runtime-extent entries='   # apply_factory_k64j.EXTENT_LOG_MARKER (F22)
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


# Q4, the four-user 64-row draft pass (quad_draft.py; QWEN_FAST_QUAD_DRAFT, default off; needs four users and
# QWEN_FAST_PACKED_PROPOSAL, _PAIR_ROW_EXACT, _ROUND_B1 and _FUSED_COMMIT_LIVE_BANKS). It promises
# '[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=64/16 rows=64 sdpa=S conv=C', logged EXACTLY once, after the
# first quad bucket captured and replayed, and under QWEN_FAST_PACKED_AUDIT one '[QUAD-DRAFT] round=R built=B ms=X'
# per quad round. A fallback ('[QUAD-DRAFT] fallback round=R reason=...': a failed build or replay, or too little
# DRAM for a fresh build) or '[PINDIAG] quad draft disabled' (two failures in a row, or a missing requirement)
# fails the arm: a quad that quietly served pairs would otherwise pass as a quad arm. The quad serves every round
# with four packable users, so over at least QUAD_FLOOR_ROUNDS four-user rounds ('[PACKED-SELECT] ... users=4')
# it must serve QUAD_SHARE_FLOOR of them, and at least one as soon as there is one. Both kinds of line are logged
# only under QWEN_FAST_PACKED_AUDIT=1 (the arm always passes it), so the flag needs it: without it this count
# would pass on no lines at all. The marker must name the modes the arm asked for (QWEN_FAST_QUAD_SDPA, default
# fold; QWEN_FAST_QUAD_CONV, default 110), over slots 0-3 at 64 rows: an image whose defaults differ, or a sub-flag
# that never reached the server, would otherwise pass as the arm it was not. '0' or unset with no sub-flag is off:
# no report, the parent's output.
# QWEN_FAST_QUAD_DRAFT_AUDIT=all|N: one '[QUAD-AUDIT] round=R equal=E stage=S users=4 checks=C' per audited quad
# round (all of them, or the first N): every line equal=1, and as many lines as audited rounds. The audited rounds
# are counted from the round lines, so the audit needs QWEN_FAST_PACKED_AUDIT=1 (without it zero lines would match
# zero counted rounds), and it must cover at least QUAD_AUDIT_FLOOR rounds (N, if N is smaller): plan Q3.1's
# '[QUAD-AUDIT] lines = quad rounds, at least 20' - a run with a handful of four-user rounds proves nothing.
QUAD_DRAFT_FLAG = 'QWEN_FAST_QUAD_DRAFT'
QUAD_SDPA_FLAG = 'QWEN_FAST_QUAD_SDPA'
QUAD_CONV_FLAG = 'QWEN_FAST_QUAD_CONV'
QUAD_AUDIT_FLAG = 'QWEN_FAST_QUAD_DRAFT_AUDIT'
QUAD_MARKER = '[PINDIAG] quad draft engaged'
QUAD_DISABLED_MARKER = '[PINDIAG] quad draft disabled'
QUAD_FALLBACK_MARKER = '[QUAD-DRAFT] fallback'
QUAD_ROUND_LINE = re.compile(r'\[QUAD-DRAFT\] round=([0-9]+) built=([01]) ms=([0-9.]+)')
QUAD_AUDIT_LINE = re.compile(r'\[QUAD-AUDIT\] round=([0-9]+|None) equal=([01]) stage=(\S+) users=([0-9]+) checks=([0-9]+)')
QUAD_MASK_LINE = re.compile(r'\[QUAD-DRAFT\] mask round=(\S+) intact=([01]) mismatched=([0-9]+) chip=([0-9]+)')
QUAD_ENGAGED_LINE = re.compile(r'\[PINDIAG\] quad draft engaged slots=\[([0-9,]+)\] heads=(\S+) rows=([0-9]+) '
                               r'sdpa=(\S+) conv=(\S+)')
FOUR_USER_SELECT = re.compile(r'\[PACKED-SELECT\] round=([0-9]+) pairs=.*? users=4 ')
QUAD_FLOOR_ROUNDS = 10
QUAD_SHARE_FLOOR = 0.95
QUAD_AUDIT_FLOOR = 20


def quad_draft_markers(environ, users):
    """{flag: [marker]} for QWEN_FAST_QUAD_DRAFT when `environ` sets it on a four-user arm."""
    if environ.get(QUAD_DRAFT_FLAG) == '1' and users == 4:
        return {QUAD_DRAFT_FLAG: [QUAD_MARKER]}
    return {}


def quad_draft_summary(log_text):
    """What the quad's lines say: the marker count, its rounds (and builds), the four-user rounds, fallbacks,
    disables, the audit lines and the mask read-backs."""
    rounds = [match.groups() for match in QUAD_ROUND_LINE.finditer(log_text)]
    audits = [match.groups() for match in QUAD_AUDIT_LINE.finditer(log_text)]
    masks = [match.groups() for match in QUAD_MASK_LINE.finditer(log_text)]
    ms = sorted(float(line[2]) for line in rounds if line[1] == '0')
    return dict(
        markers=log_text.count(QUAD_MARKER), rounds=len(rounds), builds=sum(1 for line in rounds if line[1] == '1'),
        four_user_rounds=len(FOUR_USER_SELECT.findall(log_text)),
        fallbacks=log_text.count(QUAD_FALLBACK_MARKER), disabled=log_text.count(QUAD_DISABLED_MARKER),
        audits=len(audits), audits_equal=sum(1 for line in audits if line[1] == '1'),
        audit_stages=sorted({line[2] for line in audits if line[1] != '1'}),
        prepare_ms_median=ms[len(ms) // 2] if ms else None,
        masks=len(masks), masks_clobbered=sum(1 for line in masks if line[1] == '0'))


def quad_draft_report(environ, users, log_text):
    """The summary and its problems (under 'problems'); None with the flag off ('0' or unset) and no sub-flag."""
    names = (QUAD_DRAFT_FLAG, QUAD_SDPA_FLAG, QUAD_CONV_FLAG, QUAD_AUDIT_FLAG)
    if environ.get(QUAD_DRAFT_FLAG) in (None, '', '0') and not any(environ.get(name) for name in names[1:]):
        return None
    summary = quad_draft_summary(log_text)
    problems = []
    flag = environ.get(QUAD_DRAFT_FLAG)
    if flag not in (None, '0', '1'):
        problems.append('%s: 0 or 1, not %r' % (QUAD_DRAFT_FLAG, flag))
    if flag != '1':
        if any(environ.get(name) for name in names[1:]):
            problems.append('%s / %s / %s without %s=1 do nothing' % (QUAD_SDPA_FLAG, QUAD_CONV_FLAG, QUAD_AUDIT_FLAG,
                                                                       QUAD_DRAFT_FLAG))
        summary['problems'] = problems
        return summary
    if users != 4:
        problems.append('%s serves four users only (users=%d)' % (QUAD_DRAFT_FLAG, users))
    for name in ('QWEN_FAST_PACKED_PROPOSAL', 'QWEN_FAST_PAIR_ROW_EXACT', 'QWEN_FAST_ROUND_B1',
                 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'):
        if environ.get(name) != '1':
            problems.append('%s needs %s=1' % (QUAD_DRAFT_FLAG, name))
    if environ.get('QWEN_FAST_PACKED_AUDIT') != '1':
        problems.append('%s needs QWEN_FAST_PACKED_AUDIT=1: its [QUAD-DRAFT] round lines and the four-user '
                        '[PACKED-SELECT] rounds they are counted against are logged only under it' % QUAD_DRAFT_FLAG)
    if summary['markers'] > 1:
        problems.append('%s: the marker is logged %d times, not once' % (QUAD_DRAFT_FLAG, summary['markers']))
    engaged = QUAD_ENGAGED_LINE.search(log_text)
    if engaged is not None:
        sdpa = environ.get(QUAD_SDPA_FLAG) or 'fold'
        wanted = dict(slots='0,1,2,3', heads='64/16' if sdpa == 'fold' else '32/8x2', rows='64', sdpa=sdpa,
                      conv=environ.get(QUAD_CONV_FLAG) or '110')
        summary['engaged'] = dict(zip(('slots', 'heads', 'rows', 'sdpa', 'conv'), engaged.groups()))
        differ = [name for name in wanted if summary['engaged'][name] != wanted[name]]
        if differ:
            problems.append('%s: the marker reads %s, not the %s the arm asked for' % (
                QUAD_DRAFT_FLAG, ' '.join('%s=%s' % (name, summary['engaged'][name]) for name in differ),
                ' '.join('%s=%s' % (name, wanted[name]) for name in differ)))
    elif summary['markers']:
        problems.append('%s: the marker does not name its slots, heads, rows, sdpa and conv' % QUAD_DRAFT_FLAG)
    if summary['disabled']:
        first = next(line for line in log_text.splitlines() if QUAD_DISABLED_MARKER in line)
        problems.append('%s: the quad gave up (%s)' % (QUAD_DRAFT_FLAG, first[-160:]))
    if summary['fallbacks']:
        first = next(line for line in log_text.splitlines() if QUAD_FALLBACK_MARKER in line)
        problems.append('%s: %d round(s) fell back to the pairs (first: %s)' % (QUAD_DRAFT_FLAG, summary['fallbacks'],
                                                                                first[-160:]))
    four = summary['four_user_rounds']
    if environ.get('QWEN_FAST_PACKED_AUDIT') == '1' and four:
        if not summary['rounds']:
            problems.append('%s: no quad round in %d four-user round(s)' % (QUAD_DRAFT_FLAG, four))
        elif four >= QUAD_FLOOR_ROUNDS and summary['rounds'] < QUAD_SHARE_FLOOR * four:
            problems.append('%s: %d quad rounds in %d four-user rounds (%.2f), under %.2f' % (
                QUAD_DRAFT_FLAG, summary['rounds'], four, summary['rounds'] / four, QUAD_SHARE_FLOOR))
    audit = environ.get(QUAD_AUDIT_FLAG)
    if audit:
        if audit != 'all' and not (audit.isdigit() and audit == str(int(audit)) and int(audit) >= 1):
            problems.append("%s: 'all' or a positive count, not %r" % (QUAD_AUDIT_FLAG, audit))
        elif environ.get('QWEN_FAST_PACKED_AUDIT') != '1':
            problems.append('%s needs QWEN_FAST_PACKED_AUDIT=1: its [QUAD-DRAFT] round lines count the audited '
                            'rounds' % QUAD_AUDIT_FLAG)
        else:
            expected = summary['rounds'] if audit == 'all' else min(int(audit), summary['rounds'])
            if summary['audits'] != expected:
                problems.append('%s: %d audit lines for %d audited quad rounds' % (QUAD_AUDIT_FLAG, summary['audits'],
                                                                                  expected))
            floor = QUAD_AUDIT_FLOOR if audit == 'all' else min(QUAD_AUDIT_FLOOR, int(audit))
            if summary['audits'] < floor:
                problems.append('%s: %d audited quad round(s), under the %d an audit arm needs' % (
                    QUAD_AUDIT_FLAG, summary['audits'], floor))
            if summary['audits_equal'] != summary['audits']:
                first = next(match.group(0) for match in QUAD_AUDIT_LINE.finditer(log_text) if match.group(2) != '1')
                problems.append('%s: %d audit line(s) not equal (first: %s)' % (
                    QUAD_AUDIT_FLAG, summary['audits'] - summary['audits_equal'], first[:200]))
    summary['problems'] = problems
    return summary


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
    required.update(quad_draft_markers(environ, users))
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
    slice or readahead adds the stage-4 factory's q-slice line as a fourth, and extent (0x20) K64j's
    runtime-extent line after it."""
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
    if 'extent' in served:
        markers.append(SDPA_EXTENT_MARKER)
    return markers


def sdpa_mode_names(environ):
    """QWEN_FAST_SDPA_MODES as pooled_attention_replay.sdpa_modes splits it (no validation here:
    a value that reader refuses fails the run on its own). Under QWEN_FAST_EXTENT_REPLAY=1 (S2) the
    extent readers add 'extent' to the environment's modes themselves (s2-design decision 5: the env
    keeps v235's tail,share,slice), so the promised modes line is 'modes=extent,share,slice,tail', the
    factory line flags=0x27 and K64j's F22 runtime-extent line is required too."""
    names = {name.strip() for name in (environ.get('QWEN_FAST_SDPA_MODES') or '').split(',') if name.strip()}
    if environ.get(EXTENT_REPLAY_FLAG) == '1':
        names.add('extent')
    return names


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
    # Q4: under 'quad_draft' only when one of its flags is set.
    quad = quad_draft_report(environ, users, log_text)
    if quad is not None:
        missing.extend(quad.pop('problems'))
        variable_user['quad_draft'] = quad
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
                 trace_region_bytes=1073741824, command=None):
    """The fast T16 + speculation recipe the four-user cycle bench serves
    (qwen-fp2u-image.yml), so the packed round under test is the one the 200
    tok/s/user work actually measures.

    --served-model-name is 'qwen-longctx', not this gate's own name: stream_once
    (longctx_cycle_bench.py, reused here exactly) hard-codes model='qwen-longctx' in
    its request payload, so any other served name 404s every stream in milliseconds
    (gate 1, run 35556533480 - a false negative that looked like readiness with zero
    decode rounds actually run). `command` (--server-argv platform: platform_argv) replaces
    the recipe; its streams then name its served name (stream_kwargs)."""
    command = command or engine_argv(port, users, context, trace_region_bytes)
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
    parser.add_argument('--server-argv', choices=SERVER_ARGV_MODES, default='gate',
                        help='gate (default): the recipe argv above. platform: the platform\'s argv '
                             '(platform_argv), which the C2 serving image\'s contract rewrites to its profile; '
                             'the served argv is read back from its [QWEN-C2] line')
    parser.add_argument('--served-model-name', default=PLATFORM_SERVED_NAME,
                        help='--server-argv platform: the served name the platform argv passes and the '
                             'streams request')
    parser.add_argument('--expect-profile', default=None,
                        help='--server-argv platform: the C2 profile the contract must report serving')
    parser.add_argument('--snapshot', default=MODEL,
                        help='the target snapshot the real-text tokenizer loads (the agent\'s container '
                             'mounts the hub at /models, not /models/hub)')
    parser.add_argument('--readiness-seconds', type=int, default=900,
                        help='how long the server may take to answer /health')
    parser.add_argument('--prompt-lengths', default=None,
                        help='real text only: comma-separated prompt lengths, one per stream (--users, or '
                             '--sequential-users), replacing --prompt-tokens for every user (a length per user, '
                             'real_text_prompts targets). Needs an image that serves any prompt length')
    parser.add_argument('--drops', default=None,
                        help='lifecycle, detail mode only, comma-separated: %s' % DROP_GRAMMAR)
    parser.add_argument('--user-max-tokens', default=None,
                        help='lifecycle, detail mode only: U:M gives user U its own max_tokens; comma-separated')
    parser.add_argument('--user-ignore-eos', default=None,
                        help='lifecycle, detail mode only: comma-separated users sent with ignore_eos=True')
    parser.add_argument('--alive-check', type=int, nargs='?', const=1, default=0, metavar='N',
                        help='after every stream, N more requests at once (the shortest prompt, 8 tokens each; '
                             'N=1 without a value): the engine survived what the streams did and gave every '
                             'seat back (report[\'alive_after\'])')
    parser.add_argument('--alive-seconds', type=int, default=ALIVE_SECONDS,
                        help='--alive-check: how long all N may take to answer')
    return parser


# --drops (the C2 serving gate's lifecycle arms). The pre-first-byte kinds (seconds, prefill, build)
# are fired by StreamWatch's thread, which shuts the stream's socket down; the others by the stream
# itself at a chunk. prefill and build read the server log, so they need the target user's prompt
# length to be unique (--prompt-lengths): that is how a ledger line names its request's user.
DROP_GRAMMAR = ('U:N closes user U\'s stream after N text chunks; U:@S if no byte arrived within S seconds; '
                'U:prefill+S S seconds after the server log shows its prefill began; U:build when the server log '
                'shows its prefill done and its engine build begun (the first byte only comes after the build); '
                'U:live=K at its first chunk while exactly K streams are live; U+V+W:N closes all of them together '
                'once each has N chunks (one wall time: a multiple drop in one round)')
LOG_DROP_KINDS = ('prefill', 'build')
ALIVE_SECONDS = 600


def parse_drop(value, part):
    """One --drops WHEN: ('chunks', N), ('seconds', S), ('prefill', S), ('build', 0) or ('live', K)."""
    try:
        if value.startswith('@'):
            drop = ('seconds', float(value[1:]))
        elif value.startswith('prefill+'):
            drop = ('prefill', float(value[len('prefill+'):]))
        elif value == 'build':
            drop = ('build', 0)
        elif value.startswith('live='):
            drop = ('live', int(value[len('live='):]))
        else:
            drop = ('chunks', int(value))
    except ValueError:
        raise ValueError('--drops: %r is not N, @S, prefill+S, build or live=K' % part)
    if drop[1] < 0 or (drop[1] == 0 and drop[0] not in ('prefill', 'build')):
        raise ValueError('--drops: %r must be positive' % part)
    return drop


def describe_drop(drop):
    kind, value = drop
    if kind == 'barrier':
        return 'barrier %s at %d chunks' % ('+'.join(str(user) for user in value[1]), value[0])
    if kind == 'prefill':
        return 'prefill+%s' % value
    if kind == 'build':
        return 'build'
    if kind == 'live':
        return 'live=%d' % value
    return '%s %s' % (kind, value)


def user_events(options, streams):
    """--drops, --user-max-tokens and --user-ignore-eos as {user: value}, or ValueError. A barrier
    (U+V+W:N) gives each member ('barrier', (N, (U, V, W)))."""
    def user_number(text, name, part):
        try:
            user = int(text)
        except ValueError:
            raise ValueError('%s: %r is not USER:VALUE' % (name, part))
        if not 0 <= user < streams:
            raise ValueError('%s: %r names no stream of 0..%d' % (name, part, streams - 1))
        return user

    def pairs(text, name):
        out = {}
        for part in [p.strip() for p in (text or '').split(',') if p.strip()]:
            user, _, value = part.partition(':')
            user = user_number(user, name, part)
            if user in out or not value:
                raise ValueError('%s: %r names no stream of 0..%d once' % (name, part, streams - 1))
            out[user] = value
        return out

    drops = {}
    for part in [p.strip() for p in (options.drops or '').split(',') if p.strip()]:
        who, _, value = part.partition(':')
        if not value:
            raise ValueError('--drops: %r is not USER:WHEN' % part)
        members = [user_number(text, '--drops', part) for text in who.split('+')]
        if len(set(members)) != len(members) or any(member in drops for member in members):
            raise ValueError('--drops: %r names a stream twice' % part)
        drop = parse_drop(value, part)
        if len(members) > 1:
            if drop[0] != 'chunks':
                raise ValueError('--drops: %r - a barrier takes a chunk count' % part)
            drop = ('barrier', (drop[1], tuple(sorted(members))))
        for member in members:
            drops[member] = drop
    budgets = {}
    for user, value in pairs(options.user_max_tokens, '--user-max-tokens').items():
        try:
            budgets[user] = int(value)
        except ValueError:
            raise ValueError('--user-max-tokens: %r is not an integer' % value)
        if budgets[user] < 1:
            raise ValueError('--user-max-tokens: %r must be positive' % value)
    ignore_eos = set()
    for part in [p.strip() for p in (options.user_ignore_eos or '').split(',') if p.strip()]:
        try:
            user = int(part)
        except ValueError:
            raise ValueError('--user-ignore-eos: %r is not a user' % part)
        if not 0 <= user < streams:
            raise ValueError('--user-ignore-eos: %r names no stream of 0..%d' % (part, streams - 1))
        ignore_eos.add(user)
    return dict(drops=drops, max_tokens=budgets, ignore_eos=sorted(ignore_eos))


def user_stream(options, index, kwargs, watch=None):
    """(max_tokens, stream_once keywords) for one user: the arm's, with that user's events applied.
    Every stream of a watched arm reports to the watch (the live count needs all of them)."""
    events = getattr(options, 'events', None) or {}
    kwargs = dict(kwargs)
    if watch is not None:
        kwargs['watch'] = watch
    if index in (events.get('ignore_eos') or ()):
        kwargs['ignore_eos'] = True
    return (events.get('max_tokens') or {}).get(index, options.max_tokens), kwargs


# The admission markers one request leaves in the server log, in order (serving_lifecycle,
# memory_ledger, serving_runtime):
#   prefill   "[PINDIAG] prefill gate: held='<id>'"                    its prefill begins
#             "[MEMLEDGER] phase=prefill point=before prompt=<L> ..."  the same request's prompt length
#   build     "[MEMLEDGER] phase=prefill point=after req=<id[-12:]>"   prefill done, the engine build begins
#   engine    "[PINDIAG] dram after engine <id[:48]>: ..."             the engine is built
#   admitted  "[PINDIAG] prefill gate: held=None was='<id>'"           it joined the decoding set
# The two ledger lines need QWEN_FAST_MEMORY_LEDGER=1, which the C2 serving image sets. The first token
# reaches the client only after the build (serving_lifecycle samples the prefill's token, then
# bridge_factory builds the engine, in the same step), so a drop during the build is timed on the log,
# never on a chunk count.
PREFILL_HELD = re.compile(r"\[PINDIAG\] prefill gate: held='([^']+)'")
PREFILL_RELEASED = re.compile(r"\[PINDIAG\] prefill gate: held=None was='([^']+)'")
LEDGER_PREFILL_BEFORE = re.compile(r'\[MEMLEDGER\] phase=prefill point=before prompt=([0-9]+) ')
LEDGER_PREFILL_AFTER = re.compile(r'\[MEMLEDGER\] phase=prefill point=after req=(\S+) ')
ENGINE_BUILT = re.compile(r'\[PINDIAG\] dram after engine (\S+): ')
WATCH_POLL_SECONDS = 0.05
BARRIER_WAIT_SECONDS = 600.0


class StreamWatch(object):
    """The lifecycle arms' one clock (time.perf_counter, the streams' own): when each stream started,
    got its first chunk and ended, what the server log said about each request's admission (tailed
    every WATCH_POLL_SECONDS by a thread, each marker stamped when it was read), and the --drops that
    act on them. stream_once calls begin/opened/chunk/end; the thread fires the pre-first-byte drops
    (seconds, prefill, build) by shutting the stream's socket down, and the stream records the drop
    (cancelled). report() puts every event against the markers: the phase each one hit (queued,
    prefill, build, decode) and how many streams were live then."""

    def __init__(self, drops, lengths, log_path, clock=time.perf_counter, poll=WATCH_POLL_SECONDS):
        self.drops = dict(drops)
        self.lengths = list(lengths or ())
        self.log_path = Path(log_path)
        self.clock, self.poll = clock, poll
        self.epoch = clock()
        self.state = threading.Condition()
        self.started, self.first, self.ended, self.chunks = {}, {}, {}, {}
        self.sockets, self.fired, self.cancels, self.missed = {}, {}, {}, {}
        self.delivered, self.undelivered = set(), {}
        self.requests, self.order = {}, []
        self.current = None
        self.ledger_seen = False
        self.offset, self.partial = 0, b''
        self.barriers = {}
        for kind, value in self.drops.values():
            if kind == 'barrier':
                self.barriers.setdefault(value[1], dict(chunks=value[0], members=list(value[1]), arrived={},
                                                        released=None, complete=None))
        self.thread = None
        self.stopping = False

    # --- the server log ------------------------------------------------------------------

    def read_log(self):
        try:
            with open(str(self.log_path), 'rb') as handle:
                handle.seek(self.offset)
                data = handle.read()
        except OSError:
            return
        if not data:
            return
        self.offset += len(data)
        lines = (self.partial + data).split(b'\n')
        self.partial = lines.pop()
        now = self.clock()
        with self.state:
            for line in lines:
                self.note(line.decode('utf-8', 'replace'), now)
            self.state.notify_all()

    def request(self, request_id):
        if request_id not in self.requests:
            self.requests[request_id] = {}
            self.order.append(request_id)
        return self.requests[request_id]

    def note(self, line, now):
        match = PREFILL_RELEASED.search(line)
        if match:
            self.request(match.group(1)).setdefault('admitted', now)
            return
        match = PREFILL_HELD.search(line)
        if match:
            self.current = match.group(1)
            self.request(self.current).setdefault('prefill', now)
            return
        match = LEDGER_PREFILL_BEFORE.search(line)
        if match:
            self.ledger_seen = True
            if self.current is not None:
                self.request(self.current).setdefault('prompt', int(match.group(1)))
            return
        match = LEDGER_PREFILL_AFTER.search(line)
        if match:
            self.ledger_seen = True
            for request_id in self.order:
                if request_id.endswith(match.group(1)):
                    self.requests[request_id].setdefault('build', now)
            return
        match = ENGINE_BUILT.search(line)
        if match:
            for request_id in self.order:
                if request_id[:48] == match.group(1):
                    self.requests[request_id].setdefault('engine', now)

    def user_by_length(self, length):
        return self.lengths.index(length) if length is not None and self.lengths.count(length) == 1 else None

    def request_of(self, user):
        """The request the server log ties to `user` before its first byte: the one whose ledger line
        carried the user's (unique) prompt length."""
        for request_id in self.order:
            if self.user_by_length(self.requests[request_id].get('prompt')) == user:
                return self.requests[request_id]
        return None

    # --- the streams (stream_once) ----------------------------------------------------------

    def begin(self, user, when):
        with self.state:
            self.started[user] = when

    def opened(self, user, sock):
        with self.state:
            self.sockets[user] = sock
            if sock is None:
                self.undelivered.setdefault(user, 'no socket under the response: a cancel cannot be delivered')
            if user in self.cancels:
                self._shutdown(user)

    def live(self, when):
        return sum(1 for user, first in self.first.items()
                   if first <= when and (user not in self.ended or self.ended[user] > when))

    def _fire(self, user, when, reason):
        self.fired[user] = dict(t=when, reason=reason, live=self.live(when))
        return reason

    def chunk(self, user, tokens, when):
        """After each text chunk: the reason to go away now, or None."""
        with self.state:
            if user not in self.first:
                self.first[user] = when
                self.state.notify_all()
            self.chunks[user] = tokens
            kind, value = self.drops.get(user, (None, None))
            if user in self.fired:
                return None
            if kind in ('seconds',) + LOG_DROP_KINDS:
                self.missed.setdefault(user, 'the first byte came before the drop was due')
                return None
            if kind == 'chunks' and tokens >= value:
                return self._fire(user, when, 'after %d chunks' % tokens)
            if kind == 'live' and self.live(when) == value:
                return self._fire(user, when, 'at %d live streams' % value)
            if kind == 'barrier' and tokens >= value[0]:
                barrier = self.barriers[value[1]]
                barrier['arrived'].setdefault(user, when)
                self.state.notify_all()
                deadline = when + BARRIER_WAIT_SECONDS
                while barrier['released'] is None and not self._barrier_ready(barrier) and self.clock() < deadline:
                    self.state.wait(self.poll)
                if barrier['released'] is None:
                    barrier['released'] = self.clock()
                    barrier['complete'] = all(member in barrier['arrived'] for member in barrier['members'])
                    self.state.notify_all()
                return self._fire(user, self.clock(), 'barrier %s at %d chunks' % (
                    '+'.join(str(member) for member in barrier['members']), value[0]))
            return None

    def _barrier_ready(self, barrier):
        return all(member in barrier['arrived'] or member in self.ended for member in barrier['members'])

    def end(self, user, when):
        with self.state:
            self.ended.setdefault(user, when)
            self.state.notify_all()

    def cancelled(self, user):
        """The reason the watch cancelled `user`'s stream - only once the socket was really shut down:
        a cancel that could not be delivered leaves the stream running, and it is recorded as such."""
        with self.state:
            return self.cancels.get(user) if user in self.delivered else None

    # --- the pre-first-byte drops --------------------------------------------------------------

    def _shutdown(self, user):
        sock = self.sockets.get(user)
        if sock is None:
            return
        try:
            import socket
            sock.shutdown(socket.SHUT_RDWR)
            self.delivered.add(user)
        except (OSError, ValueError, AttributeError) as error:
            self.undelivered[user] = '%s: %s' % (type(error).__name__, error)

    def due(self, user, kind, value, now):
        if kind == 'seconds':
            return now - self.started[user] >= value and 'no byte within %s s' % value
        marks = self.request_of(user)
        if not marks:
            return None
        if kind == 'prefill' and marks.get('prefill') is not None and now - marks['prefill'] >= value:
            return '%s s into its prefill' % value
        if kind == 'build' and marks.get('build') is not None:
            return 'its engine build began'
        return None

    def check(self, now):
        with self.state:
            for user, (kind, value) in sorted(self.drops.items()):
                if (kind not in ('seconds',) + LOG_DROP_KINDS or user in self.fired or user in self.first
                        or user in self.ended or user not in self.started):
                    continue
                reason = self.due(user, kind, value, now)
                if reason:
                    self._fire(user, now, reason)
                    self.cancels[user] = reason
                    self._shutdown(user)

    def run(self):
        while not self.stopping:
            self.read_log()
            self.check(self.clock())
            time.sleep(self.poll)

    def start(self):
        self.thread = threading.Thread(target=self.run, name='stream-watch', daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stopping = True
        if self.thread is not None:
            self.thread.join(5.0)
        self.read_log()

    # --- the record -----------------------------------------------------------------------------

    def seconds(self, when):
        return None if when is None else round(when - self.epoch, 3)

    def phase_at(self, user, when, marks):
        first = self.first.get(user)
        if first is not None and when >= first:
            return 'decode'
        if marks.get('build') is not None and when >= marks['build']:
            return 'build'
        if marks.get('prefill') is not None and when >= marks['prefill']:
            return 'prefill' if self.ledger_seen else 'prefill-or-build'
        return 'queued'

    def report(self, results=()):
        """Every drop against the markers (fired or not, when, the phase, the live count), every
        user's timeline, the barriers. A request is tied to its user by the stream's id (the engine id
        extends it) or, before a first byte, by its ledger prompt length."""
        with self.state:
            stream_ids = {user: (entry or {}).get('request_id') for user, entry in enumerate(results or ())}
            owners = {}
            for request_id in self.order:
                owner = next((user for user, stream_id in stream_ids.items()
                              if stream_id and request_id.startswith(stream_id)), None)
                owners[request_id] = owner if owner is not None else self.user_by_length(
                    self.requests[request_id].get('prompt'))
            users = sorted(set(self.started) | set(range(len(results or ()))))
            timeline, marks_of = {}, {}
            for user in users:
                ids = [request_id for request_id in self.order if owners[request_id] == user]
                marks = self.requests[ids[0]] if len(ids) == 1 else {}
                marks_of[user] = marks
                timeline[str(user)] = dict(
                    request_ids=ids, started_s=self.seconds(self.started.get(user)),
                    prefill_s=self.seconds(marks.get('prefill')), build_s=self.seconds(marks.get('build')),
                    engine_s=self.seconds(marks.get('engine')), admitted_s=self.seconds(marks.get('admitted')),
                    first_chunk_s=self.seconds(self.first.get(user)), ended_s=self.seconds(self.ended.get(user)),
                    chunks=self.chunks.get(user, 0))
            events = {}
            for user, drop in sorted(self.drops.items()):
                entry = dict(kind=drop[0], spec=describe_drop(drop), fired=user in self.fired)
                fired = self.fired.get(user)
                if fired:
                    entry.update(fired_s=self.seconds(fired['t']), reason=fired['reason'], live=fired['live'],
                                 phase=self.phase_at(user, fired['t'], marks_of.get(user) or {}))
                    if user in self.cancels:
                        # A cancel acts only once the socket is shut; one never delivered left the stream running.
                        entry['delivered'] = user in self.delivered
                        if not entry['delivered']:
                            entry['undelivered'] = self.undelivered.get(user) or 'the stream never opened'
                else:
                    entry['missed'] = self.missed.get(user) or 'never due before the stream ended'
                events[str(user)] = entry
            barriers = [dict(members=b['members'], chunks=b['chunks'], complete=b['complete'],
                             released_s=self.seconds(b['released']),
                             arrived_s={str(user): self.seconds(when) for user, when in sorted(b['arrived'].items())})
                        for _, b in sorted(self.barriers.items())]
            return dict(ledger_markers=self.ledger_seen, requests_seen=len(self.order), events=events,
                        users=timeline, barriers=barriers)


def alive_check(port, prompt, count, stream_timeout, deadline_seconds, kwargs, clock=time.perf_counter):
    """`count` requests at once (the engine's seats), `prompt` for 8 tokens each, after every stream
    has ended: each must answer within `deadline_seconds` of the first's start - a seat a drop or a
    cancel never gave back leaves one queued past it. -> (entries, every one answered)."""
    results = [None] * count
    threads = [threading.Thread(target=stream_once, args=(port, prompt, 8, results, index, stream_timeout),
                                kwargs=dict(kwargs), daemon=True) for index in range(count)]
    started = clock()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(0.0, deadline_seconds - (clock() - started)))
    entries = []
    for index in range(count):
        entry = results[index]
        if threads[index].is_alive() or entry is None:
            entry = dict(error='no answer within %d s (a seat not given back?)' % deadline_seconds)
        entries.append(entry)
    return entries, all(entry and not entry.get('error') and entry.get('text') for entry in entries)


# The ledger's allocator reading before each prefill (memory_ledger, 'phase=prefill point=before'),
# one line per chip: with nothing else resident it is the idle allocation, which a lifecycle arm
# compares before its first request and after every stream has ended (the ledger stays flat).
LEDGER_IDLE_CHIP = re.compile(r'\[MEMLEDGER\] phase=prefill point=before prompt=([0-9]+) chip([0-9]+) '
                              r'allocated=([0-9.]+)GB')


def ledger_readings(log_text):
    """[{prompt, chips: {'0': allocated GB, ...}}], one per prefill, in log order."""
    readings = []
    for match in LEDGER_IDLE_CHIP.finditer(log_text):
        prompt, chip, allocated = int(match.group(1)), match.group(2), float(match.group(3))
        if not readings or chip in readings[-1]['chips'] or readings[-1]['prompt'] != prompt:
            readings.append(dict(prompt=prompt, chips={}))
        readings[-1]['chips'][chip] = allocated
    return readings


def ledger_report(log_text, alive_index=None):
    """The memory ledger read independently of flag_marker_report (which a mixed-length arm may not
    survive): the P7 residual check's status (attach time), the idle reading before the first prefill,
    the one before the first prefill after `alive_index` readings (the alive check's first request,
    after every stream ended), and their difference per chip in GB (idle_drift_gb)."""
    residual = LEDGER_RESIDUAL.search(log_text)
    readings = ledger_readings(log_text)
    first = readings[0] if readings else None
    after = readings[alive_index] if alive_index is not None and alive_index < len(readings) else None
    drift = None
    if first is not None and after is not None:
        drift = {chip: round(after['chips'][chip] - first['chips'][chip], 3)
                 for chip in sorted(first['chips']) if chip in after['chips']}
    return dict(residual_status=residual.group(1) if residual else None, readings=len(readings),
                first_idle=first, idle_after_streams=after, idle_drift_gb=drift)


# S2, C2-PACKED-ANY (s2-design.md W11). QWEN_FAST_EXTENT_REPLAY=1 - set only by the c2-packed profiles, never
# the image ENV - builds the block's extent readers (extent_attention_replay): one K64j 0x27 SDPA program
# serving every live user at its own 256-key family. The gate-only knobs are never in a profile or the image
# ENV; c2_serving_gate sets them per arm with -e, and qwen_configuration records that they reached this
# process: QWEN_FAST_EXTENT_AUDIT=1 reads every packed round's words, cur_pos, masks and tables back and
# compares them with the host (one [EXTENT-AUDIT] line per packed round, MISMATCH on a difference; W3);
# QWEN_FAST_PACKED_CAPTURE_POSITION=<p> captures the block at p instead of C - 256 (G3b; W3 logs the
# override); QWEN_FAST_GATE_FORCE_CAP=8 caps every packed commit at 8 rows (the M4 forced-cap pair; W4's
# [PACKED] lines carry cap=). Every line format below is its producer's own, as the producer writes it (each
# constant names its source; test_c2_serving_gate_s2.ProducerContractTests renders the producer's lines and
# parses them here whenever the checkout carries it), and key=value lines are read by field name, so a field a
# producer adds never hides a line. s2_report reads all of it into report['s2'] and puts what fails an arm into
# flag_markers['missing'], so gate_passed carries it:
#   flag on:  the admission line (W7), the engaged line (W1), K64j's F22 line, and a 'packed extent round'
#             line (W3, logged inside verify: the executed path, memory graft-mounted-is-not-graft-executed)
#             for every packed round's [PACKED] lines; with the audit, no MISMATCH, every word and cur_pos
#             read back, and an audit line for every packed extent round; never a packed round below 128;
#   flag off: none of S2_MARKERS (this is not the profile's path);
#   either:   no cap-refused (the block backstop, W3/W4) or replay-deadline line; a capture knob that the
#             block logged taking.
# Everything else - holds, refusals, narrowing, releases, before-points, trace-region readings, per-user
# paths, cap events and boundary crossings - is recorded for c2_serving_gate's plan verdicts.
EXTENT_REPLAY_FLAG = 'QWEN_FAST_EXTENT_REPLAY'
EXTENT_AUDIT_FLAG = 'QWEN_FAST_EXTENT_AUDIT'
CAPTURE_POSITION_FLAG = 'QWEN_FAST_PACKED_CAPTURE_POSITION'
FORCE_CAP_FLAG = 'QWEN_FAST_GATE_FORCE_CAP'
S2_GATE_KNOBS = (EXTENT_AUDIT_FLAG, CAPTURE_POSITION_FLAG, FORCE_CAP_FLAG)
S2_ADMISSION_MARKER = '[PINDIAG] packed-any admission'   # W7, packed_any_admission.admit
S2_ENGAGED_MARKER = '[PINDIAG] extent replay engaged'    # W1, extent_attention_replay.ENGAGED_MARKER
S2_ROUND_MARKER = '[PINDIAG] packed extent round'        # W3, packed_verifier.EXTENT_ROUND_MARKER
# W3, packed_verifier.note_extent_round: 'round=R live=L families=[seg:E,...] idle=[seg,...] capped=[seg:limit,...]'
# - each live segment with its family E (extent_rounds reads the E after the colon; a bare E is read too).
S2_ROUND_LINE = re.compile(r'\[PINDIAG\] packed extent round round=([0-9]+) live=([0-9]+) families=\[([^\]\n]*)\]'
                           r'(?: idle=\[([^\]\n]*)\])?(?: capped=\[([^\]\n]*)\])?')
EXTENT_AUDIT_MARKER = '[EXTENT-AUDIT] '
# W3, packed_verifier.audit_extent: 'round=R segments=S words_ok=W cur_pos_ok=C mask_ok=M tables_ok=T rotated=I
# ms=X', read by field name (W3's rotated= sits between tables_ok and ms). The MISMATCH line is its own.
EXTENT_AUDIT_LINE = re.compile(r'\[EXTENT-AUDIT\] (round=[0-9]+[^\n]*)')
EXTENT_AUDIT_REQUIRED = ('round', 'segments', 'words_ok', 'cur_pos_ok')
EXTENT_AUDIT_MISMATCH = '[EXTENT-AUDIT] MISMATCH'
LINE_FIELD = re.compile(r'([a-z_]+)=(\S+)')
CAP_REFUSED_MARKER = '[PINDIAG] packed extent cap refused'       # W3 commit_user backstop
DEADLINE_MARKER = '[PINDIAG] replay deadline exceeded'           # W3 replay watchdog
CAPTURE_OVERRIDE_LINE = re.compile(r'\[PINDIAG\] packed capture position override=([0-9]+)')
# W6b, serving_prefill_admission: DRAM_HOLD_LINE once per held request with the decodes running when it was
# held; 'released' once it fits, 'lifted' when no decode is left to wait for (admitted anyway, the bridge
# backstop decides), 'unavailable' when the predicate read no DRAM (it holds nothing). The predicate is asked
# whenever a prompt waits - with every seat decoding too, where a hold changes nothing (churn has more users than
# seats) - so only a hold with a seat free is a hold that failed the fit: c2_serving_gate judges the decodes
# against the profile's seats. A hold that began with every seat decoding logs no new line when a seat frees and
# it still does not fit; the wrapper's decision line (once per distinct state) shows it: no partials, the gate
# not held, allowed=0 and the queues hidden is the DRAM hold's state and no other's (admission()).
DRAM_HOLD_MARKER = '[PINDIAG] dram hold prompt='
DRAM_HOLD_LINE = re.compile(r'\[PINDIAG\] dram hold prompt=(\S+) largest_free=(\S+) need=(\S+) request=(\S+) '
                            r'decodes=([0-9]+)')
DRAM_RELEASED_MARKER = '[PINDIAG] dram hold released '
DRAM_LIFTED_MARKER = '[PINDIAG] dram hold lifted '
DRAM_UNAVAILABLE_MARKER = '[PINDIAG] dram hold unavailable '
DRAM_HELD_STATE = re.compile(r'\[PINDIAG\] one fresh prefill per step: partials=0 decodes=([0-9]+) gate_held=False '
                             r'allowed=0 hidden=True')
QUARANTINED_MARKER = '[PINDIAG] request quarantined:'            # D2: a RequestRefused (W6b's backstop included)
# W6c, dflash_packed_proposal_coordinator.RELEASED_LINE: 'quad=0|1 pairs=[[a, b], ...]', logged at every detach
# once a coordinator exists (serving_worker_hook.release_dead_proposals), from serving_lifecycle before the step
# reaches the hook - so just ahead of that step's '[PHASE] execute ... finished=[ids]' line.
RELEASED_LINE = re.compile(r'\[PACKED-PROPOSE\] released quad=([0-9]+) pairs=(\[[^\n]*\]|[0-9]+)')
QUAD_BUILT_LINE = re.compile(r'\[QUAD-DRAFT\] round=[0-9]+ built=1 ')
# A quad round is QUAD_ROUND_LINE (above: quad_draft.ROUND_LINE, logged every quad round).
# serving_worker_hook (QWEN_FAST_PHASE_LOG=1, the image's): finished= is the step's sorted finished request ids.
PHASE_FINISHED = re.compile(r'\[PHASE\] execute total=\S+ new=\S+ cached=\S+ spec=\S+ finished=\[([^\]\n]*)\]')
NARROWED_MARKER = '[PINDIAG] packed survivor narrowed'           # W5a, serving_packed_step.NARROWED_MARKER
NARROWING_REFUSED_MARKER = '[PINDIAG] packed survivor narrowing refused'
ABORTED_LINE = re.compile(r'\[PINDIAG\] packed refused round aborted ([0-9]+)/([0-9]+) FINISHED_ABORTED')   # W5b
# W6a, the any-request engine line (serving_request_factory): the ladder is a Python tuple through loguru's {} -
# '(2048,)' under the extent flag, '(256, 512, 1024, 2048)' for a short prompt without it - or 'unavailable (..)'.
PROPOSAL_LADDER = re.compile(r'proposal ladder (\([0-9, ]*\)|\[[0-9, ]*\]|unavailable[^\n]*)')
# The ledger: its phase point ahead of each prefill (serving_runtime, record('prefill', point='before prompt=N'))
# and W6d's before/after points (memory_ledger.MemoryLedger.before/after: 'before op=<op>[ point=<p>] chip<n>
# largest_free=..MB free=..GB estimate=..MB margin=..MB floor=..MB' then 'trace_used=..MB trace_largest_free=..MB'
# or 'trace=unavailable', a margin below zero included; 'dram unavailable (..)' or 'error=..' when it read nothing).
# A message past the ledger's LINE_BUDGET continues on '[MEMLEDGER] ...' lines, joined back on here.
LEDGER_CONTINUATION = '[MEMLEDGER] ...'
LEDGER_BEFORE = re.compile(r'\[MEMLEDGER\] phase=(\S+) point=before ?([^\n]*?) chip([0-9]+) allocated=([0-9.]+)GB '
                           r'free=([0-9.]+)GB largest_free=([0-9.]+)MB')
LEDGER_BEFORE_UNREAD = re.compile(r'\[MEMLEDGER\] phase=\S+ point=before [^\n]*?dram unavailable')
LEDGER_OP_POINT = re.compile(r'\[MEMLEDGER\] (before|after) op=(\S+)(?: point=(\S+))? chip([0-9]+) ([^\n]*)')
LEDGER_OP_UNREAD = re.compile(r'\[MEMLEDGER\] (before|after) op=(\S+)(?: point=(\S+))? '
                              r'(dram unavailable[^\n]*|error=[^\n]*)')
LEDGER_SIZE = re.compile(r'(?<![a-z_])(largest_free|free|estimate|margin|floor|cost|trace_used|trace_largest_free)='
                         r'(-?[0-9.]+)(MB|GB)')
S2_MARKERS = (S2_ADMISSION_MARKER, S2_ENGAGED_MARKER, S2_ROUND_MARKER, EXTENT_AUDIT_MARKER, CAP_REFUSED_MARKER,
              SDPA_EXTENT_MARKER)
S2_MIN_LIVE_START = 128   # extent_attention_replay.MIN_LIVE_START: no packed round below it
# The prefill point's operation (s2-design 3.2 item 2, serving_prefill_admission's one set of defaults, UNVERIFIED:
# M8 and M11 calibrate them): the transient of a prompt of at least PREFILL_TRANSIENT_FROM tokens, 0 below. W6d's
# points carry their own estimate (the engine build's peak, the capture's estimate).
PREFILL_TRANSIENT_GB, PREFILL_TRANSIENT_FROM = 0.30, 2048
EXTENT_KEYS = 256


def extent_of(position):
    """The 256-key family a row at `position` reads (extent_attention_replay.extent)."""
    return (position // EXTENT_KEYS + 1) * EXTENT_KEYS


def s2_relevant(environ, log_text):
    """Whether an arm gets report['s2']: the flag or a gate knob is set, or the log carries an S2 marker."""
    return (environ.get(EXTENT_REPLAY_FLAG) == '1' or any(environ.get(name) for name in S2_GATE_KNOBS)
            or any(marker in log_text for marker in S2_MARKERS))


def round_families(text):
    """The families of a round line's families=[...]: W3 writes segment:E per live segment; a bare E is read too."""
    families = []
    for item in (text or '').split(','):
        value = item.strip().rsplit(':', 1)[-1].strip()
        if value.isdigit():
            families.append(int(value))
    return families


def extent_round_entries(log_text):
    rounds = []
    for match in S2_ROUND_LINE.finditer(log_text):
        idle = [value for value in re.findall(r'[0-9]+', match.group(4) or '')]
        capped = [(int(segment), int(limit))
                  for segment, limit in re.findall(r'([0-9]+):([0-9]+)', match.group(5) or '')]
        rounds.append(dict(round=int(match.group(1)), live=int(match.group(2)), families=round_families(match.group(3)),
                           idle=len(idle), capped=capped))
    return rounds


def extent_rounds(log_text):
    """The 'packed extent round' lines: how many, by live count, the most distinct live families in one
    round, the rounds with two or more, the capped segments and the rounds with idle segments."""
    rounds = extent_round_entries(log_text)
    by_live = {}
    for entry in rounds:
        by_live[str(entry['live'])] = by_live.get(str(entry['live']), 0) + 1
    return dict(count=len(rounds), by_live=by_live,
                max_families=max((len(set(entry['families'])) for entry in rounds), default=0),
                multi_family_rounds=sum(1 for entry in rounds if len(set(entry['families'])) >= 2),
                capped_rounds=sum(1 for entry in rounds if entry['capped']),
                capped_segments=sum(len(entry['capped']) for entry in rounds),
                idle_rounds=sum(1 for entry in rounds if entry['idle']),
                families_seen=sorted(set(value for entry in rounds for value in entry['families']))[:64])


def extent_audit_entries(log_text):
    """(the audit lines, each {round, segments, words_ok, cur_pos_ok[, ms]}, and the malformed ones)."""
    entries, malformed = [], []
    for match in EXTENT_AUDIT_LINE.finditer(log_text):
        fields = dict(LINE_FIELD.findall(match.group(1)))
        try:
            entry = dict((name, int(fields[name])) for name in EXTENT_AUDIT_REQUIRED)
        except (KeyError, ValueError):
            malformed.append(match.group(0).strip()[:160])
            continue
        try:
            entry['ms'] = float(fields['ms'])
        except (KeyError, ValueError):
            entry['ms'] = None
        entries.append(entry)
    return entries, malformed


def extent_audit(log_text):
    """The extent audit's lines: how many, the malformed ones, the mismatch lines, the rounds that did not
    read back every segment's word and cur_pos, and the median ms it added."""
    lines, malformed = extent_audit_entries(log_text)
    mismatches = [line.strip()[:240] for line in log_text.splitlines() if EXTENT_AUDIT_MISMATCH in line]
    incomplete = [entry['round'] for entry in lines
                  if entry['words_ok'] != entry['segments'] or entry['cur_pos_ok'] != entry['segments']]
    times = [entry['ms'] for entry in lines if entry['ms'] is not None]
    return dict(lines=len(lines), malformed=len(malformed), malformed_lines=malformed[:2], mismatches=len(mismatches),
                mismatch_lines=mismatches[:4], incomplete_rounds=incomplete[:8], incomplete=len(incomplete),
                median_ms=round(statistics.median(times), 3) if times else None)


def refused_rounds_report(log_text):
    """refuse_round's lines against W5b's aborted lines: each refused round must end all of its requests
    (index 1..count) as FINISHED_ABORTED through the quarantine (s2-design W5b; M6)."""
    refused = sum(1 for line in log_text.splitlines() if REFUSED_ROUND_MARKER in line and '[PACKED]' in line)
    groups, current = [], None
    for match in ABORTED_LINE.finditer(log_text):
        index, count = int(match.group(1)), int(match.group(2))
        if index == 1 or current is None or current['count'] != count or current['seen'] + 1 != index:
            current = dict(count=count, seen=0)
            groups.append(current)
        current['seen'] = index
    complete = sum(1 for group in groups if group['seen'] == group['count'])
    return dict(refused=refused, aborted_lines=sum(group['seen'] for group in groups), aborted_groups=len(groups),
                complete_groups=complete, accounted=refused == 0 or (len(groups) == refused and complete == refused))


def dram_holds(log_text):
    """W6b's hold lines: each hold's decodes (a seat was free when fewer than the seats decode), the decodes of
    every held state the wrapper logged, and the released, lifted and unavailable lines."""
    lines = log_text.splitlines()
    decodes = [int(match.group(5)) for match in DRAM_HOLD_LINE.finditer(log_text)]

    def picked(marker):
        return [line.strip()[:240] for line in lines if marker in line]

    held, lifted, unavailable = picked(DRAM_HOLD_MARKER), picked(DRAM_LIFTED_MARKER), picked(DRAM_UNAVAILABLE_MARKER)
    return dict(holds=len(decodes), hold_decodes=decodes[:64], lines=held[:4],
                held_states=sorted(set(int(value) for value in DRAM_HELD_STATE.findall(log_text))),
                released=len(picked(DRAM_RELEASED_MARKER)), lifted=len(lifted), lifted_lines=lifted[:2],
                unavailable=len(unavailable), unavailable_lines=unavailable[:2])


def pair_count(text):
    """The pairs a release line names: [[a, b], ...] (W6c) or a bare count."""
    return int(text) if text.isdigit() else len(re.findall(r'\[[0-9, ]+\]', text))


def finished_ids(text):
    """The request ids of a [PHASE] line's finished=[...] (a Python list through loguru's {})."""
    quoted = re.findall(r"'([^']*)'", text)
    return quoted if quoted else [part.strip() for part in text.split(',') if part.strip()]


def proposal_releases(log_text):
    """W6c's release lines against the departures they must answer (M11: a released quad at every quad
    member's detach). In log order a quad is formed from a quad round line (quad_draft.ROUND_LINE, every quad
    round) until a quad=1 release frees it; a departure is a finished request in a step's [PHASE] execute line,
    and the detach's release line comes just ahead of that line (RELEASED_LINE). A step with a departure while a
    quad was formed (as the step began) must carry a quad=1 release, else it is 'unreleased'."""
    formed, at_start, step = False, None, []
    lines = quads = pairs = quad_rounds = steps = departures = quad_departures = 0
    unreleased = []
    for line in log_text.splitlines():
        if QUAD_ROUND_LINE.search(line):
            formed = True
            quad_rounds += 1
            continue
        match = RELEASED_LINE.search(line)
        if match:
            quad = int(match.group(1))
            lines += 1
            quads += quad
            pairs += pair_count(match.group(2))
            if at_start is None:
                at_start = formed
            step.append(quad)
            if quad:
                formed = False
            continue
        match = PHASE_FINISHED.search(line)
        if match:
            finished = finished_ids(match.group(1))
            if finished:
                steps += 1
                departures += len(finished)
                began = formed if at_start is None else at_start
                if began:
                    quad_departures += 1
                    if not any(step):
                        unreleased.append('departure %d (%s): %s' % (
                            steps, ','.join(value[-12:] for value in finished),
                            'released quad=0' if step else 'no release line'))
            at_start, step = None, []
    return dict(lines=lines, quad=quads, pairs=pairs, quad_rounds=quad_rounds, departures=departures,
                departure_steps=steps, quad_departures=quad_departures, unreleased=len(unreleased),
                unreleased_steps=unreleased[:4])


def ladder_of(text):
    """A proposal ladder as a list of its buckets ('(2048,)' -> [2048]), or the text when it names none."""
    if text.startswith('(') or text.startswith('['):
        return [int(value) for value in re.findall(r'[0-9]+', text)]
    return text.strip()[:80]


def memledger_messages(log_text):
    """Every [MEMLEDGER] message from its marker on, each continuation line ('[MEMLEDGER] ...', the ledger's
    LINE_BUDGET split) joined back on exactly as it was cut."""
    messages, joinable = [], False
    for line in log_text.splitlines():
        if LEDGER_CONTINUATION in line:
            if joinable:
                messages[-1] += line.split(LEDGER_CONTINUATION, 1)[1].rstrip('\r')
            continue
        index = line.find('[MEMLEDGER] ')
        joinable = index >= 0
        if joinable:
            messages.append(line[index:].rstrip('\r'))
    return messages


def ledger_sizes(text):
    """{field: GB} of a ledger message's MB/GB fields (largest_free, free, estimate, margin, floor, ...)."""
    return dict((name, float(value) / (1000.0 if unit == 'MB' else 1.0))
                for name, value, unit in LEDGER_SIZE.findall(text))


def before_points(log_text):
    """Every 'before' ledger point - the prefill's phase point ('before prompt=', its estimate the prefill
    transient) and W6d's ahead of each engine build, pair and quad capture and single-user rebuild (its own
    estimate and margin) - per chip: its largest free block, the operation's estimate and the margin = largest
    free - estimate (W6d's own, a negative one included). floor_gb is the smallest margin (s2-design 3.3: G5
    judges it); a point that read no DRAM ('dram unavailable', 'error=') is 'unread', never a pass."""
    points, unread = [], []
    for message in memledger_messages(log_text):
        match = LEDGER_OP_POINT.match(message)
        if match:
            if match.group(1) != 'before':
                continue
            sizes = ledger_sizes(match.group(5))
            largest, estimate, margin = sizes.get('largest_free'), sizes.get('estimate'), sizes.get('margin')
            if margin is None and largest is not None and estimate is not None:
                margin = largest - estimate
            points.append(dict(op=match.group(2), detail=(match.group(3) or '')[:80], chip=int(match.group(4)),
                               largest_free_gb=None if largest is None else round(largest, 4),
                               estimate_gb=None if estimate is None else round(estimate, 4),
                               margin_gb=None if margin is None else round(margin, 4),
                               logged_floor_gb=sizes.get('floor')))
            continue
        match = LEDGER_OP_UNREAD.match(message)
        if match:
            if match.group(1) == 'before':
                unread.append(message.strip()[:160])
            continue
        if LEDGER_BEFORE_UNREAD.match(message):
            unread.append(message.strip()[:160])
            continue
        match = LEDGER_BEFORE.match(message)
        if match:
            detail, largest = match.group(2).strip(), float(match.group(6)) / 1000.0
            prompt = re.search(r'prompt=([0-9]+)', detail)
            estimate = (PREFILL_TRANSIENT_GB if prompt and int(prompt.group(1)) >= PREFILL_TRANSIENT_FROM else 0.0) \
                if match.group(1) == 'prefill' else None
            points.append(dict(op=match.group(1), detail=detail[:80], chip=int(match.group(3)),
                               largest_free_gb=round(largest, 4), estimate_gb=estimate,
                               margin_gb=None if estimate is None else round(largest - estimate, 4),
                               logged_floor_gb=None))
    judged = [point for point in points if point['margin_gb'] is not None]
    floor = min(judged, key=lambda point: point['margin_gb']) if judged else None
    by_op = {}
    for point in points:
        by_op[point['op']] = by_op.get(point['op'], 0) + 1
    logged = [point['logged_floor_gb'] for point in points if point['logged_floor_gb'] is not None]
    return dict(points=len(points), judged=len(judged), by_op=by_op, ops=sorted(by_op),
                unestimated=sorted(set(point['op'] for point in points if point['margin_gb'] is None)),
                unread=len(unread), unread_lines=unread[:4], floor_gb=floor['margin_gb'] if floor else None,
                floor_point=floor, logged_floor_gb=min(logged) if logged else None)


def trace_region(log_text):
    """The trace region's readings on W6d's before/after points (s2-design Q18, B9): how many, how many said
    'trace=unavailable' (this ttnn has no TRACE view), the most used and the smallest largest free block."""
    used, largest, unavailable, lines = [], [], 0, []
    for message in memledger_messages(log_text):
        match = LEDGER_OP_POINT.match(message)
        if not match:
            continue
        if 'trace=unavailable' in message:
            unavailable += 1
            continue
        sizes = ledger_sizes(match.group(5))
        if 'trace_used' not in sizes:
            continue
        used.append(sizes['trace_used'])
        if 'trace_largest_free' in sizes:
            largest.append(sizes['trace_largest_free'])
        if len(lines) < 16:
            lines.append(message.strip()[:240])
    return dict(readings=len(used), unavailable=unavailable, max_used_gb=round(max(used), 4) if used else None,
                min_largest_free_gb=round(min(largest), 4) if largest else None, lines=lines)


def user_paths(log_text, streams, prompt_lengths):
    """acceptance_report.path_records, with per user the packed rounds, the cap events (a packed commit
    capped below its 16 rows: W4's cap=), the 256-key boundaries its rounds crossed, and any packed round
    below S2_MIN_LIVE_START; the records encoded as the report carries them."""
    from acceptance_report import encode_paths, path_records
    found = path_records(log_text, streams, prompt_lengths)
    users, low = {}, []
    for user, records in sorted(found['users'].items()):
        caps = [record['cap'] for record in records if record['path'] == 'P' and record['cap'] is not None]
        extents = [extent_of(record['position']) for record in records if record['position'] is not None]
        crossings = sum(1 for before, after in zip(extents, extents[1:]) if after != before)
        low.extend('user %d round %d at %d' % (user, record['round'], record['position']) for record in records
                   if record['path'] == 'P' and record['position'] is not None and record['position'] < S2_MIN_LIVE_START)
        users[str(user)] = dict(rounds=len(records), packed=sum(1 for record in records if record['path'] == 'P'),
                                sequential=sum(1 for record in records if record['path'] == 'S'),
                                cap_fields=len(caps), cap_events=sum(1 for cap in caps if cap < 16),
                                max_cap=max(caps) if caps else None, boundaries=crossings,
                                encoded=encode_paths(records))
    return dict(users=users, unattributed=len(found['unattributed']), malformed=found['malformed'],
                position_checks=found['position_checks'], position_mismatches=found['position_mismatches'],
                position_mismatch_count=found['position_mismatch_count'], packed_below_floor=low[:8])


def s2_report(environ, log_text, streams=None, prompt_lengths=None):
    """What an S2-relevant arm's server log says (the section comment above), with the arm-failing problems
    under 'problems'."""
    on = environ.get(EXTENT_REPLAY_FLAG) == '1'
    audit_on = environ.get(EXTENT_AUDIT_FLAG) == '1'
    capture = (environ.get(CAPTURE_POSITION_FLAG) or '').strip() or None
    lines = log_text.splitlines()
    rounds = extent_rounds(log_text)
    audit = extent_audit(log_text)
    packed_lines = sum(1 for line in lines if '[PACKED] request=' in line)
    other = dict(prestage=h1a_summary(log_text)['audit_mismatches'],
                 fused=h1b_summary(log_text)['audit_mismatches'] + log_text.count(FUSED_AUDIT_MISMATCH_MARKER),
                 pair_mask=(pair_mask_audit_summary(log_text) or {}).get('clobbered', 0))
    markers = dict(admission=S2_ADMISSION_MARKER in log_text, engaged=S2_ENGAGED_MARKER in log_text,
                   f22=SDPA_EXTENT_MARKER in log_text)
    first = lambda marker: [line.strip()[:240] for line in lines if marker in line][:2]
    region = trace_region(log_text)
    report = dict(
        extent_replay=on, audit=audit_on, capture_position=capture, force_cap=environ.get(FORCE_CAP_FLAG) or None,
        markers=markers, admission_lines=first(S2_ADMISSION_MARKER), engaged_lines=first(S2_ENGAGED_MARKER),
        rounds=rounds, packed_lines=packed_lines, extent_audit=audit,
        capture_overrides=sorted(set(CAPTURE_OVERRIDE_LINE.findall(log_text))),
        cap_refused=sum(1 for line in lines if CAP_REFUSED_MARKER in line),
        deadline=sum(1 for line in lines if DEADLINE_MARKER in line),
        idle_commits=sum(1 for line in lines if PADDED_IDLE_COMMIT_MARKER in line),
        refused=refused_rounds_report(log_text),
        narrowed=sum(1 for line in lines if NARROWED_MARKER in line),
        narrowing_refused=sum(1 for line in lines if NARROWING_REFUSED_MARKER in line),
        dram_hold=dram_holds(log_text),
        quarantined=sum(1 for line in lines if QUARANTINED_MARKER in line), quarantined_lines=first(QUARANTINED_MARKER),
        releases=proposal_releases(log_text),
        quads_built=len(QUAD_BUILT_LINE.findall(log_text)),
        ladders=[ladder_of(text) for text in PROPOSAL_LADDER.findall(log_text)][:32],
        before=before_points(log_text),
        trace_region=region, trace_region_lines=region['lines'],
        other_audits=other)
    try:
        from acceptance_report import live_rate
        report['live4'] = live_rate(log_text, 4)
    except Exception as error:
        report['live4'] = dict(error='%s: %s' % (type(error).__name__, error))
    try:
        report['paths'] = user_paths(log_text, streams or [], prompt_lengths)
    except Exception as error:
        report['paths'] = dict(error='%s: %s' % (type(error).__name__, error))
    problems = []
    if on:
        for key, marker, what in (('admission', S2_ADMISSION_MARKER, 'the packed-any admission (W7) never ran'),
                                  ('engaged', S2_ENGAGED_MARKER, 'no extent reader was built (W1)'),
                                  ('f22', SDPA_EXTENT_MARKER, 'no 0x20 program was built: the K64j runtime-extent '
                                                              'factory never ran')):
            if not markers[key]:
                problems.append('%s=1: no "%s" line: %s' % (EXTENT_REPLAY_FLAG, marker, what))
        if packed_lines and not rounds['count']:
            problems.append('%s=1: %d [PACKED] lines but no "%s" line: the packed rounds did not run the extent verify '
                            '(W3)' % (EXTENT_REPLAY_FLAG, packed_lines, S2_ROUND_MARKER))
        if audit_on:
            if audit['mismatches']:
                problems.append('%s: %d MISMATCH lines (%s)' % (EXTENT_AUDIT_FLAG, audit['mismatches'],
                                                                 '; '.join(audit['mismatch_lines'][:2])))
            if audit['malformed']:
                problems.append('%s: %d audit lines without %s (%s): the audit\'s line is not the one this gate reads'
                                % (EXTENT_AUDIT_FLAG, audit['malformed'], '/'.join(EXTENT_AUDIT_REQUIRED),
                                   '; '.join(audit['malformed_lines'])))
            audited = set(entry['round'] for entry in extent_audit_entries(log_text)[0])
            unaudited = sorted(set(entry['round'] for entry in extent_round_entries(log_text)) - audited)
            if audit['lines'] < rounds['count'] or unaudited:
                problems.append('%s: %d audit lines for %d packed extent rounds (unaudited rounds %s): a packed round '
                                'went unaudited' % (EXTENT_AUDIT_FLAG, audit['lines'], rounds['count'], unaudited[:8]))
            if audit['incomplete']:
                problems.append('%s: %d rounds did not read back every segment\'s word and cur_pos (rounds %s)'
                                % (EXTENT_AUDIT_FLAG, audit['incomplete'], audit['incomplete_rounds']))
        low = (report['paths'] or {}).get('packed_below_floor') or []
        if low:
            problems.append('%s=1: packed rounds below position %d (%s): the admission floor broke'
                            % (EXTENT_REPLAY_FLAG, S2_MIN_LIVE_START, '; '.join(low[:3])))
    else:
        leaked = sorted(marker for marker in S2_MARKERS if marker in log_text)
        if leaked:
            problems.append('%s is off, but the server log carries S2 markers (%s): this is not the profile\'s path'
                            % (EXTENT_REPLAY_FLAG, '; '.join(leaked)))
    if report['cap_refused']:
        problems.append('%d "%s" lines: a commit past its boundary cap reached the block backstop (an outage for '
                        'the round)' % (report['cap_refused'], CAP_REFUSED_MARKER))
    if report['deadline']:
        problems.append('"%s": a replay overran its deadline and the engine exited' % DEADLINE_MARKER)
    if capture is not None and capture not in report['capture_overrides']:
        problems.append('%s=%s but no "[PINDIAG] packed capture position override=%s" line: the knob never reached '
                        'the block' % (CAPTURE_POSITION_FLAG, capture, capture))
    report['problems'] = problems
    return report


def add_s2_report(report, environ, log_text, streams, prompt_lengths):
    """report['s2'] for an S2-relevant arm, its problems added to flag_markers['missing'] (so gate_passed
    carries them); a failure here is itself such a problem, never a lost report."""
    if not s2_relevant(environ, log_text):
        return
    try:
        report['s2'] = s2_report(environ, log_text, streams, prompt_lengths)
    except Exception as error:
        report['s2'] = dict(error='%s: %s' % (type(error).__name__, str(error)[:300]),
                            problems=['s2_report failed: %s: %s' % (type(error).__name__, str(error)[:300])])
    markers = report.setdefault('flag_markers', dict(found={}, missing=[]))
    markers.setdefault('missing', []).extend(report['s2']['problems'])


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
    platform = options.server_argv == 'platform'
    if options.expect_profile is not None and not platform:
        parser.error('--expect-profile needs --server-argv platform: only the C2 contract reports a profile')
    if options.readiness_seconds < 1:
        parser.error('--readiness-seconds must be positive')
    try:
        options.events = user_events(options, options.sequential_users or options.users)
    except ValueError as error:
        parser.error(str(error))
    if any(options.events.values()) and not detail_mode(options):
        parser.error('--drops, --user-max-tokens and --user-ignore-eos need detail streams (real text or --eos stop)')
    if options.alive_check < 0 or options.alive_seconds < 1:
        parser.error('--alive-check needs a count of at least 1 and --alive-seconds a positive limit')
    log_drops = sorted(user for user, (kind, _) in options.events['drops'].items() if kind in LOG_DROP_KINDS)
    if log_drops:
        # The server log names a request's user before its first byte only by the prompt length its
        # ledger line carries, so each such user needs a length no other user has.
        try:
            import real_text_prompts
            lengths = real_text_prompts.parse_targets(options.prompt_lengths) if options.prompt_lengths else None
        except ValueError:
            lengths = None   # refused below with its own message
        if lengths is not None and not all(user < len(lengths) and lengths.count(lengths[user]) == 1
                                           for user in log_drops):
            parser.error('--drops prefill+S and build need a --prompt-lengths length no other user has (users %s)'
                         % log_drops)
        if options.prompt_lengths is None:
            parser.error('--drops prefill+S and build need --prompt-lengths: the server log names a request\'s user '
                         'by its prompt length')
    if options.prompt_lengths is not None:
        if not real_text:
            parser.error('--prompt-lengths needs --prompt-source real-text')
        try:
            import real_text_prompts
            options.prompt_lengths = real_text_prompts.parse_targets(options.prompt_lengths)
        except ValueError as error:
            parser.error(str(error))
        streams = options.sequential_users or options.users
        if len(options.prompt_lengths) != streams:
            parser.error('--prompt-lengths gives %d lengths for %d streams (--users, or --sequential-users)'
                         % (len(options.prompt_lengths), streams))
        # Under the platform argv the contract clamps max_tokens to what the context leaves; the
        # recipe argv has no contract, so there every prompt must leave its whole budget.
        room = options.context - (1 if platform else options.max_tokens)
        too_long = [length for length in options.prompt_lengths if length > room]
        if too_long:
            parser.error('--prompt-lengths %s exceed %d (--context %d less %s)' % (
                too_long, room, options.context, 'one token' if platform else '--max-tokens'))
        return options
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
    The arms keep their --context: KV sizing and the 4 x 131k fit are computed from it.
    None under --prompt-lengths, where every user has its own."""
    if getattr(options, 'prompt_lengths', None):
        return None
    return min(options.prompt_tokens, options.context - options.max_tokens)


def detail_mode(options):
    """Anything but the default mode (synthetic prompts, --eos ignore). Only here does the gate
    ask for detail streams and add the acceptance, rate and configuration diagnostics; the
    default mode's requests, report and stdout stay exactly what they were."""
    return not (options.prompt_source == 'synthetic' and options.eos == 'ignore')


def stream_kwargs(options):
    """stream_once's keywords. None at all in the default mode (synthetic, --eos ignore), so every
    existing call is unchanged - the payload stream_once sends and the fields it records. Under
    --server-argv platform the streams also name the platform's served model."""
    kwargs = {} if not detail_mode(options) else dict(ignore_eos=options.eos == 'ignore', detail=True)
    if getattr(options, 'server_argv', 'gate') == 'platform':
        kwargs['model'] = options.served_model_name
    return kwargs


def real_text_stream_problems(results, prompt_lengths=None):
    """A real-text arm has no reference to fail a bad stream, so each stream must itself have
    completed: no error, some text, a finish_reason of stop (EOS) or length (the budget), and -
    given the built lengths - a usage.prompt_tokens equal to its prompt's length, the cheap proof
    that the server served the built ids with no template or truncation of its own."""
    problems = []
    for index, entry in enumerate(results):
        entry = entry or {}
        if entry.get('dropped') and not entry.get('error'):
            continue   # a lifecycle drop (--drops): the client went away on purpose
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


# What qwen_configuration records (report['configuration_scope']): every QWEN<n>_* flag (QWEN_*, and
# QWEN35_GDN_*, QWEN36_* that the stock model code reads), every TT_*, MESH_DEVICE and OMP_NUM_THREADS.
# Reports from before the scope was recorded carry QWEN_* only; real_text_compare compares two reports
# on the names both scopes cover.
CONFIGURATION_SCOPE = 'qwen-tt'
CONFIGURATION_PREFIX = re.compile(r'(?:QWEN[0-9]*_|TT_)')
CONFIGURATION_NAMES = ('MESH_DEVICE', 'OMP_NUM_THREADS')


def qwen_configuration(environ):
    """The configuration flags the server process inherits (CONFIGURATION_SCOPE): what
    real_text_compare.py diffs between a concurrent and a sequential arm, whose flag sets differ
    (v157/v159 vs v158/v160), and between a served arm and a tracked reference."""
    return {name: environ[name] for name in sorted(environ)
            if CONFIGURATION_PREFIX.match(name) or name in CONFIGURATION_NAMES}


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
                      qwen_configuration=qwen_configuration(os.environ), configuration_scope=CONFIGURATION_SCOPE)
    platform = options.server_argv == 'platform'
    if platform:
        report.update(server_argv='platform', served_model_name=options.served_model_name,
                      expect_profile=options.expect_profile, snapshot=options.snapshot)
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
            if options.prompt_lengths:
                report['prompt_lengths_requested'] = list(options.prompt_lengths)
                built = real_text_prompts.build_prompts(streams, None, model=options.snapshot,
                                                        log=lambda line: print(line, flush=True),
                                                        targets=options.prompt_lengths)
            else:
                built = real_text_prompts.build_prompts(streams, report['real_text_target'], model=options.snapshot,
                                                        log=lambda line: print(line, flush=True))
            real_text_prompts.write_prompts(options.results / REAL_TEXT_PROMPTS, built)
            report['real_text'] = real_text_prompts.summary(built)
            prompts = [entry['tokens'] for entry in built['users']]
            print('[REALTEXT] %d prompts of %s tokens (target %s) in %.1f s, %d tokenizer calls; corpus %s files, '
                  '%s characters, sha256 %s' % (
                      streams, report['real_text']['prompt_lengths'], report['real_text_target'],
                      built['seconds']['total'], built['tokenizer_calls'], built['corpus'].get('files'),
                      built['corpus'].get('characters'), str(built['corpus'].get('sha256'))[:16]), flush=True)

        def prompt(index):
            if prompts is not None:
                return prompts[index]
            return prompt_for(options.prompt_base, options.prompt_user_offset, index, options.prompt_tokens)

        if platform:
            process, handle, log_path, command = start_server(
                options.port, options.users, options.context, options.results, 'server.log',
                readiness_seconds=options.readiness_seconds, trace_region_bytes=options.trace_region_bytes,
                command=platform_argv(options.port, options.served_model_name))
        else:
            process, handle, log_path, command = start_server(
                options.port, options.users, options.context, options.results, 'server.log',
                readiness_seconds=options.readiness_seconds, trace_region_bytes=options.trace_region_bytes)
        report['command'] = command
        report['ready'] = True

        results = [None] * streams
        kwargs = stream_kwargs(options)
        events = options.events if any(options.events.values()) else None
        watch = None
        if events:
            report['user_events'] = dict(drops={str(u): describe_drop(drop) for u, drop in sorted(events['drops'].items())},
                                         max_tokens={str(u): m for u, m in sorted(events['max_tokens'].items())},
                                         ignore_eos=events['ignore_eos'])
            if events['drops']:
                watch = StreamWatch(events['drops'], (report.get('real_text') or {}).get('prompt_lengths'),
                                    log_path).start()
        try:
            if options.sequential_users:
                # One request at a time: each is the only stream on the server, which is
                # what a single-stream reference means.
                for index in range(streams):
                    budget, user_kwargs = user_stream(options, index, kwargs, watch)
                    stream_once(options.port, prompt(index), budget, results, index, options.stream_timeout,
                                **user_kwargs)
            threads = [] if options.sequential_users else [threading.Thread(
                target=stream_once,
                args=(options.port, prompt(index), user_stream(options, index, kwargs, watch)[0], results, index,
                      options.stream_timeout),
                kwargs=user_stream(options, index, kwargs, watch)[1])
                for index in range(options.users)]
            for position, index in enumerate(request_order(options) if threads else []):
                if position and options.stagger:
                    time.sleep(options.stagger)
                threads[index].start()
            for thread in threads:
                thread.join()
        finally:
            if watch is not None:
                watch.stop()
                report['lifecycle'] = watch.report(results)
        report['streams'] = results
        alive_index = None
        if options.alive_check:
            # The engine outlived what the streams did (drops, a cancel, a one-token request) and gave
            # every seat back: --alive-check N more requests at once, the shortest prompt, 8 tokens
            # each, after every stream has ended, all answered within --alive-seconds.
            alive_index = len(ledger_readings(log_path.read_text(errors='replace') if log_path.is_file() else ''))
            shortest = min(range(streams), key=lambda index: len(prompt(index)))
            report['alive_after'], report['alive'] = alive_check(
                options.port, prompt(shortest), options.alive_check, options.stream_timeout, options.alive_seconds,
                kwargs)
            print('[ALIVE] after the streams: %s (%d at once)' % (report['alive'], options.alive_check), flush=True)
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
                                  max_tokens=user_stream(options, index, {})[0],
                                  finish_reason=(entry or {}).get('finish_reason'),
                                  actual_len=len((entry or {}).get('text') or ''))
                if entry and entry.get('error'):
                    comparison['error'] = entry['error']
                if events:
                    comparison.update(dropped=(entry or {}).get('dropped'),
                                      ignore_eos=index in events['ignore_eos'])
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

        if platform:
            report['platform'] = platform_report(log_text, options.expect_profile)
            report['dram'] = dram_report(log_text)
            report['ledger'] = ledger_report(log_text, alive_index)
            try:
                report['flag_markers'] = flag_marker_report(os.environ, streams, log_text,
                                                            prompt_tokens=marker_prompt_tokens(options, report))
            except Exception as error:
                # Mixed lengths and more streams than seats reach shapes the marker floors were never
                # written for: a failure there is recorded as a missing marker, never a lost report.
                report['flag_markers'] = dict(found={}, missing=['flag_marker_report failed: %s: %s' % (
                    type(error).__name__, str(error)[:300])])
        else:
            report['flag_markers'] = flag_marker_report(os.environ, streams, log_text,
                                                        prompt_tokens=marker_prompt_tokens(options, report))
        if detail:
            # S2 (s2-design W11): only an arm with the extent flag, a gate knob or an S2 line gets report['s2'];
            # every other arm's report keeps exactly its keys.
            add_s2_report(report, os.environ, log_text, results, (report.get('real_text') or {}).get('prompt_lengths'))
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
        if platform:
            report['gate_passed'] = bool(report['gate_passed'] and not report['platform']['problems'])
        if options.alive_check:
            report['gate_passed'] = bool(report['gate_passed'] and report['alive'])
    except BaseException as error:
        report['fatal'] = '%s: %s' % (type(error).__name__, str(error)[:600])
    finally:
        stop_server(process, handle)
        log_path = options.results / 'server.log'
        if platform and 'platform' not in report:
            # A run that died before its streams still says what the contract launched, if anything.
            try:
                text = log_path.read_text(errors='replace') if log_path.is_file() else ''
                report['platform'] = platform_report(text, options.expect_profile)
                report['dram'] = dram_report(text)
            except Exception as error:
                report['platform'] = dict(problems=['platform report failed: %s' % error])
        if platform and report.get('command') is not None:
            # What served is the contract's rewrite of what was launched (read-the-launched-argv):
            # 'command' is the served argv (None when the contract logged none), the platform's own
            # is 'command_requested'.
            served = (report.get('platform') or {}).get('served_argv')
            report['command_requested'] = report['command']
            report['command'] = list(report['command'][:3]) + served if isinstance(served, list) else None
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
