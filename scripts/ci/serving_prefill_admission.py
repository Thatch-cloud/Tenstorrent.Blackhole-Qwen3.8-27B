"""C2-any: at most one fresh prompt per prefill step, on the scheduler class the engine actually runs.

Run 36211578069 (G4 ladder, profile c2, nine users submitted at once) killed the EngineCore with
`Fast serving requires one complete fresh prefill: prefill_slot=None new=[three ids] cached=[]`
(serving_lifecycle._execute). The fast prefill path takes one capture and runs one prompt; its GDN
prefill scratch and host RoPE table are single-occupancy. The stock TTScheduler admits every waiting
prompt that fits max-num-batched-tokens into one prefill step (131328 under c2), so three short
prompts that arrive together are batched into one step, and the lifecycle refuses that step.

The fast path's own one-in-flight scheduler never runs. serving_fast_policy.validate_fast_config
sets scheduler_cls to serving_one_in_flight.OneInFlightScheduler. Then the TT platform's
check_and_update_config, in its non-lane branch (plugin bf77cd63 platform.py:1080-1081), overwrites
it with 'vllm_tt_plugin.scheduler.TTScheduler'. The last writer wins, which run 35690327326 had
already shown. Before this module, `exact` was safe only structurally (it admits only
staged-context prompts, and two of those cannot fit one step), and the c2 gate was safe only
because it staggered arrivals by 0.25 s.

THE RULE is lever_n_model_patch.patch_scheduler's, the M2 graft that rewrites TTScheduler.
_schedule_prefill_only in the plugin's scheduler.py (the m3native arm mounts it). It is reused
here, not re-invented:
- if partial prefills are running: allowed = their count, and both waiting queues are hidden, so
  nothing new joins a partial (run 35689293766);
- else, if the lifecycle's prefill gate (sys.modules['_qwen_prefill_gate'].held) names a request:
  allowed = 0, and the queues are hidden (run 35714211185);
- else: allowed = 1, and the queues stay visible, so exactly one fresh prompt is admitted
  (run 35690327326, new=[A, B]).
The waiting loop then sees max(0, min(saved_max - decodes, allowed)). A gate that is absent or
unreadable counts as not held.

Here it is applied at runtime, as a wrapper around the class's own _schedule_prefill_only. It is
not a textual graft, because the serving image ships the plugin's scheduler.py unpatched. The
wrapper writes min(saved_max, allowed + decodes) before delegating. The plugin then subtracts the
decodes (max(0, max_num_running_reqs - len(pure_decodes))) and lands on exactly the graft's value
(serving_one_in_flight's compensation). The queues are hidden and merged back with the file's own
idiom: create_request_queue(self.policy) and prepend_requests, as _schedule_decode_only and the
graft do. test_serving_prefill_admission checks that this matches the graft call for call.

WHERE and WHEN: the lifecycle installs it beside the D2 quarantine consumer
(serving_request_quarantine), on the class the worker's own config names
(serving_request_quarantine.scheduler_class, which on the TT platform is TTScheduler). Both share
the EngineCore process (uniproc executor). The lifecycle does this only under
QWEN_FAST_ANY_REQUEST=1 (the c2 profiles), so `exact` schedules exactly as before. On TT a step is
either all prefill or all decode, and that is unchanged. A prefill step still pauses the running
decodes for that step, as the stock scheduler already did. The difference is that three
simultaneous arrivals now take three one-prompt prefill steps instead of one three-prompt step
that kills the engine. Decodes are never dropped from `running`. When nothing can be admitted
(every seat decoding, or the gate held), the plugin's default mode falls back to a decode-only
step (scheduler.py:140-142).

Two markers show whether it ran (memory: graft-mounted-is-not-graft-executed):
    [PINDIAG] one fresh prefill per step installed on <module.Class>   (once, at install)
    [PINDIAG] one fresh prefill per step live in <Class>               (once, the first prefill
                                                                        step: the platform warmup)
There is also one decision line per distinct (partials, decodes, gate held, allowed, hidden) state.
That keeps the log bounded; the graft logged every call.

S2 W6b, THE DRAM ADMISSION HOLD (s2-design.md section 3.2 item 2; QWEN_FAST_EXTENT_REPLAY=1 only). Beside the
64-row block (3.84 GB per chip) four per-request engines leave about 1.17 GB per chip (d); the fourth arrival's
prefill and engine build see about 1.97 GB (d), or about 1.48 GB (d) while a departed user's quad and pair
traces are still held (W6c releases them at detach). A prefill or build that runs out of DRAM kills the engine
and every live user with it, and a refusal raised after the prefill (the bridge's RequestRefused backstop,
serving_request_factory.dram_backstop) has already spent the prefill. So the worker parks a predicate under
DRAM_KEY at attach (serving_request_factory.register_dram_admission), the prefill gate's sys.modules pattern,
and the wrapper asks it before one fresh prompt is let in - and only then: with a partial in flight, the gate
held, or every seat decoding (max-num-seqs) the waiting loop admits nobody anyway, so nothing is asked or logged.
It is asked about the request whose need binds among those the waiting loop may admit this step
(admission_candidates: a head vLLM skips for a blocked status, and the requests up to the first unblocked one).
While it refuses, allowed = 0 and both queues are hidden, exactly as behind a held gate: the plugin falls back
to a decode-only step, the running users keep decoding, and the hold resolves when one of them finishes and its
engine is freed. With no decode left to wait for nothing can free DRAM, so the prompt is admitted anyway
('lifted') and the backstop decides after its prefill: a hold never livelocks the engine.
Two timing facts of vLLM 0.25.1 and the plugin shape it:
- THE READING IS ONE STEP STALE AFTER A FINISH. vLLM names a request that finished in the NEXT step's output
  (Scheduler.finished_req_ids, scheduler.py:2108, :1105), and the worker detaches it - its engine and dead
  proposal traces freed - at the start of executing that step (serving_lifecycle._execute), before any prefill
  the step carries. A refusal on a step whose output will name finished requests is therefore DEFERRED one step:
  held, but not a hold - no hold line, the held state untouched - and decided on the next step's fresh reading.
- A HELD PASS TAKES THE FINISHED IDS WITH IT. schedule() hands finished_req_ids to its output and starts a new
  set (:1105, :1210). A held pass schedules nothing, so with a decode running the plugin discards it and runs a
  decode-only pass (plugin bf77cd63 scheduler.py:136-143) whose output would name no finished request: the
  worker would never detach it or free its engine, and the next decode step would find a bridge vLLM no longer
  schedules (serving_vllm_packed.ordered_tickets refuses that set). So after a DRAM-held pass the plugin
  discards, the wrapper puts the ids back (carry_finished) for the decode-only pass to carry.
The need is ONE set of defaults, here (dram_need): the engine build's resident cost and transient margin, the
prefill transient of a prompt of PREFILL_TRANSIENT_FROM tokens or more, and the coordinator's DRAM reserve.
THE SPLIT (G5 churn, gate v79, GitHub run 36368363993). The need used to be asked of the smallest largest free
block alone, as if an engine were one buffer. An engine is some 1,400 buffers per chip (the ledger's
engine_request item: 2,134 to 3,416 over the two chips), and v79 built replacement engines wholly in the holes
departed users left, the largest block unchanged (its engine5, engine8 and engine9; v70 and v71 once each). At
each of v79's holds with three decoding, 2.319 GB per chip was free - 1.48 times the 1568.4 MB need - beside a
largest block of 1079.7, 1320.7 or 1321.8 MB: fragmented, not short, and the hold failed G5 with a seat free.
So a reading now fits only when all three terms hold (split_short), each on the smallest chip:
  free        the total free less STRANDED_BYTES (what no request can use) covers the need, whose 200 MB build
              margin stays in it;
  contiguous  the largest free block holds the reserve plus LARGEST_BUFFER_BYTES (contiguous_need): the one
              buffer that must land in one block, with the reserve still free beside it. At admission a prompt of
              PREFILL_TRANSIENT_FROM tokens or more also needs its prefill transient in that block
              (admission_contiguous_need: 696.4 MB at the 256 MiB reserve), because a prefill takes what it
              leaves from the largest block, not the holes: v79's 110000-token prompt at 2.077 GB free took the
              block from 1886.4 to 1822.6 MB. Without it a long prompt was admitted beside the same 396.4 MB
              block the backstop asks after the prefill, so the prefill's residue alone could push the backstop
              into refusing a spent 120k prefill, and the unmeasured transient (Q8/Q9) had no contiguous cover;
  trace       the trace region's largest free block holds TRACE_CONTIGUOUS_BYTES (an engine's traces), where the
              pool can read the region; where it cannot, this term holds nothing.
The bridge's backstop (serving_request_factory.dram_backstop) and the proposal coordinator's fresh pair and quad
captures and its pair release (dflash_packed_proposal_coordinator.capture_headroom, under the same flag) apply
the same terms, with the contiguous term that carries no prefill transient: the backstop runs after the prefill,
so a long prompt keeps the 300 MB gap between its admission and its backstop that it had before the split
(1568.4 against 1268.4 MB then). No predicate, an unreadable one, or a reading that is unavailable holds nothing:
the rule above, call for call.
    [PINDIAG] dram hold prompt=<n> largest_free=<MB> need=<MB> request=<id> decodes=<d> free=<MB>
        trace_largest_free=<MB> short=<terms>                                         once per (request, decodes)
    [PINDIAG] dram hold released prompt=<n> ...                                        once, when it fits
    [PINDIAG] dram hold lifted prompt=<n> ...                                          no decode left to wait for
    [PINDIAG] dram admission deferred one step prompt=<n> ... finished=[<ids>] ...     a stale reading, not a hold
    [PINDIAG] dram admission carried finished=[<ids>] ...                              the ids put back
    [PINDIAG] dram admission fit prompt=<n> ... decodes=<d> free=<MB> ...              once per request admitted
                                                                                       without a hold: the reading
                                                                                       it was admitted on (M11)
Only lines that start with DRAM_HOLD are holds (what a gate counts). Each state logs once; the state kept to
decide that is the held (request, decodes) and the last line noted, so it stays bounded whatever the traffic.

DECODE CREDIT (QWEN_FAST_DECODE_STEPS_PER_ADMISSION=R, default 0 = off; lever N's alternation, docs/lever-N). When
several fresh prompts wait, each admission's prefill and engine build holds the gate for 2 to 3 s and no live user
decodes (the admission freeze: four arrivals together stall every earlier user for about 14 s). With R > 0 the
wrapper owes the running decodes R decode-only steps after each admission: the next R calls answer allowed = 0
with both queues hidden, exactly as behind a held gate or a DRAM hold, so the plugin runs a decode-only pass for
each. A step is a schedule() call, and the credit
- is armed by an admission (the call let one prompt in and scheduled it), whatever is waiting at that time;
- is paid by each decode-only pass that follows it: its own holds, and a pass a held gate or a DRAM hold forced;
- is dropped by a schedule() call that asks for no prefill (nothing pending, so the decodes ran), by a call that
  finds no decode running, and by a call that finds a partial prefill in flight, so a credit never outlives the
  burst it was armed in: a lone arrival, or the first one after the burst, is admitted at once;
- holds only while a decode is running, so it never idles the device, and never holds a partial prefill.
It moves time from the later users' first token to the earlier users' decode. It is exactness-neutral by design (it
changes scheduling order only), UNVERIFIED on hardware: it runs decode rounds with 1 to 3 live users in the middle
of a burst, which the scheduler never runs today, so token identity rests on packed == solo and the arm must be
judged real-text exact against solo, as every TP4 arm is. The finished ids of a held pass are put back for the decode pass as under the DRAM hold (carry_finished). R = 0 wraps schedule() in nothing and
every call is what it was. A bad value of the flag is refused (logged as REFUSED) and installs the cap with R = 0.
    [PINDIAG] decode credit installed on <class>: steps=<R>               once, at install, with R > 0
    [PINDIAG] decode credit armed steps=<R> decodes=<d> waiting=<n>      once per admission
    [PINDIAG] decode credit REFUSED <value>: ...                          a bad flag value; the cap stays, credit off
    [PINDIAG] decode credit hold left=<k> decodes=<d>                    once per held pass
    [PINDIAG] decode credit carried finished=[<ids>] ...                 the ids put back past the held pass
"""

import importlib
import os
import sys
import types

from serving_request_quarantine import scheduler_class

# serving_lifecycle.PREFILL_GATE_KEY. Duplicated rather than imported, so that the scheduler side
# does not import the lifecycle. The test pins the two equal.
GATE_KEY = '_qwen_prefill_gate'
WRAPPED = '_qwen_one_fresh_prefill'
METHOD = '_schedule_prefill_only'
INSTALLED = '[PINDIAG] one fresh prefill per step installed on '
LIVE = '[PINDIAG] one fresh prefill per step live in '

# S2 W6b, the DRAM admission hold (the module docstring): the key the worker parks its predicate under, and the
# lines the hold logs (DRAM_HOLD is the prefix a gate counts).
DRAM_KEY = '_qwen_dram_admission'
DRAM_HOLD = '[PINDIAG] dram hold prompt='
# The first five fields keep the order a gate parses (lever_n_m3native_gate.DRAM_HOLD_LINE); the split's readings
# follow them (the module docstring).
DRAM_HOLD_LINE = DRAM_HOLD + '{} largest_free={} need={} request={} decodes={} free={} trace_largest_free={} short={}'
DRAM_RELEASED_LINE = ('[PINDIAG] dram hold released prompt={} largest_free={} need={} request={} free={} '
                      'trace_largest_free={}')
DRAM_LIFTED_LINE = ('[PINDIAG] dram hold lifted prompt={} largest_free={} need={} request={} free={} short={}: no '
                    'decode is left to free DRAM, so the prompt is admitted and the bridge backstop decides')
DRAM_UNAVAILABLE_LINE = '[PINDIAG] dram hold unavailable request={}: {} (not held)'
DRAM_DEFERRED_LINE = ('[PINDIAG] dram admission deferred one step prompt={} largest_free={} need={} request={} '
                      'finished={} free={} short={}: the reading still counts their engines, which this step '
                      'detaches first')
DRAM_CARRIED_LINE = ('[PINDIAG] dram admission carried finished={} past the discarded prefill pass into the '
                     'decode-only step')
DRAM_FIT_LINE = ('[PINDIAG] dram admission fit prompt={} largest_free={} need={} request={} decodes={} free={} '
                 'trace_largest_free={}')
# DECODE CREDIT (the module docstring): decode-only steps owed after an admission; 0 (the default) is off.
STEPS_FLAG = 'QWEN_FAST_DECODE_STEPS_PER_ADMISSION'
MAX_STEPS = 64
CREDIT_ARMED_LINE = '[PINDIAG] decode credit armed steps={} decodes={} waiting={}'
CREDIT_HOLD_LINE = '[PINDIAG] decode credit hold left={} decodes={}'
CREDIT_CARRIED_LINE = ('[PINDIAG] decode credit carried finished={} past the discarded prefill pass into the '
                       'decode-only step')
CREDIT_REFUSED_LINE = '[PINDIAG] decode credit REFUSED {!r}: {}; the one-fresh-prefill cap is installed with the credit off'
CREDIT_INSTALLED_LINE = '[PINDIAG] decode credit installed on {}: steps={}'
MEGABYTE = 10 ** 6
# The need: one set of defaults, per chip (s2-design.md section 3.2 item 2). M8 and M11 calibrate them.
ENGINE_BUILD_BYTES = 800 * MEGABYTE          # (m) an engine with one 2048 proposal bucket (v26, run 36218104858)
ENGINE_BUILD_MARGIN_BYTES = 200 * MEGABYTE   # (e) the build's transient above what stays resident
PREFILL_TRANSIENT_BYTES = 300 * MEGABYTE     # (e, UNVERIFIED Q8) the prefill of a long prompt beside the block
PREFILL_TRANSIENT_FROM = 2048                # prompts shorter than this: no prefill transient (e)
# (m) What any prefill leaves in the largest block until the backstop reads it: 2047-token prefills took 86.9 MB
# (v26, 1475.8 to 1388.9 MB) and 85.2 MB (v23, 3873.9 to 3788.7 MB), v79's 1536-token ones 47 to 71 MB and its long
# ones 0 to 76.2 MB. The admission asks at least this much of the block beside the backstop's term, so a short prompt
# it admits is not refused after its prefill (a hold costs a wait; a refusal throws the prefill away).
PREFILL_RESIDUE_BYTES = 100 * MEGABYTE
# THE SPLIT's terms (the module docstring; gate v79, GitHub run 36368363993, gate/churn/server.log), per chip.
# (m) Free DRAM no request can use, left out of the free term. With no user live v79 read free 4.490 GB beside a
# largest block of 4191.7 MB (298.3 MB apart; 4.489 GB and 4190.5 MB, 298.5 MB, at its end): the first prefill's
# model_after_prefill (161 MB in 742 buffers) stays resident for the process among small holes. Rounded up.
STRANDED_BYTES = 300 * MEGABYTE
# (e, UNVERIFIED Q8/Q9) The largest single buffer an engine build, a long prefill or a proposal capture allocates:
# never measured, so a conservative bound. The largest sized in source is 21.3 MB (a single-user capture's
# (1,1,2080,5120) bf16 history; the prepare_publication transient is 20.97 MB). With the 256 MiB reserve the largest
# block must hold 396.4 MB (contiguous_need); v79's smallest largest block at any hold or before point was 673.2 MB.
# The ledger's largest= (memory_ledger) only BOUNDS it from below, and only on the items a request allocates
# (memory_ledger.REQUEST_ITEMS: engine_request, model_after_prefill, quad_intermediates, quad_placeholders): the
# startup items are no measure of it (v79's P0 model.embedding is one 1.271 GB buffer per chip, lm_head 675 MB, each
# kv_caches buffer about 286 MB; all resident before any admission, none asked of the block), and the walk sees only
# what is resident, never a transient peak (prefill intermediates, engine-build scratch), which is freed before it
# walks and is the Q8/Q9 risk itself. A request item above this bound says raise it (the c2 gate fails on one); one
# at or below it does not show the bound is enough.
LARGEST_BUFFER_BYTES = 128 * MEGABYTE
# (m) The trace region's largest free block an engine build needs: each v79 engine's traces took 41.6 MB (its
# ledger's trace_used, 39.6 to 81.2 MB across one build), and at the holds with three decoding the region's largest
# free block read 49.0 MB (its before points at 02:14:03 and 02:17:29). 44 MB leaves 2.4 MB over the traces;
# 48 MB would leave 1 MB of v79's reading, so a slightly worse layout would hold again.
TRACE_CONTIGUOUS_BYTES = 44 * MEGABYTE
SPLIT_TERMS = ('free', 'contiguous', 'trace')

# EIGHT SEATS (tp4/seats8, review 3): the three bounds that were never measured at eight engines beside two 64-row blocks
# (PREFILL_TRANSIENT_BYTES, LARGEST_BUFFER_BYTES, ENGINE_BUILD_BYTES) are tunable from the profile's env, in whole
# megabytes per chip, so a window can change them without an image rebuild. The module constants above stay the
# DEFAULTS: with the variable unset (every profile today) each bound is exactly the constant, so the four-seat decisions
# are unchanged byte for byte. A value that is not a non-negative whole number of megabytes is refused (ValueError), never
# read as a default: a profile that names a bound means it. Read at each decision, never at import.
TUNING_FLAGS = (('QWEN_FAST_DRAM_ENGINE_BUILD_MB', 'ENGINE_BUILD_BYTES'),
                ('QWEN_FAST_DRAM_PREFILL_TRANSIENT_MB', 'PREFILL_TRANSIENT_BYTES'),
                ('QWEN_FAST_DRAM_LARGEST_BUFFER_MB', 'LARGEST_BUFFER_BYTES'))


def tuned_bytes(flag, default, environ=None):
    """The bound in bytes: `default` when `flag` is unset, else the flag's whole megabytes (ValueError otherwise)."""
    value = (os.environ if environ is None else environ).get(flag)
    if value is None:
        return default
    text = value.strip()
    if not text.isascii() or not text.isdigit():
        raise ValueError('%s must be a non-negative whole number of megabytes, got %r' % (flag, value))
    return int(text) * MEGABYTE


def engine_build_bytes(environ=None):
    return tuned_bytes('QWEN_FAST_DRAM_ENGINE_BUILD_MB', ENGINE_BUILD_BYTES, environ)


def prefill_transient_bytes(environ=None):
    return tuned_bytes('QWEN_FAST_DRAM_PREFILL_TRANSIENT_MB', PREFILL_TRANSIENT_BYTES, environ)


def largest_buffer_bytes(environ=None):
    return tuned_bytes('QWEN_FAST_DRAM_LARGEST_BUFFER_MB', LARGEST_BUFFER_BYTES, environ)


# THE LONG-PREFILL TIER (tp4/seats8-262k, B5; optional): a 253,920-token prefill may need more DRAM beside the blocks than a 2,048-token one, and the flat
# PREFILL_TRANSIENT_BYTES charges every prompt of PREFILL_TRANSIENT_FROM tokens or more the same. With BOTH flags set - the prompt length from which the tier
# applies (whole tokens, at least PREFILL_TRANSIENT_FROM) and its transient in whole megabytes per chip - a prompt at or past that length is charged the tier's
# bytes instead of the flat ones, and a shorter one the flat ones as ever. With both unset (every profile today) prefill_transient is exactly what it was; one
# without the other, or a value that is not a whole number, is refused (ValueError), never read as unset. Read at each decision, never at import.
LONG_FROM_FLAG = 'QWEN_FAST_DRAM_PREFILL_LONG_FROM'
LONG_MB_FLAG = 'QWEN_FAST_DRAM_PREFILL_LONG_MB'


def long_prefill_tier(environ=None):
    """(from_tokens, bytes) of the long-prefill tier, or None when both flags are unset; ValueError for any other state."""
    environ = os.environ if environ is None else environ
    start, size = environ.get(LONG_FROM_FLAG), environ.get(LONG_MB_FLAG)
    if start is None and size is None:
        return None
    if start is None or size is None:
        raise ValueError('%s and %s are set together or not at all' % (LONG_FROM_FLAG, LONG_MB_FLAG))
    text = start.strip()
    if not text.isascii() or not text.isdigit() or int(text) < PREFILL_TRANSIENT_FROM:
        raise ValueError('%s must be a whole number of tokens of at least %d, got %r' % (LONG_FROM_FLAG, PREFILL_TRANSIENT_FROM, start))
    return int(text), tuned_bytes(LONG_MB_FLAG, 0, environ)


def tuning_problems(environ=None):
    """What is wrong with the tuning flags in `environ`, one string each; [] when every set one parses."""
    problems = []
    for flag, constant in TUNING_FLAGS:
        try:
            tuned_bytes(flag, 0, environ)
        except ValueError as failure:
            problems.append(str(failure))
    try:
        long_prefill_tier(environ)
    except ValueError as failure:
        problems.append(str(failure))
    return problems



def _log(message, *values):
    try:
        from loguru import logger
    except ImportError:
        print(message.format(*values), flush=True)
        return
    logger.info(message, *values)


def gate_held(modules=None):
    """The request the lifecycle holds in its prefill phase, read the way the graft reads it.

    Absent, unset or unreadable all mean None ("not held"). That is the behaviour before the gate
    existed. The scheduler must never hard-depend on the baked tree."""
    modules = sys.modules if modules is None else modules
    try:
        return getattr(modules.get(GATE_KEY), 'held', None)
    except BaseException:
        return None


def admission(partials, held):
    """(allowed, hide) for one prefill step: lever_n_model_patch.patch_scheduler's rule.

    allowed is how many requests the step may run (the partials, or one fresh prompt, or none
    while the gate is held). hide says whether both waiting queues are blanked for the step."""
    if type(partials) is not int or partials < 0:
        raise ValueError('Non-negative integer partial prefill count required')
    if partials:
        return partials, True
    if held is not None:
        return 0, True
    return 1, False


def engine_build_peak():
    """An engine build's DRAM peak per chip: what stays resident and the build's own transient."""
    return engine_build_bytes() + ENGINE_BUILD_MARGIN_BYTES


def _reserve(reserve):
    if type(reserve) is not int or reserve < 0:
        raise ValueError('A non-negative integer DRAM reserve in bytes is required, got %r' % (reserve,))
    return reserve


def prefill_transient(prompt_tokens):
    """The prefill's transient per chip: prefill_transient_bytes() (PREFILL_TRANSIENT_BYTES unless the profile tunes it)
    from PREFILL_TRANSIENT_FROM tokens on (a length that cannot be read counts as long), none below; and from the long-prefill
    tier's length on (long_prefill_tier, both of its flags set) the tier's bytes instead."""
    long_prompt = type(prompt_tokens) is not int or prompt_tokens >= PREFILL_TRANSIENT_FROM
    if not long_prompt:
        return 0
    tier = long_prefill_tier()
    if tier is not None and (type(prompt_tokens) is not int or prompt_tokens >= tier[0]):
        return tier[1]                # an unreadable length counts as long here too
    return prefill_transient_bytes()


def dram_need(prompt_tokens, reserve):
    """The DRAM per chip a fresh prompt needs before its prefill is scheduled (THE SPLIT's free term): the engine
    build's peak, the prefill's transient (prefill_transient) and the reserve."""
    return engine_build_peak() + prefill_transient(prompt_tokens) + _reserve(reserve)


def backstop_need(reserve):
    """What the post-prefill backstop requires (serving_request_factory.dram_backstop): the prefill has run,
    so the build's peak and the reserve."""
    return engine_build_peak() + _reserve(reserve)


def contiguous_need(reserve):
    """THE SPLIT's contiguous term: the largest free block must hold the largest single buffer and leave the
    reserve beside it (396.4 MB at the 256 MiB default). What the backstop and the coordinator's captures ask."""
    return _reserve(reserve) + largest_buffer_bytes()


def prefill_residue(prompt_tokens):
    """What the admission keeps free in the largest block for the prefill: its transient (prefill_transient), and
    never less than PREFILL_RESIDUE_BYTES, which a short prefill also takes from the block."""
    return max(prefill_transient(prompt_tokens), PREFILL_RESIDUE_BYTES)


def admission_contiguous_need(prompt_tokens, reserve):
    """The contiguous term the admission asks of a fresh prompt: contiguous_need plus the prefill's residue
    (prefill_residue), which the prefill takes from the largest block (the module docstring). 696.4 MB for a
    prompt of PREFILL_TRANSIENT_FROM tokens or more at the 256 MiB reserve, 496.4 MB below it; the backstop, after
    the prefill, asks contiguous_need alone (396.4 MB), so a long prompt reaches it with 300 MB of the block to
    spare and a short one with 100 MB, above the 86.9 MB the largest short prefill seen took."""
    return contiguous_need(reserve) + prefill_residue(prompt_tokens)


def split_short(free, largest, need, reserve, trace_largest=None, contiguous=None):
    """The terms of THE SPLIT (the module docstring) a reading is short of, in SPLIT_TERMS order; () when it fits.

    free and largest are the smallest chip's total free and largest free DRAM block; need what the operation needs
    with the reserve in it (dram_need, backstop_need, or a capture's estimate plus the reserve); trace_largest the
    smallest chip's largest free trace-region block, None where it cannot be read (that term then holds nothing);
    contiguous the block the operation needs (the admission's admission_contiguous_need), contiguous_need(reserve)
    when None. Gate v79 (run 36368363993) held 2.319 GB free beside a 1079.7 MB largest block against a 1568.4 MB
    need: free less STRANDED_BYTES is 2019 MB, the block holds a 120000-token prompt's 696.4 MB contiguous need, so
    it fits."""
    short = []
    if contiguous is None:
        contiguous = contiguous_need(reserve)
    elif type(contiguous) is not int or contiguous < 0:
        raise ValueError('A non-negative integer contiguous need in bytes is required, got %r' % (contiguous,))
    if free - STRANDED_BYTES < need:
        short.append('free')
    if largest < contiguous:
        short.append('contiguous')
    if trace_largest is not None and trace_largest < TRACE_CONTIGUOUS_BYTES:
        short.append('trace')
    return tuple(short)


def largest_free(pool):
    """(the smallest largest-free-DRAM-block over the chips, None), read through the pool's allocator
    statistics (serving_buffer_pool.ServingBufferPool.dram_statistics), or (None, why) when they cannot be read.
    THE SPLIT reads it beside the total free (dram_reading)."""
    statistics = getattr(pool, 'dram_statistics', None)
    if not callable(statistics):
        return None, 'pool without device statistics'
    try:
        report = statistics()
        if isinstance(report, dict):
            return None, str(report.get('unavailable', 'no statistics'))
        return min(int(chip['largest_free']) for chip in report), None
    except Exception as failure:
        return None, '%s: %s' % (type(failure).__name__, str(failure)[:120])


def trace_largest_free(pool):
    """(the smallest largest free trace-region block over the chips, None), read through the pool's
    trace_statistics (serving_buffer_pool.ServingBufferPool.trace_statistics, a BufferType.TRACE view), or
    (None, why) when the pool or this ttnn cannot read the region."""
    statistics = getattr(pool, 'trace_statistics', None)
    if not callable(statistics):
        return None, 'pool without trace statistics'
    try:
        report = statistics()
        if isinstance(report, dict):
            return None, str(report.get('unavailable', 'no statistics'))
        return min(int(chip['largest_free']) for chip in report), None
    except Exception as failure:
        return None, '%s: %s' % (type(failure).__name__, str(failure)[:120])


def dram_reading(pool):
    """(reading, None), or (None, why) when the pool's DRAM statistics cannot be read: what THE SPLIT reads, the
    smallest free and the smallest largest free block over the chips (free, largest_free), and the trace region's
    smallest largest free block (trace_largest_free, None when unread, trace_unread saying why)."""
    statistics = getattr(pool, 'dram_statistics', None)
    if not callable(statistics):
        return None, 'pool without device statistics'
    try:
        report = statistics()
        if isinstance(report, dict):
            return None, str(report.get('unavailable', 'no statistics'))
        reading = dict(free=min(int(chip['free']) for chip in report),
                       largest_free=min(int(chip['largest_free']) for chip in report))
    except Exception as failure:
        return None, '%s: %s' % (type(failure).__name__, str(failure)[:120])
    reading['trace_largest_free'], reading['trace_unread'] = trace_largest_free(pool)
    return reading, None


def dram_predicate(pool, reserve):
    """The predicate the worker registers, admits(prompt_tokens) -> (ok, detail): ok is False only when the
    pool's reading exists and is short of a term of THE SPLIT for dram_need, its contiguous term the admission's
    (admission_contiguous_need: the prefill transient in the block for a long prompt), detail['short'] naming them.
    An unavailable reading admits, and detail['unavailable'] says why (the attach refuses a pool without statistics
    under the flag, W7)."""
    _reserve(reserve)

    def admits(prompt_tokens):
        need = dram_need(prompt_tokens, reserve)
        reading, reason = dram_reading(pool)
        if reading is None:
            return True, dict(largest_free=None, need=need, unavailable=reason)
        short = split_short(reading['free'], reading['largest_free'], need, reserve, reading['trace_largest_free'],
                            contiguous=admission_contiguous_need(prompt_tokens, reserve))
        return not short, dict(largest_free=reading['largest_free'], need=need, free=reading['free'],
                               trace_largest_free=reading['trace_largest_free'], short=short)

    return admits


def register_dram_predicate(admits, modules=None):
    """Park `admits` under DRAM_KEY. Returns the callable that removes it again, only while it is still this
    one (the attach's scope calls it before the pool the predicate reads is closed)."""
    if not callable(admits):
        raise ValueError('A callable DRAM admission predicate is required')
    modules = sys.modules if modules is None else modules
    holder = types.ModuleType(DRAM_KEY)
    holder.admits = admits
    modules[DRAM_KEY] = holder

    def unregister():
        if modules.get(DRAM_KEY) is holder:
            del modules[DRAM_KEY]

    return unregister


def dram_admits(modules=None):
    """The registered predicate, or None. Absent or unreadable means no hold: the scheduler never
    hard-depends on the worker."""
    modules = sys.modules if modules is None else modules
    try:
        admits = getattr(modules.get(DRAM_KEY), 'admits', None)
    except BaseException:
        return None
    return admits if callable(admits) else None


def _blocked(check, request):
    """vLLM's own Scheduler._is_blocked_waiting_status on the request's status. A scheduler without it, or a
    check that raises, blocks nothing."""
    if not callable(check):
        return False
    try:
        return bool(check(getattr(request, 'status', None)))
    except Exception:
        return False


def admission_candidates(scheduler):
    """The waiting requests the base scheduler's waiting loop may admit this step, in the order it takes them.

    vLLM 0.25.1 FCFS takes `skipped_waiting or waiting` (scheduler.py:1867-1869) and reads the head with
    peek_request (:650), which is the queue's iteration order. A head with a blocked status (a structured-output
    grammar still compiling, remote KV, a paused stream: _is_blocked_waiting_status, :1853-1858) is popped and
    skipped unless that status is promoted in this very pass (:652-664), and the loop goes on to the next. So the
    request admitted is any of the blocked heads or the first request that is not blocked: all of them, up to
    and including that one. Empty when nothing waits."""
    check = getattr(scheduler, '_is_blocked_waiting_status', None)
    candidates = []
    for name in ('skipped_waiting', 'waiting'):
        queue = getattr(scheduler, name, None)
        if not queue:
            continue
        try:
            for request in queue:
                candidates.append(request)
                if not _blocked(check, request):
                    return candidates
        except Exception:
            continue
    return candidates


def prompt_tokens(request):
    """A vLLM Request's prompt length (num_prompt_tokens, else len(prompt_token_ids)), or None."""
    value = getattr(request, 'num_prompt_tokens', None)
    if type(value) is int:
        return value
    try:
        return len(request.prompt_token_ids)
    except Exception:
        return None


def binding_request(candidates):
    """The candidate whose need binds: dram_need and admission_contiguous_need both grow with the prompt, and a
    prompt whose length cannot be read counts as the longest. The first of equals; None when there is none."""
    if not candidates:
        return None

    def length(request):
        tokens = prompt_tokens(request)
        return float('inf') if tokens is None else tokens

    return max(candidates, key=length)


def finished_since_last_step(scheduler):
    """The requests vLLM finished since its last schedule() (Scheduler.finished_req_ids, scheduler.py:2108),
    sorted: the output of the step being scheduled names them (:1105), and the worker detaches them at the start
    of executing it, before any prefill (serving_lifecycle._execute). () when there are none or they cannot be
    read."""
    try:
        return tuple(sorted(getattr(scheduler, 'finished_req_ids', None) or (), key=str))
    except Exception:
        return ()


def _megabytes(value):
    return 'unread' if value is None else '%.1fMB' % (value / MEGABYTE)


def _terms(short):
    """THE SPLIT's short terms for a log line: 'free+contiguous', 'none', or 'unread' (a predicate that names none)."""
    if short is None:
        return 'unread'
    return '+'.join(short) if short else 'none'


def _note(state, log, key, template, *values):
    """Log `template` unless the last line noted was for this same key: one line per distinct state, keeping
    only the last key, so the state is bounded however many requests pass."""
    if state.get('dram_noted') != key:
        state['dram_noted'] = key
        log(template, *values)


def dram_hold(scheduler, decodes, state, log, modules=None, candidates=None):
    """Whether the fresh prompt this step would admit waits (S2 W6b, the module docstring). The wrapper asks only
    when a seat is free and nothing else holds the step. False when no predicate is registered, nothing waits,
    the predicate raises or reads nothing, the prompt fits, or no decode is left to wait for. True when it does
    not fit - silently deferred for one step when the reading still counts the engines of requests that finished
    since the last step. `state` keeps the held (request, decodes) and the last line noted, so each state logs
    once and the state stays bounded."""
    admits = dram_admits(modules)
    if admits is None:
        return False
    request = binding_request(admission_candidates(scheduler) if candidates is None else candidates)
    if request is None:
        return False
    request_id = getattr(request, 'request_id', None)
    prompt = prompt_tokens(request)
    try:
        ok, detail = admits(prompt)
        largest, need, unavailable = detail['largest_free'], detail['need'], detail.get('unavailable')
        # THE SPLIT's readings (dram_predicate); a predicate that names none logs them 'unread'.
        free, trace, short = detail.get('free'), detail.get('trace_largest_free'), detail.get('short')
    except Exception as failure:
        ok, largest, need, unavailable = True, None, None, '%s: %s' % (type(failure).__name__, str(failure)[:120])
        free = trace = short = None
    if largest is None:
        _note(state, log, ('unavailable', request_id), DRAM_UNAVAILABLE_LINE, request_id, unavailable or 'no reading')
        return False
    held = state.get('dram_held')
    if ok:
        if held is not None and held[0] == request_id:
            state['dram_held'] = None
            log(DRAM_RELEASED_LINE, prompt, _megabytes(largest), _megabytes(need), request_id, _megabytes(free),
                _megabytes(trace))
        else:
            # The reading a prompt was admitted on without a hold, once per request: under churn (M11) the heap and
            # the trace region no longer drain between cycles, so each admission's reading is what shows a ratchet.
            _note(state, log, ('fit', request_id), DRAM_FIT_LINE, prompt, _megabytes(largest), _megabytes(need),
                  request_id, decodes, _megabytes(free), _megabytes(trace))
        return False
    finished = finished_since_last_step(scheduler)
    if finished:
        # Stale: this step's output names them, and the worker frees their engines before any prefill in it. Wait
        # one step for a reading without them; this is not a hold, so neither the hold line nor the held state.
        _note(state, log, ('deferred', request_id, finished), DRAM_DEFERRED_LINE, prompt, _megabytes(largest),
              _megabytes(need), request_id, list(finished), _megabytes(free), _terms(short))
        return True
    if not decodes:
        state['dram_held'] = None
        _note(state, log, ('lifted', request_id), DRAM_LIFTED_LINE, prompt, _megabytes(largest), _megabytes(need),
              request_id, _megabytes(free), _terms(short))
        return False
    if held != (request_id, decodes):
        state['dram_held'] = (request_id, decodes)
        log(DRAM_HOLD_LINE, prompt, _megabytes(largest), _megabytes(need), request_id, decodes, _megabytes(free),
            _megabytes(trace), _terms(short))
    return True


KV_FLAG = 'QWEN_FAST_KV_RESERVATION'


def kv_reservation_requested(environ=None):
    """QWEN_FAST_KV_RESERVATION, strictly (serving_kv_reservation.enabled's rule, read here so that an image without that module
    still schedules as before while the flag is off): unset or '0' off, '1' on, anything else refused (ValueError)."""
    value = (os.environ if environ is None else environ).get(KV_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (KV_FLAG, value))
    return value == '1'


LEVERN_FLAG = 'QWEN_FAST_LEVER_N'


def levern_requested(environ=None):
    """QWEN_FAST_LEVER_N, strictly (levern_policy.enabled's rule, read here so that an image without that module still
    schedules as before while the flag is off): unset or '0' off, '1' on, anything else refused (ValueError)."""
    value = (os.environ if environ is None else environ).get(LEVERN_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (LEVERN_FLAG, value))
    return value == '1'


def decode_steps_per_admission(environ=None):
    """R, the decode-only steps owed after an admission (STEPS_FLAG): a whole number from 0 to MAX_STEPS in plain
    digits, 0 when unset. Anything else is a configuration error, not a silent 0."""
    value = (os.environ if environ is None else environ).get(STEPS_FLAG)
    if value is None:
        return 0
    if not value.isascii() or not value.isdigit() or int(value) > MAX_STEPS:
        raise ValueError('%s must be a whole number of steps from 0 to %d, got %r' % (STEPS_FLAG, MAX_STEPS, value))
    return int(value)


def carry_finished(scheduler, result, decodes, log, line=DRAM_CARRIED_LINE):
    """After a DRAM-held pass: put back the finished request ids it took when the plugin will discard it.

    schedule() handed them to `result` and started a new set (scheduler.py:1105, :1210). With a decode running and
    nothing scheduled, the plugin's default mode discards this pass and schedules a decode-only one (plugin
    bf77cd63 scheduler.py:136-143), whose output must name them or the worker never detaches them (the module
    docstring). Not when no decode runs, or in a forced mode: the plugin then returns this pass as it is, and it
    already names them. Returns the ids carried."""
    if not decodes or getattr(result, 'total_num_scheduled_tokens', None) != 0:
        return ()
    mode = getattr(scheduler, '_forced_mode', None)
    if mode is not None and getattr(mode, 'name', None) != 'DEFAULT':
        return ()
    taken = getattr(result, 'finished_req_ids', None)
    if not taken:
        return ()
    scheduler.finished_req_ids = set(taken) | set(getattr(scheduler, 'finished_req_ids', None) or ())
    carried = tuple(sorted(taken, key=str))
    log(line, list(carried))
    return carried


def waiting_capacity(saved_max, decodes, allowed):
    """The max_num_running_reqs the base scheduler's waiting loop sees: the graft's
    max(0, min(saved_max - decodes, allowed))."""
    return max(0, min(saved_max - decodes, allowed))


def _module_queue_factory(original):
    """create_request_queue(self.policy), by the name the plugin's scheduler.py binds it to (the
    graft's idiom). Falls back to vLLM's own when the method's module does not bind it."""
    create = getattr(original, '__globals__', {}).get('create_request_queue')
    if create is None:
        create = importlib.import_module('vllm.v1.core.sched.request_queue').create_request_queue
    return lambda scheduler: create(scheduler.policy)


def _enqueue(queue, request):
    """Put `request` in a request queue (vLLM's RequestQueue.add_request; the test fakes are lists)."""
    add = getattr(queue, 'add_request', None)
    if callable(add):
        add(request)
    else:
        queue.append(request)


def _dequeue_request(queue, request):
    """Take `request` out of a request queue (vLLM's RequestQueue.remove_request; the test fakes are lists). Absent is fine."""
    remover = getattr(queue, 'remove_request', None)
    try:
        if callable(remover):
            remover(request)
        else:
            queue.remove(request)
    except (ValueError, KeyError):
        pass


def new_state():
    return dict(live=False, seen=set(), dram_held=None, dram_noted=None, credit=0, asked=False)


def wrap(original, *, queue_factory, log, steps=None, state=None, kv=None, levern=None):
    """The wrapper installed as <class>._schedule_prefill_only around `original`. `steps` is R, the decode credit
    (decode_steps_per_admission() when None); `state` is shared with the schedule() wrapper (wrap_schedule); `kv` is
    serving_kv_reservation (QWEN_FAST_KV_RESERVATION=1: install() passes it), None for every profile that does not turn the
    reservation on, whose steps are then exactly what they were. `levern` is levern_scheduler.LevernRuntime
    (QWEN_FAST_LEVER_N=1: install() passes it): this step's request is capped at levern_policy.step_budget tokens, so vLLM
    splits a long prefill at the model's own 2,048-token boundaries; None (every profile without the flag) caps nothing."""
    steps = decode_steps_per_admission() if steps is None else steps
    if type(steps) is not int or steps < 0:
        raise ValueError('A non-negative integer decode credit is required, got %r' % (steps,))
    state = new_state() if state is None else state

    def _schedule_prefill_only(self):
        decodes = sum(1 for request in self.running if not request.is_prefill_chunk)
        partials = len(self.running) - decodes
        held = gate_held()
        # LEVER N, THE MERGED ROUTE (levern_scheduler.LevernRuntime.merged): which prefill this pass advances, which are hidden from it and which are
        # ended before it allocates anything. The pass then sees ONE prefill: the active partial, or the one fresh prompt as the only waiting request.
        plan = None
        if levern is not None and levern.merged is not None:
            plan = levern.plan_pass(self, decodes, held)
            partials = 1 if plan.active is not None else 0
            if plan.override_gate:
                held = None
        allowed, hide = admission(partials, held)
        state['asked'] = True
        # DECODE CREDIT: with a decode running and a credit owed, this call is a decode step, not an admission.
        credit_held = False
        if steps:
            if partials or not decodes:
                state['credit'] = 0
            elif state['credit'] and not hide:
                allowed, hide, credit_held = 0, True, True
        # Seats a fresh prompt may not take: the decoders and, under admission v2, every prefill the pass hides (parked and quarantined ones still hold one).
        seats_held = decodes + (len(plan.hidden) if plan is not None else 0)

        def candidates():
            if plan is not None and plan.fresh is not None:
                return [plan.fresh]
            return admission_candidates(self)

        def holds():
            # KV RESERVATION (QWEN_FAST_KV_RESERVATION=1, serving_kv_reservation): asked first, when this step would admit a fresh
            # prompt, so a prompt whose worst-case blocks do not fit what is unreserved waits as behind a held gate - before the
            # DRAM is even read. Never lifted: with nothing running the pool is empty and any request the contract admitted fits.
            # It reads `running` WHOLE (a hidden prefill still holds its worst-case blocks), so it is asked before the pass hides anything.
            kv_flag = (kv is not None and not hide and waiting_capacity(self.max_num_running_reqs, seats_held, allowed) > 0
                       and kv.hold(self, candidates(), decodes, state, log))
            # S2 W6b: asked only when this step would admit a fresh prompt - nothing in flight, the gate free and a
            # seat free (with every seat decoding the waiting loop admits nobody, and nothing is asked or logged).
            # When the prompt does not fit the DRAM left, it waits as behind a held gate.
            dram_flag = (not kv_flag and not hide and waiting_capacity(self.max_num_running_reqs, seats_held, allowed) > 0
                         and dram_hold(self, decodes, state, log, candidates=candidates()))
            return kv_flag, dram_flag

        kv_held, dram_held = holds()
        if (kv_held or dram_held) and plan is not None and plan.preempted is not None:
            # The short prompt would not fit: the long prefill it would have parked simply continues (no decode-only pass in its place).
            plan.demote()
            partials, held = 1, gate_held()
            allowed, hide = admission(partials, held)
            seats_held = decodes + len(plan.hidden)
            kv_held = dram_held = False
        if kv_held:
            allowed, hide = 0, True
        if dram_held:
            allowed, hide = 0, True
        if steps and state['credit'] and hide and decodes and not partials:
            # A decode-only pass follows whatever held this one (the credit, the gate or the DRAM hold): it pays a step.
            state['credit'] -= 1
            if credit_held:
                log(CREDIT_HOLD_LINE, state['credit'], decodes)
        if not state['live']:
            state['live'] = True
            log(LIVE + '{}', type(self).__name__)
        decision = (partials, decodes, held is not None, allowed, hide)
        if decision not in state['seen']:
            state['seen'].add(decision)
            log('[PINDIAG] one fresh prefill per step: partials={} decodes={} gate_held={} allowed={} hidden={}',
                *decision)
        saved_max = self.max_num_running_reqs
        saved_waiting = self.waiting
        saved_skipped = getattr(self, 'skipped_waiting', None)
        # LEVER N (QWEN_FAST_LEVER_N=1): this step advances one request, the partial in flight or the one fresh prompt the
        # waiting loop admits, by at most its step budget (a multiple of 2,048 ending before the final step, levern_policy).
        restore_cap = capped = None
        hidden_running = []
        single = None
        if levern is not None:
            if plan is None:
                restore_cap, target, _fresh, budget = levern.plan_cap(self, partials, allowed, hide, decodes)
            else:
                levern.commit_plan(plan)
                hidden_running = list(plan.hidden)
                target = plan.active if partials else (plan.fresh if allowed > 0 and not hide else None)
                if target is not None:
                    restore_cap, _target, _fresh, budget = levern.cap_for(self, target, target is plan.fresh, decodes)
                if target is not None and target is plan.fresh:
                    single = plan.fresh
            if restore_cap is not None:
                capped = (target, levern.computed)
        if hidden_running:
            # The pass must not see the prefills it parks or ends: they leave `running` for the call and come back after it.
            self.running = [request for request in self.running if all(request is not other for other in hidden_running)]
        # The plugin writes max(0, value - len(pure_decodes)) before calling the base scheduler, so
        # the value written here carries the decodes it is about to remove.
        self.max_num_running_reqs = min(saved_max, allowed + decodes)
        if hide or single is not None:
            self.waiting = queue_factory(self)
            if saved_skipped is not None:
                self.skipped_waiting = queue_factory(self)
            if single is not None:
                _enqueue(self.waiting, single)
        admitted_single = False
        try:
            result = original(self)
            if single is not None:
                admitted_single = any(getattr(value, 'req_id', None) == getattr(single, 'request_id', None)
                                      for value in getattr(result, 'scheduled_new_reqs', None) or ())
        finally:
            if restore_cap is not None:
                restore_cap()
            if hide or single is not None:
                # Anything the base scheduler put back (a preemption) is merged ahead of what
                # was hidden, exactly as _schedule_decode_only and the graft merge it.
                if single is not None:
                    left = [request for request in list(self.waiting) + list(getattr(self, 'skipped_waiting', None) or ())
                            if request is not single]
                    if left:
                        saved_waiting.prepend_requests(left)
                    if admitted_single:
                        for queue in (saved_waiting, saved_skipped):
                            if queue is not None:
                                _dequeue_request(queue, single)
                    if saved_skipped is not None:
                        self.skipped_waiting = saved_skipped
                else:
                    if self.waiting:
                        saved_waiting.prepend_requests(self.waiting)
                    if saved_skipped is not None:
                        if self.skipped_waiting:
                            saved_skipped.prepend_requests(self.skipped_waiting)
                        self.skipped_waiting = saved_skipped
                self.waiting = saved_waiting
            if hidden_running:
                self.running.extend(hidden_running)
            self.max_num_running_reqs = saved_max
        if capped is not None and not (dram_held or kv_held or credit_held):
            # The pass is about to run: it must be the one the cap was computed for (levern_scheduler.verify), or the engine stops here.
            levern.verify(result, capped)
        if dram_held:
            # S2 W6b: the pass the plugin is about to discard took the finished ids; its decode-only pass carries them.
            carry_finished(self, result, decodes, log)
        elif kv_held:
            carry_finished(self, result, decodes, log, kv.CARRIED_LINE)
        elif credit_held:
            carry_finished(self, result, decodes, log, CREDIT_CARRIED_LINE)
        elif steps and not hide and allowed and getattr(result, 'total_num_scheduled_tokens', 0):
            # An admission: the decodes it paused are owed R steps before the next prompt (the module docstring).
            state['credit'] = steps
            log(CREDIT_ARMED_LINE, steps, decodes, len(self.waiting) + len(getattr(self, 'skipped_waiting', None) or ()))
        return result

    setattr(_schedule_prefill_only, WRAPPED, True)
    _schedule_prefill_only.__wrapped__ = original
    return _schedule_prefill_only


def wrap_schedule(original, state):
    """The wrapper installed as <class>.schedule when the decode credit is on: a schedule() call that asked for no
    prefill (nothing pending, or a forced decode mode) was a decode-only step, and it drops the credit."""

    def schedule(self, *args, **kwargs):
        state['asked'] = False
        result = original(self, *args, **kwargs)
        if not state['asked']:
            state['credit'] = 0
        return result

    setattr(schedule, WRAPPED, True)
    schedule.__wrapped__ = original
    return schedule


def install(config, *, importer=importlib.import_module, log=None, queue_factory=None):
    """Wrap the configured scheduler class's _schedule_prefill_only. Returns its qualified name.

    Idempotent. A subclass of an already-wrapped class inherits the wrapper and is not wrapped
    again. Raises if the class cannot be resolved or has no _schedule_prefill_only (vLLM's stock
    Scheduler has none): the caller then serves as it did before this module existed."""
    log = _log if log is None else log
    cls = scheduler_class(config, importer)
    name = '%s.%s' % (cls.__module__, cls.__qualname__)
    original = getattr(cls, METHOD, None)
    if not callable(original):
        raise ValueError('%s has no %s: not a TT scheduler, so there is no prefill step to cap' % (name, METHOD))
    if getattr(original, WRAPPED, False):
        return name
    if queue_factory is None:
        queue_factory = _module_queue_factory(original)
    try:
        steps = decode_steps_per_admission()
    except ValueError as refusal:
        # A typo in the credit flag must never decide whether the one-fresh-prefill cap and the DRAM hold are installed.
        steps = 0
        log(CREDIT_REFUSED_LINE, os.environ.get(STEPS_FLAG), refusal)
    schedule = getattr(cls, 'schedule', None)
    if steps and not callable(schedule):
        raise ValueError('%s has no schedule: the decode credit cannot see its decode-only steps' % name)
    kv = None
    if kv_reservation_requested():
        # QWEN_FAST_KV_RESERVATION=1: the reservation rule (serving_kv_reservation, overlay only), on the vLLM it was proved
        # against; a vLLM that moved the pool's count refuses the install by name, before anything is wrapped.
        import serving_kv_reservation as kv

        problems = kv.install_check()
        if problems:
            raise ValueError('%s=1 cannot be installed on %s: %s' % (kv.FLAG, name, '; '.join(problems)))
    levern = None
    if levern_requested():
        # QWEN_FAST_LEVER_N=1: the cap and the alternation (levern_scheduler). Never optional: a class that cannot take
        # them fails the install by name, and the lifecycle refuses to attach without them, rather than serve a chunked
        # profile that never chunks. The per-admission decode credit is the same idea per admission and the two are
        # mutually exclusive.
        import levern_policy
        import levern_scheduler

        if steps:
            raise ValueError('%s and %s=1 are mutually exclusive: the Lever N alternation subsumes the per-admission '
                             'decode credit' % (STEPS_FLAG, levern_policy.FLAG))
        if not callable(schedule) or not callable(getattr(cls, '_schedule_decode_only', None)):
            raise ValueError('%s has no schedule or no _schedule_decode_only: Lever N cannot yield a step to the decoders'
                             % name)
        # Beside prefix reuse (QWEN_PREFIX_REUSE=1) it is the MERGED route's runtime: the peeked cap, the single-candidate pass, the pre-pass
        # quarantine, the short lane, the governor and the kill switch (levern_scheduler).
        merged = levern_policy.merged_config() if os.environ.get('QWEN_PREFIX_REUSE') == '1' else None
        levern = levern_scheduler.LevernRuntime(log=log, merged=merged)
    state = new_state()
    setattr(cls, METHOD, wrap(original, queue_factory=queue_factory, log=log, steps=steps, state=state, kv=kv, levern=levern))
    log(INSTALLED + '{}', name)
    if kv is not None:
        log(kv.INSTALLED_LINE, name)
    if steps:
        setattr(cls, 'schedule', wrap_schedule(schedule, state))
        log(CREDIT_INSTALLED_LINE, name, steps)
    if levern is not None:
        setattr(cls, 'schedule', levern_scheduler.wrap_schedule(schedule, levern))
        cfg = levern.cfg
        log(levern_policy.INSTALLED_LINE, name, cfg.step, cfg.solo, cfg.share, cfg.rounds if cfg.rounds else '-',
            cfg.max_rounds)
    return name
