"""Explicit attachment of the admitted combined runtime to a loaded TT worker."""

from contextlib import ExitStack, contextmanager
from functools import partial
import json
import os
import re
import time

from dflash_device import PreparedDraftWeights, pindiag
import memory_ledger
import trace_census
from serving_buffer_pool import ServingBufferPool, dram_line
from serving_cache_owner import ServingCacheOwner
from serving_fast_policy import STICKY_SESSIONS_FLAG, any_request_enabled, sticky_sessions_enabled, validate_fast_config
from serving_lifecycle import FastServingLifecycle
from serving_page_binding import VerifierPageBinding
from serving_request_factory import attach_source_check, from_prefill, sequential_captures
from serving_request_factory import extent_replay_enabled, register_dram_admission
from serving_runner_bridge import FastRunnerBridge


SKIP_BLOCK_STREAM_FLAG = 'QWEN_FAST_SKIP_BLOCK_STREAM'
SINGLE_GATEUP_FLAG = 'QWEN_FAST_SINGLE_GATEUP'
SINGLE_GATEUP_SHAPE = 'the single gate/up copy'
M3_SHAPE = 'the 64-row block'
C2_ANY_SHAPE = 'C2-any with no packed block'
PADDED_BLOCK_FLAG = 'QWEN_FAST_PADDED_BLOCK'
# Eight seats on TWO 64-row M3 blocks (default off): block A over pool slots (0..3), block B over (4..7), each exactly
# the qualified 4-user block, run back to back in one step. '1' (or unset) is today's one block at four requests; '2'
# is admitted at exactly eight scheduler requests and nowhere else; any other value is refused.
M3_BLOCKS_FLAG = 'QWEN_FAST_M3_BLOCKS'
M3_BLOCKS_USERS = 8
M3_BLOCKS_MARKER = '[PINDIAG] M3 blocks={} over pool slots {} (QWEN_FAST_M3_BLOCKS={}); each block is the qualified 4-user 64-row block'
CAPTURE_PROGRAMS_MARKER = '[PINDIAG] packed blocks capture block={} programs={}->{}'
CLOSE_FAILED_MARKER = '[PINDIAG] a sibling block did not close without the fence: {}'
CAPTURE_POSITION_FLAG = 'QWEN_FAST_PACKED_CAPTURE_POSITION'
CAPTURE_POSITION_MARKER = '[PINDIAG] packed capture position override='
EXTENT_REPLAY_FLAG = 'QWEN_FAST_EXTENT_REPLAY'
# Sticky sessions (QWEN_FAST_STICKY_SESSIONS=1 only): one line per admitted request's engine build,
# '<marker><request> ms=<build> frontier=<R, 0 cold> prompt=<P>'.
STICKY_ENGINE_MARKER = '[PINDIAG] sticky engine built req='


def m3_blocks(environ=None):
    """QWEN_FAST_M3_BLOCKS, strictly: unset or '1' is one M3 block (today's), '2' is two (eight seats), and anything
    else - an empty value included - is a configuration error naming the flag, refused before anything is built."""
    value = (os.environ if environ is None else environ).get(M3_BLOCKS_FLAG, '1')
    if value not in ('1', '2'):
        raise ValueError('%s must be 1 or 2, got %r' % (M3_BLOCKS_FLAG, value))
    return int(value)


def m3_blocks_for(policy, environ=None):
    """How many M3 blocks this attach builds (1 or 2), refusing - ValueError naming QWEN_FAST_M3_BLOCKS - a malformed
    value and the value 2 at any scheduler request count but eight: two blocks are the eight-seat shape, and at four
    requests the flag would be a second, silent way to ask for something the four-seat attach already is."""
    environ = os.environ if environ is None else environ
    blocks = m3_blocks(environ)
    policy = policy() if callable(policy) else policy
    if blocks == 2 and policy['scheduler_requests'] != M3_BLOCKS_USERS:
        raise ValueError('%s=2 builds two 4-user M3 blocks and is admitted at exactly %d scheduler requests, not %s'
                         % (M3_BLOCKS_FLAG, M3_BLOCKS_USERS, policy['scheduler_requests']))
    return blocks


def m3_shape(policy, environ=None):
    """(met, description): whether this is the 64-row M3 block - four scheduler requests,
    QWEN_FAST_FOUR_AS_TWO=0 and QWEN_FAST_PACKED_STEP=1 - or, under QWEN_FAST_M3_BLOCKS=2, the two M3 blocks
    of eight scheduler requests (each block exactly that same 4-user block), and a short description of the
    shape actually configured, for a marker (it names the block count only when it is not one). `policy` is
    validate_fast_config's dict, or a zero-argument callable returning it."""
    environ = os.environ if environ is None else environ
    policy = policy() if callable(policy) else policy
    users, four_as_two, packed = (policy['scheduler_requests'], environ.get('QWEN_FAST_FOUR_AS_TWO', 'unset'),
                                  environ.get('QWEN_FAST_PACKED_STEP', 'unset'))
    blocks = m3_blocks(environ)
    met = users == (4 if blocks == 1 else M3_BLOCKS_USERS) and four_as_two == '0' and packed == '1'
    return met, 'users=%s FOUR_AS_TWO=%s PACKED_STEP=%s%s' % (users, four_as_two, packed,
                                                             '' if blocks == 1 else ' M3_BLOCKS=%d' % blocks)


def c2_any_without_block(environ=None):
    """Whether this is C2-any with no packed block: QWEN_FAST_ANY_REQUEST=1 and QWEN_FAST_PACKED_STEP
    not 1 (the c2 profile). attach_combined_runtime then builds no block and caps every per-request
    engine at the sequential widths (1, 2, 4), so, as at the 64-row block, no target MLP ever sees 16
    rows and FusedT16Arm and the block stream behind it are never read."""
    environ = os.environ if environ is None else environ
    return environ.get('QWEN_FAST_ANY_REQUEST') == '1' and environ.get('QWEN_FAST_PACKED_STEP', 'unset') != '1'


def register_reader_reason(policy, environ=None):
    """Why the serial block stream may be replaced by the register-epilogue reader over the
    native w_gate_up (combined_runtime's block_stream=None branch), as a (shape, detail)
    pair for the startup marker - or None, the default, when the stream is required.

    Both flags are default-off and both are admitted at the 64-row M3 block ONLY (four
    scheduler requests, QWEN_FAST_FOUR_AS_TWO=0, QWEN_FAST_PACKED_STEP=1), the one shape
    where no target MLP ever sees 16 rows - the per-request engines capture 1, 2 and 4 rows
    and the block 64 - so FusedT16Arm (fused_t16_scope.py:49-51) and the stream behind it
    are never read:
    - QWEN_FAST_SINGLE_GATEUP=1: the model builds no w_gate_up, so there is nothing to
      stream (and combined_runtime installs no FusedT16Arm either). At any OTHER shape it
      is REFUSED (ValueError): there the arm serves every 16-row verify, and replacing its
      BF4 register-epilogue projection with the native w1/w3 path changes the target's
      arithmetic, so committed tokens could change.
    - QWEN_FAST_SKIP_BLOCK_STREAM=1: elsewhere None, the stream is built as always (and
      serving_startup.weight_streams says so in a marker).

    The same holds for C2-any with no packed block (c2_any_without_block, the c2 profile): its
    engines capture 1, 2 and 4 rows and nothing else runs a verify, so both flags are admitted there
    too. Run 36219636175 refused SINGLE_GATEUP there, and turning it off would build w_gate_up and
    its stream again, about as much DRAM as dropping the block freed.

    The policy is evaluated only once either flag is set."""
    environ = os.environ if environ is None else environ
    single = environ.get(SINGLE_GATEUP_FLAG) == '1'
    if not single and environ.get(SKIP_BLOCK_STREAM_FLAG) != '1':
        return None
    met, shape = m3_shape(policy, environ)
    if not met and c2_any_without_block(environ):
        if single:
            return (SINGLE_GATEUP_SHAPE, 'QWEN_FAST_SINGLE_GATEUP=1 builds no w_gate_up to stream')
        return (C2_ANY_SHAPE, 'register-epilogue reader on native w_gate_up')
    if single:
        if not met:
            raise ValueError('QWEN_FAST_SINGLE_GATEUP=1 is admitted only at the 64-row M3 block '
                             '(users=4 FOUR_AS_TWO=0 PACKED_STEP=1; users=8 with QWEN_FAST_M3_BLOCKS=2), not ' + shape)
        return (SINGLE_GATEUP_SHAPE, 'QWEN_FAST_SINGLE_GATEUP=1 builds no w_gate_up to stream')
    if not met:
        return None
    return (M3_SHAPE, 'register-epilogue reader on native w_gate_up')


def padded_block_admission(policy, environ=None):
    """QWEN_FAST_PADDED_BLOCK (variable-user packed rounds M2, default off): None while the flag
    is off - the policy is not even read - else the fewest live users a round of the block may
    serve (QWEN_FAST_PADDED_BLOCK_MIN_USERS, default 2; packed_verifier.padded_block_min_users),
    which the 64-row block is then built with. Admitted at the 64-row M3 block ONLY, by the same
    rule as QWEN_FAST_SINGLE_GATEUP (m3_shape): every other shape is REFUSED (ValueError). The
    padded round is the M3 trace with idle segments on page 0 - measured exact at that trace
    (v188) and nowhere else - and two idle segments are all page 0's two tile rows hold."""
    environ = os.environ if environ is None else environ
    if environ.get(PADDED_BLOCK_FLAG, '0') == '0':
        return None
    from packed_verifier import padded_block_min_users

    minimum = padded_block_min_users(environ)   # refuses any value but '1' here
    met, shape = m3_shape(policy, environ)
    if not met:
        raise ValueError('QWEN_FAST_PADDED_BLOCK=1 is admitted only at the 64-row M3 block '
                         '(users=4 FOUR_AS_TWO=0 PACKED_STEP=1; users=8 with QWEN_FAST_M3_BLOCKS=2), not ' + shape)
    return minimum


WARM_MARKER = '[PINDIAG] four-card eager prefill warmed before the packed traces'
PREFILL_PROGRAMS_MARKER = '[PINDIAG] four-card prefill programs='
ENGINE_PROGRAMS_MARKER = '[PINDIAG] four-card engine programs='
WARM_SLOT_TOKENS = 64
WARM_LONG_PROMPTS = (2048 + 64, 4096)


def program_count(model):
    """The mesh's program-cache entry count, or None where the model has no mesh that reports it (a test double)."""
    count = getattr(getattr(model, 'mesh_device', None), 'num_program_cache_entries', None)
    if not callable(count):
        return None
    try:
        return int(count())
    except Exception:
        return None


def prefill_warm_before_traces(runner, model, scheduler_requests, environ=None, operations=None):
    """Four cards only (QWEN_FAST_TP not '2'): run the model's eager-prefill warmup BEFORE the packed blocks capture their
    traces, and return True when it did. Two cards return False before touching anything, so the pair is unchanged.

    The fast path never runs the model's own prefill warmup (qwen36_model.prefill_paged_slots: 'call the batched warmup
    first ... pre-warmed programs, no post-park compile'): serving runs trace_mode=decode_only, so warmup_model_prefill returns
    at once, and serving_startup.warmup replaces the plugin's warmup, so G1's _qwen_prefix_warm_eager (#48536) never runs either.
    The first eager prefill of the process then compiled the whole prefill program set AFTER the attach-time packed traces
    were captured, and every later packed replay could overwrite what it left in the holes the capture freed: v172 came out
    as EOS or garbage (the GDN scratch's zero sources), and with the scratch moved before the traces (110e919b) v188 and
    v159 hung in the first prefill chunk whose programs had compiled after the capture.

    In order: (a) the persistent B=1 GDN scratch, claimed in the ledger; (b) a page table at the width the plugin gives every
    prefill (runner.max_num_blocks_per_req; SDPA pads the table to a multiple of 32 and is keyed on that shape), refusing the
    attach when the runner has no such width; (c) the masked-bucket warmup under the bound scratch, unbound in a finally;
    (d) one served-entry prefill per scheduler slot, plus a 2048+64 and a 4096 prompt on slot 0, so the chunk loop, the
    write-slot programs and the logits run through the entry the requests use. A model without the scratch method (a test
    double) is left alone."""
    environ = os.environ if environ is None else environ
    if environ.get('QWEN_FAST_TP', '2') == '2':
        return False
    ensure = getattr(model, '_ensure_gdn_prefill_scratch', None)
    if ensure is None:
        return False
    width = getattr(runner, 'max_num_blocks_per_req', None)
    if type(width) is not int or width <= 0:
        raise ValueError('The four-card eager prefill warm needs the plugin runner\'s max_num_blocks_per_req (the width '
                         'every prefill page table has), not %r' % (width,))
    import torch

    began, programs_before = time.perf_counter(), program_count(model)
    ensure()
    # The ledger (a no-op unless QWEN_FAST_MEMORY_LEDGER=1) claims the scratch here, under its own item, so the first
    # prefill's model_after_prefill - c2_smoke_check's four-card rule - reads only what that prefill allocated.
    memory_ledger.record('P5', point='prefill_scratch', model_prefill_scratch=getattr(model, '_gdn_prefill_scratch', None))
    page_table = torch.arange(width, dtype=torch.int32).reshape(1, width)
    previous = model._bind_gdn_prefill_scratch()
    try:
        model.warmup_prefill_masked_buckets(page_table)
    finally:
        model._unbind_gdn_prefill_scratch(previous)
    slots = tuple(range(int(scheduler_requests)))
    for slot in slots:
        model.prefill_paged_slots([torch.zeros((1, WARM_SLOT_TOKENS), dtype=torch.int64)], page_table, [slot],
                                  valid_lens=[WARM_SLOT_TOKENS])
    for length in WARM_LONG_PROMPTS:
        model.prefill_paged_slots([torch.zeros((1, length), dtype=torch.int64)], page_table, [0], valid_lens=[length])
    if operations is not None:
        operations.synchronize_device(model.mesh_device)
    pindiag(WARM_MARKER + ': page_table_blocks={} slots={} long_prompts={} programs={}->{} ms={:.0f}', width, len(slots),
            list(WARM_LONG_PROMPTS), programs_before, program_count(model), (time.perf_counter() - began) * 1000.0)
    memory_ledger.record('P5', point='prefill_warm')
    return True


def prefill_tripwire(model, capture_factory, bridge_factory, environ=None):
    """(capture_factory, bridge_factory), wrapped at four cards to log one line per prefill segment (one per prompt
    without chunked prefill): '[PINDIAG] four-card prefill programs=A->B window=W prompt=P', the program-cache count
    before and after the segment's device work and how many of those entries the window snapshot compiled
    (dflash_prefill_window: keyed on the prompt's geometry, so it cannot be warmed). c2_smoke_check fails a prefill with
    B-A-W above zero: a program compiled after the packed traces were captured, the #48536 sequence.

    Counted around the capture's segment (PrefillWindowCapture.capture runs its own segment), not at the bridge: a
    request that ends at its first token - the platform's max_tokens=1 warmup, the first prefill after the attach, and
    an instant EOS - is never bridged, and must be counted all the same. The bridge (the engine build) logs its own
    '[PINDIAG] four-card engine programs=A->B req=R' line: those programs compile after the capture by design, at both
    widths, so the line is a fact to read and not a rule. The pair, and a model that cannot report its cache, get the
    factories back unwrapped."""
    environ = os.environ if environ is None else environ
    if environ.get('QWEN_FAST_TP', '2') == '2' or program_count(model) is None:
        return capture_factory, bridge_factory
    import dflash_prefill_window

    dflash_prefill_window.set_program_counter(lambda: program_count(model))

    def counted_segment(segment, position):
        @contextmanager
        def counted():
            dflash_prefill_window.window_programs(reset=True)
            before = program_count(model)
            try:
                # QWEN_FAST_STALL_DEADLINE_S / QWEN_FAST_TRACE_CENSUS_GRAPH: the segment is a watched scope and, with the graph
                # census on, its allocations are held against every live trace (eager work, never inside a trace capture).
                with trace_census.eager_segment('prefill segment prompt=%s' % position), segment() as value:
                    yield value
            finally:
                pindiag(PREFILL_PROGRAMS_MARKER + '{}->{} window={} prompt={}', before, program_count(model),
                        dflash_prefill_window.window_programs(), position)
        return counted

    def counted_capture(position, start=0):
        capture = capture_factory(position, start)
        segment = getattr(capture, 'segment', None)
        if callable(segment):
            # An instance attribute: capture() and the lifecycle's continuations both enter it through self.segment.
            capture.segment = counted_segment(segment, position)
        return capture

    def counted_bridge(state, capture):
        before = program_count(model)
        try:
            return bridge_factory(state, capture)
        finally:
            pindiag(ENGINE_PROGRAMS_MARKER + '{}->{} req={}', before, program_count(model),
                    str(getattr(state, 'req_id', None))[:48])

    return counted_capture, counted_bridge


def packed_capture_position(environ=None):
    """QWEN_FAST_PACKED_CAPTURE_POSITION (S2 design W3, B1; GATE ONLY, unset by default): the
    position every packed block of this attach captures at, or None for the block's own default
    (packed_verifier: C - 256, the last native chunk family). G3b sets it per arm to run the same
    served path below C - the flag-off block captured in family F (4352, 16640: families the pinned
    validate_ticket admits and the pool already holds tables for) against the extent block at the
    same position - and nothing else may: W8 pins that no traffic profile and no image ENV sets it.
    A strict decimal integer; anything else (an empty value included) is a configuration error,
    refused before anything is built. The range is the block's own to refuse."""
    environ = os.environ if environ is None else environ
    text = environ.get(CAPTURE_POSITION_FLAG)
    if text is None:
        return None
    if type(text) is not str or re.fullmatch('0|[1-9][0-9]*', text) is None:
        raise ValueError('%s must be a decimal integer position, got %r' % (CAPTURE_POSITION_FLAG, text))
    return int(text)


def extent_replay_requested(environ=None):
    """QWEN_FAST_EXTENT_REPLAY (S2 C2-packed-any, design W2/W3; default off), strictly: unset or '0'
    is off, '1' is on, and anything else is a configuration error, refused before anything is built.
    On, the attach builds the pool WITH its extent storage (ServingBufferPool extent_replay=True) and
    then requires every packed block over it to be the extent block; the block keys on that storage
    alone (PackedVerifierEngine.extent), so without this the flag would build today's family block,
    which packs only [131072, 131312], and nothing at attach would say so. Whether the extent path
    is admitted at all is packed_any_admission's (design W7); this makes the flag reach the storage
    and proves the block took it."""
    value = (os.environ if environ is None else environ).get(EXTENT_REPLAY_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (EXTENT_REPLAY_FLAG, value))
    return value == '1'


def complete_blocks_two_phase(blocks, model=None, log=None):
    """QWEN_FAST_M3_BLOCKS=2 (A1c): finish the construction of blocks built with defer_capture=True. Each block has
    already allocated its persistent state (the initial snapshot, checkpoints, taps) and the first phase here runs
    every block's warm forward and builds every fixture (extent words, masks, retained storage); only then does any
    block capture a trace, and the publication warm and the reseed come after the LAST capture. The rule is
    packed_verifier's CONSTRUCTION ORDER: a buffer a block keeps across rounds that is allocated after a capture
    lands in the holes that capture freed, and every replay overwrites it. Built one after the other, block B's
    fixture inputs, taps, checkpoints and extent words would have been exactly that.

    A log line per capture gives the program-cache count before and after each block: block B compiles zero programs
    after block A's capture (the probe window reads it)."""
    log = pindiag if log is None else log
    try:
        for block in blocks:
            block.warm_and_fixture()
        for index, block in enumerate(blocks):
            before = program_count(model) if model is not None else None
            block.capture_traces()
            log(CAPTURE_PROGRAMS_MARKER, index, before, program_count(model) if model is not None else None)
        for index, block in enumerate(blocks):
            if index:
                block.warm_publication = False      # block A's warm covered the same plan on the shared program cache
            block.finish_construction()
    except BaseException:
        # The failing block closed itself without the device fence (it may be hung); a sibling still under construction
        # would be closed by the attach scope WITH the fence and block there on the same hung device. Close it the same
        # way, here, then let the original failure through.
        for block in blocks:
            try:
                block.close(wait=False)
            except BaseException as error:
                log(CLOSE_FAILED_MARKER, repr(error)[:200])
        raise


@contextmanager
def attach_combined_runtime(worker, operations, *, directory, runtime_root, fixtures,
                            native_attention_evidence, block_stream, kv_publication_evidence,
                            eos_ids, cancelled):
    import torch
    from dflash_combined_request import combined_runtime
    from dflash_prefill_window import PrefillWindowCapture
    from dflash_request_runtime import TARGET_TAPS
    from gdn_snapshot import ActiveSnapshot
    from models.common.sampling.generator import SamplingGenerator
    from models.tt_transformers.tt.ccl import TT_CCL
    from sampling_link_policy import sampler_links

    policy = validate_fast_config(worker.vllm_config)
    # QWEN_FAST_M3_BLOCKS (default 1; 2 only at eight scheduler requests): read strictly before anything is built, so a
    # malformed value, or 2 at any other request count, is refused here by name.
    m3_blocks_count = m3_blocks_for(policy)
    # Measurement-only, env-gated admission of one named grafted binary (K64 kernel
    # graft): inert unless QWEN_FAST_RUNTIME_BINARY_SHA256 is set, and it neither
    # touches the hash-pinned sources nor lowers the pin itself. Must run before
    # combined_runtime() is entered below, since dflash_t16_native_scope.admit runs inside it.
    from runtime_binary_override import install as override_runtime_binary
    binary_record = override_runtime_binary(runtime_root, log=pindiag)
    # Measurement-only, env-gated admission of the device profiler into the mandatory
    # block-stream recipe (attribution only, never a throughput claim): inert unless
    # QWEN_FAST_PROFILED_BLOCK_STREAM=1, and edits no recipe file. Must run before
    # combined_runtime() is entered below, since scoped_block_stream's require_hardware
    # check runs inside it.
    from profiled_block_stream_override import install as admit_profiled_block_stream
    admit_profiled_block_stream(log=pindiag)
    # The serial weight stream stays mandatory except at the shape register_reader_reason
    # admits (each flag default-off), where the register-epilogue reader over the native
    # w_gate_up replaces it. Read before anything is built: QWEN_FAST_SINGLE_GATEUP=1 at any
    # other shape is refused here, whatever recipe startup handed over.
    reader = register_reader_reason(policy)
    # QWEN_FAST_PADDED_BLOCK (default off): the 64-row block's fewest live users per round, or
    # None. Refused here at any other shape, before anything is built, like the single copy.
    padded_min_users = padded_block_admission(policy)
    # QWEN_FAST_SOLO_LANE (D0, default off; strictly '0' or '1'): the one-user 16-row block beside M3, gate only. Refused
    # here, before anything is built, at any shape, width or flag set serving_solo_lane does not admit; off, nothing in
    # this attach differs from before it existed (not even the import).
    solo_lane = None
    if os.environ.get('QWEN_FAST_SOLO_LANE', '0') != '0':
        import serving_solo_lane

        solo_lane = serving_solo_lane.solo_lane_admission(m3_shape(policy), log=pindiag)
    # QWEN_FAST_LANE (default off; strictly '0' or '1'): one fast lane beside standard lanes, gate only. It rides on the solo
    # lane (a solo round IS the fast user's extra round), so the solo admission above must have passed. Off, not even the import.
    lane_config = None
    if os.environ.get('QWEN_FAST_LANE', '0') != '0':
        import serving_fast_lane

        lane_config = serving_fast_lane.lane_admission(solo_lane, seats=policy['scheduler_requests'], log=pindiag)
    # QWEN_FAST_PACKED_CAPTURE_POSITION (S2 G3b, gate only, default unset): parsed here, before
    # anything is built; each block refuses a position its capacity cannot capture at.
    capture_position = packed_capture_position()
    # S2 C2-packed-any (QWEN_FAST_EXTENT_REPLAY, default off; strictly '0' or '1', read here before anything
    # is built): packed rounds at any position through the K64j extent readers. Admitted here or nowhere,
    # host only and before anything is built, so a refusal builds nothing: the M3 shape, C2-any, eight-row
    # groups, the tree-scratch precondition, the modes, the K64j binary and kernels, and the pinned hardware
    # evidence (packed_any_admission, design W7). Off (unset or '0'), nothing here runs, not even the import
    # - the module ships in the C2 overlay only, as serving_lifecycle's lazy quarantine import does - and
    # the attach is today's.
    extent_replay = extent_replay_requested()
    if extent_replay:
        import packed_any_admission

        packed_any_admission.extent_replay_enabled()   # strictly '1' from here: any other value is refused
        packed_any_admission.admit(runtime_root, m3=m3_shape(policy), binary_record=binary_record, log=pindiag)
    if (native_attention_evidence is None or kv_publication_evidence is None
            or (block_stream is None and reader is None)
            or (block_stream is not None and 'pipeline_evidence' in block_stream)):
        raise ValueError('Native T16, direct KV publication and serial weight-stream recipe required')
    runner = worker.model_runner
    model = runner.model.model[0]
    # One draft history pair per scheduler slot, allocated FIRST: no request exists
    # yet, so no request trace does either. A request's verify trace bakes the
    # addresses of the intermediates its capture frees; the next request's
    # persistent history, allocated afterwards, lands in those holes and every
    # replay of the first trace overwrites it - per chip, since each chip's
    # allocator reuses independently (runs 35477522469, 35479238722; precedent
    # docs/experiment-execution.md, feature_prefix.py). Nothing earlier in this
    # attachment captures a trace, and the plugin's own warmup is replaced by
    # serving_startup.warmup, so this is the earliest point the fast path controls.
    # (Under QWEN_PREFIX_REUSE=1 that warmup first runs the prefix-reuse model graft's
    # warm, serving_startup.prefix_warm: a transient restore round trip through the
    # model's persistent B=1 prefill scratch, and no trace.)
    # Registered first so it closes last, after the lifecycle has closed every
    # device that borrows from it.
    scopes = ExitStack()
    try:
        # The helpers allocate nothing; the pool needs them to allocate every request's
        # GDN snapshot sets (initial, carry, one checkpoint set per capture) in the
        # exact shape the engine's own allocate() would have, before any trace.
        layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
        if len(layers) != 48:
            raise ValueError('All forty-eight native GDN layers required')
        helpers = [ActiveSnapshot(layer, operations, direct=True) for layer in layers]

        # The page table has to cover the admitted context. It was a fixed 68 pages,
        # which is 68 x 64 = 4352 tokens, while this image's T16 gate demands position
        # 32768 - so the fast path could not decode at ANY context, at one user or two
        # (runs 35472072127, 35473307362). 68 stays the floor because
        # ServingCacheOwner requires at least that many physical pages. The pool's
        # fixture inputs are this wide too: a request's table must match exactly.
        page_width = max(68, -(-int(worker.vllm_config.model_config.max_model_len) // 64))

        def rope(positions):
            # The native rotary construction ModelBatch uses, so the pooled tables are
            # the same kind of tensor the fixture would have built.
            from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode
            return rot_mats_decode(model.mesh_device, model.args.rope_head_dim,
                                   model.args.max_seq_len, model.args.rope_theta, positions)

        # Widths with multiplicity for the geometry from_prefill gives every engine:
        # sixteen verifier rows, the output budget, a T16 cap. A request whose plan
        # needs more is refused at admission by the engine, never allocated late.
        from verifier_engine import capture_bucket_rows

        # The packed block's shape, when QWEN_FAST_PACKED_STEP=1 asks for one, follows the
        # scheduler's request count (packed_shapes.serving_shape): two requests take the
        # 32-row M1 block, four the 64-row M3 block, and any other count builds no block,
        # so the sequential step serves the rounds and the stage line says so. Chosen
        # before the pool, which lends the block(s) their per-user replay page tables for
        # exactly that shape; with the switch unset the pool is built as it always was.
        #
        # Four requests are the one count with a choice: QWEN_FAST_FOUR_AS_TWO (default ON
        # at four requests) builds TWO 32-row M1 blocks instead of the single 64-row M3
        # block - block A over pool slots (0, 1), block B over (2, 3) - because two proven,
        # workaround-free 32-row rounds (M1b, 109 ms measured, run 35500352729) cost far
        # less than the 64-row round's two-tile workarounds (1453 ms, run 35544598063) for
        # the same four users. QWEN_FAST_FOUR_AS_TWO=0 keeps the single M3 block.
        packed_requested = os.environ.get('QWEN_FAST_PACKED_STEP') == '1'
        four_as_two = False
        # QWEN_FAST_M3_BLOCKS=2: two 64-row M3 blocks over pool slots (0..3) and (4..7), the same multi-block path.
        m3_blocks_two = False
        packed_shapes = ()
        if m3_blocks_count == 2 and not packed_requested:
            raise ValueError('%s=2 builds packed blocks and needs QWEN_FAST_PACKED_STEP=1, not %s'
                             % (M3_BLOCKS_FLAG, os.environ.get('QWEN_FAST_PACKED_STEP', 'unset')))
        if packed_requested:
            from packed_shapes import m1_shape, m3_shape as m3_block_shape, serving_shape

            if m3_blocks_count == 2:
                if os.environ.get('QWEN_FAST_FOUR_AS_TWO', '1') != '0':
                    raise ValueError('%s=2 needs QWEN_FAST_FOUR_AS_TWO=0, not %s'
                                     % (M3_BLOCKS_FLAG, os.environ.get('QWEN_FAST_FOUR_AS_TWO', 'unset')))
                m3_blocks_two = True
                packed_shapes = (m3_block_shape(page_width), m3_block_shape(page_width))
            elif policy['scheduler_requests'] == 4 and os.environ.get('QWEN_FAST_FOUR_AS_TWO', '1') != '0':
                four_as_two = True
                packed_shapes = (m1_shape(page_width), m1_shape(page_width))
            else:
                shape = serving_shape(policy['scheduler_requests'], page_width)
                if shape is None:
                    pindiag('[PINDIAG] QWEN_FAST_PACKED_STEP=1 builds no packed block for {} scheduler requests '
                            '(two take the 32-row M1 block, four the 64-row M3 block); the sequential step '
                            'serves the rounds', policy['scheduler_requests'])
                else:
                    packed_shapes = (shape,)
        # S2 (QWEN_FAST_EXTENT_REPLAY=1) serves its rounds through the packed block alone - the pool lends
        # the block its extent storage and the block keys on it - so with no block to build the flag would
        # build nothing and every round would run sequentially. The admission's M3 shape check already makes
        # this unreachable in serving; refused here too, before the pool, the first allocation.
        if extent_replay and not packed_shapes:
            raise ValueError('%s=1 serves its rounds through the packed block, and this attach builds none '
                             '(QWEN_FAST_PACKED_STEP=%s, %d scheduler requests)%s'
                             % (EXTENT_REPLAY_FLAG, os.environ.get('QWEN_FAST_PACKED_STEP', 'unset'),
                                policy['scheduler_requests'],
                                # Eight requests are the two-block shape: say which flag builds it.
                                '; eight requests need %s=2 (two 64-row M3 blocks)' % M3_BLOCKS_FLAG
                                if policy['scheduler_requests'] == M3_BLOCKS_USERS else ''))
        # The per-request engines' capture widths, and the pool's buckets that hold them:
        # the full T16 set by default and beside the 32-row block, the sequential widths
        # (1, 2, 4) beside the 64-row block, whose four engines' 8- and 16-row captures do
        # not fit (packed_shapes.sequential_capture_rows; run 35509307389, image v52, OOM
        # in the first engine with 32.9 of 33.1 GB per chip allocated). The rounds the
        # block serves are drafted at its width (serving_packed_step.proposal_rows); the
        # survivors decode sequentially at four rows per round.
        #
        # Two 32-row blocks retain the SAME total state as the one 64-row block - every
        # per-segment term in packed_verifier.py scales from users x rows_per_user, 4 x 16
        # either way, not from one block's own width - so nothing here has re-measured a
        # device with two 32-row blocks built beside four untrimmed engines and found the
        # headroom the single 64-row measurement did not have (run 35509307389: 32.9 of
        # 33.1 GB allocated with ONE untrimmed engine's captures still pending). Trimming
        # is the safe assumption for four_as_two too: `sequential_capture_rows(m1_shape)`
        # would answer 16 (correctly - a LONE 32-row block never needed the trim), so the
        # four-user, two-block case is decided explicitly here instead, by total rows
        # rather than by either shape alone.
        from packed_shapes import M3_SEQUENTIAL_CAPTURE_ROWS, sequential_capture_rows

        if four_as_two or m3_blocks_two:
            capture_rows = M3_SEQUENTIAL_CAPTURE_ROWS
        else:
            capture_rows = sequential_capture_rows(packed_shapes[0] if packed_shapes else None)
        # C2-any with no packed block at all (the c2 profile sets QWEN_FAST_PACKED_STEP=0): its
        # engines drop replay attention and the T16 gate, so they must capture the sequential
        # widths whatever the block would have asked. The block could never engage under c2 - it
        # needs every live user inside [131072, 131328), and c2 caps prompts at 123136 - yet it held
        # 3.84 GB per chip, and run 36218104858 died building the fourth engine with 0.79 GB free
        # (engines cost 0.80 GB at one proposal bucket, 1.35 GB at four).
        no_block_any_request = any_request_enabled() and not packed_shapes
        if no_block_any_request:
            capture_rows = M3_SEQUENTIAL_CAPTURE_ROWS
        # QWEN_FAST_ANY_REQUEST (C2-any, plan S1; default off). Its engines drop replay
        # attention and the per-request T16 gate, which is only sound where every capture is
        # narrower than a replayed block (serving_request_factory.sequential_captures): beside
        # the four-user block. Anywhere else the gate would still refuse every request that is
        # not the frozen benchmark shape, so the attach is refused instead, before anything is
        # built. And the gate's source qualification - every pinned component source hashed
        # against the frozen evidence - runs here, once, in its place: a mismatch fails the
        # attach, as it failed the first admission before (image v42).
        if any_request_enabled():
            if not sequential_captures(capture_rows):
                raise ValueError('QWEN_FAST_ANY_REQUEST=1 needs per-request captures narrower than a replayed block '
                                 '(beside the four-user block: QWEN_FAST_PACKED_STEP=1 with four scheduler '
                                 'requests); this attach caps them at %d rows' % capture_rows)
            attach_source_check()
        bucket_rows = capture_bucket_rows(policy['verifier_rows'], policy['output_budget'], capture_rows)
        trimmed = capture_rows != policy['verifier_rows']
        if trimmed and no_block_any_request:
            pindiag('[PINDIAG] per-request captures trimmed to widths {} for C2-any with no packed block',
                    tuple(sorted(set(bucket_rows))))
        elif trimmed:
            pindiag('[PINDIAG] per-request captures trimmed to widths {} for the four-user block',
                    tuple(sorted(set(bucket_rows))))
        if m3_blocks_two:
            pindiag(M3_BLOCKS_MARKER, 2, '(0, 1, 2, 3) and (4, 5, 6, 7)', 2)
        # packed_shapes covers each distinct shape once (validate_packed_shapes still
        # refuses a shape repeated there); four_as_two's two blocks share one (2, 16) shape,
        # so its multiplicity is named separately, and the pool lends each block asking for
        # it its OWN independent replay table set (ServingBufferPool.packed_replay).
        distinct_shapes = tuple(dict.fromkeys((shape.users, shape.rows_per_user) for shape in packed_shapes))
        solo_shape_value = None
        if solo_lane is not None:
            # D0: the pool also lends the solo block its own one-user extent storage (a (1, 16) set, beside M3's (4, 16)).
            from packed_shapes import solo_shape

            solo_shape_value = solo_shape(page_width)
            distinct_shapes = distinct_shapes + ((solo_shape_value.users, solo_shape_value.rows_per_user),)
        if packed_shapes:
            # The packed block's own replay grouping (QWEN_FAST_REPLAY_GROUP_ROWS), which
            # may differ from the per-request bucket slots' fixed four-row grouping above -
            # packed_verifier.py is the only reader of this value.
            from packed_verifier import replay_group_rows
        # S2: one mask per packed draft (each fixed pair, and the quad), allocated with the pool before any trace,
        # so the drafts need not upload theirs after the request traces exist (serving_buffer_pool's last
        # section). Flag off, no keyword at all. The T16 block: the draft weights below are built at 16 rows.
        draft_masks = {}
        if extent_replay:
            from dflash_packed_proposal_coordinator import pooled_draft_mask_shapes

            draft_masks = pooled_draft_mask_shapes(policy['scheduler_requests'], 16)
        # S2 v86 (run 36416471352): every traced draft's head outputs - each slot's single-user draft, each fixed
        # pair's and the quad's - copied at the end of its pass into a set allocated with the pool before any trace,
        # so no other trace's replay can write what the round reads back (serving_buffer_pool's last section).
        # Flag off, no keyword at all.
        draft_outputs = {}
        if extent_replay:
            from dflash_packed_proposal_coordinator import pooled_draft_output_shapes

            draft_outputs = pooled_draft_output_shapes(policy['scheduler_requests'], 16)
        pool = ServingBufferPool(operations, model.mesh_device, users=policy['scheduler_requests'],
            helpers=helpers, page_width=page_width, bucket_rows=bucket_rows,
            feature_taps=len(TARGET_TAPS), rope=rope,
            **({} if not packed_shapes else dict(packed_shapes=distinct_shapes,
                packed_replay_group_rows=replay_group_rows(),
                **({'packed_replicas': {distinct_shapes[0]: len(packed_shapes)}} if four_as_two or m3_blocks_two else {}),
                # S2: the extent storage in place of the per-family tables; flag off, no keyword at all.
                **({'extent_replay': True} if extent_replay else {}))),
            **({'draft_masks': draft_masks} if draft_masks else {}),
            **({'draft_outputs': draft_outputs} if draft_outputs else {}))
        scopes.callback(pool.close)
        if extent_replay:
            # The pool must hold the extent storage the S2 block keys on (design W2: without it the block
            # would be the per-family one, packed only in [131072, 131312], and nothing would say so), and
            # (design B4) its DRAM statistics must be readable, since the scheduler-side DRAM hold reads them
            # for every request. Either fails the attach here, before any trace (the pool closes with the
            # scopes).
            packed_any_admission.admit_pool(pool, log=pindiag)
        memory_ledger.record('P2', buffer_pool=pool)
        owner = ServingCacheOwner(operations, runner, model)
        # Built once and shared by every request: two TT_CCL objects cycling semaphore
        # handles over one mesh is cross-request interference, not concurrency.
        collectives = TT_CCL(model.mesh_device)
        sampler = SamplingGenerator(args=model.args, mesh_device=model.mesh_device,
            tt_ccl=TT_CCL(model.mesh_device))
        sampler.set_trace_bucket(1)
        from serving_gather_experiment import from_environment

        experiment = from_environment(directory, runtime_root)

        scopes.enter_context(sampler_links(sampler.tt_sampling, 4))
        trace_census.note_collectives(collectives, model, sampler)
        memory_ledger.record('P3', serving_collectives=collectives, serving_sampler=sampler, gather_experiment=experiment)
        audit = scopes.enter_context(combined_runtime(operations, model, directory=directory,
            runtime_root=runtime_root, native_attention_evidence=native_attention_evidence,
            block_stream=block_stream, kv_publication_evidence=kv_publication_evidence,
            **({'single_gateup_admitted': True} if reader is not None and reader[0] == SINGLE_GATEUP_SHAPE
               else {})))
        memory_ledger.record('P4', combined_runtime=audit)
        # The draft weights, uploaded once for every request: they are the first
        # buffers a request allocates, so they took the lowest hole an earlier
        # request's verify trace left, ahead of the history (PreparedDraftWeights).
        # After combined_runtime, because T16 native proposal preparation must run
        # inside its source-bound admission scope; still before any request. The
        # geometry is the one from_prefill asks every device for, and lend() refuses
        # a device asking for anything else.
        _, draft_layers, projection, selector = fixtures
        weights = PreparedDraftWeights(operations, model.mesh_device, draft_layers, projection, selector,
            block_rows=16, live_query_qk=False, native_proposal_attention=True)
        scopes.callback(weights.close)
        memory_ledger.record('P5', draft_weights=weights)
        prefill_warm_before_traces(runner, model, policy['scheduler_requests'], operations=operations)
        # The device step. By default the sequential one - correct, not yet fast: one
        # weight pass per user per round - and describe() records that cost so a benchmark
        # reading it is not mistaken for the goal. QWEN_FAST_PACKED_STEP=1 builds the
        # packed verify block instead, one pass serving every user at the shape chosen
        # above, and builds it HERE: after the pool and the shared draft weights, whose
        # buffers it restores from and checks, and before the lifecycle admits a request,
        # whose traces would otherwise bake over its buffers (packed_verifier.py,
        # CONSTRUCTION ORDER). Registered after the weights so it closes before them,
        # with the scope. Off by default: the sequential step stays the serving default
        # until the packed step's gates pass (docs/packed-device-step-plan-2026-09-20.md).
        from serving_sequential_step import describe as describe_sequential_step, sequential_packed_step

        packed_step, step_description = sequential_packed_step, describe_sequential_step()
        if packed_shapes:
            from packed_verifier import PackedVerifierEngine
            from serving_packed_step import PackedStep, describe as describe_packed_step

            # One block per configured shape, each bound to its own disjoint pool slots in
            # scheduler order - four_as_two's block A over (0, 1), block B over (2, 3) - so
            # every block keeps its fixed carry-identity binding for the pool's whole life
            # and no round ever rebinds a segment. A single configured shape (the m1
            # two-user or m3 four-user default) gets no `pool_slots=` at all, so its block
            # is built exactly as it always was: slots 0..users-1, in order.
            packed_blocks, slot = [], 0
            if capture_position is not None:
                pindiag('{}{} (gate only)', CAPTURE_POSITION_MARKER, capture_position)
            for shape in packed_shapes:
                packed_block = PackedVerifierEngine(operations, model, helpers, sampler, pool=pool, shared_weights=weights,
                                                    shape=shape, feature_taps=TARGET_TAPS,
                                                    **({'pool_slots': tuple(range(slot, slot + shape.users))}
                                                       if four_as_two or m3_blocks_two else {}),
                                                    # QWEN_FAST_M3_BLOCKS=2 (A1c): every block allocates and warms
                                                    # first, then every block captures (complete_blocks_two_phase above).
                                                    **({'defer_capture': True} if m3_blocks_two else {}),
                                                    **({'padded_min_users': padded_min_users}
                                                       if padded_min_users is not None else {}),
                                                    # S2 G3b's gate-only knob; unset, no keyword at all.
                                                    **({'capture_position': capture_position}
                                                       if capture_position is not None else {}),
                                                    # Round-fence plan H1b (QWEN_FAST_FUSED_COMMIT, default
                                                    # off): the block's T_proj traces use the one shared
                                                    # TT_CCL every request's device uses. On four cards
                                                    # (fused commit off) S2 B6's eager publication warm
                                                    # needs it too, or it is skipped; the pair is unchanged.
                                                    **({'collectives': collectives}
                                                       if os.environ.get('QWEN_FAST_FUSED_COMMIT') == '1'
                                                       or (os.environ.get('QWEN_FAST_EXTENT_REPLAY') == '1'
                                                           and os.environ.get('QWEN_FAST_TP', '2') != '2') else {}))
                scopes.callback(packed_block.close)
                packed_blocks.append(packed_block)
                if not m3_blocks_two:
                    memory_ledger.record('P6', point='block%d' % len(packed_blocks), packed_block=packed_block)
                slot += shape.users
            if m3_blocks_two:
                complete_blocks_two_phase(packed_blocks, model)
                for index, packed_block in enumerate(packed_blocks, 1):
                    memory_ledger.record('P6', point='block%d' % index, packed_block=packed_block)
            # D0: the one-user block, after M3 (a block is built before any request exists, and the pool's slot 0 is
            # bound by carry identity to M3's segment 0 AND to this block's only segment), over slot 0 alone.
            solo_block = None
            if solo_lane is not None:
                solo_block = PackedVerifierEngine(operations, model, helpers, sampler, pool=pool, shared_weights=weights,
                                                  shape=solo_shape_value, feature_taps=TARGET_TAPS,
                                                  pool_slots=(solo_lane['slot'],),
                                                  **({'capture_position': capture_position}
                                                     if capture_position is not None else {}),
                                                  # The same shared collectives as the M3 blocks (on four cards
                                                  # under extent replay), or the 16-row publication shapes are
                                                  # never warmed and compile mid-request.
                                                  **({'collectives': collectives}
                                                     if os.environ.get('QWEN_FAST_FUSED_COMMIT') == '1'
                                                     or (os.environ.get('QWEN_FAST_EXTENT_REPLAY') == '1'
                                                         and os.environ.get('QWEN_FAST_TP', '2') != '2') else {}))
                scopes.callback(solo_block.close)
                memory_ledger.record('P6', point='solo', packed_block=solo_block)
            # QWEN_FAST_M3_BLOCKS=2: refused unless BOTH blocks report that their captured trace reads every carry in
            # place (QWEN_FAST_VERIFY_T1 #3). Every block's commit DMA writes the model's native slot 0, so a block whose
            # trace read slot 0 instead of its own carries would be fed another block's state: the attach is refused.
            if m3_blocks_two:
                refused_blocks = [index for index, packed_block in enumerate(packed_blocks)
                                  if getattr(packed_block, 'carries_in_place', False) is not True]
                if refused_blocks:
                    raise ValueError('%s=2 needs every M3 block to read its carries in place (QWEN_FAST_VERIFY_T1 #3); '
                                     'blocks %s do not report carries_in_place' % (M3_BLOCKS_FLAG, refused_blocks))
            # S2: every block must be the extent block under the flag, and none may be without it. The
            # block keys on the pool's storage alone (PackedVerifierEngine.extent), so this is the one
            # attach-time proof that the flag reached the storage and the storage the block (design W2,
            # W3); refused before the lifecycle admits a request, and the scopes close what was built.
            extents = [getattr(packed_block, 'extent', False)
                       for packed_block in packed_blocks + ([solo_block] if solo_block is not None else [])]
            if any(value is not extent_replay for value in extents):
                raise ValueError('%s=%d, but the packed blocks built are extent=%r'
                                 % (EXTENT_REPLAY_FLAG, int(extent_replay), extents))
            # The step bound to its block (or blocks), carrying the per-round ticket-width
            # policy the worker hook asks before drafting.
            packed_step = PackedStep(packed_blocks if four_as_two or m3_blocks_two else packed_blocks[0],
                                     **({'solo': solo_block} if solo_block is not None else {}),
                                     # QWEN_FAST_M3_BLOCKS=2: each block's ticket width is its own (PackedStep.proposal_groups).
                                     **({'per_block_widths': True} if m3_blocks_two else {}))
            if m3_blocks_two:
                # New arrivals fill a block that has exactly one live user first, else the fuller block that is not full
                # (ServingBufferPool.place_blocks), so a lone user is rare and a block runs packed whenever it can.
                pool.place_blocks(tuple(tuple(range(index * 4, index * 4 + 4)) for index in range(2)))
            step_description = dict(describe_packed_step(),
                **(dict(blocks=[packed_block.describe() for packed_block in packed_blocks])
                   if four_as_two or m3_blocks_two else dict(block=packed_blocks[0].describe())),
                **(dict(solo=solo_block.describe()) if solo_block is not None else {}))
        elif packed_requested:
            step_description = dict(step_description,
                packed_block_skipped='no packed block shape for %d scheduler requests' % policy['scheduler_requests'])
        if extent_replay:
            # The executed path is the admitted one (memory graft-mounted-is-not-graft-executed): every block
            # is the extent block and every segment reader reports runtime_extent, or the attach fails here,
            # before the lifecycle admits a request (the scopes close the block, the weights and the pool).
            packed_any_admission.admit_blocks(packed_blocks + ([solo_block] if solo_block is not None else []),
                                              log=pindiag)
        # One line with every pre-trace address - the pooled history pairs, each named
        # shared weight and, when built, the packed block's taps, checkpoints and carries -
        # so a diverged address from the shard check can be placed against what was
        # allocated before any request; and which device step serves the rounds.
        print(json.dumps(dict(stage='serving_buffer_pool', **pool.describe(), draft_weights=weights.describe(),
                              device_step=step_description)), flush=True)
        # The device allocator after everything the attach allocates - weights, KV pool,
        # buffer pool, draft weights, the block and its traces - so the log shows the
        # headroom the per-request engines have (run 35509307389 found 214 MB of it).
        pindiag('[PINDIAG] dram after attach: {}', dram_line(pool))
        memory_ledger.record('P7', point='after_attach')

        # Sticky sessions (QWEN_FAST_STICKY_SESSIONS, read once at attach; default off). On, every
        # prefill capture wraps the prefix-reuse route (QWEN_PREFIX_REUSE=1 sends every prefill,
        # cold or resumed, through it), a granted hit's capture counts from its R, and each engine
        # build is timed (STICKY_ENGINE_MARKER). Off, the factories below build exactly what they
        # always did.
        sticky = sticky_sessions_enabled()
        lanes = None
        if lane_config is not None:
            lanes = serving_fast_lane.LaneRuntime(lane_config, log=pindiag)

        def capture_factory(position, start=0):
            owner.validate()
            # The allocator just before this user's prefill; the bridge factory reads it
            # again just after, so the pair bounds what the prefill leaves resident.
            # QWEN_FAST_ADMISSION_DIAG_TRIM=1 (the traffic twins): the first admission's point only.
            if memory_ledger.admission_diag('prefill_before'):
                memory_ledger.record('prefill', point='before prompt=%d' % position)
            if not sticky:
                if start:
                    raise ValueError('A prefill resumed at %d needs %s=1' % (start, STICKY_SESSIONS_FLAG))
                return PrefillWindowCapture(operations, model, position, TARGET_TAPS)
            return PrefillWindowCapture(operations, model, position, TARGET_TAPS, start=start, prefix_route=True)

        def bridge_factory(state, capture):
            owner.validate()
            if memory_ledger.admission_diag('prefill_after'):
                memory_ledger.record('prefill', point='after req=%s' % memory_ledger.short_id(state.req_id),
                                     request=str(state.req_id), model_after_prefill=model)
            if len(state.block_ids) != 1:
                raise ValueError('One scheduler KV group required')
            blocks = tuple(state.block_ids[0])
            if (not blocks or len(blocks) > page_width or len(set(blocks)) != len(blocks)
                    or any(type(block) is not int or not 0 <= block < owner.physical_pages for block in blocks)):
                raise ValueError('Unique physical pages from the admitted cache required')
            pages = torch.full((1, page_width), blocks[0], dtype=torch.int32)
            pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            def create_request():
                return from_prefill(operations, model, sampler, pages, helpers,
                    state=state, capture=capture, fixtures=fixtures, eos_ids=eos_ids,
                    collectives=collectives, buffer_pool=pool, shared_weights=weights,
                    **(dict(capture_rows=capture_rows) if trimmed else {}))

            # QWEN_FAST_LANE: the lane the request is granted decides which pool slots its engine may borrow (the fast request
            # slot 0 alone, standard requests the others), so the carry the solo block is bound to is the fast request's.
            # Admission is the worker's: a second fast request is downgraded to standard here, never queued or refused.
            grant = None
            if lanes is not None:
                grant = lanes.admit(state.req_id, state.sampling_params, slot0_free=not pool.slots[0].lent)
            if sticky:
                began = time.perf_counter()
            trace_census.engine_begin()
            create_request = trace_census.build_guard(create_request, operations, model.mesh_device, state.req_id)
            try:
                if grant is None:
                    request = create_request() if experiment is None else experiment.create(create_request)
                else:
                    with pool.slot_order(grant.slot_order):
                        request = create_request() if experiment is None else experiment.create(create_request)
            except BaseException:
                if lanes is not None:
                    lanes.release(state.req_id)
                raise
            if sticky:
                # Sticky sessions: the engine build per request, so a gate can split a hit's TTFT into
                # its tail prefill and the build phase 1 still pays (STICKY_ENGINE_MARKER).
                pindiag(STICKY_ENGINE_MARKER + '{} ms={:.1f} frontier={} prompt={}', str(state.req_id)[:48],
                        (time.perf_counter() - began) * 1000.0, state.num_computed_tokens,
                        len(state.prompt_token_ids))
            # The allocator after this request's engine and its captures: one line per
            # admitted request, so the log shows what each costs and what is left.
            # Under QWEN_FAST_ADMISSION_DIAG_TRIM=1 the line is off (it reads the allocator of every chip, and
            # the first engine's ledger point carries the same reading) and only the first engine is walked.
            if not memory_ledger.trim_enabled():
                pindiag('[PINDIAG] dram after engine {}: {}', str(state.req_id)[:48], dram_line(pool))
            if memory_ledger.admission_diag('engine'):
                memory_ledger.engine_admitted(str(state.req_id), engine_request=request)
            trace_census.census_engine(str(state.req_id), request, operations)
            try:
                binding = VerifierPageBinding(request.engine, blocks, physical_pages=owner.physical_pages)
                return FastRunnerBridge(runner, request, binding, validate_storage=owner.validate)
            except BaseException:
                if lanes is not None:
                    lanes.release(state.req_id)
                request.close(state.req_id)
                raise

        # S2 W6b (QWEN_FAST_EXTENT_REPLAY=1 only; unset, nothing is registered and the scheduler admits exactly
        # as before): the scheduler-side DRAM admission hold's predicate (serving_prefill_admission), read through
        # this pool, is parked before the lifecycle serves a request; the scope removes it before the pool closes.
        if extent_replay_enabled():
            scopes.callback(register_dram_admission(pool))
        capture_factory, bridge_factory = prefill_tripwire(model, capture_factory, bridge_factory)
        lifecycle = FastServingLifecycle(worker, config=worker.vllm_config,
            capture_factory=capture_factory, bridge_factory=bridge_factory, eos_ids=eos_ids,
            cancelled=cancelled, packed_step=packed_step,
            **({'lanes': lanes} if lanes is not None else {}))
    except BaseException as failure:
        # Closing the scopes can itself raise (the block-stream scope checks at exit
        # that every layer ran the fused candidate, which nothing has at attach), and
        # run 35496483954 logged only that, losing the attach failure it was
        # handling. Keep the original: log the closing error and re-raise the first.
        # And log the failure BEFORE closing: the scopes' exits fence the device
        # (the block's close, then the admitted runtime's shared-QK scope first), and a
        # device left hung by the failed attach blocks the first fence for good - run
        # 35507675630 (image v51) sat in gdn_shared_qk_scope's exit for the rest of its
        # 900 s with nothing in the log to say so.
        pindiag('[PINDIAG] attach failed with {}: {}; closing the attach scopes now (the packed block if built, '
                'the draft weights, the admitted combined runtime from its shared-QK scope, the sampler links, '
                'the pool) - their exits fence the device, and a hung device blocks the first fence',
                type(failure).__name__, str(failure)[:300])
        try:
            scopes.close()
        except BaseException as closing:
            try:
                from loguru import logger
                logger.info('[PINDIAG] attach failed with {}: {}; closing the scopes then raised {}: {}',
                            type(failure).__name__, str(failure)[:300], type(closing).__name__, str(closing)[:200])
            except BaseException:
                pass
        raise failure
    try:
        yield dict(lifecycle=lifecycle, runtime=audit, cache_owner=owner,
            serving_qualified=False, performance_qualified=False)
    finally:
        memory_ledger.record('P13', point='before_shutdown')
        lifecycle.close()
        scopes.close()
