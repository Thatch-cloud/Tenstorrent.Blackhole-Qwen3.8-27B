"""Round-fence plan H1a: pre-stage the next packed verify inside the drafts' fence window, guarded
by a VALUE DIFF, and the round's fence diet. Three flags, every one default off:

  QWEN_FAST_PRESTAGE        the pre-stage (this module, packed_verifier.verify)
  QWEN_FAST_PRESTAGE_AUDIT  reads back a rotating AUDIT_BUFFERS of the staged buffers after each
                            verify-time write and compares them with the verify-time values
  QWEN_FAST_ROUND_FENCES    the fence diet (gdn_records.RetainedGDNBlock, packed_verifier)

WHAT THE PRE-STAGE BUYS. packed_verifier.stage_packed restages ~140 captured buffers before every
verify (tokens, positions, rotary tables, singleton positions, per-row and per-tile page tables,
each reader's positions word and bundle tables) and fences (F1); verify() first re-checks every
binding (bind_ms). All of it but the tokens depends only on each user's next frontier and page
table, both known once the round's commits are done. So the drafts' fence window - the host
waiting for the two pair proposals (dflash_packed_proposal_coordinator.prepare) - computes and
writes every verify input except the tokens (`BlockPrestage.prestage`, with no fence of its own:
the window's fence F9 follows it) and validates the block's bindings.

WHY A VALUE DIFF. The pages are refreshed at execute time, after the window (serving_packed_bridge,
page_binding.refresh), and an unrefreshed page entry names the user's own first block, so a missed
page change would write K/V into the wrong place. The pre-stage therefore keeps no key of what it
thinks the values depend on: it keeps the raw host VALUES it wrote, per destination (`Snapshot`).
At verify the host values are recomputed from the verify-time tickets and tables - the ground
truth, through the same packed_values stage_packed uses, T2 K/V guard included - and only the
buffers whose value differs are copied: the tokens always, a page change a few more. Nothing is
fenced (F1 goes: the trace follows on the same in-order CQ0), and verify() skips validate_bindings
(it ran in the window).

THE ONE INVARIANT LEFT is that nothing else wrote the fixture's buffers between the pre-stage and
the verify. The WRITE EPOCH enforces it: every other writer of fixture inputs bumps it (`bump`) -
the full stage_packed (so the padded probe's restage), a prefill or a Lever N prefill chunk
(serving_lifecycle, the worker hook's pass-through), an admission or a detach (the hook), and the
verify itself - and a snapshot whose epoch moved takes today's full stage_packed. A snapshot serves
one verify at most.

FAILURE. A pre-stage that raises drops its snapshot (the epoch was bumped before its first copy)
and never fails the round: the next verify takes the full path, which restages every buffer. The
pre-stage never poisons a reader and never moves a reader's `start` (both only at verify, as
today). The diff write poisons on failure exactly as stage_packed does.

THE AUDIT (QWEN_FAST_PRESTAGE_AUDIT, with QWEN_FAST_PRESTAGE). After every verify-time write, diff
or full, AUDIT_BUFFERS destinations in rotation are read back from both chips and compared with
the verify-time values. A mismatch is logged (AUDIT_MARKER ... mismatches=N) and the round is
restaged in full before its trace, so the round stays exact; the gate fails the arm on it.

ROUND FENCES (QWEN_FAST_ROUND_FENCES; gdn_records.RetainedGDNBlock.use_round_fences). F3: the
retained block's replay drops the fence after the (blocking) trace and the second validate. F8: the
round's last commit no longer fences; the replay is armed by the drafts' fence F9
(`WhileWaiting.fenced` -> note_round_fence) or, when no draft fenced, by one fence at replay. The
first commit skips its validate when the round's replay validated (validated_this_round). A replay
still needs every segment decided, and a poisoned block still refuses.

H1b (fused_commit.py, QWEN_FAST_FUSED_COMMIT): the same window also writes each live segment's
T_proj RoPE tables for its next frontier (WhileWaiting -> FusedCommit.stage_window), before the
pre-stage; they are not fixture inputs and move no epoch.

THE EIGHT-SEAT HOST GAP (tp4/hostgap; every flag below default off, byte-identical off). With two packed 64-row
blocks the window leaves the verify pre-stage out (serving_packed_step.while_waiting_groups): the fixture write epoch
is one counter, so a second block's pre-stage or the first block's verify would kill the other's snapshot, and 744 of
788 measured verifies took the full stage (22.7 ms of device idle a round).

  QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE=1   1a-lite: under today's ONE epoch the block that verifies first is pre-staged in
                            the two-block window. Between its pre-stage and its verify nothing bumps the epoch (the
                            other block's pre-stage is left out; A's own diff bumps it only after A consumed its
                            snapshot), so this is the one-block mechanism four seats already qualified. The second
                            block takes the full stage, as today. Needs QWEN_FAST_PRESTAGE=1 and two blocks.
  QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS=1   1a proper (needs the flag above): each block's OWN writes (its pre-stage,
                            its verify-time diff, its full stage_packed) bump a per-fixture epoch instead of the global
                            one, so BOTH blocks keep a usable snapshot; every external writer still bumps the global
                            epoch and kills both. A snapshot is usable only while (global, local) both stand. Engaged at
                            attach only when the blocks' fixtures, replay readers and extent storage are distinct
                            objects, and disengaged (global bumps again) the moment a staging destination of one block
                            shares a chip-local address with the other's.
  QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT=1   after each verify-time diff write, every destination is read back from
                            every chip and compared with the verify-time (full) values, then the round is staged IN
                            FULL anyway, so the trace sees the control's inputs. [PACKED-PRESTAGE-FULLAUDIT].
  QWEN_FAST_TP4_WINDOW_VALIDATE=1      1c: a verify that took a usable snapshot (its window ran validate_bindings)
                            skips the retained block's second binding check right before its trace; under the audit
                            the skipped check still runs, as a shadow, and must pass.
  QWEN_FAST_TP4_ENTRY_DIET=1           1d: one storage check per distinct validator a step, and an incremental
                            page-allocation validation (serving_packed_bridge, serving_page_binding).
  QWEN_FAST_TP4_HOSTGAP_LOG=1          stage 0: [PACKED-ENTRY], [PACKED-HOSTGAP-VERIFY], [PACKED-HOSTGAP-SELECT],
                            [PACKED-HOSTGAP-WINDOW] and [PINDIAG] gc lines; new lines only, no existing line changes.
"""

import os
import sys
import time

import round_host

PRESTAGE_FLAG = 'QWEN_FAST_PRESTAGE'
PRESTAGE_AUDIT_FLAG = 'QWEN_FAST_PRESTAGE_AUDIT'
ROUND_FENCES_FLAG = 'QWEN_FAST_ROUND_FENCES'
TWO_BLOCK_FLAG = 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE'
BLOCK_EPOCHS_FLAG = 'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS'
FULL_AUDIT_FLAG = 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT'
WINDOW_VALIDATE_FLAG = 'QWEN_FAST_TP4_WINDOW_VALIDATE'
ENTRY_DIET_FLAG = 'QWEN_FAST_TP4_ENTRY_DIET'
HOSTGAP_LOG_FLAG = 'QWEN_FAST_TP4_HOSTGAP_LOG'
HOSTGAP_FLAGS = (TWO_BLOCK_FLAG, BLOCK_EPOCHS_FLAG, FULL_AUDIT_FLAG, WINDOW_VALIDATE_FLAG, ENTRY_DIET_FLAG,
                 HOSTGAP_LOG_FLAG)

# Once at attach, from the block that engaged the flag (packed_verifier).
ENGAGED_MARKER = '[PINDIAG] verify prestage engaged'
FENCES_ENGAGED_MARKER = '[PINDIAG] round fences engaged'
# Per round: the window's pre-stage (or why it dropped), the verify's path, the audit, and the
# step's fence line (serving_packed_step, under QWEN_FAST_PACKED_AUDIT).
WINDOW_MARKER = '[PACKED-PRESTAGE-WINDOW]'
MARKER = '[PACKED-PRESTAGE]'
AUDIT_MARKER = '[PACKED-PRESTAGE-AUDIT]'
# tp4/hostgap: once per block at attach (mode=first|blocks), once for the per-block epochs, a refusal with its reason, and per
# audited verify the full read-back of every destination.
TWO_BLOCK_ENGAGED_MARKER = '[PINDIAG] verify prestage two-block engaged'
TWO_BLOCK_REFUSED_MARKER = '[PINDIAG] verify prestage two-block refused'
BLOCK_EPOCHS_ENGAGED_MARKER = '[PINDIAG] verify prestage block epochs engaged'
BLOCK_EPOCHS_REFUSED_MARKER = '[PINDIAG] verify prestage block epochs refused'
# Lever N merged route (docs/lever-n-prefix-merged-route.md section 8): the external-writer class. A step of a split prefill that wrote no decode
# slot is a DISJOINT writer: it bumps no epoch while every persistent address it writes is disjoint from every block's staging destinations.
EXTERNAL_WRITER_ENGAGED_MARKER = '[PINDIAG] verify prestage external writer engaged'
EXTERNAL_WRITER_REFUSED_MARKER = '[PINDIAG] verify prestage external writer refused'
FULL_AUDIT_MARKER = '[PACKED-PRESTAGE-FULLAUDIT]'
ENTRY_MARKER = '[PACKED-ENTRY]'
HOSTGAP_VERIFY_MARKER = '[PACKED-HOSTGAP-VERIFY]'
HOSTGAP_SELECT_MARKER = '[PACKED-HOSTGAP-SELECT]'
HOSTGAP_WINDOW_MARKER = '[PACKED-HOSTGAP-WINDOW]'
GC_MARKER = '[PINDIAG] gc'
GC_LOG_MS = 5.0
FENCES_MARKER = '[PACKED-FENCES]'
AUDIT_BUFFERS = 8


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % name)
    return value == '1'


def enabled(environ=None):
    """QWEN_FAST_PRESTAGE=1. Any value but 0 or 1 is a configuration error."""
    return _flag(PRESTAGE_FLAG, environ)


def audit_enabled(environ=None):
    """QWEN_FAST_PRESTAGE_AUDIT=1 under QWEN_FAST_PRESTAGE=1 (alone it audits nothing: the gate
    and the arm refuse it; here it is inert)."""
    return _flag(PRESTAGE_AUDIT_FLAG, environ) and enabled(environ)


def round_fences_enabled(environ=None):
    """QWEN_FAST_ROUND_FENCES=1."""
    return _flag(ROUND_FENCES_FLAG, environ)


def any_enabled(environ=None):
    """Whether the worker hook has a window callable to build at all (either flag)."""
    return enabled(environ) or round_fences_enabled(environ)


def two_block_enabled(environ=None):
    """QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE=1 (1a-lite; engaged only with QWEN_FAST_PRESTAGE=1 and two blocks, engage_two_block)."""
    return _flag(TWO_BLOCK_FLAG, environ)


def block_epochs_enabled(environ=None):
    """QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS=1 (1a proper; needs the two-block flag)."""
    return _flag(BLOCK_EPOCHS_FLAG, environ)


def full_audit_enabled(environ=None):
    """QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT=1 under QWEN_FAST_PRESTAGE=1 (alone it audits nothing)."""
    return _flag(FULL_AUDIT_FLAG, environ) and enabled(environ)


def window_validate_enabled(environ=None):
    """QWEN_FAST_TP4_WINDOW_VALIDATE=1 under QWEN_FAST_PRESTAGE=1 (1c)."""
    return _flag(WINDOW_VALIDATE_FLAG, environ) and enabled(environ)


def skip_next_binding_check(block):
    """1c (QWEN_FAST_TP4_WINDOW_VALIDATE): a context manager under which the retained block's NEXT validate_bindings call is a no-op and every
    later one is the block's own. RetainedGDNBlock.replay (a pinned source, never edited) makes its first check right before the trace, with nothing
    between its entry and that check that validates (fence_at_replay only synchronizes); the check after the trace's sync, when there are no round
    fences, is the second call and still runs. The instance attribute is put back (or removed) on exit, raise or not."""
    import contextlib

    @contextlib.contextmanager
    def scope():
        had = 'validate_bindings' in vars(block)
        previous = vars(block).get('validate_bindings')
        inner = block.validate_bindings
        state = {'skipped': 0}

        def once():
            if not state['skipped']:
                state['skipped'] = 1
                return None
            return inner()

        block.validate_bindings = once
        try:
            yield state
        finally:
            if had:
                block.validate_bindings = previous
            else:
                del block.validate_bindings

    return scope()


def entry_diet_enabled(environ=None):
    """QWEN_FAST_TP4_ENTRY_DIET=1 (1d: one storage check per validator, incremental page validation)."""
    return _flag(ENTRY_DIET_FLAG, environ)


def hostgap_log_enabled(environ=None):
    """QWEN_FAST_TP4_HOSTGAP_LOG=1 (stage 0: new log lines only)."""
    return _flag(HOSTGAP_LOG_FLAG, environ)


# The fixture write epoch: one counter per process (every packed block's fixture lives in this
# process), bumped by every writer of fixture inputs other than the pre-stage it invalidates.
_EPOCH = [0, 'start']


def epoch():
    return _EPOCH[0]


def last_bump():
    return _EPOCH[1]


def bump(reason):
    """Another writer touched (or may have touched) a packed fixture's inputs: every snapshot
    taken before now is stale. Host only, never raises."""
    _EPOCH[0] += 1
    _EPOCH[1] = str(reason)


# tp4/hostgap. The mode the attach engaged (engage_two_block): 'first' (1a-lite: the first block to verify is pre-staged under
# the one global epoch), 'blocks' (1a proper: per-fixture epochs), or None (today's rule). _LOCAL holds the per-fixture epoch
# {id(fixture): (count, last reason)}, moved only in 'blocks' mode; _ADDRESSES the chip-local addresses of each block's staging
# destinations, filled by the block's first pre-stage. Process state, like _EPOCH.
_MODE = dict(first=False, blocks=False)
_LOCAL = {}
_ADDRESSES = {}


def two_block_mode():
    """'blocks', 'first' or None."""
    return 'blocks' if _MODE['blocks'] else 'first' if _MODE['first'] else None


def local_epoch(fixture):
    return _LOCAL.get(id(fixture), (0, 'start'))[0]


def bump_fixture(fixture, reason):
    """A write to THIS fixture's own inputs (its pre-stage, its verify-time diff, its full stage_packed). With the per-block
    epochs engaged only this fixture's snapshot goes stale; otherwise it is `bump`, argument for argument. Host only."""
    if _MODE['blocks']:
        _LOCAL[id(fixture)] = (local_epoch(fixture) + 1, str(reason))
    else:
        bump(reason)


def last_local_bump(fixture):
    return _LOCAL.get(id(fixture), (0, 'start'))[1]


def leaf_ids(value, seen=None):
    """The ids of every non-container object inside nested lists, tuples and dicts (the extent storage's tensors)."""
    seen = set() if seen is None else seen
    if isinstance(value, (list, tuple)):
        for item in value:
            leaf_ids(item, seen)
    elif isinstance(value, dict):
        for item in value.values():
            leaf_ids(item, seen)
    elif value is not None:
        seen.add(id(value))
    return seen


def two_block_refusal(blocks):
    """Why the two-block pre-stage cannot engage on these blocks, or None: one block, or a block built without the pre-stage
    state (QWEN_FAST_PRESTAGE=1), or two blocks sharing one fixture."""
    blocks = tuple(blocks)
    if len(blocks) < 2:
        return 'one packed block (QWEN_FAST_M3_BLOCKS=2 needed)'
    if any(getattr(block, 'prestaged', None) is None for block in blocks):
        return 'a block was built without the pre-stage (QWEN_FAST_PRESTAGE=1 needed)'
    if len({id(getattr(block, 'fixture', None)) for block in blocks}) != len(blocks):
        return 'two blocks share one fixture'
    return None


def block_epochs_refusal(blocks):
    """Why per-block epochs cannot engage, or None: every block has its own fixture, replay reader and extent storage
    (distinct objects, no tensor in two blocks' extent storage)."""
    readers = [getattr(block.fixture, 'replay_reader', None) for block in blocks]
    if any(reader is None for reader in readers):
        return 'a block has no replay reader'
    if len({id(reader) for reader in readers}) != len(blocks):
        return 'two blocks share one replay reader'
    seen, total = set(), 0
    for block in blocks:
        leaves = leaf_ids(getattr(block, 'extent_storage', None))
        total += len(leaves)
        seen |= leaves
    if len(seen) != total:
        return 'two blocks share extent storage'
    return None


def block_label(block, index=None):
    """The label the hostgap lines name a block by ('A', 'B', ...): set at attach, '-' for a block never labelled."""
    label = getattr(block, 'hostgap_label', None)
    if label is None and index is not None:
        label = 'ABCDEFGH'[index % 8]
        try:
            block.hostgap_label = label
        except Exception:
            pass
    return label or '-'


def engage_two_block(blocks, environ=None):
    """Attach (serving_packed_step.PackedStep, several blocks): reset the mode and engage what the flags ask for. The first is
    1a-lite (QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE), the second per-block epochs (QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS, which needs
    the first). One refusal line, with its reason, per refused flag; the mode stays what it was able to engage. Returns the mode."""
    _MODE.update(first=False, blocks=False)
    _LOCAL.clear()
    _ADDRESSES.clear()
    blocks = tuple(blocks)
    # Every host-gap flag is read once here: a value but 0 or 1 raises at the attach, not at the first step that reads it.
    for name in HOSTGAP_FLAGS:
        _flag(name, environ)
    asked, per_block = two_block_enabled(environ), block_epochs_enabled(environ)
    if per_block and not asked:
        log_line('%s reason=%s' % (BLOCK_EPOCHS_REFUSED_MARKER, '%s_needs_%s=1' % (BLOCK_EPOCHS_FLAG, TWO_BLOCK_FLAG)))
    if not asked:
        return None
    reason = two_block_refusal(blocks)
    if reason is not None:
        log_line('%s reason=%s' % (TWO_BLOCK_REFUSED_MARKER, reason.replace(' ', '_')))
        return None
    _MODE['first'] = True
    mode = 'first'
    if per_block:
        reason = block_epochs_refusal(blocks)
        if reason is None:
            _MODE['blocks'] = True
            mode = 'blocks'
            log_line('%s blocks=%d' % (BLOCK_EPOCHS_ENGAGED_MARKER, len(blocks)))
        else:
            log_line('%s reason=%s' % (BLOCK_EPOCHS_REFUSED_MARKER, reason.replace(' ', '_')))
    for index, block in enumerate(blocks):
        log_line('%s block=%s users=%d mode=%s window_validate=%d audit=%d' % (
            TWO_BLOCK_ENGAGED_MARKER, block_label(block, index), getattr(block, 'users', 0), mode,
            int(window_validate_enabled(environ)), int(full_audit_enabled(environ))))
    return mode


# The external writers (register_external_writer): {name: frozenset of (chip, address)}, and whether the disjoint class is live: engaged by the
# first registration, and dropped for the process (the safe direction: every step bumps the global epoch again) by an address overlap with a block's
# staging destinations. `disjoint` counts the steps that took the class (what a gate reads beside the audited arm).
_EXTERNAL = {}
_SCOPE = dict(route=False, refused=None, disjoint=0, global_bumps=0)


def register_external_writer(name, addresses):
    """The chip-local (chip, address) pairs of the persistent device buffers an external writer writes between rounds (the Lever N route's B=1
    scratch: rec_state, conv_states and conv_carry of every GDN layer). Returns 'engaged', or 'overlap' when they intersect the staging destinations
    of a block that already registered them (the class is then off for the process). Blocks that register later are checked in
    register_destinations."""
    pairs = frozenset(addresses)
    if not pairs:
        _SCOPE.update(route=False, refused='no addresses')
        log_line('%s name=%s reason=no_addresses' % (EXTERNAL_WRITER_REFUSED_MARKER, name))
        return 'overlap'
    for key, other in _ADDRESSES.items():
        if not pairs.isdisjoint(other):
            _EXTERNAL.pop(name, None)
            _SCOPE.update(route=False, refused='overlap')
            log_line('%s name=%s reason=overlap' % (EXTERNAL_WRITER_REFUSED_MARKER, name))
            return 'overlap'
    _EXTERNAL[name] = pairs
    _SCOPE.update(route=True, refused=None)
    log_line('%s name=%s addresses=%d' % (EXTERNAL_WRITER_ENGAGED_MARKER, name, len(pairs)))
    return 'engaged'


def disjoint_engaged():
    """Whether a step that wrote no decode slot may skip the global epoch bump: an external writer registered, no overlap seen, and the per-block
    epochs engaged (the staging destinations the writer was checked against are only known then; with one global epoch every writer bumps)."""
    return bool(_SCOPE['route'] and _EXTERNAL and _MODE['blocks'])


def note_disjoint(reason):
    """A disjoint writer ran: no epoch moves; counted, so the host-gap lines and a gate still see it. Host only, never raises."""
    _SCOPE['disjoint'] += 1
    _SCOPE['last_disjoint'] = str(reason)


def disjoint_count():
    return _SCOPE['disjoint']


def reset_external_writers():
    """Forget every external writer (a new attach, a test)."""
    _EXTERNAL.clear()
    _SCOPE.update(route=False, refused=None, disjoint=0, global_bumps=0)
    _SCOPE.pop('last_disjoint', None)


def register_destinations(block, destinations, addresses):
    """Per-block epochs only, once per block (its first pre-stage, before any bump or copy): the chip-local addresses of the
    staging destinations against every other registered block's. A shared address means one block's write could change what the
    other's snapshot holds, so the per-block epochs disengage (global bumps again, the safe direction) and say so. Returns
    'overlap' on a disengage, else None."""
    if not _MODE['blocks'] or id(block.fixture) in _ADDRESSES:
        return None
    mine = frozenset((chip, address) for destination in destinations for chip, address in enumerate(addresses(destination)))
    for key, other in _ADDRESSES.items():
        if key != id(block.fixture) and not mine.isdisjoint(other):
            _MODE['blocks'] = False
            log_line('%s reason=%s' % (BLOCK_EPOCHS_REFUSED_MARKER, 'staging_destinations_overlap_between_blocks'))
            return 'overlap'
    for name, writes in _EXTERNAL.items():
        if not mine.isdisjoint(writes):
            # An external writer's persistent buffers share an address with this block's staging destinations: a step of it could change what the
            # block's snapshot holds, so the disjoint class is off for the process and every such step bumps the global epoch again.
            _EXTERNAL.clear()
            _SCOPE.update(route=False, refused='overlap')
            log_line('%s name=%s reason=overlap' % (EXTERNAL_WRITER_REFUSED_MARKER, name))
            break
    _ADDRESSES[id(block.fixture)] = mine
    return None


# Stage 0 scratch (QWEN_FAST_TP4_HOSTGAP_LOG): values one step leaves for the line another function writes.
_SCRATCH = {}


def note_scratch(key, value):
    _SCRATCH[key] = value


def take_scratch(key, default=None):
    return _SCRATCH.pop(key, default)


def add_collect_split(read_ms, merge_ms):
    """Stage 0: one quad's readback split (quad_draft_tp.read_quad_outputs), summed over the round's quads until select_round
    writes its line."""
    total = _SCRATCH.setdefault('collect', [0.0, 0.0])
    total[0] += read_ms
    total[1] += merge_ms


def entry_line(checks_ms):
    """Stage 0: the step entry's [PACKED-ENTRY] line - what serving_packed_bridge.execute_packed_decode left (admit, the storage check, vLLM's
    _update_states, the reservation checks, the page refreshes and how many wrote the device) and the packed step's own checks
    (validate, group, ineligible), with the whole entry's wall time up to now. One line a step; nothing left, nothing written."""
    drain_gc_lines()
    entry = take_scratch('entry')
    if entry is None:
        return
    try:
        log_line('%s admit_ms=%.2f update_states_ms=%.2f storage_ms=%.2f reservation_ms=%.2f refresh_ms=%.2f refresh_writes=%d '
                 'checks_ms=%.2f entry_ms=%.2f' % (ENTRY_MARKER, entry['admit_ms'], entry['update_states_ms'], entry['storage_ms'],
                                                  entry['reservation_ms'], entry['refresh_ms'], entry['refresh_writes'], checks_ms,
                                                  (time.perf_counter() - entry['started']) * 1000))
    except BaseException:
        pass


def thread_ms():
    """This thread's CPU time in ms, beside wall time: GC and host compute count here, a descheduled thread does not. Read only
    under QWEN_FAST_TP4_HOSTGAP_LOG (0.0 otherwise): with the flag off the staging path makes no extra clock call."""
    try:
        if not hostgap_log_enabled():
            return 0.0
        return time.thread_time() * 1000
    except Exception:
        return 0.0


GC_PENDING_MAX = 64
_GC = dict(installed=False, started=0.0, callback=None, pending=[])


def install_gc_log():
    """Stage 0: one [PINDIAG] gc line for any collection over GC_LOG_MS (the 50-170 ms host stalls: gen-2 collections or not).
    The callback only APPENDS (generation, collected, ms) to a bounded list (as publication_diagnostics does): a collection can
    start inside the logger's own emit, so it never logs. `drain_gc_lines` writes the lines, from window_line and entry_line.
    Idempotent; a callback never raises."""
    if _GC['installed']:
        return False
    import gc

    def callback(phase, info):
        try:
            if phase == 'start':
                _GC['started'] = time.perf_counter()
                return
            ms = (time.perf_counter() - _GC['started']) * 1000
            pending = _GC['pending']
            if ms >= GC_LOG_MS and len(pending) < GC_PENDING_MAX:
                pending.append((info.get('generation'), info.get('collected'), ms))
        except BaseException:
            pass

    gc.callbacks.append(callback)
    _GC['callback'] = callback
    _GC['installed'] = True
    return True


def uninstall_gc_log():
    """Remove the callback and forget what it held (tests; a process never needs it)."""
    import gc

    callback = _GC['callback']
    if callback is not None:
        try:
            gc.callbacks.remove(callback)
        except ValueError:
            pass
    _GC.update(installed=False, started=0.0, callback=None, pending=[])


def drain_gc_lines():
    """Write one [PINDIAG] gc line per collection the callback recorded since the last drain. Never raises."""
    try:
        pending = _GC['pending']
        taken = pending[:]
        del pending[:len(taken)]
        for generation, collected, ms in taken:
            log_line('%s gen=%s collected=%s ms=%.2f' % (GC_MARKER, generation, collected, ms))
    except BaseException:
        pass


def log_line(text):
    """One line into the server log: loguru where it exists, stderr otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, file=sys.stderr, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:
        pass


def same_value(expected, actual):
    """Whether two host tensors hold the same values in the same shape and dtype."""
    import torch

    return (actual.dtype == expected.dtype and tuple(actual.shape) == tuple(expected.shape)
            and bool(torch.equal(actual, expected)))


def same_readback(expected, actual):
    """Whether a chip's readback holds the staged values (element for element; float64 holds
    every int32, uint32 and bfloat16 value exactly)."""
    import torch

    expected, actual = expected.reshape(-1), actual.reshape(-1)
    if actual.numel() != expected.numel():
        return False
    try:
        return bool(torch.equal(actual.to(torch.float64), expected.to(torch.float64)))
    except (RuntimeError, TypeError):
        return actual.tolist() == expected.tolist()


class Snapshot:
    """What one pre-stage wrote: every destination of the fixture's staging list, in order, and
    the raw host value written into each - None for the tokens, never pre-staged."""

    __slots__ = ('epoch', 'destinations', 'values', 'buffers', 'ms', 'round', 'local', 'cpu_ms', 'key')

    def __init__(self, epoch_value, destinations, values, buffers, ms, round_number, local=0, cpu_ms=0.0, key=None):
        self.epoch, self.destinations, self.values = epoch_value, tuple(destinations), list(values)
        self.buffers, self.ms, self.round = buffers, ms, round_number
        # tp4/hostgap: the fixture's own epoch when the block epochs are engaged (always 0 otherwise, so `usable` is today's check).
        self.local, self.cpu_ms = local, cpu_ms
        # tp4/round-host KEYED (round_host.KEYED_FLAG): what the values were computed from (KeyedInputs), None otherwise.
        self.key = key


class KeyedInputs:
    """tp4/round-host KEYED: the inputs every staged value but the tokens is a pure function of, as the pre-stage computed them: per segment
    the start and a COPY of the page table (engine.pages is rewritten in place by the entry's page refresh), the reader objects whose words and
    tables are among the values, the index of the tokens destination, and whether the T2 K/V guard (a function of positions and tables alone)
    would pass. Built only under the flag."""

    __slots__ = ('starts', 'tables', 'readers', 'token_index', 'kv_ok')

    def __init__(self, starts, tables, readers, token_index, kv_ok):
        self.starts, self.tables, self.readers = tuple(starts), tuple(tables), tuple(readers)
        self.token_index, self.kv_ok = token_index, kv_ok


class BlockPrestage:
    """The pre-stage state of one PackedVerifierEngine (built by it under QWEN_FAST_PRESTAGE)."""

    def __init__(self, block, *, audit):
        self.block, self.audit = block, bool(audit)
        self.snapshot = None
        self.dropped = 'no-snapshot'
        # The host tensors of the last unfenced write, kept alive until the next write replaces
        # them - after at least one fence (F9, or the verify's blocking trace).
        self.inflight = ()
        self.cursor = 0
        self.counts = dict(prestaged=0, dropped=0, diff=0, full=0, audited=0, mismatches=0)
        # The last verify's split, for serving_packed_step's fence line.
        self.last = dict(path='off', buffers=0, reason='-', prestage_ms=0.0, diff_ms=0.0, write_ms=0.0)
        # tp4/hostgap: the full read-back audit (every destination, then the round staged in full) and the thread CPU time of
        # the last verify-time write, for the stage 0 line.
        self.full_audit = full_audit_enabled()
        self.last_cpu_ms = 0.0
        if self.full_audit:
            self.counts.update(full_audited=0, full_mismatches=0)

    # -- the window ---------------------------------------------------------------------------
    def drop(self, reason):
        self.snapshot = None
        self.dropped = str(reason).replace(' ', '_')[:120]
        self.counts['dropped'] += 1

    def users_for(self, requests):
        """The next round's users in segment order, from the live requests' frontiers and page
        tables - (placeholder tokens, start, table) per live segment, idle segments filled as
        the verify fills them - and the live count. Raises for a round this block will not
        serve as one pass."""
        block = self.block
        live = [request for request in requests if not getattr(request.session, 'finished', False)]
        users = [None] * block.users
        for request in live:
            segment = block.segment_of(request.engine)
            if users[segment] is not None:
                raise ValueError('two live requests through segment %d' % segment)
            users[segment] = ((0,) * block.rows_per_user, request.session.position, request.engine.pages)
        segments = tuple(segment for segment, user in enumerate(users) if user is not None)
        if len(segments) != block.users:
            if not block.pads(len(segments)):
                raise ValueError('%d live requests is not a round of this block' % len(segments))
            users = block.padded_users(users, segments)
        return users, len(segments)

    def prestage_requests(self, requests):
        """The window's pre-stage for these requests (serving_worker_hook, through WhileWaiting)."""
        users, live = self.users_for(requests)
        self.prestage(users, live=live)

    def prestage(self, users, live=None):
        """Write every verify input but the tokens for `users` (segment order), validate the
        bindings, and keep the snapshot. No fence: the window's F9 follows."""
        from packed_verifier import packed_values, write_packed

        block = self.block
        self.snapshot = None
        self.dropped = 'prestage-incomplete'
        round_number = block.rounds + 1
        if block.phase != 'idle':
            raise ValueError('the block is %s, not idle' % block.phase)
        # Invalidates any older snapshot before the first copy: a pre-stage that fails part way
        # leaves nothing a verify could diff against.
        # tp4/hostgap, per-block epochs: the bump is this fixture's own and comes after the values (their destinations are
        # first checked against the other block's), still before the first copy; every other mode bumps first, as before.
        per_block = _MODE['blocks']
        if not per_block:
            bump('prestage')
        started = time.perf_counter()
        cpu_started = thread_ms()
        block.validate_bindings()
        values, readers = packed_values(block.operations, block.model, block.fixture, block.shape, users, guard=False)
        if per_block:
            import packed_verifier

            register_destinations(block, [value[0] for value in values],
                                  lambda destination: packed_verifier.addresses(block.operations, destination))
            bump_fixture(block.fixture, 'prestage')
        tokens = block.fixture.tokens
        indices = [index for index, value in enumerate(values) if value[0] is not tokens]
        self.inflight = write_packed(block.operations, block.model, values, readers, indices=indices, fence=False,
                                     poison=False)
        key = self.keyed_inputs(users, values, readers) if round_host.keyed_enabled() else None
        ms = (time.perf_counter() - started) * 1000
        self.snapshot = Snapshot(epoch(), [value[0] for value in values],
                                 [None if value[0] is tokens else value[1:] for value in values],
                                 len(indices), ms, round_number, local=local_epoch(block.fixture),
                                 cpu_ms=thread_ms() - cpu_started, key=key)
        self.dropped = None
        self.counts['prestaged'] += 1
        log_line('%s round=%d buffers=%d ms=%.2f live=%s' % (WINDOW_MARKER, round_number, len(indices), ms,
                                                            '-' if live is None else live))

    # -- the verify ---------------------------------------------------------------------------
    def usable(self):
        """(snapshot, None) when the snapshot can serve this verify, else (None, reason)."""
        snapshot = self.snapshot
        if snapshot is None:
            return None, self.dropped or 'no-snapshot'
        if snapshot.epoch != epoch():
            return None, 'epoch:%s' % last_bump()
        if snapshot.local != local_epoch(self.block.fixture):
            return None, 'epoch:%s' % last_local_bump(self.block.fixture)
        return snapshot, None

    def keyed_inputs(self, users, values, readers):
        """tp4/round-host KEYED, at the pre-stage: what its values were computed from (KeyedInputs), or None when this block's values hold no
        tokens destination to key on. `users` are the pre-stage's (placeholder tokens, start, table) per segment; the tables are copied."""
        block = self.block
        fixture = block.fixture
        token_index = next((index for index, value in enumerate(values) if value[0] is fixture.tokens), None)
        if token_index is None:
            return None
        kv_ok = True
        if getattr(fixture, 'kv_chains', False):
            # The T2 K/V guard (packed_values keeps it for every verify-time call) is a function of the positions and the tables alone: its
            # verdict for this key is taken now, without raising (a conflict declines the key; the diff then raises exactly as today).
            import verify_trace_t2

            positions = next(value[1] for value in values if value[0] is fixture.positions)
            pages = next(value[1] for value in values if value[0] is fixture.pages)
            guarded = verify_trace_t2.block_users(positions, pages, block.shape.rows_per_user, block.shape.users)
            kv_ok = verify_trace_t2.kv_conflict(guarded) is None
        return KeyedInputs([user[1] for user in users], [user[2].clone() for user in users], readers, token_index, kv_ok)

    def key_matches(self, snapshot, users):
        """tp4/round-host KEYED, at the verify: whether every staged value but the tokens is still the snapshot's - the key (start and page
        table per segment, the readers) is what the values are a pure function of, and it is compared as VALUES against the verify-time
        tickets and tables. Host only; False declines to today's value diff."""
        key = snapshot.key
        block = self.block
        if key is None or not key.kv_ok or len(users) != len(key.starts):
            return False
        fixture = block.fixture
        if snapshot.destinations[key.token_index] is not fixture.tokens:
            return False
        reader = getattr(fixture, 'replay_reader', None)
        readers = () if reader is None else tuple(reader.readers)
        if len(readers) != len(key.readers) or any(own is not kept for own, kept in zip(readers, key.readers)):
            return False
        if tuple(user[1] for user in users) != key.starts:
            return False
        if any(not same_value(kept, user[2]) for kept, user in zip(key.tables, users)):
            return False
        import verify_trace_t2

        # The T2 audit's diagnostic lines come from packed_values: an audited verify takes the diff that writes them.
        return not verify_trace_t2.audit_enabled()

    def keyed_write(self, snapshot, users):
        """tp4/round-host KEYED: write the tokens buffer alone (the one value the key does not hold), exactly the write the value diff would
        have made when only the tokens differ. The tokens are validated and built by packed_verifier.packed_host_tokens (packed_host_inputs'
        own calls), each reader validates its start before any copy as packed_values has them do. Returns (written, readers)."""
        from packed_verifier import packed_host_tokens, write_packed

        block = self.block
        operations = block.operations
        tokens = packed_host_tokens(users, block.shape, block.model.args.vocab_size)
        readers = snapshot.key.readers
        for own, user in zip(readers, users, strict=True):
            own.validate(user[1])
        value = (block.fixture.tokens, tokens, operations.uint32, operations.ROW_MAJOR_LAYOUT)
        try:
            self.inflight = write_packed(operations, block.model, [value], readers, indices=[0], fence=False)
        except BaseException:
            bump('verify-failed')
            raise
        return 1, readers

    def audit_keyed(self, snapshot, values, changed, users):
        """tp4/round-host AUDIT, on a round whose key stood: the claim KEYED rests on, checked against the diff that ran (and whose write
        stands) - the tokens destination is the only one that differs from the snapshot, and the tokens KEYED would have written are the
        diff's own. Logs [ROUND-HOST-AUDIT] kind=keyed; the smoke check fails the arm on equal=0."""
        from packed_verifier import packed_host_tokens

        block = self.block
        key = snapshot.key
        equal = changed == [key.token_index]
        if equal:
            import torch

            keyed = packed_host_tokens(users, block.shape, block.model.args.vocab_size)
            kept = values[key.token_index][1]
            equal = keyed.dtype == kept.dtype and tuple(keyed.shape) == tuple(kept.shape) and bool(torch.equal(keyed, kept))
        round_host.count('audit_equal' if equal else 'audit_unequal')
        if equal:
            round_host.note_path('keyed')       # the audited arm's ledger counts the rounds whose key stood and was confirmed
        round_host.log_line('%s kind=keyed equal=%d checked=%d changed=%s' % (
            round_host.AUDIT_MARKER, int(equal), len(values), ','.join(str(index) for index in changed[:8])))

    def stage(self, entries, segments, snapshot, reason):
        """The verify-time write: the diff against `snapshot` when there is one, else today's
        full stage_packed. Returns the buffers written; logs MARKER; audits."""
        from packed_verifier import packed_values, write_packed

        block = self.block
        round_number = block.rounds + 1
        users = block.segment_users(entries, segments)
        if len(segments) < block.users:
            users = block.padded_users(users, segments)
        self.snapshot = None
        self.dropped = 'no-snapshot'
        prestage_ms = 0.0 if snapshot is None else snapshot.ms
        started = time.perf_counter()
        cpu_started = thread_ms()
        values = None
        # tp4/round-host KEYED: the key still holds, so the tokens are the only value that moved (audited, the diff below runs and the
        # claim is checked against it instead).
        would_key = snapshot is not None and round_host.keyed_enabled() and self.key_matches(snapshot, users)
        keyed = would_key and not round_host.audit_enabled()
        if keyed:
            diffed = time.perf_counter()
            written, readers = self.keyed_write(snapshot, users)
            for own, user in zip(readers, users, strict=True):
                own.start = user[1]
            bump_fixture(block.fixture, 'verify')
            path, reason = 'diff', '-'
            self.counts['diff'] += 1
            self.counts['keyed'] = self.counts.get('keyed', 0) + 1
            round_host.count('keyed')
            round_host.note_path('keyed')
        elif snapshot is not None:
            if round_host.keyed_enabled() and snapshot.key is not None and not would_key:
                round_host.count('keyed_declined')
            values, readers = packed_values(block.operations, block.model, block.fixture, block.shape, users)
            if len(values) != len(snapshot.destinations) or any(
                    value[0] is not destination for value, destination in zip(values, snapshot.destinations)):
                snapshot, reason = None, 'destinations'
        if keyed:
            pass
        elif snapshot is None:
            diffed = started
            written = block.stage_packed_inputs(entries)
            path = 'full'
            self.counts['full'] += 1
        else:
            changed = [index for index, (value, kept) in enumerate(zip(values, snapshot.values))
                       if kept is None or kept[1:] != value[2:] or not same_value(kept[0], value[1])]
            if would_key:
                self.audit_keyed(snapshot, values, changed, users)
            diffed = time.perf_counter()
            try:
                self.inflight = write_packed(block.operations, block.model, values, readers, indices=changed,
                                             fence=False)
            except BaseException:
                bump('verify-failed')
                raise
            for own, user in zip(readers, users, strict=True):
                own.start = user[1]
            bump_fixture(block.fixture, 'verify')
            written, path, reason = len(changed), 'diff', '-'
            self.counts['diff'] += 1
        finished = time.perf_counter()
        self.last_cpu_ms = thread_ms() - cpu_started
        self.last = dict(path=path, buffers=written, reason=reason, prestage_ms=prestage_ms,
                         diff_ms=(diffed - started) * 1000, write_ms=(finished - diffed) * 1000)
        log_line('%s round=%d path=%s buffers=%d reason=%s live=%d' % (
            MARKER, round_number, path, written, str(reason).replace(' ', '_')[:120], len(segments)))
        if self.audit:
            self.audit_round(users, round_number, path)
        if self.full_audit:
            self.full_audit_round(values, users, round_number, path)
        return written

    def full_audit_round(self, values, users, round_number, path='diff'):
        """QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT, after a verify-time write: EVERY destination read back from every chip
        and compared with the verify-time (full-stage) value. After a DIFF write the round is then staged in full anyway, so
        the trace sees what today's verify sees whatever the audit found; a mismatch is logged and fails the arm at the gate.
        After a FULL write (path=full, `values` is None: the full stage is the reference itself) the same read-back runs
        against the values the verify computes now: a mismatch there is an artifact of the comparator (layout, padding, dtype),
        never of the lever, and the smoke judges the two paths separately. Nothing is restaged after a full write."""
        from packed_verifier import packed_values, stage_packed

        block = self.block
        operations = block.operations
        if values is None:
            values, unused = packed_values(operations, block.model, block.fixture, block.shape, users, guard=False)
        mismatched, checked = [], 0
        for index, value in enumerate(values):
            for chip, shard in enumerate(operations.get_device_tensors(value[0])):
                checked += 1
                if not same_readback(value[1], operations.to_torch(shard)):
                    mismatched.append('%d.%d' % (index, chip))
        self.counts['full_audited'] += checked
        self.counts['full_mismatches'] += len(mismatched)
        log_line('%s block=%s round=%d path=%s buffers=%d checked=%d mismatches=%d%s' % (
            FULL_AUDIT_MARKER, block_label(block), round_number, path, len(values), checked, len(mismatched),
            (' at=%s' % ','.join(mismatched[:8])) if mismatched else ''))
        if path == 'diff':
            stage_packed(operations, block.model, block.fixture, block.shape, users)

    def audit_round(self, users, round_number, path):
        """QWEN_FAST_PRESTAGE_AUDIT: AUDIT_BUFFERS destinations in rotation, read back from both
        chips, against the verify-time values; a mismatch restages the round in full."""
        from packed_verifier import packed_values, stage_packed

        block = self.block
        operations = block.operations
        values, unused = packed_values(operations, block.model, block.fixture, block.shape, users, guard=False)
        count = len(values)
        # `first` is where this round's rotation starts (the marker's first=), not the smallest
        # index checked: past the end of the list the eight wrap round to index 0.
        first = self.cursor % count if count else -1
        indices = sorted({(self.cursor + offset) % count for offset in range(min(AUDIT_BUFFERS, count))})
        self.cursor = (self.cursor + AUDIT_BUFFERS) % count if count else 0
        mismatched = []
        for index in indices:
            destination, value = values[index][0], values[index][1]
            for chip, shard in enumerate(operations.get_device_tensors(destination)):
                if not same_readback(value, operations.to_torch(shard)):
                    mismatched.append('%d.%d' % (index, chip))
        self.counts['audited'] += len(indices)
        self.counts['mismatches'] += len(mismatched)
        log_line('%s round=%d path=%s checked=%d first=%d mismatches=%d%s' % (
            AUDIT_MARKER, round_number, path, len(indices), first, len(mismatched),
            (' at=%s' % ','.join(mismatched[:8])) if mismatched else ''))
        if mismatched:
            # The round must stay exact whatever the audit found: every buffer again, fenced
            # (stage_packed bumps the epoch and sets every reader's start).
            stage_packed(operations, block.model, block.fixture, block.shape, users)


class WhileWaiting:
    """What the drafts' fence window runs for one packed block: `coordinator.prepare(...,
    while_waiting=this)` calls it just before its fence F9, after both pairs are enqueued, and
    `fenced()` right after that fence. Built by serving_packed_step.PackedStep.while_waiting only
    when the coming round is this block's packed round and a flag is on."""

    def __init__(self, block, requests, prestage=True):
        self.block, self.requests = block, list(requests)
        self.token = None
        # False: this window is one of several (CompositeWindow) and the verify pre-stage is left out of it - see
        # serving_packed_step.PackedStep.while_waiting_groups. True, the default, is the one block's window as it was.
        self.prestage = prestage
        # tp4/hostgap (QWEN_FAST_TP4_HOSTGAP_LOG): this window's own timing, read by the line `window_line` writes; a window inside a
        # CompositeWindow leaves the line to it.
        self.timing = dict(stage_window_ms=0.0, prestage_ms=0.0, prestaged=0, ended=0.0)
        self.logs_own_line = True

    def __call__(self):
        block = self.block
        if getattr(block, 'round_fences', False):
            # Taken before the fence: only commits enqueued before it may be armed by it.
            self.token = block.fence_token()
        fused = getattr(block, 'fused', None)
        started = time.perf_counter()
        if fused is not None:
            # Round-fence plan H1b (fused_commit.py): the next round's T_proj RoPE tables for each
            # live segment. Never raises (a segment that fails is staged at its commit), so the
            # pre-stage below always runs.
            fused.stage_window(self.requests)
        staged_window = time.perf_counter()
        prestaged = getattr(block, 'prestaged', None)
        try:
            if prestaged is not None and self.prestage:
                prestaged.prestage_requests(self.requests)
                self.timing['prestaged'] = 1
        finally:
            self.timing.update(stage_window_ms=(staged_window - started) * 1000,
                               prestage_ms=(time.perf_counter() - staged_window) * 1000, ended=time.perf_counter())

    def drop(self, failure):
        """The pre-stage raised (the coordinator catches it): its snapshot is gone and the round
        goes on - the verify takes the full path."""
        prestaged = getattr(self.block, 'prestaged', None)
        if prestaged is None:
            return
        reason = 'prestage-failed:%s' % type(failure).__name__
        prestaged.drop(reason)
        log_line('%s round=%d dropped=%s detail=%s' % (WINDOW_MARKER, self.block.rounds + 1, reason,
                                                       str(failure).replace(' ', '_')[:120]))

    def fenced(self):
        """F9 has just drained CQ0: arm the retained block's replay (round fences)."""
        if self.logs_own_line and hostgap_log_enabled():
            window_line([self])
        if self.token is not None:
            self.block.note_round_fence(self.token)
            self.token = None


class CompositeWindow:
    """QWEN_FAST_M3_BLOCKS=2: the drafts' fence window over SEVERAL packed blocks, one WhileWaiting each, in the blocks'
    order. `coordinator.prepare(..., while_waiting=this)` calls it once before its fence and `fenced()` once after, as it
    does a single block's window; a window that raises is dropped on its own (its `drop`) and the others still run, so
    one block's failed window never costs another block its round fence."""

    def __init__(self, windows):
        self.windows = list(windows)
        if not self.windows:
            raise ValueError('A composite window needs at least one block window')
        self.ended = 0.0
        for window in self.windows:
            window.logs_own_line = False

    def __call__(self):
        try:
            self.run_windows()
        finally:
            self.ended = time.perf_counter()

    def run_windows(self):
        for window in self.windows:
            try:
                window()
            except Exception as failure:
                try:
                    window.drop(failure)
                except Exception:
                    pass

    def drop(self, failure):
        for window in self.windows:
            try:
                window.drop(failure)
            except Exception:
                pass

    def fenced(self):
        """F9 has just drained CQ0: every block's round fence is armed; the first failure is raised after the rest ran."""
        first = None
        if hostgap_log_enabled():
            window_line(self.windows, self.ended)
        for window in self.windows:
            try:
                window.fenced()
            except Exception as failure:
                first = first or failure
        if first is not None:
            raise first


def window_line(windows, ended=None):
    """Stage 0 / D3 (QWEN_FAST_TP4_HOSTGAP_LOG): one line a round for the drafts' window, written right after its fence F9 -
    each block's T_proj table staging and pre-stage in ms (and whether it pre-staged), the window's host total, and how long the
    host then waited in the fence. A fence_wait_ms near 0 with a long window_ms is a window that overran the quads' device
    time: the pre-stages cost the lever device idle. Never raises."""
    drain_gc_lines()
    try:
        now = time.perf_counter()
        if ended is None:
            ended = max(window.timing['ended'] for window in windows)
        timing = [window.timing for window in windows]
        log_line('%s round=%d blocks=%d stage_window_ms=%s prestage_ms=%s prestaged=%s window_ms=%.2f fence_wait_ms=%.2f' % (
            HOSTGAP_WINDOW_MARKER, windows[0].block.rounds + 1, len(windows),
            ','.join('%.2f' % item['stage_window_ms'] for item in timing),
            ','.join('%.2f' % item['prestage_ms'] for item in timing),
            ','.join(str(item['prestaged']) for item in timing),
            sum(item['stage_window_ms'] + item['prestage_ms'] for item in timing), (now - ended) * 1000))
    except BaseException:
        pass
