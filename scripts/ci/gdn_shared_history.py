"""One GDN per-token state history shared by both 64-row blocks (QWEN_FAST_GDN_SHARED_HISTORY, default off, gate profiles only).

Design and proof: docs/tp4-gdn-shared-history.md. In short: every GDN layer's verify launch writes a (16, 12, 128, 128) bf16
history per packed user (6 MiB per chip); only the deferred commit reads it. The two eight-seat blocks verify one after the
other, so once block A's commits have read its history, block B may write the same buffers. This module owns that one set
(4 users x 48 layers = 192 tensors) and the rule that makes it exact: a block's history is never overwritten before every
commit of the block that wrote it has been enqueued on the one command queue (`claim`).

What this module does and does not touch:

- `SharedHistory.capture(block)` is opened by the block around its VERIFY capture only. Inside it `active()` is the pool and
  `gdn_user_batch_tp.execute` takes each launch's states from it. The warm forward, a sequential engine's capture and every
  other launch run with `active()` None and allocate privately, exactly as before.
- The first block to capture allocates the 192 tensors, in launch order; the second is handed the same tensors in the same
  order after each one's shape is checked, and its capture must ask for exactly as many.
- The pool, not the blocks, owns the tensors. `holds(tensor)` is what `tp_addresses.release_owned` consults (through
  sys.modules, so nothing imports this module unless the flag is on) to leave them alone; the last `detach` frees them.
- `claim(block)` is the commit-before-reuse guard `PackedVerifierEngine.verify` calls: deferred commits of the block that last
  wrote the history are flushed first (same `flush_commits` the engine already has, site `shared-history`); a block with
  undecided segments refuses the round.

QWEN_FAST_GDN_SHARED_HISTORY unset or '0': off, '1': on, anything else is refused (never read as off). Refused at attach unless
serving is four cards with two M3 blocks and the K5-A launch is off. QWEN_FAST_GDN_SHARED_HISTORY_KV_GROW is a separate,
gate-only marker for a profile whose KV pool was sized with the freed bytes (`pool_problem`); it never changes a pool by itself.

Stdlib only; overlay only.
"""

from contextlib import contextmanager
import os

FLAG = 'QWEN_FAST_GDN_SHARED_HISTORY'
GROW_FLAG = 'QWEN_FAST_GDN_SHARED_HISTORY_KV_GROW'
FLAGS = (FLAG, GROW_FLAG)

# The served 8-seat shape: two blocks of four 16-row users, 48 GDN layers, 12 value heads and 128 x 128 states a chip.
BLOCKS = 2
USERS = 4
ROWS_PER_USER = 16
BLOCK_ROWS = USERS * ROWS_PER_USER
SOLO_SHAPE = (1, ROWS_PER_USER)
GDN_LAYERS = 48
HEADS = 12
STATE_SIDE = 128
BYTES_PER_VALUE = 2
TENSORS = USERS * GDN_LAYERS
# KV: a 64-token block is 557,056 B a chip (the profiles' own arithmetic, test_tp4_262k8_best_profiles).
KV_BLOCK_BYTES = 557056
KV_BLOCK_TOKENS = 64
# The production pool and the high-band edge its DRAM arithmetic allows (test_tp4_262k8_best_profiles.test_the_dram_arithmetic).
PRODUCTION_POOL_BLOCKS = 19968
PRODUCTION_POOL_EDGE = 19979
SEATS = 8
WINDOW_TOKENS = 262144
# A full 262k reservation is its window's blocks plus the spare (serving_kv_reservation.request_blocks at the window).
FULL_WINDOW_BLOCKS = WINDOW_TOKENS // KV_BLOCK_TOKENS + 1
TARGET_WINDOWS = 5
POOL_TOKENS_ENV = 'QWEN36_MAX_TOKENS_ALL_USERS'

ENGAGED_MARKER = '[PINDIAG] gdn shared history engaged'
FLUSH_MARKER = '[PINDIAG] gdn shared history flush'
CLOSED_MARKER = '[PINDIAG] gdn shared history closed'
REFUSED_MARKER = '[PINDIAG] gdn shared history refused'


def _strict(flag, environ=None):
    value = (os.environ if environ is None else environ).get(flag, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (flag, value))
    return value == '1'


def enabled(environ=None):
    """QWEN_FAST_GDN_SHARED_HISTORY, strictly: unset or '0' is off, '1' is on, anything else is refused."""
    return _strict(FLAG, environ)


def grow_enabled(environ=None):
    """QWEN_FAST_GDN_SHARED_HISTORY_KV_GROW, strictly (see the module docstring: a marker, not an action)."""
    return _strict(GROW_FLAG, environ)


def requested(environ=None):
    """Whether either flag is set to anything but '0' (the packed verifier's cheap test before it imports this module)."""
    source = os.environ if environ is None else environ
    return any(source.get(flag, '0') != '0' for flag in FLAGS)


# ---- accounting -----------------------------------------------------------------------------------------------------------

def history_bytes_per_user(layers=GDN_LAYERS, rows=ROWS_PER_USER, heads=HEADS):
    """One user's history over every GDN layer on one chip: 16 x 12 x 128 x 128 x 2 B x 48 = 288 MiB."""
    return rows * heads * STATE_SIDE * STATE_SIDE * BYTES_PER_VALUE * layers


def history_bytes_per_block(users=USERS):
    return users * history_bytes_per_user()


def freed_bytes(blocks=BLOCKS, users=USERS):
    """What sharing one history set over `blocks` blocks frees on each chip: every block but the one that owns it."""
    if type(blocks) is not int or blocks < 1:
        raise ValueError('At least one block required')
    return (blocks - 1) * history_bytes_per_block(users)


def kv_blocks_gained(freed=None):
    """Whole KV blocks the freed bytes hold (floor)."""
    return (freed_bytes() if freed is None else freed) // KV_BLOCK_BYTES


def grown_pool_edge():
    """The largest pool the production DRAM arithmetic allows once the freed bytes are spent, rounded down to 64 blocks (the
    profiles' own rounding)."""
    return (PRODUCTION_POOL_EDGE + kv_blocks_gained()) // 64 * 64


def windows_that_fit(pool_blocks):
    """Whole full-262k reservations a pool admits (its usable blocks are the pool less vLLM's null block)."""
    return max(pool_blocks - 1, 0) // FULL_WINDOW_BLOCKS


def pool_tokens_for(pool_blocks, seats=SEATS):
    """QWEN36_MAX_TOKENS_ALL_USERS that makes the worker build `pool_blocks` (ceil(tokens / 64) + seats)."""
    return (pool_blocks - seats) * KV_BLOCK_TOKENS


def pool_problem(profile):
    """What is wrong with how a profile uses the growth marker; None when nothing (and for every profile without it)."""
    env = profile.get('env', {})
    if env.get(GROW_FLAG, '0') == '0' and env.get(FLAG, '0') == '0':
        return None
    for flag in FLAGS:
        if env.get(flag, '0') not in ('0', '1'):
            return '%s must be 0 or 1, got %r' % (flag, env.get(flag))
    if not profile.get('gate_only'):
        return 'profile %s sets %s or %s but is not gate_only: the shared history is a gate arm until the card window passes' % (
            profile.get('name'), FLAG, GROW_FLAG)
    if env.get(GROW_FLAG, '0') != '1':
        return None
    if env.get(FLAG, '0') != '1':
        return '%s=1 needs %s=1: the pool may only grow by bytes the sharing freed' % (GROW_FLAG, FLAG)
    engine = profile.get('engine', {})
    blocks, seats = engine.get('num-gpu-blocks-override'), engine.get('max-num-seqs')
    if type(blocks) is not int or seats != SEATS:
        return '%s=1 needs an eight-seat profile with an integer num-gpu-blocks-override, got %r blocks and %r seats' % (
            GROW_FLAG, blocks, seats)
    if blocks <= PRODUCTION_POOL_BLOCKS:
        return '%s=1 but the pool is %d blocks, not above the production %d: nothing was grown' % (
            GROW_FLAG, blocks, PRODUCTION_POOL_BLOCKS)
    limit = grown_pool_edge()
    if blocks > limit:
        return '%s=1 names a pool of %d blocks; the freed %d bytes allow %d at most' % (
            GROW_FLAG, blocks, freed_bytes(), limit)
    if blocks % 64:
        return '%s=1 names a pool of %d blocks, not a multiple of 64' % (GROW_FLAG, blocks)
    tokens = env.get(POOL_TOKENS_ENV)
    if tokens is None or not str(tokens).isdigit() or int(tokens) != pool_tokens_for(blocks, seats):
        return '%s must be %d for a pool of %d blocks, got %r' % (POOL_TOKENS_ENV, pool_tokens_for(blocks, seats), blocks, tokens)
    if windows_that_fit(blocks) < TARGET_WINDOWS:
        return 'a pool of %d blocks admits %d full 262k reservations, not %d' % (blocks, windows_that_fit(blocks), TARGET_WINDOWS)
    return None


# ---- refusals ---------------------------------------------------------------------------------------------------------------

def refusal(environ=None):
    """Why the sharing cannot be turned on in this environment, or None (also None with both flags off). Raises on a malformed
    flag value."""
    source = os.environ if environ is None else environ
    on, grow = enabled(source), grow_enabled(source)
    if grow and not on:
        return '%s=1 needs %s=1' % (GROW_FLAG, FLAG)
    if not on:
        return None
    if source.get('QWEN_FAST_TP', '2') != '4':
        return '%s needs four-card serving (QWEN_FAST_TP=4)' % FLAG
    if source.get('QWEN_FAST_M3_BLOCKS', '1') != '2':
        return '%s needs two M3 blocks (QWEN_FAST_M3_BLOCKS=2): one block has nothing to share with' % FLAG
    if source.get('QWEN_FAST_GDN_SEQ_BLOCK', '0') not in ('', '0'):
        return ('%s is refused with the K5-A launch (QWEN_FAST_GDN_SEQ_BLOCK): its states are allocated inside pinned sources '
                'and are not plumbed' % FLAG)
    return None


# ---- the pool ---------------------------------------------------------------------------------------------------------------

_POOL = None
_ACTIVE = None
_HELD = {}


def active():
    """The pool a verify capture is open on, else None (every other launch allocates privately)."""
    return _ACTIVE


def holds(tensor):
    """Whether the pool owns `tensor` (tp_addresses.release_owned leaves such tensors to it)."""
    return id(tensor) in _HELD


def current():
    return _POOL


def reset():
    """Drop the process-wide pool without freeing anything (tests only)."""
    global _POOL, _ACTIVE
    _POOL, _ACTIVE = None, None
    _HELD.clear()


def join(block, *, log=None, environ=None):
    """The shared pool `block` attaches to, or None with the flag off. A flag the environment cannot honour, or a block the
    sharing is not built for, raises ValueError (fail closed, at attach), logged first."""
    global _POOL
    problem = refusal(environ)
    if problem is None and not enabled(environ):
        return None
    if problem is not None:
        if log is not None:
            log('%s reason=%s' % (REFUSED_MARKER, problem))
        raise ValueError(problem)
    if (getattr(block, 'users', None), getattr(block, 'rows_per_user', None)) == SOLO_SHAPE:
        # D0's one-user block (QWEN_FAST_SOLO_LANE) is a separate block with its own history: it never shares.
        return None
    if _POOL is None or _POOL.closed:
        _POOL = SharedHistory(log=log)
    _POOL.attach(block)
    return _POOL


class _Held:
    """The pool's references to the blocks and the device operations. Slots only, so `vars()` fails on it: the memory ledger walks
    every instance dict reachable from a block it itemises, and the pool (reachable from each block) must not lead it into the OTHER
    block's buffers, which would shift one block's bytes onto the other's line."""

    __slots__ = ('blocks', 'writer', 'builder', 'capturing', 'operations')


def _held(name):
    return property(lambda self: getattr(self._held, name), lambda self, value: setattr(self._held, name, value))


class SharedHistory:
    blocks, writer, builder, capturing, operations = (_held(name) for name in _Held.__slots__)

    def __init__(self, *, log=None):
        self.log = log
        self._held = _Held()
        self.blocks = []
        self.tensors = []
        self.shape = None
        self.builder = None
        self.expected = None
        self.capturing = None
        self.cursor = 0
        self.writer = None
        self.operations = None
        self.closed = False
        self.counts = dict(captures=0, reused=0, claims=0, flushes=0, flushed_commits=0)

    def say(self, line):
        if self.log is not None:
            self.log(line)

    # -- membership
    def attach(self, block):
        if self.closed:
            raise ValueError('The shared history is closed')
        shape = (getattr(block, 'users', None), getattr(block, 'rows_per_user', None), getattr(block, 'block_rows', None))
        if shape != (USERS, ROWS_PER_USER, BLOCK_ROWS):
            raise ValueError('The shared history is built for %d-user %d-row-per-user %d-row blocks, not %r'
                             % (USERS, ROWS_PER_USER, BLOCK_ROWS, shape))
        if any(known is block for known in self.blocks):
            raise ValueError('This block is already attached to the shared history')
        if len(self.blocks) >= BLOCKS:
            raise ValueError('The shared history serves %d blocks; a third block cannot attach' % BLOCKS)
        self.blocks.append(block)

    def detach(self, block, operations=None):
        """The block closed: release its claim. The last one out frees the tensors."""
        global _ACTIVE
        if not any(known is block for known in self.blocks):
            return
        self.blocks = [known for known in self.blocks if known is not block]
        if self.writer is block:
            self.writer = None
        if self.capturing is block:
            self.capturing = None
            if _ACTIVE is self:
                _ACTIVE = None
        if self.blocks:
            return
        self.release(operations if operations is not None else self.operations)

    def release(self, operations):
        global _ACTIVE
        count, freed = len(self.tensors), len(self.tensors) * self.tensor_bytes()
        seen = set()
        for tensor in self.tensors:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            _HELD.pop(id(tensor), None)
            if operations is not None:
                operations.deallocate(tensor)
        self.tensors, self.closed = [], True
        if _ACTIVE is self:
            _ACTIVE = None
        self.say('%s tensors=%d bytes_per_chip=%d' % (CLOSED_MARKER, count, freed))

    def tensor_bytes(self):
        if self.shape is None:
            return 0
        total = BYTES_PER_VALUE
        for side in self.shape:
            total *= side
        return total

    # -- capture
    @contextmanager
    def capture(self, block, operations):
        """Open the pool around `block`'s verify capture. The first block to capture builds the tensors; the other is handed
        them. A capture that raises leaves what it built to the pool's release (the block's close)."""
        global _ACTIVE
        if self.closed or not any(known is block for known in self.blocks):
            raise ValueError('Only an attached block may capture on the shared history')
        if self.capturing is not None or _ACTIVE is not None:
            raise ValueError('A shared-history capture is already open')
        if self.builder is block:
            raise ValueError('The block that built the shared history cannot capture on it twice')
        building = self.builder is None
        self.operations = operations
        self.capturing, self.cursor, _ACTIVE = block, 0, self
        try:
            yield self
        except BaseException:
            raise
        else:
            want = TENSORS if building else self.expected
            if self.cursor != want:
                raise ValueError('The verify capture took %d history tensors; the shared set is %d' % (self.cursor, want))
            if building:
                self.builder, self.expected = block, self.cursor
            self.counts['captures'] += 1
        finally:
            self.capturing = None
            if _ACTIVE is self:
                _ACTIVE = None

    def take(self, operations, mesh, rows, heads):
        """The next launch's (rows, heads, 128, 128) bf16 DRAM history: built by the first capture, handed to the second."""
        if self.capturing is None or _ACTIVE is not self:
            raise ValueError('The shared history hands tensors out only inside a verify capture')
        shape = (rows, heads, STATE_SIDE, STATE_SIDE)
        if shape != (ROWS_PER_USER, HEADS, STATE_SIDE, STATE_SIDE):
            raise ValueError('The shared history is %r a launch; a launch asked for %r'
                             % ((ROWS_PER_USER, HEADS, STATE_SIDE, STATE_SIDE), shape))
        index = self.cursor
        self.cursor += 1
        if self.builder is None:
            if self.shape is None:
                self.shape = shape
            states = operations.empty(shape, device=mesh, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                                      memory_config=operations.DRAM_MEMORY_CONFIG)
            self.tensors.append(states)
            _HELD[id(states)] = states
            return states
        if index >= len(self.tensors):
            raise ValueError('The second capture asked for more history tensors than the first built (%d)' % len(self.tensors))
        states = self.tensors[index]
        if tuple(states.shape) != shape:
            raise ValueError('History tensor %d is %r; the launch asked for %r' % (index, tuple(states.shape), shape))
        self.counts['reused'] += 1
        return states

    # -- commit before reuse
    def claim(self, block):
        """Called by PackedVerifierEngine.verify before the block stages anything. Another block's history must be fully read
        first: its deferred commits are enqueued now (same queue, so before this block's trace); a block that has verified
        and not decided every segment refuses the round."""
        if self.closed or not any(known is block for known in self.blocks):
            raise ValueError('Only an attached block may verify on the shared history')
        other = self.writer
        if other is not None and other is not block and getattr(other, 'phase', None) not in ('failed', 'closed'):
            held = len(getattr(other, 'deferred_commits', None) or ())
            if held:
                self.counts['flushes'] += 1
                self.counts['flushed_commits'] += held
                other.flush_commits('shared-history')
            pending = history_pending(other)
            if pending:
                raise ValueError('Commit-before-reuse refused: the block that last wrote the shared history has %s'
                                 % pending)
        self.writer = block
        self.counts['claims'] += 1

    def describe(self):
        return dict(blocks=len(self.blocks), tensors=len(self.tensors), tensor_bytes=self.tensor_bytes(),
                    shared_bytes=len(self.tensors) * self.tensor_bytes(), counts=dict(self.counts), closed=self.closed)

    def engaged_line(self, block):
        """The attach marker one block logs once its construction is done: its role (the first to capture owns the build), the
        shared set and what it frees and buys."""
        freed = freed_bytes(BLOCKS)
        return ('%s role=%s blocks=%d users=%d layers=%d tensors=%d tensor_bytes=%d freed_per_chip=%d kv_blocks_gained=%d' % (
            ENGAGED_MARKER, 'owner' if self.builder is block else 'sharer', len(self.blocks), USERS, GDN_LAYERS,
            len(self.tensors), self.tensor_bytes(), freed, kv_blocks_gained(freed)))


def history_pending(block):
    """Why `block` still needs its history, or '' when every segment it verified has been committed and every commit trace
    enqueued."""
    phase = getattr(block, 'phase', None)
    if phase in ('verifying', 'verified'):
        return 'segments verified and not committed (phase %s)' % phase
    if getattr(block, 'pending_segments', None):
        return 'uncommitted segments %s' % sorted(block.pending_segments)
    if getattr(block, 'deferred_commits', None):
        return '%d deferred commit traces not enqueued' % len(block.deferred_commits)
    return ''
