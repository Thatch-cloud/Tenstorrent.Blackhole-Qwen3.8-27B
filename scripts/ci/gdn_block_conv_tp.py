"""QWEN_FAST_TP4_GDN_BLOCK_CONV (tp4/vglue V1): one convolution-and-gates launch per GDN layer for all four packed users.

Served (gdn_user_batch_conv.run_user_batched_projected), a GDN layer runs gdn_decode_conv_gates once per packed user on that
user's 16-row piece and windows: four launches per layer, the last three of them waiting on the first (192 launches, 3.97 ms of
kernels and 2.25 ms of dispatch stalls per replay). The op is row-parallel - every output row is the four-tap FIR over that
row's own window rows, a silu, and the two gates of that row's own a and b - and a work unit is one (tile row, channel tile), so
a call at batch 64 over the block, with the four windows stacked to (1, 64, C), computes every user's rows in one launch with
the same per-element instructions (the op's own validation admits it: 1 <= B <= Bx <= Bmax). Per layer this stage runs

  1. canon block   the projection block with users 1 and 3 canonicalised (what the served pieces of those users are),
  2. window stack  the T2 packed windows of the four users -> four (1, 64, C) block windows (raw),
  3. ONE gdn_decode_conv_gates(block, block windows, batch=64) -> (conv, beta, g) blocks; it advances the block windows,
  4. unstack       conv, beta, g, z (the canon block's columns 2560..4095) and the advanced windows back to the per-user
                   (1, 16, .) tensors the recurrence and the commit DMA read, rows 16-31 zero.

Steps 1, 2 and 4 are gdn_rows_dma_tp launches (three of them), so the layer's convolution costs four launches where it cost
four plus the split's and the z slices'. The user tensors are the same tensors the loop would have made (same shape, dtype,
layout, DRAM), the per-user windows are the T2 windows themselves (advanced in place, as the served op advances them), and
K5-A runs as it does.

Exactness is not by construction: the op's per-element arithmetic does not depend on where a row sits, but that a row's result
is the same in tile row 1 and in faces 2-3 as in tile row 0 rests on the SFPU treating every lane alike. QWEN_FAST_TP4_VGLUE_AUDIT
runs the served per-user path beside this one (its own windows and its own conv_gates call) and holds both for the in-trace
compare: conv, beta, g, z and every advanced window, every user, every chip, as int16 bit patterns.

Stdlib only at import (ttnn-free until launch), py 3.7.
"""

import os

import gdn_rows_dma_tp as rows_dma
import tp4_vglue
import tp_shapes
from tp_addresses import release_owned

USER_ROWS = rows_dma.USER_ROWS
SPREAD_FLAG = 'QWEN_FAST_TP4_CONV_GATES_SPREAD'            # gdn_conv_gates_spread.FLAG (a literal here so an unset flag imports nothing)
SPREAD_AUDIT_FLAG = 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT'
SLOTS = 4


class BlockStage(object):
    """What run_user_batched_projected takes in place of the per-user loop's conv gates and z slice: `users` is one
    ((conv, beta, g), z) per user, `owned` the block-level temporaries (freed with the layer), `held` the audit's tensors
    (outside `owned`), `entries` the audit entries per user."""

    def __init__(self, users, owned, held, entries):
        self.users, self.owned, self.held, self.entries = users, owned, held, entries


def plan_windows(users):
    """The window stack as tasks over sources u * 4 + slot (each user's four windows in slot order) and destinations slot."""
    tasks = []
    for slot in range(SLOTS):
        for destination, page, first, second in rows_dma.stack_users(users, tp_shapes.active().gdn_qkv):
            def remap(half):
                source, source_page, mode = half
                return (source * SLOTS + slot, source_page, mode)
            tasks.append((slot, page, remap(first), remap(second)))
    return tasks


def plan_unstack(users, width):
    """The block back to per-user tensors: sources conv, beta, g, canon block, block windows 0-3 (indices 0-7); destinations
    conv u (0-3), beta u (4-7), g u (8-11), z u (12-15), then the users' windows u * 4 + slot (16-31). `width` is the
    projection's logical width."""
    found = tp_shapes.active()
    qkv_columns = rows_dma.tile_columns(found.gdn_qkv)
    z_first = found.gdn_qkv // 32
    z_columns = (found.gdn_a_col - found.gdn_qkv) // 32
    block_columns = rows_dma.tile_columns(width)
    tasks = []

    def add(plan, source, destination_base, destination_stride=1, destination_offset=0):
        for destination, page, first, second in plan:
            tasks.append((destination_base + destination * destination_stride + destination_offset, page,
                          (source, first[1], first[2]), (source, second[1], second[2])))

    add(rows_dma.unstack_users(users, qkv_columns, 0, qkv_columns), 0, 0)
    add(rows_dma.unstack_users(users, 1, 0, 1), 1, 4)
    add(rows_dma.unstack_users(users, 1, 0, 1), 2, 8)
    add(rows_dma.unstack_users(users, block_columns, z_first, z_columns), 3, 12)
    for slot in range(SLOTS):
        add(rows_dma.unstack_users(users, qkv_columns, 0, qkv_columns), 4 + slot, 16, SLOTS, slot)
    return tasks


def problem(groups, packed_windows, operations):
    """Why the block launch cannot serve this layer (None when it can)."""
    if packed_windows is None:
        return 'the T2 packed windows are not engaged'
    if len(groups) < 2:
        return 'fewer than two users'
    # ttnn.Shape takes an int index only (no slice): index the two leading dimensions one by one.
    if any((piece.shape[0], piece.shape[1]) != (1, USER_ROWS) for piece, initial, history in groups):
        return 'a user is not a 16-row segment'
    if len(packed_windows) != len(groups) or any(len(user) != SLOTS for user in packed_windows):
        return 'the packed windows are not four per user'
    return None


def stage(mesh, projected, groups, packed_windows, taps, dt_bias, neg_exp_A, operations, note_fallback=None):
    """The BlockStage for one layer, or None (nothing left allocated, nothing advanced that the served loop reads) when the
    launches do not fit: the caller's loop then runs the served per-user conv gates."""
    from gdn_user_batch_conv import conv_gates

    def refuse(reason):
        if note_fallback is not None:
            note_fallback(reason)
        return None

    reason = problem(groups, packed_windows, operations)
    if reason is not None:
        return refuse(reason)
    found = tp_shapes.active()
    users, width = len(groups), projected.shape[-1]
    rows = users * USER_ROWS
    dram, dtype, layout = operations.DRAM_MEMORY_CONFIG, operations.bfloat16, operations.TILE_LAYOUT
    owned, per_user, held, entries, spread_entries = [], [], [], [], []

    def make(shape):
        tensor = operations.empty(shape, dtype=dtype, layout=layout, device=mesh, memory_config=dram)
        owned.append(tensor)
        return tensor

    try:
        canon = make((1, rows, width))
        rows_dma.launch(mesh, [projected], [canon], rows_dma.canon_block(users, width))
        block_windows = [make((1, rows, found.gdn_qkv)) for slot in range(SLOTS)]
        flat_windows = [window for user in packed_windows for window in user]
        rows_dma.launch(mesh, flat_windows, block_windows, plan_windows(users))
        spread = None
        if os.environ.get(SPREAD_FLAG, '0') != '0' or os.environ.get(SPREAD_AUDIT_FLAG, '0') != '0':
            # QWEN_FAST_TP4_CONV_GATES_SPREAD (F1, gdn_conv_gates_spread): the same launch with the gate tiles on cores of their own; None
            # (one logged line) when it cannot take the call, and then this is the served call below. Unset, none of this runs.
            import gdn_conv_gates_spread

            if gdn_conv_gates_spread.enabled():
                spread = gdn_conv_gates_spread.launch(operations, mesh, canon, block_windows, taps, dt_bias, neg_exp_A, rows,
                                                      found.gdn_qkv, found.gdn_a_col, found.gdn_b_col, held, spread_entries)
        if spread is None:
            conv, beta, g = conv_gates(operations, canon, block_windows, taps, dt_bias, neg_exp_A, rows)
        else:
            conv, beta, g = spread
        owned.extend([conv, beta, g])
        conv_users = [make((1, USER_ROWS, found.gdn_qkv)) for user in range(users)]
        beta_users = [make((1, USER_ROWS, found.gdn_nv)) for user in range(users)]
        g_users = [make((1, USER_ROWS, found.gdn_nv)) for user in range(users)]
        z_users = [make((1, USER_ROWS, found.gdn_a_col - found.gdn_qkv)) for user in range(users)]
        rows_dma.launch(mesh, [conv, beta, g, canon, *block_windows],
                        [*conv_users, *beta_users, *g_users, *z_users, *flat_windows], plan_unstack(users, width))
        if tp4_vglue.audit_enabled():
            entries = hold_served(mesh, groups, packed_windows, conv_users, beta_users, g_users, z_users, taps, dt_bias,
                                  neg_exp_A, operations, held)
            if spread_entries:
                # F1's three entries (block conv, beta and g against the served op's on shared scratch windows) ride
                # the first user's list: the layer's retained record frees them with the rest.
                entries[0].extend(spread_entries)
    except rows_dma.Unsupported as error:
        release_owned(operations, owned + held)
        return refuse(str(error))
    except BaseException:
        release_owned(operations, owned + held)
        raise
    for user in range(users):
        per_user.append(((conv_users[user], beta_users[user], g_users[user]), z_users[user]))
    per_user_tensors = {id(tensor) for pair in per_user for tensor in (*pair[0], pair[1])}
    return BlockStage(per_user, [tensor for tensor in owned if id(tensor) not in per_user_tensors], held, entries)


def hold_served(mesh, groups, packed_windows, conv_users, beta_users, g_users, z_users, taps, dt_bias, neg_exp_A, operations,
                held):
    """QWEN_FAST_TP4_VGLUE_AUDIT: per user, the served path's conv, beta, g (one conv_gates call on the user's own piece and
    its own freshly built windows), z (the served slice of the piece) and advanced windows, against copies of the block
    launch's; every copy and every served tensor is held outside `owned` (the launch's tensors are freed with the layer, so the
    compare after the replay reads the copies). Returns the per-user audit entries."""
    from gdn_conv_windows import build_windows
    from gdn_user_batch_conv import conv_gates

    found = tp_shapes.active()
    dram = operations.DRAM_MEMORY_CONFIG
    entries = []
    for user, (piece, initial, history) in enumerate(groups):
        own = []
        served_windows = build_windows(mesh, piece, history)
        held.extend(served_windows)
        served = conv_gates(operations, piece, served_windows, taps, dt_bias, neg_exp_A, USER_ROWS)
        held.extend(served)
        z = operations.slice(piece, (0, 0, found.gdn_qkv), (1, USER_ROWS, found.gdn_a_col), memory_config=dram)
        held.append(z)
        pairs = [('conv', conv_users[user], served[0]), ('beta', beta_users[user], served[1]),
                 ('g', g_users[user], served[2]), ('z', z_users[user], z)]
        pairs += [('window %d' % slot, packed_windows[user][slot], served_windows[slot]) for slot in range(SLOTS)]
        for label, mine, theirs in pairs:
            copy = operations.clone(mine, memory_config=dram)
            held.append(copy)
            own.append(dict(label='block conv user %d %s' % (user, label), mine=copy, served=theirs))
        entries.append(own)
    return entries
