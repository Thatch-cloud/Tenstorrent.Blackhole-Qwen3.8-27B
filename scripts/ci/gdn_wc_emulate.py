"""CPU emulation of what gdn_wc_reader.cpp and gdn_wc_writer.cpp do to bytes (tests and the card-M gate's host model; never loaded by the image).

A tile is a flat int16 vector of 1024 elements in FACE ORDER (face 0 = rows 0-15 x columns 0-15, face 1 = rows 0-15 x columns 16-31, face 2 = rows
16-31 x columns 0-15, face 3 = rows 16-31 x columns 16-31; each face row-major 16 x 16), the layout of a bfloat16 TILE page in DRAM and in a
circular buffer. Everything the kernels do is a copy of byte ranges inside such pages (or the canonical rule on whole words), so the emulation is
the kernels' loops transcribed over these vectors, with the very same offsets: a row of a face is 16 elements (32 bytes), a face 256 (512 bytes),
a half tile 512 (1024 bytes). The compute kernel is NOT emulated here: `Compute` is a stand-in with the served kernel's contract (the four window
tiles in, one conv tile and the four tiles passed through out; the gate tiles in, beta and g out), used to compare routes, not arithmetic.

Torch is imported at top level (tests only).
"""

import torch

import gdn_wc as wc

PAGE = wc.PAGE // 2          # elements of a tile
HALF = wc.HALF // 2
FACE = wc.FACE // 2
ROW = wc.ROW // 2
SPAN = wc.SPAN // 2


# ---- tile layout helpers ----

def tile_to_faces(matrix):
    """A (32, 32) int16 tile -> its 1024-element face-order vector."""
    faces = [matrix[0:16, 0:16], matrix[0:16, 16:32], matrix[16:32, 0:16], matrix[16:32, 16:32]]
    return torch.cat([face.contiguous().reshape(-1) for face in faces])


def faces_to_tile(vector):
    """A face-order vector -> the (32, 32) tile."""
    matrix = torch.zeros(32, 32, dtype=torch.int16)
    faces = [vector[index * FACE:(index + 1) * FACE].reshape(16, 16) for index in range(4)]
    matrix[0:16, 0:16], matrix[0:16, 16:32], matrix[16:32, 0:16], matrix[16:32, 16:32] = faces
    return matrix


def canonical(vector, denorm=True):
    """The kernels' canonical_half over int16 elements: a zero exponent (-0 and every denormal) comes out +0 (CANON_DENORM on), else only -0."""
    raw = vector.to(torch.int32) & 0xFFFF
    if denorm:
        return torch.where((raw & 0x7F80) == 0, torch.zeros_like(vector), vector)
    return torch.where(raw == 0x8000, torch.zeros_like(vector), vector)


class WriteOnce(dict):
    """{(tensor, user, slot, page): tile}: a page written twice is a planner bug."""

    def put(self, key, tile):
        if key in self:
            raise AssertionError('page %r written twice' % (key,))
        self[key] = tile.clone()


# ---- the reader ----

def reader_conv_instance(inst, shape, x, history, taps, denorm=True):
    """gdn_wc_reader.cpp's conv loop body for instance `inst`: returns (window tiles [st1, st2, st3, x] as the CB holds them, the four tap tiles).
    x: {page: tile}, history[user][index]: {page: tile} (only rows 0 of faces 0 and 1 of histories 1..3 are read), taps[j]: {page: tile}."""
    bt, cc = divmod(inst, shape['channel_tiles'])
    window = [torch.zeros(PAGE, dtype=torch.int16) for _ in range(wc.K)]
    window[3] = x[bt * shape['width_tiles'] + cc].clone()
    # the 64-byte spans of the history pages: row 0 of face 0 and of face 1, 32 elements each, of which the first 16 are used
    spans = {}
    for user_local in range(2):
        user = 2 * bt + user_local
        for index in range(1, wc.HISTORY):
            page = history[user][index][cc]
            for face in range(2):
                spans[(user_local, index, face)] = page[face * FACE:face * FACE + SPAN].clone()
    # canonicalise the second user's half of the block tile in place (words of faces 2-3)
    window[3][HALF:] = canonical(window[3][HALF:], denorm)
    for s in range(1, wc.HISTORY):
        slot = s - 1
        for face_index in range(4):
            user_local, face = face_index >> 1, face_index & 1
            base = face_index * FACE
            for r in range(wc.HISTORY - s):
                window[slot][base + r * ROW:base + (r + 1) * ROW] = spans[(user_local, r + s, face)][:ROW]
            length = (16 - wc.HISTORY + s) * ROW
            window[slot][base + (wc.HISTORY - s) * ROW:base + (wc.HISTORY - s) * ROW + length] = window[3][base:base + length]
    return window, [taps[j][cc].clone() for j in range(wc.K)]


def reader_gate_tile(gi, shape, x, dt_bias, neg_exp_A, denorm=True):
    """The gate loop body for gate tile `gi` (bt_g = gi // gate_columns, t = gi % gate_columns): (a tile, b tile, dt_bias tile, neg_exp_A tile)."""
    bt_g, t = divmod(gi, shape['gate_columns'])
    h0 = t * wc.TILE
    h1 = min(h0 + wc.TILE, shape['heads'])
    tiles = []
    for column in (shape['a_col'], shape['b_col']):
        dst = torch.zeros(PAGE, dtype=torch.int16)
        for h in range(h0, h1):
            c = column + h
            page = x[bt_g * shape['width_tiles'] + c // wc.TILE]
            sc, dc = c % wc.TILE, h - h0
            for r in range(wc.TILE):
                source = ((r // 16) * 2 + sc // 16) * FACE + (r % 16) * 16 + sc % 16
                target = ((r // 16) * 2 + dc // 16) * FACE + (r % 16) * 16 + dc % 16
                dst[target] = page[source]
        dst[HALF:] = canonical(dst[HALF:], denorm)
        tiles.append(dst)
    return tiles[0], tiles[1], dt_bias[t].clone(), neg_exp_A[t].clone()


def reader_z_tile(zi, shape, x, denorm=True):
    """The z loop body: the projection tile bt * Wt + z_first + column with its second half canonical."""
    bt, column = divmod(zi, shape['z_columns'])
    tile = x[bt * shape['width_tiles'] + shape['z_first'] + column].clone()
    tile[HALF:] = canonical(tile[HALF:], denorm)
    return tile


# ---- the writer ----

def user_page(tile, half):
    """A user's output page from a block tile: half `half` of the tile in the page's half 0, zeros in half 1 (the local copy into the staging page,
    whose upper half was zeroed once)."""
    page = torch.zeros(PAGE, dtype=torch.int16)
    page[:HALF] = tile[half * HALF:(half + 1) * HALF]
    return page


def writer_conv_instance(inst, shape, out_tile, shift_tiles, written):
    bt, cc = divmod(inst, shape['channel_tiles'])
    for half in range(2):
        user = 2 * bt + half
        written.put(('conv', user, 0, cc), user_page(out_tile, half))
        for slot in range(wc.SLOTS):
            written.put(('window', user, slot, cc), user_page(shift_tiles[slot], half))


def writer_gate_tile(gi, shape, beta_tile, g_tile, written):
    bt, t = divmod(gi, shape['gate_columns'])
    for half in range(2):
        user = 2 * bt + half
        written.put(('beta', user, 0, t), user_page(beta_tile, half))
        written.put(('g', user, 0, t), user_page(g_tile, half))


def writer_z_tile(zi, shape, z_tile, written):
    bt, column = divmod(zi, shape['z_columns'])
    for half in range(2):
        written.put(('z', 2 * bt + half, 0, column), user_page(z_tile, half))


# ---- the launch ----

def emulate(work_plan, x, history, taps, dt_bias, neg_exp_A, compute, denorm=True):
    """Every core's reader, the compute stand-in and every core's writer over the plan. `compute.instance(window, tap) -> (out, shift)` and
    `compute.gate(a, b, dtb, nega) -> (beta, g)` take and return face-order tiles. Returns the WriteOnce map of everything written."""
    shape = work_plan['shape']
    written = WriteOnce()
    for item in work_plan['work']:
        for inst in range(item['start'], item['start'] + item['n_inst']):
            window, tap_tiles = reader_conv_instance(inst, shape, x, history, taps, denorm)
            out_tile, shift_tiles = compute.instance(window, tap_tiles)
            writer_conv_instance(inst, shape, out_tile, shift_tiles, written)
        for gi in range(item['gate_start'], item['gate_start'] + item['g_n']):
            a, b, dtb, nega = reader_gate_tile(gi, shape, x, dt_bias, neg_exp_A, denorm)
            beta_tile, g_tile = compute.gate(a, b, dtb, nega)
            writer_gate_tile(gi, shape, beta_tile, g_tile, written)
        for zi in range(item['z_start'], item['z_start'] + item['n_z']):
            writer_z_tile(zi, shape, reader_z_tile(zi, shape, x, denorm), written)
    return written


# ---- the compute stand-in: the served kernel's contract on whole tiles (float32 per element, rounded to bf16) ----

def _to_float(bits):
    return bits.contiguous().view(torch.bfloat16).to(torch.float32)


def _to_bits(values):
    return values.to(torch.bfloat16).contiguous().view(torch.int16)


class Compute(object):
    """gdn_decode_conv_gates' contract on one instance: conv = silu(sum_j window_j * tap_j) with the tap's row 0 broadcast over the rows, the window
    passed through (the writer stores it as the advanced shift register); gates: beta = sigmoid(b), g = neg_exp_A * softplus(a + dt_bias) with the
    parameters' row 0 broadcast. Row-parallel and per element, so it is a faithful stand-in for a ROUTING comparison (it cannot show the real
    kernel's arithmetic)."""

    def instance(self, window, taps):
        frames = [_to_float(faces_to_tile(tile)) for tile in window]
        tap_rows = [_to_float(faces_to_tile(tile))[0:1, :] for tile in taps]
        accumulated = sum(frame * tap for frame, tap in zip(frames, tap_rows))
        out = tile_to_faces(_to_bits(torch.nn.functional.silu(accumulated)))
        return out, [tile.clone() for tile in window]

    def gate(self, a, b, dtb, nega):
        a_values, b_values = _to_float(faces_to_tile(a)), _to_float(faces_to_tile(b))
        dt_row, neg_row = _to_float(faces_to_tile(dtb))[0:1, :], _to_float(faces_to_tile(nega))[0:1, :]
        beta = tile_to_faces(_to_bits(torch.sigmoid(b_values)))
        g = tile_to_faces(_to_bits(neg_row * torch.nn.functional.softplus(a_values + dt_row)))
        return beta, g
