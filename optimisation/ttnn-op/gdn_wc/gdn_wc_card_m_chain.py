"""The served per-user path the WC card-M gate holds the one-launch stage against (gdn_wc_card_m.py), in plain ttnn calls that need no particular chip count.

Per user: the piece the served split cuts (a tile slice for users 0 and 2, the untilize / slice / tilize round trip for users 1 and 3: the pinned
gdn_device_loop_state.resident_piece over ttnn.slice, as DeviceLoopState.served_pieces does), the T2 packed windows (the production window kernel, which runs on a
1x1 mesh), ONE served gdn_decode_conv_gates call on the user's own piece and windows at batch 16 (the windows advance in place, as the commit reads them), and
the z slice of the piece. This is the reference gdn_block_conv_tp.hold_served builds in-trace for the audit, so the gate and the audit hold WC to the same path.
"""


def served_path(ttnn, mesh, found, tensors):
    import gdn_conv_windows_packed as packed
    import gdn_device_loop_state as pinned
    import gdn_user_batch_conv

    users, block = tensors['users'], tensors['x']
    width = block.shape[-1]
    dram = ttnn.DRAM_MEMORY_CONFIG
    owned = []
    pieces = [pinned.resident_piece(ttnn, ttnn.slice(block, (0, 16 * user, 0), (1, 16 * user + 16, width)), owned) for user in range(users)]
    windows = packed.build_windows_packed(mesh, [(pieces[user], tensors['histories'][user]) for user in range(users)], operations=ttnn)
    made = dict(conv=[], beta=[], g=[], z=[], window=windows, owned=owned, pieces=pieces)
    for user in range(users):
        conv, beta, g = gdn_user_batch_conv.conv_gates(ttnn, pieces[user], windows[user], tensors['taps'], tensors['dt_bias'], tensors['neg_exp_A'], 16)
        made['conv'].append(conv)
        made['beta'].append(beta)
        made['g'].append(g)
        made['z'].append(ttnn.slice(pieces[user], (0, 0, found.gdn_qkv), (1, 16, found.gdn_a_col), memory_config=dram))
    return made
