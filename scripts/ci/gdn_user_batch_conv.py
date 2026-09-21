"""The packed users' convolution/gates per user, then ONE recurrence launch for all four.

`gdn_batched_conv.run_batched_projected` is the per-user block: DMA windows, the K-stack
conv+gates op, then the recurrence and norm/gate. Run once per packed user it costs four
of everything. This runs the convolution half unchanged, per user - the windows kernel
reads one history per launch (gdn_conv_windows.cpp) and the conv+gates op is a compiled
op that cannot be multi-instanced from the host - and then hands all four users'
`(conv, beta, g, initial, z)` sets to one `gdn_user_batch.execute`, which places each
user on its own 24-core share of the grid inside a single program.

Every tensor the caller gets back has the per-user shape the per-user path produces, and
the result dictionaries carry the same keys, so `restore_prefix`, `gdn_conv_prefix_copy`,
`gdn_records` and `gdn_commit_dma` see nothing new.

Deferred publication only. The per-user path writes the block-start entry into the shared
working state before each segment runs (`DeviceLoopState._recurrence`), which a batched
launch cannot reproduce: hoisting the four state moves ahead of one launch would leave
every user reading the LAST user's convolution state. Deferred packed decode does not do
that copy at all, which is why it is the only configuration accepted here - and it is the
configuration the four-user serving round runs in.
"""

from gdn_multitoken_conv import addresses, release_owned, validate_projected

import gdn_user_batch


def validate_options(dma_windows, packed_checkpoints, defer_conv_publication, prefix_zero_reuse):
    for name, value in (('dma_windows', dma_windows), ('packed_checkpoints', packed_checkpoints),
                        ('defer_conv_publication', defer_conv_publication),
                        ('prefix_zero_reuse', prefix_zero_reuse)):
        if type(value) is not bool:
            raise ValueError('Explicit bool %s option required' % name)
    if not (dma_windows and packed_checkpoints and defer_conv_publication):
        raise ValueError('Batched packed users require DMA windows, packed checkpoints and deferred publication')


def run_user_batched_projected(mesh, users, taps, dt_bias, neg_exp_A, norm_w, kernels, operations=None, *,
                               dma_windows=True, packed_checkpoints=True, defer_conv_publication=True,
                               prefix_zero_reuse=False, output_memory=None):
    """`users` is one `(projected, initial, conv_states)` triple per packed user.

    Returns one result dictionary per user, in user order, each shaped exactly like
    `run_batched_projected`'s and owning only its own tensors.
    """
    if operations is None:
        import ttnn as operations
    from gdn_conv_windows import build_windows

    validate_options(dma_windows, packed_checkpoints, defer_conv_publication, prefix_zero_reuse)
    groups = [tuple(user) for user in users]
    if not 1 <= len(groups) <= gdn_user_batch.MAX_USERS or any(len(user) != 3 for user in groups):
        raise ValueError('One to %d packed (projected, initial, conv_states) users required'
                         % gdn_user_batch.MAX_USERS)
    if len(taps) != 4 or any(tuple(tap.shape) != (1, 1, 5120) for tap in taps):
        raise ValueError('Four channel-wise convolution taps required')

    owned, shared = [], []
    per_user, widths, bindings = [], [], []
    try:
        weights = operations.to_memory_config(norm_w, operations.DRAM_MEMORY_CONFIG)
        if addresses(operations, weights) != addresses(operations, norm_w):
            shared.append(weights)
        for projected, initial, conv_states in groups:
            rows = validate_projected(tuple(projected.shape), conv_states)
            if rows < 2:
                raise ValueError('A batched packed user is a multirow segment; T1 uses the native path')
            bindings.append([addresses(operations, state) for state in conv_states])
            mine = []
            windows = build_windows(mesh, projected, conv_states)
            mine.extend(windows)
            packed = operations.transformer.gdn_decode_conv_gates(projected, windows, taps, projected, projected,
                dt_bias, neg_exp_A, batch=rows, memory_config=operations.DRAM_MEMORY_CONFIG,
                channels=5120, a_col=8192, b_col=8216)
            mine.extend(packed)
            z = operations.slice(projected, (0, 0, 5120), (1, rows, 8192),
                                 memory_config=operations.DRAM_MEMORY_CONFIG)
            if addresses(operations, z) != addresses(operations, projected):
                mine.append(z)
            owned.extend(mine)
            widths.append(rows)
            per_user.append(dict(windows=windows, owned=mine,
                                 inputs=(packed[0], packed[1], packed[2], initial, z, weights)))
        produced = gdn_user_batch.execute(mesh, [user['inputs'] for user in per_user], kernels, operations,
                                          output_memory=output_memory)
        # Owned BEFORE anything below can raise. The per-user loop asserts on state
        # addresses, and a raise part way through it would otherwise strand the outputs
        # of every user it had not reached yet: `execute` has already handed ownership
        # over, so nothing else would ever free them.
        owned.extend(value for pair in produced for value in pair)
        results = []
        for index, ((projected, initial, conv_states), user, rows, (output, states)) in enumerate(
                zip(groups, per_user, widths, produced, strict=True)):
            user['owned'].extend((output, states))
            if [addresses(operations, state) for state in conv_states] != bindings[index]:
                raise AssertionError('Batched convolution changed stable state addresses')
            results.append(dict(output=output, states=states, conv_prefixes=[None] * rows,
                owned=user['owned'] + (shared if index == 0 else []),
                packed_conv_states=user['windows'], mesh=mesh, packed_checkpoints=True,
                prefix_zero_reuse=prefix_zero_reuse, materialized_conv_prefixes=(),
                available_conv_prefixes=tuple(range(1, rows + 1)), hoisted_input=True,
                batched_convolution=True, dma_windows=True, norm_batch=False,
                deferred_conv_publication=True, user_batched=True, packed_users=len(groups)))
        return results
    except BaseException:
        release_owned(operations, owned + shared)
        raise
