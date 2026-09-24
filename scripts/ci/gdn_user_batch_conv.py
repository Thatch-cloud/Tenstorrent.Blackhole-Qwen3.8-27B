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

Under QWEN_FAST_VERIFY_T2=1 (verify_trace_t2, cut #1) the four window launches become one
(gdn_conv_windows_packed.build_windows_packed) ahead of the per-user conv gates: the same
pieces and histories in, the same 16 window tensors out, so every conv gates op still reads
its own user's windows and nothing in the layer writes a piece or a history in between.
An input the packed op does not serve (Unsupported) takes the served per-user path, counted
as windows_fallback. QWEN_FAST_VERIFY_T2_AUDIT=1 also builds the served windows beside them
(G1: packed_verifier compares every layer on the first round, then two per round); those are
handed over as `audit_windows`, outside `owned`, so retain_checkpoint_histories never frees
them inside the trace - ModelBatch releases them with the retained records.
"""

from gdn_multitoken_conv import addresses, release_owned, validate_projected

import gdn_user_batch
from verify_trace_t2 import audit_enabled as t2_audit, cut as t2_cut, fell_back as t2_fell_back, note as t2_note


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
    # QWEN_FAST_VERIFY_T2 (#1): the served windows the audit builds beside the packed ones,
    # per user. Never in `owned` (see the module docstring); freed here only on failure.
    audit, audit_owned = {}, []
    packed_windows = None
    try:
        weights = operations.to_memory_config(norm_w, operations.DRAM_MEMORY_CONFIG)
        if addresses(operations, weights) != addresses(operations, norm_w):
            shared.append(weights)
        if t2_cut('windows'):
            from gdn_conv_windows_packed import Unsupported, build_windows_packed
            try:
                packed_windows = build_windows_packed(mesh, [(projected, conv_states)
                                                             for projected, initial, conv_states in groups],
                                                      operations=operations)
            except Unsupported as reason:
                t2_note('windows_fallback')
                t2_fell_back('windows', reason)
            else:
                owned.extend(window for user in packed_windows for window in user)
                t2_note('windows')
        for index, (projected, initial, conv_states) in enumerate(groups):
            rows = validate_projected(tuple(projected.shape), conv_states)
            if rows < 2:
                raise ValueError('A batched packed user is a multirow segment; T1 uses the native path')
            bindings.append([addresses(operations, state) for state in conv_states])
            mine = []
            if packed_windows is None:
                windows = build_windows(mesh, projected, conv_states)
            else:
                windows = list(packed_windows[index])
                if t2_audit():
                    served = build_windows(mesh, projected, conv_states)
                    audit_owned.extend(served)
                    audit[index] = served
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
                deferred_conv_publication=True, user_batched=True, packed_users=len(groups),
                **({'audit_windows': audit[index]} if index in audit else {})))
        return results
    except BaseException:
        release_owned(operations, owned + shared + audit_owned)
        raise
