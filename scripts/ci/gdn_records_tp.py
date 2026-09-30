"""gdn_records.retain_checkpoint_histories for any served width.

The pinned function checks that the retained histories and the caller's output are independent on both chips with a
literal `range(2)`; gdn_records.py is held unedited by the K5 evidence (test_gdn_seq_block), so this twin counts the
chips from the bindings themselves. tp_addresses.install() rebinds the pinned name at QWEN_FAST_TP=4 only.
"""

from gdn_records import block_histories
from tp_addresses import addresses, release_owned


def retain_checkpoint_histories(operations, result, output):
    histories = block_histories(result)
    bindings = [addresses(operations, value) for value in [*histories, output]]
    if any(len({binding[chip] for binding in bindings}) != len(bindings) for chip in range(len(bindings[0]))):
        raise ValueError('Histories and caller-owned output must be independent on every chip')
    protected = set(bindings)
    owned = {addresses(operations, value): value for value in result['owned']}
    history_addresses = {addresses(operations, value) for value in histories}
    if len(history_addresses) != len(histories) or not history_addresses.issubset(owned):
        raise ValueError('Every independent history must have retained ownership')
    for binding in owned:
        if binding not in protected and any(any(current == saved for current, saved in zip(binding, keep, strict=True))
                                            for keep in protected):
            raise ValueError('Scratch partially aliases a protected history or output')
    release_owned(operations, [value for binding, value in owned.items() if binding not in protected])
    result['owned'] = [value for binding, value in owned.items() if binding in history_addresses]
