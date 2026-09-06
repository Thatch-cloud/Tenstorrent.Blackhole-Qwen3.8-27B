"""Greedy prefix selection only; device verification and state commit are external."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    emitted: tuple
    accepted: int
    state_rows: int
    next_input: int | None
    finished: bool


def select_prefix(proposals, target_argmax, *, vocab_size, eos_ids=(), max_proposals=15):
    """Target rows consume [already-emitted seed, *proposals]; each predicts its successor."""
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("Expected positive target vocabulary size")
    if type(max_proposals) is not int or max_proposals not in (15, 31):
        raise ValueError('Explicit T16 or T32 proposal capacity required')
    proposals, target_argmax, eos_ids = tuple(proposals), tuple(target_argmax), tuple(eos_ids)
    if len(proposals) > max_proposals or len(target_argmax) != len(proposals) + 1:
        raise ValueError('Expected bounded K proposals and K+1 target prediction rows')
    if any(type(token) is not int or not 0 <= token < vocab_size
           for token in (*proposals, *target_argmax, *eos_ids)):
        raise ValueError("Token outside target vocabulary")
    accepted = 0
    for proposed, predicted in zip(proposals, target_argmax):
        if proposed != predicted:
            break
        accepted += 1
        if proposed in eos_ids:
            return Decision(proposals[:accepted], accepted, accepted, None, True)
    correction = target_argmax[accepted]
    finished = correction in eos_ids
    return Decision(proposals[:accepted] + (correction,), accepted, accepted + 1,
                    None if finished else correction, finished)
