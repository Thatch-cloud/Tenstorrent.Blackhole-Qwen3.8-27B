"""A correctness-first device step for a packed block: one user at a time.

`serving_packed_bridge.execute_packed_decode` takes the device step as a
parameter. This is the simplest implementation that is CORRECT: it steps each
packed request in turn, through exactly the single-user machinery that already
works, and returns the outputs in the scheduler's order.

**This is not the batched verifier and does not pretend to be.** It spends one
full pass over the 19.92 GB of dense projections per user per round, so per-user
throughput is the single-user rate divided by the number of users - the very thing
`docs/batch-spec-tasks-2026-09-19.md` rules out as a path to 200 tok/s per user.
What it buys is a working two-user serving path today, and a place for the real
packed step to drop in behind the same interface tomorrow.

Its one genuine job beyond looping is ORDER. Probe 35436807668 measured the
scheduler presenting the pair as `cached=['B','A']`, not in creation order, and
`execute_packed_decode` checks that each output's request id matches the entry it
was produced for. Stepping in entry order is what keeps that true.
"""


def sequential_packed_step(entries, *, cancelled):
    """Step every packed request in the scheduler's order, one at a time."""
    if not entries:
        raise ValueError('A packed step needs at least one admitted request')
    outputs = []
    for entry in entries:
        request, ticket = entry['request'], entry['ticket']
        if ticket.request_id != entry['request_id']:
            raise ValueError('Each packed entry must carry its own prepared ticket')
        output = request.step(ticket.request_id, cancelled=cancelled)
        if output is None or output.request_id != entry['request_id']:
            raise ValueError('A packed request must commit its own output')
        outputs.append(output)
    return outputs


def describe():
    """What this costs, so a benchmark reading it is not mistaken for the goal."""
    return dict(name='sequential', weight_passes_per_round='one per user',
                per_user_rate='single-user rate divided by users',
                batched=False)
