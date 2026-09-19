"""Execute one scheduled step covering several packed decode requests.

`FastRunnerBridge.execute_decode` admits a single resident request, refreshes that
request's page binding, and runs `request.step`. Packed, every one of those is per
user except two things that must happen exactly once: `runner._update_states`,
which consumes the whole SchedulerOutput, and the device step itself - which is the
entire point, since one pass over the weights is what packing buys.

The device step is injected rather than reached for. The packed fixture lives in
`ModelBatch` behind `verifier_pack`, and keeping it a parameter is what lets this
contract be exercised without a device.
"""

from serving_vllm_packed import admit_packed_scheduler_output, packed_model_runner_output


def execute_packed_decode(bridges, scheduled, *, cancelled, packed_step):
    """Admit a packed step, bind every user's pages, run one device step.

    `bridges` maps request id to its live bridge. The returned output follows the
    SCHEDULER's order, because that is the order the packed block's row segments
    were matched to.
    """
    from serving_vllm_state import apply_committed_output, validate_runner_reservation

    if not bridges or not callable(packed_step):
        raise ValueError('Live packed bridges and an explicit device step required')
    if any(bridge.failed for bridge in bridges.values()):
        raise ValueError('Failed runner bridge cannot execute another block')
    runners = {id(bridge.runner) for bridge in bridges.values()}
    if len(runners) != 1:
        raise ValueError('Packed requests must share one model runner')
    entries = admit_packed_scheduler_output([bridge.request for bridge in bridges.values()], scheduled)
    ordered = [dict(entry, bridge=bridges[entry['request_id']]) for entry in entries]
    runner = ordered[0]['bridge'].runner
    try:
        for entry in ordered:
            bridge = entry['bridge']
            if bridge.validate_storage is not None:
                bridge.validate_storage()
        # once, because it consumes the whole SchedulerOutput rather than one request
        runner._update_states(scheduled)
        for entry in ordered:
            bridge, ticket = entry['bridge'], entry['ticket']
            validate_runner_reservation(runner, bridge.state, ticket)
            if len(bridge.state.block_ids) != 1:
                raise ValueError('One explicit target KV page group required')
            bridge.page_binding.refresh(bridge.state.block_ids[0], position=ticket.position,
                                        rows=len(ticket.tokens))
        outputs = packed_step(ordered, cancelled=cancelled)
        if len(outputs) != len(ordered):
            raise ValueError('The packed step must commit one output per packed request')
        for entry, output in zip(ordered, outputs):
            if output.request_id != entry['request_id']:
                raise ValueError('Packed outputs must follow the scheduled order')
            apply_committed_output(runner, entry['bridge'].state, output)
        return packed_model_runner_output(outputs)
    except BaseException:
        for bridge in bridges.values():
            bridge.failed = True
        raise
