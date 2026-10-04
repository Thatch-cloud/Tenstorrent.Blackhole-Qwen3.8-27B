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

import time

from serving_vllm_packed import admit_packed_scheduler_output, packed_model_runner_output


def execute_packed_decode(bridges, scheduled, *, cancelled, packed_step):
    """Admit a packed step, bind every user's pages, run one device step.

    `bridges` maps request id to its live bridge. The returned output follows the
    SCHEDULER's order, because that is the order the packed block's row segments
    were matched to.
    """
    from serving_vllm_state import apply_committed_output, validate_runner_reservation
    import verify_prestage

    # tp4/hostgap: QWEN_FAST_TP4_HOSTGAP_LOG times the step entry's parts (stage 0: one [PACKED-ENTRY] line a step, written by the
    # packed step once it has its own checks' time); QWEN_FAST_TP4_ENTRY_DIET runs each distinct storage validator once (1d).
    # Both unset, the body below is today's.
    log, diet = verify_prestage.hostgap_log_enabled(), verify_prestage.entry_diet_enabled()
    stamps = [time.perf_counter()] if log else None
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
        if log:
            stamps.append(time.perf_counter())
        validated = []
        for entry in ordered:
            bridge = entry['bridge']
            if bridge.validate_storage is not None:
                if diet:
                    # 1d: every bridge holds the same bound owner.validate, which re-inspects every paged-KV buffer: the same check
                    # once, at the same point, before any device work. A validator not equal to an earlier one still runs.
                    if any(bridge.validate_storage == earlier for earlier in validated):
                        continue
                    validated.append(bridge.validate_storage)
                bridge.validate_storage()
        if log:
            stamps.append(time.perf_counter())
        # once, because it consumes the whole SchedulerOutput rather than one request
        runner._update_states(scheduled)
        if log:
            stamps.append(time.perf_counter())
        # One loop, as ever: a user's reservation check, then its page refresh, then the next user's.
        reservation_ms = refresh_ms = 0.0
        writes = 0
        for entry in ordered:
            bridge, ticket = entry['bridge'], entry['ticket']
            if log:
                lap = time.perf_counter()
            validate_runner_reservation(runner, bridge.state, ticket, len(ordered))
            if len(bridge.state.block_ids) != 1:
                raise ValueError('One explicit target KV page group required')
            if log:
                lapped = time.perf_counter()
                reservation_ms += (lapped - lap) * 1000
            wrote = bridge.page_binding.refresh(bridge.state.block_ids[0], position=ticket.position,
                                                rows=len(ticket.tokens))
            if log:
                refresh_ms += (time.perf_counter() - lapped) * 1000
                writes += 1 if wrote is True else 0
        if log:
            stamps.append(time.perf_counter())
            verify_prestage.note_scratch('entry', dict(
                admit_ms=(stamps[1] - stamps[0]) * 1000, storage_ms=(stamps[2] - stamps[1]) * 1000,
                update_states_ms=(stamps[3] - stamps[2]) * 1000, reservation_ms=reservation_ms,
                refresh_ms=refresh_ms, refresh_writes=writes, started=stamps[0]))
        outputs = packed_step(ordered, cancelled=cancelled)
        if len(outputs) != len(ordered):
            raise ValueError('The packed step must commit one output per packed request')
        for entry, output in zip(ordered, outputs):
            if output.request_id != entry['request_id']:
                raise ValueError('Packed outputs must follow the scheduled order')
            apply_committed_output(runner, entry['bridge'].state, output, len(ordered))
        return packed_model_runner_output(outputs)
    except BaseException:
        for bridge in bridges.values():
            bridge.failed = True
        raise
