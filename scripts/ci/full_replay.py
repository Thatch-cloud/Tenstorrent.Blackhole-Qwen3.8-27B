"""Exact two-block trace reuse gate; forced fixtures, not a request throughput benchmark."""

from attention_batch import capture_operation
from model_batch import ModelBatch
from verifier_inputs import stage_inputs


def validate_fixture(rows, first_prefix, second_prefix, oracle_length):
    if any(type(value) is not int for value in (rows, first_prefix, second_prefix, oracle_length)):
        raise ValueError('Integer replay geometry required')
    if rows not in (2, 16, 32) or first_prefix not in (0, 1, rows) or second_prefix not in (1, rows):
        raise ValueError('Bounded replay fixture required')
    if oracle_length < first_prefix + rows + 3:
        raise ValueError('Oracle must include the replay block and corrected continuation')


def verify_replay(model, prompt, oracle, pages, helpers, checkpoints, initial, *, rows, first_prefix, second_prefix,
                  prefill, decode, save, restore, state_digest, live_digest, kv_digest, inactive_digest, local_host,
                  norm_batch=False):
    validate_fixture(rows, first_prefix, second_prefix, len(oracle))
    import torch
    import ttnn

    length = len(prompt)
    mesh = model.mesh_device
    prefill(prompt)
    reference = [decode(oracle[index], length + index, False).reshape(-1, model.args.vocab_size)[:1]
                 for index in range(first_prefix + rows)]
    expected_output = torch.cat(reference[first_prefix:], dim=0)
    expected_end_state, expected_end_kv = live_digest(), kv_digest(length + first_prefix + rows)
    prefill(prompt)
    for index in range(first_prefix + second_prefix):
        decode(oracle[index], length + index, False)
    expected_commit_state, expected_commit_kv = live_digest(), kv_digest(length + first_prefix + second_prefix)
    expected_correction = [decode(oracle[index], length + index, False)
                           for index in range(first_prefix + second_prefix, first_prefix + second_prefix + 2)]
    expected_final_state, expected_final_kv = live_digest(), kv_digest(length + first_prefix + second_prefix + 2)
    prefill(prompt)
    expected_inactive = inactive_digest()
    fixture = ModelBatch(model, oracle[:rows], length, pages, helpers, checkpoints, rows,
        serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True,
        hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True, batch_conv=True,
        packed_checkpoints=True, retain_records=True, ordered_cache=True, norm_batch=norm_batch)
    captured, output = None, None
    try:
        save(initial)
        captured, output = capture_operation(ttnn, mesh, fixture.run)
        restore(initial)
        ttnn.execute_trace(mesh, captured, cq_id=0, blocking=True)
        expected_first = torch.cat(reference[:rows], dim=0)
        if any(not torch.equal(value.reshape(rows, model.args.vocab_size), expected_first) for value in local_host(output)):
            raise AssertionError('Initial captured block differs from native before reuse')
        fixture.retained.commit(first_prefix, dma=True, synchronize=True)
        stage_inputs(fixture, oracle[first_prefix:first_prefix + rows], length + first_prefix)
        fixture.retained.replay(lambda: ttnn.execute_trace(mesh, captured, cq_id=0, blocking=True))
        if fixture.retained.replay_epoch != 1:
            raise AssertionError('Exactly one successful replay epoch required')
        if any(not torch.equal(value.reshape(rows, model.args.vocab_size), expected_output) for value in local_host(output)):
            raise AssertionError('Changed-token/position replay logits differ from native')
        if live_digest() != expected_end_state or kv_digest(length + first_prefix + rows) != expected_end_kv:
            raise AssertionError('Replayed block did not refresh native GDN/KV state')
        if state_digest(checkpoints) != expected_end_state:
            raise AssertionError('Replayed end checkpoint remained stale')
        fixture.retained.commit(second_prefix, dma=True, synchronize=True)
        if live_digest() != expected_commit_state or kv_digest(length + first_prefix + second_prefix) != expected_commit_kv:
            raise AssertionError('Second decision did not use refreshed prefix histories')
        if inactive_digest() != expected_inactive:
            raise AssertionError('Replay changed an inactive native GDN slot')
        for index, expected in zip(range(first_prefix + second_prefix, first_prefix + second_prefix + 2), expected_correction, strict=True):
            if not torch.equal(decode(oracle[index], length + index, True), expected):
                raise AssertionError('Corrected continuation after replay differs from native')
        if live_digest() != expected_final_state or kv_digest(length + first_prefix + second_prefix + 2) != expected_final_kv:
            raise AssertionError('Corrected replay continuation GDN/KV differs')
        return dict(length=length, rows=rows, first_prefix=first_prefix, second_prefix=second_prefix,
            replay_epoch=fixture.retained.replay_epoch, logits_exact=True, refreshed_prefixes_exact=True,
            valid_kv_exact=True, inactive_slots_exact=True, correction_steps=2,
            scope='Two blocks using one captured verifier with changed metadata; no drafter or throughput claim')
    finally:
        if captured is not None:
            ttnn.release_trace(mesh, captured)
        if output is not None:
            ttnn.deallocate(output)
        fixture.close()
